"""Configuration objects for the credit-risk scorecard monitoring pipeline.

Everything that encodes a *business rule* lives here so that the analytical
modules stay generic.  Adding a fifth scorecard, changing the segment order,
re-pointing a reference period or tightening a threshold is a configuration
change, never a code change.

Configuration can be supplied as:

* a Python ``dict``                     -> :func:`MonitoringConfig.from_dict`
* a YAML file (requires ``PyYAML``)     -> :func:`load_config`
* a JSON file                           -> :func:`load_config`
* interactive prompts for the reference
  periods only                          -> :mod:`model_monitoring.reference_periods`

The module deliberately has no pandas-heavy logic: it is importable from
tests without touching a dataset.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Module level defaults (the names used in the project brief)
# ---------------------------------------------------------------------------

SEGMENT_ORDER: List[str] = [
    "CAT A", "CAT B", "CAT C1",
    "CAT C2", "CAT D", "CAT E",
]

#: ``A_TO_E_INCREASING``  -> ``CAT A`` is the SAFEST  and ``CAT E`` the RISKIEST
#: ``A_TO_E_DECREASING``  -> ``CAT A`` is the RISKIEST and ``CAT E`` the SAFEST
RISK_DIRECTION: str = "A_TO_E_INCREASING"

DEFAULT_MODEL_NAMES: List[str] = ["MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskDirection(str, Enum):
    """Which end of :data:`SEGMENT_ORDER` carries the most credit risk."""

    A_TO_E_INCREASING = "A_TO_E_INCREASING"
    A_TO_E_DECREASING = "A_TO_E_DECREASING"


class ReferenceMode(str, Enum):
    """Which population is used as the PSI/CSI baseline."""

    #: The model development sample (the population the scorecard was fitted on).
    DEVELOPMENT = "DEVELOPMENT"
    #: A separately approved production reference population (post-implementation
    #: baseline, e.g. the first stable quarter after go-live).
    PRODUCTION_BASELINE = "PRODUCTION_BASELINE"


class VariableType(str, Enum):
    """How a characteristic is binned for CSI."""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    #: Already-binned scorecard input (WOE bands, approved bands).  Treated as
    #: categorical but the bin order is taken from the supplied definition.
    BINNED = "BINNED"


class VariableRole(str, Enum):
    """What the column represents inside the scorecard."""

    FEATURE = "FEATURE"          # a raw model input / characteristic
    SUBSCORE = "SUBSCORE"        # variable-level points contribution
    SCORE = "SCORE"              # final / aggregate scorecard score


class Severity(str, Enum):
    """Alert severity labels used across every sheet."""

    GREEN = "GREEN"   # within normal monitoring limits
    AMBER = "AMBER"   # investigation required
    RED = "RED"       # significant breach requiring escalation
    GREY = "GREY"     # insufficient data / metric unavailable

    @property
    def rank(self) -> int:
        """Ordering used for 'worst severity' aggregation and for sorting."""
        return _SEVERITY_RANK[self.value]


_SEVERITY_RANK: Dict[str, int] = {"GREEN": 0, "GREY": 1, "AMBER": 2, "RED": 3}


def severity_rank(value: Any) -> int:
    """Rank of a severity label, tolerant of plain strings and ``None``."""
    if value is None:
        return _SEVERITY_RANK["GREY"]
    if isinstance(value, Severity):
        return value.rank
    return _SEVERITY_RANK.get(str(value).strip().upper(), _SEVERITY_RANK["GREY"])


def worst_severity(values: Iterable[Any]) -> Severity:
    """Return the most serious severity in ``values`` (GREEN when empty)."""
    worst = Severity.GREEN
    for value in values:
        if severity_rank(value) > worst.rank:
            worst = Severity(str(value).strip().upper()) if not isinstance(value, Severity) else value
    return worst


class MultiRecordPolicy(str, Enum):
    """How to collapse several applications of one customer inside a period.

    Used by quarter-on-quarter customer migration, which must compare one
    segment per customer per period.
    """

    LATEST = "LATEST"      # keep the most recent LOGIN_DATE
    EARLIEST = "EARLIEST"  # keep the first LOGIN_DATE
    WORST = "WORST"        # keep the riskiest segment (conservative)
    BEST = "BEST"          # keep the safest segment
    DROP = "DROP"          # exclude the customer entirely (strictest)


class SmoothingMethod(str, Enum):
    """Zero-frequency treatment for PSI/CSI."""

    #: Add a configurable pseudocount to every category count *before*
    #: normalising.  Statistically defensible (additive / Laplace smoothing),
    #: keeps proportions summing to exactly 1 and never divides by zero.
    PSEUDOCOUNT = "PSEUDOCOUNT"
    #: Drop bins where either side is empty, and flag the comparison.  Keeps
    #: the arithmetic pure at the cost of an incomplete sum; the number of
    #: dropped bins is reported.
    DROP_EMPTY_BINS = "DROP_EMPTY_BINS"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_date(value: Any, field_name: str = "date") -> Optional[date]:
    """Coerce ``value`` to a :class:`datetime.date` (``None`` passes through)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt) + 2], fmt).date()
        except ValueError:
            continue
    try:  # last resort: let pandas try
        import pandas as pd

        parsed = pd.to_datetime(text, errors="coerce")
        if parsed is not None and not pd.isna(parsed):
            return parsed.date()
    except Exception:  # pragma: no cover - defensive
        pass
    raise ConfigError(f"Could not parse {field_name!r} from value {value!r}; expected YYYY-MM-DD.")


def _enum(enum_cls, value, field_name: str):
    """Coerce ``value`` into ``enum_cls`` with a helpful error message."""
    if isinstance(value, enum_cls):
        return value
    if value is None:
        raise ConfigError(f"{field_name} is required (one of {[e.value for e in enum_cls]}).")
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return enum_cls(text)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid {field_name}={value!r}. Allowed values: {[e.value for e in enum_cls]}."
        ) from exc


class ConfigError(ValueError):
    """Raised for any invalid or internally inconsistent configuration."""


# ---------------------------------------------------------------------------
# Segment configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentConfig:
    """Ordered risk grades produced by every scorecard.

    ``segment_order`` is the *display* order.  ``risk_direction`` tells the
    pipeline which end of that list is the riskiest, so that upgrades,
    downgrades, risk ranks and the Excel colour palette stay correct even if a
    bank reverses the convention.
    """

    segment_order: List[str] = field(default_factory=lambda: list(SEGMENT_ORDER))
    risk_direction: RiskDirection = RiskDirection.A_TO_E_INCREASING
    #: How many of the riskiest segments make up the "riskier segment share"
    #: head-line metric.  Default: the two riskiest grades.
    riskier_segment_count: int = 2
    #: Label applied to blank/NaN scorecard output.
    missing_label: str = "MISSING"
    #: Label applied to values outside ``segment_order``.
    invalid_label: str = "INVALID"

    def __post_init__(self) -> None:
        self.risk_direction = _enum(RiskDirection, self.risk_direction, "risk_direction")
        self.segment_order = [str(s).strip() for s in self.segment_order]
        if len(self.segment_order) < 2:
            raise ConfigError("segment_order must contain at least two segments.")
        if len(set(self.segment_order)) != len(self.segment_order):
            raise ConfigError(f"segment_order contains duplicates: {self.segment_order}")
        if not 1 <= int(self.riskier_segment_count) < len(self.segment_order):
            raise ConfigError(
                "riskier_segment_count must be >=1 and < number of segments "
                f"(got {self.riskier_segment_count} for {len(self.segment_order)} segments)."
            )
        self.riskier_segment_count = int(self.riskier_segment_count)

    # -- derived views ------------------------------------------------------
    @property
    def n_segments(self) -> int:
        return len(self.segment_order)

    @property
    def risk_rank(self) -> Dict[str, int]:
        """Map segment -> ordinal risk rank where **higher means riskier**."""
        n = self.n_segments
        if self.risk_direction is RiskDirection.A_TO_E_INCREASING:
            return {seg: idx + 1 for idx, seg in enumerate(self.segment_order)}
        return {seg: n - idx for idx, seg in enumerate(self.segment_order)}

    @property
    def segments_by_risk(self) -> List[str]:
        """Segments ordered safest -> riskiest."""
        return sorted(self.segment_order, key=lambda s: self.risk_rank[s])

    @property
    def riskier_segments(self) -> List[str]:
        """The ``riskier_segment_count`` riskiest segments."""
        return self.segments_by_risk[-self.riskier_segment_count:]

    @property
    def safest_segment(self) -> str:
        return self.segments_by_risk[0]

    @property
    def riskiest_segment(self) -> str:
        return self.segments_by_risk[-1]

    def is_valid(self, value: Any) -> bool:
        return isinstance(value, str) and value.strip() in set(self.segment_order)


# ---------------------------------------------------------------------------
# Smoothing / threshold configuration
# ---------------------------------------------------------------------------


@dataclass
class SmoothingConfig:
    """Documented zero-frequency treatment, surfaced in the Excel report."""

    method: SmoothingMethod = SmoothingMethod.PSEUDOCOUNT
    #: Additive pseudocount added to *every* bin count before normalising.
    #: 0.5 is the Jeffreys prior; 1.0 is Laplace.  Keep small relative to the
    #: smallest expected bin count.
    pseudocount: float = 0.5
    #: Numerical tolerance used when asserting that bin contributions sum to
    #: the reported total.
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        self.method = _enum(SmoothingMethod, self.method, "smoothing.method")
        self.pseudocount = float(self.pseudocount)
        if self.pseudocount < 0:
            raise ConfigError("smoothing.pseudocount must be >= 0.")
        if self.method is SmoothingMethod.PSEUDOCOUNT and self.pseudocount == 0:
            raise ConfigError(
                "smoothing.method=PSEUDOCOUNT with pseudocount=0 cannot protect against "
                "zero-frequency bins. Use a positive pseudocount or method=DROP_EMPTY_BINS."
            )
        self.tolerance = float(self.tolerance)

    def describe(self) -> str:
        if self.method is SmoothingMethod.PSEUDOCOUNT:
            return (
                f"Additive (pseudocount) smoothing: p_i = (n_i + {self.pseudocount:g}) / "
                f"(N + k x {self.pseudocount:g}) applied to BOTH the reference and the "
                "monitoring distribution before the PSI/CSI sum."
            )
        return (
            "Empty-bin exclusion: bins with a zero count on either side are dropped from "
            "the sum and counted in 'bins_dropped'. The remaining proportions are NOT "
            "re-normalised, so the reported index is a lower bound."
        )


@dataclass
class ThresholdBand:
    """A two-sided AMBER/RED threshold band for a 'higher is worse' metric."""

    amber: float
    red: float

    def __post_init__(self) -> None:
        self.amber = float(self.amber)
        self.red = float(self.red)
        if self.red < self.amber:
            raise ConfigError(f"ThresholdBand red ({self.red}) must be >= amber ({self.amber}).")

    def classify(self, value: Optional[float]) -> Severity:
        """Classify an **unrounded** metric value."""
        if value is None:
            return Severity.GREY
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return Severity.GREY
        if numeric != numeric:  # NaN
            return Severity.GREY
        if numeric >= self.red:
            return Severity.RED
        if numeric >= self.amber:
            return Severity.AMBER
        return Severity.GREEN

    def label(self) -> str:
        return f"GREEN < {self.amber:g} <= AMBER < {self.red:g} <= RED"


@dataclass
class ThresholdConfig:
    """All monitoring thresholds in one place, overridable by model governance.

    The PSI/CSI bands default to the commonly cited industry convention
    (0.10 / 0.25).  These are *monitoring guidelines*, not universal
    regulatory limits -- see ``docs/ASSUMPTIONS_AND_LIMITATIONS.md``.
    """

    psi: ThresholdBand = field(default_factory=lambda: ThresholdBand(0.10, 0.25))
    csi: ThresholdBand = field(default_factory=lambda: ThresholdBand(0.10, 0.25))
    #: Tighter band applied to variables flagged ``critical: true``.
    csi_critical: ThresholdBand = field(default_factory=lambda: ThresholdBand(0.05, 0.15))
    #: Per-variable overrides, e.g. {"MODEL_1_VARIABLE_3": {"amber": .05, "red": .12}}
    csi_variable_overrides: Dict[str, ThresholdBand] = field(default_factory=dict)

    # -- sample-size gates -------------------------------------------------
    #: Below this many reference records a PSI/CSI value is reported but
    #: labelled GREY (not trustworthy).
    min_reference_records: int = 500
    min_monitoring_records: int = 200
    #: Below this many records in a CSI bin comparison the variable is GREY.
    min_variable_records: int = 100

    # -- data quality bands (percentage points unless stated) --------------
    missingness_increase_pp: ThresholdBand = field(default_factory=lambda: ThresholdBand(5.0, 10.0))
    missingness_level_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(20.0, 50.0))
    unseen_category_rate_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(1.0, 5.0))
    duplicate_rate_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(1.0, 5.0))
    invalid_segment_rate_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(0.5, 2.0))
    missing_segment_rate_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(2.0, 5.0))
    #: Absolute relative change in quarterly record count vs previous quarter.
    record_count_change_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(30.0, 50.0))
    #: Share of the population sitting in a single segment.
    concentration_pct: ThresholdBand = field(default_factory=lambda: ThresholdBand(50.0, 70.0))
    #: Net migration toward riskier segments, in percentage points of matched customers.
    net_downgrade_pp: ThresholdBand = field(default_factory=lambda: ThresholdBand(5.0, 10.0))
    #: Share of the riskiest segments; increase vs reference in percentage points.
    riskier_share_increase_pp: ThresholdBand = field(default_factory=lambda: ThresholdBand(5.0, 10.0))

    # -- outcome performance (lower is worse -> handled explicitly) --------
    #: Minimum acceptable Gini; below ``red`` escalate, below ``amber`` investigate.
    min_gini_amber: float = 0.30
    min_gini_red: float = 0.20
    #: Minimum matured-outcome count before performance metrics are trusted.
    min_outcome_records: int = 300
    #: Minimum number of bads before AUC/Gini/KS are trusted.
    min_bad_count: int = 30

    def __post_init__(self) -> None:
        self.psi = _coerce_band(self.psi, "thresholds.psi")
        self.csi = _coerce_band(self.csi, "thresholds.csi")
        self.csi_critical = _coerce_band(self.csi_critical, "thresholds.csi_critical")
        self.csi_variable_overrides = {
            str(k): _coerce_band(v, f"thresholds.csi_variable_overrides[{k}]")
            for k, v in (self.csi_variable_overrides or {}).items()
        }
        for name in (
            "missingness_increase_pp", "missingness_level_pct", "unseen_category_rate_pct",
            "duplicate_rate_pct", "invalid_segment_rate_pct", "missing_segment_rate_pct",
            "record_count_change_pct", "concentration_pct", "net_downgrade_pp",
            "riskier_share_increase_pp",
        ):
            setattr(self, name, _coerce_band(getattr(self, name), f"thresholds.{name}"))
        for name in ("min_reference_records", "min_monitoring_records", "min_variable_records",
                     "min_outcome_records", "min_bad_count"):
            setattr(self, name, int(getattr(self, name)))
        if self.min_gini_red > self.min_gini_amber:
            raise ConfigError("thresholds.min_gini_red must be <= min_gini_amber.")

    def csi_band(self, variable: str, critical: bool = False) -> ThresholdBand:
        """Resolve the CSI band for one variable (override > critical > default)."""
        if variable in self.csi_variable_overrides:
            return self.csi_variable_overrides[variable]
        return self.csi_critical if critical else self.csi

    def classify_gini(self, gini: Optional[float]) -> Severity:
        """Gini is a 'higher is better' metric, so the band is inverted."""
        if gini is None or gini != gini:
            return Severity.GREY
        if gini < self.min_gini_red:
            return Severity.RED
        if gini < self.min_gini_amber:
            return Severity.AMBER
        return Severity.GREEN


def _coerce_band(value: Any, field_name: str) -> ThresholdBand:
    if isinstance(value, ThresholdBand):
        return value
    if isinstance(value, Mapping):
        if "amber" not in value or "red" not in value:
            raise ConfigError(f"{field_name} must define both 'amber' and 'red'.")
        return ThresholdBand(amber=value["amber"], red=value["red"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return ThresholdBand(amber=value[0], red=value[1])
    raise ConfigError(f"{field_name} must be a mapping with 'amber'/'red' keys, got {value!r}.")


# ---------------------------------------------------------------------------
# Variable specifications
# ---------------------------------------------------------------------------


@dataclass
class VariableSpec:
    """One monitored characteristic of one scorecard."""

    name: str                                   # column name in the DataFrame
    var_type: VariableType = VariableType.NUMERIC
    role: VariableRole = VariableRole.FEATURE
    label: Optional[str] = None                 # human readable name for reports
    #: Approved numeric bin edges (monotonic).  When given they take precedence
    #: over quantile binning, which is the whole point of "use the approved
    #: scorecard binning".
    bins: Optional[List[float]] = None
    #: Approved category order / grouping for CATEGORICAL and BINNED variables.
    categories: Optional[List[str]] = None
    #: Explicit mapping raw value -> reported bin, for WOE/approved bands.
    category_map: Optional[Dict[str, str]] = None
    #: Business-critical flag -> tighter CSI band (``thresholds.csi_critical``).
    critical: bool = False
    #: Companion subscore column, monitored as its own numeric characteristic.
    subscore_column: Optional[str] = None
    #: Number of quantile bins when ``bins`` is not supplied.
    n_bins: Optional[int] = None
    #: Skip this variable (kept in config for documentation purposes).
    enabled: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ConfigError("VariableSpec.name must be a non-empty column name.")
        self.var_type = _enum(VariableType, self.var_type, f"variable[{self.name}].type")
        self.role = _enum(VariableRole, self.role, f"variable[{self.name}].role")
        self.label = self.label or self.name
        if self.bins is not None:
            edges = [float(b) for b in self.bins]
            if len(edges) < 2:
                raise ConfigError(f"variable[{self.name}].bins needs at least 2 edges.")
            if any(b <= a for a, b in zip(edges, edges[1:])):
                raise ConfigError(f"variable[{self.name}].bins must be strictly increasing: {edges}")
            self.bins = edges
            self.var_type = VariableType.NUMERIC
        if self.categories is not None:
            self.categories = [str(c) for c in self.categories]
        if self.category_map is not None:
            self.category_map = {str(k): str(v) for k, v in self.category_map.items()}
        if self.n_bins is not None:
            self.n_bins = int(self.n_bins)
            if self.n_bins < 2:
                raise ConfigError(f"variable[{self.name}].n_bins must be >= 2.")


@dataclass
class VariableDiscoveryConfig:
    """Regex-driven discovery of a scorecard's columns.

    Real scorecard extracts carry hundreds or thousands of columns, so
    enumerating them by hand is not practical.  Discovery resolves the actual
    column list against the dataset header at run time; every resolved column
    still becomes a fully-fledged :class:`VariableSpec`, so explicit overrides
    and discovery can be mixed freely.
    """

    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    #: Suffix identifying variable-level subscores (monitored separately).
    subscore_suffix: Optional[str] = "_SUBSCORE"
    #: Treat discovered subscore columns as monitored characteristics.
    monitor_subscores: bool = True
    #: Columns with <= this many distinct values are treated as categorical
    #: when the dtype is numeric but clearly discrete (e.g. 0/1 flags).
    categorical_max_unique: int = 15
    enabled: bool = True

    def __post_init__(self) -> None:
        self.include_patterns = [str(p) for p in (self.include_patterns or [])]
        self.exclude_patterns = [str(p) for p in (self.exclude_patterns or [])]
        for pattern in self.include_patterns + self.exclude_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(f"Invalid discovery regex {pattern!r}: {exc}") from exc
        self.categorical_max_unique = int(self.categorical_max_unique)

    def matches(self, column: str) -> bool:
        if not self.enabled or not self.include_patterns:
            return False
        if not any(re.search(p, column) for p in self.include_patterns):
            return False
        if any(re.search(p, column) for p in self.exclude_patterns):
            return False
        return True

    def is_subscore(self, column: str) -> bool:
        return bool(self.subscore_suffix) and column.endswith(self.subscore_suffix)


# ---------------------------------------------------------------------------
# Reference period
# ---------------------------------------------------------------------------


@dataclass
class ReferencePeriodConfig:
    """The baseline population of ONE scorecard.

    Each scorecard owns its own reference window because scorecards are
    developed, implemented and retrained at different times.  The pipeline
    never substitutes one model's window for another's, and never silently
    falls back to "the whole dataset".
    """

    model: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    mode: ReferenceMode = ReferenceMode.DEVELOPMENT
    #: Optional pandas ``query`` string selecting an approved production
    #: baseline population (only meaningful for PRODUCTION_BASELINE).
    baseline_filter: Optional[str] = None
    #: Free-text provenance for the governance sheet.
    source_note: str = ""
    #: Resolved at run time by :mod:`reference_periods` (not user supplied).
    record_count: Optional[int] = None
    configured_by: str = "CONFIG_FILE"   # CONFIG_FILE | INTERACTIVE | DEFAULT

    def __post_init__(self) -> None:
        self.model = str(self.model).strip()
        self.mode = _enum(ReferenceMode, self.mode, f"reference[{self.model}].mode")
        self.start_date = _to_date(self.start_date, f"reference[{self.model}].start_date")
        self.end_date = _to_date(self.end_date, f"reference[{self.model}].end_date")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ConfigError(
                f"reference[{self.model}]: start_date {self.start_date} must be <= "
                f"end_date {self.end_date}."
            )

    @property
    def is_configured(self) -> bool:
        return self.start_date is not None and self.end_date is not None

    @property
    def label(self) -> str:
        if not self.is_configured:
            return "NOT CONFIGURED"
        return f"{self.start_date.isoformat()} to {self.end_date.isoformat()}"

    def describe(self) -> str:
        return f"{self.label} [{self.mode.value}]"


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Everything the pipeline needs to monitor ONE scorecard."""

    name: str
    segment_column: str
    #: Final / aggregate scorecard score, when available and its definition is
    #: known.  Monitored as a continuous characteristic -- never confused with
    #: the categorical segment.
    score_column: Optional[str] = None
    #: Model probability of default, for calibration / observed-vs-expected.
    pd_column: Optional[str] = None
    variables: List[VariableSpec] = field(default_factory=list)
    discovery: Optional[VariableDiscoveryConfig] = None
    reference: Optional[ReferencePeriodConfig] = None
    #: Optional pandas ``query`` restricting the population scored by this model
    #: (e.g. a product or channel cut).
    population_filter: Optional[str] = None
    description: str = ""
    owner: str = ""
    version: str = ""
    enabled: bool = True
    #: Populated by :meth:`resolve_variables`; True once discovery has run.
    variables_resolved: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ConfigError("ModelConfig.name must be non-empty.")
        self.segment_column = str(self.segment_column).strip()
        if not self.segment_column:
            raise ConfigError(f"model[{self.name}].segment_column must be non-empty.")
        self.variables = [v if isinstance(v, VariableSpec) else VariableSpec(**dict(v))
                          for v in (self.variables or [])]
        if self.discovery is not None and not isinstance(self.discovery, VariableDiscoveryConfig):
            self.discovery = VariableDiscoveryConfig(**dict(self.discovery))
        if self.reference is not None and not isinstance(self.reference, ReferencePeriodConfig):
            payload = dict(self.reference)
            payload.setdefault("model", self.name)
            self.reference = ReferencePeriodConfig(**payload)
        if self.reference is None:
            self.reference = ReferencePeriodConfig(model=self.name)

    # -- variable resolution ----------------------------------------------
    def explicit_variable_names(self) -> List[str]:
        return [v.name for v in self.variables]

    def resolve_variables(
        self,
        available_columns: Sequence[str],
        *,
        dtype_hints: Optional[Mapping[str, str]] = None,
        logger=None,
    ) -> Tuple[List[VariableSpec], List[str]]:
        """Expand discovery patterns and drop variables absent from the data.

        Returns ``(resolved_specs, warnings)``.  Explicit specs always win over
        discovered ones; subscore companion columns are added as separate
        monitored characteristics with ``role=SUBSCORE``.
        """
        warnings: List[str] = []
        available = list(available_columns)
        available_set = set(available)
        resolved: Dict[str, VariableSpec] = {}

        # 1) explicit specifications (authoritative)
        for spec in self.variables:
            if not spec.enabled:
                continue
            if spec.name not in available_set:
                warnings.append(
                    f"[{self.name}] configured variable '{spec.name}' is not present in the "
                    "dataset and will be skipped."
                )
                continue
            resolved[spec.name] = spec
            if spec.subscore_column:
                if spec.subscore_column in available_set:
                    resolved.setdefault(
                        spec.subscore_column,
                        VariableSpec(
                            name=spec.subscore_column,
                            var_type=VariableType.NUMERIC,
                            role=VariableRole.SUBSCORE,
                            label=f"{spec.label} (subscore)",
                            critical=spec.critical,
                            n_bins=spec.n_bins,
                        ),
                    )
                else:
                    warnings.append(
                        f"[{self.name}] subscore column '{spec.subscore_column}' declared for "
                        f"'{spec.name}' is missing from the dataset."
                    )

        # 2) discovery
        if self.discovery and self.discovery.enabled:
            for column in available:
                if column in resolved:
                    continue
                if not self.discovery.matches(column):
                    continue
                is_sub = self.discovery.is_subscore(column)
                if is_sub and not self.discovery.monitor_subscores:
                    continue
                hint = (dtype_hints or {}).get(column)
                var_type = VariableType.NUMERIC
                if hint == "categorical":
                    var_type = VariableType.CATEGORICAL
                resolved[column] = VariableSpec(
                    name=column,
                    var_type=var_type,
                    role=VariableRole.SUBSCORE if is_sub else VariableRole.FEATURE,
                    label=column,
                )

        # 3) the final score, monitored as a continuous characteristic
        if self.score_column:
            if self.score_column in available_set:
                resolved.setdefault(
                    self.score_column,
                    VariableSpec(
                        name=self.score_column,
                        var_type=VariableType.NUMERIC,
                        role=VariableRole.SCORE,
                        label=f"{self.name} final score",
                        critical=True,
                    ),
                )
            else:
                warnings.append(
                    f"[{self.name}] score_column '{self.score_column}' is not in the dataset; "
                    "score-level CSI and calibration will be unavailable."
                )

        ordered = sorted(resolved.values(), key=lambda s: (s.role.value, s.name))
        self.variables = ordered
        self.variables_resolved = True
        if logger is not None:
            for message in warnings:
                logger.warning(message)
        return ordered, warnings

    def required_columns(self) -> List[str]:
        """Columns this model needs, for column-pruned loading."""
        columns = [self.segment_column]
        if self.score_column:
            columns.append(self.score_column)
        if self.pd_column:
            columns.append(self.pd_column)
        for spec in self.variables:
            columns.append(spec.name)
            if spec.subscore_column:
                columns.append(spec.subscore_column)
        return list(dict.fromkeys(columns))

    def variable(self, name: str) -> Optional[VariableSpec]:
        for spec in self.variables:
            if spec.name == name:
                return spec
        return None


# ---------------------------------------------------------------------------
# Data source / outcome / runtime / excel
# ---------------------------------------------------------------------------


@dataclass
class DataSourceConfig:
    """Where the customer-level scorecard extract comes from."""

    path: Optional[str] = None
    fmt: str = "auto"                 # auto | csv | excel | parquet | dataframe
    sheet_name: Optional[Any] = 0     # Excel only
    delimiter: str = ","
    encoding: str = "utf-8"
    #: Read CSV in chunks of this many rows (memory-efficient path).
    chunksize: Optional[int] = None
    #: Hard cap on rows read (smoke tests only -- never used silently).
    nrows: Optional[int] = None
    #: Load only the columns the configuration actually needs.  This is the
    #: single biggest memory saving on wide scorecard extracts.
    prune_columns: bool = True
    #: Downcast numerics and convert low-cardinality text to ``category``.
    optimise_dtypes: bool = True
    #: Strings treated as missing on read, in addition to pandas defaults.
    na_values: List[str] = field(default_factory=lambda: ["", "NA", "N/A", "NULL", "null", "None", "-", "?", "#N/A"])

    def __post_init__(self) -> None:
        self.fmt = str(self.fmt).strip().lower()
        if self.fmt not in {"auto", "csv", "excel", "parquet", "dataframe"}:
            raise ConfigError(f"data_source.fmt={self.fmt!r} not supported.")
        if self.chunksize is not None:
            self.chunksize = int(self.chunksize)
        if self.nrows is not None:
            self.nrows = int(self.nrows)

    def resolved_format(self) -> str:
        if self.fmt != "auto":
            return self.fmt
        if not self.path:
            return "dataframe"
        suffix = Path(self.path).suffix.lower()
        if suffix in {".csv", ".txt", ".tsv", ".gz", ".bz2", ".zip"}:
            return "csv"
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            return "excel"
        if suffix in {".parquet", ".pq"}:
            return "parquet"
        raise ConfigError(
            f"Cannot infer format from path {self.path!r}; set data_source.fmt explicitly."
        )


@dataclass
class OutcomeConfig:
    """Optional default / delinquency outcome, enabling performance metrics.

    When no outcome is supplied the pipeline reports performance metrics as
    ``NOT AVAILABLE`` with the missing dependency named.  It never fabricates
    discrimination statistics from segment distributions alone.
    """

    outcome_column: Optional[str] = None          # binary 1 = bad / event
    bad_values: List[Any] = field(default_factory=lambda: [1, "1", True, "Y", "BAD", "DEFAULT"])
    good_values: List[Any] = field(default_factory=lambda: [0, "0", False, "N", "GOOD"])
    #: Continuous score used for AUC/KS.  Defaults to the model's score column,
    #: falling back to the ordinal segment rank (documented as a coarse proxy).
    score_column: Optional[str] = None
    #: True when a HIGHER score means HIGHER risk.
    score_higher_is_riskier: bool = False
    #: Expected PD for observed-vs-expected / calibration.
    pd_column: Optional[str] = None
    #: Performance window: an account is only classified good/bad once this many
    #: months have elapsed since LOGIN_DATE.
    observation_window_months: int = 12
    #: Column flagging a matured performance window; derived when absent.
    maturity_column: Optional[str] = None
    #: Date the outcome data was extracted; defaults to max(LOGIN_DATE).
    outcome_as_of: Optional[date] = None
    #: Drop immature accounts instead of treating a not-yet-bad account as good.
    exclude_immature: bool = True
    #: Origination date column for vintage analysis (defaults to LOGIN_DATE).
    vintage_date_column: Optional[str] = None

    def __post_init__(self) -> None:
        self.outcome_as_of = _to_date(self.outcome_as_of, "outcome.outcome_as_of")
        self.observation_window_months = int(self.observation_window_months)
        if self.observation_window_months < 0:
            raise ConfigError("outcome.observation_window_months must be >= 0.")

    @property
    def enabled(self) -> bool:
        return bool(self.outcome_column)


@dataclass
class ExcelConfig:
    """Workbook generation options (presentation only -- no business logic)."""

    output_dir: str = "output"
    filename_template: str = "Scorecard_Monitoring_Report_{timestamp}.xlsx"
    #: Rows shown on the 'top N' dashboard tables.
    top_n: int = 25
    #: Hard cap per detail sheet.  Overflow is written to a CSV sidecar and the
    #: sheet carries an explicit truncation banner -- data is never dropped
    #: silently.  Excel's own hard limit is 1,048,576 rows.
    max_rows_per_sheet: int = 200_000
    #: Convert detail ranges into real Excel tables (slower on huge sheets).
    use_excel_tables: bool = True
    excel_table_row_limit: int = 50_000
    #: Add native Excel charts to the dashboard.
    include_charts: bool = True
    #: Also export matplotlib PNGs (useful for decks / email).
    export_png_charts: bool = False
    #: Conditional formatting is expensive on very large ranges.
    conditional_formatting_row_limit: int = 50_000
    freeze_panes: bool = True
    autofilter: bool = True

    def __post_init__(self) -> None:
        self.top_n = int(self.top_n)
        self.max_rows_per_sheet = max(1, int(self.max_rows_per_sheet))
        if self.max_rows_per_sheet > 1_000_000:
            self.max_rows_per_sheet = 1_000_000


@dataclass
class RuntimeConfig:
    """Execution-level behaviour."""

    #: Quarter handling.  Incomplete (partial) quarters are identified and
    #: excluded from the primary comparison unless this is switched on.
    include_partial_quarters: bool = False
    #: Minimum records for a quarter to be monitored at all.
    min_quarter_records: int = 30
    #: Customer-level multi-record policy for QoQ migration.
    multi_record_policy: MultiRecordPolicy = MultiRecordPolicy.LATEST
    #: Process CSI variables in batches of this size (memory control).
    csi_variable_batch_size: int = 50
    #: Only monitor these variables (empty = all configured/discovered).
    csi_variable_whitelist: List[str] = field(default_factory=list)
    #: Keep only the N highest-CSI variables in the *summary* sheet.  The full
    #: detail table is always retained.
    csi_top_n: Optional[int] = None
    #: Number of quantile bins for numeric characteristics without approved bins.
    default_numeric_bins: int = 10
    #: Categories below this share of the reference population are grouped
    #: into OTHER.
    rare_category_min_share: float = 0.01
    #: Which population a model's characteristic distributions are measured on.
    #: ``MODEL_OUTPUT`` (default): only records the scorecard actually scored,
    #: which is the right denominator when a model covers part of the book.
    #: ``ALL_RECORDS``: every record in the period, so applications the model
    #: failed to score appear in the MISSING bin and move CSI directly.
    #: Unscored volume is monitored either way by MER and Data Quality.
    csi_scope: str = "MODEL_OUTPUT"
    #: Keep GREEN rows in the alert register.  They are what makes the
    #: dashboard's GREEN count meaningful (evaluated and passed, rather than
    #: not evaluated); set False for a breach-only register.
    alert_include_green: bool = True
    #: Also compute CSI against the previous quarter (using the reference bin
    #: boundaries).  Off by default: it doubles the CSI workload on wide
    #: datasets, and the fixed-reference view is the governance requirement.
    csi_include_qoq: bool = False
    #: Include MISSING/INVALID scorecard output as explicit bins in the segment
    #: PSI.  Default False: PSI is computed over the configured segments only,
    #: and missing / invalid output is monitored separately in MER and Data
    #: Quality.  Set True to let unscored volume move PSI directly.
    psi_include_missing_segment: bool = False
    #: Decimal places used for *display* (threshold tests always use raw values).
    display_decimals: int = 4
    #: Fail fast instead of prompting when a reference period is missing.
    non_interactive: bool = False
    random_seed: int = 42
    log_level: str = "INFO"
    log_to_file: bool = True

    def __post_init__(self) -> None:
        self.multi_record_policy = _enum(
            MultiRecordPolicy, self.multi_record_policy, "runtime.multi_record_policy"
        )
        self.csi_variable_batch_size = max(1, int(self.csi_variable_batch_size))
        self.default_numeric_bins = int(self.default_numeric_bins)
        if self.default_numeric_bins < 2:
            raise ConfigError("runtime.default_numeric_bins must be >= 2.")
        self.rare_category_min_share = float(self.rare_category_min_share)
        if not 0.0 <= self.rare_category_min_share < 1.0:
            raise ConfigError("runtime.rare_category_min_share must be in [0, 1).")
        if self.csi_top_n is not None:
            self.csi_top_n = int(self.csi_top_n)
        self.csi_variable_whitelist = [str(v) for v in (self.csi_variable_whitelist or [])]
        self.display_decimals = int(self.display_decimals)
        self.csi_scope = str(self.csi_scope).strip().upper()
        if self.csi_scope not in {"MODEL_OUTPUT", "ALL_RECORDS"}:
            raise ConfigError(
                "runtime.csi_scope must be 'MODEL_OUTPUT' or 'ALL_RECORDS', got "
                f"{self.csi_scope!r}."
            )


# ---------------------------------------------------------------------------
# Top level configuration
# ---------------------------------------------------------------------------


@dataclass
class MonitoringConfig:
    """Root configuration object consumed by :mod:`model_monitoring.main`."""

    # identification / governance
    report_title: str = "Credit Risk Scorecard Monitoring Report"
    entity_name: str = "NBFC Risk Analytics"
    framework_version: str = "1.0.0"
    prepared_by: str = ""
    governance_note: str = (
        "PSI/CSI thresholds are conventional monitoring guidelines. A breach triggers "
        "investigation under the model risk management policy; it does not by itself "
        "invalidate a scorecard, nor does compliance with them establish regulatory approval."
    )

    # common columns
    lan_column: str = "LAN_ID"
    customer_column: str = "CUSTOMER_ID"
    date_column: str = "LOGIN_DATE"
    extra_columns: List[str] = field(default_factory=list)

    # components
    models: List[ModelConfig] = field(default_factory=list)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    outcome: OutcomeConfig = field(default_factory=OutcomeConfig)
    excel: ExcelConfig = field(default_factory=ExcelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    #: Set at run time; used in filenames and the dashboard header.
    execution_timestamp: Optional[datetime] = None
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.models = [m if isinstance(m, ModelConfig) else ModelConfig(**dict(m))
                       for m in (self.models or [])]
        if not isinstance(self.segments, SegmentConfig):
            self.segments = SegmentConfig(**dict(self.segments))
        if not isinstance(self.thresholds, ThresholdConfig):
            self.thresholds = ThresholdConfig(**dict(self.thresholds))
        if not isinstance(self.smoothing, SmoothingConfig):
            self.smoothing = SmoothingConfig(**dict(self.smoothing))
        if not isinstance(self.data_source, DataSourceConfig):
            self.data_source = DataSourceConfig(**dict(self.data_source))
        if not isinstance(self.outcome, OutcomeConfig):
            self.outcome = OutcomeConfig(**dict(self.outcome))
        if not isinstance(self.excel, ExcelConfig):
            self.excel = ExcelConfig(**dict(self.excel))
        if not isinstance(self.runtime, RuntimeConfig):
            self.runtime = RuntimeConfig(**dict(self.runtime))
        if self.execution_timestamp is None:
            self.execution_timestamp = datetime.now()
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ConfigError(f"Duplicate model names in configuration: {names}")

    # -- accessors ---------------------------------------------------------
    @property
    def active_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]

    def model(self, name: str) -> ModelConfig:
        for candidate in self.models:
            if candidate.name == name:
                return candidate
        raise ConfigError(f"Unknown model {name!r}. Configured: {[m.name for m in self.models]}")

    @property
    def common_columns(self) -> List[str]:
        columns = [self.lan_column, self.customer_column, self.date_column]
        columns.extend(self.extra_columns or [])
        if self.outcome.enabled:
            columns.append(self.outcome.outcome_column)
            if self.outcome.score_column:
                columns.append(self.outcome.score_column)
            if self.outcome.pd_column:
                columns.append(self.outcome.pd_column)
            if self.outcome.maturity_column:
                columns.append(self.outcome.maturity_column)
            if self.outcome.vintage_date_column:
                columns.append(self.outcome.vintage_date_column)
        return [c for c in dict.fromkeys(columns) if c]

    def required_columns(self) -> List[str]:
        columns = list(self.common_columns)
        for model in self.active_models:
            columns.extend(model.required_columns())
        return list(dict.fromkeys(columns))

    def resolve_variables(self, available_columns: Sequence[str], *, dtype_hints=None,
                          logger=None) -> List[str]:
        """Resolve every model's variable list against the dataset header."""
        all_warnings: List[str] = []
        for model in self.active_models:
            _, warnings = model.resolve_variables(
                available_columns, dtype_hints=dtype_hints, logger=logger
            )
            all_warnings.extend(warnings)
        return all_warnings

    def monitored_variables(self, model_name: str) -> List[VariableSpec]:
        """Variables for one model after whitelist / enabled filtering."""
        model = self.model(model_name)
        whitelist = set(self.runtime.csi_variable_whitelist or [])
        specs = [s for s in model.variables if s.enabled]
        if whitelist:
            specs = [s for s in specs if s.name in whitelist]
        return specs

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON/YAML-safe dictionary, used for the Configuration sheet."""
        return _jsonify(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MonitoringConfig":
        """Build a configuration from a nested mapping.

        Accepts both the compact ``models: {MODEL_1: {...}}`` mapping form and
        the explicit ``models: [{name: MODEL_1, ...}]`` list form.
        """
        data = dict(payload or {})
        models = data.get("models")
        if isinstance(models, Mapping):
            expanded = []
            for name, spec in models.items():
                spec = dict(spec or {})
                spec.setdefault("name", name)
                spec.setdefault("segment_column", f"{name}_SEGMENT")
                expanded.append(spec)
            data["models"] = expanded
        # variables may be given as a mapping too
        for model_payload in data.get("models") or []:
            if not isinstance(model_payload, Mapping):
                continue
            model_payload = dict(model_payload)
            variables = model_payload.get("variables")
            if isinstance(variables, Mapping):
                model_payload["variables"] = [
                    {**(dict(spec or {})), "name": var_name}
                    for var_name, spec in variables.items()
                ]
            if "type" in model_payload:
                model_payload["var_type"] = model_payload.pop("type")
        # normalise the 'type' alias inside variable specs
        normalised_models = []
        for model_payload in data.get("models") or []:
            model_payload = dict(model_payload)
            variables = []
            for spec in model_payload.get("variables") or []:
                spec = dict(spec)
                if "type" in spec:
                    spec["var_type"] = spec.pop("type")
                variables.append(spec)
            model_payload["variables"] = variables
            normalised_models.append(model_payload)
        if normalised_models or "models" in data:
            data["models"] = normalised_models
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"Unknown configuration keys: {sorted(unknown)}. Allowed: {sorted(known)}"
            )
        return cls(**data)

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Structural self-check.  Returns non-fatal warnings; raises on errors."""
        warnings: List[str] = []
        if not self.active_models:
            raise ConfigError("No enabled models configured -- nothing to monitor.")
        for model in self.active_models:
            if not model.variables and not (model.discovery and model.discovery.enabled):
                warnings.append(
                    f"[{model.name}] has neither explicit variables nor a discovery rule; "
                    "only segment-level stability (PSI) will be produced."
                )
            if model.reference is None or not model.reference.is_configured:
                warnings.append(
                    f"[{model.name}] reference period is not configured; it must be supplied "
                    "interactively or in the configuration file before PSI/CSI can run."
                )
            if model.reference and model.reference.mode is ReferenceMode.PRODUCTION_BASELINE \
                    and not model.reference.baseline_filter and not model.reference.is_configured:
                warnings.append(
                    f"[{model.name}] PRODUCTION_BASELINE mode needs either a date window or a "
                    "baseline_filter."
                )
        if self.outcome.enabled and not (
            self.outcome.score_column or any(m.score_column for m in self.active_models)
        ):
            warnings.append(
                "An outcome column is configured but no continuous score column is available; "
                "AUC/KS will fall back to the ordinal segment rank, which is a coarse proxy."
            )
        return warnings

    def describe_models(self) -> str:
        lines = []
        for model in self.models:
            state = "" if model.enabled else " (disabled)"
            ref = model.reference.describe() if model.reference else "NOT CONFIGURED"
            lines.append(
                f"  - {model.name}{state}: segment='{model.segment_column}', "
                f"variables={len(model.variables)}, reference={ref}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    """Recursively convert dataclass output into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime,)):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_config(path: str | os.PathLike) -> MonitoringConfig:
    """Load a YAML (preferred) or JSON configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    payload: Any
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                "PyYAML is required for YAML configuration files. Install it with "
                "'pip install PyYAML', or use the JSON configuration instead."
            ) from exc
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise ConfigError(f"Unsupported configuration extension {suffix!r}; use .yaml or .json.")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Configuration root must be a mapping, got {type(payload).__name__}.")
    config = MonitoringConfig.from_dict(payload)
    config.config_path = str(config_path.resolve())
    return config


def save_config(config: MonitoringConfig, path: str | os.PathLike) -> Path:
    """Persist the effective configuration (handy for reproducibility)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    if out.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            return out
        except ImportError:
            out = out.with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def default_config(
    model_names: Sequence[str] = tuple(DEFAULT_MODEL_NAMES),
    *,
    n_variables_per_model: int = 0,
    data_path: Optional[str] = None,
) -> MonitoringConfig:
    """Convention-driven configuration for the illustrative naming scheme.

    ``MODEL_i_SEGMENT`` / ``MODEL_i_VARIABLE_j`` / ``MODEL_i_VARIABLE_j_SUBSCORE``.
    Discovery patterns are used rather than hardcoded names so the same config
    works whether a model has 5 or 5,000 characteristics.
    """
    models: List[ModelConfig] = []
    for name in model_names:
        explicit = [
            VariableSpec(name=f"{name}_VARIABLE_{j}", subscore_column=f"{name}_VARIABLE_{j}_SUBSCORE")
            for j in range(1, n_variables_per_model + 1)
        ]
        models.append(
            ModelConfig(
                name=name,
                segment_column=f"{name}_SEGMENT",
                score_column=f"{name}_TOTAL_SCORE",
                variables=explicit,
                discovery=VariableDiscoveryConfig(
                    include_patterns=[rf"^{re.escape(name)}_VARIABLE_"],
                    exclude_patterns=[r"_WOE$"],
                    subscore_suffix="_SUBSCORE",
                ),
                reference=ReferencePeriodConfig(model=name),
            )
        )
    source = DataSourceConfig(path=data_path) if data_path else DataSourceConfig()
    return MonitoringConfig(models=models, data_source=source)


__all__ = [
    "SEGMENT_ORDER", "RISK_DIRECTION", "DEFAULT_MODEL_NAMES",
    "RiskDirection", "ReferenceMode", "VariableType", "VariableRole", "Severity",
    "MultiRecordPolicy", "SmoothingMethod",
    "ConfigError", "SegmentConfig", "SmoothingConfig", "ThresholdBand", "ThresholdConfig",
    "VariableSpec", "VariableDiscoveryConfig", "ReferencePeriodConfig", "ModelConfig",
    "DataSourceConfig", "OutcomeConfig", "ExcelConfig", "RuntimeConfig", "MonitoringConfig",
    "load_config", "save_config", "default_config", "severity_rank", "worst_severity",
]
