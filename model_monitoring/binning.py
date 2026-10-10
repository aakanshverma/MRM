"""Reference-period binning for characteristic stability monitoring.

The single most important rule in CSI binning: **bin boundaries are fitted once
on the reference population and then reused unchanged in every monitoring
quarter.**  Re-bucketing each quarter (for example, taking fresh deciles every
quarter) forces each quarter's distribution back towards uniform and therefore
hides exactly the drift CSI exists to detect.

Priority order for numeric characteristics
------------------------------------------
1. approved scorecard / WOE boundaries supplied in ``variables[].bins``
2. quantile boundaries fitted on the reference period
3. distinct-value fallback when the column is degenerate (near-constant)

Priority order for categorical characteristics
----------------------------------------------
1. approved grouping supplied in ``variables[].category_map``
2. approved category list supplied in ``variables[].categories``
3. reference categories, with rare levels grouped into ``OTHER``

Special bins
------------
``MISSING`` (null in the source), ``OTHER`` (rare/grouped reference levels) and
``UNSEEN`` (a level that never appeared in the reference period) are explicit,
always-present bins.  Keeping them visible is deliberate: a characteristic whose
missingness jumps from 2% to 30% is a production incident, and the bin detail
sheet must show it rather than quietly dropping those records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import RuntimeConfig, VariableSpec, VariableType
from .logging_config import get_logger
from .utils import MISSING_BIN, OTHER_BIN, UNSEEN_BIN, is_numeric

logger = get_logger("binning")

# Source labels reported on the CSI sheets.
SOURCE_APPROVED_BINS = "APPROVED_BINS"
SOURCE_REFERENCE_QUANTILES = "REFERENCE_QUANTILES"
SOURCE_DISTINCT_VALUES = "REFERENCE_DISTINCT_VALUES"
SOURCE_CATEGORY_MAP = "APPROVED_CATEGORY_MAP"
SOURCE_APPROVED_CATEGORIES = "APPROVED_CATEGORY_LIST"
SOURCE_REFERENCE_CATEGORIES = "REFERENCE_CATEGORIES"


def canonical_value(value: Any) -> Any:
    """Canonical string form of a raw category value.

    Fitting and application must agree on the key, and they do not see the same
    dtype: a reference column may arrive as ``int64`` while the monitoring
    column arrives as ``float64`` (because a quarter contains a null), turning
    ``0`` into ``0.0``.  Normalising integral floats to their integer form and
    booleans to ``True``/``False`` keeps the lookup stable across quarters.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:   # NaN
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return repr(numeric) if not np.isfinite(numeric) else f"{numeric:.12g}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text


@dataclass
class BinApplication:
    """Result of applying a bin definition to one population."""

    labels: pd.Series                  # bin label per record
    counts: pd.Series                  # counts aligned to the definition's bins
    total: int
    missing_count: int
    unseen_count: int
    other_count: int
    infinite_count: int = 0

    @property
    def missing_pct(self) -> Optional[float]:
        return None if self.total == 0 else self.missing_count / self.total * 100.0

    @property
    def unseen_pct(self) -> Optional[float]:
        return None if self.total == 0 else self.unseen_count / self.total * 100.0

    @property
    def other_pct(self) -> Optional[float]:
        return None if self.total == 0 else self.other_count / self.total * 100.0

    @property
    def unseen_values(self) -> List[str]:
        return getattr(self, "_unseen_values", [])


@dataclass
class BinDefinition:
    """Immutable binning scheme fitted on a model's reference period."""

    variable: str
    var_type: VariableType
    bins: List[str]                               # ordered bin labels
    source: str
    edges: Optional[np.ndarray] = None            # numeric only
    value_to_bin: Optional[Dict[Any, str]] = None  # categorical only
    reference_total: int = 0
    reference_missing: int = 0
    reference_distinct: int = 0
    notes: List[str] = field(default_factory=list)
    usable: bool = True
    reason: str = ""

    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def is_numeric(self) -> bool:
        return self.edges is not None

    def describe(self) -> str:
        if self.is_numeric and self.edges is not None:
            interior = self.edges[1:-1]
            shown = ", ".join(f"{e:,.4g}" for e in interior[:12])
            if len(interior) > 12:
                shown += ", ..."
            return f"{self.source}: edges [{shown}] (outer edges extended to +/-inf)"
        real = [b for b in self.bins if b not in (MISSING_BIN, OTHER_BIN, UNSEEN_BIN)]
        shown = ", ".join(real[:12]) + (", ..." if len(real) > 12 else "")
        return f"{self.source}: levels [{shown}]"

    # -- application -------------------------------------------------------
    def apply(self, series: pd.Series) -> BinApplication:
        """Assign every record of ``series`` to one of :attr:`bins`."""
        total = int(len(series))
        if total == 0:
            empty = pd.Series([], dtype="object")
            return BinApplication(
                labels=empty,
                counts=pd.Series(0, index=pd.Index(self.bins, name="BIN"), dtype="int64"),
                total=0, missing_count=0, unseen_count=0, other_count=0,
            )

        missing_mask = series.isna()
        infinite_count = 0
        unseen_values: List[str] = []

        if self.is_numeric:
            values = pd.to_numeric(series, errors="coerce")
            # values that fail numeric coercion become missing; count them
            missing_mask = values.isna()
            numeric_array = values.to_numpy(dtype="float64", na_value=np.nan)
            infinite_count = int(np.isinf(numeric_array).sum())
            interval_labels = self.bins[: len(self.edges) - 1]
            assigned = pd.cut(
                values, bins=self.edges, labels=interval_labels,
                right=True, include_lowest=True, ordered=False,
            ).astype("object")
            labels = pd.Series(assigned, index=series.index, dtype="object")
            labels[missing_mask] = MISSING_BIN
            # Anything still unassigned (should not happen with +/-inf edges)
            stranded = labels.isna()
            if stranded.any():
                labels[stranded] = MISSING_BIN
                missing_mask = missing_mask | stranded
            unseen_count = 0
            other_count = 0
        else:
            as_text = series.astype("object").map(canonical_value)
            missing_mask = as_text.isna() | (as_text.astype("string").fillna("") == "")
            mapping = self.value_to_bin or {}
            mapped = as_text.map(lambda v: mapping.get(v)) if mapping else pd.Series(
                None, index=series.index, dtype="object"
            )
            labels = pd.Series(mapped, index=series.index, dtype="object")
            unseen_mask = labels.isna() & ~missing_mask
            if unseen_mask.any():
                unseen_values = sorted(
                    {str(v) for v in as_text[unseen_mask].dropna().unique()[:25]}
                )
            labels[unseen_mask] = UNSEEN_BIN
            labels[missing_mask] = MISSING_BIN
            unseen_count = int(unseen_mask.sum())
            other_count = int((labels == OTHER_BIN).sum())

        counts = labels.value_counts(dropna=False)
        counts.index = counts.index.map(lambda v: str(v))
        aligned = counts.reindex(self.bins).fillna(0).astype("int64")
        aligned.index.name = "BIN"
        unaccounted = int(counts.sum() - aligned.sum())
        if unaccounted:  # pragma: no cover - defensive
            logger.warning(
                "Variable '%s': %d record(s) fell outside the fitted bins and were dropped "
                "from the CSI distribution.", self.variable, unaccounted,
            )
        application = BinApplication(
            labels=labels,
            counts=aligned,
            total=total,
            missing_count=int(missing_mask.sum()),
            unseen_count=unseen_count,
            other_count=other_count if not self.is_numeric else int(
                aligned.get(OTHER_BIN, 0)
            ),
            infinite_count=infinite_count,
        )
        application._unseen_values = unseen_values  # type: ignore[attr-defined]
        return application


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_bin_definition(
    reference_series: pd.Series,
    spec: VariableSpec,
    runtime: RuntimeConfig,
    *,
    kind_hint: Optional[str] = None,
) -> BinDefinition:
    """Fit a :class:`BinDefinition` on one variable's reference population."""
    notes: List[str] = []
    total = int(len(reference_series))
    missing = int(reference_series.isna().sum())

    if total == 0:
        return BinDefinition(
            variable=spec.name, var_type=spec.var_type, bins=[MISSING_BIN],
            source="UNAVAILABLE", reference_total=0, reference_missing=0,
            usable=False,
            reason="The reference population contains no records for this characteristic.",
        )

    non_missing = reference_series.dropna()
    if non_missing.empty:
        return BinDefinition(
            variable=spec.name, var_type=spec.var_type, bins=[MISSING_BIN],
            source="UNAVAILABLE", reference_total=total, reference_missing=missing,
            usable=False,
            reason=(
                "Every reference-period value for this characteristic is missing, so no bin "
                "boundaries can be derived."
            ),
        )

    treat_as_numeric = _treat_as_numeric(reference_series, spec, kind_hint, runtime)
    if treat_as_numeric:
        return _fit_numeric(reference_series, non_missing, spec, runtime, notes, total, missing)
    return _fit_categorical(reference_series, non_missing, spec, runtime, notes, total, missing)


def _treat_as_numeric(
    series: pd.Series, spec: VariableSpec, kind_hint: Optional[str], runtime: RuntimeConfig
) -> bool:
    if spec.bins is not None:
        return True
    if spec.var_type in (VariableType.CATEGORICAL, VariableType.BINNED):
        return False
    if spec.category_map or spec.categories:
        return False
    if not is_numeric(series):
        return False
    if kind_hint == "categorical":
        return False
    return True


def _fit_numeric(
    series: pd.Series,
    non_missing: pd.Series,
    spec: VariableSpec,
    runtime: RuntimeConfig,
    notes: List[str],
    total: int,
    missing: int,
) -> BinDefinition:
    values = pd.to_numeric(non_missing, errors="coerce").dropna()
    array = values.to_numpy(dtype="float64")
    n_infinite = int(np.isinf(array).sum())
    finite = array[np.isfinite(array)]
    if n_infinite:
        notes.append(
            f"{n_infinite:,} infinite value(s) in the reference period are clipped into the "
            "outer bins; the boundaries themselves are fitted on finite values only."
        )
    if finite.size == 0:
        return BinDefinition(
            variable=spec.name, var_type=VariableType.NUMERIC, bins=[MISSING_BIN],
            source="UNAVAILABLE", reference_total=total, reference_missing=missing,
            usable=False, notes=notes,
            reason="No finite reference values; numeric boundaries cannot be fitted.",
        )

    if spec.bins is not None:
        interior = [float(b) for b in spec.bins]
        source = SOURCE_APPROVED_BINS
        notes.append(
            "Approved scorecard boundaries used verbatim; the outer edges are extended to "
            "+/-inf so out-of-range values are retained rather than becoming missing."
        )
    else:
        requested = spec.n_bins or runtime.default_numeric_bins
        quantiles = np.nanquantile(finite, np.linspace(0.0, 1.0, requested + 1))
        interior = sorted({float(q) for q in quantiles[1:-1]})
        source = SOURCE_REFERENCE_QUANTILES
        if len(interior) < requested - 1:
            notes.append(
                f"{requested} quantile bins were requested but the reference distribution "
                f"supports only {len(interior) + 1} distinct boundar(y/ies) (ties in the "
                "underlying values)."
            )

    edges = np.array([-np.inf, *interior, np.inf], dtype="float64")
    edges = np.unique(edges)
    if edges.size < 3:
        # Degenerate: a (near-)constant column.  Fall back to distinct values so
        # the characteristic is still monitored rather than silently skipped.
        distinct = sorted({float(v) for v in np.unique(finite)})[:50]
        if len(distinct) <= 25:
            notes.append(
                "The reference distribution is (near-)constant, so the characteristic is "
                "monitored on its distinct values instead of quantile bins."
            )
            value_to_bin = {canonical_value(v): _format_number(v) for v in distinct}
            bins = [_format_number(v) for v in distinct] + [UNSEEN_BIN, MISSING_BIN]
            return BinDefinition(
                variable=spec.name, var_type=VariableType.CATEGORICAL,
                bins=list(dict.fromkeys(bins)), source=SOURCE_DISTINCT_VALUES,
                value_to_bin=value_to_bin, reference_total=total,
                reference_missing=missing, reference_distinct=len(distinct), notes=notes,
            )
        notes.append("Only one usable boundary could be derived; the comparison is coarse.")
        edges = np.array([-np.inf, float(np.nanmedian(finite)), np.inf])

    labels = _interval_labels(edges)
    bins = [*labels, MISSING_BIN]
    return BinDefinition(
        variable=spec.name,
        var_type=VariableType.NUMERIC,
        bins=bins,
        source=source,
        edges=edges,
        reference_total=total,
        reference_missing=missing,
        reference_distinct=int(values.nunique()),
        notes=notes,
    )


def _fit_categorical(
    series: pd.Series,
    non_missing: pd.Series,
    spec: VariableSpec,
    runtime: RuntimeConfig,
    notes: List[str],
    total: int,
    missing: int,
) -> BinDefinition:
    as_text = non_missing.astype("object").map(canonical_value)
    counts = as_text.value_counts()
    distinct = int(counts.shape[0])
    value_to_bin: Dict[Any, str] = {}

    if spec.category_map:
        for raw, target in spec.category_map.items():
            value_to_bin[canonical_value(raw)] = str(target)
        ordered = list(dict.fromkeys(spec.category_map.values()))
        unmapped = [v for v in counts.index if canonical_value(v) not in value_to_bin]
        if unmapped:
            for value in unmapped:
                value_to_bin[canonical_value(value)] = OTHER_BIN
            notes.append(
                f"{len(unmapped)} reference level(s) are absent from the approved category map "
                f"and were grouped into {OTHER_BIN}."
            )
            ordered.append(OTHER_BIN)
        source = SOURCE_CATEGORY_MAP
        bins = list(dict.fromkeys([*ordered, UNSEEN_BIN, MISSING_BIN]))
    elif spec.categories:
        approved = [str(c) for c in spec.categories]
        approved_keys = {canonical_value(c) for c in approved}
        for value in approved:
            value_to_bin[canonical_value(value)] = value
        extra = [str(v) for v in counts.index if canonical_value(v) not in approved_keys]
        if extra:
            for value in extra:
                value_to_bin[canonical_value(value)] = OTHER_BIN
            notes.append(
                f"{len(extra)} reference level(s) outside the approved category list were "
                f"grouped into {OTHER_BIN}."
            )
        source = SOURCE_APPROVED_CATEGORIES
        bins = list(dict.fromkeys([*approved, *( [OTHER_BIN] if extra else [] ), UNSEEN_BIN, MISSING_BIN]))
    else:
        shares = counts / max(1, int(counts.sum()))
        keep = [str(v) for v, share in shares.items() if share >= runtime.rare_category_min_share]
        rare = [str(v) for v in counts.index if str(v) not in set(keep)]
        if not keep:  # every level is rare -> keep the most frequent few
            keep = [str(v) for v in counts.index[: min(10, distinct)]]
            rare = [str(v) for v in counts.index if str(v) not in set(keep)]
            notes.append(
                "No level reached runtime.rare_category_min_share; the 10 most frequent "
                "reference levels are monitored individually."
            )
        for value in keep:
            value_to_bin[canonical_value(value)] = value
        for value in rare:
            value_to_bin[canonical_value(value)] = OTHER_BIN
        if rare:
            notes.append(
                f"{len(rare)} level(s) below {runtime.rare_category_min_share:.1%} of the "
                f"reference population were grouped into {OTHER_BIN}."
            )
        source = SOURCE_REFERENCE_CATEGORIES
        bins = list(dict.fromkeys([*sorted(keep), *( [OTHER_BIN] if rare else [] ), UNSEEN_BIN, MISSING_BIN]))

    return BinDefinition(
        variable=spec.name,
        var_type=VariableType.BINNED if spec.var_type is VariableType.BINNED else VariableType.CATEGORICAL,
        bins=bins,
        source=source,
        value_to_bin=value_to_bin,
        reference_total=total,
        reference_missing=missing,
        reference_distinct=distinct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4g}"


def _interval_labels(edges: np.ndarray) -> List[str]:
    """``['<= 10', '(10, 20]', '> 20']`` style labels, sorted by bin order."""
    labels: List[str] = []
    for low, high in zip(edges, edges[1:]):
        if np.isneginf(low):
            labels.append(f"01. <= {_format_number(high)}")
        elif np.isposinf(high):
            labels.append(f"{len(labels) + 1:02d}. > {_format_number(low)}")
        else:
            labels.append(
                f"{len(labels) + 1:02d}. ({_format_number(low)}, {_format_number(high)}]"
            )
    return labels


def fit_bin_definitions(
    reference_frame: pd.DataFrame,
    specs: Sequence[VariableSpec],
    runtime: RuntimeConfig,
    *,
    kind_hints: Optional[Dict[str, str]] = None,
) -> Dict[str, BinDefinition]:
    """Fit bin definitions for several variables at once."""
    definitions: Dict[str, BinDefinition] = {}
    for spec in specs:
        if spec.name not in reference_frame.columns:
            continue
        definitions[spec.name] = fit_bin_definition(
            reference_frame[spec.name], spec, runtime,
            kind_hint=(kind_hints or {}).get(spec.name),
        )
    return definitions


__all__ = [
    "BinDefinition", "BinApplication", "canonical_value",
    "fit_bin_definition", "fit_bin_definitions",
    "SOURCE_APPROVED_BINS", "SOURCE_REFERENCE_QUANTILES", "SOURCE_DISTINCT_VALUES",
    "SOURCE_CATEGORY_MAP", "SOURCE_APPROVED_CATEGORIES", "SOURCE_REFERENCE_CATEGORIES",
]
