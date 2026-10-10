"""Schema validation, frame preparation and data-quality monitoring.

Three responsibilities:

1.  :func:`validate_schema` -- fail fast on a structurally unusable extract
    (missing key columns, unparseable dates, no scorecard output at all).
2.  :func:`prepare_frame` -- add the derived columns every downstream module
    relies on: the quarter label and a *normalised* segment column per model
    where every value is either a configured segment, ``MISSING`` or
    ``INVALID``.  Nothing downstream has to re-implement that cleaning.
3.  :func:`run_data_quality_checks` -- the Data Quality sheet: missingness,
    duplicates, invalid segments and dates, outliers and infinities,
    unexpected categories, record-count anomalies, model-output availability
    and data freshness.

Every check emits the same row schema (:data:`CHECK_COLUMNS`) so the Excel
writer and the alert engine can consume them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    MonitoringConfig,
    ReferencePeriodConfig,
    Severity,
    ThresholdConfig,
    VariableRole,
    VariableType,
)
from .logging_config import ExecutionLog, get_logger
from .utils import (
    MISSING_BIN,
    QuarterInfo,
    count_infinite,
    is_numeric,
    pct,
    to_quarter_series,
)

logger = get_logger("data_validation")

#: Internal derived columns.  Prefixed to avoid colliding with source columns.
QUARTER_COLUMN = "__QUARTER"
SEGMENT_PREFIX = "__SEGMENT_"
SEGMENT_RANK_PREFIX = "__SEGRANK_"
MATURITY_COLUMN = "__OUTCOME_MATURED"
OUTCOME_COLUMN = "__OUTCOME_BINARY"

CHECK_COLUMNS = [
    "CHECK_ID", "CATEGORY", "CHECK", "MODEL", "QUARTER", "SCOPE",
    "METRIC", "VALUE", "UNIT", "THRESHOLD", "SEVERITY",
    "RECORDS_AFFECTED", "DETAIL", "RECOMMENDED_ACTION",
]


class ValidationError(RuntimeError):
    """Raised when the dataset cannot support any monitoring at all."""


@dataclass
class ValidationReport:
    """Outcome of schema validation."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    date_min: Optional[date] = None
    date_max: Optional[date] = None
    models_with_output: List[str] = field(default_factory=list)
    models_without_output: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError(
                "Input validation failed:\n  - " + "\n  - ".join(self.errors)
            )

    def summary(self) -> str:
        span = (
            f"{self.date_min} to {self.date_max}" if self.date_min and self.date_max else "unknown"
        )
        return (
            f"{self.row_count:,} rows / {self.column_count} columns; date span {span}; "
            f"{len(self.models_with_output)} model(s) with usable output; "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


def validate_schema(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
) -> ValidationReport:
    """Check that the extract can support monitoring at all."""
    errors: List[str] = []
    warnings: List[str] = []

    # -- mandatory common columns -----------------------------------------
    for label, column in (
        ("date_column", config.date_column),
        ("lan_column", config.lan_column),
        ("customer_column", config.customer_column),
    ):
        if column not in frame.columns:
            if label == "date_column":
                errors.append(
                    f"Mandatory {label} '{column}' is missing. Quarterly monitoring is "
                    "impossible without an application date."
                )
            else:
                warnings.append(
                    f"{label} '{column}' is missing; counts that depend on it will be "
                    "reported as NOT AVAILABLE and customer-level migration is disabled."
                )

    if frame.empty:
        errors.append("The dataset contains zero rows.")

    # -- dates -------------------------------------------------------------
    date_min = date_max = None
    if config.date_column in frame.columns:
        dates = pd.to_datetime(frame[config.date_column], errors="coerce")
        valid = dates.dropna()
        if valid.empty:
            errors.append(
                f"No parseable values in '{config.date_column}'. Expected dates such as "
                "YYYY-MM-DD."
            )
        else:
            date_min, date_max = valid.min().date(), valid.max().date()
            n_bad = int(dates.isna().sum())
            if n_bad:
                warnings.append(
                    f"{n_bad:,} record(s) have a missing/invalid '{config.date_column}' and "
                    "cannot be assigned to a monitoring quarter."
                )
            future = int((valid.dt.date > date.today()).sum())
            if future:
                warnings.append(
                    f"{future:,} record(s) have a future '{config.date_column}' "
                    f"(> {date.today().isoformat()}); check the extract logic."
                )

    # -- scorecard output --------------------------------------------------
    with_output: List[str] = []
    without_output: List[str] = []
    for model in config.active_models:
        column = model.segment_column
        if column not in frame.columns:
            without_output.append(model.name)
            warnings.append(
                f"[{model.name}] segment column '{column}' is missing; this scorecard "
                "cannot be monitored and will be skipped."
            )
            continue
        non_null = int(frame[column].notna().sum())
        if non_null == 0:
            without_output.append(model.name)
            warnings.append(
                f"[{model.name}] segment column '{column}' is present but entirely empty; "
                "this scorecard cannot be monitored."
            )
            continue
        with_output.append(model.name)
        valid_mask = _valid_segment_mask(frame[column], config)
        invalid = int((frame[column].notna() & ~valid_mask).sum())
        if invalid:
            unexpected = sorted(
                {
                    str(v) for v in frame.loc[frame[column].notna() & ~valid_mask, column]
                    .astype("string").dropna().unique()[:10]
                }
            )
            warnings.append(
                f"[{model.name}] {invalid:,} record(s) hold a segment value outside the "
                f"configured list. Examples: {unexpected}."
            )

    if not with_output:
        errors.append(
            "None of the configured scorecards has usable segment output. Check "
            "model[*].segment_column against the extract's column names."
        )

    # -- duplicate keys ----------------------------------------------------
    if config.lan_column in frame.columns:
        duplicated = int(frame[config.lan_column].duplicated().sum())
        if duplicated:
            warnings.append(
                f"{duplicated:,} duplicated '{config.lan_column}' value(s). Record counts and "
                "distributions may be inflated; see the Data Quality sheet."
            )

    report = ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        row_count=int(len(frame)),
        column_count=int(frame.shape[1]),
        date_min=date_min,
        date_max=date_max,
        models_with_output=with_output,
        models_without_output=without_output,
    )
    for message in warnings:
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
    for message in errors:
        logger.error(message)
        if execution_log is not None:
            execution_log.error(message)
    return report


def model_scope_mask(
    frame: pd.DataFrame, config: MonitoringConfig, model, *, scope: str = "MODEL_OUTPUT"
) -> pd.Series:
    """Rows that belong to one scorecard's population.

    A scorecard often covers part of the book (a product, a channel, or a newly
    implemented model still ramping up).  Measuring its characteristics against
    *every* row in the extract then reports 90% missingness for a perfectly
    healthy model, which is noise, not a finding.  ``MODEL_OUTPUT`` scopes the
    population to records the model actually scored; coverage of the rest is
    monitored separately by MER (missing output %) and Data Quality (DQ-041).
    """
    mask = pd.Series(True, index=frame.index)
    if model.population_filter:
        try:
            selected = frame.query(model.population_filter).index
            mask &= frame.index.isin(selected)
        except Exception as exc:  # pragma: no cover - configuration error
            logger.warning(
                "[%s] population_filter could not be evaluated (%s); using all records.",
                model.name, exc,
            )
    if str(scope).upper() == "MODEL_OUTPUT" and model.segment_column in frame.columns:
        mask &= frame[model.segment_column].notna()
    return mask.fillna(False)


def _valid_segment_mask(series: pd.Series, config: MonitoringConfig) -> pd.Series:
    allowed = set(config.segments.segment_order)
    as_text = series.astype("string").str.strip()
    return as_text.isin(allowed).fillna(False)


# ---------------------------------------------------------------------------
# 2. Frame preparation
# ---------------------------------------------------------------------------


def prepare_frame(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
) -> pd.DataFrame:
    """Return a copy with derived quarter / normalised-segment / outcome columns.

    * ``__QUARTER`` -- calendar quarter label from ``LOGIN_DATE`` (NA when the
      date is unusable).
    * ``__SEGMENT_<MODEL>`` -- the scorecard output cleaned to exactly one of
      the configured segments, ``MISSING`` or ``INVALID``.
    * ``__SEGRANK_<MODEL>`` -- ordinal risk rank (higher = riskier), NaN for
      MISSING/INVALID.
    * ``__OUTCOME_BINARY`` / ``__OUTCOME_MATURED`` -- when an outcome is configured.
    """
    prepared = frame.copy()
    segments = config.segments

    if config.date_column in prepared.columns:
        prepared[config.date_column] = pd.to_datetime(
            prepared[config.date_column], errors="coerce"
        )
        prepared[QUARTER_COLUMN] = to_quarter_series(prepared[config.date_column])
    else:
        prepared[QUARTER_COLUMN] = pd.Series(pd.NA, index=prepared.index, dtype="string")

    rank_map = segments.risk_rank
    for model in config.active_models:
        target = f"{SEGMENT_PREFIX}{model.name}"
        if model.segment_column not in prepared.columns:
            prepared[target] = segments.missing_label
            prepared[f"{SEGMENT_RANK_PREFIX}{model.name}"] = np.nan
            continue
        raw = prepared[model.segment_column].astype("string").str.strip()
        allowed = set(segments.segment_order)
        normalised = pd.Series(segments.invalid_label, index=prepared.index, dtype="object")
        normalised[raw.isin(allowed).fillna(False)] = raw[raw.isin(allowed).fillna(False)]
        normalised[raw.isna() | (raw == "")] = segments.missing_label
        prepared[target] = normalised
        prepared[f"{SEGMENT_RANK_PREFIX}{model.name}"] = (
            normalised.map(rank_map).astype("float64")
        )

    if config.outcome.enabled:
        prepared = _prepare_outcome(prepared, config, execution_log)
    return prepared


def _prepare_outcome(
    frame: pd.DataFrame, config: MonitoringConfig, execution_log: Optional[ExecutionLog]
) -> pd.DataFrame:
    """Normalise the outcome flag and derive performance-window maturity.

    Right-censoring matters: an account whose window has not elapsed is NOT a
    good customer, it is an unknown.  Such records are flagged immature and
    (by default) excluded from performance metrics rather than counted as good.
    """
    outcome_cfg = config.outcome
    column = outcome_cfg.outcome_column
    if column not in frame.columns:
        message = (
            f"Outcome column '{column}' is configured but absent from the extract; "
            "outcome-based performance monitoring is unavailable."
        )
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
        frame[OUTCOME_COLUMN] = np.nan
        frame[MATURITY_COLUMN] = False
        return frame

    series = frame[column]
    bad_values = {str(v).strip().upper() for v in outcome_cfg.bad_values}
    good_values = {str(v).strip().upper() for v in outcome_cfg.good_values}
    as_text = series.astype("string").str.strip().str.upper()
    binary = pd.Series(np.nan, index=frame.index, dtype="float64")
    binary[as_text.isin(bad_values).fillna(False)] = 1.0
    binary[as_text.isin(good_values).fillna(False)] = 0.0
    unmapped = int((series.notna() & binary.isna()).sum())
    if unmapped:
        examples = sorted({str(v) for v in series[series.notna() & binary.isna()].unique()[:8]})
        message = (
            f"{unmapped:,} outcome value(s) matched neither outcome.bad_values nor "
            f"outcome.good_values and were treated as unknown. Examples: {examples}."
        )
        logger.warning(message)
        if execution_log is not None:
            execution_log.warning(message)
    frame[OUTCOME_COLUMN] = binary

    # -- maturity ----------------------------------------------------------
    if outcome_cfg.maturity_column and outcome_cfg.maturity_column in frame.columns:
        frame[MATURITY_COLUMN] = (
            frame[outcome_cfg.maturity_column].astype("string").str.strip().str.upper()
            .isin({"1", "TRUE", "Y", "YES", "MATURE", "MATURED"})
            .fillna(False)
        )
    else:
        origin_column = outcome_cfg.vintage_date_column or config.date_column
        if origin_column in frame.columns:
            as_of = outcome_cfg.outcome_as_of
            if as_of is None:
                observed = pd.to_datetime(frame[config.date_column], errors="coerce").max()
                as_of = observed.date() if pd.notna(observed) else date.today()
            cutoff = pd.Timestamp(as_of) - pd.DateOffset(
                months=outcome_cfg.observation_window_months
            )
            origin = pd.to_datetime(frame[origin_column], errors="coerce")
            frame[MATURITY_COLUMN] = (origin <= cutoff).fillna(False)
        else:  # pragma: no cover - no date at all is already a fatal error
            frame[MATURITY_COLUMN] = False

    immature_bads = int(((frame[MATURITY_COLUMN] == False) & (frame[OUTCOME_COLUMN] == 1)).sum())  # noqa: E712
    if immature_bads:
        logger.info(
            "%d bad outcome(s) sit inside an unmatured performance window; they are retained "
            "as bads but their cohort is flagged incomplete.", immature_bads,
        )
    return frame


# ---------------------------------------------------------------------------
# 3. Data quality checks
# ---------------------------------------------------------------------------


def _check_row(
    check_id: str,
    category: str,
    check: str,
    *,
    model: str = "ALL",
    quarter: str = "ALL",
    scope: str = "DATASET",
    metric: str = "",
    value: Any = None,
    unit: str = "",
    threshold: str = "",
    severity: Severity = Severity.GREEN,
    records: Any = None,
    detail: str = "",
    action: str = "",
) -> Dict[str, Any]:
    return {
        "CHECK_ID": check_id,
        "CATEGORY": category,
        "CHECK": check,
        "MODEL": model,
        "QUARTER": quarter,
        "SCOPE": scope,
        "METRIC": metric,
        "VALUE": value,
        "UNIT": unit,
        "THRESHOLD": threshold,
        "SEVERITY": severity.value if isinstance(severity, Severity) else str(severity),
        "RECORDS_AFFECTED": records,
        "DETAIL": detail,
        "RECOMMENDED_ACTION": action,
    }


def run_data_quality_checks(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    quarters: Sequence[QuarterInfo],
    reference_periods: Optional[Dict[str, ReferencePeriodConfig]] = None,
    max_variables_scanned: int = 2_000,
    execution_log: Optional[ExecutionLog] = None,
) -> pd.DataFrame:
    """Build the Data Quality table.

    ``frame`` must already have been through :func:`prepare_frame`.
    """
    thresholds = config.thresholds
    rows: List[Dict[str, Any]] = []
    total_rows = len(frame)

    incomplete = {q.label for q in quarters if not q.is_complete}
    rows.extend(_dataset_level_checks(frame, config, thresholds, quarters))
    for model in config.active_models:
        rows.extend(
            _model_level_checks(frame, config, model, thresholds, quarters, incomplete)
        )
    rows.extend(
        _variable_level_checks(
            frame, config, thresholds, max_variables_scanned=max_variables_scanned
        )
    )
    rows.extend(_freshness_checks(frame, config, quarters))

    table = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    if not table.empty:
        table["__SEVRANK"] = table["SEVERITY"].map(
            {"RED": 3, "AMBER": 2, "GREY": 1, "GREEN": 0}
        ).fillna(0)
        table = (
            table.sort_values(["__SEVRANK", "CATEGORY", "MODEL", "QUARTER"], ascending=[False, True, True, True])
            .drop(columns="__SEVRANK")
            .reset_index(drop=True)
        )
    material = int((table["SEVERITY"].isin(["RED", "AMBER"])).sum()) if not table.empty else 0
    logger.info(
        "Data quality: %d check(s) executed on %s rows, %d material issue(s).",
        len(table), f"{total_rows:,}", material,
    )
    if execution_log is not None and material:
        execution_log.warning(f"{material} material data-quality issue(s) raised.")
    return table


def _dataset_level_checks(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    thresholds: ThresholdConfig,
    quarters: Sequence[QuarterInfo],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(frame)

    # -- duplicate LAN -----------------------------------------------------
    if config.lan_column in frame.columns:
        dup_count = int(frame[config.lan_column].duplicated().sum())
        rate = pct(dup_count, total, 0.0) or 0.0
        rows.append(_check_row(
            "DQ-001", "DUPLICATION", f"Duplicate {config.lan_column} values",
            metric="Duplicate rate", value=rate, unit="%",
            threshold=thresholds.duplicate_rate_pct.label(),
            severity=thresholds.duplicate_rate_pct.classify(rate),
            records=dup_count,
            detail=(
                f"{dup_count:,} of {total:,} rows repeat a loan account number. The loan "
                "account number is expected to be unique per application."
            ),
            action=(
                "Confirm with the data engineering team whether re-scored applications are "
                "intentionally retained. De-duplicate before interpreting population counts."
            ),
        ))

    # -- duplicate customer+date ------------------------------------------
    if config.customer_column in frame.columns and config.date_column in frame.columns:
        dup_cust = int(frame.duplicated(subset=[config.customer_column, config.date_column]).sum())
        rate = pct(dup_cust, total, 0.0) or 0.0
        rows.append(_check_row(
            "DQ-002", "DUPLICATION", "Duplicate customer + login date",
            metric="Duplicate rate", value=rate, unit="%",
            threshold=thresholds.duplicate_rate_pct.label(),
            severity=thresholds.duplicate_rate_pct.classify(rate),
            records=dup_cust,
            detail=(
                f"{dup_cust:,} rows share a customer identifier and login date, which usually "
                "indicates a join fan-out or a re-scoring event."
            ),
            action="Apply runtime.multi_record_policy consistently; investigate the extract join.",
        ))
        repeat_customers = int(
            frame[config.customer_column].value_counts().gt(1).sum()
        ) if total else 0
        rows.append(_check_row(
            "DQ-003", "DUPLICATION", "Customers with multiple applications",
            metric="Customers affected", value=repeat_customers, unit="count",
            threshold="informational", severity=Severity.GREEN,
            records=repeat_customers,
            detail=(
                f"{repeat_customers:,} customer(s) appear more than once. Quarter-on-quarter "
                f"migration collapses these using policy "
                f"'{config.runtime.multi_record_policy.value}'."
            ),
            action="No action if repeat applications are expected for this product.",
        ))

    # -- invalid / missing dates ------------------------------------------
    if config.date_column in frame.columns:
        dates = pd.to_datetime(frame[config.date_column], errors="coerce")
        bad = int(dates.isna().sum())
        rate = pct(bad, total, 0.0) or 0.0
        rows.append(_check_row(
            "DQ-010", "DATE INTEGRITY", f"Unusable {config.date_column}",
            metric="Unusable date rate", value=rate, unit="%",
            threshold="GREEN < 0.5 <= AMBER < 2 <= RED",
            severity=Severity.RED if rate >= 2 else (Severity.AMBER if rate >= 0.5 else Severity.GREEN),
            records=bad,
            detail=f"{bad:,} record(s) cannot be assigned to a monitoring quarter.",
            action="Fix the source date format or exclude the records with a documented rationale.",
        ))
        future = int((dates.dt.date > date.today()).sum())
        if future:
            rows.append(_check_row(
                "DQ-011", "DATE INTEGRITY", "Future-dated applications",
                metric="Future-dated records", value=future, unit="count",
                threshold="any occurrence is an exception", severity=Severity.AMBER,
                records=future,
                detail=f"{future:,} record(s) are dated after today ({date.today().isoformat()}).",
                action="Check the extract's timezone handling and any placeholder dates.",
            ))
        very_old = int((dates.dt.year < 1990).sum())
        if very_old:
            rows.append(_check_row(
                "DQ-012", "DATE INTEGRITY", "Implausibly old application dates",
                metric="Records before 1990", value=very_old, unit="count",
                threshold="any occurrence is an exception", severity=Severity.AMBER,
                records=very_old,
                detail=f"{very_old:,} record(s) are dated before 1990 (likely placeholders).",
                action="Treat as missing dates or correct at source.",
            ))

    # -- incomplete quarters ----------------------------------------------
    for quarter in quarters:
        if quarter.is_complete:
            continue
        rows.append(_check_row(
            "DQ-020", "COVERAGE", "Incomplete calendar quarter",
            quarter=quarter.label, scope="QUARTER",
            metric="Records in quarter", value=quarter.record_count, unit="count",
            threshold="full calendar quarter required",
            severity=Severity.GREY,
            records=quarter.record_count,
            detail=(
                f"{quarter.label} spans {quarter.period_start} to {quarter.period_end} but the "
                f"data covers {quarter.observed_start} to {quarter.observed_end}. "
                + (quarter.completeness_note or "")
            ),
            action=(
                "Excluded from the primary quarterly comparison. Re-run once the quarter closes, "
                "or set runtime.include_partial_quarters=true to monitor it explicitly."
            ),
        ))

    # -- record count anomalies -------------------------------------------
    complete = [q for q in quarters if q.is_complete or config.runtime.include_partial_quarters]
    for previous, current in zip(complete, complete[1:]):
        change = pct(current.record_count - previous.record_count, previous.record_count)
        if change is None:
            continue
        severity = thresholds.record_count_change_pct.classify(abs(change))
        if severity is Severity.GREEN:
            continue
        rows.append(_check_row(
            "DQ-030", "VOLUME", "Unexpected change in application volume",
            quarter=current.label, scope="QUARTER",
            metric="QoQ record count change", value=change, unit="%",
            threshold=f"+/- {thresholds.record_count_change_pct.label()}",
            severity=severity,
            records=current.record_count,
            detail=(
                f"{current.label} holds {current.record_count:,} records against "
                f"{previous.record_count:,} in {previous.label} ({change:+.1f}%)."
            ),
            action=(
                "Confirm against business volumes. A population shift of this size will also "
                "move PSI and must not be read as model deterioration on its own."
            ),
        ))
    return rows


def _model_level_checks(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    model,
    thresholds: ThresholdConfig,
    quarters: Sequence[QuarterInfo],
    incomplete_quarters: Optional[set] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    segment_column = f"{SEGMENT_PREFIX}{model.name}"
    if segment_column not in frame.columns:
        return rows
    segments = config.segments
    incomplete_quarters = incomplete_quarters or set()

    def _grade(severity: Severity, quarter_label: str) -> tuple:
        """An incomplete quarter cannot support a conclusion, so it is GREY.

        The rate is still reported -- it is useful intelligence -- but a partial
        quarter must not raise an escalation that a full quarter would not, and
        must not crowd genuine findings out of the priority list.
        """
        if quarter_label in incomplete_quarters and severity is not Severity.GREEN:
            return Severity.GREY, (
                f" {quarter_label} is an incomplete calendar quarter, so this rate is "
                "provisional and is reported as a data gap rather than a breach; re-assess "
                "once the quarter closes."
            )
        return severity, ""

    # -- model output availability by quarter -----------------------------
    for quarter in quarters:
        subset = frame.loc[frame[QUARTER_COLUMN] == quarter.label, segment_column]
        total = int(len(subset))
        if total == 0:
            rows.append(_check_row(
                "DQ-040", "MODEL OUTPUT", "No scorecard output in quarter",
                model=model.name, quarter=quarter.label, scope="MODEL-QUARTER",
                metric="Records", value=0, unit="count",
                threshold="at least 1 scored record",
                severity=Severity.GREY, records=0,
                detail=f"No records at all for {model.name} in {quarter.label}.",
                action="Check whether the scorecard was live in this quarter.",
            ))
            continue
        missing = int((subset == segments.missing_label).sum())
        invalid = int((subset == segments.invalid_label).sum())
        missing_rate = pct(missing, total, 0.0) or 0.0
        invalid_rate = pct(invalid, total, 0.0) or 0.0
        if missing:
            severity, note = _grade(
                thresholds.missing_segment_rate_pct.classify(missing_rate), quarter.label
            )
            rows.append(_check_row(
                "DQ-041", "MODEL OUTPUT", "Missing scorecard output",
                model=model.name, quarter=quarter.label, scope="MODEL-QUARTER",
                metric="Missing segment rate", value=missing_rate, unit="%",
                threshold=thresholds.missing_segment_rate_pct.label(),
                severity=severity,
                records=missing,
                detail=(
                    f"{missing:,} of {total:,} records in {quarter.label} have no "
                    f"{model.name} segment.{note}"
                ),
                action=(
                    "Investigate scoring-engine coverage, feature availability and any "
                    "policy-reject population that bypasses the scorecard."
                ),
            ))
        if invalid:
            observed = sorted(
                {
                    str(v) for v in frame.loc[
                        (frame[QUARTER_COLUMN] == quarter.label)
                        & (frame[segment_column] == segments.invalid_label),
                        model.segment_column,
                    ].astype("string").dropna().unique()[:8]
                }
            )
            severity, note = _grade(
                thresholds.invalid_segment_rate_pct.classify(invalid_rate), quarter.label
            )
            rows.append(_check_row(
                "DQ-042", "MODEL OUTPUT", "Unexpected segment value",
                model=model.name, quarter=quarter.label, scope="MODEL-QUARTER",
                metric="Invalid segment rate", value=invalid_rate, unit="%",
                threshold=thresholds.invalid_segment_rate_pct.label(),
                severity=severity,
                records=invalid,
                detail=(
                    f"{invalid:,} record(s) hold a value outside "
                    f"{segments.segment_order}. Observed: {observed}.{note}"
                ),
                action=(
                    "Correct the scoring-engine output mapping. Invalid values are excluded "
                    "from PSI and migration, which biases those metrics."
                ),
            ))
    return rows


def _variable_level_checks(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    thresholds: ThresholdConfig,
    *,
    max_variables_scanned: int = 2_000,
) -> List[Dict[str, Any]]:
    """Missingness level, infinities and outliers per monitored characteristic."""
    rows: List[Dict[str, Any]] = []
    scanned = 0
    for model in config.active_models:
        scope = model_scope_mask(frame, config, model, scope=config.runtime.csi_scope)
        scoped = frame.loc[scope]
        total = int(len(scoped))
        if total == 0:
            continue
        scope_note = (
            f"Measured on the {total:,} record(s) {model.name} actually scored, not the whole "
            "extract; coverage of unscored applications is reported separately (DQ-041)."
        )
        for spec in config.monitored_variables(model.name):
            if scanned >= max_variables_scanned:
                rows.append(_check_row(
                    "DQ-059", "COVERAGE", "Variable-level scan truncated",
                    model=model.name, scope="VARIABLE",
                    metric="Variables scanned", value=scanned, unit="count",
                    threshold=f"max_variables_scanned={max_variables_scanned}",
                    severity=Severity.GREY,
                    detail=(
                        "The dataset-level variable scan stopped at the configured cap. CSI "
                        "still covers every monitored characteristic."
                    ),
                    action="Raise max_variables_scanned or rely on the CSI sheets.",
                ))
                return rows
            if spec.name not in scoped.columns:
                continue
            scanned += 1
            series = scoped[spec.name]
            missing = int(series.isna().sum())
            missing_rate = pct(missing, total, 0.0) or 0.0
            severity = thresholds.missingness_level_pct.classify(missing_rate)
            if severity is not Severity.GREEN:
                rows.append(_check_row(
                    "DQ-050", "MISSINGNESS", "High missingness in model variable",
                    model=model.name, scope="VARIABLE",
                    metric=f"{spec.name} missing rate", value=missing_rate, unit="%",
                    threshold=thresholds.missingness_level_pct.label(),
                    severity=severity, records=missing,
                    detail=(
                        f"{missing:,} of {total:,} records have no value for '{spec.name}' "
                        f"({spec.role.value.lower()}). {scope_note}"
                    ),
                    action=(
                        "Confirm the feature pipeline is populating this characteristic. High "
                        "missingness pushes mass into the MISSING bin and inflates CSI."
                    ),
                ))
            if is_numeric(series):
                n_inf = count_infinite(series)
                if n_inf:
                    rows.append(_check_row(
                        "DQ-051", "DATA INTEGRITY", "Infinite values in model variable",
                        model=model.name, scope="VARIABLE",
                        metric=f"{spec.name} infinite count", value=n_inf, unit="count",
                        threshold="any occurrence is an exception",
                        severity=Severity.AMBER, records=n_inf,
                        detail=(
                            f"{n_inf:,} +/-inf value(s) in '{spec.name}'. They are clipped into "
                            "the outer bins for CSI, which can mask the underlying defect."
                        ),
                        action="Fix the upstream division/ratio calculation producing infinities.",
                    ))
                outliers, fence = _iqr_outliers(series)
                outlier_rate = pct(outliers, total, 0.0) or 0.0
                if outlier_rate >= 1.0:
                    rows.append(_check_row(
                        "DQ-052", "DATA INTEGRITY", "Outliers beyond the 3xIQR fence",
                        model=model.name, scope="VARIABLE",
                        metric=f"{spec.name} outlier rate", value=outlier_rate, unit="%",
                        threshold="GREEN < 1 <= AMBER < 5 <= RED",
                        severity=Severity.RED if outlier_rate >= 5 else Severity.AMBER,
                        records=outliers,
                        detail=(
                            f"{outliers:,} value(s) lie outside {fence}. Extreme values are "
                            "retained (never deleted) and fall into the outer bins."
                        ),
                        action=(
                            "Check for sentinel codes such as 9999 or -1 being treated as real "
                            "values, and confirm the approved bin boundaries still apply."
                        ),
                    ))
            else:
                try:
                    n_unique = int(series.nunique(dropna=True))
                except Exception:  # pragma: no cover
                    continue
                if n_unique > 200 and spec.var_type is VariableType.CATEGORICAL:
                    rows.append(_check_row(
                        "DQ-053", "DATA INTEGRITY", "Very high cardinality categorical",
                        model=model.name, scope="VARIABLE",
                        metric=f"{spec.name} distinct values", value=n_unique, unit="count",
                        threshold="investigate above 200",
                        severity=Severity.AMBER, records=n_unique,
                        detail=(
                            f"'{spec.name}' has {n_unique:,} distinct values. Rare levels are "
                            "grouped into OTHER, so CSI is driven by the grouping rule."
                        ),
                        action="Supply an approved category grouping via variables[].category_map.",
                    ))
    return rows


def _iqr_outliers(series: pd.Series, multiplier: float = 3.0) -> tuple[int, str]:
    """Count values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` ignoring NaN/inf."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))]
    if finite.empty:
        return 0, "n/a"
    q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return 0, "degenerate IQR"
    low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
    count = int(((finite < low) | (finite > high)).sum())
    return count, f"[{low:,.4g}, {high:,.4g}]"


def _freshness_checks(
    frame: pd.DataFrame, config: MonitoringConfig, quarters: Sequence[QuarterInfo]
) -> List[Dict[str, Any]]:
    """Data freshness and feature-availability checks."""
    rows: List[Dict[str, Any]] = []
    if config.date_column not in frame.columns:
        return rows
    dates = pd.to_datetime(frame[config.date_column], errors="coerce").dropna()
    if dates.empty:
        return rows
    latest = dates.max().date()
    lag_days = (date.today() - latest).days
    severity = Severity.GREEN
    if lag_days > 120:
        severity = Severity.RED
    elif lag_days > 45:
        severity = Severity.AMBER
    rows.append(_check_row(
        "DQ-060", "FRESHNESS", "Data freshness",
        metric="Days since latest application", value=lag_days, unit="days",
        threshold="GREEN <= 45 < AMBER <= 120 < RED",
        severity=severity,
        detail=(
            f"The most recent application in the extract is dated {latest.isoformat()}, "
            f"{lag_days} day(s) before the run date {date.today().isoformat()}."
        ),
        action=(
            "Confirm the extract covers the full reporting quarter before circulating the pack."
        ),
    ))

    # -- feature availability per model -----------------------------------
    for model in config.active_models:
        specs = config.monitored_variables(model.name)
        if not specs:
            continue
        present = [s for s in specs if s.name in frame.columns]
        missing_cols = [s.name for s in specs if s.name not in frame.columns]
        if missing_cols:
            rows.append(_check_row(
                "DQ-061", "FRESHNESS", "Configured characteristic absent from extract",
                model=model.name, scope="MODEL",
                metric="Missing characteristics", value=len(missing_cols), unit="count",
                threshold="0 expected",
                severity=Severity.AMBER, records=len(missing_cols),
                detail="Absent: " + ", ".join(missing_cols[:20])
                       + (" ..." if len(missing_cols) > 20 else ""),
                action="Re-point the column mapping or restore the feature in the extract.",
            ))
        fully_empty = [s.name for s in present if int(frame[s.name].notna().sum()) == 0]
        if fully_empty:
            rows.append(_check_row(
                "DQ-062", "FRESHNESS", "Characteristic present but entirely empty",
                model=model.name, scope="MODEL",
                metric="Empty characteristics", value=len(fully_empty), unit="count",
                threshold="0 expected",
                severity=Severity.RED, records=len(fully_empty),
                detail="Empty: " + ", ".join(fully_empty[:20])
                       + (" ..." if len(fully_empty) > 20 else ""),
                action=(
                    "A scorecard input that is never populated means the live score is not the "
                    "developed score. Escalate to the model owner immediately."
                ),
            ))
    return rows


def variable_missingness_checks(
    csi_frame: pd.DataFrame, thresholds: ThresholdConfig
) -> pd.DataFrame:
    """Derive 'missingness increased vs reference' checks from the CSI table.

    Kept here so every data-quality row shares one schema, but driven by the CSI
    results because that is where reference-vs-monitoring missingness is
    measured on a like-for-like basis.
    """
    if csi_frame is None or csi_frame.empty:
        return pd.DataFrame(columns=CHECK_COLUMNS)
    required = {"MODEL", "QUARTER", "VARIABLE", "MISSING_PCT_REF", "MISSING_PCT_MON"}
    if not required.issubset(csi_frame.columns):
        return pd.DataFrame(columns=CHECK_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for record in csi_frame.to_dict("records"):
        change = record.get("MISSINGNESS_CHANGE_PP")
        if change is None or (isinstance(change, float) and change != change):
            continue
        severity = thresholds.missingness_increase_pp.classify(abs(float(change)))
        if severity is Severity.GREEN:
            continue
        direction = "increased" if float(change) > 0 else "decreased"
        rows.append(_check_row(
            "DQ-055", "MISSINGNESS", "Material change in missingness vs reference",
            model=str(record.get("MODEL")), quarter=str(record.get("QUARTER")),
            scope="VARIABLE",
            metric=f"{record.get('VARIABLE')} missingness change",
            value=float(change), unit="pp",
            threshold=f"+/- {thresholds.missingness_increase_pp.label()}",
            severity=severity,
            records=record.get("MONITORING_COUNT"),
            detail=(
                f"Missingness {direction} from {record.get('MISSING_PCT_REF')}% in the "
                f"reference period to {record.get('MISSING_PCT_MON')}% in "
                f"{record.get('QUARTER')} ({float(change):+.2f} pp)."
            ),
            action=(
                "Trace the feature pipeline for this characteristic. A missingness shift of this "
                "size changes the effective scorecard input and must be explained before the "
                "CSI movement is attributed to genuine population change."
            ),
        ))

    # -- unseen categories -------------------------------------------------
    if "UNSEEN_CATEGORY_PCT" in csi_frame.columns:
        for record in csi_frame.to_dict("records"):
            unseen = record.get("UNSEEN_CATEGORY_PCT")
            if unseen is None or (isinstance(unseen, float) and unseen != unseen) or float(unseen) <= 0:
                continue
            severity = thresholds.unseen_category_rate_pct.classify(float(unseen))
            if severity is Severity.GREEN:
                continue
            rows.append(_check_row(
                "DQ-056", "UNEXPECTED CATEGORY", "Categories not seen in the reference period",
                model=str(record.get("MODEL")), quarter=str(record.get("QUARTER")),
                scope="VARIABLE",
                metric=f"{record.get('VARIABLE')} unseen-category rate",
                value=float(unseen), unit="%",
                threshold=thresholds.unseen_category_rate_pct.label(),
                severity=severity,
                records=record.get("UNSEEN_CATEGORY_COUNT"),
                detail=(
                    f"{record.get('UNSEEN_CATEGORY_COUNT')} record(s) "
                    f"({float(unseen):.2f}%) in {record.get('QUARTER')} carry a value that never "
                    "appeared in the reference period, so no development bin exists for them."
                ),
                action=(
                    "Check for new product codes, bureau schema changes or re-coded categories. "
                    "Unseen levels cannot be scored as developed and need a mapping decision."
                ),
            ))
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


__all__ = [
    "QUARTER_COLUMN", "SEGMENT_PREFIX", "SEGMENT_RANK_PREFIX", "OUTCOME_COLUMN",
    "MATURITY_COLUMN", "CHECK_COLUMNS", "ValidationError", "ValidationReport",
    "validate_schema", "prepare_frame", "run_data_quality_checks",
    "variable_missingness_checks",
]
