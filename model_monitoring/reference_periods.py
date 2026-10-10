"""Per-scorecard reference (development / production baseline) period handling.

This module enforces the rule that gives the whole report its meaning: **every
scorecard is compared against its own approved baseline.**  Four scorecards
built, implemented and retrained at different times do not share a baseline,
and the pipeline must never paper over that.

Concretely it refuses to:

* use the entire dataset as a reference population by default,
* reuse one model's reference window for another model,
* proceed with an unconfigured reference period without the user saying so.

Configuration can arrive from a config file, a Python dict, CLI overrides, or
interactive prompts -- and whichever route is used, the resolved window, mode,
record count and any warnings are carried into the Excel report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .config import (
    ConfigError,
    ModelConfig,
    MonitoringConfig,
    ReferenceMode,
    ReferencePeriodConfig,
)
from .data_validation import QUARTER_COLUMN, SEGMENT_PREFIX
from .logging_config import ExecutionLog, get_logger
from .utils import QuarterInfo, build_quarters, pct

logger = get_logger("reference_periods")


class ReferencePeriodError(RuntimeError):
    """Raised when no usable reference population can be established."""


@dataclass
class ReferenceResolution:
    """The resolved reference population of one scorecard."""

    model: str
    period: ReferencePeriodConfig
    mask: pd.Series
    record_count: int
    scored_count: int
    unique_lans: Optional[int]
    unique_customers: Optional[int]
    share_of_dataset_pct: Optional[float]
    observed_start: Optional[date]
    observed_end: Optional[date]
    sufficient: bool
    warnings: List[str] = field(default_factory=list)
    quarters_covered: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.period.label

    @property
    def mode(self) -> str:
        return self.period.mode.value

    def describe(self) -> str:
        return (
            f"{self.model}: {self.label} [{self.mode}] -> {self.record_count:,} record(s), "
            f"{self.scored_count:,} scored"
            + ("" if self.sufficient else "  ** INSUFFICIENT **")
        )

    def as_row(self) -> Dict[str, Any]:
        return {
            "MODEL": self.model,
            "REFERENCE_MODE": self.mode,
            "REFERENCE_START": self.period.start_date.isoformat() if self.period.start_date else None,
            "REFERENCE_END": self.period.end_date.isoformat() if self.period.end_date else None,
            "REFERENCE_LABEL": self.label,
            "BASELINE_FILTER": self.period.baseline_filter or "",
            "CONFIGURED_BY": self.period.configured_by,
            "RECORDS_IN_REFERENCE": self.record_count,
            "SCORED_RECORDS_IN_REFERENCE": self.scored_count,
            "UNIQUE_LANS": self.unique_lans,
            "UNIQUE_CUSTOMERS": self.unique_customers,
            "SHARE_OF_DATASET_PCT": None if self.share_of_dataset_pct is None
            else round(self.share_of_dataset_pct, 2),
            "OBSERVED_START": self.observed_start.isoformat() if self.observed_start else None,
            "OBSERVED_END": self.observed_end.isoformat() if self.observed_end else None,
            "QUARTERS_COVERED": ", ".join(self.quarters_covered),
            "SUFFICIENT_SAMPLE": self.sufficient,
            "SOURCE_NOTE": self.period.source_note,
            "WARNINGS": " | ".join(self.warnings),
        }


REFERENCE_SUMMARY_COLUMNS = [
    "MODEL", "REFERENCE_MODE", "REFERENCE_START", "REFERENCE_END", "REFERENCE_LABEL",
    "BASELINE_FILTER", "CONFIGURED_BY", "RECORDS_IN_REFERENCE",
    "SCORED_RECORDS_IN_REFERENCE", "UNIQUE_LANS", "UNIQUE_CUSTOMERS",
    "SHARE_OF_DATASET_PCT", "OBSERVED_START", "OBSERVED_END", "QUARTERS_COVERED",
    "SUFFICIENT_SAMPLE", "SOURCE_NOTE", "WARNINGS",
]


# ---------------------------------------------------------------------------
# Dataset span
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpan:
    """Available date range, used to validate every reference window."""

    min_date: date
    max_date: date
    record_count: int

    def contains(self, value: date) -> bool:
        return self.min_date <= value <= self.max_date

    def describe(self) -> str:
        return (
            f"{self.min_date.isoformat()} to {self.max_date.isoformat()} "
            f"({self.record_count:,} records)"
        )


def dataset_span(frame: pd.DataFrame, config: MonitoringConfig) -> DatasetSpan:
    """Date coverage of the extract."""
    if config.date_column not in frame.columns:
        raise ReferencePeriodError(
            f"Cannot determine the dataset span: '{config.date_column}' is missing."
        )
    dates = pd.to_datetime(frame[config.date_column], errors="coerce").dropna()
    if dates.empty:
        raise ReferencePeriodError(
            f"Cannot determine the dataset span: no parseable values in "
            f"'{config.date_column}'."
        )
    return DatasetSpan(
        min_date=dates.min().date(), max_date=dates.max().date(), record_count=int(len(frame))
    )


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def build_reference_mask(
    frame: pd.DataFrame, config: MonitoringConfig, model: ModelConfig
) -> Tuple[pd.Series, List[str]]:
    """Boolean mask selecting one model's reference population.

    Never returns "all rows" implicitly: a window or an explicit baseline filter
    must be present, otherwise the caller gets an error.
    """
    warnings: List[str] = []
    period = model.reference
    if period is None:
        raise ReferencePeriodError(f"[{model.name}] has no reference period object.")

    if not period.is_configured and not period.baseline_filter:
        raise ReferencePeriodError(
            f"[{model.name}] reference period is not configured. Supply "
            f"models.{model.name}.reference.start_date/end_date, pass "
            f"--reference {model.name}=START:END, or answer the interactive prompt. "
            "The pipeline will not fall back to the full dataset."
        )

    mask = pd.Series(True, index=frame.index)

    if period.is_configured:
        dates = pd.to_datetime(frame[config.date_column], errors="coerce")
        mask &= dates.notna()
        mask &= dates.dt.date >= period.start_date
        mask &= dates.dt.date <= period.end_date

    if period.baseline_filter:
        try:
            selected = frame.query(period.baseline_filter).index
            mask &= frame.index.isin(selected)
        except Exception as exc:
            raise ReferencePeriodError(
                f"[{model.name}] reference.baseline_filter could not be evaluated "
                f"({period.baseline_filter!r}): {exc}"
            ) from exc
        if period.mode is not ReferenceMode.PRODUCTION_BASELINE:
            warnings.append(
                f"A baseline_filter is applied although the reference mode is "
                f"{period.mode.value}; confirm this is intended."
            )

    if model.population_filter:
        try:
            selected = frame.query(model.population_filter).index
            mask &= frame.index.isin(selected)
        except Exception as exc:
            raise ReferencePeriodError(
                f"[{model.name}] population_filter could not be evaluated "
                f"({model.population_filter!r}): {exc}"
            ) from exc

    return mask.fillna(False), warnings


def resolve_reference_periods(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
    strict: bool = False,
) -> Dict[str, ReferenceResolution]:
    """Resolve, validate and describe each scorecard's reference population.

    Models whose reference population cannot be built are reported and skipped
    (or, with ``strict=True``, abort the run) -- they are never silently given
    somebody else's baseline.
    """
    span = dataset_span(frame, config)
    resolutions: Dict[str, ReferenceResolution] = {}
    minimum = config.thresholds.min_reference_records

    for model in config.active_models:
        period = model.reference
        try:
            mask, warnings = build_reference_mask(frame, config, model)
        except ReferencePeriodError as exc:
            logger.error(str(exc))
            if execution_log is not None:
                execution_log.error(str(exc))
            if strict:
                raise
            continue

        subset = frame.loc[mask]
        record_count = int(len(subset))
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        if segment_column in subset.columns:
            scored = int(
                (~subset[segment_column].isin(
                    [config.segments.missing_label, config.segments.invalid_label]
                )).sum()
            )
        else:
            scored = 0

        # -- date-window validation against the available data -------------
        if period.is_configured:
            if not span.contains(period.start_date):
                warnings.append(
                    f"Reference start {period.start_date.isoformat()} lies outside the dataset "
                    f"span ({span.describe()}); the effective window is truncated."
                )
            if not span.contains(period.end_date):
                warnings.append(
                    f"Reference end {period.end_date.isoformat()} lies outside the dataset "
                    f"span ({span.describe()}); the effective window is truncated."
                )

        observed_start = observed_end = None
        quarters_covered: List[str] = []
        if record_count and config.date_column in subset.columns:
            dates = pd.to_datetime(subset[config.date_column], errors="coerce").dropna()
            if not dates.empty:
                observed_start, observed_end = dates.min().date(), dates.max().date()
            if QUARTER_COLUMN in subset.columns:
                quarters_covered = sorted(
                    {str(q) for q in subset[QUARTER_COLUMN].dropna().unique()}
                )

        if record_count == 0:
            warnings.append(
                "The reference window selects ZERO records. PSI and CSI cannot be computed "
                "against it; check the dates, the baseline filter and the population filter."
            )
        # Adequacy is judged on the SCORED count, not the row count. The expected
        # distribution is estimated only from records that carry a grade, so a
        # window holding 1,836 rows of which 272 are scored is a 272-observation
        # baseline -- which is what PSI and CSI are actually computed against.
        sufficient = scored >= minimum
        if 0 < scored < minimum:
            warnings.append(
                f"Only {scored:,} of {record_count:,} record(s) in the reference period carry a "
                f"valid scorecard segment, below the configured minimum of {minimum:,}. PSI/CSI "
                "will be reported but labelled GREY: the expected proportions are estimated from "
                "that scored sub-population alone and are unstable at this size."
            )
        elif 0 < record_count < minimum:
            warnings.append(
                f"Only {record_count:,} record(s) in the reference period, below the configured "
                f"minimum of {minimum:,}. PSI/CSI will be reported but labelled GREY: a baseline "
                "this small produces unstable expected proportions."
            )
        if record_count and scored == 0:
            warnings.append(
                "No record in the reference period carries a valid scorecard segment, so there "
                "is no expected distribution to compare against."
            )

        resolution = ReferenceResolution(
            model=model.name,
            period=period,
            mask=mask,
            record_count=record_count,
            scored_count=scored,
            unique_lans=int(subset[config.lan_column].nunique())
            if config.lan_column in subset.columns else None,
            unique_customers=int(subset[config.customer_column].nunique())
            if config.customer_column in subset.columns else None,
            share_of_dataset_pct=pct(record_count, span.record_count),
            observed_start=observed_start,
            observed_end=observed_end,
            sufficient=sufficient,
            warnings=warnings,
            quarters_covered=quarters_covered,
        )
        period.record_count = record_count
        resolutions[model.name] = resolution

        for message in warnings:
            logger.warning("[%s] %s", model.name, message)
            if execution_log is not None:
                execution_log.warning(f"[{model.name}] {message}")
        logger.info("Reference population %s", resolution.describe())

    if not resolutions:
        raise ReferencePeriodError(
            "No scorecard has a usable reference population. Configure "
            "models.<MODEL>.reference.start_date / end_date for each scorecard."
        )

    # -- guard against an accidentally shared window ----------------------
    windows: Dict[Tuple[Optional[date], Optional[date]], List[str]] = {}
    for name, resolution in resolutions.items():
        key = (resolution.period.start_date, resolution.period.end_date)
        windows.setdefault(key, []).append(name)
    for key, names in windows.items():
        if len(names) > 1 and key != (None, None):
            logger.info(
                "Models %s share the reference window %s to %s. That is legitimate when the "
                "scorecards were implemented together, but confirm it is not a copy-paste error.",
                names, key[0], key[1],
            )
    return resolutions


def reference_masks(resolutions: Mapping[str, ReferenceResolution]) -> Dict[str, pd.Series]:
    """``{model: mask}`` view for the metric modules."""
    return {name: resolution.mask for name, resolution in resolutions.items()}


def reference_summary_frame(
    resolutions: Mapping[str, ReferenceResolution]
) -> pd.DataFrame:
    """Tabular reference-period metadata for the Configuration sheet."""
    rows = [resolution.as_row() for resolution in resolutions.values()]
    return pd.DataFrame(rows, columns=REFERENCE_SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# Quarter selection
# ---------------------------------------------------------------------------


def select_monitoring_quarters(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
) -> Tuple[List[QuarterInfo], List[QuarterInfo]]:
    """Return ``(all_quarters, monitored_quarters)``.

    Complete calendar quarters are used by default.  Incomplete quarters are
    identified, reported and excluded from the primary comparison unless
    ``runtime.include_partial_quarters`` is explicitly enabled.
    """
    dates = pd.to_datetime(frame[config.date_column], errors="coerce")
    all_quarters = build_quarters(dates, min_records=config.runtime.min_quarter_records)
    if not all_quarters:
        raise ReferencePeriodError(
            "No monitoring quarter could be derived from the extract; check the login dates."
        )

    include_partial = config.runtime.include_partial_quarters
    monitored: List[QuarterInfo] = []
    for quarter in all_quarters:
        if not quarter.is_complete and not include_partial:
            message = (
                f"{quarter.label} is an incomplete calendar quarter "
                f"({quarter.observed_start} to {quarter.observed_end}) and is EXCLUDED from the "
                "primary quarterly comparison. Set runtime.include_partial_quarters=true to "
                "monitor it explicitly."
            )
            logger.warning(message)
            if execution_log is not None:
                execution_log.warning(message)
            continue
        if quarter.record_count < config.runtime.min_quarter_records:
            message = (
                f"{quarter.label} holds only {quarter.record_count:,} record(s), below "
                f"runtime.min_quarter_records={config.runtime.min_quarter_records}. It is "
                "retained but its metrics are labelled GREY."
            )
            logger.warning(message)
            if execution_log is not None:
                execution_log.warning(message)
        monitored.append(quarter)

    if not monitored:
        raise ReferencePeriodError(
            "Every quarter in the extract is incomplete, so there is nothing to monitor. "
            "Enable runtime.include_partial_quarters to proceed with partial quarters."
        )
    logger.info(
        "Monitoring %d of %d quarter(s): %s",
        len(monitored), len(all_quarters), ", ".join(q.label for q in monitored),
    )
    return all_quarters, monitored


# ---------------------------------------------------------------------------
# Non-interactive overrides
# ---------------------------------------------------------------------------


def apply_reference_mapping(
    config: MonitoringConfig, mapping: Mapping[str, Any], *, source: str = "PYTHON_DICT"
) -> None:
    """Apply ``{model: {...}}`` or ``{model: "START:END"}`` reference overrides."""
    for model_name, payload in mapping.items():
        model = config.model(model_name)
        if isinstance(payload, str):
            if ":" not in payload:
                raise ConfigError(
                    f"Reference override for {model_name} must look like 'START:END' "
                    f"(got {payload!r})."
                )
            start, _, end = payload.partition(":")
            payload = {"start_date": start.strip(), "end_date": end.strip()}
        payload = dict(payload)
        payload.setdefault("model", model_name)
        payload.setdefault("mode", model.reference.mode if model.reference else ReferenceMode.DEVELOPMENT)
        period = ReferencePeriodConfig(**payload)
        period.configured_by = source
        model.reference = period
        logger.info(
            "[%s] reference period set from %s: %s", model_name, source, period.describe()
        )


def parse_cli_reference(values: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Parse ``--reference MODEL_1=2023-01-01:2023-12-31[:MODE]`` arguments."""
    mapping: Dict[str, Dict[str, Any]] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ConfigError(
                f"--reference expects MODEL=START:END[:MODE], got {raw!r}."
            )
        model_name, _, window = raw.partition("=")
        parts = window.split(":")
        if len(parts) < 2:
            raise ConfigError(
                f"--reference expects MODEL=START:END[:MODE], got {raw!r}."
            )
        payload: Dict[str, Any] = {
            "start_date": parts[0].strip(),
            "end_date": parts[1].strip(),
        }
        if len(parts) >= 3 and parts[2].strip():
            payload["mode"] = parts[2].strip()
        mapping[model_name.strip()] = payload
    return mapping


# ---------------------------------------------------------------------------
# Interactive configuration
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:  # pragma: no cover
        return False


def models_needing_configuration(config: MonitoringConfig) -> List[ModelConfig]:
    """Models whose reference period is still unknown."""
    pending: List[ModelConfig] = []
    for model in config.active_models:
        period = model.reference
        if period is None or (not period.is_configured and not period.baseline_filter):
            pending.append(model)
    return pending


def prompt_reference_periods(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    only_missing: bool = True,
    input_func=input,
    output=print,
) -> None:
    """Ask the user for each scorecard's reference period, one model at a time.

    Validations applied to every answer:

    * the dates parse,
    * ``start <= end``,
    * the window overlaps the available data,
    * the resulting record count is shown, and a window below
      ``thresholds.min_reference_records`` must be explicitly confirmed.
    """
    span = dataset_span(frame, config)
    targets = models_needing_configuration(config) if only_missing else list(config.active_models)
    if not targets:
        return

    output("")
    output("=" * 78)
    output("Configure monitoring reference periods")
    output("=" * 78)
    output(f"Dataset span: {span.describe()}")
    output(
        "Each scorecard is configured separately: they may have been developed, implemented "
        "or retrained at different times."
    )
    output(
        "Reference modes -- DEVELOPMENT: the model development sample. "
        "PRODUCTION_BASELINE: an approved post-implementation reference population."
    )
    output("Enter dates as YYYY-MM-DD. Type 'q' at any prompt to abort the run.")

    for model in targets:
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        scored = (
            int((~frame[segment_column].isin(
                [config.segments.missing_label, config.segments.invalid_label]
            )).sum())
            if segment_column in frame.columns else 0
        )
        output("")
        output(f"{model.name}:")
        output(
            f"  segment column : {model.segment_column} "
            f"({scored:,} record(s) with a valid segment)"
        )
        output(f"  characteristics: {len(model.variables)} monitored")
        if model.description:
            output(f"  description    : {model.description}")

        mode = _prompt_mode(model, input_func, output)
        while True:
            start = _prompt_date("  Start date (YYYY-MM-DD): ", input_func, output)
            end = _prompt_date("  End date   (YYYY-MM-DD): ", input_func, output)
            if start > end:
                output(
                    f"  ! Start date {start} is after end date {end}. Please re-enter both."
                )
                continue
            if end < span.min_date or start > span.max_date:
                output(
                    f"  ! The window {start} to {end} does not overlap the dataset "
                    f"({span.describe()}). Please re-enter both."
                )
                continue
            if start < span.min_date or end > span.max_date:
                output(
                    f"  ! Note: the window extends beyond the available data "
                    f"({span.describe()}); only the overlapping records will be used."
                )

            candidate = ReferencePeriodConfig(
                model=model.name, start_date=start, end_date=end, mode=mode
            )
            candidate.configured_by = "INTERACTIVE"
            previous = model.reference
            model.reference = candidate
            try:
                mask, _ = build_reference_mask(frame, config, model)
            except ReferencePeriodError as exc:
                output(f"  ! {exc}")
                model.reference = previous
                continue
            count = int(mask.sum())
            scored_in_window = (
                int((~frame.loc[mask, segment_column].isin(
                    [config.segments.missing_label, config.segments.invalid_label]
                )).sum())
                if segment_column in frame.columns else 0
            )
            share = pct(count, span.record_count) or 0.0
            output(
                f"  -> {count:,} record(s) in {start} to {end} ({share:.1f}% of the dataset); "
                f"{scored_in_window:,} with a valid {model.name} segment."
            )
            if count == 0:
                output(
                    "  ! That window contains no records and cannot serve as a baseline. "
                    "Please choose another."
                )
                model.reference = previous
                continue
            minimum = config.thresholds.min_reference_records
            if scored_in_window < minimum:
                output(
                    f"  ! WARNING: only {scored_in_window:,} of those record(s) carry a valid "
                    f"{model.name} segment, below the configured minimum of {minimum:,}. The "
                    "expected proportions come from that scored sub-population alone, so they "
                    "are unstable and every PSI/CSI value will be labelled GREY."
                )
                if not _prompt_yes_no("  Use this window anyway? [y/N]: ", input_func, output):
                    model.reference = previous
                    continue
            note = input_func("  Provenance note (optional, press Enter to skip): ").strip()
            if note.lower() == "q":
                raise KeyboardInterrupt("Reference period configuration aborted by the user.")
            candidate.source_note = note
            break

    output("")
    output("Reference periods configured:")
    for model in config.active_models:
        output(f"  {model.name}: {model.reference.describe()}")
    output("=" * 78)
    output("")


def _prompt_mode(model: ModelConfig, input_func, output) -> ReferenceMode:
    default = model.reference.mode if model.reference else ReferenceMode.DEVELOPMENT
    while True:
        answer = input_func(
            f"  Reference mode [DEVELOPMENT/PRODUCTION_BASELINE] (default {default.value}): "
        ).strip()
        if answer.lower() == "q":
            raise KeyboardInterrupt("Reference period configuration aborted by the user.")
        if not answer:
            return default
        try:
            return ReferenceMode(answer.strip().upper().replace("-", "_").replace(" ", "_"))
        except ValueError:
            output("  ! Enter DEVELOPMENT or PRODUCTION_BASELINE.")


def _prompt_date(prompt: str, input_func, output) -> date:
    while True:
        answer = input_func(prompt).strip()
        if answer.lower() == "q":
            raise KeyboardInterrupt("Reference period configuration aborted by the user.")
        if not answer:
            output("  ! A date is required (YYYY-MM-DD).")
            continue
        parsed = pd.to_datetime(answer, errors="coerce", format=None)
        if parsed is None or pd.isna(parsed):
            output(f"  ! Could not read {answer!r} as a date. Use YYYY-MM-DD.")
            continue
        return parsed.date()


def _prompt_yes_no(prompt: str, input_func, output) -> bool:
    while True:
        answer = input_func(prompt).strip().lower()
        if answer == "q":
            raise KeyboardInterrupt("Reference period configuration aborted by the user.")
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        output("  ! Please answer y or n.")


def ensure_reference_periods(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    execution_log: Optional[ExecutionLog] = None,
    input_func=input,
    output=print,
) -> None:
    """Make sure every active model has a reference period, prompting if needed."""
    pending = models_needing_configuration(config)
    if not pending:
        return
    names = ", ".join(model.name for model in pending)
    if config.runtime.non_interactive or not _is_interactive():
        raise ReferencePeriodError(
            f"Reference period(s) missing for: {names}. The pipeline will not default to the "
            "full dataset. Supply them in the configuration file "
            "(models.<MODEL>.reference.start_date / end_date), pass "
            "--reference MODEL=START:END on the command line, or run interactively. "
            + ("(runtime.non_interactive is enabled.)" if config.runtime.non_interactive
               else "(stdin is not a TTY, so prompting is unavailable.)")
        )
    message = f"Reference period(s) missing for: {names}. Prompting the user."
    logger.info(message)
    if execution_log is not None:
        execution_log.info(message)
    prompt_reference_periods(frame, config, input_func=input_func, output=output)


__all__ = [
    "ReferencePeriodError", "ReferenceResolution", "REFERENCE_SUMMARY_COLUMNS",
    "DatasetSpan", "dataset_span", "build_reference_mask", "resolve_reference_periods",
    "reference_masks", "reference_summary_frame", "select_monitoring_quarters",
    "apply_reference_mapping", "parse_cli_reference", "prompt_reference_periods",
    "models_needing_configuration", "ensure_reference_periods",
]
