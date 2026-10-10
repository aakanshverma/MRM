"""Characteristic Stability Index.

Formula
-------
::

    CSI_j = sum_i (A_ij - E_ij) * ln(A_ij / E_ij)

for characteristic ``j`` over its bins ``i``, where ``E`` is the reference
(development or approved production baseline) distribution and ``A`` the
monitoring-quarter distribution.  It is the same divergence as PSI applied to a
characteristic's bins rather than the scorecard's risk grades, so the
implementation reuses :func:`model_monitoring.psi_calculator.compute_distribution_index`.

What CSI adds over PSI
----------------------
PSI says *the scored population moved*.  CSI says *which input moved*, which is
what an investigation actually needs.  Each model uses its own reference period
and its own characteristic mapping -- a model's bin boundaries are never applied
to another model's variables.

Scale
-----
Real extracts carry thousands of characteristics, so the calculation:

* slices one column at a time (never materialising a wide reference frame),
* processes variables in configurable batches with explicit garbage collection,
* supports a whitelist for partial runs, and
* keeps the full detail table while allowing the executive views to show only
  the top-N movers.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .binning import BinApplication, BinDefinition, fit_bin_definition
from .config import (
    MonitoringConfig,
    Severity,
    ThresholdBand,
    VariableSpec,
    VariableType,
)
from .data_validation import QUARTER_COLUMN, model_scope_mask
from .logging_config import ExecutionLog, get_logger
from .psi_calculator import (
    VS_PREVIOUS_QUARTER,
    VS_REFERENCE,
    classify_index,
    compute_distribution_index,
)
from .utils import MISSING_BIN, OTHER_BIN, UNSEEN_BIN, QuarterInfo, chunked, round_or_none

logger = get_logger("csi")

CSI_SUMMARY_COLUMNS = [
    "MODEL", "QUARTER", "COMPARISON", "VARIABLE", "VARIABLE_LABEL", "VARIABLE_TYPE",
    "VARIABLE_ROLE", "CRITICAL", "REFERENCE_LABEL", "REFERENCE_MODE",
    "REFERENCE_COUNT", "MONITORING_COUNT", "N_BINS", "BINNING_SOURCE", "BINNING_DEFINITION",
    "CSI", "CSI_DISPLAY", "SEVERITY", "THRESHOLD_AMBER", "THRESHOLD_RED",
    "TOP_CONTRIBUTING_BIN", "TOP_BIN_CONTRIBUTION", "TOP_BIN_SHARE_PCT",
    "MAX_ABS_PROP_SHIFT_PP", "MISSING_PCT_REF", "MISSING_PCT_MON", "MISSINGNESS_CHANGE_PP",
    "UNSEEN_CATEGORY_COUNT", "UNSEEN_CATEGORY_PCT", "UNSEEN_CATEGORY_EXAMPLES",
    "OTHER_BIN_PCT_MON", "INFINITE_COUNT_MON", "SMOOTHING", "BINS_DROPPED",
    "CONTRIBUTION_SUM_ERROR", "CALCULABLE", "STATUS_REASON", "RECOMMENDED_ACTION",
]

CSI_DETAIL_COLUMNS = [
    "MODEL", "QUARTER", "COMPARISON", "VARIABLE", "VARIABLE_TYPE", "REFERENCE_LABEL",
    "BIN", "BIN_ORDER", "REFERENCE_COUNT", "MONITORING_COUNT",
    "REFERENCE_PROPORTION", "MONITORING_PROPORTION", "PROPORTION_CHANGE_PP",
    "CSI_CONTRIBUTION", "CONTRIBUTION_SHARE_PCT", "TOTAL_CSI", "SEVERITY",
]

BIN_DEFINITION_COLUMNS = [
    "MODEL", "VARIABLE", "VARIABLE_LABEL", "VARIABLE_TYPE", "VARIABLE_ROLE", "CRITICAL",
    "REFERENCE_LABEL", "REFERENCE_MODE", "BINNING_SOURCE", "N_BINS",
    "REFERENCE_RECORDS", "REFERENCE_MISSING", "REFERENCE_DISTINCT_VALUES",
    "DEFINITION", "USABLE", "NOTES",
]


@dataclass
class CsiResult:
    """CSI tables plus the fitted bin definitions (governance evidence)."""

    summary: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CSI_SUMMARY_COLUMNS))
    detail: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CSI_DETAIL_COLUMNS))
    definitions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BIN_DEFINITION_COLUMNS)
    )
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def top_variables(
        self, n: int = 25, comparison: str = VS_REFERENCE, quarter: Optional[str] = None
    ) -> pd.DataFrame:
        """Highest-CSI rows, for the executive dashboard."""
        if self.summary.empty:
            return self.summary
        subset = self.summary[self.summary["COMPARISON"] == comparison]
        if quarter is not None:
            subset = subset[subset["QUARTER"] == quarter]
        subset = subset[subset["CALCULABLE"] == True]  # noqa: E712
        return subset.sort_values("CSI", ascending=False).head(max(0, int(n)))

    def count_above(self, threshold: float, comparison: str = VS_REFERENCE,
                    quarter: Optional[str] = None) -> int:
        """Number of model-quarter-variable combinations with CSI >= threshold."""
        if self.summary.empty:
            return 0
        subset = self.summary[
            (self.summary["COMPARISON"] == comparison) & (self.summary["CALCULABLE"] == True)  # noqa: E712
        ]
        if quarter is not None:
            subset = subset[subset["QUARTER"] == quarter]
        return int((subset["CSI"] >= threshold).sum())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_csi(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    reference_masks: Dict[str, pd.Series],
    monitored_quarters: Sequence[QuarterInfo],
    execution_log: Optional[ExecutionLog] = None,
    kind_hints: Optional[Dict[str, str]] = None,
) -> CsiResult:
    """CSI for every model / quarter / monitored characteristic."""
    runtime = config.runtime
    thresholds = config.thresholds
    smoothing = config.smoothing
    quarter_labels = [q.label for q in monitored_quarters]

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    definition_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Pre-compute row masks once; slicing by index keeps peak memory to a
    # single column rather than a full copy of the reference population.
    quarter_masks: Dict[str, pd.Series] = {
        label: (frame[QUARTER_COLUMN] == label) for label in quarter_labels
    }

    for model in config.active_models:
        # Characteristic distributions are measured on the population this
        # scorecard actually scored (see runtime.csi_scope); unscored volume is
        # monitored by MER and Data Quality rather than silently inflating CSI.
        scope = model_scope_mask(frame, config, model, scope=runtime.csi_scope)
        quarter_index: Dict[str, pd.Index] = {
            label: frame.index[mask & scope] for label, mask in quarter_masks.items()
        }
        mask = reference_masks.get(model.name)
        if mask is None:
            message = (
                f"[{model.name}] reference population unavailable; CSI is skipped for this "
                "scorecard."
            )
            logger.warning(message)
            if execution_log is not None:
                execution_log.warning(message)
            skipped.append({"MODEL": model.name, "VARIABLE": "ALL", "REASON": message})
            continue

        reference_index = frame.index[
            mask.reindex(frame.index, fill_value=False) & scope
        ]
        reference_label = model.reference.label if model.reference else "NOT CONFIGURED"
        reference_mode = model.reference.mode.value if model.reference else "UNKNOWN"
        specs = config.monitored_variables(model.name)
        specs = [s for s in specs if s.name in frame.columns]
        if not specs:
            message = (
                f"[{model.name}] no monitored characteristics are present in the extract; only "
                "segment-level PSI will be produced."
            )
            logger.warning(message)
            if execution_log is not None:
                execution_log.warning(message)
            skipped.append({"MODEL": model.name, "VARIABLE": "ALL", "REASON": message})
            continue

        logger.info(
            "[%s] CSI on %d characteristic(s) x %d quarter(s) (batch size %d).",
            model.name, len(specs), len(quarter_labels), runtime.csi_variable_batch_size,
        )

        for batch in chunked(specs, runtime.csi_variable_batch_size):
            for spec in batch:
                definition = fit_bin_definition(
                    frame.loc[reference_index, spec.name], spec, runtime,
                    kind_hint=(kind_hints or {}).get(spec.name),
                )
                reference_application = definition.apply(
                    frame.loc[reference_index, spec.name]
                ) if definition.usable else None

                definition_rows.append({
                    "MODEL": model.name,
                    "VARIABLE": spec.name,
                    "VARIABLE_LABEL": spec.label,
                    "VARIABLE_TYPE": definition.var_type.value,
                    "VARIABLE_ROLE": spec.role.value,
                    "CRITICAL": bool(spec.critical),
                    "REFERENCE_LABEL": reference_label,
                    "REFERENCE_MODE": reference_mode,
                    "BINNING_SOURCE": definition.source,
                    "N_BINS": definition.n_bins,
                    "REFERENCE_RECORDS": definition.reference_total,
                    "REFERENCE_MISSING": definition.reference_missing,
                    "REFERENCE_DISTINCT_VALUES": definition.reference_distinct,
                    "DEFINITION": definition.describe(),
                    "USABLE": bool(definition.usable),
                    "NOTES": " ".join(definition.notes) if definition.notes else definition.reason,
                })

                if not definition.usable or reference_application is None:
                    skipped.append({
                        "MODEL": model.name, "VARIABLE": spec.name,
                        "REASON": definition.reason or "Bin definition unusable.",
                    })
                    for label in quarter_labels:
                        summary_rows.append(_unavailable_row(
                            model.name, label, VS_REFERENCE, spec, definition,
                            reference_label, reference_mode, thresholds,
                            reason=definition.reason or "Bin definition unusable.",
                        ))
                    continue

                band = thresholds.csi_band(spec.name, critical=spec.critical)
                previous_application: Optional[BinApplication] = None

                for position, label in enumerate(quarter_labels):
                    monitoring_application = definition.apply(
                        frame.loc[quarter_index[label], spec.name]
                    )

                    summary_row, rows = _csi_rows(
                        model_name=model.name,
                        quarter=label,
                        comparison=VS_REFERENCE,
                        spec=spec,
                        definition=definition,
                        reference_application=reference_application,
                        monitoring_application=monitoring_application,
                        reference_label=reference_label,
                        reference_mode=reference_mode,
                        config=config,
                        band=band,
                        min_reference=thresholds.min_reference_records,
                        min_monitoring=thresholds.min_variable_records,
                    )
                    summary_rows.append(summary_row)
                    detail_rows.extend(rows)

                    if runtime.csi_include_qoq and previous_application is not None:
                        summary_row, rows = _csi_rows(
                            model_name=model.name,
                            quarter=label,
                            comparison=VS_PREVIOUS_QUARTER,
                            spec=spec,
                            definition=definition,
                            reference_application=previous_application,
                            monitoring_application=monitoring_application,
                            reference_label=quarter_labels[position - 1],
                            reference_mode="PREVIOUS_QUARTER",
                            config=config,
                            band=band,
                            min_reference=thresholds.min_variable_records,
                            min_monitoring=thresholds.min_variable_records,
                        )
                        summary_rows.append(summary_row)
                        detail_rows.extend(rows)
                    previous_application = monitoring_application
            gc.collect()

    summary = pd.DataFrame(summary_rows, columns=CSI_SUMMARY_COLUMNS)
    detail = pd.DataFrame(detail_rows, columns=CSI_DETAIL_COLUMNS)
    definitions = pd.DataFrame(definition_rows, columns=BIN_DEFINITION_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(
            ["MODEL", "COMPARISON", "QUARTER", "CSI"], ascending=[True, True, True, False]
        ).reset_index(drop=True)
    if not detail.empty:
        detail = detail.sort_values(
            ["MODEL", "COMPARISON", "QUARTER", "VARIABLE", "BIN_ORDER"]
        ).reset_index(drop=True)

    amber = int((summary["SEVERITY"] == "AMBER").sum()) if not summary.empty else 0
    red = int((summary["SEVERITY"] == "RED").sum()) if not summary.empty else 0
    logger.info(
        "CSI: %d model-quarter-variable row(s); %d AMBER, %d RED; %d characteristic(s) skipped.",
        len(summary), amber, red, len(skipped),
    )
    return CsiResult(summary=summary, detail=detail, definitions=definitions, skipped=skipped)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _csi_rows(
    *,
    model_name: str,
    quarter: str,
    comparison: str,
    spec: VariableSpec,
    definition: BinDefinition,
    reference_application: BinApplication,
    monitoring_application: BinApplication,
    reference_label: str,
    reference_mode: str,
    config: MonitoringConfig,
    band: ThresholdBand,
    min_reference: int,
    min_monitoring: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    smoothing = config.smoothing
    decimals = config.runtime.display_decimals

    index = compute_distribution_index(
        reference_application.counts.to_numpy(dtype="float64"),
        monitoring_application.counts.reindex(reference_application.counts.index)
        .fillna(0).to_numpy(dtype="float64"),
        [str(b) for b in reference_application.counts.index],
        smoothing,
        min_reference_total=min_reference,
        min_monitoring_total=min_monitoring,
    )
    severity, reason = classify_index(
        index, band, min_reference_total=min_reference, min_monitoring_total=min_monitoring
    )
    if index.calculable and index.contribution_sum_error > smoothing.tolerance:
        logger.error(
            "[%s] %s %s: bin contributions do not sum to the reported CSI (error %.3e).",
            model_name, quarter, spec.name, index.contribution_sum_error,
        )

    top_bin, top_contribution, top_share = index.top_contributor()
    missing_reference = reference_application.missing_pct
    missing_monitoring = monitoring_application.missing_pct
    missingness_change = (
        None if (missing_reference is None or missing_monitoring is None)
        else missing_monitoring - missing_reference
    )

    summary_row = {
        "MODEL": model_name,
        "QUARTER": quarter,
        "COMPARISON": comparison,
        "VARIABLE": spec.name,
        "VARIABLE_LABEL": spec.label,
        "VARIABLE_TYPE": definition.var_type.value,
        "VARIABLE_ROLE": spec.role.value,
        "CRITICAL": bool(spec.critical),
        "REFERENCE_LABEL": reference_label,
        "REFERENCE_MODE": reference_mode,
        "REFERENCE_COUNT": index.reference_total,
        "MONITORING_COUNT": index.monitoring_total,
        "N_BINS": index.n_bins,
        "BINNING_SOURCE": definition.source,
        "BINNING_DEFINITION": definition.describe(),
        "CSI": index.total,
        "CSI_DISPLAY": round_or_none(index.total, decimals),
        "SEVERITY": severity.value,
        "THRESHOLD_AMBER": band.amber,
        "THRESHOLD_RED": band.red,
        "TOP_CONTRIBUTING_BIN": top_bin,
        "TOP_BIN_CONTRIBUTION": round_or_none(top_contribution, decimals),
        "TOP_BIN_SHARE_PCT": round_or_none(top_share, 2),
        "MAX_ABS_PROP_SHIFT_PP": round_or_none(index.max_abs_shift_pp(), 2),
        "MISSING_PCT_REF": round_or_none(missing_reference, 2),
        "MISSING_PCT_MON": round_or_none(missing_monitoring, 2),
        "MISSINGNESS_CHANGE_PP": round_or_none(missingness_change, 2),
        "UNSEEN_CATEGORY_COUNT": monitoring_application.unseen_count,
        "UNSEEN_CATEGORY_PCT": round_or_none(monitoring_application.unseen_pct, 2),
        "UNSEEN_CATEGORY_EXAMPLES": ", ".join(monitoring_application.unseen_values[:8]),
        "OTHER_BIN_PCT_MON": round_or_none(monitoring_application.other_pct, 2),
        "INFINITE_COUNT_MON": monitoring_application.infinite_count,
        "SMOOTHING": _smoothing_label(config),
        "BINS_DROPPED": index.bins_dropped,
        "CONTRIBUTION_SUM_ERROR": index.contribution_sum_error,
        "CALCULABLE": index.calculable,
        "STATUS_REASON": reason,
        "RECOMMENDED_ACTION": recommend_action(
            severity=severity,
            csi=index.total,
            variable=spec,
            definition=definition,
            top_bin=top_bin,
            missingness_change=missingness_change,
            unseen_pct=monitoring_application.unseen_pct,
            calculable=index.calculable,
        ),
    }

    detail_rows: List[Dict[str, Any]] = []
    for order, row in enumerate(index.to_rows(), start=1):
        detail_rows.append({
            "MODEL": model_name,
            "QUARTER": quarter,
            "COMPARISON": comparison,
            "VARIABLE": spec.name,
            "VARIABLE_TYPE": definition.var_type.value,
            "REFERENCE_LABEL": reference_label,
            "BIN": row["CATEGORY"],
            "BIN_ORDER": order,
            "REFERENCE_COUNT": row["REFERENCE_COUNT"],
            "MONITORING_COUNT": row["MONITORING_COUNT"],
            "REFERENCE_PROPORTION": row["REFERENCE_PROPORTION"],
            "MONITORING_PROPORTION": row["MONITORING_PROPORTION"],
            "PROPORTION_CHANGE_PP": row["PROPORTION_CHANGE_PP"],
            "CSI_CONTRIBUTION": row["CONTRIBUTION"],
            "CONTRIBUTION_SHARE_PCT": row["CONTRIBUTION_SHARE_PCT"],
            "TOTAL_CSI": index.total,
            "SEVERITY": severity.value,
        })
    return summary_row, detail_rows


def _unavailable_row(
    model_name: str,
    quarter: str,
    comparison: str,
    spec: VariableSpec,
    definition: BinDefinition,
    reference_label: str,
    reference_mode: str,
    thresholds,
    *,
    reason: str,
) -> Dict[str, Any]:
    """A GREY row so an unmonitorable characteristic is visible, not absent."""
    band = thresholds.csi_band(spec.name, critical=spec.critical)
    row = {column: None for column in CSI_SUMMARY_COLUMNS}
    row.update({
        "MODEL": model_name,
        "QUARTER": quarter,
        "COMPARISON": comparison,
        "VARIABLE": spec.name,
        "VARIABLE_LABEL": spec.label,
        "VARIABLE_TYPE": definition.var_type.value,
        "VARIABLE_ROLE": spec.role.value,
        "CRITICAL": bool(spec.critical),
        "REFERENCE_LABEL": reference_label,
        "REFERENCE_MODE": reference_mode,
        "REFERENCE_COUNT": definition.reference_total,
        "MONITORING_COUNT": 0,
        "N_BINS": definition.n_bins,
        "BINNING_SOURCE": definition.source,
        "BINNING_DEFINITION": definition.describe(),
        "CSI": np.nan,
        "CSI_DISPLAY": None,
        "SEVERITY": Severity.GREY.value,
        "THRESHOLD_AMBER": band.amber,
        "THRESHOLD_RED": band.red,
        "UNSEEN_CATEGORY_COUNT": 0,
        "BINS_DROPPED": 0,
        "CONTRIBUTION_SUM_ERROR": 0.0,
        "CALCULABLE": False,
        "STATUS_REASON": reason,
        "RECOMMENDED_ACTION": (
            "Restore the characteristic in the extract or remove it from the monitoring "
            "configuration. A scorecard input that cannot be measured cannot be assured."
        ),
    })
    return row


def _smoothing_label(config: MonitoringConfig) -> str:
    from .psi_calculator import _smoothing_label as label

    return label(config.smoothing)


def recommend_action(
    *,
    severity: Severity,
    csi: Optional[float],
    variable: VariableSpec,
    definition: BinDefinition,
    top_bin: Optional[str],
    missingness_change: Optional[float],
    unseen_pct: Optional[float],
    calculable: bool,
) -> str:
    """Concrete next step for the model owner, not a generic platitude."""
    if not calculable:
        return (
            "Metric unavailable. Confirm the characteristic is populated in both the reference "
            "period and the monitoring quarter before drawing any conclusion."
        )
    parts: List[str] = []
    if severity is Severity.GREY:
        parts.append(
            "Insufficient observations for a reliable reading; re-assess once volumes recover "
            "or widen the comparison window."
        )
    elif severity is Severity.RED:
        parts.append(
            f"Escalate to the model owner: the distribution of '{variable.name}' has shifted "
            "materially against its reference period."
        )
    elif severity is Severity.AMBER:
        parts.append(f"Investigate the shift in '{variable.name}'.")
    else:
        parts.append("No action; the characteristic is stable against its reference period.")

    if severity in (Severity.AMBER, Severity.RED):
        if top_bin == MISSING_BIN:
            parts.append(
                "The movement is concentrated in the MISSING bin, so this is most likely a data "
                "supply problem rather than a change in customer mix -- check the feature "
                "pipeline and any upstream bureau/API outage first."
            )
        elif top_bin == UNSEEN_BIN:
            parts.append(
                "The movement is concentrated in values never seen in the reference period: "
                "check for re-coded categories or a new product/channel."
            )
        elif top_bin == OTHER_BIN:
            parts.append(
                "The movement sits in the grouped OTHER bin; supply an approved category "
                "grouping so the driver is visible."
            )
        elif top_bin:
            parts.append(f"The largest contribution comes from bin '{top_bin}'.")
        if variable.role.value == "SCORE":
            parts.append(
                "This is the final score distribution: a material shift here changes approval "
                "rates and cut-off behaviour directly."
            )
        elif variable.role.value == "SUBSCORE":
            parts.append(
                "This is a variable-level subscore: check both the underlying characteristic and "
                "the points mapping applied in production."
            )
        if variable.critical:
            parts.append("Flagged business-critical, so the tighter threshold band applies.")
    if missingness_change is not None and abs(missingness_change) >= 5.0:
        parts.append(
            f"Missingness moved {missingness_change:+.1f} pp versus the reference period; "
            "resolve that before attributing the shift to population change."
        )
    if unseen_pct is not None and unseen_pct >= 1.0:
        parts.append(
            f"{unseen_pct:.1f}% of records carry an unseen level and cannot be scored as "
            "developed; a mapping decision is required."
        )
    return " ".join(parts)


__all__ = [
    "CSI_SUMMARY_COLUMNS", "CSI_DETAIL_COLUMNS", "BIN_DEFINITION_COLUMNS",
    "CsiResult", "compute_csi", "recommend_action",
]
