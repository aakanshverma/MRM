"""Unified alert register.

Every metric produced by the pipeline is funnelled through one table with one
schema, so that a reviewer has a single place to answer "what needs attention
this quarter, and why".

Severity labels
---------------
``GREEN``  within normal monitoring limits
``AMBER``  investigation required
``RED``    significant breach requiring escalation
``GREY``   insufficient data, or the metric could not be calculated

Two design rules matter as much as the thresholds:

* **No misleading alerts.**  A metric computed on too little data is ``GREY``
  with the sample size stated -- never ``RED`` (false alarm) and never
  ``GREEN`` (false comfort).
* **A breach triggers investigation, not rejection.**  Every AMBER/RED row
  carries a concrete next step and an open investigation status; none of them
  says "retire the model".
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MonitoringConfig, Severity, severity_rank
from .logging_config import get_logger
from .psi_calculator import VS_PREVIOUS_QUARTER, VS_REFERENCE
from .reference_periods import ReferenceResolution
from .segment_migration import COMPARISON_QOQ, COMPARISON_REFERENCE, MIGRATION_CAVEAT
from .utils import NOT_AVAILABLE, round_or_none

logger = get_logger("alerts")

ALERT_COLUMNS = [
    "ALERT_ID", "CATEGORY", "MODEL", "QUARTER", "METRIC", "VARIABLE_OR_SEGMENT",
    "METRIC_VALUE", "UNIT", "THRESHOLD", "SEVERITY", "REFERENCE_PERIOD",
    "MONITORING_PERIOD", "EXPLANATION", "RECOMMENDED_ACTION",
    "INVESTIGATION_STATUS", "MANUAL_REVIEW_STATUS", "MODEL_OWNER_NOTES",
]

STATUS_OPEN = "OPEN - PENDING MODEL OWNER REVIEW"
STATUS_ESCALATED = "OPEN - ESCALATION REQUIRED"
STATUS_NONE = "NOT REQUIRED"
STATUS_DATA_GAP = "OPEN - DATA GAP TO RESOLVE"

_NOT_REJECTION = (
    "A stability breach triggers investigation under the model risk policy; it does not by "
    "itself invalidate the scorecard."
)


def _status(severity: Severity) -> str:
    if severity is Severity.RED:
        return STATUS_ESCALATED
    if severity is Severity.AMBER:
        return STATUS_OPEN
    if severity is Severity.GREY:
        return STATUS_DATA_GAP
    return STATUS_NONE


def _alert(
    alert_id: str,
    category: str,
    *,
    model: str,
    quarter: str,
    metric: str,
    subject: str,
    value: Any,
    unit: str,
    threshold: str,
    severity: Severity,
    reference_period: str,
    monitoring_period: str,
    explanation: str,
    action: str,
) -> Dict[str, Any]:
    return {
        "ALERT_ID": alert_id,
        "CATEGORY": category,
        "MODEL": model,
        "QUARTER": quarter,
        "METRIC": metric,
        "VARIABLE_OR_SEGMENT": subject,
        "METRIC_VALUE": value,
        "UNIT": unit,
        "THRESHOLD": threshold,
        "SEVERITY": severity.value if isinstance(severity, Severity) else str(severity),
        "REFERENCE_PERIOD": reference_period,
        "MONITORING_PERIOD": monitoring_period,
        "EXPLANATION": explanation,
        "RECOMMENDED_ACTION": action,
        "INVESTIGATION_STATUS": _status(
            severity if isinstance(severity, Severity) else Severity(str(severity))
        ),
        "MANUAL_REVIEW_STATUS": "NOT STARTED",
        "MODEL_OWNER_NOTES": "",
    }


# ---------------------------------------------------------------------------
# Builders per source
# ---------------------------------------------------------------------------


def _psi_alerts(psi_summary: pd.DataFrame, config: MonitoringConfig) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if psi_summary is None or psi_summary.empty:
        return alerts
    band = config.thresholds.psi
    for index, record in enumerate(psi_summary.to_dict("records"), start=1):
        comparison = record["COMPARISON"]
        fixed = comparison == VS_REFERENCE
        severity = Severity(str(record["SEVERITY"]))
        value = record.get("PSI")
        top_segment = record.get("TOP_CONTRIBUTING_SEGMENT") or NOT_AVAILABLE
        metric = (
            "PSI (fixed reference)" if fixed else "PSI (quarter-on-quarter)"
        )
        if fixed:
            explanation = (
                f"The {record['MODEL']} segment distribution in {record['QUARTER']} "
                f"({record['MONITORING_COUNT']:,} records) compared against its own "
                f"{record['REFERENCE_MODE']} reference period {record['REFERENCE_LABEL']} "
                f"({record['REFERENCE_COUNT']:,} records). "
            )
        else:
            explanation = (
                f"The {record['MODEL']} segment distribution in {record['QUARTER']} compared "
                f"against the immediately preceding quarter {record['REFERENCE_LABEL']}. This "
                "view catches recent deterioration that a fixed development baseline can mask. "
            )
        if value is not None and value == value:
            explanation += (
                f"PSI = {value:.4f}, with the largest contribution from '{top_segment}' "
                f"({record.get('TOP_CONTRIBUTION_SHARE_PCT') or 0:.0f}% of the total) and a "
                f"maximum segment shift of {record.get('MAX_ABS_PROP_SHIFT_PP') or 0:.1f} "
                "percentage points."
            )
        if record.get("STATUS_REASON"):
            explanation += f" {record['STATUS_REASON']}"

        if severity is Severity.RED:
            action = (
                f"Escalate: identify what changed in the applicants landing in '{top_segment}'. "
                "Review the CSI sheet for the characteristics driving the shift, confirm no "
                "scoring-engine or policy change coincides with it, and document the conclusion "
                f"in the model monitoring record. {_NOT_REJECTION}"
            )
        elif severity is Severity.AMBER:
            action = (
                f"Investigate the movement in '{top_segment}' against sourcing-mix and policy "
                "changes, and check whether the quarter-on-quarter PSI shows the shift is "
                "recent or long-standing."
            )
        elif severity is Severity.GREY:
            action = (
                "Resolve the data limitation before drawing a conclusion; the sample is too "
                "small for a reliable population-stability read."
            )
        else:
            action = "No action. Continue routine quarterly monitoring."
        alerts.append(_alert(
            f"PSI-{index:04d}",
            "POPULATION STABILITY",
            model=str(record["MODEL"]),
            quarter=str(record["QUARTER"]),
            metric=metric,
            subject=str(top_segment),
            value=round_or_none(value, config.runtime.display_decimals),
            unit="index",
            threshold=band.label(),
            severity=severity,
            reference_period=str(record["REFERENCE_LABEL"]),
            monitoring_period=str(record["QUARTER"]),
            explanation=explanation,
            action=action,
        ))
    return alerts


def _csi_alerts(csi_summary: pd.DataFrame, config: MonitoringConfig) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if csi_summary is None or csi_summary.empty:
        return alerts
    include_green = config.runtime.alert_include_green
    for index, record in enumerate(csi_summary.to_dict("records"), start=1):
        severity = Severity(str(record["SEVERITY"]))
        if severity is Severity.GREEN and not include_green:
            continue
        value = record.get("CSI")
        comparison = record.get("COMPARISON")
        metric = (
            "CSI (fixed reference)" if comparison == VS_REFERENCE
            else "CSI (quarter-on-quarter)"
        )
        explanation = (
            f"Characteristic '{record['VARIABLE']}' ({str(record.get('VARIABLE_TYPE'))}, role "
            f"{record.get('VARIABLE_ROLE')}) binned on {record.get('BINNING_SOURCE')} with "
            f"{record.get('N_BINS')} bins. Reference {record.get('REFERENCE_COUNT')} records vs "
            f"monitoring {record.get('MONITORING_COUNT')} records."
        )
        if value is not None and value == value:
            explanation += (
                f" CSI = {value:.4f}; largest contribution from bin "
                f"'{record.get('TOP_CONTRIBUTING_BIN')}'."
            )
        if record.get("MISSINGNESS_CHANGE_PP") not in (None, 0):
            explanation += (
                f" Missingness moved {record['MISSINGNESS_CHANGE_PP']:+.2f} pp "
                f"({record.get('MISSING_PCT_REF')}% -> {record.get('MISSING_PCT_MON')}%)."
            )
        if record.get("UNSEEN_CATEGORY_COUNT"):
            explanation += (
                f" {record['UNSEEN_CATEGORY_COUNT']} record(s) carry an unseen level "
                f"({record.get('UNSEEN_CATEGORY_EXAMPLES')})."
            )
        if record.get("STATUS_REASON"):
            explanation += f" {record['STATUS_REASON']}"
        alerts.append(_alert(
            f"CSI-{index:05d}",
            "CHARACTERISTIC STABILITY",
            model=str(record["MODEL"]),
            quarter=str(record["QUARTER"]),
            metric=metric,
            subject=str(record["VARIABLE"]),
            value=round_or_none(value, config.runtime.display_decimals),
            unit="index",
            threshold=(
                f"GREEN < {record.get('THRESHOLD_AMBER')} <= AMBER < "
                f"{record.get('THRESHOLD_RED')} <= RED"
                + (" (business-critical band)" if record.get("CRITICAL") else "")
            ),
            severity=severity,
            reference_period=str(record.get("REFERENCE_LABEL")),
            monitoring_period=str(record["QUARTER"]),
            explanation=explanation,
            action=str(record.get("RECOMMENDED_ACTION") or ""),
        ))
    return alerts


def _migration_alerts(
    migration_summary: pd.DataFrame, config: MonitoringConfig
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if migration_summary is None or migration_summary.empty:
        return alerts
    band = config.thresholds.net_downgrade_pp
    for index, record in enumerate(migration_summary.to_dict("records"), start=1):
        severity = Severity(str(record["SEVERITY"]))
        comparison = record["COMPARISON"]
        metric = (
            "Net migration to riskier grades (vs reference)"
            if comparison == COMPARISON_REFERENCE
            else "Net migration to riskier grades (quarter-on-quarter)"
        )
        explanation = str(record.get("INTERPRETATION") or "")
        if severity is Severity.RED:
            action = (
                "Escalate to the model owner and credit policy: a material share of matched "
                "customers has moved to riskier grades. Separate genuine credit deterioration "
                "from sourcing-mix, policy and scoring-configuration change before concluding. "
                + _NOT_REJECTION
            )
        elif severity is Severity.AMBER:
            action = (
                f"Investigate the dominant flow ({record.get('LARGEST_DOWNGRADE_FLOW') or 'n/a'}) "
                "and confirm whether the same customers are genuinely riskier or simply "
                "re-applying under different circumstances."
            )
        elif severity is Severity.GREY:
            action = (
                "Customer-level migration is not measurable here. Read the Segment Distribution "
                "and PSI sheets for the population-level shift instead."
            )
        else:
            action = "No action. Matched-customer migration is within normal limits."
        alerts.append(_alert(
            f"MIG-{index:04d}",
            "SEGMENT MIGRATION",
            model=str(record["MODEL"]),
            quarter=str(record["TO_PERIOD"]),
            metric=metric,
            subject=str(record.get("LARGEST_DOWNGRADE_FLOW") or "n/a"),
            value=record.get("NET_DOWNGRADE_PP"),
            unit="pp",
            threshold=band.label(),
            severity=severity,
            reference_period=str(record["FROM_PERIOD"]),
            monitoring_period=str(record["TO_PERIOD"]),
            explanation=explanation,
            action=action,
        ))
    return alerts


def _data_quality_alerts(
    data_quality: pd.DataFrame, config: MonitoringConfig
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if data_quality is None or data_quality.empty:
        return alerts
    include_green = config.runtime.alert_include_green
    for index, record in enumerate(data_quality.to_dict("records"), start=1):
        severity = Severity(str(record["SEVERITY"]))
        if severity is Severity.GREEN and not include_green:
            continue
        alerts.append(_alert(
            f"DQ-{index:05d}",
            f"DATA QUALITY / {record.get('CATEGORY')}",
            model=str(record.get("MODEL") or "ALL"),
            quarter=str(record.get("QUARTER") or "ALL"),
            metric=str(record.get("CHECK")),
            subject=str(record.get("METRIC") or record.get("SCOPE") or ""),
            value=record.get("VALUE"),
            unit=str(record.get("UNIT") or ""),
            threshold=str(record.get("THRESHOLD") or ""),
            severity=severity,
            reference_period="n/a",
            monitoring_period=str(record.get("QUARTER") or "ALL"),
            explanation=str(record.get("DETAIL") or ""),
            action=str(record.get("RECOMMENDED_ACTION") or ""),
        ))
    return alerts


def _performance_alerts(
    performance: pd.DataFrame, config: MonitoringConfig
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if performance is None or performance.empty:
        return alerts
    thresholds = config.thresholds
    for index, record in enumerate(performance.to_dict("records"), start=1):
        severity = Severity(str(record.get("PERFORMANCE_SEVERITY") or Severity.GREY.value))
        gini = record.get("GINI")
        explanation = (
            f"Discrimination for {record['MODEL']} in {record['QUARTER']}: Gini "
            f"{gini if gini is not None else NOT_AVAILABLE}, AUC {record.get('AUC')}, KS "
            f"{record.get('KS_STATISTIC')} on {record.get('MATURED_RECORDS')} matured record(s) "
            f"({record.get('BAD_COUNT')} bad). Rank ordering: {record.get('RANK_ORDERING')}. "
            f"{record.get('PERFORMANCE_STATUS') or ''}"
        )
        if severity is Severity.RED:
            action = (
                "Escalate: discrimination has fallen below the governance floor. Commission a "
                "full performance review covering recalibration, re-development and interim "
                "overlay options. Confirm first that the outcome window and bad definition are "
                "correct, since a data error produces the same symptom."
            )
        elif severity is Severity.AMBER:
            action = (
                "Investigate the fall in discriminatory power alongside the stability metrics; "
                "a population shift and a performance drop together usually point to a "
                "different applicant mix rather than a broken model."
            )
        elif severity is Severity.GREY:
            action = (
                "Performance cannot be assessed for this quarter. Do not infer that the model "
                "is performing: re-assess once the performance window has matured."
            )
        else:
            action = "No action. Discrimination remains within the governance floor."
        alerts.append(_alert(
            f"PERF-{index:04d}",
            "PREDICTIVE PERFORMANCE",
            model=str(record["MODEL"]),
            quarter=str(record["QUARTER"]),
            metric="Gini (discriminatory power)",
            subject="ALL SEGMENTS",
            value=gini,
            unit="index",
            threshold=(
                f"GREEN >= {thresholds.min_gini_amber:g} > AMBER >= "
                f"{thresholds.min_gini_red:g} > RED"
            ),
            severity=severity,
            reference_period=f"{record.get('OBSERVATION_WINDOW_MONTHS')}-month outcome window",
            monitoring_period=str(record["QUARTER"]),
            explanation=explanation,
            action=action,
        ))
        verdict = str(record.get("CALIBRATION_VERDICT") or "")
        if verdict and not verdict.startswith(NOT_AVAILABLE):
            calibration_severity = (
                Severity.GREEN if verdict.startswith("WITHIN TOLERANCE")
                else (Severity.RED if verdict.startswith("UNDER-PREDICTING") else Severity.AMBER)
            )
            alerts.append(_alert(
                f"CAL-{index:04d}",
                "PREDICTIVE PERFORMANCE",
                model=str(record["MODEL"]),
                quarter=str(record["QUARTER"]),
                metric="Calibration (observed vs expected PD)",
                subject="ALL SEGMENTS",
                value=record.get("OBSERVED_VS_EXPECTED_RATIO"),
                unit="observed/expected",
                threshold="|z| <= 1.96 (95% binomial tolerance)",
                severity=calibration_severity,
                reference_period=f"expected PD column, {record.get('OBSERVATION_WINDOW_MONTHS')}-month window",
                monitoring_period=str(record["QUARTER"]),
                explanation=(
                    f"Observed bad rate {record.get('EVENT_RATE_PCT')}% against an expected "
                    f"{record.get('EXPECTED_BAD_RATE_PCT')}% (ratio "
                    f"{record.get('OBSERVED_VS_EXPECTED_RATIO')}, z={record.get('CALIBRATION_Z')}). "
                    f"{verdict}."
                ),
                action=(
                    "No action; the calibration holds." if calibration_severity is Severity.GREEN
                    else "Review the PD calibration and the central tendency assumption with the "
                         "model owner before the next pricing or provisioning cycle."
                ),
            ))
    return alerts


def _mer_alerts(
    mer: pd.DataFrame, config: MonitoringConfig,
    resolutions: Mapping[str, ReferenceResolution],
) -> List[Dict[str, Any]]:
    """Sample-size, concentration and reference-adequacy alerts."""
    alerts: List[Dict[str, Any]] = []
    thresholds = config.thresholds
    counter = 0

    for name, resolution in resolutions.items():
        counter += 1
        severity = Severity.GREEN if resolution.sufficient else Severity.GREY
        alerts.append(_alert(
            f"REF-{counter:04d}",
            "REFERENCE ADEQUACY",
            model=name,
            quarter="ALL",
            metric="Reference population size",
            subject=f"{resolution.mode} baseline",
            value=resolution.scored_count,
            unit="scored records",
            threshold=f">= {thresholds.min_reference_records:,} scored records",
            severity=severity,
            reference_period=resolution.label,
            monitoring_period="ALL",
            explanation=(
                f"{name} is monitored against {resolution.label} [{resolution.mode}] holding "
                f"{resolution.record_count:,} record(s), of which {resolution.scored_count:,} "
                f"carry a valid segment ({resolution.share_of_dataset_pct or 0:.1f}% of the "
                f"extract). "
                + (" ".join(resolution.warnings) if resolution.warnings else "")
            ),
            action=(
                "No action; the baseline is adequate." if resolution.sufficient else
                "Agree a larger or re-approved baseline with the model owner. Expected "
                "proportions from a small reference sample make every PSI/CSI reading unstable, "
                "so all of this model's stability metrics are reported as GREY."
            ),
        ))

    if mer is None or mer.empty:
        return alerts

    for index, record in enumerate(mer.to_dict("records"), start=1):
        model = str(record["MODEL"])
        quarter = str(record["QUARTER"])
        reference_label = str(record.get("REFERENCE_LABEL"))

        # -- insufficient monitoring sample -------------------------------
        total = record.get("TOTAL_RECORDS") or 0
        if total < thresholds.min_monitoring_records:
            alerts.append(_alert(
                f"SMP-{index:04d}", "SAMPLE ADEQUACY",
                model=model, quarter=quarter,
                metric="Monitoring quarter sample size", subject="ALL SEGMENTS",
                value=total, unit="records",
                threshold=f">= {thresholds.min_monitoring_records:,} records",
                severity=Severity.GREY,
                reference_period=reference_label, monitoring_period=quarter,
                explanation=(
                    f"{quarter} holds only {total:,} record(s) for {model}, below the configured "
                    f"minimum of {thresholds.min_monitoring_records:,}. Stability metrics for "
                    "this quarter are reported but labelled GREY rather than GREEN or RED, to "
                    "avoid a misleading alert from sampling noise."
                ),
                action=(
                    "Confirm whether the low volume is expected (product wind-down, channel "
                    "pause) or an extract defect, and consider aggregating two quarters for a "
                    "stability read."
                ),
            ))

        # -- incomplete quarter -------------------------------------------
        if not record.get("QUARTER_COMPLETE", True):
            alerts.append(_alert(
                f"INC-{index:04d}", "COVERAGE",
                model=model, quarter=quarter,
                metric="Incomplete monitoring quarter", subject="ALL SEGMENTS",
                value=total, unit="records",
                threshold="full calendar quarter",
                severity=Severity.GREY,
                reference_period=reference_label, monitoring_period=quarter,
                explanation=(
                    f"{quarter} ({record.get('QUARTER_START')} to {record.get('QUARTER_END')}) is "
                    "not a complete calendar quarter in this extract, so its metrics are not "
                    "comparable with closed quarters. It is included because "
                    "runtime.include_partial_quarters is enabled."
                ),
                action="Re-run the pack once the quarter closes before relying on these figures.",
            ))

        # -- segment concentration ----------------------------------------
        concentration = record.get("SEGMENT_CONCENTRATION_PCT")
        if concentration is not None and concentration == concentration:
            severity = thresholds.concentration_pct.classify(concentration)
            if severity is not Severity.GREEN:
                alerts.append(_alert(
                    f"CON-{index:04d}", "SCORECARD OUTPUT",
                    model=model, quarter=quarter,
                    metric="Concentration in a single segment",
                    subject=str(record.get("MOST_POPULATED_SEGMENT")),
                    value=concentration, unit="%",
                    threshold=thresholds.concentration_pct.label(),
                    severity=severity,
                    reference_period=reference_label, monitoring_period=quarter,
                    explanation=(
                        f"{concentration:.1f}% of scored applications in {quarter} fall in "
                        f"'{record.get('MOST_POPULATED_SEGMENT')}' (HHI "
                        f"{record.get('SEGMENT_HHI')}). A scorecard that assigns most of the book "
                        "to one grade has little practical discriminatory value, whatever its "
                        "development statistics showed."
                    ),
                    action=(
                        "Review the grade boundaries with the model owner: either the cut-offs "
                        "no longer match the current population or the scorecard needs "
                        "re-calibration."
                    ),
                ))

        # -- riskier-segment share increase -------------------------------
        riskier_change = record.get("RISKIER_SEGMENT_SHARE_CHANGE_PP")
        if riskier_change is not None and riskier_change == riskier_change and riskier_change > 0:
            severity = thresholds.riskier_share_increase_pp.classify(riskier_change)
            if severity is not Severity.GREEN:
                alerts.append(_alert(
                    f"RSK-{index:04d}", "SCORECARD OUTPUT",
                    model=model, quarter=quarter,
                    metric="Increase in riskier-segment share",
                    subject=", ".join(config.segments.riskier_segments),
                    value=riskier_change, unit="pp",
                    threshold=thresholds.riskier_share_increase_pp.label(),
                    severity=severity,
                    reference_period=reference_label, monitoring_period=quarter,
                    explanation=(
                        f"The share of applications in the riskiest grade(s) "
                        f"({', '.join(config.segments.riskier_segments)}) is "
                        f"{record.get('RISKIER_SEGMENT_SHARE_PCT')}% in {quarter} against "
                        f"{record.get('RISKIER_SEGMENT_SHARE_REF_PCT')}% in the reference period, "
                        f"a move of {riskier_change:+.1f} pp. Average segment rank moved "
                        f"{record.get('AVERAGE_SEGMENT_RANK_CHANGE')} on a 1-"
                        f"{config.segments.n_segments} scale."
                    ),
                    action=(
                        "Quantify the expected impact on approval rates, pricing and expected "
                        "credit loss, and confirm with the business whether the sourcing mix "
                        "changed deliberately."
                    ),
                ))
    return alerts


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_alerts(
    config: MonitoringConfig,
    *,
    psi_summary: Optional[pd.DataFrame] = None,
    csi_summary: Optional[pd.DataFrame] = None,
    migration_summary: Optional[pd.DataFrame] = None,
    data_quality: Optional[pd.DataFrame] = None,
    performance: Optional[pd.DataFrame] = None,
    mer: Optional[pd.DataFrame] = None,
    resolutions: Optional[Mapping[str, ReferenceResolution]] = None,
) -> pd.DataFrame:
    """Assemble the single alert register from every metric source."""
    rows: List[Dict[str, Any]] = []
    rows.extend(_psi_alerts(psi_summary, config))
    rows.extend(_csi_alerts(csi_summary, config))
    rows.extend(_migration_alerts(migration_summary, config))
    rows.extend(_data_quality_alerts(data_quality, config))
    rows.extend(_performance_alerts(performance, config))
    rows.extend(_mer_alerts(mer, config, resolutions or {}))

    table = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    if table.empty:
        return table
    table["__SEVRANK"] = table["SEVERITY"].map(severity_rank)
    table = (
        table.sort_values(
            ["__SEVRANK", "CATEGORY", "MODEL", "QUARTER"],
            ascending=[False, True, True, False],
        )
        .drop(columns="__SEVRANK")
        .reset_index(drop=True)
    )
    counts = table["SEVERITY"].value_counts().to_dict()
    logger.info(
        "Alerts: %d row(s) -- RED %d, AMBER %d, GREY %d, GREEN %d.",
        len(table), counts.get("RED", 0), counts.get("AMBER", 0),
        counts.get("GREY", 0), counts.get("GREEN", 0),
    )
    return table


def alert_counts(alerts: pd.DataFrame) -> Dict[str, int]:
    """``{'RED': n, 'AMBER': n, 'GREY': n, 'GREEN': n}`` with zeros present."""
    base = {severity.value: 0 for severity in Severity}
    if alerts is None or alerts.empty:
        return base
    base.update({str(k): int(v) for k, v in alerts["SEVERITY"].value_counts().items()})
    return base


def overall_status(alerts: pd.DataFrame) -> str:
    """Head-line monitoring status for the dashboard."""
    counts = alert_counts(alerts)
    if counts.get("RED"):
        return "RED - ESCALATION REQUIRED"
    if counts.get("AMBER"):
        return "AMBER - INVESTIGATION REQUIRED"
    if counts.get("GREY") and not counts.get("GREEN"):
        return "GREY - INSUFFICIENT DATA TO CONCLUDE"
    if counts.get("GREY"):
        return "GREEN WITH DATA GAPS"
    return "GREEN - WITHIN MONITORING LIMITS"


def top_concerns(
    alerts: pd.DataFrame, n: int = 5, *, latest_quarter: Optional[str] = None
) -> pd.DataFrame:
    """The ``n`` most serious stability concerns, for the dashboard."""
    if alerts is None or alerts.empty:
        return pd.DataFrame(columns=["RANK", "SEVERITY", "MODEL", "QUARTER", "METRIC",
                                     "VARIABLE_OR_SEGMENT", "METRIC_VALUE", "CONCERN"])
    subset = alerts[alerts["SEVERITY"].isin(["RED", "AMBER"])].copy()
    if subset.empty:
        subset = alerts[alerts["SEVERITY"] == "GREY"].copy()
    if latest_quarter is not None and not subset.empty:
        preferred = subset[subset["QUARTER"].isin([latest_quarter, "ALL"])]
        if not preferred.empty:
            subset = preferred
    if subset.empty:
        return pd.DataFrame(
            [{
                "RANK": 1, "SEVERITY": "GREEN", "MODEL": "ALL", "QUARTER": "ALL",
                "METRIC": "No breach", "VARIABLE_OR_SEGMENT": "n/a", "METRIC_VALUE": None,
                "CONCERN": "No metric breached its monitoring threshold in this cycle.",
            }],
        )
    subset["__SEVRANK"] = subset["SEVERITY"].map(severity_rank)
    subset["__ABS"] = pd.to_numeric(subset["METRIC_VALUE"], errors="coerce").abs().fillna(0)
    subset = subset.sort_values(["__SEVRANK", "__ABS"], ascending=[False, False])
    # One concern per model+metric: five rows saying the same thing about five
    # columns of one model is not five concerns, and it crowds out the rest.
    subset = subset.drop_duplicates(subset=["MODEL", "METRIC"])
    if len(subset) < n:
        remainder = alerts[alerts["SEVERITY"].isin(["RED", "AMBER", "GREY"])].copy()
        remainder["__SEVRANK"] = remainder["SEVERITY"].map(severity_rank)
        remainder["__ABS"] = (
            pd.to_numeric(remainder["METRIC_VALUE"], errors="coerce").abs().fillna(0)
        )
        remainder = remainder.sort_values(["__SEVRANK", "__ABS"], ascending=[False, False])
        subset = pd.concat([subset, remainder]).drop_duplicates(
            subset=["MODEL", "METRIC", "VARIABLE_OR_SEGMENT"]
        )
    subset = subset.head(max(1, n))
    rows = []
    for rank, record in enumerate(subset.to_dict("records"), start=1):
        rows.append({
            "RANK": rank,
            "SEVERITY": record["SEVERITY"],
            "MODEL": record["MODEL"],
            "QUARTER": record["QUARTER"],
            "METRIC": record["METRIC"],
            "VARIABLE_OR_SEGMENT": record["VARIABLE_OR_SEGMENT"],
            "METRIC_VALUE": record["METRIC_VALUE"],
            "CONCERN": str(record["EXPLANATION"])[:500],
        })
    return pd.DataFrame(rows)


def priority_investigations(alerts: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Recommended priority investigations, deduplicated by model + metric."""
    columns = ["PRIORITY", "SEVERITY", "MODEL", "QUARTER", "METRIC",
               "VARIABLE_OR_SEGMENT", "RECOMMENDED_ACTION", "INVESTIGATION_STATUS"]
    if alerts is None or alerts.empty:
        return pd.DataFrame(columns=columns)
    subset = alerts[alerts["SEVERITY"].isin(["RED", "AMBER"])].copy()
    if subset.empty:
        return pd.DataFrame(
            [{
                "PRIORITY": 1, "SEVERITY": "GREEN", "MODEL": "ALL", "QUARTER": "ALL",
                "METRIC": "Routine monitoring", "VARIABLE_OR_SEGMENT": "n/a",
                "RECOMMENDED_ACTION": (
                    "No investigation required this cycle. Continue quarterly monitoring and "
                    "re-confirm the reference periods at the next model review."
                ),
                "INVESTIGATION_STATUS": STATUS_NONE,
            }],
            columns=columns,
        )
    subset["__SEVRANK"] = subset["SEVERITY"].map(severity_rank)
    subset["__ABS"] = pd.to_numeric(subset["METRIC_VALUE"], errors="coerce").abs().fillna(0)
    subset = subset.sort_values(
        ["__SEVRANK", "__ABS", "QUARTER"], ascending=[False, False, False]
    )
    subset = subset.drop_duplicates(subset=["MODEL", "METRIC", "VARIABLE_OR_SEGMENT"])

    # Round-robin across model + metric families so the list leads with breadth.
    # Eight consecutive rows about one model's characteristics is a true but
    # unhelpful answer to "what should we look at first": the second model's
    # unseen-category problem never makes the page.
    ordered: List[Dict[str, Any]] = []
    groups: List[List[Dict[str, Any]]] = [
        group.to_dict("records")
        for _, group in subset.groupby(["MODEL", "METRIC"], sort=False)
    ]
    groups.sort(key=lambda group: (-severity_rank(group[0]["SEVERITY"]), -group[0]["__ABS"]))
    while groups and len(ordered) < n:
        for group in list(groups):
            if not group:
                groups.remove(group)
                continue
            ordered.append(group.pop(0))
            if len(ordered) >= n:
                break
    subset = pd.DataFrame(ordered, columns=subset.columns) if ordered else subset.head(0)
    rows = []
    for priority, record in enumerate(subset.to_dict("records"), start=1):
        rows.append({
            "PRIORITY": priority,
            "SEVERITY": record["SEVERITY"],
            "MODEL": record["MODEL"],
            "QUARTER": record["QUARTER"],
            "METRIC": record["METRIC"],
            "VARIABLE_OR_SEGMENT": record["VARIABLE_OR_SEGMENT"],
            "RECOMMENDED_ACTION": record["RECOMMENDED_ACTION"],
            "INVESTIGATION_STATUS": record["INVESTIGATION_STATUS"],
        })
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "ALERT_COLUMNS", "build_alerts", "alert_counts", "overall_status",
    "top_concerns", "priority_investigations",
    "STATUS_OPEN", "STATUS_ESCALATED", "STATUS_NONE", "STATUS_DATA_GAP",
]
