"""Model Effectiveness Report (MER) metrics.

The MER is the volume-and-coverage backbone of the pack: for every scorecard
and every monitoring quarter it answers "how many applications were scored,
how many were not, where did they land, and how does that compare with the
baseline and with last quarter".

A deliberate separation runs through this module:

* **Model stability** -- population counts, segment mix, PSI, CSI.  Answerable
  from scorecard output alone.
* **Model predictive performance** -- bad rates, Gini, KS, rank ordering.
  Answerable only with matured outcome labels, and produced by
  :mod:`model_monitoring.performance_metrics`.

Segment output on its own supports the first and says nothing about the second,
so performance columns are merged in as ``NOT AVAILABLE`` when no outcome
exists rather than being approximated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MonitoringConfig, Severity, severity_rank
from .data_validation import QUARTER_COLUMN, SEGMENT_PREFIX, SEGMENT_RANK_PREFIX
from .logging_config import get_logger
from .psi_calculator import VS_PREVIOUS_QUARTER, VS_REFERENCE, segment_counts
from .reference_periods import ReferenceResolution
from .utils import NOT_AVAILABLE, QuarterInfo, pct, periods_overlap, round_or_none, safe_divide

logger = get_logger("mer")

MER_BASE_COLUMNS = [
    "MODEL", "QUARTER", "QUARTER_START", "QUARTER_END", "QUARTER_COMPLETE",
    "QUARTER_IN_REFERENCE_WINDOW", "REFERENCE_LABEL", "REFERENCE_MODE", "REFERENCE_RECORDS",
    "TOTAL_RECORDS", "UNIQUE_LANS", "UNIQUE_CUSTOMERS",
    "DUPLICATE_LAN_COUNT", "DUPLICATE_LAN_PCT",
    "DUPLICATE_CUSTOMER_DATE_COUNT", "RECORDS_PER_CUSTOMER",
    "SCORED_RECORDS", "MISSING_SEGMENT_COUNT", "MISSING_SEGMENT_PCT",
    "INVALID_SEGMENT_COUNT", "INVALID_SEGMENT_PCT", "MISSING_OUTPUT_PCT",
    "MOST_POPULATED_SEGMENT", "MOST_POPULATED_SEGMENT_PCT",
    "SEGMENT_CONCENTRATION_PCT", "SEGMENT_HHI", "DISTINCT_SEGMENTS_OBSERVED",
    "EMPTY_SEGMENTS", "AVERAGE_SEGMENT_RANK", "AVERAGE_SEGMENT_RANK_REF",
    "AVERAGE_SEGMENT_RANK_CHANGE", "RISKIER_SEGMENT_SHARE_PCT",
    "RISKIER_SEGMENT_SHARE_REF_PCT", "RISKIER_SEGMENT_SHARE_CHANGE_PP",
    "POPULATION_CHANGE_VS_REFERENCE_PCT", "POPULATION_CHANGE_VS_PREV_QUARTER_PCT",
    "PREVIOUS_QUARTER", "PREVIOUS_QUARTER_RECORDS",
]

SEGMENT_DISTRIBUTION_COLUMNS = [
    "MODEL", "QUARTER", "SEGMENT", "RISK_RANK", "RISK_BAND",
    "CUSTOMER_COUNT", "POPULATION_PCT", "REFERENCE_PCT", "CHANGE_PP",
    "REFERENCE_COUNT", "COUNT_CHANGE", "RELATIVE_CHANGE_PCT",
    "QUARTER_TOTAL", "REFERENCE_TOTAL", "IS_RISKIER_SEGMENT", "MATERIAL_SHIFT",
]

MODEL_SUMMARY_COLUMNS = [
    "MODEL", "DESCRIPTION", "OWNER", "VERSION", "SEGMENT_COLUMN",
    "CHARACTERISTICS_MONITORED", "REFERENCE_LABEL", "REFERENCE_MODE",
    "REFERENCE_RECORDS", "REFERENCE_SUFFICIENT",
    "QUARTERS_ANALYSED", "FIRST_QUARTER", "LATEST_QUARTER", "TOTAL_RECORDS",
    "TOTAL_UNIQUE_CUSTOMERS", "LATEST_QUARTER_RECORDS",
    "LATEST_PSI", "LATEST_PSI_SEVERITY", "MAX_PSI", "MAX_PSI_QUARTER",
    "MEAN_PSI", "LATEST_QOQ_PSI", "LATEST_QOQ_PSI_SEVERITY",
    "CSI_VARIABLES_AMBER", "CSI_VARIABLES_RED", "WORST_CSI_VARIABLE", "WORST_CSI",
    "LATEST_MISSING_OUTPUT_PCT", "LATEST_RISKIER_SHARE_PCT",
    "RISKIER_SHARE_CHANGE_PP", "LATEST_GINI", "GINI_SEVERITY",
    "OVERALL_STATUS", "HEADLINE",
]


# ---------------------------------------------------------------------------
# Quarterly MER
# ---------------------------------------------------------------------------


def compute_mer(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    resolutions: Mapping[str, ReferenceResolution],
    monitored_quarters: Sequence[QuarterInfo],
) -> pd.DataFrame:
    """One row per model-quarter with every volume / coverage / mix metric."""
    segments = config.segments
    rows: List[Dict[str, Any]] = []

    for model in config.active_models:
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        rank_column = f"{SEGMENT_RANK_PREFIX}{model.name}"
        resolution = resolutions.get(model.name)
        reference_frame = frame.loc[resolution.mask] if resolution is not None else None
        reference_stats = _segment_stats(reference_frame, config, model.name) \
            if reference_frame is not None else None

        previous: Optional[Dict[str, Any]] = None
        for quarter in monitored_quarters:
            subset = frame.loc[frame[QUARTER_COLUMN] == quarter.label]
            stats = _segment_stats(subset, config, model.name)
            total = int(len(subset))

            duplicate_lans = (
                int(subset[config.lan_column].duplicated().sum())
                if config.lan_column in subset.columns else None
            )
            duplicate_customer_date = (
                int(subset.duplicated(subset=[config.customer_column, config.date_column]).sum())
                if {config.customer_column, config.date_column}.issubset(subset.columns) else None
            )
            unique_lans = (
                int(subset[config.lan_column].nunique())
                if config.lan_column in subset.columns else None
            )
            unique_customers = (
                int(subset[config.customer_column].nunique())
                if config.customer_column in subset.columns else None
            )

            average_rank = (
                float(subset[rank_column].mean())
                if rank_column in subset.columns and subset[rank_column].notna().any() else None
            )
            reference_average_rank = (
                float(reference_frame[rank_column].mean())
                if reference_frame is not None and rank_column in reference_frame.columns
                and reference_frame[rank_column].notna().any() else None
            )

            row: Dict[str, Any] = {
                "MODEL": model.name,
                "QUARTER": quarter.label,
                "QUARTER_START": quarter.period_start.isoformat(),
                "QUARTER_END": quarter.period_end.isoformat(),
                "QUARTER_COMPLETE": quarter.is_complete,
                "QUARTER_IN_REFERENCE_WINDOW": bool(
                    model.reference is not None
                    and periods_overlap(
                        quarter.period_start, quarter.period_end,
                        model.reference.start_date, model.reference.end_date,
                    )
                ),
                "REFERENCE_LABEL": model.reference.label if model.reference else NOT_AVAILABLE,
                "REFERENCE_MODE": model.reference.mode.value if model.reference else NOT_AVAILABLE,
                "REFERENCE_RECORDS": resolution.record_count if resolution else None,
                "TOTAL_RECORDS": total,
                "UNIQUE_LANS": unique_lans,
                "UNIQUE_CUSTOMERS": unique_customers,
                "DUPLICATE_LAN_COUNT": duplicate_lans,
                "DUPLICATE_LAN_PCT": round_or_none(pct(duplicate_lans, total), 2)
                if duplicate_lans is not None else None,
                "DUPLICATE_CUSTOMER_DATE_COUNT": duplicate_customer_date,
                "RECORDS_PER_CUSTOMER": round_or_none(
                    safe_divide(total, unique_customers), 3
                ) if unique_customers else None,
                "SCORED_RECORDS": stats["scored"],
                "MISSING_SEGMENT_COUNT": stats["missing"],
                "MISSING_SEGMENT_PCT": round_or_none(pct(stats["missing"], total), 2),
                "INVALID_SEGMENT_COUNT": stats["invalid"],
                "INVALID_SEGMENT_PCT": round_or_none(pct(stats["invalid"], total), 2),
                "MISSING_OUTPUT_PCT": round_or_none(
                    pct(stats["missing"] + stats["invalid"], total), 2
                ),
                "MOST_POPULATED_SEGMENT": stats["top_segment"],
                "MOST_POPULATED_SEGMENT_PCT": round_or_none(stats["top_share_pct"], 2),
                "SEGMENT_CONCENTRATION_PCT": round_or_none(stats["top_share_pct"], 2),
                "SEGMENT_HHI": round_or_none(stats["hhi"], 4),
                "DISTINCT_SEGMENTS_OBSERVED": stats["distinct"],
                "EMPTY_SEGMENTS": ", ".join(stats["empty_segments"]),
                "AVERAGE_SEGMENT_RANK": round_or_none(average_rank, 3),
                "AVERAGE_SEGMENT_RANK_REF": round_or_none(reference_average_rank, 3),
                "AVERAGE_SEGMENT_RANK_CHANGE": round_or_none(
                    None if (average_rank is None or reference_average_rank is None)
                    else average_rank - reference_average_rank, 3
                ),
                "RISKIER_SEGMENT_SHARE_PCT": round_or_none(stats["riskier_share_pct"], 2),
                "RISKIER_SEGMENT_SHARE_REF_PCT": round_or_none(
                    reference_stats["riskier_share_pct"] if reference_stats else None, 2
                ),
                "RISKIER_SEGMENT_SHARE_CHANGE_PP": round_or_none(
                    None if (reference_stats is None or stats["riskier_share_pct"] is None
                             or reference_stats["riskier_share_pct"] is None)
                    else stats["riskier_share_pct"] - reference_stats["riskier_share_pct"], 2
                ),
                "POPULATION_CHANGE_VS_REFERENCE_PCT": round_or_none(
                    pct(total - resolution.record_count, resolution.record_count)
                    if resolution and resolution.record_count else None, 2
                ),
                "POPULATION_CHANGE_VS_PREV_QUARTER_PCT": round_or_none(
                    pct(total - previous["TOTAL_RECORDS"], previous["TOTAL_RECORDS"])
                    if previous and previous["TOTAL_RECORDS"] else None, 2
                ),
                "PREVIOUS_QUARTER": previous["QUARTER"] if previous else None,
                "PREVIOUS_QUARTER_RECORDS": previous["TOTAL_RECORDS"] if previous else None,
            }
            # per-segment share columns, always all configured segments
            for segment in segments.segment_order:
                row[f"PCT_{segment}"] = round_or_none(stats["shares_pct"].get(segment), 2)
                row[f"N_{segment}"] = int(stats["counts"].get(segment, 0))
            rows.append(row)
            previous = row

    columns = list(MER_BASE_COLUMNS)
    for segment in segments.segment_order:
        columns.extend([f"N_{segment}", f"PCT_{segment}"])
    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        table = table.sort_values(["MODEL", "QUARTER"]).reset_index(drop=True)
    logger.info("MER: %d model-quarter row(s).", len(table))
    return table


def _segment_stats(
    subset: Optional[pd.DataFrame], config: MonitoringConfig, model_name: str
) -> Dict[str, Any]:
    """Segment counts, shares, concentration and riskier-share for one population."""
    segments = config.segments
    column = f"{SEGMENT_PREFIX}{model_name}"
    empty = {
        "counts": {}, "shares_pct": {}, "scored": 0, "missing": 0, "invalid": 0,
        "top_segment": None, "top_share_pct": None, "hhi": None, "distinct": 0,
        "empty_segments": list(segments.segment_order), "riskier_share_pct": None,
    }
    if subset is None or subset.empty or column not in subset.columns:
        return empty

    values = subset[column].astype("object")
    missing = int((values == segments.missing_label).sum())
    invalid = int((values == segments.invalid_label).sum())
    counts = segment_counts(subset, model_name, segments, include_missing=False)
    scored = int(counts.sum())
    if scored == 0:
        result = dict(empty)
        result.update({"missing": missing, "invalid": invalid})
        return result

    shares = counts / scored
    top_segment = str(shares.idxmax())
    riskier = [s for s in segments.riskier_segments]
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares_pct": {str(k): float(v) * 100.0 for k, v in shares.items()},
        "scored": scored,
        "missing": missing,
        "invalid": invalid,
        "top_segment": top_segment,
        "top_share_pct": float(shares.max()) * 100.0,
        "hhi": float((shares ** 2).sum()),
        "distinct": int((counts > 0).sum()),
        "empty_segments": [str(k) for k, v in counts.items() if v == 0],
        "riskier_share_pct": float(sum(shares.get(s, 0.0) for s in riskier)) * 100.0,
    }


# ---------------------------------------------------------------------------
# Segment distribution sheet
# ---------------------------------------------------------------------------


def segment_distribution_frame(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    resolutions: Mapping[str, ReferenceResolution],
    monitored_quarters: Sequence[QuarterInfo],
    material_shift_pp: float = 5.0,
) -> pd.DataFrame:
    """Model-quarter-segment table with reference comparison."""
    segments = config.segments
    rows: List[Dict[str, Any]] = []
    risk_rank = segments.risk_rank
    riskier = set(segments.riskier_segments)
    n_segments = segments.n_segments

    for model in config.active_models:
        resolution = resolutions.get(model.name)
        if resolution is not None:
            reference_counts = segment_counts(
                frame.loc[resolution.mask], model.name, segments
            )
            reference_total = int(reference_counts.sum())
        else:
            reference_counts = None
            reference_total = 0

        for quarter in monitored_quarters:
            subset = frame.loc[frame[QUARTER_COLUMN] == quarter.label]
            counts = segment_counts(subset, model.name, segments)
            quarter_total = int(counts.sum())
            for segment in segments.segment_order:
                count = int(counts.get(segment, 0))
                share = pct(count, quarter_total)
                reference_count = (
                    int(reference_counts.get(segment, 0)) if reference_counts is not None else None
                )
                reference_share = (
                    pct(reference_count, reference_total)
                    if reference_counts is not None and reference_total else None
                )
                change = (
                    None if (share is None or reference_share is None)
                    else share - reference_share
                )
                rank = risk_rank.get(segment)
                rows.append({
                    "MODEL": model.name,
                    "QUARTER": quarter.label,
                    "SEGMENT": segment,
                    "RISK_RANK": rank,
                    "RISK_BAND": _risk_band_label(rank, n_segments),
                    "CUSTOMER_COUNT": count,
                    "POPULATION_PCT": round_or_none(share, 2),
                    "REFERENCE_PCT": round_or_none(reference_share, 2),
                    "CHANGE_PP": round_or_none(change, 2),
                    "REFERENCE_COUNT": reference_count,
                    "COUNT_CHANGE": None if reference_count is None else count - reference_count,
                    "RELATIVE_CHANGE_PCT": round_or_none(
                        pct(count - reference_count, reference_count)
                        if reference_count else None, 2
                    ),
                    "QUARTER_TOTAL": quarter_total,
                    "REFERENCE_TOTAL": reference_total or None,
                    "IS_RISKIER_SEGMENT": segment in riskier,
                    "MATERIAL_SHIFT": bool(change is not None and abs(change) >= material_shift_pp),
                })
    table = pd.DataFrame(rows, columns=SEGMENT_DISTRIBUTION_COLUMNS)
    if not table.empty:
        table = table.sort_values(["MODEL", "QUARTER", "RISK_RANK"]).reset_index(drop=True)
    return table


def _risk_band_label(rank: Optional[int], n_segments: int) -> str:
    """Plain-text risk band so the report never relies on colour alone."""
    if rank is None:
        return "UNRANKED"
    position = (rank - 1) / max(1, n_segments - 1)
    if position <= 0.2:
        return "1-LOWEST RISK"
    if position <= 0.4:
        return "2-LOW RISK"
    if position <= 0.6:
        return "3-MEDIUM RISK"
    if position <= 0.8:
        return "4-HIGH RISK"
    return "5-HIGHEST RISK"


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_mer_with_psi(mer: pd.DataFrame, psi_summary: pd.DataFrame) -> pd.DataFrame:
    """Attach fixed-reference and quarter-on-quarter PSI to the MER table."""
    if mer.empty:
        return mer
    enriched = mer.copy()
    for comparison, prefix in ((VS_REFERENCE, "PSI"), (VS_PREVIOUS_QUARTER, "PSI_QOQ")):
        if psi_summary is None or psi_summary.empty:
            enriched[prefix] = np.nan
            enriched[f"{prefix}_SEVERITY"] = Severity.GREY.value
            continue
        subset = psi_summary[psi_summary["COMPARISON"] == comparison][
            ["MODEL", "QUARTER", "PSI_DISPLAY", "SEVERITY", "TOP_CONTRIBUTING_SEGMENT"]
        ].rename(columns={
            "PSI_DISPLAY": prefix,
            "SEVERITY": f"{prefix}_SEVERITY",
            "TOP_CONTRIBUTING_SEGMENT": f"{prefix}_TOP_SEGMENT",
        })
        enriched = enriched.merge(subset, on=["MODEL", "QUARTER"], how="left")
        enriched[f"{prefix}_SEVERITY"] = enriched[f"{prefix}_SEVERITY"].fillna(Severity.GREY.value)
    return enriched


def enrich_mer_with_csi(mer: pd.DataFrame, csi_summary: pd.DataFrame,
                        thresholds) -> pd.DataFrame:
    """Attach per-model-quarter CSI counts and the worst drifting characteristic."""
    if mer.empty:
        return mer
    enriched = mer.copy()
    if csi_summary is None or csi_summary.empty:
        enriched["CSI_VARIABLES_MONITORED"] = 0
        enriched["CSI_VARIABLES_AMBER"] = 0
        enriched["CSI_VARIABLES_RED"] = 0
        enriched["WORST_CSI_VARIABLE"] = NOT_AVAILABLE
        enriched["WORST_CSI"] = np.nan
        return enriched

    subset = csi_summary[csi_summary["COMPARISON"] == VS_REFERENCE]
    grouped = subset.groupby(["MODEL", "QUARTER"], observed=True)
    aggregated = grouped.agg(
        CSI_VARIABLES_MONITORED=("VARIABLE", "nunique"),
        CSI_VARIABLES_AMBER=("SEVERITY", lambda s: int((s == "AMBER").sum())),
        CSI_VARIABLES_RED=("SEVERITY", lambda s: int((s == "RED").sum())),
        WORST_CSI=("CSI", "max"),
    ).reset_index()
    worst = (
        subset.sort_values("CSI", ascending=False)
        .drop_duplicates(subset=["MODEL", "QUARTER"])[["MODEL", "QUARTER", "VARIABLE"]]
        .rename(columns={"VARIABLE": "WORST_CSI_VARIABLE"})
    )
    aggregated = aggregated.merge(worst, on=["MODEL", "QUARTER"], how="left")
    enriched = enriched.merge(aggregated, on=["MODEL", "QUARTER"], how="left")
    for column in ("CSI_VARIABLES_MONITORED", "CSI_VARIABLES_AMBER", "CSI_VARIABLES_RED"):
        enriched[column] = enriched[column].fillna(0).astype("int64")
    enriched["WORST_CSI_VARIABLE"] = enriched["WORST_CSI_VARIABLE"].fillna(NOT_AVAILABLE)
    enriched["WORST_CSI"] = enriched["WORST_CSI"].map(lambda v: round_or_none(v, 4))
    return enriched


def enrich_mer_with_performance(
    mer: pd.DataFrame, performance: Optional[pd.DataFrame], config: MonitoringConfig
) -> pd.DataFrame:
    """Merge outcome-performance metrics, or mark them NOT AVAILABLE."""
    if mer.empty:
        return mer
    enriched = mer.copy()
    performance_columns = [
        "EVENT_RATE_PCT", "NON_EVENT_RATE_PCT", "MATURED_RECORDS", "BAD_COUNT",
        "AUC", "GINI", "KS_STATISTIC", "RANK_ORDERING", "PERFORMANCE_SEVERITY",
        "PERFORMANCE_STATUS",
    ]
    if performance is None or performance.empty:
        reason = (
            "NOT AVAILABLE - no outcome column configured"
            if not config.outcome.enabled
            else "NOT AVAILABLE - no matured outcome records"
        )
        for column in performance_columns:
            enriched[column] = np.nan if column not in {"RANK_ORDERING", "PERFORMANCE_SEVERITY",
                                                        "PERFORMANCE_STATUS"} else reason
        enriched["PERFORMANCE_SEVERITY"] = Severity.GREY.value
        enriched["PERFORMANCE_STATUS"] = reason
        return enriched
    available = [c for c in performance_columns if c in performance.columns]
    subset = performance[["MODEL", "QUARTER", *available]]
    enriched = enriched.merge(subset, on=["MODEL", "QUARTER"], how="left")
    if "PERFORMANCE_SEVERITY" in enriched.columns:
        enriched["PERFORMANCE_SEVERITY"] = enriched["PERFORMANCE_SEVERITY"].fillna(
            Severity.GREY.value
        )
    if "PERFORMANCE_STATUS" in enriched.columns:
        enriched["PERFORMANCE_STATUS"] = enriched["PERFORMANCE_STATUS"].fillna(
            "NOT AVAILABLE - quarter has no matured outcome records"
        )
    return enriched


# ---------------------------------------------------------------------------
# Model-level summary
# ---------------------------------------------------------------------------


def model_level_summary(
    mer: pd.DataFrame,
    config: MonitoringConfig,
    *,
    resolutions: Mapping[str, ReferenceResolution],
    psi_summary: Optional[pd.DataFrame] = None,
    csi_summary: Optional[pd.DataFrame] = None,
    performance: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per scorecard summarising the whole monitoring window."""
    rows: List[Dict[str, Any]] = []
    for model in config.active_models:
        model_mer = mer[mer["MODEL"] == model.name] if not mer.empty else mer
        resolution = resolutions.get(model.name)
        latest = model_mer.sort_values("QUARTER").iloc[-1].to_dict() if not model_mer.empty else {}
        first_quarter = model_mer["QUARTER"].min() if not model_mer.empty else None
        latest_quarter = model_mer["QUARTER"].max() if not model_mer.empty else None

        psi_rows = (
            psi_summary[(psi_summary["MODEL"] == model.name)
                        & (psi_summary["COMPARISON"] == VS_REFERENCE)]
            if psi_summary is not None and not psi_summary.empty else pd.DataFrame()
        )
        qoq_rows = (
            psi_summary[(psi_summary["MODEL"] == model.name)
                        & (psi_summary["COMPARISON"] == VS_PREVIOUS_QUARTER)]
            if psi_summary is not None and not psi_summary.empty else pd.DataFrame()
        )
        latest_psi = latest_psi_severity = max_psi = max_psi_quarter = mean_psi = None
        if not psi_rows.empty:
            ordered = psi_rows.sort_values("QUARTER")
            latest_psi = round_or_none(ordered["PSI"].iloc[-1], 4)
            latest_psi_severity = ordered["SEVERITY"].iloc[-1]
            if ordered["PSI"].notna().any():
                max_psi = round_or_none(ordered["PSI"].max(), 4)
                max_psi_quarter = ordered.loc[ordered["PSI"].idxmax(), "QUARTER"]
                mean_psi = round_or_none(ordered["PSI"].mean(), 4)
        latest_qoq = latest_qoq_severity = None
        if not qoq_rows.empty:
            ordered = qoq_rows.sort_values("QUARTER")
            latest_qoq = round_or_none(ordered["PSI"].iloc[-1], 4)
            latest_qoq_severity = ordered["SEVERITY"].iloc[-1]

        csi_amber = csi_red = 0
        worst_csi_variable, worst_csi = NOT_AVAILABLE, None
        if csi_summary is not None and not csi_summary.empty:
            subset = csi_summary[
                (csi_summary["MODEL"] == model.name)
                & (csi_summary["COMPARISON"] == VS_REFERENCE)
                & (csi_summary["QUARTER"] == latest_quarter)
            ]
            if not subset.empty:
                csi_amber = int((subset["SEVERITY"] == "AMBER").sum())
                csi_red = int((subset["SEVERITY"] == "RED").sum())
                if subset["CSI"].notna().any():
                    worst = subset.loc[subset["CSI"].idxmax()]
                    worst_csi_variable = str(worst["VARIABLE"])
                    worst_csi = round_or_none(worst["CSI"], 4)

        latest_gini = gini_severity = None
        if performance is not None and not performance.empty:
            subset = performance[
                (performance["MODEL"] == model.name) & (performance["QUARTER"] == latest_quarter)
            ]
            if not subset.empty:
                latest_gini = round_or_none(subset["GINI"].iloc[0], 4)
                gini_severity = subset.get("PERFORMANCE_SEVERITY", pd.Series([None])).iloc[0]

        severities = [
            latest_psi_severity, latest_qoq_severity, gini_severity,
            Severity.RED.value if csi_red else (Severity.AMBER.value if csi_amber else Severity.GREEN.value),
        ]
        overall = max(
            (s for s in severities if s is not None),
            key=severity_rank,
            default=Severity.GREY.value,
        )
        rows.append({
            "MODEL": model.name,
            "DESCRIPTION": model.description,
            "OWNER": model.owner,
            "VERSION": model.version,
            "SEGMENT_COLUMN": model.segment_column,
            "CHARACTERISTICS_MONITORED": len(config.monitored_variables(model.name)),
            "REFERENCE_LABEL": model.reference.label if model.reference else NOT_AVAILABLE,
            "REFERENCE_MODE": model.reference.mode.value if model.reference else NOT_AVAILABLE,
            "REFERENCE_RECORDS": resolution.record_count if resolution else None,
            "REFERENCE_SUFFICIENT": resolution.sufficient if resolution else False,
            "QUARTERS_ANALYSED": int(model_mer["QUARTER"].nunique()) if not model_mer.empty else 0,
            "FIRST_QUARTER": first_quarter,
            "LATEST_QUARTER": latest_quarter,
            "TOTAL_RECORDS": int(model_mer["TOTAL_RECORDS"].sum()) if not model_mer.empty else 0,
            "TOTAL_UNIQUE_CUSTOMERS": latest.get("UNIQUE_CUSTOMERS"),
            "LATEST_QUARTER_RECORDS": latest.get("TOTAL_RECORDS"),
            "LATEST_PSI": latest_psi,
            "LATEST_PSI_SEVERITY": latest_psi_severity or Severity.GREY.value,
            "MAX_PSI": max_psi,
            "MAX_PSI_QUARTER": max_psi_quarter,
            "MEAN_PSI": mean_psi,
            "LATEST_QOQ_PSI": latest_qoq,
            "LATEST_QOQ_PSI_SEVERITY": latest_qoq_severity or Severity.GREY.value,
            "CSI_VARIABLES_AMBER": csi_amber,
            "CSI_VARIABLES_RED": csi_red,
            "WORST_CSI_VARIABLE": worst_csi_variable,
            "WORST_CSI": worst_csi,
            "LATEST_MISSING_OUTPUT_PCT": latest.get("MISSING_OUTPUT_PCT"),
            "LATEST_RISKIER_SHARE_PCT": latest.get("RISKIER_SEGMENT_SHARE_PCT"),
            "RISKIER_SHARE_CHANGE_PP": latest.get("RISKIER_SEGMENT_SHARE_CHANGE_PP"),
            "LATEST_GINI": latest_gini,
            "GINI_SEVERITY": gini_severity or Severity.GREY.value,
            "OVERALL_STATUS": overall,
            "HEADLINE": _headline(
                model.name, latest_quarter, latest_psi, latest_psi_severity, latest_qoq,
                csi_red, csi_amber, worst_csi_variable, latest.get("RISKIER_SEGMENT_SHARE_CHANGE_PP"),
            ),
        })
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS)


def _headline(
    model: str,
    quarter: Optional[str],
    psi: Optional[float],
    severity: Optional[str],
    qoq: Optional[float],
    csi_red: int,
    csi_amber: int,
    worst_variable: str,
    riskier_change: Optional[float],
) -> str:
    """One sentence a credit committee can read without the rest of the pack."""
    if quarter is None:
        return f"{model}: no monitored quarter available."
    parts = [f"{model} in {quarter}:"]
    if psi is None:
        parts.append("PSI not calculable against the configured reference period.")
    else:
        label = {"GREEN": "low", "AMBER": "moderate", "RED": "significant"}.get(
            str(severity), "indeterminate"
        )
        parts.append(f"fixed-reference PSI {psi:.4f} ({label} population shift)")
        if qoq is not None:
            parts.append(f"and quarter-on-quarter PSI {qoq:.4f}")
    if csi_red or csi_amber:
        parts.append(
            f"; {csi_red} characteristic(s) above the RED CSI threshold and {csi_amber} above "
            f"AMBER, worst being {worst_variable}"
        )
    else:
        parts.append("; no characteristic breached its CSI threshold")
    if riskier_change is not None:
        direction = "toward riskier grades" if riskier_change > 0 else "toward safer grades"
        parts.append(f"; riskier-segment share moved {riskier_change:+.1f} pp {direction}")
    parts.append(
        ". A stability breach triggers investigation, not automatic model rejection."
    )
    return " ".join(parts).replace(" ;", ";").replace(" .", ".")


__all__ = [
    "MER_BASE_COLUMNS", "SEGMENT_DISTRIBUTION_COLUMNS", "MODEL_SUMMARY_COLUMNS",
    "compute_mer", "segment_distribution_frame", "enrich_mer_with_psi",
    "enrich_mer_with_csi", "enrich_mer_with_performance", "model_level_summary",
]
