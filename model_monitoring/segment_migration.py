"""Segment migration analysis.

Two comparisons are produced for every scorecard:

1. **Reference vs monitoring quarter** -- how the baseline population's grades
   compare with the current quarter's.
2. **Previous quarter vs current quarter** -- the short-horizon view that a
   fixed development baseline can hide.

Both are computed on **matched customers**: the same customer observed in both
periods.  Comparing two unmatched populations and calling the difference
"migration" is a category error -- that is a distribution shift, which the PSI
and Segment Distribution sheets already cover.  When too few customers appear
in both periods the migration view is reported as ``GREY`` with the match rate
stated, never silently replaced by a distribution comparison.

Interpretation caveat carried into the report: migration between two
independently scored snapshots can reflect changes in the applicant
population, underwriting policy, customer behaviour or the scoring
configuration itself.  It is not, on its own, evidence of model deterioration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import MonitoringConfig, MultiRecordPolicy, SegmentConfig, Severity
from .data_validation import QUARTER_COLUMN, SEGMENT_PREFIX, SEGMENT_RANK_PREFIX
from .logging_config import get_logger
from .reference_periods import ReferenceResolution
from .utils import QuarterInfo, pct, periods_overlap, round_or_none

logger = get_logger("migration")

COMPARISON_REFERENCE = "REFERENCE_VS_QUARTER"
COMPARISON_QOQ = "PREVIOUS_VS_CURRENT_QUARTER"

MIGRATION_MATRIX_COLUMNS = [
    "MODEL", "COMPARISON", "FROM_PERIOD", "TO_PERIOD", "FROM_SEGMENT", "TO_SEGMENT",
    "FROM_RISK_RANK", "TO_RISK_RANK", "RANK_CHANGE", "MOVEMENT",
    "CUSTOMER_COUNT", "ROW_PCT", "COLUMN_PCT", "TOTAL_PCT",
]

MIGRATION_SUMMARY_COLUMNS = [
    "MODEL", "COMPARISON", "FROM_PERIOD", "TO_PERIOD", "MULTI_RECORD_POLICY",
    "FROM_CUSTOMERS", "TO_CUSTOMERS", "MATCHED_CUSTOMERS", "MATCH_RATE_PCT",
    "UNCHANGED", "UPGRADES", "DOWNGRADES", "UNCHANGED_PCT", "UPGRADE_PCT",
    "DOWNGRADE_PCT", "NET_DOWNGRADE_PP", "AVERAGE_RANK_MOVEMENT",
    "AVERAGE_ABS_RANK_MOVEMENT", "MULTI_NOTCH_DOWNGRADES", "MULTI_NOTCH_DOWNGRADE_PCT",
    "LARGEST_FLOW", "LARGEST_FLOW_COUNT", "LARGEST_FLOW_PCT",
    "LARGEST_DOWNGRADE_FLOW", "LARGEST_DOWNGRADE_FLOW_COUNT",
    "SEVERITY", "CALCULABLE", "OVERLAPS_REFERENCE", "STATUS_REASON", "INTERPRETATION",
]

MIGRATION_CAVEAT = (
    "Migration between two independently scored snapshots can reflect a change in the "
    "applicant population, in underwriting or sourcing policy, in customer behaviour, or in "
    "the scoring configuration. It is not by itself evidence of model deterioration and must "
    "be read alongside PSI, CSI and (where available) outcome performance."
)


# ---------------------------------------------------------------------------
# Customer-level collapsing
# ---------------------------------------------------------------------------


def collapse_to_customer(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    model_name: str,
    *,
    policy: Optional[MultiRecordPolicy] = None,
) -> pd.DataFrame:
    """One row per customer, applying the configured multi-record policy.

    Returns a frame indexed by customer with columns ``SEGMENT`` and ``RANK``.
    Records whose segment is MISSING or INVALID are dropped first: an unscored
    application cannot take part in a grade migration.
    """
    policy = policy or config.runtime.multi_record_policy
    customer_column = config.customer_column
    segment_column = f"{SEGMENT_PREFIX}{model_name}"
    rank_column = f"{SEGMENT_RANK_PREFIX}{model_name}"

    if customer_column not in frame.columns or segment_column not in frame.columns:
        return pd.DataFrame(columns=["SEGMENT", "RANK"])

    columns = [customer_column, segment_column, rank_column]
    if config.date_column in frame.columns:
        columns.append(config.date_column)
    subset = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    subset = subset[subset[rank_column].notna()]
    if subset.empty:
        return pd.DataFrame(columns=["SEGMENT", "RANK"])

    if policy is MultiRecordPolicy.DROP:
        counts = subset[customer_column].value_counts()
        keep = counts[counts == 1].index
        subset = subset[subset[customer_column].isin(keep)]
    elif policy in (MultiRecordPolicy.LATEST, MultiRecordPolicy.EARLIEST):
        if config.date_column in subset.columns:
            ascending = policy is MultiRecordPolicy.EARLIEST
            subset = subset.sort_values(config.date_column, ascending=ascending)
        subset = subset.drop_duplicates(subset=[customer_column], keep="first")
    elif policy is MultiRecordPolicy.WORST:
        subset = subset.sort_values(rank_column, ascending=False).drop_duplicates(
            subset=[customer_column], keep="first"
        )
    elif policy is MultiRecordPolicy.BEST:
        subset = subset.sort_values(rank_column, ascending=True).drop_duplicates(
            subset=[customer_column], keep="first"
        )

    collapsed = subset.set_index(customer_column)[[segment_column, rank_column]]
    collapsed.columns = ["SEGMENT", "RANK"]
    collapsed["SEGMENT"] = collapsed["SEGMENT"].astype("object")
    return collapsed


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def _movement_label(rank_change: float) -> str:
    """Upgrade = move to a SAFER grade (lower risk rank)."""
    if rank_change < 0:
        return "UPGRADE"
    if rank_change > 0:
        return "DOWNGRADE"
    return "UNCHANGED"


def build_migration(
    from_population: pd.DataFrame,
    to_population: pd.DataFrame,
    segments: SegmentConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Matched-customer migration between two collapsed populations."""
    risk_rank = segments.risk_rank
    order = list(segments.segment_order)

    from_count = int(len(from_population))
    to_count = int(len(to_population))
    if from_population.empty or to_population.empty:
        return pd.DataFrame(), {
            "from_customers": from_count, "to_customers": to_count, "matched": 0,
            "calculable": False,
            "reason": "One of the two populations has no scored customer, so no customer can be "
                      "matched across periods.",
        }

    joined = from_population.join(
        to_population, how="inner", lsuffix="_FROM", rsuffix="_TO"
    )
    matched = int(len(joined))
    if matched == 0:
        return pd.DataFrame(), {
            "from_customers": from_count, "to_customers": to_count, "matched": 0,
            "calculable": False,
            "reason": (
                "No customer appears in both periods, so customer-level migration cannot be "
                "measured. The population-level shift is covered by PSI and the Segment "
                "Distribution sheet."
            ),
        }

    joined["RANK_CHANGE"] = joined["RANK_TO"] - joined["RANK_FROM"]
    grid = pd.MultiIndex.from_product([order, order], names=["FROM_SEGMENT", "TO_SEGMENT"])
    counts = (
        joined.groupby(["SEGMENT_FROM", "SEGMENT_TO"], observed=True)
        .size()
        .rename("CUSTOMER_COUNT")
    )
    counts.index = counts.index.set_names(["FROM_SEGMENT", "TO_SEGMENT"])
    matrix = counts.reindex(grid, fill_value=0).reset_index()

    row_totals = matrix.groupby("FROM_SEGMENT", observed=True)["CUSTOMER_COUNT"].transform("sum")
    column_totals = matrix.groupby("TO_SEGMENT", observed=True)["CUSTOMER_COUNT"].transform("sum")
    matrix["FROM_RISK_RANK"] = matrix["FROM_SEGMENT"].map(risk_rank)
    matrix["TO_RISK_RANK"] = matrix["TO_SEGMENT"].map(risk_rank)
    matrix["RANK_CHANGE"] = matrix["TO_RISK_RANK"] - matrix["FROM_RISK_RANK"]
    matrix["MOVEMENT"] = matrix["RANK_CHANGE"].map(_movement_label)
    matrix["ROW_PCT"] = np.where(
        row_totals > 0, matrix["CUSTOMER_COUNT"] / row_totals * 100.0, np.nan
    )
    matrix["COLUMN_PCT"] = np.where(
        column_totals > 0, matrix["CUSTOMER_COUNT"] / column_totals * 100.0, np.nan
    )
    matrix["TOTAL_PCT"] = matrix["CUSTOMER_COUNT"] / matched * 100.0

    upgrades = int((joined["RANK_CHANGE"] < 0).sum())
    downgrades = int((joined["RANK_CHANGE"] > 0).sum())
    unchanged = int((joined["RANK_CHANGE"] == 0).sum())
    multi_notch = int((joined["RANK_CHANGE"] >= 2).sum())

    flows = matrix[matrix["CUSTOMER_COUNT"] > 0].sort_values("CUSTOMER_COUNT", ascending=False)
    largest = flows.iloc[0] if not flows.empty else None
    downgrade_flows = flows[flows["MOVEMENT"] == "DOWNGRADE"]
    largest_downgrade = downgrade_flows.iloc[0] if not downgrade_flows.empty else None

    stats = {
        "from_customers": from_count,
        "to_customers": to_count,
        "matched": matched,
        "match_rate_pct": pct(matched, min(from_count, to_count)),
        "upgrades": upgrades,
        "downgrades": downgrades,
        "unchanged": unchanged,
        "upgrade_pct": pct(upgrades, matched),
        "downgrade_pct": pct(downgrades, matched),
        "unchanged_pct": pct(unchanged, matched),
        "net_downgrade_pp": (pct(downgrades, matched) or 0.0) - (pct(upgrades, matched) or 0.0),
        "average_rank_movement": float(joined["RANK_CHANGE"].mean()),
        "average_abs_rank_movement": float(joined["RANK_CHANGE"].abs().mean()),
        "multi_notch_downgrades": multi_notch,
        "multi_notch_downgrade_pct": pct(multi_notch, matched),
        "largest_flow": (
            f"{largest['FROM_SEGMENT']} -> {largest['TO_SEGMENT']}" if largest is not None else None
        ),
        "largest_flow_count": int(largest["CUSTOMER_COUNT"]) if largest is not None else None,
        "largest_flow_pct": float(largest["TOTAL_PCT"]) if largest is not None else None,
        "largest_downgrade_flow": (
            f"{largest_downgrade['FROM_SEGMENT']} -> {largest_downgrade['TO_SEGMENT']}"
            if largest_downgrade is not None else None
        ),
        "largest_downgrade_flow_count": (
            int(largest_downgrade["CUSTOMER_COUNT"]) if largest_downgrade is not None else None
        ),
        "calculable": True,
        "reason": "",
    }
    return matrix, stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compute_segment_migration(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    resolutions: Mapping[str, ReferenceResolution],
    monitored_quarters: Sequence[QuarterInfo],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(migration_matrix, migration_summary)`` for every scorecard."""
    segments = config.segments
    thresholds = config.thresholds
    policy = config.runtime.multi_record_policy
    matrix_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    if config.customer_column not in frame.columns:
        logger.warning(
            "Customer column '%s' is absent; customer-level migration cannot be produced.",
            config.customer_column,
        )
        return (
            pd.DataFrame(columns=MIGRATION_MATRIX_COLUMNS),
            pd.DataFrame(columns=MIGRATION_SUMMARY_COLUMNS),
        )

    for model in config.active_models:
        resolution = resolutions.get(model.name)
        reference_population = (
            collapse_to_customer(frame.loc[resolution.mask], config, model.name, policy=policy)
            if resolution is not None else pd.DataFrame(columns=["SEGMENT", "RANK"])
        )
        reference_label = model.reference.label if model.reference else "NOT CONFIGURED"

        quarter_populations: Dict[str, pd.DataFrame] = {}
        for quarter in monitored_quarters:
            quarter_populations[quarter.label] = collapse_to_customer(
                frame.loc[frame[QUARTER_COLUMN] == quarter.label], config, model.name,
                policy=policy,
            )

        labels = [q.label for q in monitored_quarters]
        for position, label in enumerate(labels):
            current = quarter_populations[label]

            # -- reference vs quarter -------------------------------------
            overlaps = bool(
                model.reference is not None
                and periods_overlap(
                    monitored_quarters[position].period_start,
                    monitored_quarters[position].period_end,
                    model.reference.start_date, model.reference.end_date,
                )
            )
            matrix, stats = build_migration(reference_population, current, segments)
            summary_rows.append(_summary_row(
                model.name, COMPARISON_REFERENCE, reference_label, label, policy,
                stats, thresholds, segments, overlaps_reference=overlaps,
            ))
            if not matrix.empty:
                matrix.insert(0, "TO_PERIOD", label)
                matrix.insert(0, "FROM_PERIOD", reference_label)
                matrix.insert(0, "COMPARISON", COMPARISON_REFERENCE)
                matrix.insert(0, "MODEL", model.name)
                matrix_rows.append(matrix)

            # -- previous quarter vs current ------------------------------
            if position == 0:
                continue
            previous_label = labels[position - 1]
            matrix, stats = build_migration(
                quarter_populations[previous_label], current, segments
            )
            summary_rows.append(_summary_row(
                model.name, COMPARISON_QOQ, previous_label, label, policy,
                stats, thresholds, segments,
            ))
            if not matrix.empty:
                matrix.insert(0, "TO_PERIOD", label)
                matrix.insert(0, "FROM_PERIOD", previous_label)
                matrix.insert(0, "COMPARISON", COMPARISON_QOQ)
                matrix.insert(0, "MODEL", model.name)
                matrix_rows.append(matrix)

    matrix_frame = (
        pd.concat(matrix_rows, ignore_index=True)[MIGRATION_MATRIX_COLUMNS]
        if matrix_rows else pd.DataFrame(columns=MIGRATION_MATRIX_COLUMNS)
    )
    summary_frame = pd.DataFrame(summary_rows, columns=MIGRATION_SUMMARY_COLUMNS)
    logger.info(
        "Migration: %d summary row(s), %d matrix cell(s).", len(summary_frame), len(matrix_frame)
    )
    return matrix_frame, summary_frame


def _summary_row(
    model_name: str,
    comparison: str,
    from_period: str,
    to_period: str,
    policy: MultiRecordPolicy,
    stats: Dict[str, Any],
    thresholds,
    segments: SegmentConfig,
    *,
    overlaps_reference: bool = False,
) -> Dict[str, Any]:
    matched = stats.get("matched", 0)
    calculable = bool(stats.get("calculable"))
    net = stats.get("net_downgrade_pp")
    if overlaps_reference:
        stats = dict(stats)
        stats["reason"] = (
            (stats.get("reason") or "")
            + f" {to_period} overlaps the reference window, so most matched customers are the "
              "same records on both sides of the comparison. Migration measured here reflects "
              "repeat applications within the baseline, not change against it."
        ).strip()

    if not calculable:
        severity = Severity.GREY
    elif matched < thresholds.min_monitoring_records:
        severity = Severity.GREY
        stats["reason"] = (
            f"Only {matched:,} matched customer(s), below the configured minimum of "
            f"{thresholds.min_monitoring_records:,}; the migration rates are not reliable."
        )
    else:
        # Only movement TOWARD risk raises an alert; a net upgrade is reported but GREEN.
        severity = thresholds.net_downgrade_pp.classify(net if net and net > 0 else 0.0)

    return {
        "MODEL": model_name,
        "COMPARISON": comparison,
        "FROM_PERIOD": from_period,
        "TO_PERIOD": to_period,
        "MULTI_RECORD_POLICY": policy.value,
        "FROM_CUSTOMERS": stats.get("from_customers"),
        "TO_CUSTOMERS": stats.get("to_customers"),
        "MATCHED_CUSTOMERS": matched,
        "MATCH_RATE_PCT": round_or_none(stats.get("match_rate_pct"), 2),
        "UNCHANGED": stats.get("unchanged"),
        "UPGRADES": stats.get("upgrades"),
        "DOWNGRADES": stats.get("downgrades"),
        "UNCHANGED_PCT": round_or_none(stats.get("unchanged_pct"), 2),
        "UPGRADE_PCT": round_or_none(stats.get("upgrade_pct"), 2),
        "DOWNGRADE_PCT": round_or_none(stats.get("downgrade_pct"), 2),
        "NET_DOWNGRADE_PP": round_or_none(net, 2),
        "AVERAGE_RANK_MOVEMENT": round_or_none(stats.get("average_rank_movement"), 3),
        "AVERAGE_ABS_RANK_MOVEMENT": round_or_none(stats.get("average_abs_rank_movement"), 3),
        "MULTI_NOTCH_DOWNGRADES": stats.get("multi_notch_downgrades"),
        "MULTI_NOTCH_DOWNGRADE_PCT": round_or_none(stats.get("multi_notch_downgrade_pct"), 2),
        "LARGEST_FLOW": stats.get("largest_flow"),
        "LARGEST_FLOW_COUNT": stats.get("largest_flow_count"),
        "LARGEST_FLOW_PCT": round_or_none(stats.get("largest_flow_pct"), 2),
        "LARGEST_DOWNGRADE_FLOW": stats.get("largest_downgrade_flow"),
        "LARGEST_DOWNGRADE_FLOW_COUNT": stats.get("largest_downgrade_flow_count"),
        "SEVERITY": Severity.GREY.value if overlaps_reference else severity.value,
        "CALCULABLE": calculable,
        "OVERLAPS_REFERENCE": overlaps_reference,
        "STATUS_REASON": stats.get("reason", ""),
        "INTERPRETATION": _interpretation(stats, severity, segments),
    }


def _interpretation(stats: Dict[str, Any], severity: Severity, segments: SegmentConfig) -> str:
    if not stats.get("calculable"):
        return stats.get("reason", "Migration could not be measured.") + " " + MIGRATION_CAVEAT
    net = stats.get("net_downgrade_pp") or 0.0
    average = stats.get("average_rank_movement") or 0.0
    direction = (
        "toward riskier grades" if net > 0 else
        ("toward safer grades" if net < 0 else "with no net directional movement")
    )
    lead = (
        f"{stats.get('matched', 0):,} matched customer(s): "
        f"{stats.get('unchanged_pct') or 0:.1f}% unchanged, "
        f"{stats.get('upgrade_pct') or 0:.1f}% upgraded, "
        f"{stats.get('downgrade_pct') or 0:.1f}% downgraded "
        f"(net {net:+.1f} pp {direction}; average movement "
        f"{average:+.2f} notches on a 1-{segments.n_segments} risk scale)."
    )
    if severity is Severity.RED:
        lead += (
            " This is a material deterioration in the matched book and should be escalated to "
            "the model owner and the credit policy team."
        )
    elif severity is Severity.AMBER:
        lead += " Investigate the drivers before the next monitoring cycle."
    elif severity is Severity.GREY:
        lead += " " + (stats.get("reason") or "")
    return lead + " " + MIGRATION_CAVEAT


# ---------------------------------------------------------------------------
# Pivot helper for the Excel heatmap
# ---------------------------------------------------------------------------


def migration_pivot(
    matrix: pd.DataFrame,
    model: str,
    comparison: str,
    to_period: str,
    segments: SegmentConfig,
    *,
    value: str = "ROW_PCT",
) -> pd.DataFrame:
    """Square ``from x to`` matrix ready to be written as a heatmap."""
    order = list(segments.segment_order)
    empty = pd.DataFrame(0.0, index=order, columns=order)
    if matrix.empty:
        return empty
    subset = matrix[
        (matrix["MODEL"] == model)
        & (matrix["COMPARISON"] == comparison)
        & (matrix["TO_PERIOD"] == to_period)
    ]
    if subset.empty:
        return empty
    pivot = subset.pivot_table(
        index="FROM_SEGMENT", columns="TO_SEGMENT", values=value, aggfunc="sum"
    )
    return pivot.reindex(index=order, columns=order).fillna(0.0)


__all__ = [
    "COMPARISON_REFERENCE", "COMPARISON_QOQ", "MIGRATION_MATRIX_COLUMNS",
    "MIGRATION_SUMMARY_COLUMNS", "MIGRATION_CAVEAT", "collapse_to_customer",
    "build_migration", "compute_segment_migration", "migration_pivot",
]
