"""Outcome-based predictive performance monitoring.

This module is strictly separate from the stability modules, because the two
answer different questions:

* PSI / CSI / migration  -> *is the population changing?*
* this module            -> *does the scorecard still separate good from bad?*

Segment output alone can only answer the first.  When no outcome label is
supplied every metric here is reported as ``NOT AVAILABLE`` together with the
precise missing dependency; nothing is approximated from the segment mix.

Right-censoring
---------------
A loan booked two months ago with no default yet is **not** a good customer --
it is an unknown.  Counting it as good understates the bad rate and flatters
every discrimination metric.  Accounts whose performance window has not elapsed
are therefore excluded by default (``outcome.exclude_immature``), the exclusion
is counted, and any quarter whose cohort is not fully matured is labelled
incomplete so the reader knows the bad rate will still rise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import MonitoringConfig, Severity
from .data_validation import (
    MATURITY_COLUMN,
    OUTCOME_COLUMN,
    QUARTER_COLUMN,
    SEGMENT_PREFIX,
    SEGMENT_RANK_PREFIX,
)
from .logging_config import get_logger
from .utils import NOT_AVAILABLE, QuarterInfo, pct, round_or_none, safe_divide

logger = get_logger("performance")

try:  # scikit-learn is optional; a NumPy fallback keeps the pipeline working
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - environment dependent
    _sk_roc_auc_score = None
    _HAVE_SKLEARN = False


PERFORMANCE_SUMMARY_COLUMNS = [
    "MODEL", "QUARTER", "SCORE_SOURCE", "SCORE_DIRECTION", "OBSERVATION_WINDOW_MONTHS",
    "TOTAL_RECORDS", "RECORDS_WITH_OUTCOME", "MATURED_RECORDS", "IMMATURE_EXCLUDED",
    "MATURITY_PCT", "COHORT_COMPLETE", "BAD_COUNT", "GOOD_COUNT",
    "EVENT_RATE_PCT", "NON_EVENT_RATE_PCT", "AUC", "GINI", "KS_STATISTIC",
    "KS_SEGMENT", "LIFT_TOP_SEGMENT", "BAD_CAPTURE_TOP_2_SEGMENTS_PCT",
    "RANK_ORDERING", "RANK_ORDERING_INVERSIONS", "SPEARMAN_RANK_CORRELATION",
    "EXPECTED_BAD_RATE_PCT", "OBSERVED_VS_EXPECTED_RATIO", "CALIBRATION_Z",
    "CALIBRATION_VERDICT", "PERFORMANCE_SEVERITY", "PERFORMANCE_STATUS",
]

PERFORMANCE_SEGMENT_COLUMNS = [
    "MODEL", "QUARTER", "SEGMENT", "RISK_RANK", "MATURED_RECORDS", "BAD_COUNT",
    "BAD_RATE_PCT", "SHARE_OF_POPULATION_PCT", "SHARE_OF_BADS_PCT", "LIFT",
    "CUMULATIVE_POPULATION_PCT", "CUMULATIVE_BAD_CAPTURE_PCT",
    "EXPECTED_PD_PCT", "OBSERVED_VS_EXPECTED_RATIO", "RANK_ORDER_BREAK",
]

VINTAGE_COLUMNS = [
    "MODEL", "VINTAGE_QUARTER", "SEGMENT", "ACCOUNTS", "MATURED_ACCOUNTS",
    "MATURITY_PCT", "BAD_COUNT", "BAD_RATE_PCT", "COHORT_COMPLETE", "NOTE",
]

AVAILABILITY_COLUMNS = ["METRIC", "STATUS", "MISSING_DEPENDENCY", "EXPLANATION"]


@dataclass
class PerformanceResult:
    """All outcome-based tables, plus an availability explanation."""

    summary: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=PERFORMANCE_SUMMARY_COLUMNS)
    )
    by_segment: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=PERFORMANCE_SEGMENT_COLUMNS)
    )
    vintage: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=VINTAGE_COLUMNS))
    availability: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    )
    enabled: bool = False

    @property
    def available(self) -> bool:
        return self.enabled and not self.summary.empty


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """AUC where a HIGHER ``score`` means HIGHER risk.

    Uses scikit-learn when available and falls back to the rank-based
    Mann-Whitney identity otherwise, so the pipeline never hard-depends on it.
    """
    labels = np.asarray(labels, dtype="float64")
    scores = np.asarray(scores, dtype="float64")
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    if labels.size == 0:
        return None
    positives = labels == 1
    negatives = labels == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return None
    if _HAVE_SKLEARN:
        try:
            return float(_sk_roc_auc_score(labels, scores))
        except Exception as exc:  # pragma: no cover
            logger.debug("sklearn AUC failed (%s); using the NumPy fallback.", exc)
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = float(positives.sum())
    n_neg = float(negatives.sum())
    rank_sum = float(ranks[positives].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_ks(labels: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """KS statistic and the score at which it is attained.

    KS is the maximum vertical distance between the cumulative bad and
    cumulative good distributions, ordered by increasing risk score.
    """
    labels = np.asarray(labels, dtype="float64")
    scores = np.asarray(scores, dtype="float64")
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    if labels.size == 0:
        return None, None
    n_bad = float((labels == 1).sum())
    n_good = float((labels == 0).sum())
    if n_bad == 0 or n_good == 0:
        return None, None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_bad = np.cumsum(sorted_labels == 1) / n_bad
    cumulative_good = np.cumsum(sorted_labels == 0) / n_good
    difference = np.abs(cumulative_bad - cumulative_good)
    # Evaluate only at the last record of each tied score block.
    boundaries = np.r_[np.diff(sorted_scores) != 0, True]
    if not boundaries.any():
        return 0.0, None
    index = int(np.argmax(np.where(boundaries, difference, -np.inf)))
    return float(difference[index]), float(sorted_scores[index])


def assess_rank_ordering(
    bad_rates: Sequence[Optional[float]], risk_ranks: Sequence[float]
) -> Tuple[str, int, Optional[float]]:
    """Does the bad rate rise monotonically with the configured risk rank?

    Returns ``(verdict, inversions, spearman)``.  Segments with no matured
    records are skipped rather than treated as a zero bad rate.
    """
    pairs = [
        (float(rank), float(rate))
        for rank, rate in zip(risk_ranks, bad_rates)
        if rate is not None and rate == rate
    ]
    if len(pairs) < 3:
        return f"{NOT_AVAILABLE} - fewer than 3 segments with matured outcomes", 0, None
    pairs.sort(key=lambda item: item[0])
    rates = [rate for _, rate in pairs]
    inversions = sum(1 for a, b in zip(rates, rates[1:]) if b < a)
    ranks_series = pd.Series([rank for rank, _ in pairs])
    rates_series = pd.Series(rates)
    spearman = ranks_series.corr(rates_series, method="spearman")
    spearman = None if spearman is None or spearman != spearman else float(spearman)
    if inversions == 0:
        verdict = "MONOTONIC - bad rate increases with risk grade as designed"
    elif inversions == 1:
        verdict = "MINOR BREAK - one adjacent pair of grades is inverted"
    else:
        verdict = f"BROKEN - {inversions} adjacent inversions in the bad-rate ordering"
    return verdict, inversions, round_or_none(spearman, 4)


# ---------------------------------------------------------------------------
# Availability explanation
# ---------------------------------------------------------------------------


def availability_frame(config: MonitoringConfig, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Explain exactly which fields each unavailable metric needs."""
    outcome = config.outcome
    has_outcome = outcome.enabled and (
        frame is None or (OUTCOME_COLUMN in frame.columns and frame[OUTCOME_COLUMN].notna().any())
    )
    has_pd = bool(outcome.pd_column) and (
        frame is None or outcome.pd_column in frame.columns
    )
    has_score = bool(outcome.score_column or any(m.score_column for m in config.active_models))

    def row(metric: str, ok: bool, dependency: str, explanation: str) -> Dict[str, Any]:
        return {
            "METRIC": metric,
            "STATUS": "AVAILABLE" if ok else NOT_AVAILABLE,
            "MISSING_DEPENDENCY": "" if ok else dependency,
            "EXPLANATION": explanation,
        }

    rows = [
        row("Event rate / non-event rate", has_outcome, "outcome.outcome_column",
            "Requires a binary good/bad (or default/delinquency) flag per application."),
        row("Bad rate by segment", has_outcome, "outcome.outcome_column",
            "Requires the outcome flag; computed on matured accounts only."),
        row("Rank ordering", has_outcome, "outcome.outcome_column",
            "Requires the outcome flag and at least three segments with matured records."),
        row("AUC / Gini", has_outcome, "outcome.outcome_column",
            "Requires the outcome flag. A continuous score gives a sharper estimate; with "
            "segments only, the ordinal risk rank is used and the AUC is a coarse lower bound."),
        row("KS statistic", has_outcome, "outcome.outcome_column",
            "Requires the outcome flag plus a risk ordering (score or segment rank)."),
        row("Lift / cumulative bad capture", has_outcome, "outcome.outcome_column",
            "Requires the outcome flag; computed over segments ordered riskiest first."),
        row("Calibration (observed vs expected PD)", has_outcome and has_pd,
            "outcome.outcome_column + outcome.pd_column",
            "Requires a model-estimated probability of default per application. A risk grade "
            "is not a probability and cannot be substituted for one."),
        row("Vintage analysis", has_outcome, "outcome.outcome_column + origination date",
            "Requires the outcome flag and an origination date (outcome.vintage_date_column, "
            "defaulting to the login date)."),
        row("Continuous score distribution (score CSI)", has_score,
            "models[].score_column",
            "Requires the final scorecard score. The categorical segment is monitored "
            "separately and is not a substitute for the score."),
    ]
    return pd.DataFrame(rows, columns=AVAILABILITY_COLUMNS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_performance(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    monitored_quarters: Sequence[QuarterInfo],
) -> PerformanceResult:
    """Outcome performance per model and quarter, or a clear unavailability note."""
    outcome_cfg = config.outcome
    availability = availability_frame(config, frame)

    if not outcome_cfg.enabled or OUTCOME_COLUMN not in frame.columns:
        logger.info(
            "Outcome performance monitoring is NOT AVAILABLE: no outcome column is configured. "
            "Stability monitoring (PSI/CSI/migration) is unaffected."
        )
        return PerformanceResult(availability=availability, enabled=False)

    if not frame[OUTCOME_COLUMN].notna().any():
        logger.warning(
            "The configured outcome column contains no usable good/bad values; performance "
            "metrics are reported as NOT AVAILABLE."
        )
        return PerformanceResult(availability=availability, enabled=False)

    summary_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []

    for model in config.active_models:
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        rank_column = f"{SEGMENT_RANK_PREFIX}{model.name}"
        score_column, score_source, higher_is_riskier = _resolve_score(config, model, frame)

        for quarter in monitored_quarters:
            subset = frame.loc[frame[QUARTER_COLUMN] == quarter.label]
            summary_row, rows = _quarter_performance(
                subset, config, model, quarter,
                segment_column=segment_column,
                rank_column=rank_column,
                score_column=score_column,
                score_source=score_source,
                higher_is_riskier=higher_is_riskier,
            )
            summary_rows.append(summary_row)
            segment_rows.extend(rows)

    summary = pd.DataFrame(summary_rows, columns=PERFORMANCE_SUMMARY_COLUMNS)
    by_segment = pd.DataFrame(segment_rows, columns=PERFORMANCE_SEGMENT_COLUMNS)
    vintage = _vintage_analysis(frame, config)
    logger.info(
        "Performance: %d model-quarter row(s); %d with enough matured outcomes for Gini.",
        len(summary),
        int(summary["GINI"].notna().sum()) if not summary.empty else 0,
    )
    return PerformanceResult(
        summary=summary, by_segment=by_segment, vintage=vintage,
        availability=availability, enabled=True,
    )


def _resolve_score(
    config: MonitoringConfig, model, frame: pd.DataFrame
) -> Tuple[Optional[str], str, bool]:
    """Pick the risk ordering: explicit score, model score, or segment rank."""
    outcome_cfg = config.outcome
    for candidate, source in (
        (outcome_cfg.score_column, "outcome.score_column"),
        (model.score_column, f"{model.name}.score_column"),
    ):
        if candidate and candidate in frame.columns:
            return candidate, f"{candidate} ({source})", outcome_cfg.score_higher_is_riskier
    return (
        None,
        "ordinal segment rank (no continuous score available - coarse proxy)",
        True,
    )


def _quarter_performance(
    subset: pd.DataFrame,
    config: MonitoringConfig,
    model,
    quarter: QuarterInfo,
    *,
    segment_column: str,
    rank_column: str,
    score_column: Optional[str],
    score_source: str,
    higher_is_riskier: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    thresholds = config.thresholds
    outcome_cfg = config.outcome
    segments = config.segments
    total = int(len(subset))

    with_outcome = subset[subset[OUTCOME_COLUMN].notna()]
    if outcome_cfg.exclude_immature and MATURITY_COLUMN in subset.columns:
        matured = with_outcome[with_outcome[MATURITY_COLUMN] == True]  # noqa: E712
    else:
        matured = with_outcome
    immature_excluded = int(len(with_outcome) - len(matured))
    maturity = pct(len(matured), total)
    cohort_complete = bool(maturity is not None and maturity >= 99.0)

    row: Dict[str, Any] = {column: None for column in PERFORMANCE_SUMMARY_COLUMNS}
    row.update({
        "MODEL": model.name,
        "QUARTER": quarter.label,
        "SCORE_SOURCE": score_source,
        "SCORE_DIRECTION": "HIGHER = RISKIER" if higher_is_riskier else "HIGHER = SAFER",
        "OBSERVATION_WINDOW_MONTHS": outcome_cfg.observation_window_months,
        "TOTAL_RECORDS": total,
        "RECORDS_WITH_OUTCOME": int(len(with_outcome)),
        "MATURED_RECORDS": int(len(matured)),
        "IMMATURE_EXCLUDED": immature_excluded,
        "MATURITY_PCT": round_or_none(maturity, 2),
        "COHORT_COMPLETE": cohort_complete,
        "RANK_ORDERING": f"{NOT_AVAILABLE} - insufficient matured outcomes",
        "CALIBRATION_VERDICT": f"{NOT_AVAILABLE} - no expected PD supplied",
        "PERFORMANCE_SEVERITY": Severity.GREY.value,
    })

    if matured.empty:
        row["PERFORMANCE_STATUS"] = (
            f"{NOT_AVAILABLE} - no matured outcome record in {quarter.label}. With a "
            f"{outcome_cfg.observation_window_months}-month performance window, applications "
            "from this quarter cannot yet be classified good or bad. They are NOT counted as "
            "good."
        )
        return row, []

    labels = matured[OUTCOME_COLUMN].to_numpy(dtype="float64")
    bad_count = int(np.nansum(labels == 1))
    good_count = int(np.nansum(labels == 0))
    row["BAD_COUNT"] = bad_count
    row["GOOD_COUNT"] = good_count
    row["EVENT_RATE_PCT"] = round_or_none(pct(bad_count, len(matured)), 3)
    row["NON_EVENT_RATE_PCT"] = round_or_none(pct(good_count, len(matured)), 3)

    # -- risk ordering -----------------------------------------------------
    if score_column is not None and score_column in matured.columns:
        raw = pd.to_numeric(matured[score_column], errors="coerce").to_numpy(dtype="float64")
        risk_scores = raw if higher_is_riskier else -raw
    else:
        risk_scores = matured[rank_column].to_numpy(dtype="float64")

    enough = len(matured) >= thresholds.min_outcome_records and bad_count >= thresholds.min_bad_count
    auc = compute_auc(labels, risk_scores)
    ks, ks_at = compute_ks(labels, risk_scores)
    gini = None if auc is None else 2.0 * auc - 1.0
    row["AUC"] = round_or_none(auc, 4)
    row["GINI"] = round_or_none(gini, 4)
    row["KS_STATISTIC"] = round_or_none(ks, 4)
    row["KS_SEGMENT"] = (
        None if ks_at is None else
        (_segment_for_rank(ks_at, segments) if score_column is None else round_or_none(ks_at, 4))
    )

    # -- per segment -------------------------------------------------------
    segment_rows, lift_top, capture_top2, bad_rates, ranks, breaks = _segment_performance(
        matured, config, model, quarter, segment_column, rank_column
    )
    row["LIFT_TOP_SEGMENT"] = round_or_none(lift_top, 3)
    row["BAD_CAPTURE_TOP_2_SEGMENTS_PCT"] = round_or_none(capture_top2, 2)
    verdict, inversions, spearman = assess_rank_ordering(bad_rates, ranks)
    row["RANK_ORDERING"] = verdict
    row["RANK_ORDERING_INVERSIONS"] = inversions
    row["SPEARMAN_RANK_CORRELATION"] = spearman

    # -- calibration -------------------------------------------------------
    pd_column = outcome_cfg.pd_column or model.pd_column
    if pd_column and pd_column in matured.columns:
        expected = pd.to_numeric(matured[pd_column], errors="coerce")
        if expected.notna().any():
            expected_values = expected.to_numpy(dtype="float64")
            if np.nanmax(expected_values) > 1.0:   # supplied as a percentage
                expected_values = expected_values / 100.0
            expected_rate = float(np.nanmean(expected_values))
            observed_rate = bad_count / len(matured)
            row["EXPECTED_BAD_RATE_PCT"] = round_or_none(expected_rate * 100.0, 3)
            row["OBSERVED_VS_EXPECTED_RATIO"] = round_or_none(
                safe_divide(observed_rate, expected_rate), 3
            )
            expected_bads = expected_rate * len(matured)
            variance = float(np.nansum(expected_values * (1.0 - expected_values)))
            z_score = (
                (bad_count - expected_bads) / np.sqrt(variance) if variance > 0 else None
            )
            row["CALIBRATION_Z"] = round_or_none(z_score, 3)
            if z_score is None:
                row["CALIBRATION_VERDICT"] = f"{NOT_AVAILABLE} - degenerate expected PD variance"
            elif abs(z_score) <= 1.96:
                row["CALIBRATION_VERDICT"] = (
                    "WITHIN TOLERANCE - observed bads are consistent with the expected PD "
                    "at the 95% level"
                )
            elif z_score > 0:
                row["CALIBRATION_VERDICT"] = (
                    "UNDER-PREDICTING - observed bads exceed the expected PD; the model is "
                    "optimistic and the calibration should be reviewed"
                )
            else:
                row["CALIBRATION_VERDICT"] = (
                    "OVER-PREDICTING - observed bads are below the expected PD; the model is "
                    "conservative"
                )

    # -- severity / status -------------------------------------------------
    if not enough:
        row["PERFORMANCE_SEVERITY"] = Severity.GREY.value
        row["PERFORMANCE_STATUS"] = (
            f"Metrics computed on {len(matured):,} matured record(s) with {bad_count:,} bad(s), "
            f"below the configured minimums (min_outcome_records="
            f"{thresholds.min_outcome_records:,}, min_bad_count={thresholds.min_bad_count:,}). "
            "Treat the discrimination statistics as indicative only."
        )
    else:
        severity = thresholds.classify_gini(gini)
        row["PERFORMANCE_SEVERITY"] = severity.value
        parts = [
            f"Computed on {len(matured):,} matured record(s) "
            f"({row['EVENT_RATE_PCT']}% event rate)."
        ]
        if gini is not None:
            parts.append(f"Gini {gini:.4f} against a floor of {thresholds.min_gini_amber:g}.")
        if inversions:
            parts.append(f"Rank ordering shows {inversions} inversion(s).")
        if not cohort_complete:
            parts.append(
                f"Only {maturity:.1f}% of the quarter has a matured performance window, so the "
                "bad rate will continue to rise as the cohort seasons."
            )
        if score_column is None:
            parts.append(
                "No continuous score was available, so the ordinal segment rank was used; with "
                f"{segments.n_segments} ties-heavy grades the AUC is a conservative lower bound."
            )
        row["PERFORMANCE_STATUS"] = " ".join(parts)
    if breaks:
        row["PERFORMANCE_STATUS"] = (row["PERFORMANCE_STATUS"] or "") + (
            f" Rank-order break(s) at: {', '.join(breaks)}."
        )
    return row, segment_rows


def _segment_for_rank(rank_value: float, segments) -> Optional[str]:
    for segment, rank in segments.risk_rank.items():
        if abs(rank - rank_value) < 1e-9:
            return segment
    return None


def _segment_performance(
    matured: pd.DataFrame,
    config: MonitoringConfig,
    model,
    quarter: QuarterInfo,
    segment_column: str,
    rank_column: str,
) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float], List[Optional[float]],
           List[float], List[str]]:
    """Bad rate, lift and cumulative bad capture by segment (riskiest first)."""
    segments = config.segments
    outcome_cfg = config.outcome
    pd_column = outcome_cfg.pd_column or model.pd_column
    total = int(len(matured))
    total_bads = int((matured[OUTCOME_COLUMN] == 1).sum())
    overall_rate = safe_divide(total_bads, total)

    ordered = list(reversed(segments.segments_by_risk))  # riskiest first
    rows: List[Dict[str, Any]] = []
    cumulative_population = 0
    cumulative_bads = 0
    bad_rates: List[Optional[float]] = []
    ranks: List[float] = []
    previous_rate: Optional[float] = None
    breaks: List[str] = []

    for segment in ordered:
        in_segment = matured[matured[segment_column] == segment]
        count = int(len(in_segment))
        bads = int((in_segment[OUTCOME_COLUMN] == 1).sum()) if count else 0
        rate = pct(bads, count) if count else None
        cumulative_population += count
        cumulative_bads += bads
        expected_pd = None
        observed_vs_expected = None
        if pd_column and pd_column in matured.columns and count:
            expected = pd.to_numeric(in_segment[pd_column], errors="coerce")
            if expected.notna().any():
                values = expected.to_numpy(dtype="float64")
                if np.nanmax(values) > 1.0:
                    values = values / 100.0
                expected_pd = float(np.nanmean(values)) * 100.0
                observed_vs_expected = safe_divide(rate, expected_pd)
        rows.append({
            "MODEL": model.name,
            "QUARTER": quarter.label,
            "SEGMENT": segment,
            "RISK_RANK": segments.risk_rank.get(segment),
            "MATURED_RECORDS": count,
            "BAD_COUNT": bads,
            "BAD_RATE_PCT": round_or_none(rate, 3),
            "SHARE_OF_POPULATION_PCT": round_or_none(pct(count, total), 2),
            "SHARE_OF_BADS_PCT": round_or_none(pct(bads, total_bads), 2),
            "LIFT": round_or_none(safe_divide(rate, overall_rate * 100.0 if overall_rate else None), 3),
            "CUMULATIVE_POPULATION_PCT": round_or_none(pct(cumulative_population, total), 2),
            "CUMULATIVE_BAD_CAPTURE_PCT": round_or_none(pct(cumulative_bads, total_bads), 2),
            "EXPECTED_PD_PCT": round_or_none(expected_pd, 3),
            "OBSERVED_VS_EXPECTED_RATIO": round_or_none(observed_vs_expected, 3),
            "RANK_ORDER_BREAK": False,
        })
        # ordered riskiest -> safest, so the bad rate should DECREASE
        if rate is not None and previous_rate is not None and rate > previous_rate:
            rows[-1]["RANK_ORDER_BREAK"] = True
            breaks.append(segment)
        if rate is not None:
            previous_rate = rate
        bad_rates.append(rate)
        ranks.append(float(segments.risk_rank.get(segment, np.nan)))

    lift_top = rows[0]["LIFT"] if rows else None
    capture_top2 = rows[1]["CUMULATIVE_BAD_CAPTURE_PCT"] if len(rows) > 1 else None
    return rows, lift_top, capture_top2, bad_rates, ranks, breaks


# ---------------------------------------------------------------------------
# Vintage analysis
# ---------------------------------------------------------------------------


def _vintage_analysis(frame: pd.DataFrame, config: MonitoringConfig) -> pd.DataFrame:
    """Bad rate by origination quarter and segment, with maturity stated."""
    outcome_cfg = config.outcome
    origin_column = outcome_cfg.vintage_date_column or config.date_column
    if origin_column not in frame.columns or OUTCOME_COLUMN not in frame.columns:
        return pd.DataFrame(columns=VINTAGE_COLUMNS)

    origin = pd.to_datetime(frame[origin_column], errors="coerce")
    vintage_label = origin.dt.to_period("Q").astype("string")
    rows: List[Dict[str, Any]] = []

    for model in config.active_models:
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        if segment_column not in frame.columns:
            continue
        work = pd.DataFrame({
            "VINTAGE": vintage_label,
            "SEGMENT": frame[segment_column].astype("object"),
            "OUTCOME": frame[OUTCOME_COLUMN],
            "MATURED": frame[MATURITY_COLUMN] if MATURITY_COLUMN in frame.columns else True,
        }).dropna(subset=["VINTAGE"])
        work = work[work["SEGMENT"].isin(config.segments.segment_order)]
        if work.empty:
            continue
        grouped = work.groupby(["VINTAGE", "SEGMENT"], observed=True)
        for (vintage, segment), group in grouped:
            accounts = int(len(group))
            matured_mask = group["MATURED"] == True  # noqa: E712
            matured = group[matured_mask & group["OUTCOME"].notna()]
            matured_count = int(len(matured))
            bads = int((matured["OUTCOME"] == 1).sum()) if matured_count else 0
            maturity = pct(matured_count, accounts)
            complete = bool(maturity is not None and maturity >= 99.0)
            rows.append({
                "MODEL": model.name,
                "VINTAGE_QUARTER": str(vintage),
                "SEGMENT": str(segment),
                "ACCOUNTS": accounts,
                "MATURED_ACCOUNTS": matured_count,
                "MATURITY_PCT": round_or_none(maturity, 2),
                "BAD_COUNT": bads,
                "BAD_RATE_PCT": round_or_none(pct(bads, matured_count), 3)
                if matured_count else None,
                "COHORT_COMPLETE": complete,
                "NOTE": "" if complete else (
                    f"Only {maturity:.1f}% of this vintage has completed the "
                    f"{outcome_cfg.observation_window_months}-month window; the bad rate is a "
                    "censored lower bound and is not comparable with seasoned vintages."
                    if maturity is not None else "Maturity could not be determined."
                ),
            })
    table = pd.DataFrame(rows, columns=VINTAGE_COLUMNS)
    if not table.empty:
        table = table.sort_values(["MODEL", "VINTAGE_QUARTER", "SEGMENT"]).reset_index(drop=True)
    return table


__all__ = [
    "PERFORMANCE_SUMMARY_COLUMNS", "PERFORMANCE_SEGMENT_COLUMNS", "VINTAGE_COLUMNS",
    "AVAILABILITY_COLUMNS", "PerformanceResult", "compute_performance", "compute_auc",
    "compute_ks", "assess_rank_ordering", "availability_frame",
]
