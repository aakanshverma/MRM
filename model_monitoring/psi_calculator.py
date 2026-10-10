"""Population Stability Index.

Formula
-------
::

    PSI = sum_i (A_i - E_i) * ln(A_i / E_i)

where ``A_i`` is the proportion of the monitoring population in bin ``i`` and
``E_i`` the proportion of the reference (expected) population.  Every term is
non-negative because ``(A - E)`` and ``ln(A / E)`` always carry the same sign,
so each bin's contribution is itself interpretable as "how much this bin moved".

The low-level routine :func:`compute_distribution_index` is shared with the CSI
module: PSI and CSI are the same divergence applied to different partitions
(risk segments vs. characteristic bins), and keeping one implementation means
one place where the zero-frequency treatment, the proportion normalisation and
the contribution-sum assertion live.

Zero-frequency treatment
------------------------
With six risk grades it is entirely normal for a grade to be empty in a quarter.
``ln(0)`` and division by zero are avoided with an explicit, configurable
smoothing rule (see :class:`model_monitoring.config.SmoothingConfig`) which is
printed in the report -- never a silent substitution of an arbitrary floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    ModelConfig,
    MonitoringConfig,
    SegmentConfig,
    Severity,
    SmoothingConfig,
    SmoothingMethod,
    ThresholdBand,
)
from .data_validation import QUARTER_COLUMN, SEGMENT_PREFIX
from .logging_config import get_logger
from .utils import QuarterInfo, periods_overlap, previous_quarter_label, round_or_none

logger = get_logger("psi")

#: Comparison identifiers used throughout the report.
VS_REFERENCE = "VS_REFERENCE"
VS_PREVIOUS_QUARTER = "VS_PREVIOUS_QUARTER"

PSI_SUMMARY_COLUMNS = [
    "MODEL", "QUARTER", "COMPARISON", "REFERENCE_LABEL", "REFERENCE_MODE",
    "REFERENCE_COUNT", "MONITORING_COUNT", "PSI", "PSI_DISPLAY", "SEVERITY",
    "N_BINS", "BINS_DROPPED", "TOP_CONTRIBUTING_SEGMENT", "TOP_CONTRIBUTION",
    "TOP_CONTRIBUTION_SHARE_PCT", "MAX_ABS_PROP_SHIFT_PP", "SMOOTHING",
    "CONTRIBUTION_SUM_ERROR", "CALCULABLE", "OVERLAPS_REFERENCE", "STATUS_REASON",
]

PSI_DETAIL_COLUMNS = [
    "MODEL", "QUARTER", "COMPARISON", "REFERENCE_LABEL", "REFERENCE_MODE",
    "REFERENCE_COUNT", "MONITORING_COUNT", "SEGMENT", "RISK_RANK",
    "REFERENCE_SEGMENT_COUNT", "MONITORING_SEGMENT_COUNT",
    "REFERENCE_PROPORTION", "MONITORING_PROPORTION", "PROPORTION_CHANGE_PP",
    "PSI_CONTRIBUTION", "CONTRIBUTION_SHARE_PCT", "TOTAL_PSI", "SEVERITY",
]


# ---------------------------------------------------------------------------
# Shared divergence engine (used by PSI and CSI)
# ---------------------------------------------------------------------------


@dataclass
class DistributionIndex:
    """Result of one reference-vs-monitoring distribution comparison."""

    categories: List[str]
    reference_counts: np.ndarray
    monitoring_counts: np.ndarray
    reference_proportions: np.ndarray
    monitoring_proportions: np.ndarray
    contributions: np.ndarray
    total: Optional[float]
    bins_dropped: int = 0
    smoothing: str = ""
    calculable: bool = True
    status_reason: str = ""
    #: |sum(contributions) - total| -- must be within ``SmoothingConfig.tolerance``.
    contribution_sum_error: float = 0.0
    reference_total: int = 0
    monitoring_total: int = 0

    @property
    def n_bins(self) -> int:
        return len(self.categories)

    def top_contributor(self) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """``(category, contribution, share_of_total_pct)`` for the largest mover."""
        if not self.calculable or self.contributions.size == 0:
            return None, None, None
        index = int(np.nanargmax(self.contributions))
        contribution = float(self.contributions[index])
        share = None
        if self.total and self.total > 0:
            share = contribution / self.total * 100.0
        return self.categories[index], contribution, share

    def max_abs_shift_pp(self) -> Optional[float]:
        """Largest absolute proportion movement, in percentage points."""
        if not self.calculable:
            return None
        diff = self.monitoring_proportions - self.reference_proportions
        if diff.size == 0:
            return None
        return float(np.nanmax(np.abs(diff)) * 100.0)

    def to_rows(self) -> List[Dict[str, Any]]:
        """Per-bin rows (bin-level detail sheets)."""
        rows: List[Dict[str, Any]] = []
        total = self.total if self.total else None
        for index, category in enumerate(self.categories):
            contribution = float(self.contributions[index])
            rows.append({
                "CATEGORY": category,
                "REFERENCE_COUNT": int(self.reference_counts[index]),
                "MONITORING_COUNT": int(self.monitoring_counts[index]),
                "REFERENCE_PROPORTION": float(self.reference_proportions[index]),
                "MONITORING_PROPORTION": float(self.monitoring_proportions[index]),
                "PROPORTION_CHANGE_PP": float(
                    (self.monitoring_proportions[index] - self.reference_proportions[index]) * 100.0
                ),
                "CONTRIBUTION": contribution,
                "CONTRIBUTION_SHARE_PCT": (
                    contribution / total * 100.0 if total and total > 0 else None
                ),
            })
        return rows


def compute_distribution_index(
    reference_counts: Sequence[float],
    monitoring_counts: Sequence[float],
    categories: Sequence[str],
    smoothing: SmoothingConfig,
    *,
    min_reference_total: int = 0,
    min_monitoring_total: int = 0,
) -> DistributionIndex:
    """Core PSI/CSI computation for one aligned pair of count vectors.

    ``reference_counts`` and ``monitoring_counts`` must already be aligned to
    ``categories`` -- including categories with a zero count, which is exactly
    the case the smoothing rule exists for.
    """
    categories = [str(c) for c in categories]
    reference = np.asarray(reference_counts, dtype="float64")
    monitoring = np.asarray(monitoring_counts, dtype="float64")
    if reference.shape != monitoring.shape or reference.shape[0] != len(categories):
        raise ValueError(
            "reference_counts, monitoring_counts and categories must have the same length "
            f"(got {reference.shape[0]}, {monitoring.shape[0]}, {len(categories)})."
        )
    k = len(categories)
    reference_total = int(np.nansum(reference))
    monitoring_total = int(np.nansum(monitoring))

    def _empty(reason: str) -> DistributionIndex:
        zeros = np.zeros(k, dtype="float64")
        return DistributionIndex(
            categories=categories,
            reference_counts=reference,
            monitoring_counts=monitoring,
            reference_proportions=zeros.copy(),
            monitoring_proportions=zeros.copy(),
            contributions=zeros.copy(),
            total=None,
            smoothing=smoothing.method.value,
            calculable=False,
            status_reason=reason,
            reference_total=reference_total,
            monitoring_total=monitoring_total,
        )

    if k == 0:
        return _empty("No categories to compare.")
    if reference_total <= 0:
        return _empty("Reference population is empty; the index cannot be calculated.")
    if monitoring_total <= 0:
        return _empty("Monitoring population is empty; the index cannot be calculated.")

    bins_dropped = 0
    if smoothing.method is SmoothingMethod.PSEUDOCOUNT:
        alpha = smoothing.pseudocount
        reference_proportions = (reference + alpha) / (reference_total + k * alpha)
        monitoring_proportions = (monitoring + alpha) / (monitoring_total + k * alpha)
        mask = np.ones(k, dtype=bool)
    else:  # DROP_EMPTY_BINS
        reference_proportions = reference / reference_total
        monitoring_proportions = monitoring / monitoring_total
        mask = (reference_proportions > 0) & (monitoring_proportions > 0)
        bins_dropped = int((~mask).sum())

    contributions = np.zeros(k, dtype="float64")
    if mask.any():
        ref_masked = reference_proportions[mask]
        mon_masked = monitoring_proportions[mask]
        contributions[mask] = (mon_masked - ref_masked) * np.log(mon_masked / ref_masked)
    # Guard against -0.0 and floating-point noise on identical distributions.
    contributions[np.abs(contributions) < 1e-18] = 0.0

    total = float(np.nansum(contributions))
    status_reason = ""
    if bins_dropped:
        status_reason = (
            f"{bins_dropped} of {k} bin(s) excluded by the DROP_EMPTY_BINS rule; the reported "
            "index is a lower bound."
        )
    if reference_total < min_reference_total:
        status_reason = (
            f"Reference population ({reference_total:,}) is below the configured minimum of "
            f"{min_reference_total:,}; the index is unreliable. " + status_reason
        ).strip()
    if monitoring_total < min_monitoring_total:
        status_reason = (
            f"Monitoring population ({monitoring_total:,}) is below the configured minimum of "
            f"{min_monitoring_total:,}; the index is unreliable. " + status_reason
        ).strip()

    return DistributionIndex(
        categories=categories,
        reference_counts=reference,
        monitoring_counts=monitoring,
        reference_proportions=reference_proportions,
        monitoring_proportions=monitoring_proportions,
        contributions=contributions,
        total=total,
        bins_dropped=bins_dropped,
        smoothing=smoothing.method.value,
        calculable=True,
        status_reason=status_reason,
        contribution_sum_error=float(abs(np.nansum(contributions) - total)),
        reference_total=reference_total,
        monitoring_total=monitoring_total,
    )


def classify_index(
    index: DistributionIndex,
    band: ThresholdBand,
    *,
    min_reference_total: int,
    min_monitoring_total: int,
) -> Tuple[Severity, str]:
    """Severity for an index value, with sample-size gating.

    A metric computed on too little data is ``GREY`` (unavailable), never
    ``GREEN`` and never a misleading ``RED``: an index on 20 applications says
    nothing about population stability.
    """
    if not index.calculable or index.total is None:
        return Severity.GREY, index.status_reason or "Index could not be calculated."
    if index.reference_total < min_reference_total or index.monitoring_total < min_monitoring_total:
        return Severity.GREY, (
            index.status_reason
            or f"Insufficient observations (reference {index.reference_total:,}, "
               f"monitoring {index.monitoring_total:,})."
        )
    severity = band.classify(index.total)   # NB: unrounded value is used
    return severity, index.status_reason


# ---------------------------------------------------------------------------
# Segment distributions
# ---------------------------------------------------------------------------


def segment_counts(
    frame: pd.DataFrame,
    model_name: str,
    segments: SegmentConfig,
    *,
    include_missing: bool = False,
) -> pd.Series:
    """Counts per configured segment, **including segments with zero records**.

    Reindexing against the configured segment list is what guarantees that all
    six grades take part in the PSI sum even when a grade is empty in a quarter.
    """
    column = f"{SEGMENT_PREFIX}{model_name}"
    categories = list(segments.segment_order)
    if include_missing:
        categories = categories + [segments.missing_label, segments.invalid_label]
    if column not in frame.columns or frame.empty:
        return pd.Series(0, index=pd.Index(categories, name="SEGMENT"), dtype="int64")
    counts = frame[column].astype("object").value_counts(dropna=False)
    counts.index = counts.index.map(lambda v: str(v))
    aligned = counts.reindex(categories).fillna(0).astype("int64")
    aligned.index.name = "SEGMENT"
    return aligned


def segment_distribution(
    frame: pd.DataFrame,
    model_name: str,
    segments: SegmentConfig,
    *,
    include_missing: bool = False,
) -> pd.DataFrame:
    """Counts and raw (unsmoothed) shares per segment, for reporting."""
    counts = segment_counts(frame, model_name, segments, include_missing=include_missing)
    total = int(counts.sum())
    table = counts.to_frame("COUNT")
    table["SHARE"] = (counts / total) if total else 0.0
    table["RISK_RANK"] = [segments.risk_rank.get(str(s)) for s in table.index]
    return table


# ---------------------------------------------------------------------------
# Model-level PSI
# ---------------------------------------------------------------------------


@dataclass
class PsiResult:
    """PSI tables for the whole run."""

    summary: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PSI_SUMMARY_COLUMNS))
    detail: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PSI_DETAIL_COLUMNS))

    def latest(self, model: str, comparison: str = VS_REFERENCE) -> Optional[Dict[str, Any]]:
        """Most recent monitored quarter for one model/comparison."""
        if self.summary.empty:
            return None
        subset = self.summary[
            (self.summary["MODEL"] == model) & (self.summary["COMPARISON"] == comparison)
        ]
        if subset.empty:
            return None
        return subset.sort_values("QUARTER").iloc[-1].to_dict()

    def for_comparison(self, comparison: str) -> pd.DataFrame:
        if self.summary.empty:
            return self.summary
        return self.summary[self.summary["COMPARISON"] == comparison].copy()


def compute_psi(
    frame: pd.DataFrame,
    config: MonitoringConfig,
    *,
    reference_masks: Dict[str, pd.Series],
    monitored_quarters: Sequence[QuarterInfo],
) -> PsiResult:
    """Fixed-reference and quarter-on-quarter PSI for every model and quarter.

    ``reference_masks`` maps model name -> boolean mask selecting that model's
    own reference population (built by :mod:`reference_periods`).  A model's
    mask is never reused for another model.
    """
    segments = config.segments
    thresholds = config.thresholds
    smoothing = config.smoothing
    include_missing = config.runtime.psi_include_missing_segment

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    quarter_labels = [q.label for q in monitored_quarters]

    for model in config.active_models:
        reference_config = model.reference
        reference_label = reference_config.label if reference_config else "NOT CONFIGURED"
        reference_mode = reference_config.mode.value if reference_config else "UNKNOWN"
        mask = reference_masks.get(model.name)
        if mask is None:
            logger.warning(
                "[%s] no reference population resolved; PSI against a fixed baseline is skipped.",
                model.name,
            )
            reference_counts = None
        else:
            reference_frame = frame.loc[mask]
            reference_counts = segment_counts(
                reference_frame, model.name, segments, include_missing=include_missing
            )

        # cache quarterly counts once; both comparisons reuse them
        quarter_counts: Dict[str, pd.Series] = {}
        for label in quarter_labels:
            quarter_frame = frame.loc[frame[QUARTER_COLUMN] == label]
            quarter_counts[label] = segment_counts(
                quarter_frame, model.name, segments, include_missing=include_missing
            )

        for position, label in enumerate(quarter_labels):
            monitoring_counts = quarter_counts[label]
            quarter = monitored_quarters[position]
            overlap_note = ""
            if reference_config is not None and periods_overlap(
                quarter.period_start, quarter.period_end,
                reference_config.start_date, reference_config.end_date,
            ):
                overlap_note = (
                    f"{label} overlaps this scorecard's own reference window "
                    f"({reference_config.label}), so part of the monitoring population IS the "
                    "baseline. The fixed-reference PSI is mechanically close to zero here and "
                    "must not be read as evidence of stability."
                )

            # ---- fixed reference comparison ------------------------------
            if reference_counts is not None:
                summary_row, rows = _psi_rows(
                    model=model,
                    quarter=label,
                    comparison=VS_REFERENCE,
                    reference_counts=reference_counts,
                    monitoring_counts=monitoring_counts,
                    reference_label=reference_label,
                    reference_mode=reference_mode,
                    segments=segments,
                    smoothing=smoothing,
                    band=thresholds.psi,
                    min_reference=thresholds.min_reference_records,
                    min_monitoring=thresholds.min_monitoring_records,
                    decimals=config.runtime.display_decimals,
                    tolerance=smoothing.tolerance,
                    extra_note=overlap_note,
                )
                summary_rows.append(summary_row)
                detail_rows.extend(rows)

            # ---- quarter-on-quarter comparison ---------------------------
            if position == 0:
                continue
            previous_label = quarter_labels[position - 1]
            expected_previous = previous_quarter_label(label)
            note = ""
            if previous_label != expected_previous:
                note = (
                    f"Compared against {previous_label}: {expected_previous} has no records in "
                    "the extract, so the comparison is not strictly consecutive."
                )
            summary_row, rows = _psi_rows(
                model=model,
                quarter=label,
                comparison=VS_PREVIOUS_QUARTER,
                reference_counts=quarter_counts[previous_label],
                monitoring_counts=monitoring_counts,
                reference_label=previous_label,
                reference_mode="PREVIOUS_QUARTER",
                segments=segments,
                smoothing=smoothing,
                band=thresholds.psi,
                min_reference=thresholds.min_monitoring_records,
                min_monitoring=thresholds.min_monitoring_records,
                decimals=config.runtime.display_decimals,
                tolerance=smoothing.tolerance,
                extra_note=note,
            )
            summary_rows.append(summary_row)
            detail_rows.extend(rows)

    summary = pd.DataFrame(summary_rows, columns=PSI_SUMMARY_COLUMNS)
    detail = pd.DataFrame(detail_rows, columns=PSI_DETAIL_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["MODEL", "COMPARISON", "QUARTER"]).reset_index(drop=True)
    if not detail.empty:
        detail = detail.sort_values(
            ["MODEL", "COMPARISON", "QUARTER", "RISK_RANK"]
        ).reset_index(drop=True)
    logger.info(
        "PSI: %d model-quarter comparison(s) (%d vs reference, %d quarter-on-quarter).",
        len(summary),
        int((summary["COMPARISON"] == VS_REFERENCE).sum()) if not summary.empty else 0,
        int((summary["COMPARISON"] == VS_PREVIOUS_QUARTER).sum()) if not summary.empty else 0,
    )
    return PsiResult(summary=summary, detail=detail)


def _psi_rows(
    *,
    model: ModelConfig,
    quarter: str,
    comparison: str,
    reference_counts: pd.Series,
    monitoring_counts: pd.Series,
    reference_label: str,
    reference_mode: str,
    segments: SegmentConfig,
    smoothing: SmoothingConfig,
    band: ThresholdBand,
    min_reference: int,
    min_monitoring: int,
    decimals: int,
    tolerance: float,
    extra_note: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build one summary row plus its per-segment detail rows."""
    categories = [str(c) for c in reference_counts.index]
    index = compute_distribution_index(
        reference_counts.to_numpy(dtype="float64"),
        monitoring_counts.reindex(reference_counts.index).fillna(0).to_numpy(dtype="float64"),
        categories,
        smoothing,
        min_reference_total=min_reference,
        min_monitoring_total=min_monitoring,
    )
    severity, reason = classify_index(
        index, band, min_reference_total=min_reference, min_monitoring_total=min_monitoring
    )
    reason = "; ".join(part for part in (reason, extra_note) if part)

    # Contribution-sum assertion (section 5 requirement).
    if index.calculable and index.contribution_sum_error > tolerance:
        logger.error(
            "[%s] %s %s: segment contributions do not sum to the reported PSI "
            "(error %.3e > tolerance %.3e).",
            model.name, quarter, comparison, index.contribution_sum_error, tolerance,
        )

    top_segment, top_contribution, top_share = index.top_contributor()
    summary_row = {
        "MODEL": model.name,
        "QUARTER": quarter,
        "COMPARISON": comparison,
        "REFERENCE_LABEL": reference_label,
        "REFERENCE_MODE": reference_mode,
        "REFERENCE_COUNT": index.reference_total,
        "MONITORING_COUNT": index.monitoring_total,
        "PSI": index.total,
        "PSI_DISPLAY": round_or_none(index.total, decimals),
        "SEVERITY": severity.value,
        "N_BINS": index.n_bins,
        "BINS_DROPPED": index.bins_dropped,
        "TOP_CONTRIBUTING_SEGMENT": top_segment,
        "TOP_CONTRIBUTION": round_or_none(top_contribution, decimals),
        "TOP_CONTRIBUTION_SHARE_PCT": round_or_none(top_share, 2),
        "MAX_ABS_PROP_SHIFT_PP": round_or_none(index.max_abs_shift_pp(), 2),
        "SMOOTHING": _smoothing_label(smoothing),
        "CONTRIBUTION_SUM_ERROR": index.contribution_sum_error,
        "CALCULABLE": index.calculable,
        "OVERLAPS_REFERENCE": bool(extra_note),
        "STATUS_REASON": reason,
    }

    detail_rows: List[Dict[str, Any]] = []
    for row in index.to_rows():
        detail_rows.append({
            "MODEL": model.name,
            "QUARTER": quarter,
            "COMPARISON": comparison,
            "REFERENCE_LABEL": reference_label,
            "REFERENCE_MODE": reference_mode,
            "REFERENCE_COUNT": index.reference_total,
            "MONITORING_COUNT": index.monitoring_total,
            "SEGMENT": row["CATEGORY"],
            "RISK_RANK": segments.risk_rank.get(row["CATEGORY"]),
            "REFERENCE_SEGMENT_COUNT": row["REFERENCE_COUNT"],
            "MONITORING_SEGMENT_COUNT": row["MONITORING_COUNT"],
            "REFERENCE_PROPORTION": row["REFERENCE_PROPORTION"],
            "MONITORING_PROPORTION": row["MONITORING_PROPORTION"],
            "PROPORTION_CHANGE_PP": row["PROPORTION_CHANGE_PP"],
            "PSI_CONTRIBUTION": row["CONTRIBUTION"],
            "CONTRIBUTION_SHARE_PCT": row["CONTRIBUTION_SHARE_PCT"],
            "TOTAL_PSI": index.total,
            "SEVERITY": severity.value,
        })
    return summary_row, detail_rows


def _smoothing_label(smoothing: SmoothingConfig) -> str:
    if smoothing.method is SmoothingMethod.PSEUDOCOUNT:
        return f"PSEUDOCOUNT(alpha={smoothing.pseudocount:g})"
    return "DROP_EMPTY_BINS"


def psi_from_distributions(
    reference: Dict[str, float] | pd.Series,
    monitoring: Dict[str, float] | pd.Series,
    *,
    categories: Optional[Sequence[str]] = None,
    smoothing: Optional[SmoothingConfig] = None,
) -> float:
    """Convenience helper: PSI from two count dictionaries.

    Useful for ad-hoc checks and for validating the implementation against
    hand-calculated examples (see ``tests/test_psi.py``).
    """
    smoothing = smoothing or SmoothingConfig()
    reference_series = pd.Series(reference, dtype="float64")
    monitoring_series = pd.Series(monitoring, dtype="float64")
    if categories is None:
        categories = list(dict.fromkeys([*reference_series.index, *monitoring_series.index]))
    reference_aligned = reference_series.reindex(categories).fillna(0.0)
    monitoring_aligned = monitoring_series.reindex(categories).fillna(0.0)
    index = compute_distribution_index(
        reference_aligned.to_numpy(), monitoring_aligned.to_numpy(), list(categories), smoothing
    )
    if index.total is None:
        raise ValueError(index.status_reason)
    return index.total


__all__ = [
    "VS_REFERENCE", "VS_PREVIOUS_QUARTER", "PSI_SUMMARY_COLUMNS", "PSI_DETAIL_COLUMNS",
    "DistributionIndex", "compute_distribution_index", "classify_index",
    "segment_counts", "segment_distribution", "PsiResult", "compute_psi",
    "psi_from_distributions",
]
