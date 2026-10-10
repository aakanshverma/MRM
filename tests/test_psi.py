"""PSI: hand calculations, zero-frequency handling and threshold behaviour."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from model_monitoring.config import (
    SegmentConfig,
    Severity,
    SmoothingConfig,
    ThresholdBand,
)
from model_monitoring.psi_calculator import (
    VS_PREVIOUS_QUARTER,
    VS_REFERENCE,
    classify_index,
    compute_distribution_index,
    compute_psi,
    psi_from_distributions,
    segment_counts,
    segment_distribution,
)
from model_monitoring.reference_periods import reference_masks, resolve_reference_periods, \
    select_monitoring_quarters

TOLERANCE = 1e-12
NO_SMOOTHING = SmoothingConfig(method="DROP_EMPTY_BINS")


# ---------------------------------------------------------------------------
# Hand calculations
# ---------------------------------------------------------------------------


def test_psi_matches_hand_calculation(reference_counts, shifted_counts):
    """The implementation must equal the textbook sum term by term."""
    reference_total = sum(reference_counts.values())
    monitoring_total = sum(shifted_counts.values())
    expected = sum(
        (shifted_counts[key] / monitoring_total - reference_counts[key] / reference_total)
        * math.log(
            (shifted_counts[key] / monitoring_total) / (reference_counts[key] / reference_total)
        )
        for key in reference_counts
    )
    actual = psi_from_distributions(reference_counts, shifted_counts, smoothing=NO_SMOOTHING)
    assert actual == pytest.approx(expected, abs=TOLERANCE)


def test_psi_worked_example_two_bins():
    """A two-bin example that can be checked with a calculator."""
    reference = {"GOOD": 700, "BAD": 300}
    monitoring = {"GOOD": 600, "BAD": 400}
    expected = (0.6 - 0.7) * math.log(0.6 / 0.7) + (0.4 - 0.3) * math.log(0.4 / 0.3)
    assert psi_from_distributions(reference, monitoring, smoothing=NO_SMOOTHING) == pytest.approx(
        expected, abs=TOLERANCE
    )
    # sanity check against the value computed by hand: 0.0441833
    assert psi_from_distributions(
        reference, monitoring, smoothing=NO_SMOOTHING
    ) == pytest.approx(0.0441833, abs=1e-6)


def test_psi_is_zero_for_identical_distributions(reference_counts):
    """Requirement: PSI ~ 0 when reference and monitoring match."""
    for smoothing in (SmoothingConfig(), NO_SMOOTHING):
        value = psi_from_distributions(reference_counts, reference_counts, smoothing=smoothing)
        assert value == pytest.approx(0.0, abs=1e-12)


def test_psi_is_zero_for_proportional_distributions(reference_counts):
    """PSI depends on proportions, not absolute volume."""
    doubled = {key: value * 2 for key, value in reference_counts.items()}
    assert psi_from_distributions(reference_counts, doubled, smoothing=NO_SMOOTHING) == \
        pytest.approx(0.0, abs=1e-12)


def test_psi_is_non_negative_on_random_distributions():
    """Requirement: PSI >= 0 within numerical tolerance, always."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        reference = rng.integers(1, 500, 6).astype(float)
        monitoring = rng.integers(0, 500, 6).astype(float)
        index = compute_distribution_index(
            reference, monitoring, [f"S{i}" for i in range(6)], SmoothingConfig()
        )
        assert index.total >= -1e-12
        assert (index.contributions >= -1e-12).all()


def test_psi_grows_with_the_size_of_the_shift(reference_counts):
    """Monotonicity: a bigger move must not produce a smaller index."""
    previous = 0.0
    for weight in (0.0, 0.1, 0.25, 0.5, 0.9):
        monitoring = {
            key: (1 - weight) * value + weight * (1000 if key == "CAT E" else 10)
            for key, value in reference_counts.items()
        }
        value = psi_from_distributions(reference_counts, monitoring)
        assert value >= previous - 1e-12
        previous = value


# ---------------------------------------------------------------------------
# Normalisation and contributions
# ---------------------------------------------------------------------------


def test_proportions_sum_to_one_after_smoothing(reference_counts):
    """Requirement: proportions sum to 1 after the documented normalisation."""
    monitoring = dict(reference_counts)
    monitoring["CAT E"] = 0            # an empty grade
    index = compute_distribution_index(
        list(reference_counts.values()), list(monitoring.values()),
        list(reference_counts), SmoothingConfig(),
    )
    assert index.reference_proportions.sum() == pytest.approx(1.0, abs=1e-12)
    assert index.monitoring_proportions.sum() == pytest.approx(1.0, abs=1e-12)
    assert (index.monitoring_proportions > 0).all()


def test_contributions_sum_to_the_total(reference_counts, shifted_counts):
    """Requirement: segment contributions sum to the reported PSI."""
    index = compute_distribution_index(
        list(reference_counts.values()), list(shifted_counts.values()),
        list(reference_counts), SmoothingConfig(),
    )
    assert index.contributions.sum() == pytest.approx(index.total, abs=1e-12)
    assert index.contribution_sum_error <= SmoothingConfig().tolerance


def test_all_six_segments_take_part_even_when_empty(reference_counts):
    """Requirement: all six segments are represented in the PSI calculation."""
    monitoring = {key: 0 if key in ("CAT D", "CAT E") else value
                  for key, value in reference_counts.items()}
    index = compute_distribution_index(
        list(reference_counts.values()), list(monitoring.values()),
        list(reference_counts), SmoothingConfig(),
    )
    assert index.n_bins == 6
    assert len(index.to_rows()) == 6
    assert {row["CATEGORY"] for row in index.to_rows()} == set(reference_counts)
    # the empty grades must contribute, not be silently skipped
    empty = [row for row in index.to_rows() if row["CATEGORY"] in ("CAT D", "CAT E")]
    assert all(row["CONTRIBUTION"] > 0 for row in empty)


# ---------------------------------------------------------------------------
# Zero-frequency treatment
# ---------------------------------------------------------------------------


def test_pseudocount_smoothing_avoids_division_and_log_errors(reference_counts):
    """A zero-count grade must not raise, produce inf or produce NaN."""
    monitoring = dict(reference_counts)
    monitoring["CAT E"] = 0
    value = psi_from_distributions(reference_counts, monitoring)
    assert math.isfinite(value) and value > 0


def test_drop_empty_bins_reports_how_many_it_dropped(reference_counts):
    monitoring = dict(reference_counts)
    monitoring["CAT E"] = 0
    index = compute_distribution_index(
        list(reference_counts.values()), list(monitoring.values()),
        list(reference_counts), NO_SMOOTHING,
    )
    assert index.bins_dropped == 1
    assert "lower bound" in index.status_reason
    assert math.isfinite(index.total)


def test_pseudocount_size_is_visible_and_changes_the_value(reference_counts):
    """The smoothing parameter must be a documented, effective knob."""
    monitoring = dict(reference_counts)
    monitoring["CAT E"] = 0
    small = psi_from_distributions(
        reference_counts, monitoring, smoothing=SmoothingConfig(pseudocount=0.5)
    )
    large = psi_from_distributions(
        reference_counts, monitoring, smoothing=SmoothingConfig(pseudocount=5.0)
    )
    assert small != large
    assert "0.5" in SmoothingConfig(pseudocount=0.5).describe()


def test_empty_population_is_not_calculable_rather_than_zero():
    """An empty population must be flagged, not reported as a stable PSI of 0."""
    index = compute_distribution_index([0] * 6, [1, 2, 3, 4, 5, 6], list("ABCDEF"), SmoothingConfig())
    assert index.calculable is False
    assert index.total is None
    assert "Reference population is empty" in index.status_reason

    index = compute_distribution_index([1, 2, 3, 4, 5, 6], [0] * 6, list("ABCDEF"), SmoothingConfig())
    assert index.calculable is False
    assert "Monitoring population is empty" in index.status_reason


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError):
        compute_distribution_index([1, 2], [1, 2, 3], ["a", "b"], SmoothingConfig())


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_thresholds_use_unrounded_values():
    """Requirement: alert thresholds must test the raw value, not the display value."""
    band = ThresholdBand(amber=0.10, red=0.25)
    just_below_red = 0.2499
    assert round(just_below_red, 2) == 0.25           # would look like a RED breach
    assert band.classify(just_below_red) is Severity.AMBER
    assert band.classify(0.25) is Severity.RED
    just_below_amber = 0.0999
    assert round(just_below_amber, 2) == 0.10
    assert band.classify(just_below_amber) is Severity.GREEN
    assert band.classify(0.10) is Severity.AMBER


def test_threshold_boundaries_are_inclusive_at_the_lower_edge():
    band = ThresholdBand(amber=0.10, red=0.25)
    assert band.classify(0.09999999) is Severity.GREEN
    assert band.classify(0.10000001) is Severity.AMBER
    assert band.classify(0.24999999) is Severity.AMBER
    assert band.classify(0.25000001) is Severity.RED


def test_missing_or_nan_metric_is_grey_not_green():
    band = ThresholdBand(amber=0.10, red=0.25)
    assert band.classify(None) is Severity.GREY
    assert band.classify(float("nan")) is Severity.GREY
    assert band.classify("not a number") is Severity.GREY


def test_small_samples_are_grey_rather_than_green_or_red(reference_counts):
    """A metric on too little data must never produce a confident label."""
    tiny = {key: max(1, value // 100) for key, value in reference_counts.items()}
    index = compute_distribution_index(
        list(reference_counts.values()), list(tiny.values()),
        list(reference_counts), SmoothingConfig(),
        min_reference_total=100, min_monitoring_total=500,
    )
    severity, reason = classify_index(
        index, ThresholdBand(0.10, 0.25), min_reference_total=100, min_monitoring_total=500
    )
    assert severity is Severity.GREY
    assert "below the configured minimum" in reason


def test_threshold_band_rejects_inverted_bands():
    with pytest.raises(Exception):
        ThresholdBand(amber=0.30, red=0.10)


# ---------------------------------------------------------------------------
# Segment distributions
# ---------------------------------------------------------------------------


def test_segment_counts_include_zero_count_segments(prepared, config):
    """Reindexing against the configured list is what keeps empty grades in."""
    subset = prepared[prepared["__SEGMENT_MODEL_1"] == "CAT A"]
    counts = segment_counts(subset, "MODEL_1", config.segments)
    assert list(counts.index) == config.segments.segment_order
    assert counts["CAT A"] > 0
    assert (counts.drop("CAT A") == 0).all()


def test_segment_counts_exclude_missing_and_invalid_by_default(config):
    frame = pd.DataFrame({
        "__SEGMENT_MODEL_1": ["CAT A", "CAT A", "MISSING", "INVALID", "CAT E"],
    })
    counts = segment_counts(frame, "MODEL_1", config.segments)
    assert counts.sum() == 3
    counts_with = segment_counts(frame, "MODEL_1", config.segments, include_missing=True)
    assert counts_with.sum() == 5
    assert counts_with["MISSING"] == 1 and counts_with["INVALID"] == 1


def test_segment_distribution_shares_sum_to_one(prepared, config):
    table = segment_distribution(prepared, "MODEL_1", config.segments)
    assert table["SHARE"].sum() == pytest.approx(1.0, abs=1e-12)
    assert len(table) == config.segments.n_segments


# ---------------------------------------------------------------------------
# Model-level PSI
# ---------------------------------------------------------------------------


def test_compute_psi_produces_both_comparison_types(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    comparisons = set(result.summary["COMPARISON"])
    assert comparisons == {VS_REFERENCE, VS_PREVIOUS_QUARTER}
    # one fixed-reference row per model-quarter
    fixed = result.summary[result.summary["COMPARISON"] == VS_REFERENCE]
    assert len(fixed) == len(config.active_models) * len(quarters)
    # the first quarter has no predecessor
    qoq = result.summary[result.summary["COMPARISON"] == VS_PREVIOUS_QUARTER]
    assert len(qoq) == len(config.active_models) * (len(quarters) - 1)


def test_psi_detail_contributions_sum_to_total_for_every_comparison(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    grouped = result.detail.groupby(["MODEL", "QUARTER", "COMPARISON"])
    summed = grouped["PSI_CONTRIBUTION"].sum()
    totals = grouped["TOTAL_PSI"].first()
    assert (summed - totals).abs().max() < 1e-9
    # every comparison carries all six grades
    assert (grouped["SEGMENT"].nunique() == config.segments.n_segments).all()


def test_drifting_model_breaches_and_stable_model_does_not(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    fixed = result.summary[result.summary["COMPARISON"] == VS_REFERENCE]
    drifting = fixed[(fixed["MODEL"] == "MODEL_1") & (~fixed["OVERLAPS_REFERENCE"])]
    stable = fixed[(fixed["MODEL"] == "MODEL_2") & (~fixed["OVERLAPS_REFERENCE"])]
    assert drifting["PSI"].max() > stable["PSI"].max()
    assert drifting["PSI"].max() > config.thresholds.psi.amber


def test_quarters_inside_the_reference_window_are_flagged(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    fixed = result.summary[result.summary["COMPARISON"] == VS_REFERENCE]
    flagged = fixed[fixed["OVERLAPS_REFERENCE"]]
    assert not flagged.empty
    # MODEL_1's window is 2024H1, MODEL_2's is 2024H2 -> different quarters flagged
    model_1_flagged = set(flagged[flagged["MODEL"] == "MODEL_1"]["QUARTER"])
    model_2_flagged = set(flagged[flagged["MODEL"] == "MODEL_2"]["QUARTER"])
    assert model_1_flagged == {"2024Q1", "2024Q2"}
    assert model_2_flagged == {"2024Q3", "2024Q4"}
    assert all("overlaps" in reason for reason in flagged["STATUS_REASON"])
