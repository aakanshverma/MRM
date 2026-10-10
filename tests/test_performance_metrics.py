"""Outcome performance: discrimination, censoring and graceful unavailability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitoring.config import Severity
from model_monitoring.data_validation import prepare_frame
from model_monitoring.performance_metrics import (
    assess_rank_ordering,
    availability_frame,
    compute_auc,
    compute_ks,
    compute_performance,
)
from model_monitoring.psi_calculator import compute_psi
from model_monitoring.reference_periods import reference_masks, resolve_reference_periods, \
    select_monitoring_quarters
from model_monitoring.utils import NOT_AVAILABLE

from helpers import make_config, make_frame


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_auc_of_perfect_and_inverted_rankings():
    labels = np.array([0, 0, 1, 1], dtype=float)
    assert compute_auc(labels, np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert compute_auc(labels, np.array([4, 3, 2, 1])) == pytest.approx(0.0)


def test_auc_of_a_random_score_is_about_one_half():
    rng = np.random.default_rng(2)
    labels = np.r_[np.zeros(5_000), np.ones(5_000)]
    assert compute_auc(labels, rng.random(10_000)) == pytest.approx(0.5, abs=0.03)


def test_auc_is_none_with_only_one_class():
    assert compute_auc(np.array([1.0, 1.0, 1.0]), np.array([1, 2, 3])) is None
    assert compute_auc(np.array([0.0, 0.0]), np.array([1, 2])) is None
    assert compute_auc(np.array([]), np.array([])) is None


def test_numpy_fallback_matches_sklearn():
    """The pipeline must not hard-depend on scikit-learn."""
    import model_monitoring.performance_metrics as module

    rng = np.random.default_rng(5)
    labels = rng.binomial(1, 0.25, 3_000).astype(float)
    scores = labels * 1.1 + rng.normal(0, 1, 3_000)
    with_sklearn = compute_auc(labels, scores)
    original = module._HAVE_SKLEARN
    try:
        module._HAVE_SKLEARN = False
        without_sklearn = compute_auc(labels, scores)
    finally:
        module._HAVE_SKLEARN = original
    assert with_sklearn == pytest.approx(without_sklearn, abs=1e-12)


def test_ks_is_bounded_and_zero_for_identical_distributions():
    statistic, _ = compute_ks(np.array([0.0, 1.0, 0.0, 1.0]), np.array([1, 1, 1, 1]))
    assert statistic == pytest.approx(0.0)
    rng = np.random.default_rng(7)
    labels = rng.binomial(1, 0.3, 2_000).astype(float)
    statistic, at = compute_ks(labels, labels * 3 + rng.normal(0, 1, 2_000))
    assert 0.0 <= statistic <= 1.0
    assert at is not None


def test_ks_of_perfect_separation_is_one():
    labels = np.r_[np.zeros(100), np.ones(100)]
    scores = np.r_[np.zeros(100), np.ones(100)]
    statistic, _ = compute_ks(labels, scores)
    assert statistic == pytest.approx(1.0)


def test_ks_needs_both_classes():
    assert compute_ks(np.array([1.0, 1.0]), np.array([1, 2])) == (None, None)


def test_rank_ordering_verdicts():
    verdict, inversions, spearman = assess_rank_ordering([1.0, 2.0, 4.0, 9.0], [1, 2, 3, 4])
    assert verdict.startswith("MONOTONIC") and inversions == 0 and spearman == 1.0
    verdict, inversions, _ = assess_rank_ordering([1.0, 5.0, 2.0, 9.0], [1, 2, 3, 4])
    assert verdict.startswith("MINOR BREAK") and inversions == 1
    verdict, inversions, _ = assess_rank_ordering([9.0, 1.0, 8.0, 2.0], [1, 2, 3, 4])
    assert verdict.startswith("BROKEN") and inversions == 2


def test_rank_ordering_skips_segments_without_outcomes():
    """A grade with no matured records must not be read as a zero bad rate."""
    verdict, inversions, _ = assess_rank_ordering([1.0, None, 3.0, 5.0], [1, 2, 3, 4])
    assert verdict.startswith("MONOTONIC")
    verdict, _, _ = assess_rank_ordering([1.0, None, None], [1, 2, 3])
    assert verdict.startswith(NOT_AVAILABLE)


# ---------------------------------------------------------------------------
# Unavailability
# ---------------------------------------------------------------------------


def test_no_outcome_column_reports_not_available_and_names_the_dependency():
    """Requirement: never fabricate performance metrics."""
    frame = make_frame(with_outcome=False)
    config = make_config(with_outcome=False)
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)

    assert result.enabled is False
    assert result.summary.empty
    assert not result.availability.empty
    rows = result.availability.set_index("METRIC")
    # every outcome-dependent metric must be unavailable and say why
    outcome_dependent = [
        "Event rate / non-event rate", "Bad rate by segment", "Rank ordering",
        "AUC / Gini", "KS statistic", "Lift / cumulative bad capture",
        "Calibration (observed vs expected PD)", "Vintage analysis",
    ]
    assert set(rows.loc[outcome_dependent, "STATUS"]) == {NOT_AVAILABLE}
    # the score column is still present, so score-level CSI remains available:
    # losing the outcome must not be reported as losing everything
    assert rows.loc["Continuous score distribution (score CSI)", "STATUS"] == "AVAILABLE"
    assert rows.loc["AUC / Gini", "MISSING_DEPENDENCY"] == "outcome.outcome_column"
    assert "outcome.pd_column" in rows.loc[
        "Calibration (observed vs expected PD)", "MISSING_DEPENDENCY"
    ]
    # every unavailable metric explains itself
    assert (result.availability["EXPLANATION"].str.len() > 20).all()


def test_missing_outcome_labels_do_not_break_stability_monitoring():
    """Requirement: stability monitoring must survive a missing outcome column."""
    frame = make_frame(with_outcome=False)
    config = make_config(with_outcome=False)
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    psi = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    assert not psi.summary.empty
    assert psi.summary["PSI"].notna().any()
    assert set(psi.summary["SEVERITY"]) <= {"GREEN", "AMBER", "RED", "GREY"}


def test_outcome_column_present_but_empty_is_not_available(frame, config):
    blanked = frame.copy()
    blanked["DEFAULT_FLAG"] = np.nan
    config.resolve_variables(list(blanked.columns))
    prepared = prepare_frame(blanked, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    assert result.enabled is False


def test_mer_marks_performance_not_available_without_an_outcome():
    from model_monitoring.mer_metrics import compute_mer, enrich_mer_with_performance

    frame = make_frame(with_outcome=False)
    config = make_config(with_outcome=False)
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    mer = compute_mer(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    enriched = enrich_mer_with_performance(mer, None, config)
    assert set(enriched["PERFORMANCE_SEVERITY"]) == {Severity.GREY.value}
    assert all(NOT_AVAILABLE in status for status in enriched["PERFORMANCE_STATUS"])
    assert all("no outcome column configured" in status
               for status in enriched["PERFORMANCE_STATUS"])


# ---------------------------------------------------------------------------
# Censoring
# ---------------------------------------------------------------------------


def test_immature_accounts_are_excluded_not_counted_as_good(config):
    """The core censoring rule."""
    frame = pd.DataFrame({
        "LAN_ID": [f"L{i}" for i in range(400)],
        "CUSTOMER_ID": [f"C{i}" for i in range(400)],
        # half the book is seasoned, half is three months old
        "LOGIN_DATE": pd.to_datetime(
            ["2024-02-15"] * 200 + ["2025-10-15"] * 200
        ),
        "MODEL_1_SEGMENT": (["CAT A"] * 100 + ["CAT E"] * 100) * 2,
        "MODEL_2_SEGMENT": ["CAT A"] * 400,
        "DEFAULT_FLAG": ([0] * 90 + [1] * 10) + ([0] * 70 + [1] * 30) + [0] * 200,
        "EXPECTED_PD": [0.1] * 400,
    })
    config.outcome.outcome_as_of = pd.Timestamp("2026-01-01").date()
    config.thresholds.min_outcome_records = 10
    config.thresholds.min_bad_count = 5
    # every record shares one login date, so no quarter is a full calendar
    # quarter; opt in explicitly rather than have the pipeline refuse
    config.runtime.include_partial_quarters = True
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)

    seasoned = result.summary[
        (result.summary["MODEL"] == "MODEL_1") & (result.summary["QUARTER"] == "2024Q1")
    ].iloc[0]
    assert seasoned["MATURED_RECORDS"] == 200
    assert seasoned["IMMATURE_EXCLUDED"] == 0
    assert seasoned["EVENT_RATE_PCT"] == pytest.approx(20.0)   # 40 bads / 200
    assert bool(seasoned["COHORT_COMPLETE"]) is True

    recent = result.summary[
        (result.summary["MODEL"] == "MODEL_1") & (result.summary["QUARTER"] == "2025Q4")
    ].iloc[0]
    assert recent["MATURED_RECORDS"] == 0
    assert recent["IMMATURE_EXCLUDED"] == 200
    # critically: the recent cohort is NOT reported as a 0% bad rate
    assert pd.isna(recent["EVENT_RATE_PCT"]) or recent["EVENT_RATE_PCT"] is None
    assert recent["PERFORMANCE_SEVERITY"] == Severity.GREY.value
    assert "NOT counted as good" in recent["PERFORMANCE_STATUS"]


def test_partially_matured_cohort_is_flagged_incomplete(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    partial = result.summary[
        (result.summary["MATURITY_PCT"] > 0) & (result.summary["MATURITY_PCT"] < 99)
    ]
    if not partial.empty:
        assert (partial["COHORT_COMPLETE"] == False).all()  # noqa: E712
        assert all("seasons" in status or "matured" in status
                   for status in partial["PERFORMANCE_STATUS"])


def test_vintage_cohorts_carry_a_censoring_note(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    assert not result.vintage.empty
    censored = result.vintage[result.vintage["COHORT_COMPLETE"] == False]  # noqa: E712
    if not censored.empty:
        assert all("censored lower bound" in note for note in censored["NOTE"])


# ---------------------------------------------------------------------------
# Discrimination, lift and calibration
# ---------------------------------------------------------------------------


def test_performance_metrics_are_computed_when_outcomes_exist(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    assert result.enabled
    usable = result.summary[result.summary["GINI"].notna()]
    assert not usable.empty
    assert (usable["GINI"].between(-1, 1)).all()
    assert (usable["AUC"].between(0, 1)).all()
    assert (usable["KS_STATISTIC"].between(0, 1)).all()
    # Gini = 2 * AUC - 1, within the 4-decimal display rounding of both columns
    assert ((usable["GINI"] - (2 * usable["AUC"] - 1)).abs() < 1e-3).all()


def test_cumulative_bad_capture_is_monotonic_and_ends_at_one_hundred(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    for (model, quarter), group in result.by_segment.groupby(["MODEL", "QUARTER"]):
        if group["BAD_COUNT"].sum() == 0:
            continue
        capture = group["CUMULATIVE_BAD_CAPTURE_PCT"].dropna().tolist()
        population = group["CUMULATIVE_POPULATION_PCT"].dropna().tolist()
        assert capture == sorted(capture)
        assert population == sorted(population)
        assert capture[-1] == pytest.approx(100.0, abs=0.05)
        assert population[-1] == pytest.approx(100.0, abs=0.05)


def test_segments_are_ordered_riskiest_first(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    first_group = result.by_segment.groupby(["MODEL", "QUARTER"]).first().reset_index()
    assert set(first_group["SEGMENT"]) == {config.segments.riskiest_segment}


def test_calibration_detects_under_prediction(config):
    """An expected PD well below the observed bad rate must be flagged."""
    n = 1_000
    frame = pd.DataFrame({
        "LAN_ID": [f"L{i}" for i in range(n)],
        "CUSTOMER_ID": [f"C{i}" for i in range(n)],
        "LOGIN_DATE": pd.to_datetime(["2024-02-15"] * n),
        "MODEL_1_SEGMENT": (["CAT A", "CAT B", "CAT C1", "CAT C2", "CAT D", "CAT E"] * 200)[:n],
        "MODEL_2_SEGMENT": ["CAT A"] * n,
        "DEFAULT_FLAG": [1] * 200 + [0] * 800,          # 20% observed
        "EXPECTED_PD": [0.02] * n,                      # 2% expected
    })
    config.outcome.outcome_as_of = pd.Timestamp("2026-01-01").date()
    config.thresholds.min_outcome_records = 10
    # every record shares one login date, so no quarter is a full calendar
    # quarter; opt in explicitly rather than have the pipeline refuse
    config.runtime.include_partial_quarters = True
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    row = result.summary[result.summary["MODEL"] == "MODEL_1"].iloc[0]
    assert row["OBSERVED_VS_EXPECTED_RATIO"] == pytest.approx(10.0, rel=0.05)
    assert row["CALIBRATION_VERDICT"].startswith("UNDER-PREDICTING")


def test_calibration_within_tolerance_is_reported_as_such(config):
    n = 2_000
    rng = np.random.default_rng(3)
    probability = 0.1
    frame = pd.DataFrame({
        "LAN_ID": [f"L{i}" for i in range(n)],
        "CUSTOMER_ID": [f"C{i}" for i in range(n)],
        "LOGIN_DATE": pd.to_datetime(["2024-02-15"] * n),
        "MODEL_1_SEGMENT": (["CAT A", "CAT E"] * (n // 2)),
        "MODEL_2_SEGMENT": ["CAT A"] * n,
        "DEFAULT_FLAG": rng.binomial(1, probability, n),
        "EXPECTED_PD": [probability] * n,
    })
    config.outcome.outcome_as_of = pd.Timestamp("2026-01-01").date()
    config.thresholds.min_outcome_records = 10
    # every record shares one login date, so no quarter is a full calendar
    # quarter; opt in explicitly rather than have the pipeline refuse
    config.runtime.include_partial_quarters = True
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    row = result.summary[result.summary["MODEL"] == "MODEL_1"].iloc[0]
    assert row["CALIBRATION_VERDICT"].startswith("WITHIN TOLERANCE")


def test_thin_outcome_samples_are_grey_not_red(config):
    n = 40
    frame = pd.DataFrame({
        "LAN_ID": [f"L{i}" for i in range(n)],
        "CUSTOMER_ID": [f"C{i}" for i in range(n)],
        "LOGIN_DATE": pd.to_datetime(["2024-02-15"] * n),
        "MODEL_1_SEGMENT": (["CAT A", "CAT E"] * (n // 2)),
        "MODEL_2_SEGMENT": ["CAT A"] * n,
        "DEFAULT_FLAG": [1] * 2 + [0] * (n - 2),
        "EXPECTED_PD": [0.05] * n,
    })
    config.outcome.outcome_as_of = pd.Timestamp("2026-01-01").date()
    config.thresholds.min_outcome_records = 300
    config.thresholds.min_bad_count = 30
    # every record shares one login date, so no quarter is a full calendar
    # quarter; opt in explicitly rather than have the pipeline refuse
    config.runtime.include_partial_quarters = True
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    row = result.summary[result.summary["MODEL"] == "MODEL_1"].iloc[0]
    assert row["PERFORMANCE_SEVERITY"] == Severity.GREY.value
    assert "below the configured minimums" in row["PERFORMANCE_STATUS"]


def test_score_source_is_reported_and_falls_back_to_segment_rank(frame, config):
    config.resolve_variables(list(frame.columns))
    prepared = prepare_frame(frame, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_performance(prepared, config, monitored_quarters=quarters)
    sources = result.summary.groupby("MODEL")["SCORE_SOURCE"].first()
    # MODEL_1 has a score column, MODEL_2 does not
    assert "MODEL_1_TOTAL_SCORE" in sources["MODEL_1"]
    assert "ordinal segment rank" in sources["MODEL_2"]
    assert "coarse proxy" in sources["MODEL_2"]
