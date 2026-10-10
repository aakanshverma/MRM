"""Segments and migration: risk direction, denominators, upgrades and downgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_monitoring.config import (
    MultiRecordPolicy,
    RiskDirection,
    SegmentConfig,
    Severity,
)
from model_monitoring.mer_metrics import compute_mer, segment_distribution_frame
from model_monitoring.reference_periods import resolve_reference_periods, \
    select_monitoring_quarters
from model_monitoring.segment_migration import (
    COMPARISON_QOQ,
    COMPARISON_REFERENCE,
    build_migration,
    collapse_to_customer,
    compute_segment_migration,
    migration_pivot,
)

SEGMENTS = ["CAT A", "CAT B", "CAT C1", "CAT C2", "CAT D", "CAT E"]


# ---------------------------------------------------------------------------
# Risk direction
# ---------------------------------------------------------------------------


def test_risk_ranks_follow_the_configured_direction():
    increasing = SegmentConfig(risk_direction="A_TO_E_INCREASING")
    assert increasing.risk_rank["CAT A"] == 1
    assert increasing.risk_rank["CAT E"] == 6
    assert increasing.safest_segment == "CAT A"
    assert increasing.riskiest_segment == "CAT E"
    assert increasing.riskier_segments == ["CAT D", "CAT E"]

    decreasing = SegmentConfig(risk_direction="A_TO_E_DECREASING")
    assert decreasing.risk_rank["CAT A"] == 6
    assert decreasing.risk_rank["CAT E"] == 1
    assert decreasing.safest_segment == "CAT E"
    assert decreasing.riskiest_segment == "CAT A"
    assert decreasing.riskier_segments == ["CAT B", "CAT A"]


def test_reversing_the_direction_reverses_upgrades_and_downgrades():
    """The same movement must be read as a downgrade one way and an upgrade the other."""
    from_population = pd.DataFrame({"SEGMENT": ["CAT A"], "RANK": [1.0]}, index=["C1"])
    to_population = pd.DataFrame({"SEGMENT": ["CAT E"], "RANK": [6.0]}, index=["C1"])
    _, stats = build_migration(
        from_population, to_population, SegmentConfig(risk_direction="A_TO_E_INCREASING")
    )
    assert stats["downgrades"] == 1 and stats["upgrades"] == 0

    # under the reversed convention CAT A is the riskiest, so A -> E is an upgrade
    reversed_config = SegmentConfig(risk_direction="A_TO_E_DECREASING")
    from_reversed = pd.DataFrame(
        {"SEGMENT": ["CAT A"], "RANK": [reversed_config.risk_rank["CAT A"]]}, index=["C1"]
    )
    to_reversed = pd.DataFrame(
        {"SEGMENT": ["CAT E"], "RANK": [reversed_config.risk_rank["CAT E"]]}, index=["C1"]
    )
    _, stats = build_migration(from_reversed, to_reversed, reversed_config)
    assert stats["upgrades"] == 1 and stats["downgrades"] == 0


def test_custom_segment_order_is_respected():
    config = SegmentConfig(segment_order=["LOW", "MED", "HIGH"], riskier_segment_count=1)
    assert config.risk_rank == {"LOW": 1, "MED": 2, "HIGH": 3}
    assert config.riskier_segments == ["HIGH"]


def test_invalid_segment_configuration_is_rejected():
    with pytest.raises(Exception):
        SegmentConfig(segment_order=["A", "A", "B"])
    with pytest.raises(Exception):
        SegmentConfig(segment_order=["A"])
    with pytest.raises(Exception):
        SegmentConfig(riskier_segment_count=99)


# ---------------------------------------------------------------------------
# Migration arithmetic
# ---------------------------------------------------------------------------


def _population(mapping: dict, segments: SegmentConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SEGMENT": list(mapping.values()),
            "RANK": [float(segments.risk_rank[value]) for value in mapping.values()],
        },
        index=list(mapping),
    )


def test_migration_row_percentages_use_the_from_grade_as_denominator():
    """Requirement: migration percentages must use the correct denominator."""
    segments = SegmentConfig()
    before = _population(
        {"c1": "CAT A", "c2": "CAT A", "c3": "CAT A", "c4": "CAT A", "c5": "CAT B"}, segments
    )
    after = _population(
        {"c1": "CAT A", "c2": "CAT B", "c3": "CAT B", "c4": "CAT E", "c5": "CAT B"}, segments
    )
    matrix, stats = build_migration(before, after, segments)

    row_a = matrix[matrix["FROM_SEGMENT"] == "CAT A"]
    # four customers started in CAT A: 1 stayed, 2 -> CAT B, 1 -> CAT E
    assert row_a["ROW_PCT"].sum() == pytest.approx(100.0)
    assert row_a[row_a["TO_SEGMENT"] == "CAT A"]["ROW_PCT"].iloc[0] == pytest.approx(25.0)
    assert row_a[row_a["TO_SEGMENT"] == "CAT B"]["ROW_PCT"].iloc[0] == pytest.approx(50.0)
    assert row_a[row_a["TO_SEGMENT"] == "CAT E"]["ROW_PCT"].iloc[0] == pytest.approx(25.0)
    # the one CAT B customer stayed, so that row is 100% on the diagonal
    row_b = matrix[matrix["FROM_SEGMENT"] == "CAT B"]
    assert row_b[row_b["TO_SEGMENT"] == "CAT B"]["ROW_PCT"].iloc[0] == pytest.approx(100.0)
    # TOTAL_PCT uses the matched total as denominator
    assert matrix["TOTAL_PCT"].sum() == pytest.approx(100.0)


def test_upgrade_downgrade_unchanged_counts_and_percentages():
    segments = SegmentConfig()
    before = _population({"a": "CAT C1", "b": "CAT C1", "c": "CAT C1", "d": "CAT C1"}, segments)
    after = _population({"a": "CAT C1", "b": "CAT A", "c": "CAT D", "d": "CAT E"}, segments)
    _, stats = build_migration(before, after, segments)
    assert stats["matched"] == 4
    assert stats["unchanged"] == 1
    assert stats["upgrades"] == 1              # C1 -> A is safer
    assert stats["downgrades"] == 2            # C1 -> D and C1 -> E
    assert stats["unchanged_pct"] == pytest.approx(25.0)
    assert stats["upgrade_pct"] == pytest.approx(25.0)
    assert stats["downgrade_pct"] == pytest.approx(50.0)
    assert stats["net_downgrade_pp"] == pytest.approx(25.0)
    # average ordinal movement: (0 + (1-3) + (5-3) + (6-3)) / 4 = 0.75
    assert stats["average_rank_movement"] == pytest.approx(0.75)
    assert stats["average_abs_rank_movement"] == pytest.approx((0 + 2 + 2 + 3) / 4)
    assert stats["multi_notch_downgrades"] == 2


def test_percentages_sum_to_one_hundred_across_the_three_movement_classes():
    rng = np.random.default_rng(4)
    segments = SegmentConfig()
    keys = [f"c{i}" for i in range(500)]
    before = _population({k: rng.choice(SEGMENTS) for k in keys}, segments)
    after = _population({k: rng.choice(SEGMENTS) for k in keys}, segments)
    _, stats = build_migration(before, after, segments)
    total = stats["unchanged_pct"] + stats["upgrade_pct"] + stats["downgrade_pct"]
    assert total == pytest.approx(100.0, abs=1e-9)


def test_unmatched_populations_are_not_treated_as_migration():
    """Two disjoint customer sets must not produce a migration matrix."""
    segments = SegmentConfig()
    before = _population({"a": "CAT A", "b": "CAT B"}, segments)
    after = _population({"x": "CAT D", "y": "CAT E"}, segments)
    matrix, stats = build_migration(before, after, segments)
    assert matrix.empty
    assert stats["calculable"] is False
    assert stats["matched"] == 0
    assert "No customer appears in both periods" in stats["reason"]


def test_empty_population_is_handled():
    segments = SegmentConfig()
    empty = pd.DataFrame(columns=["SEGMENT", "RANK"])
    matrix, stats = build_migration(empty, _population({"a": "CAT A"}, segments), segments)
    assert matrix.empty and stats["calculable"] is False


def test_migration_matrix_is_square_and_covers_every_segment():
    segments = SegmentConfig()
    before = _population({"a": "CAT A", "b": "CAT E"}, segments)
    after = _population({"a": "CAT B", "b": "CAT E"}, segments)
    matrix, _ = build_migration(before, after, segments)
    assert len(matrix) == segments.n_segments ** 2
    assert set(matrix["FROM_SEGMENT"]) == set(SEGMENTS)
    assert set(matrix["TO_SEGMENT"]) == set(SEGMENTS)


# ---------------------------------------------------------------------------
# Multi-record policy
# ---------------------------------------------------------------------------


def test_multi_record_policies_pick_the_intended_record(config):
    frame = pd.DataFrame({
        "CUSTOMER_ID": ["C1", "C1", "C1", "C2"],
        "LOGIN_DATE": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15"]),
        "__SEGMENT_MODEL_1": ["CAT A", "CAT E", "CAT C1", "CAT B"],
        "__SEGRANK_MODEL_1": [1.0, 6.0, 3.0, 2.0],
    })
    latest = collapse_to_customer(frame, config, "MODEL_1", policy=MultiRecordPolicy.LATEST)
    assert latest.loc["C1", "SEGMENT"] == "CAT C1"
    earliest = collapse_to_customer(frame, config, "MODEL_1", policy=MultiRecordPolicy.EARLIEST)
    assert earliest.loc["C1", "SEGMENT"] == "CAT A"
    worst = collapse_to_customer(frame, config, "MODEL_1", policy=MultiRecordPolicy.WORST)
    assert worst.loc["C1", "SEGMENT"] == "CAT E"
    best = collapse_to_customer(frame, config, "MODEL_1", policy=MultiRecordPolicy.BEST)
    assert best.loc["C1", "SEGMENT"] == "CAT A"
    dropped = collapse_to_customer(frame, config, "MODEL_1", policy=MultiRecordPolicy.DROP)
    assert "C1" not in dropped.index and "C2" in dropped.index


def test_unscored_records_never_take_part_in_migration(config):
    frame = pd.DataFrame({
        "CUSTOMER_ID": ["C1", "C2", "C3"],
        "LOGIN_DATE": pd.to_datetime(["2024-01-01"] * 3),
        "__SEGMENT_MODEL_1": ["CAT A", "MISSING", "INVALID"],
        "__SEGRANK_MODEL_1": [1.0, np.nan, np.nan],
    })
    collapsed = collapse_to_customer(frame, config, "MODEL_1")
    assert list(collapsed.index) == ["C1"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_compute_segment_migration_produces_both_comparisons(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    matrix, summary = compute_segment_migration(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    assert set(summary["COMPARISON"]) == {COMPARISON_REFERENCE, COMPARISON_QOQ}
    assert set(summary["MULTI_RECORD_POLICY"]) == {config.runtime.multi_record_policy.value}
    assert "INTERPRETATION" in summary.columns
    # the interpretation must always carry the causal caveat
    assert all("not by itself evidence of model deterioration" in text
               for text in summary["INTERPRETATION"])


def test_quarters_overlapping_the_reference_window_are_greyed(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    _, summary = compute_segment_migration(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    overlapping = summary[summary["OVERLAPS_REFERENCE"]]
    assert not overlapping.empty
    assert set(overlapping["SEVERITY"]) == {Severity.GREY.value}
    assert all("overlaps the reference window" in reason
               for reason in overlapping["STATUS_REASON"])


def test_migration_pivot_is_square_and_ordered_by_risk(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    matrix, _ = compute_segment_migration(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    pivot = migration_pivot(
        matrix, "MODEL_1", COMPARISON_QOQ, quarters[-1].label, config.segments
    )
    assert list(pivot.index) == config.segments.segment_order
    assert list(pivot.columns) == config.segments.segment_order


# ---------------------------------------------------------------------------
# Segment distribution / MER
# ---------------------------------------------------------------------------


def test_segment_distribution_has_every_segment_for_every_quarter(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    table = segment_distribution_frame(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    expected = len(config.active_models) * len(quarters) * config.segments.n_segments
    assert len(table) == expected
    grouped = table.groupby(["MODEL", "QUARTER"])
    # POPULATION_PCT is rounded to 2 dp for display, so six values can only sum
    # to 100 within the rounding granularity (6 x 0.005 = 0.03 pp). The exact
    # proportions are asserted below from the raw counts.
    assert (grouped["POPULATION_PCT"].sum() - 100.0).abs().max() <= 0.03
    exact = grouped.apply(
        lambda g: (g["CUSTOMER_COUNT"] / g["QUARTER_TOTAL"]).sum(), include_groups=False
    )
    assert (exact - 1.0).abs().max() < 1e-12
    assert set(table["RISK_BAND"]) <= {
        "1-LOWEST RISK", "2-LOW RISK", "3-MEDIUM RISK", "4-HIGH RISK", "5-HIGHEST RISK",
    }


def test_mer_segment_shares_sum_to_one_hundred(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    mer = compute_mer(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    share_columns = [f"PCT_{segment}" for segment in config.segments.segment_order]
    # displayed shares are rounded to 2 dp, so allow the rounding granularity
    totals = mer[share_columns].sum(axis=1)
    assert (totals - 100.0).abs().max() <= 0.03
    # the underlying counts must reconcile exactly
    count_columns = [f"N_{segment}" for segment in config.segments.segment_order]
    assert (mer[count_columns].sum(axis=1) == mer["SCORED_RECORDS"]).all()
    exact = mer[count_columns].div(mer["SCORED_RECORDS"], axis=0).sum(axis=1)
    assert (exact - 1.0).abs().max() < 1e-12


def test_mer_counts_are_internally_consistent(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    mer = compute_mer(
        prepared, config, resolutions=resolutions, monitored_quarters=quarters
    )
    reconstructed = (
        mer["SCORED_RECORDS"] + mer["MISSING_SEGMENT_COUNT"] + mer["INVALID_SEGMENT_COUNT"]
    )
    assert (reconstructed == mer["TOTAL_RECORDS"]).all()
    assert (mer["UNIQUE_LANS"] <= mer["TOTAL_RECORDS"]).all()
    assert (mer["UNIQUE_CUSTOMERS"] <= mer["TOTAL_RECORDS"]).all()
