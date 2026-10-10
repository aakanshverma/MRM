"""CSI: binning rules, reference-bin reuse, special bins and hand calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from model_monitoring.binning import (
    SOURCE_APPROVED_BINS,
    SOURCE_CATEGORY_MAP,
    SOURCE_REFERENCE_CATEGORIES,
    SOURCE_REFERENCE_QUANTILES,
    canonical_value,
    fit_bin_definition,
)
from model_monitoring.config import (
    RuntimeConfig,
    Severity,
    SmoothingConfig,
    VariableSpec,
    VariableType,
)
from model_monitoring.csi_calculator import compute_csi
from model_monitoring.psi_calculator import VS_PREVIOUS_QUARTER, VS_REFERENCE, \
    compute_distribution_index
from model_monitoring.reference_periods import reference_masks, resolve_reference_periods, \
    select_monitoring_quarters
from model_monitoring.utils import MISSING_BIN, OTHER_BIN, UNSEEN_BIN

RUNTIME = RuntimeConfig()
NO_SMOOTHING = SmoothingConfig(method="DROP_EMPTY_BINS")


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------


def test_csi_matches_hand_calculation():
    """CSI uses the same divergence as PSI, so the hand calculation is the same."""
    reference = [400, 300, 200, 100]
    monitoring = [250, 250, 300, 200]
    reference_total, monitoring_total = sum(reference), sum(monitoring)
    expected = sum(
        (m / monitoring_total - r / reference_total)
        * math.log((m / monitoring_total) / (r / reference_total))
        for r, m in zip(reference, monitoring)
    )
    index = compute_distribution_index(
        reference, monitoring, ["b1", "b2", "b3", "b4"], NO_SMOOTHING
    )
    assert index.total == pytest.approx(expected, abs=1e-12)


def test_csi_is_zero_when_variable_distributions_match():
    """Requirement: CSI ~ 0 when the monitoring distribution matches the reference."""
    counts = [120, 340, 220, 180, 140]
    for smoothing in (SmoothingConfig(), NO_SMOOTHING):
        index = compute_distribution_index(
            counts, counts, [f"b{i}" for i in range(5)], smoothing
        )
        assert index.total == pytest.approx(0.0, abs=1e-12)


def test_csi_is_non_negative():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n_bins = int(rng.integers(2, 12))
        reference = rng.integers(1, 800, n_bins).astype(float)
        monitoring = rng.integers(0, 800, n_bins).astype(float)
        index = compute_distribution_index(
            reference, monitoring, [f"b{i}" for i in range(n_bins)], SmoothingConfig()
        )
        assert index.total >= -1e-12


def test_identical_distribution_gives_zero_csi_end_to_end(prepared, config):
    """A quarter compared against itself must produce exactly zero CSI."""
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    quarter = quarters[0].label
    # make the reference population exactly the first monitored quarter
    mask = prepared["__QUARTER"] == quarter
    result = compute_csi(
        prepared, config, reference_masks={"MODEL_1": mask, "MODEL_2": mask},
        monitored_quarters=[quarters[0]],
    )
    self_rows = result.summary[result.summary["QUARTER"] == quarter]
    assert not self_rows.empty
    assert self_rows["CSI"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert set(self_rows["SEVERITY"]) <= {"GREEN", "GREY"}


# ---------------------------------------------------------------------------
# Numeric binning
# ---------------------------------------------------------------------------


def test_approved_bins_take_precedence_over_quantiles():
    reference = pd.Series(np.random.default_rng(1).normal(650, 60, 2_000))
    definition = fit_bin_definition(
        reference, VariableSpec(name="SCORE", bins=[300, 600, 700, 900]), RUNTIME
    )
    assert definition.source == SOURCE_APPROVED_BINS
    # the approved interior boundaries must appear verbatim
    assert set([300.0, 600.0, 700.0, 900.0]).issubset(set(definition.edges.tolist()))


def test_outer_edges_are_extended_so_out_of_range_values_are_not_lost():
    """An out-of-range value must land in an outer bin, never become MISSING."""
    reference = pd.Series(np.linspace(600, 700, 500))
    definition = fit_bin_definition(
        reference, VariableSpec(name="SCORE", bins=[600, 650, 700]), RUNTIME
    )
    applied = definition.apply(pd.Series([10.0, 650.0, 99_999.0]))
    assert applied.counts[MISSING_BIN] == 0
    assert applied.counts.sum() == 3


def test_infinite_values_are_clipped_into_outer_bins_and_counted():
    reference = pd.Series(np.linspace(0, 100, 500))
    definition = fit_bin_definition(reference, VariableSpec(name="RATIO", n_bins=4), RUNTIME)
    applied = definition.apply(pd.Series([np.inf, -np.inf, 50.0, np.nan]))
    assert applied.infinite_count == 2
    assert applied.missing_count == 1
    assert applied.counts.sum() == 4           # nothing dropped


def test_numeric_bins_are_fitted_once_and_reused_across_quarters():
    """The central anti-pattern: re-bucketing per quarter would hide drift."""
    rng = np.random.default_rng(5)
    reference = pd.Series(rng.normal(0, 1, 5_000))
    definition = fit_bin_definition(reference, VariableSpec(name="X", n_bins=5), RUNTIME)
    edges = definition.edges.copy()

    shifted = pd.Series(rng.normal(3, 1, 5_000))        # a large shift
    applied = definition.apply(shifted)
    assert np.array_equal(definition.edges, edges)      # boundaries unchanged

    # with reused bins the shift concentrates in the top bin
    top_share = applied.counts.iloc[-2] / applied.total   # -1 is MISSING
    assert top_share > 0.8
    index = compute_distribution_index(
        definition.apply(reference).counts.to_numpy(),
        applied.counts.to_numpy(), definition.bins, SmoothingConfig(),
    )
    assert index.total > 0.25                           # detected as a big shift

    # had we re-binned the monitoring quarter on its own quantiles, the
    # distribution would look uniform again and the shift would vanish
    rebinned = fit_bin_definition(shifted, VariableSpec(name="X", n_bins=5), RUNTIME)
    naive = compute_distribution_index(
        definition.apply(reference).counts.to_numpy(),
        rebinned.apply(shifted).counts.to_numpy(), definition.bins, SmoothingConfig(),
    )
    assert naive.total < index.total


def test_degenerate_numeric_column_still_produces_a_definition():
    definition = fit_bin_definition(pd.Series([5.0] * 1_000), VariableSpec(name="C"), RUNTIME)
    assert definition.usable
    assert definition.n_bins >= 2
    assert definition.notes


def test_all_missing_reference_is_marked_unusable():
    definition = fit_bin_definition(
        pd.Series([np.nan] * 100), VariableSpec(name="EMPTY"), RUNTIME
    )
    assert definition.usable is False
    assert "missing" in definition.reason.lower()


# ---------------------------------------------------------------------------
# Categorical binning
# ---------------------------------------------------------------------------


def test_rare_reference_categories_are_grouped_into_other():
    reference = pd.Series(["A"] * 700 + ["B"] * 290 + ["RARE1"] * 7 + ["RARE2"] * 3)
    definition = fit_bin_definition(
        reference, VariableSpec(name="CAT", var_type="CATEGORICAL"), RUNTIME
    )
    assert definition.source == SOURCE_REFERENCE_CATEGORIES
    assert OTHER_BIN in definition.bins
    assert definition.value_to_bin["RARE1"] == OTHER_BIN
    assert definition.value_to_bin["A"] == "A"


def test_unseen_categories_are_tracked_separately_from_missing_and_other():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    definition = fit_bin_definition(
        reference, VariableSpec(name="CAT", var_type="CATEGORICAL"), RUNTIME
    )
    applied = definition.apply(pd.Series(["A", "B", "BRAND_NEW", "ALSO_NEW", None]))
    assert applied.counts[UNSEEN_BIN] == 2
    assert applied.counts[MISSING_BIN] == 1
    assert applied.unseen_count == 2
    assert set(applied.unseen_values) == {"BRAND_NEW", "ALSO_NEW"}
    assert applied.unseen_pct == pytest.approx(40.0)


def test_missing_values_are_an_explicit_bin():
    reference = pd.Series(["A"] * 90 + [None] * 10)
    definition = fit_bin_definition(
        reference, VariableSpec(name="CAT", var_type="CATEGORICAL"), RUNTIME
    )
    assert MISSING_BIN in definition.bins
    applied = definition.apply(reference)
    assert applied.counts[MISSING_BIN] == 10
    assert applied.missing_pct == pytest.approx(10.0)


def test_approved_category_map_is_used_preferentially():
    reference = pd.Series(["SAL"] * 600 + ["SE"] * 400)
    definition = fit_bin_definition(
        reference,
        VariableSpec(
            name="EMP", var_type="BINNED",
            category_map={"SAL": "WOE_BAND_1", "SE": "WOE_BAND_2"},
        ),
        RUNTIME,
    )
    assert definition.source == SOURCE_CATEGORY_MAP
    assert "WOE_BAND_1" in definition.bins
    applied = definition.apply(pd.Series(["SAL", "SE", "SE"]))
    assert applied.counts["WOE_BAND_1"] == 1
    assert applied.counts["WOE_BAND_2"] == 2


def test_canonical_values_match_across_dtypes():
    """An int64 reference column and a float64 monitoring column must agree."""
    reference = pd.Series([0, 1, 2] * 200)
    definition = fit_bin_definition(
        reference, VariableSpec(name="FLAG", var_type="CATEGORICAL"), RUNTIME
    )
    # the monitoring quarter has a null, so pandas promotes the column to float
    applied = definition.apply(pd.Series([0.0, 1.0, 2.0, np.nan]))
    assert applied.counts[UNSEEN_BIN] == 0
    assert applied.counts[MISSING_BIN] == 1
    assert applied.counts["0"] == 1
    assert canonical_value(np.float64(3.0)) == canonical_value(3) == "3"
    assert canonical_value(True) == "True"
    assert canonical_value(None) is None


def test_bin_proportions_sum_to_one(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    grouped = result.detail.groupby(["MODEL", "QUARTER", "COMPARISON", "VARIABLE"])
    assert (grouped["REFERENCE_PROPORTION"].sum() - 1.0).abs().max() < 1e-9
    assert (grouped["MONITORING_PROPORTION"].sum() - 1.0).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# Model-level CSI
# ---------------------------------------------------------------------------


def test_csi_contributions_sum_to_total(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    grouped = result.detail.groupby(["MODEL", "QUARTER", "COMPARISON", "VARIABLE"])
    summed = grouped["CSI_CONTRIBUTION"].sum()
    totals = grouped["TOTAL_CSI"].first()
    assert (summed - totals).abs().max() < 1e-9


def test_csi_uses_each_models_own_characteristics_and_reference(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    for model in config.active_models:
        subset = result.summary[result.summary["MODEL"] == model.name]
        expected = {spec.name for spec in config.monitored_variables(model.name)}
        assert set(subset["VARIABLE"]) == expected
        # the reference label must be this model's own window
        fixed = subset[subset["COMPARISON"] == VS_REFERENCE]
        assert set(fixed["REFERENCE_LABEL"]) == {model.reference.label}
    # and the two models' windows genuinely differ
    labels = {
        model.name: model.reference.label for model in config.active_models
    }
    assert len(set(labels.values())) == len(labels)


def test_csi_subscores_and_score_are_monitored_with_their_role(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    roles = result.summary.groupby("VARIABLE")["VARIABLE_ROLE"].first().to_dict()
    assert roles["MODEL_1_VARIABLE_1_SUBSCORE"] == "SUBSCORE"
    assert roles["MODEL_1_TOTAL_SCORE"] == "SCORE"
    assert roles["MODEL_1_VARIABLE_1"] == "FEATURE"


def test_quarter_on_quarter_csi_is_optional(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    masks = reference_masks(resolutions)

    config.runtime.csi_include_qoq = False
    off = compute_csi(prepared, config, reference_masks=masks, monitored_quarters=quarters)
    assert set(off.summary["COMPARISON"]) == {VS_REFERENCE}

    config.runtime.csi_include_qoq = True
    on = compute_csi(prepared, config, reference_masks=masks, monitored_quarters=quarters)
    assert set(on.summary["COMPARISON"]) == {VS_REFERENCE, VS_PREVIOUS_QUARTER}


def test_whitelist_limits_the_characteristics_processed(prepared, config):
    config.runtime.csi_variable_whitelist = ["MODEL_1_VARIABLE_1"]
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    assert set(result.summary["VARIABLE"]) == {"MODEL_1_VARIABLE_1"}


def test_critical_variables_get_the_tighter_band(prepared, config):
    config.model("MODEL_1").variable("MODEL_1_VARIABLE_1").critical = True
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    result = compute_csi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    critical = result.summary[result.summary["VARIABLE"] == "MODEL_1_VARIABLE_1"]
    assert set(critical["THRESHOLD_AMBER"]) == {config.thresholds.csi_critical.amber}
    standard = result.summary[result.summary["VARIABLE"] == "MODEL_1_VARIABLE_2"]
    assert set(standard["THRESHOLD_AMBER"]) == {config.thresholds.csi.amber}


def test_per_variable_threshold_override_wins():
    from model_monitoring.config import ThresholdConfig

    thresholds = ThresholdConfig(csi_variable_overrides={"VAR_X": {"amber": 0.02, "red": 0.07}})
    band = thresholds.csi_band("VAR_X", critical=True)
    assert (band.amber, band.red) == (0.02, 0.07)
    assert thresholds.csi_band("VAR_Y").amber == thresholds.csi.amber


def test_recommended_action_points_at_the_missing_bin_when_that_is_the_driver():
    from model_monitoring.binning import BinDefinition
    from model_monitoring.csi_calculator import recommend_action

    definition = BinDefinition(
        variable="V", var_type=VariableType.NUMERIC, bins=["a", MISSING_BIN],
        source=SOURCE_REFERENCE_QUANTILES,
    )
    action = recommend_action(
        severity=Severity.RED, csi=0.4, variable=VariableSpec(name="V"),
        definition=definition, top_bin=MISSING_BIN, missingness_change=25.0,
        unseen_pct=0.0, calculable=True,
    )
    assert "data supply problem" in action
    assert "feature pipeline" in action
