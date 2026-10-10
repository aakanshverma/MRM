"""Reference periods: per-model independence, validation and the refusal to guess."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from model_monitoring.config import (
    ConfigError,
    ModelConfig,
    MonitoringConfig,
    ReferenceMode,
    ReferencePeriodConfig,
)
from model_monitoring.reference_periods import (
    DatasetSpan,
    ReferencePeriodError,
    apply_reference_mapping,
    build_reference_mask,
    dataset_span,
    ensure_reference_periods,
    models_needing_configuration,
    parse_cli_reference,
    prompt_reference_periods,
    reference_masks,
    reference_summary_frame,
    resolve_reference_periods,
    select_monitoring_quarters,
)

from helpers import make_config, make_frame


# ---------------------------------------------------------------------------
# Validation of the configured window
# ---------------------------------------------------------------------------


def test_start_after_end_is_rejected():
    with pytest.raises(ConfigError, match="must be <="):
        ReferencePeriodConfig(
            model="M", start_date="2024-06-30", end_date="2024-01-01"
        )


def test_equal_start_and_end_is_allowed():
    period = ReferencePeriodConfig(model="M", start_date="2024-01-01", end_date="2024-01-01")
    assert period.is_configured


def test_unparseable_date_is_rejected():
    with pytest.raises(ConfigError):
        ReferencePeriodConfig(model="M", start_date="not-a-date", end_date="2024-01-01")


def test_window_outside_the_dataset_is_warned_not_silently_accepted(prepared, config):
    config.model("MODEL_1").reference = ReferencePeriodConfig(
        model="MODEL_1", start_date="2019-01-01", end_date="2019-12-31"
    )
    resolutions = resolve_reference_periods(prepared, config)
    resolution = resolutions["MODEL_1"]
    assert resolution.record_count == 0
    assert resolution.sufficient is False
    assert any("ZERO records" in warning for warning in resolution.warnings)


def test_partially_overlapping_window_is_truncated_with_a_warning(prepared, config):
    config.model("MODEL_1").reference = ReferencePeriodConfig(
        model="MODEL_1", start_date="2023-01-01", end_date="2024-03-31"
    )
    resolutions = resolve_reference_periods(prepared, config)
    resolution = resolutions["MODEL_1"]
    assert resolution.record_count > 0
    assert any("outside the dataset span" in warning for warning in resolution.warnings)


def test_small_reference_population_is_flagged_insufficient(prepared, config):
    config.thresholds.min_reference_records = 10_000_000
    resolutions = resolve_reference_periods(prepared, config)
    for resolution in resolutions.values():
        assert resolution.sufficient is False
        assert any("below the configured minimum" in warning for warning in resolution.warnings)


# ---------------------------------------------------------------------------
# Per-model independence
# ---------------------------------------------------------------------------


def test_each_model_uses_its_own_window(prepared, config):
    """Requirement: reference-period configuration is model-specific."""
    resolutions = resolve_reference_periods(prepared, config)
    assert resolutions["MODEL_1"].label == "2024-01-01 to 2024-06-30"
    assert resolutions["MODEL_2"].label == "2024-07-01 to 2024-12-31"
    assert resolutions["MODEL_1"].mode == ReferenceMode.DEVELOPMENT.value
    assert resolutions["MODEL_2"].mode == ReferenceMode.PRODUCTION_BASELINE.value
    # the masks must select genuinely different rows
    masks = reference_masks(resolutions)
    overlap = (masks["MODEL_1"] & masks["MODEL_2"]).sum()
    assert overlap == 0
    assert masks["MODEL_1"].sum() > 0 and masks["MODEL_2"].sum() > 0


def test_one_models_window_is_never_substituted_for_another(prepared, config):
    """Changing one model's window must not move the other model's baseline."""
    first = resolve_reference_periods(prepared, config)
    before = first["MODEL_2"].record_count

    config.model("MODEL_1").reference = ReferencePeriodConfig(
        model="MODEL_1", start_date="2025-01-01", end_date="2025-03-31"
    )
    second = resolve_reference_periods(prepared, config)
    assert second["MODEL_2"].record_count == before
    assert second["MODEL_1"].record_count != first["MODEL_1"].record_count
    assert second["MODEL_1"].label == "2025-01-01 to 2025-03-31"


def test_reference_summary_reports_counts_and_modes(prepared, config):
    resolutions = resolve_reference_periods(prepared, config)
    table = reference_summary_frame(resolutions)
    assert set(table["MODEL"]) == {"MODEL_1", "MODEL_2"}
    assert (table["RECORDS_IN_REFERENCE"] > 0).all()
    assert set(table["REFERENCE_MODE"]) == {"DEVELOPMENT", "PRODUCTION_BASELINE"}
    assert table["REFERENCE_START"].notna().all()
    assert table["REFERENCE_END"].notna().all()


def test_production_baseline_mode_supports_an_explicit_filter(prepared, config):
    model = config.model("MODEL_2")
    model.reference = ReferencePeriodConfig(
        model="MODEL_2", start_date="2024-07-01", end_date="2024-12-31",
        mode="PRODUCTION_BASELINE",
        baseline_filter="MODEL_2_VARIABLE_1 > 0",
    )
    mask, warnings = build_reference_mask(prepared, config, model)
    assert mask.sum() > 0
    assert not warnings      # filter with PRODUCTION_BASELINE mode is consistent


def test_bad_baseline_filter_is_reported_clearly(prepared, config):
    model = config.model("MODEL_2")
    model.reference = ReferencePeriodConfig(
        model="MODEL_2", start_date="2024-07-01", end_date="2024-12-31",
        mode="PRODUCTION_BASELINE", baseline_filter="THIS_COLUMN_DOES_NOT_EXIST > 1",
    )
    with pytest.raises(ReferencePeriodError, match="baseline_filter"):
        build_reference_mask(prepared, config, model)


# ---------------------------------------------------------------------------
# Refusal to default to the whole dataset
# ---------------------------------------------------------------------------


def test_unconfigured_reference_refuses_to_use_the_whole_dataset(prepared, config):
    """The single most important guard-rail in the framework."""
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    with pytest.raises(ReferencePeriodError) as error:
        build_reference_mask(prepared, config, config.model("MODEL_1"))
    assert "not fall back to the full dataset" in str(error.value)


def test_non_interactive_mode_fails_loudly_when_a_window_is_missing(prepared, config):
    config.model("MODEL_2").reference = ReferencePeriodConfig(model="MODEL_2")
    config.runtime.non_interactive = True
    with pytest.raises(ReferencePeriodError) as error:
        ensure_reference_periods(prepared, config)
    message = str(error.value)
    assert "MODEL_2" in message
    assert "will not default to the full dataset" in message
    assert "--reference" in message


def test_models_needing_configuration_lists_only_the_unconfigured(config):
    assert models_needing_configuration(config) == []
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    assert [m.name for m in models_needing_configuration(config)] == ["MODEL_1"]


def test_resolution_raises_when_no_model_has_a_usable_baseline(prepared, config):
    for model in config.models:
        model.reference = ReferencePeriodConfig(model=model.name)
    with pytest.raises(ReferencePeriodError, match="No scorecard has a usable reference"):
        resolve_reference_periods(prepared, config)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_cli_reference_parsing():
    parsed = parse_cli_reference([
        "MODEL_1=2024-01-01:2024-06-30",
        "MODEL_2=2024-07-01:2024-12-31:PRODUCTION_BASELINE",
    ])
    assert parsed["MODEL_1"] == {"start_date": "2024-01-01", "end_date": "2024-06-30"}
    assert parsed["MODEL_2"]["mode"] == "PRODUCTION_BASELINE"
    with pytest.raises(ConfigError):
        parse_cli_reference(["MODEL_1"])
    with pytest.raises(ConfigError):
        parse_cli_reference(["MODEL_1=2024-01-01"])


def test_apply_reference_mapping_accepts_dicts_and_strings(config):
    apply_reference_mapping(config, {
        "MODEL_1": "2025-01-01:2025-03-31",
        "MODEL_2": {"start_date": "2025-04-01", "end_date": "2025-06-30", "mode": "DEVELOPMENT"},
    })
    assert config.model("MODEL_1").reference.start_date == date(2025, 1, 1)
    assert config.model("MODEL_1").reference.configured_by == "PYTHON_DICT"
    assert config.model("MODEL_2").reference.mode is ReferenceMode.DEVELOPMENT


def test_apply_reference_mapping_rejects_a_malformed_string(config):
    with pytest.raises(ConfigError):
        apply_reference_mapping(config, {"MODEL_1": "2025-01-01"})


# ---------------------------------------------------------------------------
# Interactive prompting
# ---------------------------------------------------------------------------


def test_prompting_asks_for_every_model_separately(prepared, config):
    """Each scorecard must be prompted for independently."""
    for model in config.models:
        model.reference = ReferencePeriodConfig(model=model.name)
    answers = iter([
        "DEVELOPMENT", "2024-01-01", "2024-06-30", "signed off 2024-08",
        "PRODUCTION_BASELINE", "2024-07-01", "2024-12-31", "",
    ])
    messages = []
    prompt_reference_periods(
        prepared, config, input_func=lambda prompt: next(answers),
        output=messages.append,
    )
    assert config.model("MODEL_1").reference.label == "2024-01-01 to 2024-06-30"
    assert config.model("MODEL_2").reference.label == "2024-07-01 to 2024-12-31"
    assert config.model("MODEL_2").reference.mode is ReferenceMode.PRODUCTION_BASELINE
    assert config.model("MODEL_1").reference.source_note == "signed off 2024-08"
    assert all(
        model.reference.configured_by == "INTERACTIVE" for model in config.active_models
    )
    joined = "\n".join(messages)
    assert "MODEL_1:" in joined and "MODEL_2:" in joined
    # the record count of the selected window must be displayed
    assert "record(s) in 2024-01-01 to 2024-06-30" in joined


def test_prompting_rejects_start_after_end_and_re_asks(prepared, config):
    config.model("MODEL_2").reference = ReferencePeriodConfig(
        model="MODEL_2", start_date="2024-07-01", end_date="2024-12-31"
    )
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    answers = iter([
        "", "2024-06-30", "2024-01-01",      # invalid: start after end
        "2024-01-01", "2024-06-30", "",      # corrected
    ])
    messages = []
    prompt_reference_periods(
        prepared, config, input_func=lambda prompt: next(answers), output=messages.append
    )
    assert config.model("MODEL_1").reference.label == "2024-01-01 to 2024-06-30"
    assert any("is after end date" in message for message in messages)


def test_prompting_rejects_a_window_with_no_records(prepared, config):
    config.model("MODEL_2").reference = ReferencePeriodConfig(
        model="MODEL_2", start_date="2024-07-01", end_date="2024-12-31"
    )
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    answers = iter([
        "", "2019-01-01", "2019-06-30",      # no overlap with the data at all
        "2024-01-01", "2024-06-30", "",
    ])
    messages = []
    prompt_reference_periods(
        prepared, config, input_func=lambda prompt: next(answers), output=messages.append
    )
    assert config.model("MODEL_1").reference.label == "2024-01-01 to 2024-06-30"
    assert any("does not overlap the dataset" in message for message in messages)


def test_prompting_requires_confirmation_for_a_thin_window(prepared, config):
    config.thresholds.min_reference_records = 100_000
    config.model("MODEL_2").reference = ReferencePeriodConfig(
        model="MODEL_2", start_date="2024-07-01", end_date="2024-12-31"
    )
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    answers = iter([
        "", "2024-01-01", "2024-01-31", "n",   # declined
        "2024-01-01", "2024-06-30", "y", "",   # accepted with a warning
    ])
    messages = []
    prompt_reference_periods(
        prepared, config, input_func=lambda prompt: next(answers), output=messages.append
    )
    assert any("below the configured minimum" in message for message in messages)
    assert config.model("MODEL_1").reference.label == "2024-01-01 to 2024-06-30"


def test_prompting_can_be_aborted(prepared, config):
    config.model("MODEL_1").reference = ReferencePeriodConfig(model="MODEL_1")
    with pytest.raises(KeyboardInterrupt):
        prompt_reference_periods(
            prepared, config, input_func=lambda prompt: "q", output=lambda *args: None
        )


# ---------------------------------------------------------------------------
# Quarters
# ---------------------------------------------------------------------------


def test_dataset_span_is_derived_from_the_login_date(prepared, config):
    span = dataset_span(prepared, config)
    assert span.min_date == date(2024, 1, 1)
    assert span.max_date == date(2025, 12, 31)
    assert span.record_count == len(prepared)
    assert "2024-01-01" in span.describe()


def test_incomplete_quarters_are_excluded_by_default():
    """Requirement: identify incomplete quarters and exclude them by default."""
    frame = make_frame(n=1_200, end="2025-11-15")     # 2025Q4 is partial
    config = make_config()
    config.resolve_variables(list(frame.columns))
    from model_monitoring.data_validation import prepare_frame

    prepared = prepare_frame(frame, config)
    all_quarters, monitored = select_monitoring_quarters(prepared, config)
    incomplete = [q.label for q in all_quarters if not q.is_complete]
    assert "2025Q4" in incomplete
    assert "2025Q4" not in [q.label for q in monitored]
    assert len(monitored) == len(all_quarters) - len(incomplete)


def test_partial_quarters_can_be_opted_into():
    frame = make_frame(n=1_200, end="2025-11-15")
    config = make_config()
    config.runtime.include_partial_quarters = True
    config.resolve_variables(list(frame.columns))
    from model_monitoring.data_validation import prepare_frame

    prepared = prepare_frame(frame, config)
    all_quarters, monitored = select_monitoring_quarters(prepared, config)
    assert "2025Q4" in [q.label for q in monitored]
    assert len(monitored) == len(all_quarters)


def test_first_quarter_is_incomplete_when_the_data_starts_mid_quarter():
    frame = make_frame(n=800, start="2024-02-10", end="2025-12-31")
    config = make_config(reference_1=("2024-02-10", "2024-06-30"))
    config.resolve_variables(list(frame.columns))
    from model_monitoring.data_validation import prepare_frame

    prepared = prepare_frame(frame, config)
    all_quarters, _ = select_monitoring_quarters(prepared, config)
    first = all_quarters[0]
    assert first.label == "2024Q1"
    assert first.is_complete is False
    assert "starts" in first.completeness_note


def test_reference_adequacy_is_judged_on_the_scored_count(prepared, config):
    """A window full of unscored rows is not an adequate baseline.

    The expected distribution is estimated only from records carrying a grade,
    so adequacy must follow the scored count -- otherwise the adequacy alert
    says GREEN while PSI (which uses the same scored population) says GREY.
    """
    # blank most of MODEL_1's grades inside its reference window
    frame = prepared.copy()
    window = (
        (frame["LOGIN_DATE"].dt.date >= config.model("MODEL_1").reference.start_date)
        & (frame["LOGIN_DATE"].dt.date <= config.model("MODEL_1").reference.end_date)
    )
    targets = frame.index[window][:-20]        # leave only 20 scored
    frame.loc[targets, "__SEGMENT_MODEL_1"] = "MISSING"
    frame.loc[targets, "__SEGRANK_MODEL_1"] = float("nan")

    config.thresholds.min_reference_records = 100
    resolutions = resolve_reference_periods(frame, config)
    model_1 = resolutions["MODEL_1"]
    assert model_1.record_count > 100          # plenty of rows...
    assert model_1.scored_count <= 20          # ...but almost none scored
    assert model_1.sufficient is False
    assert any("carry a valid scorecard segment" in warning for warning in model_1.warnings)
    # the untouched model is unaffected
    assert resolutions["MODEL_2"].sufficient is True


def test_adequacy_and_psi_severity_agree(prepared, config):
    """The adequacy verdict and the PSI label must not contradict each other."""
    from model_monitoring.psi_calculator import VS_REFERENCE, compute_psi

    config.thresholds.min_reference_records = 10_000_000
    resolutions = resolve_reference_periods(prepared, config)
    _, quarters = select_monitoring_quarters(prepared, config)
    psi = compute_psi(
        prepared, config, reference_masks=reference_masks(resolutions),
        monitored_quarters=quarters,
    )
    fixed = psi.summary[psi.summary["COMPARISON"] == VS_REFERENCE]
    assert all(not r.sufficient for r in resolutions.values())
    assert set(fixed["SEVERITY"]) == {"GREY"}
