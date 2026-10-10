"""Schema validation, frame preparation and data-quality checks."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from model_monitoring.config import Severity
from model_monitoring.data_validation import (
    MATURITY_COLUMN,
    OUTCOME_COLUMN,
    QUARTER_COLUMN,
    SEGMENT_PREFIX,
    SEGMENT_RANK_PREFIX,
    ValidationError,
    model_scope_mask,
    prepare_frame,
    run_data_quality_checks,
    validate_schema,
    variable_missingness_checks,
)
from model_monitoring.utils import build_quarters

from helpers import make_config, make_frame


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_valid_extract_passes(frame, config):
    report = validate_schema(frame, config)
    assert report.ok
    assert report.row_count == len(frame)
    assert set(report.models_with_output) == {"MODEL_1", "MODEL_2"}
    assert report.models_without_output == []


def test_missing_date_column_is_fatal(frame, config):
    report = validate_schema(frame.drop(columns=["LOGIN_DATE"]), config)
    assert report.ok is False
    assert any("Mandatory date_column" in error for error in report.errors)
    with pytest.raises(ValidationError):
        report.raise_if_failed()


def test_unparseable_dates_are_fatal(frame, config):
    broken = frame.copy()
    broken["LOGIN_DATE"] = "not a date"
    report = validate_schema(broken, config)
    assert report.ok is False
    assert any("No parseable values" in error for error in report.errors)


def test_empty_dataset_is_fatal(frame, config):
    report = validate_schema(frame.head(0), config)
    assert report.ok is False
    assert any("zero rows" in error for error in report.errors)


def test_missing_segment_column_disables_only_that_model(frame, config):
    report = validate_schema(frame.drop(columns=["MODEL_2_SEGMENT"]), config)
    assert report.ok                           # MODEL_1 still usable
    assert report.models_without_output == ["MODEL_2"]
    assert any("MODEL_2" in warning for warning in report.warnings)


def test_no_usable_scorecard_output_is_fatal(frame, config):
    report = validate_schema(
        frame.drop(columns=["MODEL_1_SEGMENT", "MODEL_2_SEGMENT"]), config
    )
    assert report.ok is False
    assert any("None of the configured scorecards" in error for error in report.errors)


def test_missing_key_columns_warn_but_do_not_stop_the_run(frame, config):
    report = validate_schema(frame.drop(columns=["CUSTOMER_ID"]), config)
    assert report.ok
    assert any("customer_column" in warning for warning in report.warnings)


def test_invalid_segment_values_are_reported(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[:25], "MODEL_1_SEGMENT"] = "CAT Z"
    report = validate_schema(broken, config)
    assert any("outside the configured list" in warning for warning in report.warnings)
    assert any("CAT Z" in warning for warning in report.warnings)


def test_future_dates_are_warned(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[0], "LOGIN_DATE"] = pd.Timestamp(date.today() + timedelta(days=400))
    report = validate_schema(broken, config)
    assert any("future" in warning for warning in report.warnings)


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def test_prepare_frame_adds_quarter_and_normalised_segments(frame, config):
    prepared = prepare_frame(frame, config)
    assert QUARTER_COLUMN in prepared.columns
    assert prepared[QUARTER_COLUMN].str.match(r"^\d{4}Q[1-4]$").all()
    for model in config.active_models:
        segment_column = f"{SEGMENT_PREFIX}{model.name}"
        rank_column = f"{SEGMENT_RANK_PREFIX}{model.name}"
        assert segment_column in prepared.columns
        allowed = set(config.segments.segment_order) | {"MISSING", "INVALID"}
        assert set(prepared[segment_column].unique()) <= allowed
        assert prepared[rank_column].dropna().between(1, 6).all()


def test_prepare_frame_classifies_missing_and_invalid_segments(config):
    frame = pd.DataFrame({
        "LAN_ID": ["L1", "L2", "L3", "L4", "L5"],
        "CUSTOMER_ID": ["C1", "C2", "C3", "C4", "C5"],
        "LOGIN_DATE": pd.to_datetime(["2024-01-15"] * 5),
        "MODEL_1_SEGMENT": ["CAT A", None, "CAT Z", "", "  CAT E  "],
        "MODEL_2_SEGMENT": ["CAT B"] * 5,
    })
    prepared = prepare_frame(frame, config)
    assert list(prepared[f"{SEGMENT_PREFIX}MODEL_1"]) == [
        "CAT A", "MISSING", "INVALID", "MISSING", "CAT E",
    ]
    ranks = prepared[f"{SEGMENT_RANK_PREFIX}MODEL_1"]
    assert ranks.iloc[0] == 1 and ranks.iloc[4] == 6
    assert pd.isna(ranks.iloc[1]) and pd.isna(ranks.iloc[2])


def test_unparseable_dates_become_na_quarters(config):
    frame = pd.DataFrame({
        "LAN_ID": ["L1", "L2"],
        "CUSTOMER_ID": ["C1", "C2"],
        "LOGIN_DATE": ["2024-01-15", "rubbish"],
        "MODEL_1_SEGMENT": ["CAT A", "CAT B"],
        "MODEL_2_SEGMENT": ["CAT A", "CAT B"],
    })
    prepared = prepare_frame(frame, config)
    assert prepared[QUARTER_COLUMN].iloc[0] == "2024Q1"
    assert pd.isna(prepared[QUARTER_COLUMN].iloc[1])


def test_outcome_normalisation_and_maturity(config):
    today = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({
        "LAN_ID": ["L1", "L2", "L3", "L4"],
        "CUSTOMER_ID": ["C1", "C2", "C3", "C4"],
        # two seasoned, two recent
        "LOGIN_DATE": pd.to_datetime(["2024-01-15", "2024-02-15", "2025-11-01", "2025-12-01"]),
        "MODEL_1_SEGMENT": ["CAT A", "CAT E", "CAT A", "CAT E"],
        "MODEL_2_SEGMENT": ["CAT A"] * 4,
        "DEFAULT_FLAG": [1, 0, 0, 1],
        "EXPECTED_PD": [0.1, 0.2, 0.3, 0.4],
    })
    config.outcome.outcome_as_of = date(2026, 1, 1)
    prepared = prepare_frame(frame, config)
    assert list(prepared[OUTCOME_COLUMN]) == [1.0, 0.0, 0.0, 1.0]
    # a 12-month window means only the 2024 applications have matured
    assert list(prepared[MATURITY_COLUMN]) == [True, True, False, False]


def test_unmappable_outcome_values_are_treated_as_unknown(config):
    frame = pd.DataFrame({
        "LAN_ID": ["L1", "L2"],
        "CUSTOMER_ID": ["C1", "C2"],
        "LOGIN_DATE": pd.to_datetime(["2024-01-15"] * 2),
        "MODEL_1_SEGMENT": ["CAT A", "CAT E"],
        "MODEL_2_SEGMENT": ["CAT A"] * 2,
        "DEFAULT_FLAG": [1, "MAYBE"],
    })
    prepared = prepare_frame(frame, config)
    assert prepared[OUTCOME_COLUMN].iloc[0] == 1.0
    assert pd.isna(prepared[OUTCOME_COLUMN].iloc[1])      # not silently a good


def test_model_scope_mask_restricts_to_the_scored_population(config):
    frame = pd.DataFrame({
        "MODEL_1_SEGMENT": ["CAT A", None, "CAT E", None],
        "MODEL_2_SEGMENT": ["CAT A"] * 4,
    })
    mask = model_scope_mask(frame, config, config.model("MODEL_1"), scope="MODEL_OUTPUT")
    assert mask.tolist() == [True, False, True, False]
    everything = model_scope_mask(frame, config, config.model("MODEL_1"), scope="ALL_RECORDS")
    assert everything.all()


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------


def _checks(frame, config):
    prepared = prepare_frame(frame, config)
    quarters = build_quarters(prepared["LOGIN_DATE"])
    return run_data_quality_checks(prepared, config, quarters=quarters)


def test_duplicate_records_are_detected(frame, config):
    doubled = pd.concat([frame, frame.head(60)], ignore_index=True)
    checks = _checks(doubled, config)
    duplicates = checks[checks["CHECK_ID"] == "DQ-001"]
    assert len(duplicates) == 1
    assert duplicates["RECORDS_AFFECTED"].iloc[0] >= 60
    assert duplicates["SEVERITY"].iloc[0] in {"AMBER", "RED"}


def test_invalid_and_missing_segments_are_detected(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[:120], "MODEL_1_SEGMENT"] = "CAT Z"
    broken.loc[broken.index[120:240], "MODEL_1_SEGMENT"] = None
    checks = _checks(broken, config)
    assert not checks[checks["CHECK_ID"] == "DQ-042"].empty     # invalid value
    assert not checks[checks["CHECK_ID"] == "DQ-041"].empty     # missing output
    invalid = checks[checks["CHECK_ID"] == "DQ-042"]
    assert all(model == "MODEL_1" for model in invalid["MODEL"])


def test_high_missingness_in_a_characteristic_is_detected(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[: int(len(broken) * 0.6)], "MODEL_1_VARIABLE_1"] = np.nan
    config.resolve_variables(list(broken.columns))
    checks = _checks(broken, config)
    missingness = checks[checks["CHECK_ID"] == "DQ-050"]
    assert not missingness.empty
    assert missingness["SEVERITY"].iloc[0] == "RED"
    assert "actually scored" in missingness["DETAIL"].iloc[0]


def test_infinite_values_are_detected(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[:5], "MODEL_1_VARIABLE_1"] = np.inf
    config.resolve_variables(list(broken.columns))
    checks = _checks(broken, config)
    assert not checks[checks["CHECK_ID"] == "DQ-051"].empty


def test_unusable_dates_are_detected(frame, config):
    broken = frame.copy()
    broken["LOGIN_DATE"] = broken["LOGIN_DATE"].astype("object")
    broken.loc[broken.index[:50], "LOGIN_DATE"] = None
    checks = _checks(broken, config)
    dates = checks[checks["CHECK_ID"] == "DQ-010"]
    assert dates["RECORDS_AFFECTED"].iloc[0] == 50


def test_incomplete_quarter_is_flagged_grey(config):
    frame = make_frame(n=900, end="2025-11-10")
    checks = _checks(frame, config)
    incomplete = checks[checks["CHECK_ID"] == "DQ-020"]
    assert not incomplete.empty
    assert set(incomplete["SEVERITY"]) == {"GREY"}
    assert "2025Q4" in set(incomplete["QUARTER"])


def test_volume_anomaly_is_detected(config):
    frame = make_frame(n=1_200)
    # drop most of one quarter to create a volume cliff
    prepared = prepare_frame(frame, config)
    keep = ~((prepared[QUARTER_COLUMN] == "2025Q2") & (prepared.index % 4 != 0))
    reduced = frame.loc[keep]
    checks = _checks(reduced, config)
    assert not checks[checks["CHECK_ID"] == "DQ-030"].empty


def test_every_check_row_has_the_standard_schema(frame, config):
    from model_monitoring.data_validation import CHECK_COLUMNS

    checks = _checks(frame, config)
    assert list(checks.columns) == CHECK_COLUMNS
    assert set(checks["SEVERITY"]) <= {"GREEN", "AMBER", "RED", "GREY"}
    assert checks["CHECK_ID"].notna().all()
    assert (checks["RECOMMENDED_ACTION"].fillna("").str.len() > 0).all()


def test_checks_are_sorted_most_severe_first(frame, config):
    broken = frame.copy()
    broken.loc[broken.index[:200], "MODEL_1_SEGMENT"] = None
    checks = _checks(broken, config)
    rank = {"RED": 3, "AMBER": 2, "GREY": 1, "GREEN": 0}
    ranks = checks["SEVERITY"].map(rank).tolist()
    assert ranks == sorted(ranks, reverse=True)


def test_variable_missingness_checks_derive_from_the_csi_table(config):
    csi = pd.DataFrame([
        {
            "MODEL": "MODEL_1", "QUARTER": "2025Q1", "VARIABLE": "V1",
            "MISSING_PCT_REF": 2.0, "MISSING_PCT_MON": 30.0,
            "MISSINGNESS_CHANGE_PP": 28.0, "MONITORING_COUNT": 500,
            "UNSEEN_CATEGORY_PCT": 0.0, "UNSEEN_CATEGORY_COUNT": 0,
        },
        {
            "MODEL": "MODEL_1", "QUARTER": "2025Q1", "VARIABLE": "V2",
            "MISSING_PCT_REF": 2.0, "MISSING_PCT_MON": 2.5,
            "MISSINGNESS_CHANGE_PP": 0.5, "MONITORING_COUNT": 500,
            "UNSEEN_CATEGORY_PCT": 7.5, "UNSEEN_CATEGORY_COUNT": 38,
        },
    ])
    checks = variable_missingness_checks(csi, config.thresholds)
    missingness = checks[checks["CHECK_ID"] == "DQ-055"]
    unseen = checks[checks["CHECK_ID"] == "DQ-056"]
    assert len(missingness) == 1 and missingness["SEVERITY"].iloc[0] == "RED"
    assert len(unseen) == 1 and unseen["SEVERITY"].iloc[0] == "RED"
    # the stable variable raises no missingness row
    assert "V2" not in set(missingness["MODEL"] + missingness["METRIC"])


def test_variable_missingness_checks_handle_an_empty_csi_table(config):
    from model_monitoring.data_validation import CHECK_COLUMNS

    empty = variable_missingness_checks(pd.DataFrame(), config.thresholds)
    assert empty.empty
    assert list(empty.columns) == CHECK_COLUMNS


def test_findings_on_an_incomplete_quarter_are_greyed_not_escalated():
    """A partial quarter cannot support a breach conclusion."""
    frame = make_frame(n=1_200, end="2025-11-10")       # 2025Q4 is partial
    config = make_config()
    # plant a high missing-output rate inside the incomplete quarter only
    prepared_for_mask = prepare_frame(frame, config)
    partial = prepared_for_mask[QUARTER_COLUMN] == "2025Q4"
    broken = frame.copy()
    broken.loc[partial, "MODEL_1_SEGMENT"] = None
    checks = _checks(broken, config)

    partial_rows = checks[
        (checks["QUARTER"] == "2025Q4") & (checks["CHECK_ID"] == "DQ-041")
    ]
    assert not partial_rows.empty
    assert set(partial_rows["SEVERITY"]) == {"GREY"}
    assert all("provisional" in detail for detail in partial_rows["DETAIL"])
    # the rate itself is still reported, not suppressed
    assert (partial_rows["VALUE"] > 50).all()


def test_findings_on_a_complete_quarter_still_escalate():
    frame = make_frame(n=1_200)                          # all quarters complete
    config = make_config()
    prepared_for_mask = prepare_frame(frame, config)
    target = prepared_for_mask[QUARTER_COLUMN] == "2025Q2"
    broken = frame.copy()
    broken.loc[target, "MODEL_1_SEGMENT"] = None
    checks = _checks(broken, config)
    rows = checks[(checks["QUARTER"] == "2025Q2") & (checks["CHECK_ID"] == "DQ-041")]
    assert not rows.empty
    assert set(rows["SEVERITY"]) == {"RED"}
