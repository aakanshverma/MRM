"""End-to-end pipeline, configuration loading and the synthetic generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from model_monitoring.config import (
    ConfigError,
    MonitoringConfig,
    default_config,
    load_config,
    save_config,
)
from model_monitoring.data_loader import DataLoadError, load_dataset, peek_columns
from model_monitoring.main import build_parser, resolve_configuration, run_pipeline
from model_monitoring.synthetic_data import (
    SCENARIOS,
    GeneratorConfig,
    generate_synthetic_dataset,
    scenario_frame,
)

from helpers import make_config, make_frame

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_sample_yaml_configuration_loads_and_validates():
    config = load_config(REPO_ROOT / "config" / "config.sample.yaml")
    assert [model.name for model in config.models] == [
        "MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"
    ]
    assert config.validate() == []
    # the four reference windows must genuinely differ
    windows = {model.name: model.reference.label for model in config.models}
    assert len(set(windows.values())) == 4
    modes = {model.reference.mode.value for model in config.models}
    assert modes == {"DEVELOPMENT", "PRODUCTION_BASELINE"}


def test_sample_json_configuration_matches_the_yaml():
    yaml_config = load_config(REPO_ROOT / "config" / "config.sample.yaml")
    json_config = load_config(REPO_ROOT / "config" / "config.sample.json")
    assert [m.name for m in yaml_config.models] == [m.name for m in json_config.models]
    for model in yaml_config.models:
        other = json_config.model(model.name)
        assert model.reference.label == other.reference.label
        assert model.reference.mode == other.reference.mode


def test_configuration_round_trips_through_disk(tmp_path):
    config = load_config(REPO_ROOT / "config" / "config.sample.yaml")
    path = save_config(config, tmp_path / "effective.json")
    reloaded = load_config(path)
    assert [m.name for m in reloaded.models] == [m.name for m in config.models]
    assert reloaded.segments.risk_direction == config.segments.risk_direction
    assert reloaded.smoothing.pseudocount == config.smoothing.pseudocount


def test_unknown_configuration_keys_are_rejected():
    with pytest.raises(ConfigError, match="Unknown configuration keys"):
        MonitoringConfig.from_dict({"not_a_real_key": 1})


def test_duplicate_model_names_are_rejected():
    with pytest.raises(ConfigError, match="Duplicate model names"):
        MonitoringConfig.from_dict({
            "models": [
                {"name": "M", "segment_column": "A"},
                {"name": "M", "segment_column": "B"},
            ]
        })


def test_missing_configuration_file_is_reported_clearly():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_cli_overrides_are_applied(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args([
        "--config", str(REPO_ROOT / "config" / "config.sample.yaml"),
        "--models", "MODEL_1",
        "--variables", "MODEL_1_VARIABLE_1",
        "--csi-top-n", "7",
        "--output", str(tmp_path),
        "--non-interactive",
        "--no-charts",
        "--include-partial-quarters",
        "--reference", "MODEL_1=2025-01-01:2025-03-31:PRODUCTION_BASELINE",
    ])
    config = resolve_configuration(arguments)
    assert [m.name for m in config.active_models] == ["MODEL_1"]
    assert config.runtime.csi_variable_whitelist == ["MODEL_1_VARIABLE_1"]
    assert config.runtime.csi_top_n == 7
    assert config.runtime.include_partial_quarters is True
    assert config.excel.include_charts is False
    assert config.model("MODEL_1").reference.label == "2025-01-01 to 2025-03-31"
    assert config.model("MODEL_1").reference.mode.value == "PRODUCTION_BASELINE"
    assert config.model("MODEL_1").reference.configured_by == "CLI_ARGUMENT"


def test_cli_rejects_an_unknown_model_name():
    parser = build_parser()
    arguments = parser.parse_args([
        "--config", str(REPO_ROOT / "config" / "config.sample.yaml"),
        "--models", "MODEL_99",
    ])
    with pytest.raises(ConfigError, match="unknown scorecard"):
        resolve_configuration(arguments)


# ---------------------------------------------------------------------------
# Variable discovery
# ---------------------------------------------------------------------------


def test_discovery_resolves_variables_against_the_dataset_header():
    """Wide extracts cannot be enumerated by hand, so discovery must work."""
    config = default_config(["MODEL_1"])
    columns = [
        "LAN_ID", "CUSTOMER_ID", "LOGIN_DATE", "MODEL_1_SEGMENT",
        *[f"MODEL_1_VARIABLE_{i}" for i in range(1, 51)],
        *[f"MODEL_1_VARIABLE_{i}_SUBSCORE" for i in range(1, 51)],
        "MODEL_1_VARIABLE_7_WOE",          # excluded by pattern
        "MODEL_2_VARIABLE_1",              # belongs to another model
        "UNRELATED_COLUMN",
    ]
    config.resolve_variables(columns)
    resolved = {spec.name for spec in config.model("MODEL_1").variables}
    assert len(resolved) == 100
    assert "MODEL_1_VARIABLE_7_WOE" not in resolved
    assert "MODEL_2_VARIABLE_1" not in resolved
    assert "UNRELATED_COLUMN" not in resolved
    roles = {spec.role.value for spec in config.model("MODEL_1").variables}
    assert roles == {"FEATURE", "SUBSCORE"}


def test_configured_but_absent_columns_are_skipped_with_a_warning():
    config = make_config()
    warnings = config.resolve_variables(["LAN_ID", "LOGIN_DATE", "MODEL_1_SEGMENT"])
    assert any("MODEL_1_VARIABLE_1" in warning for warning in warnings)
    assert config.model("MODEL_1").variables == []


def test_column_pruning_loads_only_what_is_needed(tmp_path):
    frame = make_frame(n=200)
    frame["IRRELEVANT_A"] = 1
    frame["IRRELEVANT_B"] = "x"
    path = tmp_path / "extract.csv"
    frame.to_csv(path, index=False)

    config = make_config()
    config.data_source.path = str(path)
    assert "IRRELEVANT_A" in peek_columns(config.data_source)
    result = load_dataset(config)
    assert "IRRELEVANT_A" not in result.columns_loaded
    assert "IRRELEVANT_B" not in result.columns_loaded
    assert result.columns_pruned >= 2
    assert "MODEL_1_VARIABLE_1" in result.columns_loaded


def test_chunked_csv_reading_matches_a_single_read(tmp_path):
    frame = make_frame(n=500)
    path = tmp_path / "extract.csv"
    frame.to_csv(path, index=False)

    config = make_config()
    config.data_source.path = str(path)
    whole = load_dataset(config)

    chunked_config = make_config()
    chunked_config.data_source.path = str(path)
    chunked_config.data_source.chunksize = 73
    chunked = load_dataset(chunked_config)

    assert chunked.rows_read == whole.rows_read
    assert chunked.chunks_read > 1
    assert sorted(chunked.columns_loaded) == sorted(whole.columns_loaded)


def test_missing_input_file_is_reported_with_a_remedy():
    config = make_config()
    config.data_source.path = "no/such/file.csv"
    with pytest.raises(DataLoadError, match="synthetic_data"):
        load_dataset(config)


def test_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    frame = make_frame(n=300)
    path = tmp_path / "extract.parquet"
    frame.to_parquet(path, index=False)
    config = make_config()
    config.data_source.path = str(path)
    result = load_dataset(config)
    assert result.source_format == "parquet"
    assert result.rows_read == 300


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def test_synthetic_generator_plants_every_documented_scenario():
    frame = generate_synthetic_dataset(GeneratorConfig(n_rows=6_000, seed=1))
    assert len(frame) > 6_000                       # duplicates were added
    # 4 scorecards, each with a segment, a score and characteristics
    for model in ("MODEL_1", "MODEL_2", "MODEL_3", "MODEL_4"):
        assert f"{model}_SEGMENT" in frame.columns
        assert f"{model}_TOTAL_SCORE" in frame.columns
        assert any(c.startswith(f"{model}_VARIABLE_") for c in frame.columns)
    # scenario 4: missing and invalid segment values
    values = set(frame["MODEL_1_SEGMENT"].dropna().unique())
    assert values - {"CAT A", "CAT B", "CAT C1", "CAT C2", "CAT D", "CAT E"}
    assert frame["MODEL_1_SEGMENT"].isna().any()
    # scenario 6: an unseen category appears only later in the series
    late = frame.loc[pd.to_datetime(frame["LOGIN_DATE"]) >= "2025-07-01", "MODEL_3_VARIABLE_2"]
    early = frame.loc[pd.to_datetime(frame["LOGIN_DATE"]) < "2025-01-01", "MODEL_3_VARIABLE_2"]
    assert "GIG_ECONOMY" in set(late.dropna())
    assert "GIG_ECONOMY" not in set(early.dropna())
    # scenario 11: duplicates
    assert frame["LAN_ID"].duplicated().any()
    # scenario 5: missing features
    assert frame["MODEL_1_VARIABLE_2"].isna().any()
    # outcome and censoring support
    for column in ("DEFAULT_FLAG", "PERFORMANCE_MATURED", "EXPECTED_PD"):
        assert column in frame.columns


def test_synthetic_generator_is_reproducible():
    first = generate_synthetic_dataset(GeneratorConfig(n_rows=1_200, seed=99))
    second = generate_synthetic_dataset(GeneratorConfig(n_rows=1_200, seed=99))
    pd.testing.assert_frame_equal(first, second)


def test_scenario_manifest_matches_the_documented_list():
    table = scenario_frame()
    assert len(table) == len(SCENARIOS) == 13
    assert list(table.columns) == ["ID", "SCENARIO", "WHERE", "EXPECT"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_pipeline_runs_end_to_end_and_produces_every_table(pipeline_result):
    inputs = pipeline_result.report_inputs
    assert inputs is not None
    for name, table in (
        ("mer", inputs.mer),
        ("model_summary", inputs.model_summary),
        ("segment_distribution", inputs.segment_distribution),
        ("psi summary", inputs.psi.summary),
        ("psi detail", inputs.psi.detail),
        ("csi summary", inputs.csi.summary),
        ("csi detail", inputs.csi.detail),
        ("bin definitions", inputs.csi.definitions),
        ("migration matrix", inputs.migration_matrix),
        ("migration summary", inputs.migration_summary),
        ("data quality", inputs.data_quality),
        ("alerts", inputs.alerts),
    ):
        assert not table.empty, f"{name} is empty"
    assert pipeline_result.status
    assert pipeline_result.elapsed_seconds >= 0


def test_pipeline_writes_a_workbook(tmp_path):
    frame = make_frame(n=900)
    config = make_config()
    config.excel.output_dir = str(tmp_path)
    result = run_pipeline(config, frame=frame, output=lambda *a, **k: None)
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report_path.stat().st_size > 20_000
    assert result.report_path.suffix == ".xlsx"


def test_execution_summary_names_each_models_own_reference_period(pipeline_result):
    text = "\n".join(pipeline_result.summary_lines())
    assert "EXECUTION SUMMARY" in text
    for model in pipeline_result.config.active_models:
        assert model.name in text
        assert model.reference.label in text
    assert "Quarters monitored" in text
    assert "Alerts" in text


def test_pipeline_is_deterministic():
    frame = make_frame(n=800)
    first = run_pipeline(
        make_config(), frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    second = run_pipeline(
        make_config(), frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    pd.testing.assert_frame_equal(
        first.report_inputs.psi.summary, second.report_inputs.psi.summary
    )
    pd.testing.assert_frame_equal(
        first.report_inputs.csi.summary, second.report_inputs.csi.summary
    )
    assert first.alert_summary == second.alert_summary


def test_pipeline_survives_a_model_with_no_usable_output():
    frame = make_frame(n=700)
    frame["MODEL_2_SEGMENT"] = None
    config = make_config()
    result = run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    assert [m.name for m in result.config.active_models] == ["MODEL_1"]
    assert not result.report_inputs.psi.summary.empty
    assert set(result.report_inputs.psi.summary["MODEL"]) == {"MODEL_1"}


def test_pipeline_runs_with_a_single_model_subset():
    frame = make_frame(n=700)
    config = make_config()
    config.model("MODEL_2").enabled = False
    result = run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    assert set(result.report_inputs.mer["MODEL"]) == {"MODEL_1"}


def test_reversed_risk_direction_flows_through_the_whole_pipeline():
    """Changing one configuration flag must stay consistent everywhere."""
    frame = make_frame(n=900)
    config = make_config()
    config.segments.risk_direction = "A_TO_E_DECREASING"
    config.segments = type(config.segments)(
        segment_order=config.segments.segment_order,
        risk_direction="A_TO_E_DECREASING",
    )
    result = run_pipeline(
        config, frame=frame, write_excel=False, output=lambda *a, **k: None
    )
    distribution = result.report_inputs.segment_distribution
    riskiest = distribution[distribution["SEGMENT"] == "CAT A"]
    assert riskiest["RISK_RANK"].eq(6).all()
    assert riskiest["IS_RISKIER_SEGMENT"].all()
    safest = distribution[distribution["SEGMENT"] == "CAT E"]
    assert safest["RISK_RANK"].eq(1).all()
    assert not safest["IS_RISKIER_SEGMENT"].any()


def test_cli_smoke_run_on_synthetic_data(tmp_path):
    """`python main.py` must work end to end, which is the documented contract."""
    completed = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "main.py"),
            "--generate-synthetic", "3000",
            "--data", str(tmp_path / "synthetic.csv"),
            "--config", str(REPO_ROOT / "config" / "config.sample.yaml"),
            "--output", str(tmp_path),
            "--non-interactive", "--quiet", "--no-charts",
        ],
        capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "EXECUTION SUMMARY" in completed.stdout
    assert "Output file:" in completed.stdout
    workbooks = list(tmp_path.glob("*.xlsx"))
    assert len(workbooks) == 1
    assert workbooks[0].stat().st_size > 20_000


def test_cli_refuses_to_guess_a_missing_reference_period(tmp_path):
    frame = make_frame(n=400)
    path = tmp_path / "extract.csv"
    frame.to_csv(path, index=False)
    config_payload = {
        "lan_column": "LAN_ID",
        "customer_column": "CUSTOMER_ID",
        "date_column": "LOGIN_DATE",
        "data_source": {"path": str(path)},
        "models": {
            "MODEL_1": {"segment_column": "MODEL_1_SEGMENT"},
        },
        "runtime": {"non_interactive": True},
    }
    config_path = tmp_path / "incomplete.json"
    config_path.write_text(json.dumps(config_payload))
    completed = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "main.py"),
            "--config", str(config_path), "--output", str(tmp_path), "--quiet",
        ],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 1
    combined = completed.stdout + completed.stderr
    assert "will not default to the full dataset" in combined
    assert not list(tmp_path.glob("*.xlsx"))
