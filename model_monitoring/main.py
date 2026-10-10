"""Pipeline entry point.

::

    python main.py                                   # config file + prompts
    python main.py --config config/config.sample.yaml
    python main.py --generate-synthetic 25000        # end-to-end demo run
    python main.py --reference MODEL_1=2024-01-01:2024-06-30 --non-interactive

Execution order (each step logs and feeds the next):

1.  load the configured dataset, pruning to the columns the configuration needs
2.  validate the required columns and the scorecard output
3.  ask for each model's reference period, unless already configured
4.  display the selected record counts per scorecard
5.  prepare each scorecard independently (quarter, normalised grade, outcome)
6.  quarterly MER metrics
7.  fixed-reference and quarter-on-quarter PSI
8.  CSI for every monitored characteristic
9.  segment migration
10. data-quality and outcome-performance checks
11. threshold-based alerts
12. export the Excel workbook
13. print the output path and an execution summary
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import __version__
from .alert_engine import alert_counts, build_alerts, overall_status
from .config import (
    ConfigError,
    MonitoringConfig,
    default_config,
    load_config,
    save_config,
)
from .csi_calculator import compute_csi
from .data_loader import DataLoadError, LoadResult, load_dataset
from .data_validation import (
    ValidationError,
    prepare_frame,
    run_data_quality_checks,
    validate_schema,
    variable_missingness_checks,
)
from .excel_report import ReportInputs, build_workbook
from .logging_config import ExecutionLog, setup_logging
from .mer_metrics import (
    compute_mer,
    enrich_mer_with_csi,
    enrich_mer_with_performance,
    enrich_mer_with_psi,
    model_level_summary,
    segment_distribution_frame,
)
from .performance_metrics import compute_performance
from .psi_calculator import VS_PREVIOUS_QUARTER, VS_REFERENCE, compute_psi
from .reference_periods import (
    ReferencePeriodError,
    apply_reference_mapping,
    ensure_reference_periods,
    parse_cli_reference,
    reference_masks,
    resolve_reference_periods,
    select_monitoring_quarters,
)
from .segment_migration import compute_segment_migration
from .utils import infer_dtype_hints

DEFAULT_CONFIG_CANDIDATES = (
    "config/config.yaml",
    "config/config.sample.yaml",
    "config/config.json",
    "config/config.sample.json",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_monitoring",
        description=(
            "Quarterly credit-risk scorecard monitoring pipeline: MER, PSI, CSI, segment "
            "migration, data quality, outcome performance and an Excel monitoring pack."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each scorecard is monitored against its OWN reference period. The pipeline will "
            "not fall back to using the whole dataset as a baseline: supply the periods in the "
            "configuration file, with --reference, or answer the interactive prompts."
        ),
    )
    parser.add_argument("--config", help="path to a YAML or JSON configuration file")
    parser.add_argument("--data", help="override data_source.path")
    parser.add_argument(
        "--reference", action="append", metavar="MODEL=START:END[:MODE]",
        help="reference period override; repeat per model "
             "(e.g. --reference MODEL_1=2024-01-01:2024-06-30:DEVELOPMENT)",
    )
    parser.add_argument("--output", help="output directory for the workbook")
    parser.add_argument("--filename", help="output workbook filename")
    parser.add_argument(
        "--models", help="comma-separated subset of models to monitor (others are disabled)"
    )
    parser.add_argument(
        "--variables", help="comma-separated CSI whitelist (partial run on wide extracts)"
    )
    parser.add_argument("--csi-top-n", type=int, help="top-N characteristics in executive views")
    parser.add_argument(
        "--csi-qoq", action="store_true", help="also compute quarter-on-quarter CSI"
    )
    parser.add_argument(
        "--include-partial-quarters", action="store_true",
        help="monitor incomplete calendar quarters as well (off by default)",
    )
    parser.add_argument(
        "--chunksize", type=int, help="read the CSV in chunks of this many rows"
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="fail instead of prompting for a missing reference period",
    )
    parser.add_argument("--no-charts", action="store_true", help="skip the dashboard charts")
    parser.add_argument(
        "--generate-synthetic", type=int, metavar="ROWS", nargs="?", const=25_000,
        help="generate a synthetic extract of ROWS rows and monitor it (demo / test run)",
    )
    parser.add_argument(
        "--save-effective-config", metavar="PATH",
        help="write the fully resolved configuration for reproducibility",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--quiet", action="store_true", help="suppress console logging")
    parser.add_argument("--version", action="version", version=f"model_monitoring {__version__}")
    return parser


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Everything a programmatic caller needs after a run."""

    config: MonitoringConfig
    report_path: Optional[Path]
    report_inputs: Optional[ReportInputs]
    execution_log: ExecutionLog
    elapsed_seconds: float
    alert_summary: Dict[str, int]
    status: str

    def summary_lines(self) -> List[str]:
        lines = [
            "",
            "=" * 78,
            "EXECUTION SUMMARY",
            "=" * 78,
            f"  Overall status      : {self.status}",
            f"  Alerts              : RED {self.alert_summary.get('RED', 0)}, "
            f"AMBER {self.alert_summary.get('AMBER', 0)}, "
            f"GREY {self.alert_summary.get('GREY', 0)}, "
            f"GREEN {self.alert_summary.get('GREEN', 0)}",
        ]
        inputs = self.report_inputs
        if inputs is not None:
            lines.extend([
                f"  Scorecards monitored: {len(self.config.active_models)} "
                f"({', '.join(m.name for m in self.config.active_models)})",
                f"  Quarters monitored  : {len(inputs.monitored_quarters)} "
                f"({', '.join(q.label for q in inputs.monitored_quarters)})",
                f"  Records analysed    : {inputs.load_result.rows_read:,}",
                f"  MER rows            : {len(inputs.mer):,}",
                f"  PSI comparisons     : {len(inputs.psi.summary):,}",
                f"  CSI combinations    : {len(inputs.csi.summary):,}",
                f"  Migration rows      : {len(inputs.migration_summary):,}",
                f"  Data-quality checks : {len(inputs.data_quality):,}",
                f"  Alert register rows : {len(inputs.alerts):,}",
            ])
            for model in self.config.active_models:
                resolution = inputs.resolutions.get(model.name)
                if resolution is None:
                    lines.append(f"  - {model.name}: reference period UNRESOLVED (skipped)")
                    continue
                latest = inputs.psi.latest(model.name, VS_REFERENCE)
                latest_qoq = inputs.psi.latest(model.name, VS_PREVIOUS_QUARTER)
                lines.append(
                    f"  - {model.name}: reference {resolution.label} [{resolution.mode}], "
                    f"{resolution.record_count:,} records | latest PSI "
                    f"{_format_metric(latest)} ({_severity(latest)}) | QoQ PSI "
                    f"{_format_metric(latest_qoq)} ({_severity(latest_qoq)})"
                )
        lines.extend([
            f"  Warnings / errors   : {self.execution_log.warning_count} / "
            f"{self.execution_log.error_count}",
            f"  Elapsed             : {self.elapsed_seconds:.1f}s",
        ])
        if self.report_path is not None:
            lines.extend(["", f"  Excel monitoring pack: {self.report_path}"])
        lines.append("=" * 78)
        return lines


def _format_metric(record: Optional[Dict[str, Any]]) -> str:
    if not record or record.get("PSI") is None:
        return "n/a"
    value = record.get("PSI")
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _severity(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return "GREY"
    return str(record.get("SEVERITY", "GREY"))


# ---------------------------------------------------------------------------
# Configuration assembly
# ---------------------------------------------------------------------------


def resolve_configuration(arguments: argparse.Namespace) -> MonitoringConfig:
    """Load the configuration file (or a sensible default) and apply overrides."""
    config: Optional[MonitoringConfig] = None
    if arguments.config:
        config = load_config(arguments.config)
    else:
        for candidate in DEFAULT_CONFIG_CANDIDATES:
            if Path(candidate).exists():
                config = load_config(candidate)
                break
    if config is None:
        config = default_config()
        config.config_path = "(built-in default configuration)"

    if arguments.data:
        config.data_source.path = arguments.data
        config.data_source.fmt = "auto"
    if arguments.output:
        config.excel.output_dir = arguments.output
    if arguments.filename:
        config.excel.filename_template = arguments.filename
    if arguments.models:
        wanted = {name.strip() for name in arguments.models.split(",") if name.strip()}
        unknown = wanted - {model.name for model in config.models}
        if unknown:
            raise ConfigError(
                f"--models names unknown scorecard(s): {sorted(unknown)}. Configured: "
                f"{[m.name for m in config.models]}"
            )
        for model in config.models:
            model.enabled = model.name in wanted
    if arguments.variables:
        config.runtime.csi_variable_whitelist = [
            name.strip() for name in arguments.variables.split(",") if name.strip()
        ]
    if arguments.csi_top_n is not None:
        config.runtime.csi_top_n = arguments.csi_top_n
    if arguments.csi_qoq:
        config.runtime.csi_include_qoq = True
    if arguments.include_partial_quarters:
        config.runtime.include_partial_quarters = True
    if arguments.chunksize:
        config.data_source.chunksize = arguments.chunksize
    if arguments.non_interactive:
        config.runtime.non_interactive = True
    if arguments.no_charts:
        config.excel.include_charts = False
    if arguments.log_level:
        config.runtime.log_level = arguments.log_level
    if arguments.reference:
        apply_reference_mapping(
            config, parse_cli_reference(arguments.reference), source="CLI_ARGUMENT"
        )
    config.execution_timestamp = datetime.now()
    return config


def _maybe_generate_synthetic(
    arguments: argparse.Namespace, config: MonitoringConfig, logger
) -> Optional[pd.DataFrame]:
    """Generate the synthetic extract when requested."""
    if arguments.generate_synthetic is None:
        return None
    from .synthetic_data import GeneratorConfig, generate_synthetic_dataset

    rows = int(arguments.generate_synthetic)
    logger.info("Generating a synthetic extract of %s rows for this run.", f"{rows:,}")
    frame = generate_synthetic_dataset(GeneratorConfig(n_rows=rows, seed=config.runtime.random_seed))
    target = Path(config.data_source.path or "data/synthetic_scorecard_data.csv")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        logger.info("Synthetic extract written to %s", target)
    except Exception as exc:  # pragma: no cover - a failed save must not stop the run
        logger.warning("Could not persist the synthetic extract (%s); continuing in memory.", exc)
    return frame


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config: MonitoringConfig,
    *,
    frame: Optional[pd.DataFrame] = None,
    execution_log: Optional[ExecutionLog] = None,
    logger=None,
    write_excel: bool = True,
    scenarios: Optional[pd.DataFrame] = None,
    input_func=input,
    output=print,
) -> PipelineResult:
    """Run every stage and (optionally) write the workbook.

    Importable, so the pipeline can be driven from a notebook or a scheduler
    without going through the CLI.
    """
    started = time.perf_counter()
    if logger is None or execution_log is None:
        logger, execution_log = setup_logging(
            config.runtime.log_level,
            log_file=(
                Path(config.excel.output_dir) / "monitoring_run.log"
                if config.runtime.log_to_file else None
            ),
            execution_log=execution_log,
        )

    for warning in config.validate():
        logger.warning(warning)
        execution_log.warning(warning)

    # ---- 1. load ------------------------------------------------------
    execution_log.set_stage("LOAD")
    load_result = load_dataset(config, frame=frame, execution_log=execution_log)

    # ---- 2. validate --------------------------------------------------
    execution_log.set_stage("VALIDATE")
    validation = validate_schema(load_result.frame, config, execution_log=execution_log)
    validation.raise_if_failed()
    logger.info("Validation: %s", validation.summary())
    for model in config.active_models:
        if model.name in validation.models_without_output:
            model.enabled = False
            logger.warning(
                "[%s] disabled for this run: no usable scorecard output in the extract.",
                model.name,
            )
    if not config.active_models:
        raise ValidationError(
            "Every configured scorecard was disabled because none has usable output."
        )

    # ---- 3. prepare ---------------------------------------------------
    execution_log.set_stage("PREPARE")
    prepared = prepare_frame(load_result.frame, config, execution_log=execution_log)
    kind_hints = infer_dtype_hints(prepared)

    # ---- 4. reference periods (prompting if needed) --------------------
    execution_log.set_stage("REFERENCE_PERIODS")
    ensure_reference_periods(
        prepared, config, execution_log=execution_log, input_func=input_func, output=output
    )
    resolutions = resolve_reference_periods(prepared, config, execution_log=execution_log)
    output("")
    output("Reference populations selected:")
    for name, resolution in resolutions.items():
        output(f"  {resolution.describe()}")
    missing = [m.name for m in config.active_models if m.name not in resolutions]
    for name in missing:
        config.model(name).enabled = False
        logger.error(
            "[%s] has no usable reference population and is excluded from this pack.", name
        )
    masks = reference_masks(resolutions)

    # ---- 5. quarters ---------------------------------------------------
    all_quarters, monitored_quarters = select_monitoring_quarters(
        prepared, config, execution_log=execution_log
    )

    # ---- 6. MER --------------------------------------------------------
    execution_log.set_stage("MER")
    mer = compute_mer(
        prepared, config, resolutions=resolutions, monitored_quarters=monitored_quarters
    )
    segment_distribution = segment_distribution_frame(
        prepared, config, resolutions=resolutions, monitored_quarters=monitored_quarters
    )

    # ---- 7. PSI --------------------------------------------------------
    execution_log.set_stage("PSI")
    psi = compute_psi(
        prepared, config, reference_masks=masks, monitored_quarters=monitored_quarters
    )

    # ---- 8. CSI --------------------------------------------------------
    execution_log.set_stage("CSI")
    csi = compute_csi(
        prepared, config, reference_masks=masks, monitored_quarters=monitored_quarters,
        execution_log=execution_log, kind_hints=kind_hints,
    )

    # ---- 9. migration --------------------------------------------------
    execution_log.set_stage("MIGRATION")
    migration_matrix, migration_summary = compute_segment_migration(
        prepared, config, resolutions=resolutions, monitored_quarters=monitored_quarters
    )

    # ---- 10. data quality and performance ------------------------------
    execution_log.set_stage("DATA_QUALITY")
    data_quality = run_data_quality_checks(
        prepared, config, quarters=all_quarters,
        reference_periods={name: r.period for name, r in resolutions.items()},
        execution_log=execution_log,
    )
    variable_checks = variable_missingness_checks(csi.summary, config.thresholds)
    if not variable_checks.empty:
        data_quality = pd.concat([data_quality, variable_checks], ignore_index=True)
        data_quality["__SEV"] = data_quality["SEVERITY"].map(
            {"RED": 3, "AMBER": 2, "GREY": 1, "GREEN": 0}
        ).fillna(0)
        data_quality = (
            data_quality.sort_values(
                ["__SEV", "CATEGORY", "MODEL", "QUARTER"], ascending=[False, True, True, True]
            ).drop(columns="__SEV").reset_index(drop=True)
        )

    execution_log.set_stage("PERFORMANCE")
    performance = compute_performance(prepared, config, monitored_quarters=monitored_quarters)

    # ---- enrichment ----------------------------------------------------
    mer = enrich_mer_with_psi(mer, psi.summary)
    mer = enrich_mer_with_csi(mer, csi.summary, config.thresholds)
    mer = enrich_mer_with_performance(mer, performance.summary, config)
    summary = model_level_summary(
        mer, config, resolutions=resolutions, psi_summary=psi.summary,
        csi_summary=csi.summary, performance=performance.summary,
    )

    # ---- 11. alerts ----------------------------------------------------
    execution_log.set_stage("ALERTS")
    alerts = build_alerts(
        config,
        psi_summary=psi.summary, csi_summary=csi.summary,
        migration_summary=migration_summary, data_quality=data_quality,
        performance=performance.summary, mer=mer, resolutions=resolutions,
    )
    counts = alert_counts(alerts)
    status = overall_status(alerts)

    report_inputs = ReportInputs(
        config=config,
        load_result=load_result,
        validation=validation,
        resolutions=resolutions,
        all_quarters=all_quarters,
        monitored_quarters=monitored_quarters,
        mer=mer,
        model_summary=summary,
        segment_distribution=segment_distribution,
        psi=psi,
        csi=csi,
        migration_matrix=migration_matrix,
        migration_summary=migration_summary,
        data_quality=data_quality,
        performance=performance,
        alerts=alerts,
        execution_log=execution_log,
        scenarios=scenarios,
    )

    # ---- 12. Excel -----------------------------------------------------
    report_path: Optional[Path] = None
    if write_excel:
        execution_log.set_stage("EXCEL")
        stamp = config.execution_timestamp or datetime.now()
        filename = config.excel.filename_template.format(
            timestamp=f"{stamp:%Y%m%d_%H%M%S}",
            date=f"{stamp:%Y%m%d}",
            quarter=(monitored_quarters[-1].label if monitored_quarters else "NA"),
        )
        report_path = build_workbook(report_inputs, Path(config.excel.output_dir) / filename)

    execution_log.set_stage("DONE")
    return PipelineResult(
        config=config,
        report_path=report_path,
        report_inputs=report_inputs,
        execution_log=execution_log,
        elapsed_seconds=time.perf_counter() - started,
        alert_summary=counts,
        status=status,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    execution_log = ExecutionLog()
    try:
        config = resolve_configuration(arguments)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    logger, execution_log = setup_logging(
        config.runtime.log_level,
        log_file=(
            Path(config.excel.output_dir) / "monitoring_run.log"
            if config.runtime.log_to_file else None
        ),
        execution_log=execution_log,
        quiet=arguments.quiet,
    )
    logger.info(
        "Credit risk scorecard monitoring pipeline v%s | configuration: %s",
        __version__, config.config_path,
    )
    logger.info("Scorecards configured:\n%s", config.describe_models())

    scenarios = None
    try:
        frame = _maybe_generate_synthetic(arguments, config, logger)
        if frame is not None:
            from .synthetic_data import scenario_frame

            scenarios = scenario_frame()

        result = run_pipeline(
            config, frame=frame, execution_log=execution_log, logger=logger,
            scenarios=scenarios,
        )
    except (DataLoadError, ValidationError, ReferencePeriodError, ConfigError) as exc:
        logger.error("%s", exc)
        print(f"\nPIPELINE STOPPED: {exc}", file=sys.stderr)
        print(
            "\nSee docs/TROUBLESHOOTING.md for the common causes and remedies.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nAborted by the user; no report was written.", file=sys.stderr)
        return 130
    except MemoryError:
        logger.error("Out of memory while processing the extract.")
        print(
            "\nPIPELINE STOPPED: out of memory. Memory-efficient options that do NOT drop data:"
            "\n  1. stream the input:        --chunksize 250000"
            "\n  2. convert the extract to Parquet so only required columns are read"
            "\n  3. monitor a subset of characteristics: --variables VAR_A,VAR_B"
            "\n  4. monitor one scorecard per run:       --models MODEL_1"
            "\nSee docs/TROUBLESHOOTING.md.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.error("Unexpected failure: %s", exc)
        logger.debug("%s", traceback.format_exc())
        print(f"\nUNEXPECTED ERROR: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        return 1

    if arguments.save_effective_config:
        try:
            saved = save_config(result.config, arguments.save_effective_config)
            logger.info("Effective configuration written to %s", saved)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not save the effective configuration: %s", exc)

    for line in result.summary_lines():
        print(line)
    if result.report_path is not None:
        print(f"\nOutput file: {result.report_path.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    # allow `python model_monitoring/main.py` as well as `python -m model_monitoring.main`
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from model_monitoring.main import main as _main

        raise SystemExit(_main())
    raise SystemExit(main())


__all__ = ["main", "run_pipeline", "resolve_configuration", "build_parser", "PipelineResult"]
