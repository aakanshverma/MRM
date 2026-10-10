"""Methodology, configuration and limitations content for the Excel pack.

Kept out of :mod:`model_monitoring.excel_report` so the workbook builder stays
about layout and this stays about documentation.  Everything here is derived
from the *effective* configuration of the run, so the sheet always describes
what actually happened rather than what a template claims.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import __version__
from .config import MonitoringConfig, ReferenceMode, SmoothingMethod
from .excel_style import FORMAT_COUNT, FORMAT_INDEX, FORMAT_PCT
from .reference_periods import reference_summary_frame

Block = Tuple[str, Optional[str], pd.DataFrame, Optional[Dict[str, str]]]


# ---------------------------------------------------------------------------
# Formula reference
# ---------------------------------------------------------------------------


def formula_frame(config: MonitoringConfig) -> pd.DataFrame:
    """Every metric in the pack, its formula and how to read it."""
    smoothing = config.smoothing
    n_segments = config.segments.n_segments
    rows = [
        {
            "METRIC": "PSI (Population Stability Index)",
            "FORMULA": "PSI = SUM_i (A_i - E_i) * ln(A_i / E_i)",
            "APPLIED TO": f"the {n_segments} configured risk grades",
            "INTERPRETATION": (
                "A_i = monitoring-quarter proportion in grade i; E_i = reference-period "
                "proportion. Every term is non-negative, so each grade's contribution shows how "
                "much that grade moved. Answers 'has the scored population changed', not 'is the "
                "model still predictive'."
            ),
        },
        {
            "METRIC": "PSI - quarter-on-quarter",
            "FORMULA": "same formula, E = previous quarter's distribution",
            "APPLIED TO": "consecutive monitored quarters",
            "INTERPRETATION": (
                "Detects recent deterioration that a fixed development baseline can mask: a "
                "slow drift accumulates in the fixed-reference PSI, while a sudden policy or "
                "sourcing change shows up here first."
            ),
        },
        {
            "METRIC": "CSI (Characteristic Stability Index)",
            "FORMULA": "CSI_j = SUM_i (A_ij - E_ij) * ln(A_ij / E_ij)",
            "APPLIED TO": "each monitored characteristic, subscore and final score",
            "INTERPRETATION": (
                "Same divergence as PSI applied to a characteristic's bins. PSI says the "
                "population moved; CSI says which input moved, which is what an investigation "
                "needs."
            ),
        },
        {
            "METRIC": "Zero-frequency treatment",
            "FORMULA": (
                f"p_i = (n_i + {smoothing.pseudocount:g}) / (N + k * {smoothing.pseudocount:g})"
                if smoothing.method is SmoothingMethod.PSEUDOCOUNT
                else "bins empty on either side are excluded from the sum"
            ),
            "APPLIED TO": "both the reference and the monitoring distribution, in PSI and CSI",
            "INTERPRETATION": smoothing.describe(),
        },
        {
            "METRIC": "Segment concentration",
            "FORMULA": "max_i (n_i / N), reported with HHI = SUM_i (n_i / N)^2",
            "APPLIED TO": "scored applications in a quarter",
            "INTERPRETATION": (
                "A scorecard that puts most of the book in one grade has little practical "
                "discriminatory value whatever its development statistics showed."
            ),
        },
        {
            "METRIC": "Average segment rank",
            "FORMULA": "mean of the ordinal risk rank (1 = safest)",
            "APPLIED TO": "scored applications in a quarter",
            "INTERPRETATION": (
                "A single number for 'how risky is the book this quarter', comparable across "
                "quarters only because the rank mapping is fixed by configuration."
            ),
        },
        {
            "METRIC": "Riskier-grade share",
            "FORMULA": "share of applications in the configured riskiest grade(s)",
            "APPLIED TO": ", ".join(config.segments.riskier_segments),
            "INTERPRETATION": (
                "The headline business metric: it translates directly into expected approval "
                "rates, pricing mix and expected credit loss."
            ),
        },
        {
            "METRIC": "Migration matrix",
            "FORMULA": "row % = n(from i, to j) / n(from i)",
            "APPLIED TO": "customers observed in both periods, one record each",
            "INTERPRETATION": (
                "Upgrade = move to a safer grade under the configured risk direction. The "
                "denominator is the FROM-grade population, so each row sums to 100%."
            ),
        },
        {
            "METRIC": "Net movement toward risk",
            "FORMULA": "downgrade % - upgrade %",
            "APPLIED TO": "matched customers",
            "INTERPRETATION": (
                "Positive means the matched book moved toward riskier grades. It can reflect "
                "population, policy, behaviour or scoring-configuration change, not model "
                "deterioration alone."
            ),
        },
        {
            "METRIC": "Event rate / bad rate",
            "FORMULA": "bads / matured records",
            "APPLIED TO": "accounts whose performance window has elapsed",
            "INTERPRETATION": (
                "Immature accounts are excluded, never counted as good: an account with no "
                "default yet is an unknown."
            ),
        },
        {
            "METRIC": "AUC and Gini",
            "FORMULA": "Gini = 2 * AUC - 1",
            "APPLIED TO": "matured accounts, ordered by risk score or ordinal grade",
            "INTERPRETATION": (
                "Discriminatory power. With grades only (no continuous score) the heavy ties "
                "make the AUC a conservative lower bound."
            ),
        },
        {
            "METRIC": "KS statistic",
            "FORMULA": "max |F_bad(s) - F_good(s)|",
            "APPLIED TO": "matured accounts ordered by increasing risk",
            "INTERPRETATION": "The largest separation between the cumulative bad and good curves.",
        },
        {
            "METRIC": "Lift",
            "FORMULA": "grade bad rate / overall bad rate",
            "APPLIED TO": "each grade, matured accounts",
            "INTERPRETATION": "How many times worse than average a grade performs.",
        },
        {
            "METRIC": "Cumulative bad capture",
            "FORMULA": "cumulative bads / total bads, riskiest grade first",
            "APPLIED TO": "each grade, matured accounts",
            "INTERPRETATION": (
                "A well-ordered scorecard captures bads faster than population: compare it with "
                "the cumulative population column on the same row."
            ),
        },
        {
            "METRIC": "Rank ordering",
            "FORMULA": "count of adjacent grade pairs whose bad rate is inverted",
            "APPLIED TO": "grades with matured outcomes",
            "INTERPRETATION": (
                "Reported with the Spearman correlation between grade rank and bad rate. Grades "
                "with no matured records are skipped rather than treated as a zero bad rate."
            ),
        },
        {
            "METRIC": "Calibration (observed vs expected)",
            "FORMULA": "ratio = observed bad rate / mean expected PD; z = (B - E[B]) / sd(B)",
            "APPLIED TO": "matured accounts where an expected PD exists",
            "INTERPRETATION": (
                "|z| <= 1.96 is within the 95% binomial tolerance. A risk grade is not a "
                "probability and is never substituted for one."
            ),
        },
    ]
    return pd.DataFrame(rows, columns=["METRIC", "FORMULA", "APPLIED TO", "INTERPRETATION"])


# ---------------------------------------------------------------------------
# Assumptions and limitations
# ---------------------------------------------------------------------------


def limitations_frame(config: MonitoringConfig) -> pd.DataFrame:
    """The documented assumptions and limitations of this pack."""
    rows = [
        ("Threshold provenance",
         "The 0.10 / 0.25 PSI and CSI bands are widely used monitoring conventions, not "
         "universal regulatory limits. They are configurable and should be set by model "
         "governance per scorecard. Compliance with them does not establish regulatory "
         "approval, and a breach does not establish non-compliance."),
        ("A breach is a trigger, not a verdict",
         "PSI/CSI measure distributional change. They cannot distinguish a deliberate strategy "
         "change from a data defect or genuine credit deterioration. Every breach requires "
         "investigation before any conclusion about the model."),
        ("Stability is not performance",
         "Segment output supports stability analysis only. Without matured outcome labels the "
         "pack cannot say whether the scorecard still separates good from bad, and it does not "
         "pretend to: those metrics are reported as NOT AVAILABLE."),
        ("Reference-period dependence",
         "Every PSI and CSI value is relative to the configured baseline. A different window "
         "gives a different number, which is why the window, mode and record count are printed "
         "beside every metric and why each scorecard keeps its own."),
        ("Quarters inside the reference window",
         "Where a monitoring quarter overlaps its own baseline, the fixed-reference PSI is "
         "mechanically close to zero. Such rows are flagged OVERLAPS_REFERENCE and must not be "
         "read as evidence of stability."),
        ("Smoothing changes small-sample values",
         f"With the configured pseudocount ({config.smoothing.pseudocount:g}) a grade with zero "
         "observations contributes a finite amount instead of an undefined one. On small "
         "populations the smoothed value differs materially from the unsmoothed one; the "
         "parameter is reported so the effect can be reproduced."),
        ("Binning drives CSI",
         "CSI depends on the bin definition. Approved scorecard boundaries are used where "
         "supplied; otherwise reference quantiles are fitted and reused. Characteristics whose "
         "movement sits in the grouped OTHER bin need an approved grouping before the driver "
         "can be identified."),
        ("Migration needs matched customers",
         f"Quarter-on-quarter migration joins on {config.customer_column} with policy "
         f"{config.runtime.multi_record_policy.value}. Where few customers appear in both "
         "periods the matrix is reported GREY rather than inferred from unmatched populations."),
        ("Right-censoring",
         f"With a {config.outcome.observation_window_months}-month performance window, recent "
         "cohorts are incomplete. Their bad rates are censored lower bounds, flagged "
         "COHORT_COMPLETE=FALSE, and are not comparable with seasoned vintages."),
        ("Incomplete quarters",
         "Partial calendar quarters are identified and excluded from the primary comparison "
         "unless explicitly enabled, because a part-quarter population is not comparable with a "
         "full one."),
        ("Duplicate records",
         "Duplicates are reported, not removed. Where they are material the population counts "
         "and distributions in this pack are inflated accordingly and should be re-run on a "
         "de-duplicated extract."),
        ("Characteristic population scope",
         f"runtime.csi_scope = {config.runtime.csi_scope}. MODEL_OUTPUT measures each "
         "scorecard's characteristics on the records it actually scored, which is the right "
         "denominator when a model covers part of the book; coverage of unscored applications "
         "is reported in MER and Data Quality instead. ALL_RECORDS would instead push unscored "
         "applications into the MISSING bin and let them move CSI directly."),
        ("Unscored volume",
         "Records with a missing or invalid grade are excluded from the PSI distribution by "
         "default and monitored separately in MER and Data Quality. Set "
         "runtime.psi_include_missing_segment=true to let unscored volume move PSI directly."),
        ("Single-run snapshot",
         "This pack reflects one extract at one point in time. Restatements, late-arriving "
         "bureau data and retrospective corrections will change the figures; the input file and "
         "row count are recorded for reproducibility."),
    ]
    return pd.DataFrame(rows, columns=["ASSUMPTION / LIMITATION", "DETAIL"])


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def methodology_blocks(inputs: Any) -> List[Block]:
    """``(title, note, frame, number_formats)`` blocks for the config sheet."""
    config: MonitoringConfig = inputs.config
    stamp = config.execution_timestamp or datetime.now()
    blocks: List[Block] = []

    # -- execution metadata ------------------------------------------
    execution = [
        ("Report title", config.report_title),
        ("Entity", config.entity_name),
        ("Framework version", config.framework_version),
        ("Engine version", __version__),
        ("Prepared by", config.prepared_by or "(not set)"),
        ("Execution timestamp", stamp.isoformat(timespec="seconds")),
        ("Configuration file", config.config_path or "(in-code configuration)"),
        ("Input file", inputs.load_result.source_path),
        ("Input format", inputs.load_result.source_format.upper()),
        ("Rows read", inputs.load_result.rows_read),
        ("Columns available in source", len(inputs.load_result.columns_available)),
        ("Columns loaded after pruning", len(inputs.load_result.columns_loaded)),
        ("In-memory footprint (MB)", round(inputs.load_result.memory_mb, 2)),
        ("Dataset date span",
         f"{inputs.validation.date_min} to {inputs.validation.date_max}"),
        ("Quarters present", len(inputs.all_quarters)),
        ("Quarters monitored", len(inputs.monitored_quarters)),
        ("Quarters excluded as incomplete",
         ", ".join(q.label for q in inputs.all_quarters if not q.is_complete) or "none"),
        ("Partial quarters enabled", str(config.runtime.include_partial_quarters)),
        ("Random seed", config.runtime.random_seed),
        ("Warnings raised", inputs.execution_log.warning_count),
        ("Errors raised", inputs.execution_log.error_count),
    ]
    blocks.append((
        "EXECUTION METADATA",
        "Reproducibility record for this run.",
        pd.DataFrame(execution, columns=["ITEM", "VALUE"]),
        None,
    ))

    # -- reference periods -------------------------------------------
    blocks.append((
        "REFERENCE PERIOD AND MODE PER SCORECARD",
        "DEVELOPMENT = the model development sample. PRODUCTION_BASELINE = a separately "
        "approved post-implementation reference population. Each scorecard is configured "
        "independently and no model's window is substituted for another's.",
        reference_summary_frame(inputs.resolutions),
        {
            "RECORDS_IN_REFERENCE": FORMAT_COUNT,
            "SCORED_RECORDS_IN_REFERENCE": FORMAT_COUNT,
            "UNIQUE_LANS": FORMAT_COUNT, "UNIQUE_CUSTOMERS": FORMAT_COUNT,
            "SHARE_OF_DATASET_PCT": FORMAT_PCT,
        },
    ))

    # -- model and column mapping ------------------------------------
    model_rows = []
    for model in config.models:
        specs = config.monitored_variables(model.name) if model.enabled else []
        roles = pd.Series([s.role.value for s in specs]).value_counts().to_dict() if specs else {}
        model_rows.append({
            "MODEL": model.name,
            "ENABLED": model.enabled,
            "DESCRIPTION": model.description,
            "OWNER": model.owner,
            "VERSION": model.version,
            "SEGMENT_COLUMN": model.segment_column,
            "SCORE_COLUMN": model.score_column or "(none)",
            "PD_COLUMN": model.pd_column or "(none)",
            "POPULATION_FILTER": model.population_filter or "(none)",
            "CHARACTERISTICS": len(specs),
            "FEATURES": roles.get("FEATURE", 0),
            "SUBSCORES": roles.get("SUBSCORE", 0),
            "SCORES": roles.get("SCORE", 0),
            "CRITICAL_CHARACTERISTICS": sum(1 for s in specs if s.critical),
            "DISCOVERY_PATTERNS": ", ".join(
                model.discovery.include_patterns
            ) if model.discovery else "(explicit mapping only)",
        })
    blocks.append((
        "SCORECARD AND COLUMN MAPPING",
        "Column mappings are configuration, never hardcoded: adding a fifth scorecard is a "
        "configuration change. Discovery patterns resolve a model's characteristics against the "
        "extract header at run time, which is what makes thousands of columns workable.",
        pd.DataFrame(model_rows),
        {"CHARACTERISTICS": FORMAT_COUNT},
    ))

    # -- common columns ----------------------------------------------
    common = [
        ("Loan account identifier", config.lan_column),
        ("Customer identifier", config.customer_column),
        ("Application date (quarter driver)", config.date_column),
        ("Outcome column", config.outcome.outcome_column or "(none - performance NOT AVAILABLE)"),
        ("Expected PD column", config.outcome.pd_column or "(none - calibration NOT AVAILABLE)"),
        ("Outcome score column", config.outcome.score_column or "(none - segment rank used)"),
        ("Vintage date column", config.outcome.vintage_date_column or config.date_column),
        ("Observation window (months)", config.outcome.observation_window_months),
        ("Immature accounts excluded", str(config.outcome.exclude_immature)),
        ("Additional columns carried", ", ".join(config.extra_columns) or "(none)"),
    ]
    blocks.append((
        "COMMON COLUMN MAPPING",
        None,
        pd.DataFrame(common, columns=["ROLE", "COLUMN"]),
        None,
    ))

    # -- segments -----------------------------------------------------
    segment_rows = []
    for segment in config.segments.segment_order:
        rank = config.segments.risk_rank[segment]
        segment_rows.append({
            "SEGMENT": segment,
            "DISPLAY_ORDER": config.segments.segment_order.index(segment) + 1,
            "RISK_RANK (1 = safest)": rank,
            "IS_RISKIER_SEGMENT": segment in config.segments.riskier_segments,
        })
    blocks.append((
        "SEGMENT ORDER AND RISK DIRECTION",
        f"risk_direction = {config.segments.risk_direction.value}: "
        f"'{config.segments.safest_segment}' is the safest grade and "
        f"'{config.segments.riskiest_segment}' the riskiest. Upgrades, downgrades, the risk "
        "colour ramp and the riskier-grade share all derive from these ranks, so reversing the "
        "direction reverses them consistently rather than inverting their meaning.",
        pd.DataFrame(segment_rows),
        None,
    ))

    # -- thresholds ---------------------------------------------------
    thresholds = config.thresholds
    threshold_rows = [
        ("PSI (all models)", thresholds.psi.label(), "Conventional industry monitoring band"),
        ("CSI (standard characteristics)", thresholds.csi.label(),
         "Conventional industry monitoring band"),
        ("CSI (business-critical characteristics)", thresholds.csi_critical.label(),
         "Tighter band for characteristics flagged critical: true"),
        ("Minimum reference records", f">= {thresholds.min_reference_records:,}",
         "Below this, PSI/CSI are reported but labelled GREY"),
        ("Minimum monitoring records", f">= {thresholds.min_monitoring_records:,}",
         "Below this, quarter metrics are labelled GREY"),
        ("Minimum records per characteristic", f">= {thresholds.min_variable_records:,}",
         "Below this, a characteristic's CSI is labelled GREY"),
        ("Missingness change vs reference", f"+/- {thresholds.missingness_increase_pp.label()}",
         "Percentage points"),
        ("Missingness level", thresholds.missingness_level_pct.label(), "Percent of records"),
        ("Unseen-category rate", thresholds.unseen_category_rate_pct.label(),
         "Percent of records in a level absent from the reference period"),
        ("Duplicate rate", thresholds.duplicate_rate_pct.label(), "Percent of rows"),
        ("Invalid segment rate", thresholds.invalid_segment_rate_pct.label(), "Percent of rows"),
        ("Missing segment rate", thresholds.missing_segment_rate_pct.label(), "Percent of rows"),
        ("Quarter-on-quarter volume change", f"+/- {thresholds.record_count_change_pct.label()}",
         "Percent change in record count"),
        ("Single-grade concentration", thresholds.concentration_pct.label(),
         "Percent of scored applications in one grade"),
        ("Net migration toward risk", thresholds.net_downgrade_pp.label(),
         "Percentage points of matched customers"),
        ("Riskier-grade share increase", thresholds.riskier_share_increase_pp.label(),
         "Percentage points vs reference"),
        ("Minimum Gini",
         f"GREEN >= {thresholds.min_gini_amber:g} > AMBER >= {thresholds.min_gini_red:g} > RED",
         "Higher is better, so the band is inverted"),
        ("Minimum matured outcomes",
         f">= {thresholds.min_outcome_records:,} records and "
         f">= {thresholds.min_bad_count:,} bads",
         "Below this, performance metrics are labelled GREY"),
    ]
    for variable, band in (thresholds.csi_variable_overrides or {}).items():
        threshold_rows.append(
            (f"CSI override: {variable}", band.label(), "Per-variable governance override")
        )
    blocks.append((
        "THRESHOLD CONFIGURATION",
        "All thresholds are configurable and may be overridden by model governance policy. "
        "Severity is always evaluated on the UNROUNDED metric value; rounding is applied to the "
        "displayed figure only.",
        pd.DataFrame(threshold_rows, columns=["THRESHOLD", "BAND", "NOTE"]),
        None,
    ))

    # -- binning rules ------------------------------------------------
    binning_rows = [
        ("Numeric - approved boundaries",
         "variables[].bins used verbatim; outer edges extended to +/-inf so out-of-range values "
         "are retained rather than becoming missing.", "Highest"),
        ("Numeric - reference quantiles",
         f"{config.runtime.default_numeric_bins} quantile bins fitted on the reference period "
         "and reused unchanged in every monitoring quarter.", "Used when no approved bins exist"),
        ("Numeric - degenerate column",
         "A (near-)constant reference distribution falls back to distinct values so the "
         "characteristic is still monitored rather than silently skipped.", "Fallback"),
        ("Categorical - approved grouping",
         "variables[].category_map maps raw values to approved (e.g. WOE) bands.", "Highest"),
        ("Categorical - approved list",
         "variables[].categories defines the bins; other reference levels group into OTHER.",
         "Second"),
        ("Categorical - reference levels",
         f"Levels with at least {config.runtime.rare_category_min_share:.1%} of the reference "
         "population get their own bin; rarer levels group into OTHER.", "Default"),
        ("MISSING bin",
         "Always present. Missing values are a monitored outcome, not discarded records.",
         "Always"),
        ("UNSEEN bin",
         "Monitoring values never seen in the reference period. Counted and alerted separately "
         "because no development bin exists for them.", "Always (categorical)"),
        ("Infinite values",
         "Clipped into the outer bins (never dropped) and reported in the data-quality sheet.",
         "Always (numeric)"),
        ("Re-bucketing per quarter",
         "NOT performed. Fitting fresh bins each quarter forces every quarter back toward the "
         "reference shape and hides the drift CSI exists to detect.", "Prohibited"),
    ]
    blocks.append((
        "BINNING RULES",
        "Bin definitions are fitted once per model on that model's own reference period. The "
        "exact fitted boundaries are retained on the 'Bin Definitions' sheet.",
        pd.DataFrame(binning_rows, columns=["RULE", "DEFINITION", "PRECEDENCE"]),
        None,
    ))

    # -- formulas -----------------------------------------------------
    blocks.append((
        "METRIC DEFINITIONS AND FORMULAS",
        None,
        formula_frame(config),
        None,
    ))

    # -- monitoring periods -------------------------------------------
    quarter_rows = []
    for quarter in inputs.all_quarters:
        quarter_rows.append({
            "QUARTER": quarter.label,
            "PERIOD_START": quarter.period_start.isoformat(),
            "PERIOD_END": quarter.period_end.isoformat(),
            "OBSERVED_FIRST": quarter.observed_start.isoformat() if quarter.observed_start else None,
            "OBSERVED_LAST": quarter.observed_end.isoformat() if quarter.observed_end else None,
            "RECORDS": quarter.record_count,
            "COMPLETE": quarter.is_complete,
            "MONITORED": any(q.label == quarter.label for q in inputs.monitored_quarters),
            "NOTE": quarter.completeness_note,
        })
    blocks.append((
        "MONITORING PERIODS",
        "Complete calendar quarters are used by default. Incomplete quarters are identified and "
        "excluded from the primary comparison unless runtime.include_partial_quarters is "
        "enabled.",
        pd.DataFrame(quarter_rows),
        {"RECORDS": FORMAT_COUNT},
    ))

    # -- runtime settings ---------------------------------------------
    runtime = config.runtime
    runtime_rows = [
        ("Smoothing method", config.smoothing.method.value),
        ("Smoothing pseudocount", config.smoothing.pseudocount),
        ("Contribution-sum tolerance", config.smoothing.tolerance),
        ("PSI includes missing/invalid grade", str(runtime.psi_include_missing_segment)),
        ("CSI quarter-on-quarter enabled", str(runtime.csi_include_qoq)),
        ("CSI population scope", runtime.csi_scope),
        ("CSI variable batch size", runtime.csi_variable_batch_size),
        ("CSI variable whitelist",
         ", ".join(runtime.csi_variable_whitelist) if runtime.csi_variable_whitelist
         else "(all monitored characteristics)"),
        ("CSI top-N for executive views", runtime.csi_top_n or config.excel.top_n),
        ("Default numeric bins", runtime.default_numeric_bins),
        ("Rare-category minimum share", f"{runtime.rare_category_min_share:.2%}"),
        ("Multi-record policy (migration)", runtime.multi_record_policy.value),
        ("Minimum records per quarter", runtime.min_quarter_records),
        ("Display decimals", runtime.display_decimals),
        ("Alert register includes GREEN", str(runtime.alert_include_green)),
        ("Non-interactive mode", str(runtime.non_interactive)),
        ("Max rows per Excel sheet", config.excel.max_rows_per_sheet),
        ("Conditional-formatting row limit", config.excel.conditional_formatting_row_limit),
    ]
    blocks.append((
        "RUNTIME CONFIGURATION",
        None,
        pd.DataFrame(runtime_rows, columns=["SETTING", "VALUE"]),
        None,
    ))

    # -- governance ---------------------------------------------------
    blocks.append((
        "ASSUMPTIONS AND LIMITATIONS",
        config.governance_note,
        limitations_frame(config),
        None,
    ))

    # -- data provenance ----------------------------------------------
    provenance = pd.DataFrame(inputs.load_result.as_rows(), columns=["ITEM", "VALUE"])
    validation_rows = [
        {"ITEM": "Validation result",
         "VALUE": "PASSED" if inputs.validation.ok else "FAILED"},
        {"ITEM": "Models with usable output",
         "VALUE": ", ".join(inputs.validation.models_with_output) or "none"},
        {"ITEM": "Models without usable output",
         "VALUE": ", ".join(inputs.validation.models_without_output) or "none"},
    ]
    for warning in inputs.validation.warnings:
        validation_rows.append({"ITEM": "Validation warning", "VALUE": warning})
    blocks.append((
        "INPUT PROVENANCE AND VALIDATION",
        None,
        pd.concat([provenance, pd.DataFrame(validation_rows)], ignore_index=True),
        None,
    ))
    return blocks


__all__ = ["methodology_blocks", "formula_frame", "limitations_frame"]
