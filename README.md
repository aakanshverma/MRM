# Credit Risk Scorecard Monitoring Pipeline

A production-shaped, configuration-driven Python framework that turns a
customer-level extract of scorecard outputs into a quarterly **model monitoring
pack**: MER metrics, PSI, CSI, segment migration, data-quality controls,
outcome performance and a formatted Excel workbook with an executive dashboard.

Built for an Indian NBFC setting where several scorecards run side by side,
each with its **own** development or production baseline.

---

## What it produces

A single Excel workbook per execution:

| Sheet | Contents |
|---|---|
| `01 Exec Dashboard` | Status, KPI block, per-scorecard snapshot, top five concerns, priority investigations, charts |
| `02 MER Summary` | One row per model-quarter: volumes, coverage, segment mix, PSI, CSI counts, performance |
| `03 Segment Distribution` | Model-quarter-segment mix vs the reference period, with risk-band text and heatmap |
| `04 PSI Detail` | Per-segment PSI contributions plus the PSI summary |
| `05 CSI Summary` | Every model-quarter-characteristic combination, sorted by severity |
| `06 CSI Bin Detail` | Reference vs monitoring distribution for every bin |
| `07 Segment Migration` | Matched-customer migration, 6x6 matrices and heatmaps |
| `08 Data Quality` | Missingness, duplication, date integrity, outliers, coverage, freshness |
| `09 Outcome Performance` | Bad rates, Gini/KS, lift, rank ordering, calibration, vintages — or the named missing dependencies |
| `10 Config & Methodology` | Mappings, reference periods, formulas, smoothing, thresholds, binning rules, limitations |
| `11 Alert Register` | The unified alert table with investigation status and owner notes |
| `12 Execution Log` | Warnings and exceptions raised during the run |
| `13 Bin Definitions` | The fitted reference bin boundaries (audit evidence) |
| `14 Test Scenarios` | Planted-scenario manifest (synthetic runs only) |

A ready-made example is committed at
[`output/sample/`](output/sample/) — see [`output/sample/README.md`](output/sample/README.md).

---

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.9+. `pandas`, `numpy` and `openpyxl` are required; `scikit-learn`,
`matplotlib`, `pyarrow` and `PyYAML` are optional and the pipeline degrades
gracefully without them (AUC falls back to a NumPy implementation, YAML
configuration falls back to JSON, Parquet support is skipped).

## Quick start

```bash
# 1. generate a synthetic extract and monitor it end to end
python main.py --generate-synthetic 25000 --config config/config.sample.yaml --non-interactive

# 2. or point it at your own extract and answer the reference-period prompts
python main.py --data data/my_extract.csv

# 3. run the tests
python -m pytest -q
```

`python main.py` with no arguments picks up `config/config.yaml` (or the sample
config), loads the configured dataset and prompts for any reference period that
is not already configured.

---

## The one rule that shapes everything

**Each scorecard is monitored against its own approved baseline, and the
pipeline will never invent one.**

Four scorecards developed, implemented and retrained at different times do not
share a reference population. So the pipeline refuses to:

* use the whole dataset as a baseline by default,
* reuse one model's window for another model,
* proceed past a missing reference period without being told what to use.

```text
Reference period(s) missing for: MODEL_3. The pipeline will not default to the
full dataset. Supply them in the configuration file
(models.<MODEL>.reference.start_date / end_date), pass --reference
MODEL=START:END on the command line, or run interactively.
```

Three ways to supply them:

```yaml
# 1. configuration file (reproducible, preferred)
models:
  MODEL_1:
    reference:
      mode: DEVELOPMENT            # or PRODUCTION_BASELINE
      start_date: "2024-01-01"
      end_date: "2024-06-30"
      source_note: "Signed off by MRMC on 2024-08-14"
```

```bash
# 2. command line
python main.py --reference MODEL_1=2024-01-01:2024-06-30:DEVELOPMENT \
               --reference MODEL_2=2024-07-01:2024-12-31:PRODUCTION_BASELINE
```

```text
# 3. interactive prompts, with validation and record counts
MODEL_1:
  segment column : MODEL_1_SEGMENT (24,811 record(s) with a valid segment)
  Reference mode [DEVELOPMENT/PRODUCTION_BASELINE] (default DEVELOPMENT):
  Start date (YYYY-MM-DD): 2024-01-01
  End date   (YYYY-MM-DD): 2024-06-30
  -> 4,641 record(s) in 2024-01-01 to 2024-06-30 (18.3% of the dataset);
     4,554 with a valid MODEL_1 segment.
```

Prompts validate that the dates parse, that `start <= end`, that the window
overlaps the available data, and that the resulting sample is large enough —
a window below `thresholds.min_reference_records` must be confirmed explicitly.

The chosen window, mode, record count and provenance note are printed on the
dashboard and the methodology sheet, beside every metric that depends on them.

---

## Input data

A customer-level DataFrame (CSV, Excel, Parquet or an in-memory frame).

| Column | Role |
|---|---|
| `LAN_ID` | Loan account / application identifier |
| `CUSTOMER_ID` | Customer identifier (used for migration matching) |
| `LOGIN_DATE` | Application date — drives the monitoring quarters |
| `MODEL_i_SEGMENT` | That scorecard's risk grade |
| `MODEL_i_TOTAL_SCORE` | Optional final score |
| `MODEL_i_VARIABLE_j` | Model inputs |
| `MODEL_i_VARIABLE_j_SUBSCORE` | Optional variable-level points |
| `DEFAULT_FLAG`, `EXPECTED_PD`, … | Optional outcome fields |

Column names are **configuration, not convention**. Either list the
characteristics explicitly or let discovery resolve them against the header:

```yaml
MODEL_2:
  segment_column: MODEL_2_SEGMENT
  discovery:
    include_patterns: ["^MODEL_2_VARIABLE_"]
    exclude_patterns: ["_WOE$"]
    subscore_suffix: "_SUBSCORE"
```

Discovery is what makes extracts with thousands of columns workable: the header
is read first, patterns are resolved against it, and only the resulting columns
are loaded into memory.

### Risk grades

```yaml
segments:
  segment_order: ["CAT A", "CAT B", "CAT C1", "CAT C2", "CAT D", "CAT E"]
  risk_direction: A_TO_E_INCREASING    # CAT A safest ... CAT E riskiest
  riskier_segment_count: 2
```

`risk_direction: A_TO_E_DECREASING` flips it, and everything downstream follows:
risk ranks, upgrade/downgrade classification, the riskier-grade share and the
Excel colour ramp. A reversed convention produces reversed colours, never
colours whose meaning is inverted.

---

## Metrics

Full formulas and interpretation: [`docs/METRICS.md`](docs/METRICS.md).

```text
PSI   = SUM_i  (A_i  - E_i ) * ln(A_i  / E_i )     over risk grades
CSI_j = SUM_i  (A_ij - E_ij) * ln(A_ij / E_ij)     over a characteristic's bins
```

* All configured grades take part in every comparison, **including grades with
  zero observations**.
* Zero frequencies are handled by a documented, configurable rule — additive
  (pseudocount) smoothing by default, `p_i = (n_i + 0.5) / (N + k * 0.5)` — with
  the parameter printed in the report. No silent substitution of arbitrary
  floors.
* Both a **fixed-reference** PSI and a **quarter-on-quarter** PSI are produced,
  charted separately, because they answer different questions.
* Bin-level contributions are reported and asserted to sum to the total within
  `smoothing.tolerance`.
* Thresholds are evaluated on the **unrounded** value; rounding is display only.

Default bands (conventional industry guidelines, fully configurable):

| Index | GREEN | AMBER | RED |
|---|---|---|---|
| PSI | < 0.10 | 0.10 – 0.25 | >= 0.25 |
| CSI | < 0.10 | 0.10 – 0.25 | >= 0.25 |
| CSI (business-critical) | < 0.05 | 0.05 – 0.15 | >= 0.15 |

### Outcome flags: on-us vs off-us

An outcome flag unlocks the performance half of the pack — bad rate by segment,
rank ordering, Gini/KS, lift, cumulative bad capture. Three fields decide
whether those numbers mean what a reader will assume:

```yaml
outcome:
  outcome_column: OFFUS_30P_DPD
  definition_label: "Off-us 30+ DPD ever within 12 months of login (bureau)"
  definition_basis: OFF_US        # ON_US | OFF_US | COMBINED
  outcome_level: CUSTOMER         # one flag per customer, not per application
  indeterminate_values: [INDET_1_29_DPD, NO_HIT, THIN_FILE]
  source:                         # the flag arrives as its own bureau file
    path: data/offus_30plus_flag.csv
    key_column: CUSTOMER_ID
```

A worked template is at [`config/config.offus_30plus.yaml`](config/config.offus_30plus.yaml):

```bash
python main.py --config config/config.offus_30plus.yaml --non-interactive
```

**Off-us is a proxy with a real advantage.** Bureau-observed delinquency on
other lenders' accounts covers the full through-the-door population, declines
included, so rank ordering measured on it is free of the accept-only selection
bias that limits on-book monitoring. But it is almost never the target the
scorecard was developed on, so its Gini and KS are proxy measures — not
comparable with development figures, and needing their own thresholds (a 30+
off-us target is milder than on-us 90+ default and typically scores lower).
The caveat travels with every figure that depends on it.

**The indeterminate band matters more than it looks.** 1–29 DPD is delinquent
but not bad; a customer with no bureau trade line is unknown, not current. Both
are excluded and counted, never merged into the good population. If a no-hit
code is encoded as `0` and `0` sits in `good_values`, those customers silently
become good — the bad rate is understated and every discrimination metric is
flattered. The pipeline warns when an off-us outcome declares no indeterminate
values, but it cannot spot a no-hit code disguised as a zero.

**Customer-level flags are de-duplicated.** One flag repeated across a
customer's three applications is one observation, not three: leaving the copies
in inflates the effective sample behind AUC and KS and over-weights repeat
applicants. Discrimination is computed on one record per customer (via
`runtime.multi_record_policy`), with both counts reported. Stability metrics are
unaffected.

### Stability is not performance

The framework keeps two questions apart and says so on the sheets:

* **Stability** — is the population or an input changing? PSI, CSI, migration.
  Answerable from scorecard output alone.
* **Predictive performance** — does the scorecard still separate good from bad?
  Gini, KS, lift, rank ordering, calibration. Answerable *only* with matured
  outcome labels.

With no outcome column, every performance metric is reported as
`NOT AVAILABLE` with its missing dependency named. Nothing is estimated from the
segment mix.

### Binning

Bin boundaries are fitted **once** on each model's own reference period and
reused unchanged in every quarter. Re-bucketing per quarter would force each
quarter back toward the reference shape and hide the very drift CSI exists to
detect — `tests/test_csi.py` asserts this explicitly.

Precedence: approved scorecard bins / WOE bands → reference quantiles →
distinct-value fallback for degenerate columns. `MISSING`, `OTHER` and `UNSEEN`
are explicit, always-present bins, because a characteristic whose missingness
jumps from 2% to 30% is a production incident that must be visible.

---

## Alerts

One register, one schema, four labels:

| Severity | Meaning |
|---|---|
| `GREEN` | Within normal monitoring limits |
| `AMBER` | Investigation required |
| `RED` | Significant breach requiring escalation |
| `GREY` | Insufficient data, or the metric could not be calculated |

Two rules matter as much as the thresholds:

* **No misleading alerts.** A metric computed on too little data is `GREY` with
  the sample size stated — never `RED` (false alarm) and never `GREEN` (false
  comfort).
* **A breach triggers investigation, not rejection.** Every AMBER/RED row
  carries a concrete next step; none recommends retiring a model.

---

## Project structure

```
model_monitoring/
├── main.py                 orchestration + CLI
├── config.py               every business rule, as dataclasses
├── logging_config.py       logging + the in-report execution log
├── data_loader.py          column-pruned, chunked, dtype-optimised loading
├── data_validation.py      schema checks, frame preparation, data quality
├── reference_periods.py    per-model baselines, prompts, quarter selection
├── binning.py              reference-period bin fitting
├── psi_calculator.py       PSI + the shared divergence engine
├── csi_calculator.py       CSI over characteristics, subscores and scores
├── mer_metrics.py          MER + segment distribution + model summary
├── segment_migration.py    matched-customer migration matrices
├── performance_metrics.py  AUC/Gini/KS/lift/rank order/calibration/vintage
├── alert_engine.py         the unified alert register
├── excel_report.py         workbook assembly
├── excel_style.py          theme, table writer, conditional formatting
├── methodology.py          methodology & limitations content
├── synthetic_data.py       scenario-planting data generator
└── utils.py                quarters, safe maths, dtype probing

config/     config.sample.yaml, config.sample.json, config.offus_30plus.yaml
docs/       METRICS.md, ASSUMPTIONS_AND_LIMITATIONS.md, TROUBLESHOOTING.md
tests/      243 tests across 10 modules
output/     generated workbooks; output/sample/ holds a committed example
main.py     root entry point, so `python main.py` works
```

Business logic and Excel formatting are separate throughout: the analytical
modules return DataFrames and never import `openpyxl`.

---

## Command line

```
--config PATH                   YAML or JSON configuration
--data PATH                     override the input path
--reference MODEL=START:END[:MODE]   repeatable reference override
--models MODEL_1,MODEL_2        monitor a subset
--variables VAR_A,VAR_B         CSI whitelist (partial run on wide extracts)
--csi-top-n N                   top-N characteristics in executive views
--csi-qoq                       also compute quarter-on-quarter CSI
--include-partial-quarters      monitor incomplete quarters too
--chunksize N                   stream the CSV in chunks
--non-interactive               fail instead of prompting
--generate-synthetic [ROWS]     generate and monitor a synthetic extract
--save-effective-config PATH    write the resolved configuration
--output DIR / --filename NAME  where the workbook goes
--no-charts / --quiet / --log-level
```

## Execution flow

1. load the dataset, pruning to the configured columns
2. validate the schema and the scorecard output
3. ask for each model's reference period, unless configured
4. display the selected record counts per scorecard
5. prepare each scorecard independently
6. quarterly MER metrics
7. fixed-reference and quarter-on-quarter PSI
8. CSI for every monitored characteristic
9. segment migration
10. data-quality and outcome-performance checks
11. threshold-based alerts
12. export the Excel workbook
13. print the output path and an execution summary

---

## Scale and memory

Designed for wide extracts and tens of thousands of records:

* **Column pruning** — the header is read first, so only required columns load.
* **Chunked reads** — `--chunksize` streams large CSVs.
* **Batched CSI** — characteristics are processed in configurable batches, one
  column slice at a time, with explicit garbage collection.
* **Whitelisting** — `--variables` runs a subset, for a staged pass over
  thousands of characteristics.
* **Bounded formatting** — per-cell styling up to a configurable row limit,
  column-level formats above it, so a 200,000-row sheet still opens.
* **Explicit truncation** — if a sheet hits its row cap, the full table is
  written to a CSV sidecar and the sheet carries an unmissable banner. Data is
  never silently dropped.

If the pruned dataset still does not fit, the loader says so with the measured
footprint and the concrete remedies rather than downsampling. See
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

**Measured** on this machine, a 40,000-row Parquet extract with 1,005 columns
(two scorecards, 600 monitored characteristics between them, 7 quarters):

| | |
|---|---|
| Columns pruned at load | 400 of 1,005 |
| Pruned frame in memory | 79 MB |
| CSI combinations computed | 4,200 (42,000 bin-detail rows) |
| End-to-end runtime | 48 s |
| Workbook size | 6.4 MB |

The 25,000-row, four-scorecard sample pack in `output/sample/` takes about 7
seconds.

## Adding a fifth scorecard

Configuration only — no code changes:

```yaml
models:
  MODEL_5:
    segment_column: MODEL_5_GRADE          # any naming convention
    score_column: MODEL_5_SCORE
    description: "Co-lending partner scorecard"
    reference:
      mode: PRODUCTION_BASELINE
      start_date: "2025-04-01"
      end_date: "2025-09-30"
    discovery:
      include_patterns: ["^MODEL_5_FEAT_"]
```

## Testing

```bash
python -m pytest -q                    # 243 tests
python -m pytest tests/test_psi.py -v  # one module
```

The suite validates PSI and CSI against hand calculations, asserts the
properties the brief requires (proportions summing to one, zero index on
identical distributions, non-negativity, all six grades present, unrounded
threshold evaluation, model-specific reference periods, correct migration
denominators, stability monitoring surviving a missing outcome, expected Excel
sheets and columns), and runs the CLI end to end.

The synthetic generator plants thirteen specific conditions — a stable model, a
drifting model, a zero-count grade, missing and invalid grades, missing
features, an unseen category, four different reference windows, an incomplete
quarter, a high-CSI characteristic, customer upgrades and downgrades,
duplicates, a scorecard with too little data, and a customer-level off-us 30+
DPD outcome delivered as a separate bureau file with an indeterminate band — so
a run exercises the real code paths rather than a happy path.

## Documentation

* [`docs/METRICS.md`](docs/METRICS.md) — every metric, its formula and how to read it
* [`docs/ASSUMPTIONS_AND_LIMITATIONS.md`](docs/ASSUMPTIONS_AND_LIMITATIONS.md) — what this pack does and does not establish
* [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — memory limits and malformed input

## Governance note

PSI/CSI thresholds here are **widely used monitoring conventions, not universal
regulatory requirements**. They are configurable and should be set by model
governance per scorecard. The pack is designed to support model-risk governance;
meeting generic thresholds does not by itself establish regulatory compliance,
and breaching one does not by itself invalidate a scorecard.
