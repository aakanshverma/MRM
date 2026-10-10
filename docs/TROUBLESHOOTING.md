# Troubleshooting

Covers the two failure families that matter in practice: **memory limits on
large extracts** and **malformed input data**. Each entry gives the symptom, the
cause, and the fix.

---

## Memory and performance

### Symptom: `MemoryError`, or the process is killed

The pipeline prints the remedies on exit. In order of effectiveness, none of
which drops data:

**1. Convert the extract to Parquet.** The single biggest win on a wide
extract, because only the required columns are read from disk rather than
parsed and then discarded.

```bash
python -c "import pandas as pd; pd.read_csv('extract.csv').to_parquet('extract.parquet')"
python main.py --data extract.parquet
```

**2. Stream the CSV.** Reads in chunks, optimising dtypes per chunk.

```bash
python main.py --chunksize 250000
```

**3. Monitor fewer characteristics per pass.** With thousands of columns, run a
staged pass. Nothing is lost: each run produces a complete pack for its subset.

```bash
python main.py --variables MODEL_1_VARIABLE_1,MODEL_1_VARIABLE_2,MODEL_1_VARIABLE_3
```

**4. Monitor one scorecard per run.**

```bash
python main.py --models MODEL_1
```

**5. Make sure column pruning is on** (it is by default). Confirm it is working
in the log line after loading:

```
Loaded 250,000 rows x 180 columns (420.1 MB in memory; 2,820 of 3,000 source
columns pruned) from CSV 'extract.csv'
```

If `columns pruned` is 0, either `data_source.prune_columns` is false or the
configuration is selecting every column — check the discovery patterns.

### Symptom: advisory warning about the in-memory footprint

```
The pruned dataset occupies 2,400 MB in memory, above the 2,048 MB advisory
threshold...
```

Informational; the run continues. Raise the threshold with
`MRM_MEMORY_WARN_MB=4096` or apply the strategies above.

### Symptom: the run is slow

| Cause | Fix |
|---|---|
| Thousands of characteristics × many quarters | `--variables` for a staged pass; raise `runtime.csi_variable_batch_size` to trade memory for speed |
| Quarter-on-quarter CSI enabled | set `runtime.csi_include_qoq: false` — it doubles the CSI workload |
| Conditional formatting on huge sheets | lower `excel.conditional_formatting_row_limit` |
| Excel tables on huge sheets | lower `excel.excel_table_row_limit` or set `excel.use_excel_tables: false` |
| Charts | `--no-charts` |

### Symptom: the workbook is very large or slow to open

Lower `excel.max_rows_per_sheet`. Overflow is written to a CSV sidecar beside
the workbook and the sheet carries an explicit banner, so nothing is lost:

```
NOTE: this sheet shows the first 200,000 of 1,350,000 rows (Excel row budget).
No data has been dropped from the analysis; the complete table was written to
'output/csi_bins_20260310_141522.csv'.
```

Excel's own hard limit is 1,048,576 rows per sheet.

---

## Malformed input data

### `DataLoadError: Input file not found`

Check `data_source.path`, or generate the synthetic extract to confirm the
install works:

```bash
python -m model_monitoring.synthetic_data --rows 25000
python main.py --config config/config.sample.yaml --non-interactive
```

### `DataLoadError: The dataset read from ... is empty`

Usually the delimiter. A semicolon- or tab-separated file needs:

```yaml
data_source:
  delimiter: ";"
```

### `ValidationError: Mandatory date_column 'LOGIN_DATE' is missing`

Quarterly monitoring is impossible without an application date. Map the real
column:

```yaml
date_column: APPLICATION_DATE
```

### `ValidationError: No parseable values in 'LOGIN_DATE'`

The column exists but nothing in it parses as a date. Common causes: Excel
serial numbers (`45292`), a locale format pandas cannot infer, or a text
placeholder such as `N/A`. Convert at source, or pre-process:

```python
frame["LOGIN_DATE"] = pd.to_datetime(frame["LOGIN_DATE"], format="%d-%m-%Y", errors="coerce")
```

Partially unparseable dates are not fatal: those records are reported in DQ-010
and excluded from quarter assignment.

### `ValidationError: None of the configured scorecards has usable segment output`

The segment columns are missing or entirely empty. Check the mapping against the
real header:

```python
from model_monitoring.config import load_config
from model_monitoring.data_loader import peek_columns
print(peek_columns(load_config("config/config.sample.yaml").data_source))
```

### `ReferencePeriodError: Reference period(s) missing for: MODEL_3`

By design — the pipeline will not invent a baseline. Supply it in the config, on
the command line, or interactively:

```bash
python main.py --reference MODEL_3=2024-07-01:2024-12-31:PRODUCTION_BASELINE
```

### `ReferencePeriodError: Every quarter in the extract is incomplete`

The extract covers less than one full calendar quarter, or starts and ends
mid-quarter. Either extend it or opt in explicitly:

```bash
python main.py --include-partial-quarters
```

### `ReferencePeriodError: ... baseline_filter could not be evaluated`

The filter is a pandas `query` string and must reference columns that survive
pruning. Add them to `extra_columns`:

```yaml
extra_columns: [PRODUCT, CHANNEL]
models:
  MODEL_1:
    reference:
      baseline_filter: "PRODUCT == 'PERSONAL_LOAN'"
```

### `ConfigError: Unknown configuration keys: [...]`

A typo or an obsolete key. The error lists the allowed keys; `config/config.sample.yaml`
documents every option.

### `ConfigError: PyYAML is required for YAML configuration files`

```bash
python -m pip install PyYAML
# or use the JSON configuration instead
python main.py --config config/config.sample.json
```

---

## Results that look wrong

### PSI is suspiciously close to zero

Check `OVERLAPS_REFERENCE` on the PSI sheet. A quarter inside its own baseline
window is being compared against itself. Compare later quarters, or use the
quarter-on-quarter PSI.

### Every metric is GREY

A sample-size gate is biting. Check the reference adequacy alerts and the
`STATUS_REASON` column, then either widen the baseline or lower the gates for a
genuinely small portfolio:

```yaml
thresholds:
  min_reference_records: 200
  min_monitoring_records: 50
  min_variable_records: 30
```

### CSI is enormous for one characteristic

Look at `TOP_CONTRIBUTING_BIN` on the CSI summary:

| Top bin | Likely cause |
|---|---|
| `MISSING` | feature-pipeline break — check `MISSINGNESS_CHANGE_PP` first |
| `UNSEEN` | re-coded categories, or a new product/channel |
| `OTHER` | the grouping rule is hiding the driver; supply `category_map` |
| a real bin | genuine population movement |

A missingness jump must be resolved before the movement is attributed to
population change.

### Migration shows no matched customers

Expected when applicants rarely re-apply. Confirm `CUSTOMER_ID` is populated and
is genuinely a customer key rather than a per-application identifier; check
`MATCH_RATE_PCT`. Population-level change is covered by PSI and the segment
distribution sheet instead.

### All performance metrics are NOT AVAILABLE

Either no `outcome.outcome_column` is configured, or no cohort has matured. The
`09 Outcome Performance` sheet names the missing dependency for each metric.
Check `MATURITY_PCT`: with a 12-month window, applications from the last four
quarters cannot have matured.

### Bad rates look too low across the board (off-us flag)

Almost always the no-information population being counted as good. Check
`INDETERMINATE_EXCLUDED` on the `09 Outcome Performance` sheet: if it is zero
while the bureau flag has a thin-file or no-hit code, those customers are
sitting in `good_values`. Declare them:

```yaml
outcome:
  indeterminate_values: [NO_HIT, THIN_FILE, NOT_FOUND, INDET_1_29_DPD]
```

The pipeline warns when an `OFF_US` outcome declares no indeterminate values,
but it cannot spot a no-hit population encoded as `0`.

### Gini looks poor on an off-us target

Expected to a degree, and usually not a broken model. An off-us 30+ DPD target
is a milder bad definition measured on a different population than on-us 90+
default, so it yields a lower Gini. Reusing on-us floors will flag healthy
scorecards. Set them for the definition in use:

```yaml
thresholds:
  min_gini_amber: 0.22
  min_gini_red: 0.14
```

`config/config.offus_30plus.yaml` is a worked template.

### The outcome join matched almost nothing

The run warns when the match rate is below 50%. Identifier formats are the
usual cause — leading zeros stripped by a CSV round-trip, a prefix on one side,
or an integer/string mismatch:

```python
import pandas as pd
left = pd.read_csv("extract.csv", dtype={"CUSTOMER_ID": "string"})
right = pd.read_csv("offus_flag.csv", dtype={"CUSTOMER_ID": "string"})
print(left.CUSTOMER_ID.head().tolist(), right.CUSTOMER_ID.head().tolist())
print("overlap:", len(set(left.CUSTOMER_ID) & set(right.CUSTOMER_ID)))
```

Unmatched applications keep a null flag and leave the performance metrics; they
are never dropped from the stability analysis and never counted as good.

### "Truth value of a DataFrame is ambiguous" during a merge

Something has put a DataFrame into `DataFrame.attrs`. Pandas compares `attrs`
for equality when finalising a merge or concat, so any later join raises this.
Pass frames explicitly instead of smuggling them through `attrs`.

### Bad rates look too good in recent quarters

Right-censoring. Check `COHORT_COMPLETE` — a false value means the bad rate is a
lower bound that will rise as the cohort seasons. Recent and seasoned vintages
are not comparable.

### A characteristic is missing from the CSI sheets

Check the `13 Bin Definitions` sheet: `USABLE = FALSE` with a reason means no
boundaries could be fitted (usually an entirely empty reference period). Also
check `runtime.csi_variable_whitelist` and that the column name matches the
extract exactly — absent columns are logged as warnings at load time and
appear in the `12 Execution Log` sheet.

---

## Diagnostics

Run the pipeline in-process and inspect the tables directly:

```python
from model_monitoring.config import load_config
from model_monitoring.main import run_pipeline

config = load_config("config/config.sample.yaml")
config.runtime.non_interactive = True
result = run_pipeline(config, write_excel=False)

inputs = result.report_inputs
print(inputs.psi.summary.head())
print(inputs.csi.summary.nlargest(10, "CSI"))
print(inputs.data_quality[inputs.data_quality.SEVERITY == "RED"])
for entry in inputs.execution_log.to_rows():
    print(entry)
```

Useful switches:

```bash
python main.py --log-level DEBUG                       # verbose logging
python main.py --save-effective-config out/eff.json    # the resolved config
python main.py --generate-synthetic 5000 --output /tmp # known-good smoke test
python -m pytest -q                                    # 243 tests
```

Every run also writes `output/monitoring_run.log` and surfaces warnings on the
`12 Execution Log` sheet of the workbook.
