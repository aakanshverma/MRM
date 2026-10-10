# Sample monitoring pack

`Sample_Scorecard_Monitoring_Report.xlsx` is a complete, generated monitoring
pack. It was produced by:

```bash
python -m model_monitoring.synthetic_data --rows 25000
python main.py --config config/config.sample.yaml --non-interactive \
    --output output/sample --filename Sample_Scorecard_Monitoring_Report.xlsx
```

`effective_config.json` is the fully resolved configuration for that run
(`--save-effective-config`), and `monitoring_run.log` is its log.

The underlying data is synthetic. It is not a real portfolio, and the numbers
are chosen to exercise the pipeline rather than to resemble any particular
book.

## What the pack shows

| Scorecard | Reference period | Mode | Latest PSI | Status |
|---|---|---|---|---|
| MODEL_1 | 2024-01-01 → 2024-06-30 | DEVELOPMENT | 1.3030 | **RED** — a large, sustained population shift |
| MODEL_2 | 2024-01-01 → 2024-12-31 | DEVELOPMENT | 0.0009 | GREEN — stable |
| MODEL_3 | 2024-07-01 → 2024-12-31 | PRODUCTION_BASELINE | 0.4409 | **RED** — driven by a grade that stopped being emitted |
| MODEL_4 | 2025-07-01 → 2025-09-30 | DEVELOPMENT | 0.0172 | GREY — too little data to conclude |

Four different windows and two different reference modes, which is the point:
each scorecard is monitored against its own approved baseline.

25,300 records across 11 quarters (2024Q1 – 2026Q3). The final quarter is
deliberately partial, so it is flagged incomplete and excluded from the primary
comparison.

## Things worth opening the file for

**`01 Exec Dashboard`** — the head-line status, the alert counts, the
per-scorecard snapshot with each model's own reference period, the top five
concerns and the priority investigations, plus nine charts. Fixed-reference PSI
and quarter-on-quarter PSI are charted separately and labelled, because they
answer different questions.

**`04 PSI Detail`** — MODEL_3's `CAT E` row for 2026Q2 shows a reference count
of 253 against a monitoring count of **0**. The grade still takes part in the
sum: the documented pseudocount rule gives it a monitoring proportion of
0.00022 and it contributes 0.402 of the total PSI of 0.441. That empty grade *is*
the finding, which is exactly why zero-count grades are never dropped.

**`05 CSI Summary`** — `MODEL_1_VARIABLE_3` shows missingness moving +26.5
percentage points against its reference period. The recommended action says to
resolve that before attributing the movement to population change, because a
feature-pipeline break and a change in customer mix look identical in the index
and need opposite responses.

**`06 CSI Bin Detail`** — the `MISSING` and `UNSEEN` bins carry real volume.
`MODEL_3_VARIABLE_2` picks up a `GIG_ECONOMY` level that never existed in the
reference period, so no development bin exists for those applications at all.

**`07 Segment Migration`** — 6×6 matrices with row percentages and counts side
by side, heat-mapped. Quarters that overlap their own baseline are greyed out:
there, "matched customers" would largely be records matched against themselves.

**`09 Outcome Performance`** — 2026Q2 reports `MATURED_RECORDS = 0` and
`NOT AVAILABLE`, not a flattering 0% bad rate. With a 12-month window those
applications cannot yet be classified, and the pack says so rather than
counting them as good.

**`10 Config & Methodology`** — every formula, the smoothing parameter, all
thresholds, the binning precedence rules, the monitoring periods and the full
list of assumptions and limitations.

**`13 Bin Definitions`** — the exact fitted boundaries behind every CSI figure,
so any number in the pack can be reproduced or challenged.

**`14 Test Scenarios`** — the twelve conditions the synthetic generator plants,
and where each should surface. Useful as a checklist when verifying a change.

## The off-us variant

`output/offus/` is the same four scorecards and the same stability analysis,
but with performance measured against a customer-level **off-us 30+ DPD** flag
joined on from a separate bureau-style file. PSI, CSI, migration and data
quality are identical in both packs; only the performance section differs,
which is the intended separation. See
[`output/offus/README.md`](../offus/README.md).

## Regenerating it

```bash
python main.py --config config/config.sample.yaml --non-interactive \
    --output output/sample --filename Sample_Scorecard_Monitoring_Report.xlsx
```

The synthetic generator is seeded, so the same command reproduces the same pack
(the execution timestamp aside).
