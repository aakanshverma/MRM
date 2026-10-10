# Sample pack: off-us 30+ DPD outcome

`Sample_OffUs_30plus_DPD_Report.xlsx` is the same four scorecards and the same
stability analysis as `output/sample/`, but with performance measured against a
**customer-level off-us 30+ DPD flag** joined on from a separate bureau-style
file.

```bash
python -m model_monitoring.synthetic_data --rows 25000
python main.py --config config/config.offus_30plus.yaml --non-interactive \
    --output output/offus --filename Sample_OffUs_30plus_DPD_Report.xlsx
```

The data is synthetic. Treat the numbers as an illustration of the mechanics,
not of any real portfolio.

## What the outcome configuration says

| Setting | Value |
|---|---|
| Definition | Off-us 30+ DPD ever within 12 months of login (bureau trade lines) |
| Basis | `OFF_US` — other lenders' accounts, not our own book |
| Supplied at | `CUSTOMER` level, joined on `CUSTOMER_ID` from `data/offus_30plus_flag.csv` |
| Metrics measured on | one record per customer |
| Indeterminate (excluded) | `INDET_1_29_DPD`, `NO_HIT`, `THIN_FILE`, `NOT_FOUND` |
| Window | 12 months, as of the bureau pull date |
| Expected PD | none — calibration correctly reports `NOT AVAILABLE` |
| Gini floors | 0.22 AMBER / 0.14 RED, set for *this* definition |

All of it is printed on the `09 Outcome Performance` sheet above the metrics, so
a reader cannot mistake an off-us proxy for on-book default performance.

## The finding this unlocks

Mean Gini against the off-us 30+ DPD target, across matured quarters:

| Scorecard | Quarters | Mean Gini | Worst | Verdict |
|---|---|---|---|---|
| MODEL_1 | 7 | 0.187 | 0.119 | Ranks risk on this target — mostly AMBER against a 0.22 floor |
| MODEL_2 | 7 | 0.047 | 0.018 | **RED in every quarter** — essentially no discrimination |
| MODEL_3 | 7 | 0.034 | −0.006 | **RED in every quarter**, and inverted in one |
| MODEL_4 | 1 | 0.113 | 0.113 | RED, on one thin quarter |

That is the question an off-us flag answers and segment distributions cannot:
**which of these scorecards actually rank-orders risk?** MODEL_2 and MODEL_3
have stable populations — MODEL_2's PSI is 0.0009 — and yet separate good from
bad barely at all on this target. Stability and performance are different
questions, and this pack shows a case where they disagree completely.

In this synthetic data that gap is by construction. On a real extract it is the
first thing to look at.

## Mechanics worth checking on the sheets

**`09 Outcome Performance`** — the definition banner, then per quarter:
`APPLICATIONS_BEFORE_DEDUPE` against `UNITS_IN_METRIC` (customer-level
de-duplication actually biting), `INDETERMINATE_EXCLUDED` (1–29 DPD and no-hit
customers kept out of the good population), and `PERFORMANCE_STATUS` carrying
the off-us proxy caveat on every row.

**Bad rate by segment** — ordered riskiest grade first, with `RANK_ORDER_BREAK`
marking grades whose bad rate is higher than the riskier grade above them.
MODEL_2 and MODEL_3 break repeatedly; MODEL_1 mostly does not.

**`12 Execution Log`** — the join diagnostics: match rate, duplicate-key
handling, and the indeterminate exclusion count.

**Calibration** — reported `NOT AVAILABLE` throughout, because no expected PD
matching this target exists. A PD calibrated to on-us default is not an
estimate of other lenders' delinquency, so the pack declines to compare them.

## Compared with the on-us pack

`output/sample/` runs the identical stability analysis with an on-us
`DEFAULT_FLAG`. PSI, CSI, migration and data quality are the same in both; only
the performance section differs. That is the intended separation: changing the
outcome definition changes what the pack can say about performance and nothing
about stability.
