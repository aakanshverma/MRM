# Metric reference

Every metric the pipeline produces, its formula, what it is computed on, and how
to read it. The same content is written into the workbook's
`10 Config & Methodology` sheet, generated from the *effective* configuration of
the run so it always describes what actually happened.

Notation: `E` = expected (reference) proportion, `A` = actual (monitoring)
proportion, `N` = population size, `k` = number of bins.

---

## 1. Population Stability Index (PSI)

```
PSI = SUM over i of  (A_i - E_i) * ln(A_i / E_i)
```

* **Computed over** the configured risk grades (six by default).
* **Population** the scored applications of a quarter vs the scorecard's own
  reference period.
* **Range** 0 to unbounded. Every term is non-negative, because `(A - E)` and
  `ln(A / E)` always share a sign — so each grade's contribution is itself
  interpretable as "how much this grade moved".

**Two variants, charted separately:**

| Variant | Expected distribution | Answers |
|---|---|---|
| Fixed reference | the model's DEVELOPMENT or PRODUCTION_BASELINE window | how far has the population drifted from the approved baseline? |
| Quarter-on-quarter | the immediately preceding quarter | did something change *recently*? |

A slow drift accumulates in the fixed-reference PSI while leaving the
quarter-on-quarter PSI quiet; a sudden policy or sourcing change does the
opposite. Reading them as one series hides both patterns, so the dashboard
charts them apart and labels them explicitly.

**Interpretation bands** (conventional, configurable):

| PSI | Label | Meaning |
|---|---|---|
| < 0.10 | GREEN | Low population shift |
| 0.10 – 0.25 | AMBER | Moderate shift; investigate |
| >= 0.25 | RED | Significant shift; escalate for review |

**What it does not tell you.** PSI measures distributional change, nothing else.
It cannot distinguish a deliberate strategy change from a data defect or from
genuine credit deterioration, and it says nothing about whether the scorecard
still ranks risk. A high PSI is a trigger for investigation.

**Caveat flagged in the output.** Where a monitoring quarter overlaps its own
baseline window, part of the monitoring population *is* the baseline, so PSI is
mechanically near zero. Those rows carry `OVERLAPS_REFERENCE = TRUE` and must
not be read as evidence of stability.

---

## 2. Characteristic Stability Index (CSI)

```
CSI_j = SUM over i of  (A_ij - E_ij) * ln(A_ij / E_ij)
```

for characteristic `j` over its bins `i`.

* **Computed over** each monitored characteristic's reference bins.
* **Population** by default the records the scorecard actually scored
  (`runtime.csi_scope = MODEL_OUTPUT`); set `ALL_RECORDS` to let unscored
  applications fall into the `MISSING` bin and move CSI directly.
* Applied to raw characteristics, variable-level subscores and the final score,
  each labelled with its `VARIABLE_ROLE`.

Identical mathematics to PSI, applied to a different partition. The division of
labour: **PSI says the population moved; CSI says which input moved.** An
investigation needs the second.

Same bands as PSI, with a tighter band available for characteristics flagged
`critical: true` and per-variable overrides for anything under specific
governance attention.

### Binning rules

Bin boundaries are fitted **once** on each model's own reference period and
reused unchanged in every monitoring quarter.

> Re-bucketing each quarter on its own quantiles forces every quarter back
> toward the reference shape, so the drift CSI exists to detect disappears.
> `tests/test_csi.py::test_numeric_bins_are_fitted_once_and_reused_across_quarters`
> demonstrates the difference numerically.

**Numeric**, in precedence order:

1. approved scorecard / WOE boundaries from `variables[].bins`, used verbatim
2. quantile boundaries fitted on the reference period
   (`runtime.default_numeric_bins`, default 10)
3. distinct-value fallback for a (near-)constant column

Outer edges are extended to `±inf`, so an out-of-range value lands in an outer
bin instead of becoming missing. Infinities are clipped into the outer bins and
counted separately in the data-quality sheet; they are never dropped.

**Categorical**, in precedence order:

1. approved grouping from `variables[].category_map` (e.g. WOE bands)
2. approved list from `variables[].categories`
3. reference levels, with levels below `runtime.rare_category_min_share`
   grouped into `OTHER`

**Special bins**, always present:

| Bin | Meaning |
|---|---|
| `MISSING` | null in the source |
| `OTHER` | rare or grouped reference levels |
| `UNSEEN` | a level that never appeared in the reference period |

These are explicit for a reason: a characteristic whose missingness jumps from
2% to 30% is a production incident, and a new category has no development bin at
all. Both must be visible rather than quietly absorbed.

---

## 3. Zero-frequency treatment

With six grades it is entirely normal for one to be empty in a quarter, which
makes `ln(0)` and division by zero routine rather than exceptional.

**Default — additive (pseudocount) smoothing:**

```
p_i = (n_i + alpha) / (N + k * alpha)        alpha = 0.5 by default
```

Applied to **both** the reference and the monitoring distribution before the
sum. `alpha = 0.5` is the Jeffreys prior; `alpha = 1.0` is Laplace. Proportions
stay strictly positive and sum to exactly 1, and the parameter is printed in the
report so any value can be reproduced.

**Alternative — `DROP_EMPTY_BINS`:** bins empty on either side are excluded from
the sum and counted in `BINS_DROPPED`. The remaining proportions are *not*
re-normalised, so the reported index is a lower bound — which the status reason
states.

Neither rule substitutes an arbitrary floor such as "treat 0 as 0.0001".

**Verification.** Bin contributions are asserted to sum to the reported total
within `smoothing.tolerance` (1e-9), and the error is reported in
`CONTRIBUTION_SUM_ERROR`. An empty reference or monitoring population is marked
`CALCULABLE = FALSE` with a reason, never reported as a stable index of zero.

---

## 4. MER (Model Effectiveness Report) metrics

One row per model-quarter.

| Metric | Definition |
|---|---|
| `TOTAL_RECORDS` | rows in the quarter |
| `UNIQUE_LANS` / `UNIQUE_CUSTOMERS` | distinct identifiers |
| `SCORED_RECORDS` | rows with a valid grade |
| `MISSING_SEGMENT_PCT` | share with no grade |
| `INVALID_SEGMENT_PCT` | share with a value outside the configured list |
| `MISSING_OUTPUT_PCT` | missing + invalid |
| `N_<grade>` / `PCT_<grade>` | count and share per grade |
| `MOST_POPULATED_SEGMENT` | modal grade |
| `SEGMENT_CONCENTRATION_PCT` | `max_i (n_i / N)` |
| `SEGMENT_HHI` | `SUM_i (n_i / N)^2` |
| `AVERAGE_SEGMENT_RANK` | mean ordinal risk rank (1 = safest) |
| `RISKIER_SEGMENT_SHARE_PCT` | share in the configured riskiest grades |
| `POPULATION_CHANGE_VS_REFERENCE_PCT` | `(N_q - N_ref) / N_ref` |
| `POPULATION_CHANGE_VS_PREV_QUARTER_PCT` | `(N_q - N_q-1) / N_q-1` |
| `DUPLICATE_LAN_COUNT` / `_PCT` | repeated loan account numbers |
| `RECORDS_PER_CUSTOMER` | fan-out diagnostic |
| `QUARTER_IN_REFERENCE_WINDOW` | the quarter overlaps its own baseline |

**Concentration** deserves attention in its own right: a scorecard that assigns
most of the book to one grade has little practical discriminatory value
regardless of what its development statistics showed.

---

## 5. Segment migration

Computed on **matched customers** — the same customer observed in both periods,
collapsed to one record each by `runtime.multi_record_policy`
(`LATEST`, `EARLIEST`, `WORST`, `BEST`, `DROP`).

```
row %    = n(from i, to j) / n(from i)        each row sums to 100%
column % = n(from i, to j) / n(to j)
total %  = n(from i, to j) / n(matched)
```

| Metric | Definition |
|---|---|
| Upgrade | move to a **safer** grade (lower risk rank) |
| Downgrade | move to a **riskier** grade |
| `NET_DOWNGRADE_PP` | downgrade % − upgrade % |
| `AVERAGE_RANK_MOVEMENT` | mean signed change in ordinal rank (positive = riskier) |
| `MULTI_NOTCH_DOWNGRADES` | moves of two grades or more toward risk |
| `LARGEST_FLOW` | the biggest single cell |
| `MATCH_RATE_PCT` | matched customers / smaller population |

Upgrade and downgrade are derived from the configured `risk_direction`, so
reversing the convention reverses the classification.

**Why matching matters.** Comparing two *unmatched* populations and calling the
difference "migration" is a category error — that is a distribution shift, which
PSI and the segment distribution sheet already cover. Where too few customers
appear in both periods, the migration view is `GREY` with the match rate stated
rather than silently replaced by a distribution comparison.

**Causal caveat, carried on every row.** Migration between two independently
scored snapshots can reflect a change in the applicant population, in
underwriting or sourcing policy, in customer behaviour, or in the scoring
configuration. It is not by itself evidence of model deterioration.

---

## 6. Outcome performance

Requires a matured outcome label. Without one, every metric below is reported
as `NOT AVAILABLE` with its missing dependency named.

### What the outcome flag measures

Three configuration fields decide whether the numbers mean what a reader will
assume they mean. All three are printed on the `09 Outcome Performance` sheet
above the metrics, and on every row of the summary table.

| Field | Why it changes the interpretation |
|---|---|
| `definition_basis` | `ON_US` (our own book) vs `OFF_US` (bureau-observed delinquency on **other lenders'** accounts) vs `COMBINED` |
| `indeterminate_values` | the band that is neither good nor bad, plus any no-information population |
| `outcome_level` | one flag per application, or one per customer repeated across their applications |

**Off-us outcomes are a proxy, with a real advantage.** An off-us 30+ DPD flag
covers the full through-the-door population — declines and not-disbursed
applications included — so rank ordering measured on it is free of the
accept-only selection bias that limits on-book performance monitoring. That is
genuinely valuable: it is the closest thing to an unbiased read on whether a
scorecard still separates risk across everyone it scores.

But it is almost never the target the scorecard was developed on. Its Gini and
KS are therefore **proxy** discrimination measures, not a restatement of
development performance, and they are not comparable with development-time
figures or with on-us performance. Two practical consequences:

* Set the Gini floors for *this* definition. A 30+ DPD off-us target is a
  milder bad definition measured on a different population than on-us 90+
  default, and typically yields a lower Gini. Reusing floors agreed for an
  on-us default definition will flag healthy models as failing.
* Do not validate an on-us PD calibration against an off-us target. Leave
  `pd_column` unset and the pack reports calibration as `NOT AVAILABLE`
  rather than comparing two different quantities.

The caveat travels with the metric: it appears in `PERFORMANCE_STATUS`, on the
definition banner, in the availability table and in the limitations list.

### The indeterminate band

A 30+ DPD definition leaves a middle: customers 1–29 days past due are
delinquent but not bad. Bureau data adds a second group — customers with no
trade line at all (thin file, no hit, not found), whose delinquency is simply
unknown.

Both are declared in `indeterminate_values`, excluded from every performance
metric, and counted in `INDETERMINATE_EXCLUDED`. They are never merged into the
good population.

> This is the single most damaging configuration error available here. If a
> no-bureau-hit population is encoded as `0` and `0` is in `good_values`, those
> customers silently become good: the bad rate is understated, the good
> population is diluted with unknowns, and every discrimination metric in the
> pack is flattered. The pipeline warns when an `OFF_US` outcome declares no
> indeterminate values at all, but it cannot detect a no-hit code disguised as
> a zero — that has to be confirmed with whoever supplies the flag.

Values matching none of the three lists are treated as unknown (excluded, not
good) and reported with examples, so an unexpected encoding surfaces rather
than being absorbed.

### Customer-level flags

A flag supplied per customer is repeated across that customer's applications.
Left as-is, three applications from one customer contribute three perfectly
correlated observations to AUC and KS, which inflates the effective sample size
and over-weights repeat applicants — who are not a random slice of the book.

So when `outcome_level: CUSTOMER`, discrimination is measured on one record per
customer, collapsed with `runtime.multi_record_policy`. Both counts are
reported (`APPLICATIONS_BEFORE_DEDUPE`, `UNITS_IN_METRIC`) and `METRICS_LEVEL`
states which grain was used. Stability metrics are unaffected — they are
application-level by construction.

The flag can also arrive as its own file, which is how a bureau pull usually
reaches you:

```yaml
outcome:
  source:
    path: data/offus_30plus_flag.csv
    key_column: CUSTOMER_ID      # key in the outcome file
    join_on: CUSTOMER_ID         # column in the application extract
    columns: [OFFUS_30P_DPD]
```

The join is always a LEFT join. An application with no matching outcome row
keeps a null flag and leaves the performance metrics; it is never dropped from
the stability analysis and never defaulted to good. The match rate is logged,
duplicate keys are de-duplicated rather than allowed to fan out the extract,
and a match rate below 50% raises a warning — identifier format mismatches
(leading zeros, prefixes, type coercion) are the usual cause and would
otherwise quietly destroy the performance section.

### Right-censoring

An account booked two months ago with no default yet is **not** a good
customer — it is an unknown. Counting it as good understates the bad rate and
flatters every discrimination metric.

* Accounts whose `observation_window_months` has not elapsed are excluded
  (`outcome.exclude_immature`, default true) and counted in
  `IMMATURE_EXCLUDED`.
* `MATURITY_PCT` reports how much of the cohort has seasoned.
* `COHORT_COMPLETE = FALSE` marks a quarter whose bad rate is a censored lower
  bound that will still rise. Those quarters are not comparable with seasoned
  ones, however attractive their bad rate looks.

### Metrics

| Metric | Formula / definition |
|---|---|
| Event rate | bads / matured records |
| Non-event rate | goods / matured records |
| Bad rate by segment | per-grade event rate |
| AUC | probability a random bad outranks a random good, by risk score |
| Gini | `2 * AUC - 1` |
| KS | `max over s of |F_bad(s) - F_good(s)|` |
| Lift | grade bad rate / overall bad rate |
| Cumulative bad capture | cumulative bads / total bads, riskiest grade first |
| Rank ordering | count of adjacent grade pairs whose bad rate is inverted, with the Spearman correlation between rank and bad rate |
| Observed vs expected | observed bad rate / mean expected PD |
| Calibration z | `(B - E[B]) / sqrt(SUM p_i (1 - p_i))`; `|z| <= 1.96` is within the 95% binomial tolerance |
| Vintage analysis | bad rate by origination quarter and grade, with maturity stated |

**Score source.** AUC and KS need a risk ordering. The pipeline uses an explicit
score column where one exists; otherwise it falls back to the ordinal grade
rank and says so in `SCORE_SOURCE`. With six heavily tied grades the resulting
AUC is a conservative lower bound, not a like-for-like substitute for a
continuous score.

**Rank ordering** skips grades with no matured records rather than treating them
as a zero bad rate — an empty grade is unknown, not perfect.

**Calibration** requires a model-estimated probability. A risk grade is not a
probability and is never substituted for one.

Gini is a "higher is better" metric, so its band is inverted:
`GREEN >= min_gini_amber > AMBER >= min_gini_red > RED`.

---

## 7. Data quality

Each check emits one row with the same schema (`CHECK_ID`, `CATEGORY`, `CHECK`,
`MODEL`, `QUARTER`, `SCOPE`, `METRIC`, `VALUE`, `UNIT`, `THRESHOLD`, `SEVERITY`,
`RECORDS_AFFECTED`, `DETAIL`, `RECOMMENDED_ACTION`).

| ID | Check |
|---|---|
| DQ-001/002/003 | duplicate LANs, duplicate customer+date, repeat customers |
| DQ-010/011/012 | unusable, future-dated and implausibly old dates |
| DQ-020 | incomplete calendar quarter |
| DQ-030 | unexpected quarter-on-quarter volume change |
| DQ-040/041/042 | model output availability, missing output, invalid grade values |
| DQ-050 | high missingness in a characteristic |
| DQ-051/052/053 | infinities, 3×IQR outliers, very high cardinality |
| DQ-055 | material change in missingness vs reference |
| DQ-056 | categories unseen in the reference period |
| DQ-060/061/062 | data freshness, absent characteristics, entirely empty characteristics |

Nothing is auto-corrected: duplicates and outliers are reported, not removed,
so the pack never silently changes the population it is describing.

---

## 8. Alert severities

| Severity | Meaning |
|---|---|
| `GREEN` | Within normal monitoring limits |
| `AMBER` | Investigation required |
| `RED` | Significant breach requiring escalation |
| `GREY` | Insufficient data, or the metric could not be calculated |

For "worst severity" aggregation the order is `GREEN < GREY < AMBER < RED`:
"unknown" is worse than "fine" but is not an escalation.

Thresholds are always applied to the **unrounded** metric. A PSI of 0.2499
displays as 0.25 but classifies as AMBER, because the displayed value is a
presentation choice and the alert is a governance decision.

Sample-size gates (`min_reference_records`, `min_monitoring_records`,
`min_variable_records`, `min_outcome_records`, `min_bad_count`) turn a metric
`GREY` rather than let it produce a confident label on data that cannot support
one.
