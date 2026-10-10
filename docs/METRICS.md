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
