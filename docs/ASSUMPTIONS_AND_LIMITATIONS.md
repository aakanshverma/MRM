# Assumptions and limitations

What this monitoring pack does and does not establish. The same list is written
into the workbook's `10 Config & Methodology` sheet so a reader of the Excel
file sees it without consulting the repository.

---

## Governance and interpretation

**1. Thresholds are conventions, not regulation.**
The 0.10 / 0.25 PSI and CSI bands are widely cited monitoring guidelines. They
are not universal regulatory limits. They are configurable and should be set by
model governance per scorecard, with tighter bands for business-critical
characteristics. Meeting them does not establish regulatory compliance;
breaching one does not establish non-compliance.

**2. A breach is a trigger, not a verdict.**
PSI and CSI measure distributional change. They cannot distinguish a deliberate
strategy change from a data defect or from genuine credit deterioration. Every
AMBER/RED finding requires investigation before any conclusion about the model,
and no alert in this pack recommends retiring a scorecard.

**3. Stability is not performance.**
Segment output supports stability analysis only. Without matured outcome labels
the pack cannot say whether a scorecard still separates good from bad, and it
does not try: those metrics are reported as `NOT AVAILABLE` with the missing
dependency named. A stable population does not imply a working model, and a
population shift does not by itself prove a broken one.

**4. The pack is a snapshot.**
It reflects one extract at one point in time. Restatements, late-arriving bureau
data and retrospective corrections will change the figures. The input file, row
count, column count, execution timestamp and effective configuration are
recorded for reproducibility.

---

## Reference periods

**5. Every metric is relative to its baseline.**
A different reference window produces a different PSI and CSI for the same data.
The window, mode, record count and provenance note are therefore printed beside
every metric that depends on them, and each scorecard keeps its own.

**6. Quarters inside the reference window are not evidence of stability.**
Where a monitoring quarter overlaps its own baseline, part of the monitoring
population *is* the baseline and the fixed-reference PSI is mechanically close
to zero. Those rows carry `OVERLAPS_REFERENCE = TRUE`; the same quarters are
greyed out in the migration view, where "matched customers" would largely be
records matched against themselves.

**7. A small baseline makes every metric unstable.**
Expected proportions estimated from a few hundred records carry wide sampling
error, so a PSI computed against them is noisy regardless of the monitoring
sample. Baselines below `thresholds.min_reference_records` raise a reference
adequacy alert and all of that model's stability metrics are reported `GREY`.

**8. Reference modes are declarations, not validations.**
`DEVELOPMENT` and `PRODUCTION_BASELINE` are recorded as the user states them.
The pipeline cannot verify that a window really is the development sample or
that a production baseline was formally approved; the `source_note` field exists
to carry that evidence.

---

## PSI and CSI mechanics

**9. Smoothing changes small-sample values.**
With the default pseudocount (0.5), a grade with zero observations contributes a
finite amount instead of an undefined one. On small populations the smoothed
value differs materially from the unsmoothed one. The parameter is reported so
any value can be reproduced, and `DROP_EMPTY_BINS` is available where an
unsmoothed lower bound is preferred.

**10. CSI depends on the binning choice.**
Approved scorecard boundaries are used where supplied; otherwise reference
quantiles are fitted. Two defensible binnings of the same characteristic give
different CSI values. The fitted boundaries are retained on the
`13 Bin Definitions` sheet precisely so the number can be challenged.

**11. Movement inside the `OTHER` bin is not attributable.**
Rare reference levels are grouped, so a shift *within* that group is invisible.
Characteristics whose largest contribution sits in `OTHER` need an approved
grouping before the driver can be identified — which the recommended action
says.

**12. The characteristic population is the scored population by default.**
`runtime.csi_scope = MODEL_OUTPUT` measures each scorecard's characteristics on
the records it actually scored. This is the right denominator when a model
covers part of the book, but it means an application the model *failed* to score
drops out of CSI; that volume is monitored instead by the MER missing-output
percentage and data-quality check DQ-041. Set `ALL_RECORDS` to push unscored
applications into the `MISSING` bin and let them move CSI directly.

**13. Unscored volume is excluded from PSI by default.**
Records with a missing or invalid grade do not take part in the segment
distribution (`runtime.psi_include_missing_segment = false`) and are monitored
separately. Where unscored volume is material this biases the PSI comparison
toward the scored sub-population.

**14. Invalid grade values are excluded, which biases the metrics.**
A value outside the configured list cannot be placed on the risk scale, so those
records leave the PSI distribution and the migration matrices. The exclusion is
counted and alerted rather than hidden.

---

## Migration

**15. Migration needs matched customers.**
Quarter-on-quarter migration joins on the customer identifier under
`runtime.multi_record_policy`. Where few customers appear in both periods the
matrix is `GREY`, not inferred from unmatched populations.

**16. The multi-record policy changes the answer.**
Collapsing a repeat applicant with `LATEST` versus `WORST` produces different
migration rates for the same data. The policy is stated on every migration row.

**17. Repeat applicants are not a random sample.**
Customers who re-apply differ systematically from those who do not, so matched
migration describes the repeat-application book rather than the portfolio as a
whole.

---

## Outcome performance

**18. Recent cohorts are censored.**
With the configured performance window, recent quarters are incomplete. Their
bad rates are censored lower bounds flagged `COHORT_COMPLETE = FALSE` and are
not comparable with seasoned vintages. An account with no default yet is treated
as unknown, never as good.

**19. Grade-based AUC is a lower bound.**
Where no continuous score exists the ordinal grade rank is used. With six
heavily tied grades the resulting AUC understates the discriminatory power a
continuous score would show. `SCORE_SOURCE` states which was used.

**20. Calibration needs a probability.**
Observed-versus-expected analysis requires a model-estimated PD. A risk grade is
not a probability and is never substituted for one. The z-test assumes
independent outcomes, so correlated defaults (a regional shock, a common
exposure) will overstate its significance.

**21. The bad definition is taken as given.**
`bad_values` and `good_values` are applied as configured. A changed bad
definition mid-series will move every performance metric, and the pipeline
cannot detect that from the data alone. Values matching neither list are treated
as unknown rather than good.

---

## Data handling

**22. Incomplete quarters are excluded by default.**
A part-quarter population is not comparable with a full one. Partial quarters
are identified and reported but excluded from the primary comparison unless
`runtime.include_partial_quarters` is enabled.

**23. Duplicates are reported, not removed.**
Where duplication is material, population counts and distributions in the pack
are inflated accordingly and the pack should be re-run on a de-duplicated
extract.

**24. Outliers and infinities are retained.**
Extreme values are clipped into outer bins and counted, never deleted. This
keeps the population intact but means a sentinel code such as 9999 treated as a
real value will sit in the top bin and distort the distribution. DQ-052 flags
the pattern.

**25. Discovery can over-select.**
Regex-based column discovery resolves whatever matches the header. A column that
matches the pattern but is not a model input will be monitored. Explicit
`variables` lists avoid this where the characteristic set is known.

**26. Quarterly granularity is fixed.**
Monitoring periods are calendar quarters derived from the application date.
Monthly or rolling-window monitoring would need an extension to the period
logic.

**27. Single-run, single-threaded.**
The pipeline holds the pruned dataset in memory and runs in one process. Very
large extracts need the memory strategies in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) rather than silent downsampling.
