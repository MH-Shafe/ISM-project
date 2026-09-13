# Phase 13 Specification — Temporal Generalization / Unseen-Window Evaluation of the Frozen `lgbm-graph-v1` System

Status: **PLANNING ONLY — NOT AUTHORIZED FOR EXECUTION.**
Date: 2026-08-17
Author: OpenCode (per the authorized Phase 13 planning task)
Review: pending user approval of this specification.

---

## 0. Feasibility verdict (the critical data question)

**Phase 13 "unseen-window evaluation" as literally specified is NOT FEASIBLE
under the current dataset/evaluation contract.**

- CERT r4.2 spans **2010-01-02 .. 2011-05-17** (501 days). The frozen
  chronological split consumes every labeled row: TRAIN ≤ 2011-01-31
  (395,000), CALIBRATION 2011-02-01..2011-03-31 (59,000), TEST ≥ 2011-04-01
  (47,000). TEST is the **final 47 days of the dataset**; **no later labeled
  window exists** inside r4.2. (OBSERVED — master report §22: "Unseen
  production-surrogate evaluation — not possible inside r4.2 … would require a
  new dataset or release"; `src/config.py` SPLIT_DEFAULTS.)
- Creating a "new" unseen window by re-splitting existing rows would either
  (a) reuse TEST rows (violating the TEST-once contract), (b) evaluate
  in-sample TRAIN rows (the model trained on all of TRAIN — no
  generalization claim), or (c) retrain on a rolling origin (forbidden:
  frozen system, no retraining). All are scientifically invalid here.
- A true replication window requires a **new dataset or release outside
  r4.2**, which is out of contract (principles §1: r4.2 only, no silent
  substitution) and would need separate authorization.

**Therefore Phase 13 is re-scoped to the smallest scientifically valid
alternative that stays inside r4.2 and the frozen contract:**

> **Temporal-stability analysis of the frozen system across ALL model-unseen
> days (CALIBRATION + TEST, 106 days, 106,000 rows, 353 positives) at a
> fine-grained (7-day) grid, using ONLY already-frozen outputs, with zero
> fitting, zero selection, and zero decisions from TEST.**

This is the strongest temporal-generalization evidence obtainable without new
data: the model has never seen any row after 2011-01-31, so CAL and TEST are
genuinely unseen *model* windows; no prior phase examined TEST at window
granularity (Phases 9/10/12 window analyses were CAL-only; TEST was reported
only as a block).

---

## 1. Objective

Characterize the temporal generalization of the frozen `lgbm-graph-v1`
system — score distribution, alert volume, operating point, and user
concentration — across chronological 7-day windows covering the two
model-unseen windows (CALIBRATION 2011-02-01..2011-03-31, TEST
2011-04-01..2011-05-17), against pre-registered stability bounds. Produce a
**descriptive robustness verdict** (PASS / CAUTION / FAIL). The verdict is a
risk-register statement only: **no system component changes under any
verdict.**

## 2. Research question

> Does the frozen system's behavior remain temporally stable (alert volume,
> operating point, score distribution, coverage) across all 106 model-unseen
> days, or is there systematic degradation? (OBSERVED evidence only; no
> hypothesis testing on TEST.)

Primary: CAL-window stability (decision-relevant evidence). Annex:
TEST-window stability (descriptive only, pre-registered bounds).

## 3. Dataset / version

- CERT r4.2 only (Kaggle, read-only):
  `/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2`.
- Raw data never leaves Kaggle. Phase 13 execution (when authorized) uses
  **already-frozen artifacts** plus optional deterministic re-scoring on the
  kernel; no raw-log processing.

## 4. Analytical unit

- **user × day** (unchanged). All windows are sets of user-day rows.

## 5. Temporal windows

| Block | Range | Days | Rows | Positives | 7-day windows |
|---|---|---|---|---|---|
| CALIBRATION | 2011-02-01..2011-03-31 | 59 | 59,000 | 323 | 9 (8×7 d + 3 d tail) |
| TEST | 2011-04-01..2011-05-17 | 47 | 47,000 | 30 | 7 (6×7 d + 5 d tail) |

**Total: 16 windows over 106 unique days.**

> **Correction record (transparency):** an earlier draft of this
> specification listed TEST as 6 windows (and a 15-window total). The
> calendar check (2026-08-17, verified) shows TEST = 47 days = 6×7 + 5,
> i.e. **7 windows**, so the grid is **16 windows** (9 CAL + 7 TEST). The
> addendum's "15-window dates" referred to the then-current draft; the
> authoritative pre-registered table below replaces it. No metrics were
> computed under either count — this correction is made at planning time,
> before execution.

**Pre-registered window date table (asserted verbatim before any metric is
calculated; the computed windows must equal these dates exactly):**

| # | Block | Start | End | Days |
|---|---|---|---|---|
| W0 | CAL | 2011-02-01 | 2011-02-07 | 7 |
| W1 | CAL | 2011-02-08 | 2011-02-14 | 7 |
| W2 | CAL | 2011-02-15 | 2011-02-21 | 7 |
| W3 | CAL | 2011-02-22 | 2011-02-28 | 7 |
| W4 | CAL | 2011-03-01 | 2011-03-07 | 7 |
| W5 | CAL | 2011-03-08 | 2011-03-14 | 7 |
| W6 | CAL | 2011-03-15 | 2011-03-21 | 7 |
| W7 | CAL | 2011-03-22 | 2011-03-28 | 7 |
| W8 | CAL | 2011-03-29 | 2011-03-31 | 3 (tail) |
| W9 | TEST | 2011-04-01 | 2011-04-07 | 7 |
| W10 | TEST | 2011-04-08 | 2011-04-14 | 7 |
| W11 | TEST | 2011-04-15 | 2011-04-21 | 7 |
| W12 | TEST | 2011-04-22 | 2011-04-28 | 7 |
| W13 | TEST | 2011-04-29 | 2011-05-05 | 7 |
| W14 | TEST | 2011-05-06 | 2011-05-12 | 7 |
| W15 | TEST | 2011-05-13 | 2011-05-17 | 5 (tail) |

- Grid: consecutive non-overlapping chunks of the **sorted unique calendar
  days** (the Phase 12 `cal_windows` convention, regression-tested). Windows
  are **never constructed by slicing rows** — chunking is over unique days
  only; rows are then assigned to the window of their day.
- Secondary grid (cross-check only): 15-day windows on CAL (Phase 9
  convention, 4 windows) to verify consistency with the Phase 9 record.

## 6. TRAIN / CALIBRATION / TEST usage

- **TRAIN**: not used. The frozen model was trained on TRAIN; evaluating on
  TRAIN would be in-sample and is excluded.
- **CALIBRATION**: primary evidence for the verdict (window stability under
  pre-registered bounds). All CAL per-window statistics are already recorded
  in frozen artifacts (`phase9_calibration.json`,
  `phase12_rolling_threshold.json`); the analysis re-reads records and
  re-computes from frozen per-row scores only if the optional scoring pass
  is approved.
- **TEST**: descriptive annex only (see §7).

## 7. Reuse of the existing TEST — policy

- The authoritative single TEST evaluation (Phase 7/9/12 records: AUC-ROC
  0.939157, AUC-PR 0.267776, 49 alerts, F1 0.3544) is **unchanged and remains
  the only evaluation used for any system decision**.
- Phase 13 **re-reads the frozen `phase12_test_predictions.parquet`
  (47,000 × 7: user, day, is_malicious, score, alert_p0, alert_selected,
  selected_policy)** for **pre-registered descriptive statistics** only.
- **TEST predictions are never regenerated.** The frozen parquet is reused
  as-is; no model run, no scoring pass, and no recomputation touches TEST
  prediction outputs (the optional scoring pass, §12, scores CAL only).
- TEST-derived values (a) never enter any selection, threshold, parameter,
  or bound; (b) never alter the verdict's primary component (CAL-only); (c)
  are reported in an explicit DESCRIPTIVE ANNEX, with the same pre-registered
  bounds applied as a consistency check, and are labeled OBSERVED (descriptive
  re-analysis), not as a new evaluation.
- Rationale (documented protocol decision): the TEST-once rule exists to
  prevent TEST-driven tuning; a pre-registered, non-adaptive, decision-free
  re-analysis of already-published frozen scores adds no information to any
  decision and cannot leak. This is explicitly recorded so future readers
  cannot mistake it for a second evaluation.

## 8. Existence of a genuinely unseen future window

- **None exists in r4.2** (see §0). CAL + TEST (106 days, 353 positives) is
  the complete set of model-unseen labeled days. The phase states this in its
  report and does not claim to have created a new unseen window.

## 9. Label handling

- Labels (`is_malicious`, user-day) are used **only** for descriptive
  evaluation metrics and per-scenario stratification (`per_scenario_recall`,
  `scenario_of_user` from the answer key — the established Phase 12
  convention).
- Labels never enter any feature, statistic used by a frozen component, or
  selection — there is no selection in Phase 13.
- No label-derived quantity is computed before the bounds are fixed (they are
  fixed in this document, §21).

## 10. Leakage controls

Every calculation must answer: *"Would this information have been available
at prediction time?"* Phase 13 adds no predictive information anywhere:

- All inputs are frozen outputs (scores) + labels for description.
- All bounds and reference values are **pre-registered constants in this
  document**, taken from PUBLISHED records (Phase 9/12) — never fresh TEST
  computations.
- No fitting of any kind; no normalization; no preprocessing statistics; no
  threshold; no calibration; no graph construction.
- Optional kernel scoring pass (if approved) re-scores CAL with the frozen
  model (pure prediction, deterministic; no fit). **TEST predictions are
  never regenerated** — the frozen parquet is reused and its record anchors
  asserted (schema, row count, alert count, md5) instead of any re-scoring.
- Window chunking over unique days is the regression-tested Phase 12
  convention (no row-order dependence, no future reference).

## 11. Frozen components (untouched)

- `lgbm-graph-v1` (12 features, seed 42, 186 trees, LightGBM 4.6.0,
  `src/experiments/phase7.py` FROZEN_CONFIG).
- Alert threshold `0.9186015432508062` (Phase 9 `frozen_max_f1`).
- Conformal fit (Phase 10, t0 = 0.4634739481800199) — referenced, not
  re-fitted.
- Explainability layer (Phase 11) — referenced, not re-run.
- Phase 7/9/10/11/12 artifacts: read-only. Phase 13 writes only
  `phase13_*` artifacts.

## 12. Allowed modifications

- New analysis code: `src/experiments/phase13.py` (pure functions over
  frozen scores/records), `kaggle_scripts/run_phase13.py` (optional scoring
  stage), `tests/test_phase13.py`.
- Optional single deterministic re-score of CAL with the frozen model —
  **prediction only, no fit** (provides per-row CAL scores for the window
  grid, per-window Gini, and CAL score distributions). **TEST is never
  re-scored and TEST predictions are never regenerated.**
- New artifacts `phase13_*`; this specification; the Phase 13 report;
  master-report maintenance entry (after execution); knowledge-layer notes
  (after execution).

## 13. Forbidden modifications

- Retraining `lgbm-graph-v1`; changing features, seed, tree count, or any
  hyperparameter.
- Changing the frozen threshold or any policy parameter.
- Tuning, optimizing, or selecting anything on TEST (or on CAL).
- Modifying Phase 7–12 artifacts or historical reports.
- Altering split boundaries; changing the analytical unit.
- Re-evaluating TEST for any decision; adding new TEST statistics beyond the
  pre-registered descriptive set.
- Using a dataset other than r4.2; copying raw data to the PC.

## 14. Evaluation metrics

Per 7-day window (and per block), from frozen scores + labels:

| Metric | Detail |
|---|---|
| n_rows, n_positives, n_alerts, alert_rate | volume |
| precision, recall, F1, MCC, balanced accuracy, FPR, FNR | operating point (frozen threshold applied deterministically) |
| zero_alert flag, zero_recall flag | coverage |
| Gini over per-user alert counts (block and window) | concentration (only if per-row scores are available for that block) |
| score median / p90 / p99 | distribution |
| scenario recall (per scenario with n ≥ 1) | subgroup |
| AUC-ROC, AUC-PR (block-level only, window level when n_positives ≥ 10) | ranking context, descriptive |

Block-level reference values (published records, used as pre-registered
anchors): CAL 244 alerts / 4.14/day / precision 0.6844 / recall 0.5170 /
score median 0.11596; TEST 49 alerts / 1.04/day / precision 0.2857 / recall
0.4667.

## 15. Required subgroup / scenario analysis

- Per-window and block: scenario recall for every scenario with ≥ 1
  malicious row (CAL: scenarios 1–2; TEST: scenarios 2–3).
- Per-window alerted-user counts and alerts/user max (concentration trend).
- Descriptive lead-time output (optional, pre-registered): for each malicious
  user with ≥ 1 frozen alert in CAL/TEST, days from first frozen alert to the
  user's first malicious day; distribution only, no decision use.

## 16. Threshold handling

- The frozen threshold is applied as a constant (deterministic mask
  `score ≥ 0.9186015432508062`). No threshold is fitted, bootstrapped for
  selection, or changed. Phase 9 S1–S3 results are cited as context, not
  recomputed for selection.

## 17. Conformal handling

- Referenced only: Phase 10's per-2-week CAL window coverage table (§9 of
  that report) is cited as supporting temporal-robustness evidence. No
  conformal computation runs; no overlay changes.

## 18. Explainability handling

- Referenced only: Phase 11's per-alert reason statistics are descriptive
  context. No `pred_contrib` computation runs in Phase 13.

## 19. Required baselines

1. The frozen system's own published block records (CAL: Phase 9/12; TEST:
   Phase 7/9/12) — the stability reference.
2. Cross-artifact consistency anchors (automated gates): 7-day CAL windows
   must match `phase12_rolling_threshold.json` `windows_static` (alerts,
   precision, recall, F1, score percentiles); 15-day windows must match
   `phase9_calibration.json` (window alerts [79,70,56,39], precision
   [0.633,0.829,0.625,0.615], recall [0.575,0.563,0.455,0.429]).
3. If the optional scoring pass runs (CAL only — TEST is never re-scored):
   CAL 7-day stats re-computed from per-row scores must equal the
   `phase12_rolling_threshold.json` record; the frozen
   `phase12_test_predictions.parquet` is loaded as-is and its schema/row
   count/alert count asserted against the Phase 12 record.

No new model, policy, or random baseline is required (Phase 12 already
established that no operational alternative dominates P0).

## 20. Required artifacts (after execution)

- `phase13_experiment.json` — protocol record (windows, bounds, inputs,
  verdict, rationale, evidence labels).
- `phase13_window_analysis.json` — full per-window + block tables (CAL 9,
  TEST 7, plus 15-day CAL cross-check).
- `phase13_verdict.json` — bound-by-bound PASS/CAUTION/FAIL with counts
  (CAL inputs only).
- `phase13_cost.json` — elapsed times, artifact sizes, rows processed.
- `phase13.log` — `[JOB]`-format log (execution-policy §3).
- `phase13_cal_predictions.parquet` — ONLY if the optional CAL scoring pass
  is approved (else omit; record NOT VERIFIED). TEST predictions are reused
  from the frozen `phase12_test_predictions.parquet` and never regenerated.
- `reports/phase13_temporal_stability_report.md` — the phase report.
- Optional: `phase13_lead_time.json` (if §15 lead-time output is included).

## 21. Required tests (local, before execution results are recorded)

`tests/test_phase13.py` (pure-function unit tests, no real data):

- Window partition over unique days (reuses the Phase 12 convention;
  hand-computed chunks, uneven tail, determinism).
- Pre-registered bound definitions: bounds parse as constants; bound-check
  function returns per-bound PASS/CAUTION/FAIL with the exact margin.
- Wilson 90% CI helper (hand-computed cases).
- Verdict aggregation logic: PASS when all CAL bounds hold; CAUTION for ≤ 2
  marginal violations; FAIL for ≥ 3 violations or any violation > 10%
  margin (pre-registered, tested on synthetic bound matrices).
- Artifact reloadability: `phase12_test_predictions.parquet` schema
  (47,000 × 7), `phase9_calibration.json`, `phase12_rolling_threshold.json`
  parse and expose the fields the analysis needs.
- Determinism: analysis function run twice → identical output.
- **Safety-gate tests (Section 28, implemented as unit tests over synthetic
  and frozen-artifact data):** unique-day counts (CAL 59 / TEST 47 / total
  106); windows partition unique days exactly (every day in exactly one
  window; every row in exactly one window); no window crosses the CAL/TEST
  boundary; computed 16-window dates equal the pre-registered table
  (Section 5) verbatim; the verdict function raises/rejects any non-CAL
  input (architecturally CAL-only); TEST columns never appear in verdict or
  bound inputs (a test feeds TEST window data and asserts the verdict
  function cannot consume it).
- Full suite must stay green (currently 277 passed / 4 skipped); no existing
  test may change.

## 22. Reproducibility requirements

- All analysis code deterministic and seed-free; the only RNG use is the
  optional bootstrap (seed 42, `np.random.default_rng(42)`), evaluation-only.
- Rerun gate: re-running the analysis reproduces `phase13_window_analysis.json`
  bit-identically (record `had_previous_record` / `reproduced_bit_identical`).
- Every number in the phase report must be traceable to
  `phase13_window_analysis.json` or a cited published record.
- Execution policy compliance: kernel-side steps use `[JOB]` heartbeats,
  explicit timeouts, and the gz+b64 + md5 transfer protocol; journal entries
  via `kaggle_exec.py log`.

## 23. Statistical uncertainty requirements

- Per-window precision/recall: Wilson 90% intervals (implemented + tested).
- Block AUC-ROC / AUC-PR: existing `bootstrap_ci` (n = 1,000, seed 42) on
  the frozen scores; n_skipped reported (windows with too few positives
  documented, NOT VERIFIED where undefined).
- Alert-rate windows: exact Poisson-free descriptive deviation vs block rate;
  the S2-style [0.5, 2.0]× bound is a pre-registered tolerance, not a test.
- All intervals labeled OBSERVED (evaluation-only); no claim of significance;
  wide-TEST-CI caveat restated (30 positives).

## 24. TEST-evaluation rule

- TEST is evaluated **once** per finalized experiment — that evaluation is
  the Phase 7/9/12 record and is not repeated or superseded.
- Phase 13 re-reads the frozen TEST parquet for the pre-registered
  descriptive annex; **TEST predictions are never regenerated**.
- **The verdict function receives CAL-only inputs, enforced by code
  architecture and unit tests** (it accepts only the CAL window table and
  raises on any TEST-derived input). TEST never contributes to the primary
  verdict, any bound, or any system decision. This rule is stated in the
  phase report and enforced by code structure (the verdict function takes
  CAL inputs only).

## 25. Pass / fail decision criteria (pre-registered, descriptive verdict)

Bounds (constants fixed here; CAL values from published records; TEST windows
get the same bounds in the annex only):

| # | Bound | CAL reference (published) | Applied to |
|---|---|---|---|
| B1 | every window has ≥ 1 alert (no zero-alert window) | 0 zero-alert in Phase 9/12 records | CAL windows (primary); TEST windows (annex) |
| B2 | per-window alert rate ∈ [0.5, 2.0] × block rate | 4.14/day CAL (windows 2.79–5.27) | both |
| B3 | per-window recall ≥ 0.25 when positives > 0 | CAL windows 0.429–0.575 | both |
| B4 | per-window precision ≥ 0.5 × block precision | CAL 0.684 → floor 0.342 | both |
| B5 | per-window score median ∈ [0.5, 2.0] × CAL block median (0.11596) | monotone decline documented, within band | both |
| B6 | no window with positives but zero recall | none in records | both |

Verdict rule (CAL windows only — TEST annex reported with the same bounds as
"consistent / inconsistent", never merged into the verdict):

- **PASS**: all B1–B6 hold on all 9 CAL windows.
- **CAUTION**: ≤ 2 bound violations, each within 10% margin of the bound.
- **FAIL**: ≥ 3 violations, or any violation beyond 10% margin.

**Verdict semantics:** a descriptive risk-register statement about temporal
stability. **Under every verdict the frozen system stands unchanged**; a
CAUTION/FAIL adds a documented risk (master report §21) and — like Phase 9's
documented recall drift — is recorded, not "fixed" (no adaptation allowed).

## 26. Alternatives considered and rejected (selection rationale)

| Design | Rejected because |
|---|---|
| A. Strict new unseen window after TEST | no labeled data after 2011-05-17 in r4.2 — infeasible (§0) |
| B. Within-TRAIN window evaluation | model trained on all of TRAIN — in-sample, no generalization content |
| C. Rolling-origin retraining backtest | retrains / retunes the frozen model — forbidden (§13) |
| D. **Frozen-output temporal stability across CAL+TEST (adopted)** | valid, leakage-free, zero fitting, reproducible, local-cost; strongest obtainable evidence inside r4.2 |
| E. New dataset/release surrogate | out of contract (principles §1); would require separate authorization; documented as the only path to a true replication window |

D is selected: it maximizes detection validity (all 106 model-unseen days),
leakage resistance (frozen outputs only, pre-registered bounds), statistical
validity (pre-registered, descriptive, no multiple-testing decisions),
reproducibility (deterministic, bit-identical rerun), interpretability
(per-window tables vs fixed bounds), cost (local seconds; optional kernel
scoring < 10 s), and compatibility with the TEST-once contract (TEST annex
descriptive only).

## 27. Execution constraints (for the future authorized run)

- No execution until this specification is approved verbatim or with
  recorded amendments.
- Local analysis is the default path (frozen parquet + records); the
  optional CAL re-scoring stage requires the Kaggle runtime and follows
  `docs/kaggle-execution-policy.md` (health probe, `[JOB]` heartbeats,
  explicit timeouts, gz+b64 + md5 transfer, journaling).
- TEST-once and frozen-system rules apply as written in §7/§11/§13/§24.
- **STOP-ON-FAILURE RULE:** if any safety gate of Section 28 fails at
  execution time, STOP immediately, report the failure with evidence, and
  do **not** repair the protocol, skip the gate, or adjust the data
  silently. Any fix requires an explicit amendment and re-approval before
  the analysis proceeds.

## 28. Execution safety gates (hard requirements, checked before any metric
is calculated)

1. **Unique calendar days**: all temporal windows are verified using UNIQUE
   CALENDAR DAYS. Windows are **never constructed by slicing rows**.
2. **Assertions** (fail-fast, abort on any failure):
   - CALIBRATION = exactly **59 unique days** (2011-02-01..2011-03-31).
   - TEST = exactly **47 unique days** (2011-04-01..2011-05-17).
   - Total = exactly **106 unique days**, with CAL ∩ TEST = ∅.
3. **Partition completeness**: every day (and therefore every row) belongs
   to **exactly one** window; no overlaps, no gaps.
4. **Boundary integrity**: no window crosses the CAL/TEST boundary
   (2011-03-31/2011-04-01); W0–W8 ⊆ CAL, W9–W15 ⊆ TEST.
5. **Pre-registered dates**: the 16 computed windows must equal the date
   table in Section 5 **verbatim** before any metric is calculated.
6. **Frozen-artifact reuse**: existing frozen prediction artifacts are
   reused wherever possible; TEST predictions are **never regenerated**
   (§7, §12).
7. **Verdict isolation**: the Phase 13 verdict function MUST receive
   CAL-only inputs (§24, test-enforced). TEST values must never enter:
   thresholds, bounds, window selection, verdict calculation, parameter
   selection, or model/calibration decisions.

**If any of these gates fails: STOP immediately and report the failure.
Do not repair the protocol silently.**

---

*End of specification. This document authorizes nothing; it defines the
protocol for review. Execution requires explicit approval.*