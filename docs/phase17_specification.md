# Phase 17 Specification — Targeted Calibration/Generalization Intervention on Unseen Users

Status: **SPECIFICATION COMPLETE — EXECUTION NOT AUTHORIZED** (2026-08-18).
This document pre-registers the design, constraints, leakage controls, and
decision criteria of Phase 17. It authorizes **nothing** to run. Execution
occurs only after explicit approval (Section 27). This specification phase
created no artifacts, modified no frozen file, and consulted only
read-only project records.

Every numeric anchor below was re-verified against its source artifact on
2026-08-18 (not taken from memory or chat summaries). Anchors are labeled
OBSERVED (read from a verified artifact) or cited (frozen record quoted).

---

## 0. Objective

Design a **narrowly targeted calibration/generalization experiment** on
unseen users, to be run later (if approved) as a **research-only,
evaluation-only, read-only-input** phase:

1. Diagnose whether the Phase 15 finding R15-1 — *absolute-score
   calibration failure on unseen users' benign rows* (79.8% of 1,026
   false positives within 0.10 above the frozen threshold) — can be
   addressed by a **score-to-probability calibration fitted on the
   user-disjoint CALIBRATION block only** and applied to the frozen
   user-disjoint TEST score artifact.
2. Measure, once, per pre-registered arm, whether calibration **transfers**
   to unseen users (calibration quality on TEST), and whether the
   **calibrated operating point** improves operating metrics on unseen
   users compared with the frozen threshold, **without changing the
   ranking** (mathematical identity for monotone arms, Section 10.1) and
   **without touching the frozen production system**.
3. Pre-register PASS / CAUTION / FAIL bounds, artifact names, tests,
   determinism, and the Phase 18 decision tree so that execution, when
   authorized, is mechanical and cannot be tuned against TEST.

This specification creates **only** `docs/phase17_specification.md`. No
other file is created or modified by this specification phase.

## 1. Scientific question

One question, verbatim from the Phase 17 authorization:

> Can a narrowly targeted calibration/generalization intervention improve
> performance on unseen users without degrading the frozen chronological
> system, while remaining leakage-safe and reproducible?

Decomposition (each component measurable in Phase 17):

| Component | Phase 17 measurement |
|---|---|
| "Improve performance on unseen users" | Operating-point metrics at the calibrated operating point (precision/recall/F1/alert volume) and calibration quality (Brier/ECE) on the user-disjoint TEST, vs the frozen reference arm A |
| "Without degrading the frozen chronological system" | Nothing in the frozen system is modified; immutability gates (Section 4, 15); the chronological TEST record is cited, never recomputed |
| "Leakage-safe" | Calibration fitted on user-disjoint CAL rows only; leakage rule Section 11 |
| "Reproducible" | Deterministic protocol on frozen inputs, double-run determinism gate, recorded environment (Section 16, 26) |

The question is answered as PASS / CAUTION / FAIL per Section 14. This is a
research question about the *calibration layer*, not about the model, the
features, or the production threshold.

## 2. Motivation — Phase 15 evidence (all OBSERVED, re-verified 2026-08-18)

Phase 15 (`reports/phase15_unseen_user_diagnosis_report.md`) decomposed the
unseen-user degradation measured in Phase 14. Verified anchors:

- User-disjoint TEST (108 users, 54,108 rows, 476 positives): AUC-ROC
  **0.786513147174144**, AUC-PR **0.3757395445913874** (OBSERVED,
  `phase14_experiment.json` `test_metrics`); chronological TEST record:
  AUC-ROC 0.939157 (cited, never recomputed).
- At the frozen threshold `0.9186015432508062`: **1,026 alerts**,
  precision 0.20175438596491227, recall 0.43487394957983194,
  F1 0.2756324900133156, MCC 0.2872810379712531, TP 207 / FP 819 /
  TN 52,813 / FN 269, balanced accuracy 0.7098016078448086,
  alert rate 0.018962075848303395 (OBSERVED, `phase14_experiment.json`
  `test_metrics.at_primary_threshold`).
- **R15-1 (calibration failure, the target of Phase 17)**: of the 1,026
  false positives, **819 (79.8%) lie within 0.10 above the frozen
  threshold**; FP margin: min 0.0009, median **0.0447**, max 0.0784
  (OBSERVED, `phase15_threshold_diagnostic.json`). Benign-day mass sits
  low (median 0.131) but a broad benign band hugs the threshold — the
  score scale does not transport to unseen users' benign rows.
- **R15-2 (ranking failures, NOT addressable by calibration)**: the two
  missed malicious users JJM0203 / WDD0366 never alert in 501 days;
  max scores 0.4981 / 0.7264 (OBSERVED, `phase15_detected_vs_missed.json`
  and `phase14_user_diagnostics.json` `zero_detection_users`). A monotone
  score transform cannot lift them above any threshold.
- User-disjoint CAL (108 users, 54,108 rows, **379 positives**): locally
  reproduced scores (gate-verified); **580 rows ≥ frozen threshold
  (135 TP)** vs 1,026 (207 TP) in TEST — both unseen-user cohorts alert
  12–21× more often than the chronological TEST (49) (OBSERVED,
  `phase15_threshold_diagnostic.json`).
- CAL AUC-ROC 0.7355124228200639 (OBSERVED, `phase14_calibration.json`;
  locally reproduced with abs diff 0.0 in `phase15_experiment.json`
  gate 2).
- Secondary (best-F1 on user-disjoint CAL, descriptive in Phase 14):
  threshold 0.9837865316173778 → on TEST **191 alerts**, precision
  0.7749, recall 0.3109, F1 0.4438, MCC 0.4883 (OBSERVED,
  `phase14_experiment.json` `test_metrics.at_secondary_threshold`).
- Coverage at the frozen threshold: **13/15** malicious users
  (Wilson 90% CI [0.6885, 0.9328]) (OBSERVED,
  `phase14_user_diagnostics.json` `coverage`). Coverage at the
  secondary/calibrated operating point was **not recorded** per-user in
  Phase 14 → it is a genuinely new Phase 17 measurement (Section 12.3).

Reading that motivates Phase 17 (INFERENCE, from the verified anchors
above): the alert explosion is a **calibration transport failure of the
absolute score scale** (R15-1), separable from ranking quality. A
CAL-fitted calibration mapping is the minimal, leakage-safe intervention
that can test whether the score semantics of unseen users are recoverable;
the two user misses (R15-2) are out of reach for any score transform and
are pre-registered as expected to persist (Section 14, expectation note).

## 3. Evidence hierarchy and verification rules

Priority (highest first) when any two project records disagree:

1. **Verified frozen artifacts** (`reports/artifacts/*`, md5-verified
   against their manifests).
2. **Source code and tests** (`src/`, `tests/`) — implementation truth.
3. **Final phase reports** (`reports/*.md`) — measured records.
4. **Knowledge layer** (`knowledge/`) — summaries, never authoritative.
5. **Master report** (`reports/ISM_MASTER_REPORT.md`) — cumulative summary
   with its own §25–30 validation gates.
6. **Previous chat output** — never evidence.

Rules:
- No number is written into any Phase 17 artifact or report from memory;
  it is read from the artifact, and the artifact's md5 (where applicable)
  is recorded in `phase17_experiment.json`.
- Any discrepancy is recorded in the Phase 17 report with both values,
  both files, the authoritative source, and the resolution (master report
  §26 convention); it is never silently resolved.
- Every Phase 17 statement is labeled OBSERVED / INFERENCE / HYPOTHESIS /
  LITERATURE RESULT / NOT VERIFIED (ISM principles §12).
- Anything computed from a verified artifact in this specification phase is
  labeled; nothing here is a runtime result.

## 4. Frozen baseline (immutable, verbatim from the authorization)

- Dataset: CERT r4.2 only (raw data never leaves Kaggle; never copied to
  the PC).
- Analytical unit: user × day (501,000 rows, 1,000 users, 501 full
  timeline days per user; 70 malicious users, 1,892 malicious rows).
- Production model: `lgbm-graph-v1` — LightGBM, 12 registry features
  (8 behavioral + 4 graph), seed 42, best_iteration 186, lightgbm 4.6.0
  (kernel), scale_pos_weight from chronological TRAIN.
- Production threshold: `0.9186015432508062` (frozen, Phase 9).
- Chronological split (authoritative): TRAIN ≤ 2011-01-31
  (395,000 rows / 1,539 positives), CAL 2011-02-01..2011-03-31
  (59,000 / 323), TEST ≥ 2011-04-01 (47,000 / 30, evaluated exactly once).
- Auxiliary user-disjoint model (Phase 14, itself frozen now):
  `lgbm-graph-v1-uhold` — same 12 features and LightGBM config, seed 42,
  scale_pos_weight 377.7695 (TRAIN only), early stopping on user-disjoint
  CAL AUC (patience 100, ≤ 3000), best_iteration 214. Model file
  `phase14_model.txt` (750,807 B), md5
  `3778a4d869f7231e76ec4c08c6dfa419`.

Immutability: none of the above changes in Phase 17, including in the
specification phase (verified: no file under `src/`, `reports/artifacts/`,
`reports/*.md`, `knowledge/`, `docs/` was modified on 2026-08-18 by this
specification task; the only file created is this specification).

## 5. Research-only candidate definition

Phase 17 may experiment **only** with the **calibration layer on top of
frozen scores**:

- **Candidate interventions**: score→probability mappings fitted on the
  user-disjoint CAL block (108 users) and applied to the frozen TEST score
  artifact. Three candidates (arms B, C, D) plus the frozen reference
  (arm A), fully defined in Section 9.
- **Allowed**: reading frozen artifacts; computing descriptive statistics;
  fitting calibration on CAL rows; applying transforms to frozen TEST
  scores in memory; computing pre-registered metrics; bootstrap;
  determinism double run.
- **Never allowed**: retraining any model; changing any feature; changing
  the frozen threshold; selecting methods/thresholds/hyperparameters using
  TEST; scoring chronological rows; writing transformed score tables
  (JSON summaries only); any production recommendation.
- **Data support gate** (Section 6): the candidates exist only if the
  audited inputs contain the required CAL labels + scores and TEST scores.
  All inputs exist and are verified (OBSERVED, Section 6) → candidates
  B/C/D are admissible.
- Any arm that requires data or statistics not available at prediction
  time is rejected by construction (Section 11).

## 6. Data availability audit (OBSERVED, re-verified 2026-08-18)

All inputs are **local, frozen, read-only**. No Kaggle execution, no
dataset rebuild, no raw data access is required (Section 19).

| Input | Rows / shape | Role | Verified md5 (source) |
|---|---|---|---|
| `reports/artifacts/phase14_split.json` | allocation + block counts | user-disjoint allocation (TRAIN 784 / CAL 108 / TEST 108; malicious 40/15/15) | `da013f825246d568bcfdaf19dcf2c7e7` (`phase15_experiment.json` gate 3) |
| `reports/artifacts/phase14_test_predictions.parquet` | 54,108 × 6 (user, day, is_malicious, score, alert_primary, alert_secondary) | **TEST scores** — the only TEST input; never re-scored, never re-evaluated | `71eeb3f1948e518518a53e062d5a213d` (gate 3) |
| `reports/artifacts/phase15_cal_scores.parquet` | 54,108 × 4 (user, day, is_malicious, score) | **CAL scores** (gate-verified local reproduction of the auxiliary model) — calibration fitting data | `c955a4ec6ecaf7e9baff80abeb361330` (`phase15_manifest.json`) |
| `reports/artifacts/phase14_model.txt` + `phase14_model_record.json` | model file + record | only for the CAL rescoring fallback (gate 2 alternative, Section 9/19) | `3778a4d869f7231e76ec4c08c6dfa419` |
| `reports/artifacts/phase6_merged_features.parquet` | 501,000 × 16 | read-only; only for md5 gate and (optional) user-activity tercile cross-check | `9a3b188573bb953416981dfea3379def` (gate 3) |
| `reports/artifacts/phase14_user_diagnostics.json` | 15 TEST users, per-user fields | activity tercile / onset / length class for coverage sub-analysis (reuse, no recomputation) | `cd893661a3a0c08f9066e14c0bf8dd07` (gate 3) |
| `reports/artifacts/phase14_experiment.json`, `phase14_test_metrics.json`, `phase14_calibration.json`, `phase14_bootstrap.json`, `phase14_manifest.json` | recorded Phase 14 results | reference values for gates 1–2 and arm-A reproduction | `phase14_manifest.json` |
| `reports/artifacts/phase15_experiment.json`, `phase15_threshold_diagnostic.json`, `phase15_cohort_comparison.json`, `phase15_manifest.json` | recorded Phase 15 results | R15-1 anchors; gate 2 reference | `phase15_manifest.json` |
| `reports/artifacts/phase7_freeze_lgbm-graph-v1.json`, `phase9_freeze.json`, `phase12_test_policy_results.json` | frozen production records | chronological TEST record (cited, never recomputed) | recorded in their manifests |

Structural audit (OBSERVED, 2026-08-18): both score tables have exactly
54,108 rows; TEST has 476 positives, CAL has 379 positives; no nulls, no
duplicate keys (Phase 14 gates). Scores in [0, 1]. The two cohorts share
no users with each other or with TRAIN (Phase 14 allocation, asserted at
execution). A calibration fit needs only (score, label) pairs of CAL rows
and the score column of TEST rows — both available at prediction time
(scores are outputs of the frozen auxiliary model on day-local features;
Section 11).

Feasibility verdict: **FEASIBLE, fully local, low-cost** — same class of
analysis Phase 15 already ran in 9.5 s.

## 7. User-disjoint split protocol

- Reuse the **Phase 14 allocation verbatim** (`phase14_split.json`,
  asserted equal to the pre-registered table of `docs/phase14_specification.md`
  §6). No re-allocation, no new split, no random split of any kind.
- Rule (recorded, deterministic, seed-free): malicious users sorted by
  (first_positive_day, user_id), round-robin period 14 → 40/15/15;
  benign users sorted by user_id, round-robin period 10 → 744/93/93.
- Blocks: TRAIN 784 users / 392,784 rows / 1,037 positives / 40 malicious;
  CAL 108 / 54,108 / 379 / 15; TEST 108 / 54,108 / 476 / 15.
- Roles in Phase 17: TRAIN contributes **nothing** (its scores would be
  in-sample/overfit-biased; Phase 15 §4 convention); CAL is the **only**
  fitting block (calibration + operating point); TEST is **evaluated once**
  from the frozen score artifact (Section 15).
- Block invariance: all users span the full 501-day timeline; temporal
  position is block-invariant by construction (Phase 15 §3), so any
  CAL→TEST difference is entity-level, not calendar-time (Phase 15 §9
  attribution, OBSERVED).

## 8. Calibration protocol

Fit rule (applies to every arm):

1. Calibration is fitted **only** on CAL rows: (score s, label y) pairs
   from `phase15_cal_scores.parquet` (54,108 rows, 379 positives).
2. Input order is fixed before fitting: sort by (user, day) — this fixes
   any order-sensitive fit (Platt via lbfgs is order-insensitive;
   binning/ECDF are order-insensitive by construction; the sort is
   recorded for determinism).
3. The fitted mapping is applied to the frozen TEST score column
   (54,108 rows) in memory only. No TEST row, label, score, or statistic
   enters any fit.
4. The operating point of each arm is a **single pre-registered rule**
   (Section 9) computed from CAL scores only — never from TEST, and never
   iterated.
5. Everything the transform needs is available at prediction time: the
   current day's model score. No user identity, no activity tercile, no
   feature value, no cross-user statistic is used in the base protocol
   (Section 11).
6. Fitted parameters are recorded in `phase17_calibration_fit.json` with
   their fitting data summary (n rows, positives, score range).

No calibration method, bin count, regularization value, or operating-point
rule may be changed after this specification is approved except by
documented amendment (Section 27).

## 9. Candidate arms A–D

Every arm is fully defined here. No arm may be added, removed, or altered
at execution. Arm A is the reference; B/C/D are the candidate
interventions.

**Arm A — Frozen reference (no intervention).**
- No transform. Operating point: the frozen threshold
  `0.9186015432508062`.
- Metrics recomputed from the frozen TEST artifact = the Phase 14 primary
  operating-point row; this reproduction IS gate 1 (Section 9.5) and is
  the reference for B/C/D comparisons. Reference only; not an experiment.

**Arm B — Simple calibration: Platt scaling.**
- Fit: logistic regression of y on s over the 54,108 CAL rows:
  `p̂(s) = σ(a·s + b)`, σ(z) = 1/(1+e^{−z}), fitted by maximum likelihood
  with `sklearn.linear_model.LogisticRegression(C=1.0, fit_intercept=True,
  solver="lbfgs", max_iter=1000, tol=1e-4)` on the single feature s
  (sklearn 1.9.0, PC).
- Monotone by construction if a > 0. **Pre-registered abort**: if a ≤ 0
  (numerically impossible for a well-fitted binary problem with
  monotone score separation, but not assumed), arm B is reported as
  FAILED with the fit diagnostics and the phase continues with C and D.
- Operating point: best-F1 threshold on CAL under p̂ (Section 10.3).
  By the monotone-invariance identity (Section 10.1) this equals the
  Phase 14 secondary operating point (threshold 0.9837865316173778 in raw
  score space) — an OBSERVED equality expected and checked in gate 1b.

**Arm C — Alternative calibration: binning-based recalibration.**
- Fit (on CAL rows only):
  1. K = 20 equal-frequency bins over CAL scores (boundaries at the
     quantiles k/20 of the CAL score distribution, k = 1..19; interior
     boundaries at observed score values).
  2. Within-bin positive rate r_k = tp_k / n_k. **Pre-registered merge
     rule**: any bin with n_k < 30 is merged with its smaller neighbor
     (ties: left neighbor); repeat until every bin has n ≥ 30.
  3. Monotone smoothing: pool-adjacent-violators (PAV) over the merged
     bin rates (non-decreasing constraint), producing rates r'_k.
  4. Mapping: p̂(s) = r'_k for s in bin k (step function, non-decreasing).
- If after merging fewer than 5 bins remain, arm C aborts and is reported
  (pre-registered; not expected — CAL score spread is wide, OBSERVED
  Section 6).
- Operating point: best-F1 threshold on CAL under p̂. Because p̂ is a
  step function, the operating point can alert **whole bins** — the alert
  set may therefore differ from B/D at bin boundaries (tie effects,
  Section 10.1). This is the arm's genuine difference, not a defect.

**Arm D — Calibrated probability mapping (rank-preserving).**
- Fit: empirical CDF remap `p̂(s) = ECDF_CAL(s)` = fraction of CAL scores
  ≤ s, with ties resolved by midpoint rank (a row with the same score as
  m others receives rank mid(m+1) of the tied block). Strictly monotone in
  s except at ties.
- Operating point: best-F1 threshold on CAL under p̂. By monotone
  invariance this equals the raw-CAL best-F1 point (Phase 14 secondary
  threshold in raw space) — same expected alert set as B.
- Role: the simplest rank-based recalibration; serves as a sensitivity
  check on the method choice B vs C (B and D are expected to coincide at
  the operating point; C may differ only through bin-level alerting).

**Arm comparison contract**: for strictly monotone arms (B, D) the TEST
ranking is unchanged by a mathematical identity (Section 10.1), so ranking
metrics are verification quantities (must match recorded Phase 14 values),
while operating-point and calibration-quality metrics are the comparison
quantities. Arm differences at the operating point are expected to be
**nil between B and D** and **possibly non-nil for C** (bin-level
alerting). If B/C/D coincide at the operating point, that is the
pre-registered expectation, recorded as OBSERVED, not a failure.

### 9.5 Gates (before any Phase 17 statistic is accepted)

1. **Gate 1 — arm-A reproduction on TEST**: recompute from the frozen
   TEST artifact: AUC-ROC, AUC-PR, alert count, precision, recall, F1,
   MCC, TP/FP/TN/FN at the frozen threshold. Must match the recorded
   Phase 14 values bit-exactly or within 1e-12 relative
   (`phase14_experiment.json` `test_metrics`): AUC-ROC
   0.786513147174144, AUC-PR 0.3757395445913874, 1,026 alerts, precision
   0.20175438596491227, recall 0.43487394957983194, F1
   0.2756324900133156, MCC 0.2872810379712531. Also cross-check
   `alert_primary == (score ≥ threshold)` column agreement.
2. **Gate 1b — arm-B/D operating-point equality**: B and D CAL best-F1
   thresholds must equal the recorded Phase 14 secondary threshold
   0.9837865316173778 within 1e-12 (monotone-invariance verification).
3. **Gate 2 — CAL artifact reproduction**: recompute from
   `phase15_cal_scores.parquet`: 54,108 rows, 379 positives; 580 rows ≥
   frozen threshold with 135 TP (Phase 15 §8, OBSERVED); CAL AUC-ROC =
   0.7355124228200639 within 1e-9 (Phase 14 gate 2 convention). If this
   gate fails, re-derive CAL scores locally from `phase14_model.txt`
   (lightgbm 4.7.0, PC; Phase 15 gate 1 proved max diff 5.6e-17 vs the
   kernel) and re-check; if the re-derived scores pass, record the
   replacement and its md5 in the experiment artifact; if they also fail,
   STOP (Section 20).
4. **Gate 3 — input md5s**: all Section 6 inputs md5-verified at run
   start against the recorded values; recorded in
   `phase17_experiment.json`.
5. **Gate 4 — frozen-system immutability**: the Phase 16
   final-verification md5 set (51 non-JSON artifacts) unchanged after the
   run; specifically phase14/15 score and record artifacts and the
   production freeze records (P7/P9/P12) unchanged.

## 10. Mathematical definitions

### 10.1 Monotone-invariance identity (mathematical, not measured)

For any strictly increasing transform g, the ranking of scores is
unchanged: s_i < s_j ⟺ g(s_i) < g(s_j). Consequently, for the whole
user-disjoint TEST cohort, AUC-ROC and AUC-PR (sklearn
`roc_auc_score`/`average_precision_score`, which depend only on score
order and label order) and P@k/R@k are **identical** before and after g.
The best-F1 threshold (as defined in `src/evaluation/threshold.py`
`best_f1_threshold`, ties broken toward the higher threshold) is invariant
under g up to exact tie sets: argmax over {t} maps 1:1 to argmax over
{g(t)}. For a non-decreasing step function (arm C), order is preserved
(no inversion), so AUC-ROC is unchanged; AUC-PR/P@k can change only
through tie-break effects at bin boundaries (reported OBSERVED, no
tolerance claim; Section 12.4).

Corollary (pre-registered): **no arm can improve ranking metrics on the
user-disjoint TEST**; any report claiming a ranking gain would be a
pipeline error. "Performance" for Phase 17 therefore means operating-point
and calibration-quality metrics, with ranking as a verification quantity.

### 10.2 Transforms

- Platt (B): p̂(s) = 1/(1 + exp(−(a·s + b))), fitted per Section 9.
- Binning (C): step function over merged bins with PAV-smoothed rates.
- ECDF (D): p̂(s) = (rank_mid(s) − 1/2)/54,108 over CAL rows
  (midpoint-rank convention; equivalently the fraction of CAL scores
  strictly below s plus half the fraction equal to s).

### 10.3 Operating-point rule (single, pre-registered, CAL-only)

`t_arm` = `best_f1_threshold(y_CAL, p̂_CAL)` from
`src/evaluation/threshold.py` (exact reference semantics: ascending
unique-score grid, `pred = score >= t`, strict-greater F1 update with
1e-12 tolerance). Applied once; never TEST-informed; never iterated.

### 10.4 Calibration-quality metrics (pre-registered; 10 equal-frequency bins, fixed)

- Brier: B = mean over rows of (p̂ − y)². Raw (arm A) Brier uses the raw
  score as p̂ — it is the uncalibrated reference, not a fitted quantity.
- ECE: 10 equal-frequency bins over the scores being evaluated (CAL or
  TEST; bins fitted on the evaluated cohort's own scores — this is a
  descriptive summarization, not a fit used for prediction);
  ECE = Σ_k (n_k/n) · |mean(p̂)_k − mean(y)_k|. Fixed K = 10; no tuning.
- Reliability data: (bin center p̂, empirical positive rate, n) per arm
  per cohort, stored for the report figure.
- Both computed on CAL (in-sample description of the fit) and on TEST
  (the transfer measurement). TEST values are computed once from frozen
  scores + the CAL-fitted transform.

### 10.5 Operating metrics (TEST, frozen artifact, per arm)

Alerts, alert rate (per user-day and per calendar day), TP/FP/TN/FN,
precision, recall, F1, MCC, balanced accuracy, FPR, FNR — all as
`src/evaluation/metrics.py` defines them at the arm's operating point
`t_arm` (arm A: the frozen threshold).

### 10.6 Coverage

Malicious-user coverage = fraction of the 15 TEST malicious users with
≥ 1 alert at the arm's operating point; Wilson 90% CI
(z = 1.6448536269514722); per-user alert counts; zero-alert users listed;
Gini of alert concentration over alerted users (Phase 9 convention);
alerts per user (mean/median/max).

## 11. Leakage controls

Central rule (verbatim from the authorization): **"Would this information
be available at prediction time?"** — every fit, transform, and operating
point must be answerable with yes.

- **Fitting data**: only (CAL score, CAL label) pairs. Scores are model
  outputs; labels are the frozen label column of CAL users. Nothing else.
- **No future information**: CAL users' full-501-day rows are the
  calibration population; TEST users' rows are never used in any fit.
  No statistic computed from TEST or from the chronological record enters
  any fit (chronological records are cited, not fitted).
- **No user-level conditioning in the base protocol**: activity terciles,
  onset position, or any per-user characteristic are NOT used in the
  calibration fit (a full-timeline tercile is not available at prediction
  time for a per-day decision). Per-user fields from
  `phase14_user_diagnostics.json` are used **only** in the descriptive
  coverage sub-analysis (Section 12.6), never in any fit or operating
  point.
- **No cross-user statistics**: the four frozen graph features are
  strictly-past per-user (Phase 14 §4 feasibility audit); Phase 17 adds
  no new statistic of any kind.
- **No TEST-driven selection**: methods, bin counts, regularization,
  operating-point rules are fixed in this specification; nothing computed
  on TEST chooses, ranks, or re-derives anything. Arm comparisons are
  pre-registered measurements, not selections.
- **No threshold hunting**: exactly one operating-point rule (Section
  10.3); the frozen threshold is a constant reference for arm A only.
- **No chronological TEST contact**: the 47,000-row chronological TEST is
  never read or scored (Section 15).
- **No writes to inputs**: all inputs read-only; all outputs are new JSON
  artifacts; no transformed score table is written (in-memory only,
  Section 17).
- **No raw data**: raw CERT logs/answer key remain on Kaggle; Phase 17
  touches only derived artifacts already on the PC.

## 12. Metrics

### 12.1 Primary comparison quantities (TEST, once)

- Calibration transfer: TEST Brier (calibrated vs raw) and TEST ECE per
  arm; reliability data.
- Operating point: precision, recall, F1, MCC, balanced accuracy, alert
  count/rate at each arm's operating point; user-level coverage (Wilson
  90% CI).

### 12.2 Verification quantities (must match records; not "improvements")

- Ranking metrics per arm: AUC-ROC, AUC-PR, P@10/30/50/100, R@10/30/50/100
  vs the recorded Phase 14 values (AUC-ROC 0.786513147174144, AUC-PR
  0.3757395445913874, P@10 1.0, P@30 1.0, P@50 1.0, P@100 0.95,
  R@10 0.02100840336134454, R@30 0.06302521008403361,
  R@50 0.10504201680672269, R@100 0.19957983193277312).
- Arm-B/D operating threshold vs Phase 14 secondary threshold
  0.9837865316173778 (gate 1b).

### 12.3 New measurements (not present in Phase 14/15 records)

- Coverage, per-user alerts, and Gini at the calibrated operating point(s)
  (Phase 14 recorded coverage only at the frozen threshold; per-scenario
  recall at secondary only).
- Brier/ECE/reliability for the auxiliary model's scores on unseen users
  (not previously computed).

### 12.4 Tie-effect reporting (arm C)

AUC-PR and P@k/R@k deltas for arm C are reported OBSERVED with the exact
tie sets at the affected boundaries; no tolerance is claimed. Gate 1b does
not apply to C.

### 12.5 Sub-analyses (descriptive, pre-registered)

- Coverage sub-analysis by activity tercile / length class / onset half,
  reusing per-user fields from `phase14_user_diagnostics.json` (no
  recomputation). n is small (15 users); descriptive only, no
  significance tests.
- Zero-alert users at each arm's operating point (expect JJM0203,
  WDD0366 in every arm; if any arm alerts them, that is reported as an
  OBSERVED surprise with the responsible bin/tie details).
- Scenario recall at each arm's operating point using the Phase 14
  scenario mapping (NOT VERIFIED locally — recorded per Phase 14 §4; a
  scenario with zero positive rows is reported NOT VERIFIED, never
  imputed).

### 12.6 Reporting conventions

- Operating-point metrics of the enriched user-disjoint cohort are
  cohort-conditional and NOT directly comparable to the chronological
  record (Phase 14 §4/§20 convention); reported side by side with the
  caveat, never merged.
- All values labeled OBSERVED / INFERENCE / HYPOTHESIS.

## 13. Bootstrap and uncertainty

- **User-block bootstrap** (Phase 14 primary estimator): resample TEST
  users with replacement (n = 1000, seed 42), pool rows, recompute — for
  (a) AUC-ROC and AUC-PR per arm; (b) operating-point metrics (alerts,
  precision, recall, F1) per arm. Honest for the panel structure
  (within-user row correlation). `n_skipped` recorded.
- **Calibration-transfer bootstrap** (new, pre-registered): resample CAL
  users with replacement (n = 1000, seed 42), **refit the arm's
  calibration inside each resample**, apply to the fixed frozen TEST
  score artifact, and record the TEST Brier/ECE distribution per arm
  (90% percentile intervals). This measures fit variance — the honest
  uncertainty for a 108-user calibration fit. Expected runtime is small
  (refit cost: Platt < 1 s, binning < 1 s, ECDF < 1 s per resample;
  Section 21).
- **Row-level bootstrap** (project convention, Phase 6/14): reported for
  comparability, flagged as optimistic given within-user correlation.
- **Wilson 90% CI** (z = 1.6448536269514722) for coverage.
- No refitting of the auxiliary model ever (frozen); no resampling of
  TEST for any fitting purpose.

## 14. PASS / CAUTION / FAIL success criteria (pre-registered)

All gates (Section 9.5) must pass for any verdict other than FAIL.

**PASS** (all five must hold; bounds anchored on Phase 14/15 OBSERVED
values):

1. **Calibration transfers**: at least one calibrated arm (B, C, or D)
   improves TEST Brier by ≥ 20% relative vs the raw-score Brier AND has
   TEST ECE ≤ 0.05.
2. **Operating-point improvement on unseen users**: at that arm's
   operating point on TEST — F1 ≥ 0.40, precision ≥ 0.70, alert count
   ≤ 300, and user-level coverage ≥ 8/15 (Wilson CI reported). (Anchors:
   arm A F1 0.2756 / precision 0.2018 / 1,026 alerts / coverage 13/15;
   Phase 14 secondary F1 0.4438 / precision 0.7749 / 191 alerts.)
3. **Ranking preserved**: |ΔAUC-ROC| ≤ 1e-9 vs the recorded
   0.786513147174144 for every arm (strictly monotone arms expected
   ≤ 1e-12); AUC-PR/P@k deltas reported per Section 12.4.
4. **Coverage not degraded below the pre-registered bound** (≥ 8/15) at
   every calibrated arm's operating point.
5. **Frozen system untouched**: gate 4 immutability, plus no
   chronological record read beyond the cited records, plus no
   production file modified.

**CAUTION** (any of): calibration transfers (criterion 1) but operating
criterion 2 fails (e.g., F1 < 0.40 or precision < 0.70 or alerts > 300 or
coverage < 8/15); OR operating criterion 2 holds but calibration does not
transfer (criterion 1 fails); OR a gate passes only after the documented
fallback (gate 2 replacement scores); OR arm C tie effects move AUC-PR or
P@k beyond trivial values (> 1e-3 absolute) with a documented cause.

**FAIL** (any of): any gate fails without a documented, pre-registered
fallback; no calibrated arm improves TEST Brier by ≥ 20%; ranking not
preserved for a strictly monotone arm beyond 1e-9; any TEST contact beyond
the frozen-artifact read; any frozen artifact modified; the deterministic
double run is not byte-identical; any evidence of TEST-informed selection.

**Pre-registered expectation notes** (not criteria):
- JJM0203 / WDD0366 are expected to remain zero-alert in all arms
  (ranking failures; Section 2 R15-2). Coverage ceiling 13/15.
- B and D are expected to produce identical alert sets (monotone
  invariance); C may differ only through bin-level alerting.
- PASS is consistent with "the calibrated operating point recovers the
  Phase 14 secondary behavior with its own verified pipeline" — the
  phase's new contribution is the calibration-transfer measurement and
  the arm comparison, not the existence of a better operating point.

## 15. TEST-once protection

- The **authoritative chronological TEST** (47,000 rows, 30 positives,
  2011-04-01..2011-05-17) is never read, scored, or recomputed. Its
  frozen record (AUC-ROC 0.939157, 49 alerts, etc., P12 artifacts) is
  cited only. Phase 15 §2, §9 convention.
- The **user-disjoint TEST** is not re-evaluated: no model re-run, no
  score regeneration. Only the frozen `phase14_test_predictions.parquet`
  score column is read, pre-registered transforms are applied in memory,
  and pre-registered metrics are computed once (identical in kind to
  Phase 15's read-only recomputations of alerts/margins).
- The deterministic **double-run gate** re-executes the whole protocol
  within the same authorized run and asserts byte-identical output; per
  the Phase 14 convention (report §1/§3) this is a determinism check, not
  a second evaluation.
- No TEST-derived quantity feeds any fit, threshold, selection, or
  artifact other than the pre-registered metrics themselves.

## 16. Determinism

- No RNG anywhere in transforms, gates, or metrics (bootstrap uses
  `seed 42` with the Phase 14 sampling convention).
- Fixed input order (sort by (user, day)) for any order-sensitive step.
- Fixed library versions (Section 26): Python 3.11.9, numpy 2.4.6,
  pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1, lightgbm 4.7.0
  (used only in the gate-2 fallback).
- Determinism gate: run the full Phase 17 pipeline twice; all JSON
  artifacts must be byte-identical (`determinism: True` recorded, with
  runtimes).
- Environment recorded in `phase17_experiment.json` (OBSERVED at
  execution).

## 17. Required artifacts (created ONLY at execution, after approval)

`reports/artifacts/`:

- `phase17_experiment.json` — experiment id, scope, inputs + md5s
  (gate 3), gates 1/1b/2/4 results, environment, determinism flag,
  evidence labels, frozen-system statement.
- `phase17_calibration_fit.json` — per arm: fit parameters (Platt a, b;
  merged bin boundaries + PAV rates; ECDF summary), fitting data summary,
  abort records if any.
- `phase17_calibration_metrics.json` — Brier, ECE, reliability data per
  arm per cohort (CAL, TEST); raw (arm A) values; calibration-transfer
  bootstrap distributions (Section 13).
- `phase17_operating_points.json` — per arm: operating threshold (raw and
  transformed), TEST operating metrics (Section 10.5), coverage +
  Wilson CI, per-user alerts, zero-alert users, Gini, scenario recall
  (or NOT VERIFIED), alerts per user (mean/median/max).
- `phase17_ranking_verification.json` — per arm: AUC-ROC, AUC-PR,
  P@10/30/50/100, R@10/30/50/100, deltas vs the recorded Phase 14 values,
  tie-effect notes for arm C.
- `phase17_manifest.json` — md5s of all Phase 17 JSON artifacts
  (self-excluded per the Phase 14/15 convention).
- `phase17_cost.json` — wall time per stage, total.

Also (execution-time, after approval): `src/experiments/phase17.py`,
`kaggle_scripts/run_phase17.py` (local runner), `tests/test_phase17.py`,
`reports/phase17_calibration_transfer_report.md`, and (after verified
execution) the master report + knowledge layer updates per ISM principles
§15 — **none of these exist yet**; this specification creates none of
them.

## 18. Required tests (local; before any result is recorded)

`tests/test_phase17.py`:

- Unit tests for each transform: Platt fit + monotonicity (a > 0) on
  synthetic fixtures; binning edge cases (constant scores, single-score
  CAL, n < 30 merge rule, PAV output monotone); ECDF tie handling
  (midpoint ranks); best-F1 operating-point semantics match
  `best_f1_threshold`.
- Metric tests: Brier/ECE on fixtures with hand-computed values; coverage
  and Wilson CI on fixtures; user-block bootstrap on a small synthetic
  panel (Phase 14 convention; n_skipped recorded).
- Edge cases: all-positive / all-negative CAL fixture, empty cohort,
  duplicate scores, null-free assertions on the real artifacts.
- Determinism test (double-run byte-identical JSON).
- Artifact reloadability/schema tests for all Phase 17 artifacts.
- Gate-machinery test on a small fixture (reproduction of a recorded
  number from a fixture, not from the real artifacts).
- The full local suite must pass; exact counts reported (Phase 16 suite:
  379 passed / 4 skipped).

## 19. Kaggle requirements and recovery documentation

- **Kaggle is NOT required for Phase 17.** Every input is a local frozen
  artifact (Section 6); no training, no dataset rebuild, no raw data, no
  lightgbm scoring (except the gate-2 fallback, which is local:
  `phase14_model.txt` + lightgbm 4.7.0 — Phase 15 gate 1 already proved
  this path, max abs diff 5.6e-17 vs the kernel).
- **When Kaggle WOULD be required** (documented, not planned): only if an
  input artifact were lost AND regeneration were authorized — then the
  documented recovery paths apply: derived-table rebuild runbook D
  (`docs/kaggle-execution-policy.md`) for `phase6_merged_features.parquet`;
  auxiliary-model regeneration ONLY by explicit authorization (recorded
  params, seed 42, recorded split, lightgbm 4.6.0 kernel; Phase 14
  report §2) — never executed in Phase 17.
- If any input md5 mismatches at run start: STOP, record, report; do not
  substitute any file.
- The frozen TEST predictions parquet is **irreplaceable** (TEST-once):
  if it is missing or corrupt, Phase 17 cannot run; report and await
  authorization (no regeneration ever).
- Execution-time journaling (`kaggle_scripts/kaggle_exec.py log job ok`)
  is a run-time step, not part of this specification phase.

## 20. Failure and abort conditions

- Any gate (Section 9.5) fails without a pre-registered fallback → STOP,
  record, report; no partial metrics are accepted as results.
- Arm-specific aborts (Platt a ≤ 0; binning < 5 bins after merging) → arm
  reported FAILED with diagnostics; other arms continue (pre-registered).
- Test suite failures → STOP.
- Any frozen artifact hash change during the run → STOP and investigate
  (Phase 16 convention).
- Determinism double-run not byte-identical → STOP.
- Any sign of TEST-informed selection or chronological TEST contact →
  STOP; the phase is void and must be re-authorized.
- Resource failure (crash, OOM) → artifacts written up to the failure
  point are retained for diagnosis; the run is restarted only with
  authorization (the double-run gate still applies to the completed run).

## 21. Expected computational cost

- Data: reads ≤ 110,000 rows (two 54,108-row score tables + small JSONs).
- Fits: three calibration fits on 54,108 rows (Platt, binning, ECDF) —
  seconds each.
- Metrics: once over 54,108 rows per arm.
- Bootstrap: 1000 user-block resamples × (AUC + operating metrics) per
  arm; 1000 calibration-refit resamples for the transfer interval —
  expected total wall time **< 5 minutes local** (Phase 15's heavier
  diagnostic run: 9.5 s; the calibration-refit bootstrap dominates).
- No GPU, no Kaggle quota, no large disk writes (JSON artifacts only,
  no transformed-score tables).
- Measured cost recorded in `phase17_cost.json` (OBSERVED at execution).

## 22. Risks (pre-registered R17)

- **R17-1 Calibration may not transfer**: unseen-user score semantics may
  not be recoverable by any CAL-fitted monotone map (entity-specific
  inflation patterns). Outcome: CAUTION/FAIL — informative either way;
  the phase's value is the measurement.
- **R17-2 Cohort enrichment**: operating-point metrics on the user-disjoint
  TEST are not production-comparable (Phase 14 §4/§20). Mitigation:
  reported with the standing caveat; ranking + coverage are the
  generalization evidence.
- **R17-3 Small calibration fit (108 users, 379 positives)**: bin rates
  and Platt fit are noisy; 20 bins ≈ 19 positives/bin. Mitigation: merge
  rule; calibration-transfer bootstrap (Section 13) quantifies fit
  variance honestly.
- **R17-4 Arm-C tie effects**: bin-level alerting can move AUC-PR/P@k via
  ties; reported OBSERVED with tie sets (Section 12.4); no tolerance
  claim.
- **R17-5 "Without degrading the frozen system" is by-construction, not
  measured**: Phase 17 changes nothing; production impact of any
  calibrated operating point is NOT measured (would require authorized
  future work; the chronological TEST is off-limits forever). The report
  must state this explicitly.
- **R17-6 Cross-model score spaces**: the Phase 17 calibration lives on
  the auxiliary model's score space; the frozen model's chronological
  scores are a different space (186 vs 214 trees; Phase 15 §9). No
  cross-model operating-point transfer is attempted or claimed.
- **R17-7 Scope creep**: new features, retraining, graph work, or
  user-level conditioning would invalidate the phase; pre-registered
  non-goals (Section 24) bind execution.
- **R17-8 Library sensitivity**: sklearn/numpy versions recorded
  (Section 16/26); determinism gate covers refits.
- **R17-9 Overclaiming**: PASS ≠ production improvement; the report
  distinguishes research evidence from production claims (Section 14,
  R17-5).

## 23. What remains frozen

- The production model `lgbm-graph-v1`, its weights, config, seed, and
  iteration; the auxiliary model `lgbm-graph-v1-uhold` and its record.
- The production threshold `0.9186015432508062` and the Phase 9 alert
  policy; the Phase 10 conformal method; the Phase 11 explainability
  contract.
- The feature registry, all feature definitions, and the graph table.
- The chronological split; the authoritative chronological TEST record;
  the user-disjoint Phase 14 allocation (`phase14_split.json`).
- All historical artifacts, phase reports, and previous phase results
  (including `phase14_test_predictions.parquet`,
  `phase15_cal_scores.parquet`).
- The master report, decision log, experiment index, knowledge READMEs
  (unchanged in the specification phase; updated only after an approved
  and verified execution, per ISM principles §15).
- The raw dataset (Kaggle-only) and all production decisions.

## 24. Explicit non-goals

Phase 17 does NOT: add or change features; retrain or fine-tune any model;
change the model architecture (no GNNs, transformers, deep learning);
change the frozen threshold; hunt for thresholds (one pre-registered rule
per arm); tune hyperparameters (bin count, C, operating rules fixed here);
condition calibration on user identity, activity tercile, or any
prediction-time-unavailable variable; use random or temporal splits (the
allocation is frozen and user-disjoint); re-score or re-evaluate any TEST;
write transformed score tables; execute on Kaggle; touch raw data; touch
the answer key (scenario mapping reused from Phase 14 records as-is,
NOT VERIFIED locally); recommend or implement any production change; update
the master report or knowledge layer during the specification phase;
begin Phase 18 work of any kind.

## 25. Decision tree for Phase 18 authorization (pre-registered)

Phase 17 never self-authorizes. Verdicts from Section 14 map to:

- **PASS** → a follow-up authorization may propose: (a) an operational
  calibration protocol for new users (prefix-calibration variant — to be
  specified with its own leakage audit, explicitly excluding any
  chronological TEST use); or (b) a robustness extension (e.g.,
  calibration under cohort drift), each as its own pre-registered
  specification.
- **CAUTION** → a targeted diagnostic follow-up may be proposed for the
  specific failing component (calibration transfer vs operating point),
  again as its own specification.
- **FAIL** → no Phase 18 calibration work on this mechanism; the negative
  result is recorded; alternative directions (feature work per R15-3,
  ranking interventions per R15-2) would need separate, fully specified
  phases.

Every Phase 18 path requires a new authorization with its own
specification; Phase 17 results (including "PASS") imply no standing
authorization for any further execution.

## 26. Reproducibility

- Inputs: frozen artifacts with recorded md5s (Section 6), read-only.
- Protocol: this specification is the pre-registration; constants in
  Appendix A; the approved document is the execution contract.
- Determinism: no RNG outside seeded bootstrap; double-run byte-identical
  gate (Section 16).
- Environment: recorded at execution in `phase17_experiment.json`;
  PC stack pinned in `requirements.txt` (Python 3.11.9; lightgbm 4.7.0;
  numpy 2.4.6; pandas 3.0.5; scikit-learn 1.9.0; pyarrow 25.0.1; pytest
  9.1.1) — `docs/environment.md` is the environment record.
- Artifacts: JSON summaries + manifest; every reported number traces to
  an artifact or a cited frozen record (Section 3).
- Re-execution: any authorized rerun must re-verify gates 1–4 and the
  determinism gate before its numbers are accepted.

## 27. Approval gate

This specification is approved by the project owner **for execution** only
when:

1. The project owner explicitly authorizes Phase 17 execution (this
   document's Status header updated to APPROVED, with date).
2. No change to this document after approval except by a documented,
   dated amendment appended at the end (never in-place edits of approved
   text).
3. The execution plan matches Section 9 (arms), Section 10 (math),
   Section 14 (criteria), Section 17 (artifacts) exactly.
4. The verification gate (Section 9.5) and the report requirements
   (Sections 12.6, 20, 22) are satisfied before any result is recorded
   in the master report or knowledge layer.

*This specification was created from verified project records on
2026-08-18. All OBSERVED anchors were re-read from artifacts during the
specification phase; no runtime result exists yet.*

---

## Appendix A — pre-registered constants

- Frozen threshold: `0.9186015432508062`.
- Bootstrap: n = 1000, seed 42, alpha = 0.10 (90% CIs), user-block and
  row-level (Phase 14 conventions).
- Wilson z: `1.6448536269514722` (90%).
- Platt: `LogisticRegression(C=1.0, fit_intercept=True, solver="lbfgs",
  max_iter=1000, tol=1e-4)` on the single score feature.
- Binning: K = 20 equal-frequency bins; merge rule n < 30; PAV monotone
  smoothing; abort if < 5 bins remain.
- ECDF: midpoint-rank ties; denominator 54,108 (CAL rows).
- Operating point: `best_f1_threshold` (`src/evaluation/threshold.py`),
  CAL only, once.
- ECE/Brier bins: K = 10 equal-frequency, per evaluated cohort.
- Success bounds (Section 14): TEST Brier improvement ≥ 20% relative;
  TEST ECE ≤ 0.05; F1 ≥ 0.40; precision ≥ 0.70; alerts ≤ 300;
  coverage ≥ 8/15; |ΔAUC-ROC| ≤ 1e-9 (strictly monotone arms ≤ 1e-12).
- Recorded Phase 14 TEST values (gate 1): AUC-ROC 0.786513147174144;
  AUC-PR 0.3757395445913874; 1,026 alerts; precision
  0.20175438596491227; recall 0.43487394957983194; F1
  0.2756324900133156; MCC 0.2872810379712531; secondary threshold
  0.9837865316173778.
- CAL reproduction (gate 2): 580 rows ≥ frozen threshold, 135 TP;
  CAL AUC-ROC 0.7355124228200639 (tolerance 1e-9).

## Appendix B — traceability of anchors (all OBSERVED, re-verified 2026-08-18)

| Anchor | Source artifact |
|---|---|
| TEST AUC-ROC / AUC-PR; primary operating point; secondary point | `phase14_experiment.json` (`test_metrics`, `calibration`) |
| Top-k P@k / R@k | `phase14_experiment.json` (`test_metrics.top_k`) |
| Coverage 13/15, Wilson [0.6885, 0.9328]; zero-alert users JJM0203/WDD0366; per-user terciles | `phase14_user_diagnostics.json` (`coverage`, `zero_detection_users`, `per_user`) |
| User-block bootstrap CIs (AUC-ROC [0.666449, 0.901499]; AUC-PR [0.235849, 0.504778]) | `phase14_bootstrap.json` (`user_block`) |
| CAL AUC-ROC 0.7355124228200639 | `phase14_calibration.json`; reproduced in `phase15_experiment.json` gate 2 (diff 0.0) |
| 819/1,026 FPs within 0.10; FP margin median 0.0447; CAL 580 ≥ t / 135 TP | `phase15_threshold_diagnostic.json` (`test_block`, `cal_block`) |
| JJM0203 / WDD0366 max scores 0.4981 / 0.7264 | `phase15_detected_vs_missed.json` |
| Cohort table (chronological CAL/TEST, user-disjoint CAL/TEST) | `phase15_cohort_comparison.json` |
| Input md5s | `phase15_experiment.json` gate 3; `phase15_manifest.json`; `phase14_manifest.json` |
| Block counts (784/108/108 users; 392,784/54,108/54,108 rows; 1,037/379/476 positives) | `phase14_split.json` (`blocks`) |
| Chronological TEST record (0.9392, 49 alerts, etc.) | cited from P7/P9/P12 freeze records; never recomputed |
| Source semantics (threshold, metrics, params, split) | `src/evaluation/threshold.py`, `src/evaluation/metrics.py`, `src/models/lightgbm_baseline.py`, `src/preprocessing/splits.py`, `src/config.py` |