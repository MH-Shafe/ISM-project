# Phase 18 Specification — Leakage-Safe Adaptive Risk v2 (SPECIFICATION ONLY)

Status: **SPECIFICATION ONLY — NOT APPROVED — DO NOT EXECUTE.**
Date: 2026-08-19.

This document pre-registers the design of **Phase 18 — Leakage-Safe
Adaptive Risk v2**: a research-only, evaluation-only redesign of the Phase 8
adaptive-risk layer over the frozen insider-threat detection system on
CERT r4.2. It authorizes **nothing** to run. Execution occurs only after
explicit project-owner approval (Section 28). This specification phase
created no artifacts, modified no frozen file, and consulted only read-only
project records.

Every numeric anchor below was re-verified against its source artifact on
2026-08-19 (not taken from memory or chat summaries). Anchors are labeled
OBSERVED (read from a verified artifact) or cited (frozen record quoted);
reasoned statements are labeled INFERENCE / HYPOTHESIS per ISM principles
§12.

---

## 1. Scientific question

One question, verbatim from the Phase 18 authorization:

> Does a redesigned adaptive risk formulation combining ML risk,
> personalized behavioral-deviation risk, graph-derived trust risk, and
> organizational/context risk improve trustworthy insider-risk assessment
> beyond the frozen LightGBM score without introducing leakage, unacceptable
> calibration degradation, or excessive alert volume?

Decomposition into measurable, falsifiable components (each measured once
per the pre-registered protocol of this specification):

| Component of the question | Phase 18 measurement |
|---|---|
| "Improve … beyond the frozen LightGBM score" | Ranking and operating-point metrics of the fused score vs Arm A (ML-only reference) on the user-disjoint TEST, per the pre-registered arms and PASS/CAUTION/FAIL criteria (Sections 10, 16) |
| "ML risk + behavioral-deviation + graph trust + organizational/context risk" | Four components with exact definitions (Section 6), redundancy and sanity analysis (Section 7) |
| "Trustworthy … without introducing leakage" | Hard gates 1–7, strictly-past construction, block-local peer pools, TRAIN→OOF→CAL→TEST row-role separation (Sections 9, 17) |
| "Without … unacceptable calibration degradation" | Brier/ECE reported per arm on raw fused scores (calibration separation, Section 11); no calibration transform in the main protocol |
| "Without … excessive alert volume" | Alert count/rate bounds in the success criteria (Section 16) |

The question is answered as **PASS / CAUTION / FAIL** per Section 16. This
is a research question about an **alternative risk formulation**, not about
the frozen model, the features, or the production threshold — none of which
may change (Section 3).

## 2. Evidence review (what this specification read, with labels)

All files below were read in this specification phase (2026-08-19); every
number cited elsewhere in this document was taken from these records or
their artifacts, never from memory.

- **Master report** `reports/ISM_MASTER_REPORT.md` (cumulative handoff;
  last updated 2026-08-19; latest completed phase = Phase 17).
- **Phase 7** `reports/phase7_freeze_report.md` + `reports/artifacts/
  phase7_freeze_lgbm-graph-v1.json` (OBSERVED): frozen candidate
  `lgbm-graph-v1`, 12 features, seed 42, best_iteration 186, LightGBM
  4.6.0, threshold `0.9186015432508062` (CAL max-F1), chronological TEST
  (47,000 rows, 30 positives) once: AUC-ROC `0.9391565538286849`, AUC-PR
  `0.2677761042396373`, 49 alerts, F1 `0.35443037974683544`. Department
  feature REJECTED (gain 0 in 5/5 seeds; 49/454,000 non-zero rows).
- **Phase 8** `reports/phase8_adaptive_risk_report.md` + `src/experiments/
  phase8.py` (OBSERVED): the failed adaptive-risk attempt. Formulation
  `Final Risk = α·ML + β·Trust + γ·Behavior` (no context component).
  Learned weights collapsed to ML-only `[1,0,0]` on CALIBRATION; the
  equal-weight control degraded CAL AUC-PR `0.550 → 0.218` with doubled
  alert volume; `trust_risk` was redundant with the ML score (Spearman
  +0.63 on CALIBRATION) and negatively correlated with the target (−0.05);
  `behavior_risk` had negligible standalone signal (CAL AUC-PR 0.008).
  **Decision: adaptive risk NOT adopted.** Phase 18 is a material redesign
  (new behavioral-deviation definition, explicit context component, and a
  pre-registered fusion-learning protocol with collapse detection), not a
  rerun.
- **Phase 9** `reports/phase9_alert_prioritization_report.md` (OBSERVED):
  frozen alert policy = `frozen_max_f1` threshold `0.9186015432508062`
  selected by the S1/S2/S3 rule (threshold identifiability, window alert
  rate stability, window coverage), then highest CAL F1 (0.589). TEST once:
  49 alerts, F1 0.354.
- **Phase 10** `reports/phase10_conformal_report.md` (OBSERVED): accepted
  diagnostic overlay (Mondrian split conformal, CAL-only calibration,
  monitor band (0.4635, 0.9186)). Phase 18 must not disturb it.
- **Phase 12** `reports/phase12_operational_envelope_report.md` (cited via
  decision log): policy retained unchanged; no de-duplication, no rolling
  threshold.
- **Phase 13** `reports/phase13_temporal_stability_report.md` (cited via
  decision log / risks README): CAL verdict FAIL (risk-register only);
  system unchanged.
- **Phase 14** `docs/phase14_specification.md` + `reports/
  phase14_user_holdout_report.md` + artifacts (OBSERVED): user-disjoint
  allocation (malicious: sorted by (first_positive_day, user_id),
  round-robin period 14, positions 0–7 TRAIN / 8–10 CAL / 11–13 TEST;
  benign: sorted user_id, round-robin period 10, 0–7 / 8 / 9). Blocks:
  TRAIN 784 users / 392,784 rows / 1,037 positives / 40 malicious; CAL
  108 / 54,108 / 379 / 15; TEST 108 / 54,108 / 476 / 15. Auxiliary model
  `lgbm-graph-v1-uhold` (12 frozen features, seed 42, scale_pos_weight
  377.7695 from user-disjoint TRAIN, early stopping on user-disjoint CAL
  AUC patience 100 ≤ 3000, best_iteration 214, 750,807 B). TEST once:
  AUC-ROC `0.786513147174144`, AUC-PR `0.3757395445913874`, at frozen
  threshold: 1,026 alerts, precision `0.20175438596491227`, recall
  `0.43487394957983194`, F1 `0.2756324900133156`, MCC `0.2872810379712531`,
  balanced acc `0.7098016078448086`, TP 207 / FP 819 / TN 52,813 / FN 269;
  P@10/30/50 = 1.0, P@100 = 0.95; coverage 13/15 (Wilson 90%
  [0.6885, 0.9328]); zero-alert users JJM0203, WDD0366; user-block
  bootstrap AUC-ROC CI [0.666449, 0.901499], AUC-PR CI [0.235849,
  0.504778] (n = 1000, seed 42, 90%).
- **Phase 15** `docs/phase15_specification.md` + `reports/
  phase15_unseen_user_diagnosis_report.md` + artifacts (OBSERVED): R15-1
  absolute-score calibration failure on unseen benign rows (79.8% of 1,026
  FPs within 0.10 above the frozen threshold; median margin 0.0447); R15-2
  ranking failures JJM0203 (max score 0.4981) / WDD0366 (0.7264), zero
  alerts in 501 days; R15-3 graph features intrinsically weak (row AUC
  0.50–0.69 both blocks; rare counts ~99.9% zero); feature shift small
  (|d| ≤ 0.179, PSI ≤ 0.026). CAL scores locally reproduced
  (`phase15_cal_scores.parquet`).
- **Phase 16** `reports/phase16_final_packaging_report.md` (OBSERVED):
  packaging + verification gate, no ML.
- **Phase 17** `docs/phase17_specification.md` + `reports/
  phase17_calibration_transfer_report.md` + artifacts (OBSERVED):
  calibration-transfer experiment on the same user-disjoint framework.
  Verdict FAIL (mechanical): rank-based AUC is not tie-invariant (arm C
  ΔAUC-ROC −0.0725; arm D −3.27e-6; 1,731 of 4,174 distinct TEST scores
  tied). Arm B (Platt, CAL-fitted) transferred calibration (TEST Brier
  0.007962, ECE 0.002588) and its operating point equals the Phase 14
  secondary record (191 alerts, F1 0.4438, precision 0.7749, raw threshold
  `0.9837865316173778`). **Spec §25: FAIL ⇒ no Phase 18 calibration work on
  this mechanism without separate authorization** — Phase 18 therefore does
  **not** reuse Platt/binning/ECDF recalibration (Section 11).
- **Knowledge layer** `knowledge/decisions/decision-log.md`,
  `knowledge/experiments/experiment-index.md`, `knowledge/risks/README.md`,
  `knowledge/evaluation/README.md` (summaries; never authoritative).
- **Dataset record** `reports/artifacts/dataset_report.json` (OBSERVED):
  LDAP snapshot has per-user `role` (42 distinct), `department` (22),
  `business_unit`, `functional_unit`, `team`, `supervisor`; 16,743 LDAP
  rows; monthly snapshots 2009-12..2011-05 (`src/config.py`, `src/graph/
  features.py`). **No per-user role/department table exists locally** — one
  new derived table is required at execution (Section 5).

Why the Phase 8 failure matters (INFERENCE, from the verified records):
the 2016-era component definitions (empirical-CDF normalized raw features,
simplex-grid weights selected on CALIBRATION) offered the fusion nothing
that the ML score did not already provide, so learning correctly collapsed
to ML. Phase 18 changes the components (strictly-past personalized
deviation, TRAIN-fitted trust normalization, block-local peer context) and
the learning protocol (TRAIN-OOF weights, CAL selection, TEST once) and
pre-registers collapse detection as a first-class outcome.

## 3. Frozen baseline and immutability (verbatim from the authorization)

- Dataset: CERT r4.2 only (raw data never leaves Kaggle; never copied to
  the PC).
- Analytical unit: user × day (501,000 rows, 1,000 users, 501 full
  timeline days per user; 70 malicious users, 1,892 malicious rows).
- Production model: `lgbm-graph-v1` — LightGBM, 12 registry features
  (8 behavioral + 4 graph), seed 42, best_iteration 186, lightgbm 4.6.0,
  scale_pos_weight from chronological TRAIN. NEVER modified, retrained,
  re-tuned, or re-scored in Phase 18.
- Production threshold / alert policy: `0.9186015432508062`
  (`frozen_max_f1`, Phase 9). Immutable.
- Auxiliary model (Phase 14, frozen): `lgbm-graph-v1-uhold` — 12 frozen
  features, seed 42, scale_pos_weight 377.7695, best_iteration 214, model
  file `phase14_model.txt` (750,807 B, md5 `3778a4d869f7231e76ec4c08c6dfa419`).
  Never modified; used only through its frozen score artifacts (Section 5).
- Chronological split (authoritative): TRAIN ≤ 2011-01-31 (395,000 rows /
  1,539 positives), CAL 2011-02-01..2011-03-31 (59,000 / 323), TEST ≥
  2011-04-01 (47,000 / 30, evaluated exactly once, record frozen). The
  chronological TEST is **off-limits forever**: never read, scored, or
  recomputed.
- Accepted overlays: Phase 10 conformal layer, Phase 11 explainability
  contract, Phase 12 policy envelope, Phase 13 risk register. Unchanged.
- Historical artifacts, phase reports, previous phase results: read-only.
  No Phase 18 output may overwrite or modify any of them.
- The user-disjoint allocation (`reports/artifacts/phase14_split.json`,
  md5 `da013f825246d568bcfdaf19dcf2c7e7`): reused verbatim; no
  re-allocation of any kind.

Immutability verification: before and after any authorized execution, the
Phase 17 gate-4 baseline set (151 non-JSON artifacts, plus the recorded
md5s of the score artifacts in Section 5) must be unchanged.

## 4. Evaluation cohort and protocol (pre-registered, unambiguous)

**Adopted cohort: the user-disjoint framework of Phases 14/17.** Not the
chronological split. Reasons (documented, not negotiable at execution):

1. The authoritative chronological TEST is permanently off-limits
   (TEST-once; frozen record only; cited, never recomputed).
2. The user-disjoint framework is the project's established auxiliary
   generalization evaluation with pre-registered, frozen blocks and
   frozen score artifacts (Phase 14/17 conventions).
3. TEST has 476 positives / 15 malicious users vs 30 / 5 chronological —
   the only cohort with enough support for the operating-point and
   coverage comparisons this question requires.
4. All 12 frozen features are day-local or strictly-past per-user
   (Phase 14 feasibility audit, OBSERVED), so components built from them
   remain valid under user-disjoint evaluation.

Blocks and row roles (OBSERVED, `phase14_split.json`; asserted verbatim at
execution):

| Block | Users | Rows | Positives | Malicious users | Role in Phase 18 |
|---|---|---|---|---|---|
| TRAIN | 784 | 392,784 | 1,037 | 40 | OOF cross-fit of ML scores + component computation + **fusion weight learning** |
| CALIBRATION | 108 | 54,108 | 379 | 15 | Threshold selection per arm + calibration-quality description ONLY |
| TEST | 108 | 54,108 | 476 | 15 | **Evaluated exactly once** from frozen scores + pre-registered component values |

**Conflation guard (explicit):** the user-disjoint TEST is NOT the
authoritative chronological TEST. Every Phase 18 result is
**cohort-conditional** (enriched: 15/108 = 13.9% malicious-user share,
0.88% prevalence vs 0.38% population) and is never merged into or compared
against the chronological record as if comparable (Phase 14 §4/§20
convention). The chronological TEST record is cited in Section 2 only.

TEST-once: the user-disjoint TEST is evaluated exactly once per the
finalized protocol (Section 18). The deterministic double run is a
determinism check within the same authorized run, not a second evaluation
(Phase 14/17 convention).

## 5. Data and input audit (all OBSERVED, re-verified 2026-08-19)

All inputs are frozen, read-only, md5-verified. The analysis runs **fully
local** (Phases 15/17 precedent); one derived table build requires Kaggle.

| Input | Rows / shape | Role | Verified md5 |
|---|---|---|---|
| `reports/artifacts/phase6_merged_features.parquet` | 501,000 × 16 (user, day, is_malicious, 12 frozen features) | all component computation, all blocks | `9a3b188573bb953416981dfea3379def` |
| `reports/artifacts/phase14_split.json` | allocation + block counts | user-disjoint blocks | `da013f825246d568bcfdaf19dcf2c7e7` |
| `reports/artifacts/phase15_cal_scores.parquet` | 54,108 × 4 (user, day, is_malicious, score) | **R_ML on CAL** (frozen auxiliary-model scores) | `c955a4ec6ecaf7e9baff80abeb361330` |
| `reports/artifacts/phase14_test_predictions.parquet` | 54,108 × 6 (user, day, is_malicious, score, alert_primary, alert_secondary) | **R_ML on TEST** (frozen auxiliary-model scores) — the only TEST input; never re-scored | `71eeb3f1948e518518a53e062d5a213d` |
| `reports/artifacts/phase14_model.txt` + `phase14_model_record.json` | model + record | reference for the OOF cross-fit configuration (frozen config only; the model file itself is NOT scored) | `3778a4d869f7231e76ec4c08c6dfa419` |
| `reports/artifacts/phase14_user_diagnostics.json` | per-user fields (15 TEST users) | diagnostic sub-analyses only (activity tercile, onset, length class); never in any fit | `cd893661a3a0c08f9066e14c0bf8dd07` |
| `reports/artifacts/phase14_experiment.json`, `phase14_test_metrics.json`, `phase14_calibration.json`, `phase14_bootstrap.json` | recorded Phase 14 results | Arm-A reproduction gates (Section 17, gate 1) | `phase14_manifest.json` |
| `reports/artifacts/phase17_calibration_metrics.json`, `phase17_operating_points.json` | recorded Phase 17 results | Brier/ECE anchors; tie-audit precedent | `phase17_manifest.json` |
| `reports/artifacts/phase7_freeze_lgbm-graph-v1.json`, `phase9_freeze.json` | frozen production records | cited only (chronological record never recomputed) | recorded in their manifests |
| **NEW (built at execution, Kaggle)**: `reports/artifacts/phase18_role_department.parquet` | ~1,000 × ≥ 3 (user, role, department) | role/department for the context component (Section 6.4) | recorded at build; md5-gated |

**The one new derived table** (pre-registered build plan, executed only
after approval): a Kaggle-side script reads the LDAP monthly snapshots
(raw data on Kaggle; never copied to the PC), extracts per user the
`role` and `department` from the **latest snapshot strictly before
month(d)** per the frozen graph-pipeline convention (`src/graph/
features.py` `department_by_day`), and writes `phase18_role_department.parquet`
(user, role, department). Structural gates at build: exactly 1,000 grid
users present (or documented exceptions with counts); no nulls (or
documented); role/department change counts across snapshots recorded
(OBSERVED: roles/departments are static in r4.2 — verify at execution;
if any user's value changes, the strictly-past per-day rule applies).

Feasibility verdict: **FEASIBLE, low-cost, fully deterministic** — same
class of analysis Phase 15 ran in 9.5 s and Phase 17 in ~218 s per
pipeline run, plus one ~1-minute Kaggle build + pull.

## 6. Component definitions (exact; pre-registered; no execution-time invention)

`R_final(u,d) = α·R_ML(u,d) + β·R_trust(u,d) + γ·R_context(u,d) + δ·R_behavior(u,d)`,

with `α, β, γ, δ ≥ 0`, `α + β + γ + δ = 1` (Section 8). Every component is
in `[0,1]`. All component computations are deterministic (no RNG) and
vectorized (numpy/pandas; CPU only, Section 20).

### 6.1 R_ML — ML risk (frozen/authorized score; no transform)

- **TRAIN rows**: out-of-fold predictions of the pre-registered cross-fit
  models (Section 9) — the auxiliary model's in-sample TRAIN scores are
  NOT used (overfit-biased; Phase 15 §4 convention).
- **CAL rows**: `score` column of `phase15_cal_scores.parquet`.
- **TEST rows**: `score` column of `phase14_test_predictions.parquet`.
- **Raw vs calibrated**: RAW. No calibration transform is applied anywhere
  in the main protocol (Phase 17 machinery explicitly NOT reused;
  Section 11). Scores are already in `[0, 1]` (OBSERVED: 0.0–0.9995,
  Phase 14).
- No other transformation; no clipping of R_ML.

### 6.2 R_behavior — personalized behavioral-deviation risk

Candidate formula (adopted, with all parameters pre-registered below):

`z_{u,d,f} = (x_{u,d,f} − median(H_{u,d,f})) / (MAD(H_{u,d,f}) + ε)`,

`R_behavior(u,d) = (1/|F_B|) · Σ_{f ∈ F_B} Φ(clip(|z_{u,d,f}|, 0, 3))`,

where:

- `F_B` = the 8 behavioral features: `login_count`,
  `after_hours_login_count`, `usb_connection_count`, `file_access_count`,
  `sensitive_file_access_count`, `http_activity_count`,
  `unique_device_count`, `unusual_access_count`.
- `H_{u,d,f} = { x_{u,d',f} : d' < d }` — the user's own **strictly-past**
  values (expanding window over the user's full 501-day timeline; nothing
  from day d or later).
- `median` and `MAD` (median absolute deviation, un-scaled:
  `median(|x − median(H)|)`) computed over `H` only.
- `ε = 1e-6` (numerical floor).
- **Min-history rule**: if `|H| < 14` days → `z = 0` (neutral; cold
  start). Pre-registered window length: **expanding** (all strictly-past
  days), not rolling; a rolling-window variant is explicitly rejected for
  the main protocol (adds a free parameter without a design question it
  answers) and is not a diagnostic either.
- **Zero-MAD rule**: if `MAD(H) < 1e-9` → `z = 0` (no detectable
  variation; neutral).
- **Clipping**: `|z| ≤ 3` before `Φ`.
- `Φ` = standard normal CDF (scipy; environment-pinned, Section 26).
  `Φ` is used purely as a fixed monotone map to `[0,1]`; no distributional
  claim is made.
- **Directionality**: magnitude-only in the component (unusual behavior in
  either direction is the signal); signed deviations are reported in
  diagnostics (positive vs negative deviation asymmetry) but never enter
  the component.
- **Missing/unseen behavior**: the frozen tables are null-free (Phase 14
  gates, OBSERVED); zero-filled values are treated as ordinary values.
  For a user with no prior rows at all (not possible here — full grid),
  the min-history rule applies.

### 6.3 R_trust — graph-derived trust risk (accepted features only)

`R_trust(u,d) = (1/|F_G|) · Σ_{f ∈ F_G} p̂_f(x_{u,d,f})`,

where:

- `F_G` = the 4 accepted graph features: `device_consistency_score`,
  `rare_device_usage_count`, `file_type_consistency_score`,
  `rare_file_type_access_count`.
- `p̂_f` = the **TRAIN-block ECDF percentile** of feature `f`:
  `p̂_f(v) = (rank_mid(v) − 1/2) / n_TRAIN` over the 392,784 TRAIN-block
  rows (midpoint-rank convention, as in the Phase 17 ECDF arm; here fitted
  **on TRAIN rows only** and applied as a fixed monotone map).
- **`department_file_type_mismatch_count` is NOT restored** — REJECTED in
  Phase 7 on recorded evidence (gain 0 in 5/5 seeds). No justification
  exists in Phase 18 records; reintroducing it is forbidden without a
  separate pre-TEST authorization.
- **Degenerate-feature rule**: any feature with < 2 distinct values on
  TRAIN → mapped to the constant 0.5 and flagged in the component
  diagnostics (OBSERVED expectation from Phase 15: rare counts are ~99.9%
  zero — the rule exists so the component cannot silently become
  constant).
- Fitted on TRAIN-block rows only; CAL/TEST rows never enter `p̂_f`.
  Available at prediction time (a fixed map).

### 6.4 R_context — organizational/context risk

`zc_{u,d,f} = (x_{u,d,f} − m_{f,s,P(u),d}) / (MAD_{f,s,P(u),d} + ε)`,

`R_context(u,d) = (1/|F|) · Σ_{f ∈ F} Φ(clip(|zc_{u,d,f}|, 0, 3))`,

where:

- `F` = all 12 frozen features.
- `P(u)` = the **peer group** of user u: all other users with the same
  `role` (LDAP, latest snapshot strictly before month(d), Section 5).
  **Min peer count = 5**: if `|P(u)| < 5`, fall back to same-`department`
  peers; if still `< 5`, `R_context(u,d) = 0.5` (neutral) for all rows of
  u, recorded in the component table.
- **Block-local peer pools (pre-registered leakage control)**: for rows of
  a block B, peer statistics are computed from peers' rows **within block
  B only**. TRAIN-side computations therefore never touch CAL/TEST users'
  rows, and TEST users' rows never influence any TRAIN or CAL quantity.
  (Deployment note: in production the peer pool would be all users with
  strictly-past rows; equivalence is a documented HYPOTHESIS, and the
  peer-pool size distribution per block is recorded as a diagnostic.)
- `m` and `MAD` are computed over peers' rows with `d' < d`
  (**strictly-past peer statistics; a peer's day-d row never enters**) and
  `s` = the day-of-week stratum of d: `weekday` (Mon–Fri) vs `weekend`
  (Sat–Sun) — the **workday context** stratification.
- **Min-support rule**: if the stratum-s peer window has < 20 rows,
  collapse to the unstratified strictly-past peer window; if that has
  < 20 rows, `zc = 0` (neutral).
- **Time-of-day**: enters through `after_hours_login_count` (already a
  day-level aggregation of after-hours activity) compared against peers —
  no raw-log re-processing of any kind (Section 20).
- **Role-relative device/file behavior**: enters through the peer
  comparisons of `device_consistency_score`, `file_type_consistency_score`,
  `rare_device_usage_count`, `rare_file_type_access_count`,
  `usb_connection_count`, `file_access_count`.
- **Prediction-time availability audit** (all yes): role/department from
  the latest snapshot strictly before month(d) (LDAP rule, frozen);
  peers' strictly-past rows; the current day's feature values (day-local);
  day-of-week of d (calendar). Nothing else is used. No future
  information, no same-day peer values, no label-derived quantities.
- Directionality: magnitude-only (as in 6.2); signed diagnostics only.
- Missing/unseen entities: users absent from LDAP → neutral fallback
  (recorded count at execution).

### 6.5 Component production order (fixed)

Components are computed for ALL rows of all three blocks from the frozen
inputs (Sections 5–6) in one pass, before any weight fit, before any CAL
threshold, before any TEST metric. Row order fixed by (user, day) sort.

## 7. Component redundancy and sanity analysis (pre-registered; TRAIN + CAL only)

Before any weight is learned, the following diagnostics are computed and
written (TRAIN-block rows; CAL rows where stated). They inform the report,
not the protocol (no component is added/removed/reweighted by their
outcome — the fusion decides, Section 8):

1. **Pairwise correlation**: Spearman rho of (R_ML, R_trust, R_context,
   R_behavior) on TRAIN rows, and each component vs `is_malicious`
   (point-biserial). Expectation note (HYPOTHESIS, from Phase 8): R_trust
   will be positively correlated with R_ML (graph features are in the ML
   model; Phase 8 measured +0.63 for its trust component).
2. **Standalone discrimination**: row-level AUC-ROC and AUC-PR of each
   component as a score vs `is_malicious`, per block (TRAIN, CAL) —
   mirrors Phase 15 Analysis D.
3. **Standalone calibration**: Brier and ECE (K = 10 equal-frequency
   bins, per-cohort binning; Section 11 conventions) of each component
   treated as p̂.
4. **Class-conditional distributions**: per-component decile tables of
   the positive vs negative row distributions (TRAIN and CAL).
5. **Degenerate-component detection**: any component with > 99.9% of rows
   within `[0.5 − 0.01, 0.5 + 0.01]`, or with < 3 distinct values, or
   constant → flagged; the redundancy report states it explicitly
   (OBSERVED precedent: Phase 8 `behavior_risk` CAL AUC-PR 0.008).
6. **No removal rule**: a degenerate or redundant component is NOT removed
   from the fusion (the constrained learning handles it); it is reported.

## 8. Fusion formulation (exact)

- Score: `R_final = α·R_ML + β·R_trust + γ·R_context + δ·R_behavior`.
- Constraints: `α, β, γ, δ ≥ 0`; `α + β + γ + δ = 1`. **`[1,0,0,0]` is a
  valid result** — it means the auxiliary components earned no weight
  (Phase 8 outcome); it is reported as ML-only collapse per Section 16,
  not as a pipeline error.
- **"Approximately ML-only"**: `α ≥ 0.95` with the refit-bootstrap 90% CI
  of `(1 − α)` containing 0 → interpreted as collapse (Sections 16, 24).
- **Objective (pre-registered, primary)**: mean log-loss of the fused
  score on the TRAIN-OOF rows (Section 9):
  `L = −(1/n) Σ [y·log(p) + (1−y)·log(1−p)]`, with `p` clipped to
  `[1e-6, 1 − 1e-6]` (numerical floor, pre-registered). Brier is reported
  as a secondary objective but never used for selection.
- **Solver (pre-registered, deterministic, no RNG)**: `scipy.optimize.
  minimize(method="SLSQP", x0 = equal weights over the arm's free
  dimensions, bounds = (0,1) per weight, constraint = {type:"eq",
  fun: sum(weights) − 1}, tol = 1e-12, maxiter = 1000)`. The problem is
  convex (log-loss in a linear combination with linear constraints), so
  SLSQP finds the global optimum; the starting point is fixed and the
  solve is deterministic. Weights recorded to 12 significant digits.
- Weights are learned **only** on TRAIN-OOF rows. CAL sees weights only as
  read-only inputs to threshold selection. TEST never influences weights.

## 9. Weight-learning protocol (TRAIN → OOF → learn → CAL selection → TEST once)

Row roles (fixed):

| Stage | Data | Operation |
|---|---|---|
| 1. Components | all rows, all blocks | Section 6, one pass |
| 2. OOF ML scores (TRAIN only) | TRAIN block (784 users) | K = 5 entity-disjoint folds; cross-fit (below) |
| 3. Weight learning | TRAIN-OOF rows (392,784) | Section 8 solver, per arm |
| 4. CAL selection | CAL rows (54,108) | per-arm `best_f1_threshold` (Section 13); calibration description (Section 11) |
| 5. TEST evaluation | TEST rows (54,108) | once, per Section 18 |

**Cross-fit design (TRAIN OOF ML scores) — exact and deterministic:**

1. TRAIN-block users sorted by `user_id` (string sort, as in the Phase 14
   allocation); fold k = users at sorted index `i` with `i mod 5 == k`
   (round-robin over period 5). Fold sizes ≈ 156–157 users each; the fold
   table is recorded in `phase18_experiment.json`.
2. For each fold k: fit LightGBM on the other 4 folds' rows with the
   frozen auxiliary configuration (Section 3: lr 0.03, num_leaves 31,
   min_data_in_leaf 100, feature_fraction 0.8, bagging_fraction 0.8,
   bagging_freq 1, objective binary, metric auc, seed 42, verbose −1),
   `scale_pos_weight` = neg/pos over those rows only (frozen TRAIN-only
   convention), **early stopping on fold (k+1) mod 5 AUC** (patience 100,
   ≤ 3000 rounds; deterministic pre-registered rule; `best_iteration` is
   an outcome, recorded, never tuned).
3. Predict the held-out fold's rows → OOF scores.
4. These cross-fit models are evaluation-only constructs for the OOF
   scores; they are NEVER production candidates (same status as
   `lgbm-graph-v1-uhold`).
5. No CAL or TEST row enters any cross-fit fit (entity-disjoint folds
   within TRAIN; CAL/TEST users never in a fold).

**Temporal/entity-aware fold rationale** (documented): entity-disjoint
folds (users never span folds) preserve the user-disjoint property and the
strictly-past per-user feature construction; calendar time is block-
invariant by construction (Phase 15 §3, OBSERVED), so entity folds do not
introduce temporal confounding. No random split of any kind.

## 10. Arms (complete list; nothing may be added at execution)

| Arm | Weights | Purpose |
|---|---|---|
| A | (1, 0, 0, 0) | ML-only reference; **must reproduce the recorded Phase 14/17 TEST numbers exactly** (gate 1, Section 17) |
| B | (0.25, 0.25, 0.25, 0.25) | Equal-weight sanity control (Phase 8 precedent; no expectation claimed — measured) |
| C | learned (α, β, γ, δ) | Full fusion, constrained learning |
| C1 | learned (α, δ, 0, 0) | ML + behavioral deviation |
| C2 | learned (α, β, 0, 0) | ML + graph trust |
| C3 | learned (α, 0, γ, 0) | ML + context |

Every arm uses the identical protocol (Sections 8–9, 13): same OOF scores,
same component values, same solver, same CAL threshold rule, same TEST
evaluation. No post-hoc arms; no arm may be dropped or added after
approval except by documented amendment (Section 28).

## 11. Calibration separation (explicit)

- **No calibration transform exists in the main Phase 18 protocol.** The
  fused score `R_final` is reported raw; the operating point is the CAL
  best-F1 threshold on the raw score (Section 13).
- The Phase 17 calibration machinery (Platt, binning+PAV, ECDF) is **NOT
  reused** — Phase 17 §25 forbids further calibration work on that
  mechanism without separate authorization, and Phase 18's question is
  about the risk formulation, not score-to-probability recalibration.
- Brier and ECE (K = 10 equal-frequency bins, binning on the evaluated
  cohort's own scores — descriptive summarization, Phase 17 §10.4
  convention) are reported per arm per cohort as **descriptive
  calibration quality of the raw fused score**, with the raw score
  treated as p̂ (uncalibrated reference, Phase 17 arm-A convention). These
  are report quantities; they select nothing.
- Fusion (Section 8) and any future calibration are therefore
  analytically separate by construction.

## 12. Ranking and tie handling (Phase 17 lesson R17-1, applied)

- **Pre-registered fact (OBSERVED, Phase 17)**: the empirical average-rank
  AUC estimators change when distinct scores are mapped to equal values
  even when row order is preserved. No invariance claim is made for any
  arm's transformed scores.
- The fused score is a convex combination of continuous components; ties
  are possible (e.g., coincident component vectors) but expected to be
  rare. **Tie audit (pre-registered)**: for every arm, record the number
  of distinct `R_final` values on TEST and the number of collision rows
  (rows sharing a score with another row) vs the maximum over components
  and vs arm A; report OBSERVED. If any arm's collision count exceeds
  0.1% of TEST rows, the arm is flagged CAUTION (Section 16) with the tie
  sets recorded.
- Deterministic tie handling for any downstream operation (threshold
  application, top-k): secondary sort by (user_id, day) — fixed, recorded,
  no RNG.
- P@k / R@k and AUC deltas are reported with the tie audit; no tolerance
  is claimed for tie-induced movement (Phase 17 §12.4 convention).

## 13. Thresholds and operating points (CAL-only, single rule)

- **Primary comparison**: per-arm operating point at `t_arm` =
  `best_f1_threshold(y_CAL, R_final_CAL)` from `src/evaluation/threshold.py`
  (exact reference semantics: ascending unique-score grid, `pred = score
  ≥ t`, strict-greater F1 update with 1e-12 tolerance, ties broken toward
  the higher threshold — the frozen convention that produced
  `0.9186015432508062`). Applied once per arm; never TEST-informed; never
  iterated.
- **Ranking metrics** (AUC-ROC/AUC-PR/P@k/R@k) are threshold-free and are
  the primary cross-arm comparison (Phase 17 §12.1 convention).
- The frozen production threshold is NOT applied to `R_final` (different
  score space; meaningless by construction — documented, not a defect).
- Fair-comparison statement (pre-registered): every arm receives the same
  single CAL-only selection rule; no arm is tuned, and no threshold is
  selected from TEST.

## 14. Metrics (pre-registered; all on the user-disjoint TEST, once)

Using existing `src/evaluation/metrics.py` functions (frozen semantics):

- **Ranking**: AUC-ROC, AUC-PR, P@10 / P@30 / P@50, R@50 / R@100.
- **Operating (at t_arm)**: alerts, alert rate (per user-day and per
  calendar day), TP / FP / TN / FN, precision, recall, F1, MCC, balanced
  accuracy, FPR, FNR; alerts per user (mean / median / max); Gini of
  alert concentration over alerted users (Phase 9 convention).
- **Calibration (descriptive, Section 11)**: Brier, ECE (K = 10), and
  reliability data per arm per cohort (CAL, TEST).
- **Coverage**: fraction of the 15 TEST malicious users with ≥ 1 alert at
  t_arm; Wilson 90% CI (z = `1.6448536269514722`); zero-alert users
  listed; coverage sub-analysis by activity tercile / length class /
  onset half reusing `phase14_user_diagnostics.json` fields (no
  recomputation; descriptive only).
- **Cost**: wall time per stage, peak memory, artifact sizes
  (`phase18_cost.json`).

## 15. Statistical plan (pre-registered; n = 1000, seed 42, 90% CIs)

- **Confidence level: 90%** (project convention: alpha = 0.10; Z90 =
  `1.6448536269514722`; all Phase 9–17 intervals).
- **User-block bootstrap (primary, per arm)**: resample TEST users with
  replacement (n = 1000, seed 42), pool rows, recompute AUC-ROC, AUC-PR,
  alerts, precision, recall, F1 (Phase 14 estimator; `n_skipped`
  recorded). Weights FIXED (learned once on TRAIN).
- **Weight-refit bootstrap (new, pre-registered; needed because weights
  are learned, not frozen)**: resample TRAIN users with replacement
  (n = 1000, seed 42), refit the arm's fusion weights on the resampled
  OOF rows (Section 8 solver), apply the refitted weights to the FIXED
  CAL and TEST component/score values, record: weight distributions,
  CAL threshold distribution, TEST Brier/ECE and operating-metric
  distributions (90% percentile intervals). This is the honest
  uncertainty for TRAIN-learned weights (Phase 17 calibration-transfer
  bootstrap precedent). Expected cost: Section 21.
- **Row-level bootstrap**: reported for convention comparability only
  (flagged optimistic given within-user correlation, Phase 14 §12).
- **Δ metrics**: ΔAUC-ROC and ΔAUC-PR of each arm vs arm A, with 90% CIs
  from the user-block bootstrap (paired on resamples where possible —
  same resample index; recorded).
- **Determinism**: bootstrap is seeded (42) and row-order invariant
  (Phase 14 convention); the double-run gate (Section 17, gate 7) covers
  the whole pipeline including bootstraps.
- No TEST resampling for any fitting purpose; no threshold movement from
  bootstrap (evaluation-only uncertainty, Phase 14 §12).

## 16. Success criteria — PASS / CAUTION / FAIL (numerical, pre-registered)

All gates (Section 17) must pass for any verdict other than FAIL.
Anchors (OBSERVED records): Arm A on the user-disjoint TEST — AUC-ROC
`0.786513147174144`, AUC-PR `0.3757395445913874`, 1,026 alerts, precision
`0.20175438596491227`, recall `0.43487394957983194`, F1 `0.2756324900133156`,
MCC `0.2872810379712531`, coverage 13/15, TEST Brier 0.102734, TEST ECE
0.217109 (Phase 14/17 artifacts). Phase 14 secondary operating point
(descriptive precedent): 191 alerts, F1 0.4438, precision 0.7749.

**PASS** — all of the following:

1. **Gates 1–7 all pass** (Section 17).
2. **Meaningful improvement**: at least one auxiliary arm (C, C1, C2, or
   C3) meets ALL of:
   - ΔAUC-PR vs arm A ≥ **+0.02** absolute AND the user-block bootstrap
     90% CI of ΔAUC-PR **excludes 0**;
   - ΔAUC-ROC vs arm A ≥ 0.00 with CI lower bound ≥ **−0.005** (no
     ranking degradation);
   - F1 at t_arm ≥ **0.40** (arm A: 0.2756; precedent: Phase 14
     secondary 0.4438);
   - precision ≥ **0.70** (arm A: 0.2018; precedent: 0.7749);
   - alert count ≤ **300** (arm A: 1,026; precedent: 191);
   - MCC ≥ **0.40** (arm A: 0.2873; precedent: 0.4883);
   - malicious-user coverage ≥ **8/15** (arm A: 13/15; Phase 17 bound);
   - TEST Brier ≤ 0.102734 (not worse than raw ML) AND TEST ECE ≤ **0.10**
     (arm A: 0.2171);
   - tie audit: collisions ≤ 0.1% of TEST rows (Section 12).
3. **Auxiliary weight mass**: the learned arm's `1 − α ≥ 0.05` AND the
   weight-refit bootstrap 90% CI of `(1 − α)` **excludes 0** (the
   improvement is attributable to the auxiliaries, not chance).
4. **Weight/component stability**: every learned weight's refit-bootstrap
   90% CI width ≤ **0.20**; redundancy report written (Section 7).
5. **Determinism, immutability, TEST-once, no chronological contact**:
   gates 4/6/7 plus Section 18 compliance.

**CAUTION** — any of: gates pass but (a) no arm meets the improvement
bundle yet no arm degrades beyond the material bounds (a valid "no
measurable improvement" result — the question is answered negatively
without system risk); (b) the bundle is met only by arms whose weight CI
includes 0 (improvement not attributable to auxiliaries); (c) one arm
meets the bundle while another degrades beyond the material bounds; (d)
any gate passed only via a documented fallback; (e) tie audit flags
> 0.1% collisions with a documented cause; (f) weight CI width > 0.20.

**FAIL** — any of: any gate fails without a pre-registered fallback; any
auxiliary arm **materially degrades** vs arm A: ΔAUC-ROC < **−0.01** or
ΔAUC-PR < **−0.02** or alerts > 2 × arm A alerts or F1 < arm A F1 − 0.05;
**ML-only collapse**: learned `α ≥ 0.95` AND no arm meets the improvement
bundle (a valid negative result — reported honestly, Phase 8 echo);
any TEST-informed selection; determinism or immutability violation; any
contact with the chronological TEST.

**Pre-registered expectation notes** (not criteria):
- HYPOTHESIS: R_trust will be redundant with R_ML (Phase 8 measured
  +0.63 Spearman for a similar component) — C2 may not beat A; measured,
  not assumed.
- HYPOTHESIS: JJM0203 / WDD0366 (R15-2 ranking failures, max scores
  0.4981 / 0.7264, zero alerts in 501 days) will remain zero-alert in
  every arm — no fused component can repair ranking failure without
  evidence of deviation/context signal; any arm that alerts them is an
  OBSERVED surprise reported with the responsible component analysis
  (Section 19).
- A PASS is consistent with "the redesigned formulation recovers or
  exceeds the ML-only reference on unseen users with its own verified
  pipeline"; the phase's new contribution is the measured component
  contribution + the leakage-safe fusion protocol.

## 17. Hard gates 1–7 (before any Phase 18 statistic is accepted)

1. **Gate 1 — frozen-reference reproduction (Arm A)**: recompute from the
   frozen TEST score artifact + frozen labels: AUC-ROC
   `0.786513147174144`, AUC-PR `0.3757395445913874`, alert count 1,026,
   precision `0.20175438596491227`, recall `0.43487394957983194`, F1
   `0.2756324900133156`, MCC `0.2872810379712531`, balanced acc
   `0.7098016078448086`, P@10/30/50 = 1.0, P@100 = 0.95. Must match the
   recorded Phase 14 values bit-exactly or within 1e-12 relative
   (Phase 17 gate-1 convention).
2. **Gate 2 — split/entity integrity**: allocation == `phase14_split.json`
   verbatim; block counts 784/108/108 users, 392,784/54,108/54,108 rows,
   1,037/379/476 positives, 40/15/15 malicious; pairwise user
   disjointness; fold table (Section 9) disjoint and covering all 784
   TRAIN users.
3. **Gate 3 — component leakage tests**: on hand-computed synthetic
   fixtures (Section 22): behavioral deviation uses strictly-past `H`
   only; peer statistics use peers' rows strictly before d within the
   block; min-history / min-support / zero-MAD / cold-start rules behave
   as specified; no future information in any component.
4. **Gate 4 — frozen artifact integrity**: all Section 5 input md5s
   verified at run start; immutability baselines (Section 3) unchanged
   before and after the run; `phase18_role_department.parquet` build
   gates (Section 5) pass.
5. **Gate 5 — component sanity**: all components finite, in [0,1],
   null-free; degenerate-component flags computed (Section 7.5);
   R_ML column provenance asserted (TRAIN = OOF, CAL/TEST = frozen
   artifacts).
6. **Gate 6 — weight-learning isolation**: weights depend only on
   TRAIN-OOF rows (structural test: no CAL/TEST row in any fit input);
   CAL used only for thresholds; TEST read only at the final stage.
7. **Gate 7 — determinism**: full pipeline run twice within the same
   authorized run; all artifacts byte-identical (canonical JSON;
   `determinism: True` with runtimes, Phase 14/17 convention).

**STOP semantics**: if any gate fails beyond its recorded tolerance and no
pre-registered fallback applies, the run stops, the failure is recorded
and reported, and no metric is accepted (Phase 17 §20 convention).
Pre-registered fallbacks: gate 3 failure on a fixture → fix the tested
code path and re-run the fixture (code defect, not evidence); gate 1
failure → STOP (frozen-artifact corruption or pipeline error — never
"repair" by adjusting inputs).

## 18. TEST-once discipline

- The user-disjoint TEST is evaluated exactly once per the finalized
  protocol; the deterministic double run re-executes the identical
  pipeline within the same authorized run (a determinism check, not a
  second evaluation — Phase 14 §1/§3 convention).
- The **final-evaluation artifact** (`phase18_test_metrics.json` and the
  TEST blocks of `phase18_bootstrap.json`) is first written only after
  the following are all frozen (list is exhaustive):
  1. component definitions and all parameters (Section 6);
  2. fold design and cross-fit config (Section 9);
  3. fusion formulation, solver, objective, clipping (Section 8);
  4. arm list (Section 10);
  5. threshold rule (Section 13);
  6. metric definitions (Section 14);
  7. statistical plan (Section 15);
  8. success criteria (Section 16);
  9. gates (Section 17);
  10. this specification approved unchanged (Section 28);
  11. tests pass locally (Section 22);
  12. the role table build gates pass (Section 5).
- The **authoritative chronological TEST** (47,000 rows, 30 positives,
  2011-04-01..2011-05-17) is never read, scored, or recomputed; its
  frozen record is cited only (Phase 15/17 convention).
- No TEST-derived quantity feeds any fit, threshold, selection, or
  artifact other than the pre-registered metrics themselves.

## 19. Diagnostic users and post-verdict diagnostics

- JJM0203 and WDD0366 (TEST malicious users, zero-alert in Phases 14/15/
  17; max scores 0.4981 / 0.7264) are **diagnostic-only subjects**: no
  fit, threshold, or selection ever conditions on them; no metric is
  recomputed "excluding" or "including" them to claim an improvement.
- Post-verdict diagnostics (after the verdict is locked, Section 24):
  per-arm zero-alert user lists; per-arm max score and component profile
  of JJM0203 / WDD0366 (which component is high/low — the responsible
  component analysis); scenario recall per arm (Phase 14 scenario
  mapping reused as-is; NOT VERIFIED locally; empty strata reported NOT
  VERIFIED, never imputed); component-vs-score correlations.
- Any arm that alerts either diagnostic user is reported as an OBSERVED
  surprise with the mechanism (per Phase 17 §12.5 convention).

## 20. Non-goals and method constraints

Phase 18 does NOT: retrain, modify, or re-tune any production or
auxiliary model (the only new fits are the pre-registered TRAIN OOF
cross-fit models and fusion weights, both evaluation-only); change any
feature definition or the feature registry; change the frozen threshold or
alert policy; add a calibration transform; use CNN / GNN / transformer /
deep autoencoder / GPU LightGBM / CUDA (vectorized tabular methods only);
use random or temporal splits (the allocation is frozen; folds are
entity-disjoint round-robin); re-score or re-evaluate any TEST beyond the
single authorized evaluation; touch raw logs (in particular the
14.5-GB HTTP log is never read; only the LDAP snapshots are read once for
the role table build, on Kaggle); touch the answer key (scenario mapping
reused from records as-is); copy raw data to the PC; recommend or
implement any production change; update the master report, decision log,
experiment index, risks README, or any knowledge record during this
specification phase (updates occur only after an approved and verified
execution, per ISM principles §15).

## 21. Resource plan and expected cost (CPU-only)

- **Inputs**: 501,000 merged rows + two 54,108-row score tables + small
  JSONs (all local); one Kaggle LDAP extraction (~1,000-row table, ~1
  min build + pull; raw data stays on Kaggle).
- **Component computation** (Section 6): strictly-past expanding
  statistics over 501,000 × 12 features; vectorized numpy/pandas
  (searchsorted / groupby-expanding); expected 1–5 min single-threaded
  (estimate class: Phase 5 graph build 13 s; the behavioral component is
  the heaviest — per-user expanding median/MAD). Peak memory < 3 GB
  (intermediate per-feature arrays), well inside the local 31-GiB class
  and the 4-CPU class.
- **Cross-fits** (Section 9): 5 LightGBM fits on ~314k rows each — Phase
  7 measured 4.64 s train on 395k rows → expect ≈ 25–40 s total.
- **Fusion**: 6 arms × SLSQP on 392,784 rows → seconds.
- **Bootstrap**: user-block 1000 × per-arm metrics (Phase 14: both
  bootstraps fit in 33 s runs); weight-refit 1000 × SLSQP on 392,784
  rows — the dominant cost; Phase 17's equivalent refit bootstrap cost
  ~50 s per arm → estimate 1–3 min per arm, 6–15 min total.
- **Total**: single pipeline run ≈ 10–20 min; double run ≈ 20–40 min
  (Phase 17: 218 s/run with a comparable bootstrap load). No GPU, no
  Kaggle quota for the analysis; measured cost recorded in
  `phase18_cost.json`.
- **HTTP log**: not touched (Section 20). All inputs are frozen/verified
  derived tables (Section 5).

## 22. Required tests (local; before any result is recorded)

`tests/test_phase18.py` (created only at execution, after approval):

- **Hand-computed synthetic cases** (each with a written expected value):
  behavioral deviation — strictly-past median/MAD, min-history (13 vs 14
  days), zero-MAD, ε floor, clipping, Φ mapping, cold start;
  context — peer strictly-past cutoff (a peer's day-d row must not
  count), weekday/weekend stratification, min-support 20-row fallback,
  min-peer-count 5 fallback to department, neutral 0.5 fallback,
  block-local peer pools (TRAIN computation touches no CAL/TEST row);
  trust — TRAIN-only ECDF (a CAL/TEST value must not change the map),
  midpoint-rank ties, degenerate feature → 0.5 flag;
  R_ML provenance — TRAIN rows carry OOF scores, CAL/TEST carry frozen
  artifact scores (structural assertions).
- **No-future-leakage tests**: for every component, perturbation of day
  d+1 values leaves day d outputs unchanged (future-invariance, Phase 5
  convention).
- **Fusion**: weights non-negative, sum = 1 (tolerance 1e-9), solver
  determinism (double solve bit-identical), `[1,0,0,0]` reachable
  (synthetic fixture where ML dominates → learned α = 1 within 1e-6),
  log-loss clipping bounds, ablation-arm constraints.
- **Gate machinery**: arm-A reproduction on a small fixture (recorded
  number reproduced from fixture data, not real artifacts); tie audit;
  fold-table disjointness/coverage on the real split.
- **Threshold**: `best_f1_threshold` semantics (higher-threshold tie-break,
  strict-greater F1 update) on fixtures.
- **Bootstrap**: user-block and weight-refit estimators on a small
  synthetic panel (resampling semantics, seed 42 reproducibility,
  n_skipped); Wilson CI hand-computed.
- **Determinism**: double-run byte-identical JSON artifacts.
- **Artifact reloadability**: every Section 23 artifact loads and matches
  its record; manifest md5s verify.
- **Frozen-input immutability**: Section 5 inputs md5-verified before and
  after the run; `test_frozen_inputs_still_immutable` convention
  (Phase 15).
- **Diagnostic-user non-use**: structural test that no fit input contains
  JJM0203 or WDD0366 rows (they are TEST users; TEST isolation).
- Full local suite must pass; exact counts reported (Phase 17 baseline:
  407 passed / 4 skipped at its completion; Phase 18 adds its own
  tests).

## 23. Required artifacts (created ONLY at execution, after approval; no historical artifact overwritten)

`reports/artifacts/`:

- `phase18_experiment.json` — experiment id, scope, inputs + md5s
  (gate 4), gates 1–7 results, fold table, environment, determinism flag,
  evidence labels, frozen-system statement.
- `phase18_components.parquet` — per-row component values + ML score
  provenance (TRAIN OOF / CAL / TEST frozen) for all 501,000 rows
  (~20 columns; md5-gated; the component values are features, not
  evaluation results — no per-row final fused TEST scores are written).
- `phase18_redundancy.json` — Section 7 diagnostics.
- `phase18_fusion.json` — per arm: weights (12 significant digits), fit
  records (objective value, iterations, convergence), weight-refit
  bootstrap distributions.
- `phase18_calibration.json` — per arm: t_arm, CAL Brier/ECE/reliability,
  CAL threshold stability note.
- `phase18_test_metrics.json` — Section 14 metrics per arm on TEST
  (written once, Section 18).
- `phase18_ranking_verification.json` — per arm: AUC-ROC, AUC-PR,
  P@10/30/50, R@50/100, deltas vs arm A with bootstrap CIs, tie audit.
- `phase18_bootstrap.json` — user-block, weight-refit, row-level
  distributions and CIs.
- `phase18_coverage.json` — coverage + Wilson CI per arm, per-user
  alerts, zero-alert users, Gini, sub-analyses.
- `phase18_diagnostics.json` — Section 19 post-verdict diagnostics.
- `phase18_cost.json` — wall time per stage, peak memory, artifact sizes.
- `phase18_manifest.json` — md5s of all Phase 18 JSON/parquet artifacts
  (self-excluded, Phase 14/15 convention).
- `phase18_role_department.parquet` — Section 5 (built first on Kaggle).
- `phase18.log` — [JOB]-formatted log (kernel-side convention).

Also (execution-time, after approval): `src/experiments/phase18.py`,
`kaggle_scripts/build_phase18_role_table.py` (Kaggle-side LDAP
extraction), `kaggle_scripts/run_phase18.py` (local runner),
`tests/test_phase18.py`, `reports/phase18_adaptive_risk_v2_report.md`,
and (after verified execution) the master report + knowledge layer updates
per ISM principles §15 — **none of these exist yet; this specification
creates none of them**.

## 24. Research interpretation and production boundary

- Every statement in the Phase 18 report is labeled OBSERVED (computed
  from verified artifacts), INFERENCE (reasoned from OBSERVED), HYPOTHESIS
  (expected, not demonstrated), LITERATURE RESULT (external), or NOT
  VERIFIED (absent from records) — ISM principles §12.
- **A negative result is valid and must be reported honestly**: learned
  collapse to ML-only (α ≥ 0.95), no measurable improvement, or a
  component shown redundant — each is a legitimate answer to the research
  question (Phase 8 precedent; Section 16).
- **Production boundary**: Phase 18 changes nothing in production under
  any verdict. A PASS means the fused formulation is a **candidate** for a
  separate, explicitly authorized integration/freeze phase (new spec,
  new leakage audit, production-feasibility work — none authorized here).
  CAUTION/FAIL map to targeted diagnostics or a recorded negative result
  (Phase 17 §25 convention). Phase 18 never self-authorizes anything.
- The fused score is **not** proposed as a replacement alert score; it is
  a research formulation measured on the user-disjoint cohort.

## 25. Risks (pre-registered R18)

- **R18-1 ML-only collapse echo (Phase 8)**: auxiliaries may again earn
  ~zero weight. Outcome: measured; collapse is a pre-registered verdict
  path (Section 16), reported honestly.
- **R18-2 R_trust redundancy**: graph features are already in the ML
  model (Phase 8 measured +0.63 Spearman for a similar component);
  expected redundancy is measured (Section 7), never silently removed.
- **R18-3 Context component data dependency**: requires the one new LDAP
  derived table; if the LDAP build fails (schema change, missing users),
  gates 4/5 apply and the phase stops with a recorded, documented cause —
  no substitute data may be used.
- **R18-4 Peer-group sparsity**: role groups within a 108-user block can
  be small; mitigated by the min-peer/min-support fallbacks (Section 6.4)
  and the block-local peer-pool size diagnostic.
- **R18-5 Tie creation in the fused score**: possible but expected rare
  (Section 12); bounded and audited, no invariance claim.
- **R18-6 Weight instability under 5-fold OOF**: quantified by the
  weight-refit bootstrap (Section 15); criterion 4 bounds CI width.
- **R18-7 Cohort-conditional results**: user-disjoint TEST is enriched
  (13.9% malicious-user share); operating-point metrics are NOT
  production-comparable (Phase 14 §4/§20); ranking + coverage are the
  generalization evidence; stated in the report.
- **R18-8 Scope creep**: new features, retraining, calibration,
  threshold hunting, deep methods, or random splits invalidate the phase;
  non-goals (Section 20) bind execution.
- **R18-9 Overclaiming**: PASS ≠ production improvement (Section 24);
  R17-5 precedent applies to the fused score as well (deployment impact
  NOT VERIFIED).
- **R18-10 Library sensitivity**: environment pinned and recorded
  (Section 26); determinism gate covers refits.

## 26. Determinism and environment

- No RNG anywhere except the two seeded bootstrap estimators (seed 42,
  Phase 14 sampling convention).
- Fixed input order: all rows sorted by (user, day) before any
  order-sensitive computation; fold assignment and peer windows are
  seed-free.
- PC stack (pinned, OBSERVED from `requirements.txt` / Phase 17 record):
  Python 3.11.9, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, lightgbm
  4.7.0 (OOF cross-fits), scipy 1.17.1 (SLSQP, Φ), pyarrow 25.0.1, pytest
  9.1.1. Kaggle stack for the LDAP build only: Python 3.12.13, lightgbm
  not needed (no scoring on Kaggle).
- Determinism gate: full pipeline twice, byte-identical JSON
  (`determinism: True`, runtimes recorded).
- Environment recorded in `phase18_experiment.json` (OBSERVED at
  execution).

## 27. Specification validation (applied to this document, 2026-08-19)

The following checklist was applied to this specification after writing
(no experiments were run as validation):

1. All numbered sections present (1–28 + appendices) — checked.
2. Every referenced file exists locally — checked (Section 2 list;
   `reports/artifacts/*.json` md5s quoted from manifests).
3. All numerical anchors read from authoritative records/artifacts, not
   memory — checked (Appendix B).
4. Evaluation cohort unambiguous (user-disjoint; chronological TEST
   excluded and never conflated) — Section 4.
5. No leakage: strictly-past construction, block-local peers, TRAIN→OOF→
   CAL→TEST row roles, TEST-once — Sections 6, 9, 18.
6. Roles separated: fusion (TRAIN OOF) vs calibration (none) vs
   thresholds (CAL only) vs evaluation (TEST once) — Sections 8–13.
7. Exact PASS/CAUTION/FAIL criteria with numbers — Section 16.
8. Hard gates 1–7 with tolerances and STOP semantics — Section 17.
9. Artifacts pre-registered; no historical artifact overwritten —
   Section 23.
10. Immutability explicit — Section 3.
11. CPU-only documented; resource plan with expected runtime/memory;
    HTTP log untouched — Sections 20–21.
12. TEST-once explicit with a freeze list — Section 18.
13. Diagnostic-user non-use — Section 19.
14. Deliverable = this file only (created); nothing executed — verified
    at write time.

## 28. Approval gate

This specification is approved by the project owner **for execution** only
when:

1. The project owner explicitly authorizes Phase 18 execution (this
   document's Status header updated to APPROVED, with date).
2. No change to this document after approval except by a documented,
   dated amendment appended at the end (never in-place edits of approved
   text).
3. Execution matches Sections 6 (components), 8–9 (fusion and learning),
   10 (arms), 13 (thresholds), 16 (criteria), 17 (gates), 23 (artifacts)
   exactly.
4. The verification gate (Section 17) and report requirements (Sections
   24, 25) are satisfied before any result is recorded in the master
   report or knowledge layer.

---

## Appendix A — pre-registered constants

- Frozen threshold (unchanged): `0.9186015432508062`.
- Bootstrap: n = 1000, seed 42, alpha = 0.10 (90% CIs); user-block
  primary, weight-refit (TRAIN user-block with fusion refit), row-level
  convention-comparability (Phase 14 conventions).
- Wilson z: `1.6448536269514722` (90%).
- Behavioral component: expanding strictly-past window; min history 14
  days; MAD un-scaled; ε = 1e-6; zero-MAD threshold 1e-9; clip |z| ≤ 3;
  Φ = standard normal CDF; magnitude-only.
- Trust component: TRAIN-block ECDF percentile, midpoint-rank ties,
  denominator 392,784; degenerate-feature rule < 2 distinct values → 0.5.
- Context component: role peers (min 5) → department peers (min 5) →
  neutral 0.5; strictly-past peer windows with weekday/weekend strata;
  min-support 20 rows per stratum (else unstratified, else neutral);
  block-local peer pools; LDAP rule = latest snapshot strictly before
  month(d).
- Fusion: SLSQP, x0 = equal weights, bounds (0,1), sum = 1, tol 1e-12,
  maxiter 1000, objective = mean log-loss with p clipped to
  [1e-6, 1 − 1e-6].
- OOF cross-fit: K = 5 entity-disjoint round-robin folds (user_id sort,
  i mod 5); frozen auxiliary LightGBM config; scale_pos_weight from the 4
  training folds' labels; early stopping on fold (k+1) mod 5 AUC,
  patience 100, ≤ 3000.
- Threshold rule: `best_f1_threshold` (CAL only, once per arm,
  higher-threshold tie-break, strict-greater F1 update, 1e-12 tolerance).
- ECE/Brier bins: K = 10 equal-frequency, per evaluated cohort
  (descriptive only; no transform).
- Success bounds (Section 16): ΔAUC-PR ≥ +0.02 (CI excludes 0);
  ΔAUC-ROC ≥ 0.00 (CI LB ≥ −0.005); F1 ≥ 0.40; precision ≥ 0.70; MCC ≥
  0.40; alerts ≤ 300; coverage ≥ 8/15; Brier ≤ 0.102734; ECE ≤ 0.10;
  1 − α ≥ 0.05 (CI excludes 0); weight CI width ≤ 0.20; tie collisions
  ≤ 0.1%.
- Material-degradation bounds (FAIL): ΔAUC-ROC < −0.01; ΔAUC-PR < −0.02;
  alerts > 2× arm A; F1 < arm A − 0.05.
- ML-only collapse: learned α ≥ 0.95 with no arm meeting the improvement
  bundle.

## Appendix B — traceability of anchors (all OBSERVED, re-verified 2026-08-19)

| Anchor | Source artifact |
|---|---|
| User-disjoint blocks (784/108/108 users; 392,784/54,108/54,108 rows; 1,037/379/476 pos; 40/15/15 mal) | `phase14_split.json`; `docs/phase14_specification.md` §6 |
| Auxiliary model config (12 features, seed 42, spw 377.7695, iter 214, 750,807 B) | `phase14_model_record.json`; `phase17_specification.md` §4 |
| TEST scores + metrics (AUC-ROC 0.786513…, AUC-PR 0.375739…, 1,026 alerts, precision/recall/F1/MCC, P@k/R@k, coverage 13/15, Wilson [0.6885, 0.9328], zero-alert JJM0203/WDD0366) | `phase14_test_predictions.parquet` (md5 `71eeb3f1…213d`), `phase14_experiment.json`, `phase14_test_metrics.json`, `phase14_user_diagnostics.json` (md5 `cd893661…dd07`) |
| CAL scores + AUC (0.7355124228200639; 580 ≥ t; 135 TP) | `phase15_cal_scores.parquet` (md5 `c955a4ec…1330`), `phase14_calibration.json`, `phase15_threshold_diagnostic.json` |
| TEST Brier/ECE raw (0.102734 / 0.217109); arm-B record (191 alerts, F1 0.4438, P 0.7749, threshold `0.9837865316173778`) | `phase17_calibration_metrics.json`, `phase17_operating_points.json`, `phase14_experiment.json` |
| User-block bootstrap CIs (AUC-ROC [0.666449, 0.901499]; AUC-PR [0.235849, 0.504778]) | `phase14_bootstrap.json` |
| Phase 8 failure record (collapse [1,0,0]; equal-weight AUC-PR 0.550→0.218; trust Spearman +0.63 / −0.05; behavior AUC-PR 0.008) | `reports/phase8_adaptive_risk_report.md`, `knowledge/evaluation/README.md` |
| R15-1 (79.8% FPs within 0.10; median margin 0.0447), R15-2 (max scores 0.4981 / 0.7264), R15-3 (row AUC 0.50–0.69; rare 99.9% zero) | `phase15_threshold_diagnostic.json`, `phase15_detected_vs_missed.json`, `phase15_graph_diagnosis.json` |
| Phase 17 tie lesson (C −0.0725; D −3.27e-6; 1,731/4,174 ties) | `phase17_ranking_verification.json`, `reports/phase17_calibration_transfer_report.md` §5 |
| Merged table (501,000 × 16) | `phase6_merged_features.parquet`, md5 `9a3b188573bb953416981dfea3379def` |
| LDAP fields (role 42, department 22) | `dataset_report.json` (`ldap` block) |
| Frozen production record (chronological TEST: AUC-ROC 0.9391565538286849, AUC-PR 0.2677761042396373, 49 alerts, F1 0.35443037974683544; threshold `0.9186015432508062`) | `phase7_freeze_lgbm-graph-v1.json`, `phase9_freeze.json` (cited, never recomputed) |
| Threshold/metric semantics | `src/evaluation/threshold.py`, `src/evaluation/metrics.py` |
| LightGBM frozen config | `src/models/lightgbm_baseline.py`, `phase14_model_record.json` |

*End of specification. Feasibility counts are OBSERVED from verified
artifacts; protocol details are pre-registered here and must not be
invented or changed at execution time. This specification created only
this file. Nothing was executed. No frozen file was modified.*