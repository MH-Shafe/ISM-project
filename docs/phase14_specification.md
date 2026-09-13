# Phase 14 Specification — User-Level Holdout Generalization (FEASIBILITY + PROTOCOL)

Status: **SPECIFICATION ONLY — NOT APPROVED — DO NOT EXECUTE.**

This document designs an **auxiliary, evaluation-only** user-disjoint
generalization evaluation of the frozen insider-threat detection system on
CERT r4.2. Nothing in this specification changes the frozen system, the
authoritative chronological TEST, or any historical artifact.

Evidence labels: **OBSERVED** (measured/verified), **INFERENCE** (reasoned),
**HYPOTHESIS** (untested), **LITERATURE RESULT** (external), **NOT VERIFIED**
(absent from project records). All feasibility counts below are OBSERVED
from the frozen derived table `reports/artifacts/phase6_merged_features.parquet`
(501,000 × 16; md5 `9A3B188573BB953416981DFEA3379DEF`, 1,077,577 B) — the
merged behavioral + graph + label table produced by the frozen pipeline —
unless marked otherwise.

---

## 1. Objective

Design and specify (do not execute) a **user-disjoint** TRAIN → CALIBRATION →
TEST evaluation that answers: *does the frozen insider-threat detection
system generalize to users completely unseen during model development?*
The evaluation is **auxiliary robustness evidence only**: it does not replace
the authoritative chronological evaluation (frozen since Phase 4), it makes
no production decision, and it changes nothing in the frozen system.

## 2. Research question

Primary: do ranking performance (AUC-ROC / AUC-PR / top-k) and the frozen
operating point (precision / recall / F1 / MCC / alert volume at the frozen
threshold) transfer from users seen during model development to users whose
rows never entered any training fit?

Secondary (diagnostic): do detection outcomes differ by CERT scenario, user
activity level, incident length, and temporal position of the incident?

## 3. Why the experiment is needed

The master report records an open limitation: *"Entity overlap across splits
(no user-level holdout)"* (Section 21, Evaluation risks) and lists
*"User-level holdout evaluation — deferred evaluation design question"*
(Section 22). All prior evaluations used a chronological split in which the
same users appear in TRAIN, CAL, and TEST; the frozen model literally trained
on TEST-block users' TRAIN-period rows. It is therefore **unmeasured** whether
the system's performance reflects generalization to genuinely new users or
familiarity with these users' behavioral patterns. Phase 14 measures exactly
this gap, on frozen inputs and a strictly evaluation-only auxiliary model.

## 4. Feasibility findings (OBSERVED unless noted)

Measured from the frozen merged table (501,000 rows, 501 days, 1,000 users;
every user has exactly 501 rows — full grid):

- **Population**: 1,000 users; **70 malicious users** (1,892 malicious rows,
  0.38% prevalence); 930 benign users. Matches the published foundation
  record exactly.
- **Positives per malicious user**: min 1, median 9, max 59. Distribution is
  **bimodal**: 40 users with 1–12 positive days, 30 users with 45–59 positive
  days, **no user between 13 and 44** (OBSERVED; pre-registered length
  classes "short ≤ 12" and "long ≥ 45").
- **Temporal distribution**: incident onset (first positive day) spans
  2010-06-10 .. 2011-04-29; monthly onset histogram: 2010-06 ×6, 2010-07 ×14,
  2010-08 ×6, 2010-09 ×9, 2010-10 ×11, 2010-11 ×6, 2010-12 ×4, 2011-01 ×4,
  2011-02 ×9, 2011-04 ×1. Incident spans 1–59 days (median 9).
- **Chronological overlap (context)**: 60 malicious users have positive days
  in the chronological TRAIN block (≤ 2011-01-31), 13 in CAL (02-01..03-31),
  5 in TEST (≥ 04-01) — consistent with the Phase 9 record (13 CAL malicious
  users). This is why a user-disjoint evaluation is needed and why the frozen
  model is NOT "unseen" for those users.
- **Label handling**: CDE1846 is unlabeled in the frozen table (0 rows;
  unparseable answer-key date, recorded in the foundation report) — the
  labeled population is exactly 70 users / 1,892 rows, and Phase 14 must
  reproduce this gate. CDE1846 therefore falls in the **benign** pool.
- **Scenario counts**: recorded 30 / 30 / 10 (scenarios 1/2/3) for the
  dataset==4.2 incidents (foundation report §2.1). The **per-user scenario
  mapping is NOT VERIFIED locally** (answer key is raw data, Kaggle-only);
  it is required at execution time for scenario diagnostics only.
- **Candidate allocations (deterministic rules, computed)**: three candidate
  allocations of the 70 malicious users (time-sorted round-robin; see §6):

  | Design | TRAIN | CAL | TEST | Sufficient support? |
  |---|---|---|---|---|
  | A 40/15/15 | 40 users, 1,037 pos | 15 users, 379 pos | 15 users, **476 pos** | YES (all splits ≥ 10 users, ≥ 300 pos) |
  | B 50/10/10 | 50 users, 1,193 pos | 10 users, 300 pos | 10 users, 399 pos | CAL/TEST thin for user-level diagnostics |
  | C 45/15/10 | 45 users, 1,222 pos | 15 users, 325 pos | 10 users, 345 pos | TEST thin for user-level diagnostics |

  **Design A is adopted** (justification: maximizes positive-user support in
  CAL and TEST for early stopping, threshold description, user-level
  diagnostics, and confidence intervals; TEST 476 positives is ~16× the
  chronological TEST's 30; TRAIN 1,037 positives is comparable to the frozen
  chronological TRAIN's 1,539).
- **Every split can contain enough positive users**: A gives 40/15/15
  malicious users (≥ 10 per split; all splits well above the pre-registered
  minimum of 5 users / 50 positive rows, §19).
- **Split can remain leakage-safe**: yes — see §7.
- **Feature construction remains valid under user-disjoint evaluation**: yes.
  All 12 frozen features are day-local (behavioral) or per-user strictly-past
  (graph); none depends on split membership, on other users, or on any
  statistic computed from the evaluation rows. Features are computed once on
  the full timeline, then rows are partitioned by user (§7).
- **Graph features — hidden user/entity leakage**: none in the frozen model.
  The 4 retained graph features (`device_consistency_score`,
  `rare_device_usage_count`, `file_type_consistency_score`,
  `rare_file_type_access_count`) use **only the user's own strictly-past
  first-use history** (`src/graph/features.py`; verified in this audit). The
  only cross-user feature (`department_file_type_mismatch_count`) was
  REJECTED in Phase 7 and is not in the model. No graph table rebuild is
  required under the split.
- **Any feature requiring rebuilding under the proposed split**: none.
  Rebuilding is required only after a Kaggle runtime recycle (recovery §18).
- **Feasibility verdict: FEASIBLE** — a statistically meaningful user-disjoint
  evaluation can be constructed with strong positive-user support in every
  split. The evaluation is **cohort-enriched** (TEST malicious-user share
  13.9% vs 7% population; TEST prevalence 0.88% vs 0.38%): operating-point
  metrics are conditional on the evaluation cohort and are NOT directly
  comparable to the chronological TEST record (§20). This is a documented
  property of user-level holdout with rare positives, not a defect.

## 5. Dataset requirements

- CERT r4.2 only (unchanged). Raw logs + answer key remain on Kaggle,
  read-only; raw data is never copied to the PC.
- Required inputs (frozen, all existing): `user_day_features.parquet`
  (501,000 × 11), `graph_features.parquet` (501,000 × 7), merged modeling
  table (501,000 × 16, 12 frozen features + user/day/is_malicious; local
  reference md5 `9A3B188573BB953416981DFEA3379DEF`), answer key
  (`insiders.csv`, dataset==4.2) for scenario diagnostics only.
- No new dataset, no new release, no synthetic users, no fabricated labels.
- Inputs are read-only; md5s recorded; nothing is regenerated except by the
  standard rebuild paths after a runtime recycle (§18).

## 6. User-disjoint split design (pre-registered, deterministic, frozen now)

Rule (no RNG; auditable):

1. Malicious users: sort the 70 labeled users by
   `(first_positive_day, user_id)` ascending (first_positive_day from the
   frozen label column); assign by round-robin over a period of 14:
   positions 0–7 → TRAIN, 8–10 → CAL, 11–13 → TEST. This yields exactly
   **40 / 15 / 15** users.
2. Benign users (930, incl. CDE1846): sort by `user_id`; round-robin over a
   period of 10: positions 0–7 → TRAIN, 8 → CAL, 9 → TEST. Yields
   **744 / 93 / 93** users.

Pre-registered allocation of the 70 malicious users (user, split, positive
rows, onset — all OBSERVED from the frozen label column):

| User | Split | Pos | Onset | User | Split | Pos | Onset |
|---|---|---|---|---|---|---|---|
| CSC0217 | T | 2 | 2010-06-10 | PNL0301 | T | 51 | 2010-06-14 |
| LCC0819 | T | 56 | 2010-06-16 | GTD0219 | T | 2 | 2010-06-17 |
| RMW0542 | T | 59 | 2010-06-21 | AAF0535 | T | 54 | 2010-06-28 |
| IJM0776 | T | 58 | 2010-07-06 | RAR0725 | T | 45 | 2010-07-06 |
| MOS0047 | T | 58 | 2010-07-15 | EHB0824 | T | 8 | 2010-07-22 |
| JTM0223 | T | 2 | 2010-07-22 | DIB0285 | T | 50 | 2010-07-26 |
| BDV0168 | T | 12 | 2010-07-30 | LJR0523 | T | 12 | 2010-07-31 |
| EGD0132 | T | 58 | 2010-08-02 | PSF0133 | T | 59 | 2010-08-02 |
| AJR0932 | T | 9 | 2010-09-10 | RAB0589 | T | 11 | 2010-09-13 |
| LQC0479 | T | 9 | 2010-09-14 | MCF0600 | T | 4 | 2010-09-20 |
| BLS0678 | T | 10 | 2010-09-21 | MAS0025 | T | 2 | 2010-09-29 |
| BSS0369 | T | 2 | 2010-09-30 | EHD0584 | T | 7 | 2010-10-02 |
| RGG0064 | T | 8 | 2010-10-20 | AAM0658 | T | 7 | 2010-10-23 |
| TAP0551 | T | 7 | 2010-10-23 | ABC0174 | T | 59 | 2010-10-27 |
| MPM0220 | T | 2 | 2010-11-04 | GHL0460 | T | 1 | 2010-11-09 |
| DRR0162 | T | 55 | 2010-11-11 | HJB0742 | T | 7 | 2010-11-19 |
| MDH0580 | T | 59 | 2011-01-04 | FMG0527 | T | 8 | 2011-01-05 |
| FSC0601 | T | 59 | 2011-01-18 | JRG0207 | T | 8 | 2011-01-19 |
| DCH0843 | T | 1 | 2011-02-04 | CEJ0109 | T | 54 | 2011-02-07 |
| NWT0098 | T | 58 | 2011-02-07 | MAR0955 | T | 4 | 2011-02-08 |
| KPC0073 | C | 9 | 2010-07-07 | BIH0745 | C | 1 | 2010-07-13 |
| RHL0992 | C | 59 | 2010-07-13 | XHW0498 | C | 59 | 2010-08-09 |
| CAH0936 | C | 2 | 2010-08-11 | BBS0039 | C | 2 | 2010-08-12 |
| AKR0057 | C | 57 | 2010-10-04 | BTL0226 | C | 9 | 2010-10-06 |
| IUB0565 | C | 56 | 2010-10-06 | KRL0501 | C | 59 | 2010-11-22 |
| FTM0406 | C | 8 | 2010-11-25 | MSO0222 | C | 2 | 2010-12-09 |
| PPF0435 | C | 1 | 2011-02-09 | KLH0596 | C | 1 | 2011-02-12 |
| HBO0413 | C | 54 | 2011-02-14 | | | | |
| RKD0604 | X | 8 | 2010-07-13 | JMB0308 | X | 8 | 2010-07-14 |
| JGT0221 | X | 2 | 2010-07-15 | HXL0968 | X | 59 | 2010-08-31 |
| JJM0203 | X | 48 | 2010-09-02 | VSS0154 | X | 53 | 2010-09-07 |
| CCA0046 | X | 2 | 2010-10-14 | TNM0961 | X | 56 | 2010-10-15 |
| EDB0714 | X | 58 | 2010-10-18 | MYD0978 | X | 6 | 2010-12-13 |
| CCL0068 | X | 57 | 2010-12-27 | IKR0401 | X | 53 | 2010-12-27 |
| CQW0652 | X | 56 | 2011-02-18 | WDD0366 | X | 8 | 2011-02-24 |
| JLM0364 | X | 2 | 2011-04-28 | | | | |

T = TRAIN (40 users, 1,037 positive rows; 24 short ≤ 12 / 16 long ≥ 45;
onset 2010-06-10..2011-02-08), C = CAL (15 users, 379 positive rows;
9 short / 6 long; onset 2010-07-07..2011-02-14), X = TEST (15 users,
476 positive rows; 7 short / 8 long; onset 2010-07-13..2011-04-28).
TRAIN ∩ CAL = TRAIN ∩ TEST = CAL ∩ TEST = ∅ at the user level by
construction (verified by test).

Resulting blocks (full 501-day timelines of every assigned user):

| Block | Users | Rows | Malicious rows | Malicious users | Prevalence |
|---|---|---|---|---|---|
| TRAIN | 784 | 392,784 | 1,037 | 40 | 0.26% |
| CALIBRATION | 108 | 54,108 | 379 | 15 | 0.70% |
| TEST | 108 | 54,108 | 476 | 15 | 0.88% |

Every row of a user belongs to exactly one block (users never span blocks);
every block contains the user's full calendar history (no calendar gaps).

## 7. Leakage controls

- **User disjointness**: the split is at the user level; a user's entire
  501-day timeline is in exactly one block. No row of a TEST user ever
  enters any fit.
- **No future information**: all 12 features are day-local or strictly-past
  per-user (verified by the existing future-invariance tests); the split
  does not change any feature value — features are computed on the full
  timeline before partitioning.
- **No cross-user statistics in the model**: the 12 frozen features are
  user-local (audited in §4); the only cross-user feature ever built
  (department) is rejected and excluded.
- **Evaluation rows never train the model**: TRAIN fits on TRAIN rows only;
  early stopping and the secondary threshold use CAL rows only; TEST rows
  are scored once, after training and all selection are complete.
- **Labels never features**: `is_malicious` and the scenario key are used
  for evaluation/diagnostics only (structural test).
- **scale_pos_weight**: computed from user-disjoint TRAIN labels only
  (mirrors the frozen protocol).
- **Answer key**: used only for scenario diagnostics; never in allocation
  (allocation uses the frozen label column) and never in features.
- **TEST-once**: the auxiliary TEST is evaluated once per finalized protocol;
  nothing is tuned against it (see §13).

## 8. Frozen-component rules (immutable)

- Frozen model `lgbm-graph-v1` (12 features, seed 42, 186 trees): **not
  modified, not retrained, not re-tuned**; it remains the production
  candidate. Its frozen weights never touch Phase 14 rows (those users are
  not "unseen" to it).
- Frozen feature set, feature table, and pipeline: unchanged.
- Production threshold `0.9186015432508062` and the Phase 9 policy:
  unchanged; used here as an immutable reference operating point.
- Phase 10 conformal and Phase 11 explainability layers: unchanged, not
  re-fitted, referenced only as background.
- Chronological split and the authoritative chronological TEST record:
  untouched, never re-evaluated for any decision in this phase.
- Historical artifacts and previous phase results: read-only.

## 9. Training / calibration / test protocol (evaluation-only)

1. **Auxiliary model** `lgbm-graph-v1-uhold` (evaluation-only; NEVER a
   production candidate): LightGBM 4.6.0 with the frozen hyperparameter
   config (`src/models/lightgbm_baseline.py` `default_params`: lr 0.03,
   31 leaves, min_data_in_leaf 100, feature_fraction 0.8, bagging 0.8/freq 1,
   seed 42, binary/auc), the 12 frozen features, `scale_pos_weight` from
   user-disjoint TRAIN labels, early stopping on user-disjoint CAL AUC
   (patience 100, ≤ 3000 rounds) — the same training protocol that produced
   the frozen model, applied to a user-disjoint TRAIN. `best_iteration` is an
   outcome, recorded, not tuned.
2. **Operating points** (both pre-registered):
   - **Primary**: the frozen threshold `0.9186015432508062` applied to
     auxiliary TEST scores — tests whether the frozen operating point
     transfers to unseen users.
   - **Secondary (descriptive only)**: `best_f1_threshold` on user-disjoint
     CAL applied to auxiliary TEST — separates model transfer from
     threshold-calibration transfer. It selects nothing, changes nothing.
3. **TEST evaluation (once)**: score all 54,108 TEST rows with
   `lgbm-graph-v1-uhold`; compute the pre-registered metrics (§10) and
   diagnostics (§11); write artifacts (§14).
4. **Reproduction gates** before any metric: allocation table verbatim
   (§6), user disjointness, block counts, label reproduction (1,892 / 70),
   feature columns identical to the frozen 12, frozen-input md5s.

## 10. Metrics (pre-registered; all evaluation-only)

On the auxiliary TEST, using existing `src/evaluation/metrics.py` functions:

- Ranking: **AUC-ROC**, **AUC-PR**, precision@10/30/50, recall@50/100.
- Operating point (primary and secondary): precision, recall, F1, **MCC**,
  balanced accuracy, FPR, FNR, **alert count**, **alert rate** (alerts per
  user-day and per calendar day), alerts per user (mean/median/max), Gini of
  alert concentration over alerted users (Phase 9 convention).
- **Scenario recall** per scenario present in the split (from the answer key
  at execution; a scenario with zero positive rows in a split is reported
  NOT VERIFIED for that split, never imputed).
- **Confidence intervals**: (a) row-level bootstrap (`bootstrap_ci`,
  n = 1000, seed 42) — the project convention, for comparability with the
  frozen record; (b) **user-block bootstrap** (pre-registered new estimator:
  resample TEST *users* with replacement, pool each user's rows, recompute —
  honest for the panel structure; rows within a user are correlated).
  Both reported with `n_skipped`.
- **Malicious-user coverage** at the primary operating point: fraction of
  the 15 TEST malicious users with ≥ 1 alert (Wilson 90% CI).

## 11. User-level diagnostics (all pre-registered, auxiliary TEST only)

- **Zero-detection users**: TEST malicious users with no alert (listed
  individually).
- **Fully detected users**: TEST malicious users with every positive day
  alerted.
- **Detection delay**: calendar days from the user's first positive day to
  their first alert day (users with ≥ 1 alert only).
- **First-week detection**: any alert within the first 7 positive days of an
  incident.
- **User-level AUC** (diagnostic aggregation, Phase 9 §13 convention:
  max-score per user vs any-malicious-day user label) — diagnostic only,
  never an alerting unit.
- **Stratified diagnostics** (each justified by the research question):
  - by **scenario** (question: does detection differ by incident type);
  - by **activity level**: terciles of active days (login_count > 0) among
    TEST users (OBSERVED benign range 25–495, median 346);
  - by **incident length class**: short ≤ 12 vs long ≥ 45 positive days (the
    OBSERVED bimodal classes; the 13–44 gap is empty);
  - by **temporal position**: onset calendar half (2010 H2 vs 2011) and
    incident phase (first / middle / last tercile of the user's positive
    span) — detection near onset is the operational value.

## 12. Statistical uncertainty

- TEST 476 positives / 15 malicious users (vs 30 positives in the
  chronological TEST) → materially tighter CIs expected (INFERENCE from the
  counts; measured at execution).
- User-block bootstrap is the primary uncertainty statement for
  user-generalization claims; row-level bootstrap is reported for
  comparability and flagged as optimistic given within-user correlation
  (OBSERVED structure: ~470 benign rows per user, correlated daily
  behavior).
- Per-stratum cells are small (e.g., 3–5 users per length class); strata
  are reported descriptively with Wilson 90% CIs, never as significance
  tests.
- All uncertainty estimates are evaluation-only (no refitting, no threshold
  movement).

## 13. TEST isolation rules

- The auxiliary TEST is evaluated **exactly once** per finalized protocol.
- No metric, threshold, diagnostic, or artifact of the auxiliary TEST is
  used to modify, retrain, re-tune, or re-select anything — including the
  auxiliary model itself.
- The authoritative chronological TEST (47,000 rows, 30 positives) is
  **not re-evaluated** in this phase; its frozen record is cited, never
  recomputed.
- Auxiliary TEST metrics are reported in a clearly separated, labeled block
  and are never merged into the chronological record.

## 14. Required artifacts (to be created only at execution, after approval)

- `phase14_split.json` — allocation table verbatim + block counts + md5s.
- `phase14_model.txt` — auxiliary model (lgb-graph-v1-uhold) + record.
- `phase14_calibration.json` — user-disjoint CAL metrics, early-stopping
  curve summary, secondary threshold.
- `phase14_test_predictions.parquet` — 54,108 rows × (user, day,
  is_malicious, score, alert_primary, alert_secondary).
- `phase14_test_metrics.json` — §10 metrics at both operating points.
- `phase14_user_diagnostics.json` — §11 diagnostics.
- `phase14_bootstrap.json` — row-level + user-block CIs.
- `phase14_scenario.json` — per-split scenario counts and recall.
- `phase14_cost.json` — runtime, rows, model size.
- `phase14.log` — [JOB]-formatted log.

## 15. Required tests (to be written only at execution, after approval)

- Allocation rule unit tests: round-robin rule; the pre-registered 70-user
  table reproduced verbatim; 40/15/15 and 744/93/93 counts; user
  disjointness (no user in two blocks); benign-rule reproducibility.
- Block gates: rows = 392,784 / 54,108 / 54,108; positives = 1,037 / 379 /
  476; label reproduction 1,892 / 70; CDE1846 unlabeled.
- Feature tests: frozen 12 columns present; no label/scenario column enters
  features (structural); no extra numeric columns.
- Metric wiring tests: existing metric functions on synthetic data
  (hand-computed values); operating-point application at both thresholds.
- New estimator test: user-block bootstrap (hand-computed on a tiny
  synthetic panel; user resampling semantics).
- Determinism test: double-run bit-identical.
- Reloadability tests (skip-guarded): every artifact in §14 loads and
  matches its record.

## 16. Determinism requirements

- Split: deterministic (no RNG; pre-registered rules §6).
- Model: LightGBM seed 42 on identical inputs is deterministic (verified
  across Phases 4/6/7).
- Analysis: fixed seeds (42) for both bootstrap estimators; row-order
  invariant.
- Gate: the full pipeline is run twice at execution; metrics artifacts must
  be bit-identical (`bit_identical=true` recorded), mirroring the Phase 13
  gate.
- Frozen input md5s recorded before any analysis; sources md5-verified vs
  local before pushing.

## 17. Kaggle execution requirements

- Execution requires Kaggle (raw dataset + answer key; derived tables are
  Kaggle-side). Local PC: code, tests, report (no raw data).
- Push sources via the verified gz+b64 mechanism, md5-verified kernel-side.
- Expected compute (OBSERVED analogues): training ≈ 5 s, prediction ≈ 2 s
  (Phase 7: 4.6 s / 1.3 s on similar row counts); table loads ≈ 5 s —
  comfortably inside interactive limits; still, every cell uses explicit
  `timeout` and `[JOB]` heartbeats per `docs/kaggle-execution-policy.md`.
- Pull artifacts PC-side, md5-verified against the kernel manifest; journal
  outcomes via `kaggle_scripts/kaggle_exec.py log`.

## 18. Recovery requirements

Per `docs/kaggle-execution-policy.md` runbook D:

- Runtime recycled → refresh config, restart the session, re-push sources,
  rebuild derived tables (user_day ≈ 150 s, graph ≈ 20 s — OBSERVED prior
  recovery), re-run; never recover raw data to the PC.
- Kernel death → wait ~10 s, probe, retry ≤ 3; child processes do not
  survive; relaunch only after a clean idle probe.
- Aborted interactive call > 120 s → poll for the artifact; never relaunch
  speculatively.
- Journal every noteworthy outcome with classification.

## 19. Failure / infeasibility criteria (pre-registered)

**INFEASIBLE (stop, declare, do not manufacture a split)**:

- Any block with < 5 malicious users or < 50 positive rows (does not apply
  to the adopted 40/15/15 design, which is well above both bars — the rule
  guards against any changed allocation);
- Allocation table mismatch vs §6;
- Label reproduction failure (labeled users ≠ 70 or positive rows ≠ 1,892).

**Run-time gates (fail fast, no repair, record)**:

- Any integrity gate (§9.4) failure stops the run.
- Degenerate early stopping (best_iteration == 1) → recorded, reported,
  protocol continues with the limitation flagged (mirrors Phase 6 arm B
  handling); no silent config change.
- Frozen threshold yields 0 or > 10,000 alerts on auxiliary TEST → recorded,
  interpreted via the secondary point; **no adaptation**.
- Any block's positive rows below the §19 bars at execution (e.g., after an
  unplanned re-allocation attempt) → INFEASIBLE.

## 20. Decision rules

- **No production decision can result from Phase 14.** By design, no
  threshold, model, feature, policy, or overlay changes; no chronological
  record changes; nothing is adopted or rejected for production.
- Reporting conventions (descriptive labels, pre-registered, not system
  decisions), comparing the auxiliary TEST to the frozen chronological
  record:
  - AUC-ROC: within ±0.02 → "comparable"; worse by > 0.05 → "materially
    worse"; between → "inconclusive".
  - AUC-PR: within ±20% relative → "comparable"; worse by > 40% relative →
    "materially worse"; between → "inconclusive".
  - Operating-point metrics (precision/recall/F1/alerts): **not directly
    comparable** (cohort enrichment, §4) — reported side by side with the
    caveat, never merged.
  - Malicious-user coverage ≥ 12/15 → "high user-level coverage";
    8–11/15 → "moderate"; ≤ 7/15 → "low" (Wilson CI reported).
- The final report states explicitly what can and cannot be concluded (§22).

## 21. Risks

- **Cohort enrichment (OBSERVED, design property)**: auxiliary TEST
  malicious-user share 13.9% (vs 7%) and prevalence 0.88% (vs 0.38%);
  precision/F1 at any threshold are inflated relative to production.
  Mitigation: ranking metrics + user-level coverage are the primary
  generalization evidence; operating points reported with the caveat.
- **TEST incidence-length mix skews long (OBSERVED)**: TEST = 7 short / 8
  long users vs population 40/30; median TEST positives 48 vs 9 population.
  Mitigation: length-class diagnostics reported with counts; not a design
  change.
- **Scenario imbalance per split (NOT VERIFIED locally)**: per-split
  scenario counts depend on the answer key; a split could contain 0
  scenario-3 users. Mitigation: scenario counts reported at execution;
  per-scenario metrics marked NOT VERIFIED where empty; no re-allocation
  (the allocation is frozen).
- **Auxiliary model ≠ frozen model**: conclusions attribute to the system
  config + protocol, not the exact frozen weights. Mitigation: explicit
  statement in the report.
- **Early-stopping variance on 15-user CAL**: mitigated by the
  degenerate-iteration gate (§19) and recorded best_iteration.
- **Score-scale shift**: the frozen threshold may behave differently on the
  auxiliary model's scores; the secondary CAL threshold exists to interpret
  this. No adaptation.
- **User-block bootstrap width**: 15 TEST users → wide CIs are expected and
  reported honestly (not hidden by row-level bootstrap).
- **Derived-table bit-consistency across environments**: merged-table md5
  gates may fail on pandas-version differences; structural gates then apply
  and the difference is reported (not silently accepted).
- **Kaggle ephemerality**: recycling wipes `/kaggle/working` (§18).
- **CDE1846**: unlabeled by the frozen record; Phase 14 keeps it unlabeled
  and in the benign pool (no relabeling ever).

## 22. Expected scientific interpretation (pre-registered expectations)

- If auxiliary TEST AUC-ROC/AUC-PR are "comparable" (§20) and malicious-user
  coverage is high → OBSERVED evidence that the system's ranking
  generalizes to completely unseen users (cohort caveats apply;
  generalization of the operating point is reported separately via the
  primary/secondary threshold comparison).
- If ranking drops materially → OBSERVED evidence that performance is partly
  user-specific (the frozen model learned per-user patterns); the
  chronological record still stands as the production evidence.
- If ranking transfers but the frozen threshold does not (0 or degenerate
  alert volume) → OBSERVED operating-point-transfer failure; the secondary
  CAL threshold shows whether the selection protocol can recover an
  operating point on unseen users.
- All conclusions labeled OBSERVED / INFERENCE / HYPOTHESIS; the
  chronological TEST record remains authoritative for production claims;
  the Phase 14 record is auxiliary robustness evidence only.

## 23. Explicit list of things Phase 14 is NOT allowed to change

- The frozen model `lgbm-graph-v1` (weights, config, seed, iteration).
- The frozen feature set, feature registry, or any feature definition.
- The production threshold `0.9186015432508062` and the Phase 9 alert
  policy.
- The Phase 10 conformal method and its calibration.
- The Phase 11 explainability contract (including the no-counterfactual
  rule).
- The chronological split, the authoritative chronological TEST, or any
  frozen TEST record.
- Historical artifacts, phase reports, or previous phase results.
- The master report, decision log, experiment index, evaluation README, or
  risks README (until the phase is approved, executed, and verified).
- The raw dataset; no raw data is copied to the PC.
- No new dataset, no new release, no synthetic users, no fabricated labels.
- The production candidate and all production decisions.

---

## Appendix A — pre-registered constants

- Frozen threshold (primary operating point): `0.9186015432508062`.
- Bootstrap: n = 1000, seed 42, alpha = 0.10 (90% CIs), row-level and
  user-block.
- Wilson CI: z = 1.6448536269514722 (90%).
- Early stopping: patience 100, ≤ 3000 rounds, metric AUC, CAL split.
- Incidence length classes: short ≤ 12, long ≥ 45 positive days (OBSERVED
  bimodal; 13–44 empty).
- Interpretation bands (§20): AUC-ROC ±0.02 / −0.05; AUC-PR ±20% / −40%
  relative; coverage 12+ / 8–11 / ≤ 7 of 15.

## Appendix B — open design points for approval

1. **Early stopping on the 15-user user-disjoint CAL** vs fixed 186
   iterations: this spec adopts early stopping (mirrors the frozen
   protocol; best_iteration recorded as an outcome). The fixed-186 variant
   would isolate "unseen users" as the only changed factor; it can be
   adopted by amendment if reviewers prefer.
2. **Secondary operating point** (auxiliary CAL max-F1, descriptive): kept
   to separate model transfer from threshold transfer; it is descriptive
   only and can be dropped by amendment if reviewers consider it
   unnecessary.
3. **User-block bootstrap** as the primary CI: new estimator (tested on
   synthetic panels); row-level bootstrap kept for convention comparability.
4. **Benign-user allocation** (user-id round-robin 8/1/1): deterministic and
   auditable; no stratification by activity (rejected: adds complexity
   without a design question it answers).

*End of specification. Feasibility counts are OBSERVED from the frozen
derived table; protocol details are pre-registered here and must not be
invented or changed at execution time.*