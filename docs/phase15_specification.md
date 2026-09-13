# Phase 15 Specification — Unseen-User Generalization Diagnosis

Status: APPROVED (authorized; internal consistency confirmed; inputs verified).
Last updated: 2026-08-17.

## 0. Objective

Diagnose the causes of the degradation observed in the user-disjoint Phase 14
evaluation (AUC-ROC 0.7865 vs 0.9392 chronological; precision 0.2018 vs 0.286;
1,026 vs 49 alerts) without touching the frozen production system.

This is a **diagnostic, read-only, evaluation-free** phase: no training, no
threshold choice, no production change, no chronological TEST rerun. All
statistics are descriptive; every result is labeled OBSERVED / INFERENCE /
HYPOTHESIS (ISM principles §12).

## 1. Hard constraints (inherited)

- CERT r4.2 only; raw data never leaves Kaggle.
- Do NOT rerun Phase 14; do NOT retrain or modify `lgbm-graph-v1` (12 features,
  seed 42, 186 trees, threshold `0.9186015432508062`), its chronological TEST
  record, or accepted overlays (Phases 9/10/11/12).
- The authoritative chronological TEST (47,000 rows, 2011-04-01..2011-05-17)
  is NOT read, scored, or analyzed in this phase.
- No TEST-driven tuning, no new threshold, no GNNs/transformers/adaptive risk.
- Phase 14 auxiliary user-disjoint TEST: descriptive analysis only.
- Preserve the Phase 14 allocation exactly (no re-allocation).

## 2. Inputs (all local, frozen, md5-verified; no Kaggle execution required)

| Input | Source | Role |
|---|---|---|
| `phase6_merged_features.parquet` (501,000 × 16, md5 `9a3b1885…9def`) | frozen | rows, 12 features, `is_malicious` |
| `phase14_split.json` | verified artifact | user-disjoint allocation (TRAIN 784 / CAL 108 / TEST 108) |
| `phase14_test_predictions.parquet` (54,108 × 6) | verified artifact | TEST scores, alert_primary, alert_secondary |
| `phase14_user_diagnostics.json` | verified artifact | per-user detection, onset, strata, delay |
| `phase14_model.txt` + `phase14_model_record.json` | verified artifacts | auxiliary model for CAL score reproduction |
| `phase14_calibration.json` | verified artifact | recorded CAL AUC-ROC for reproduction gate |
| `phase7_freeze_lgbm-graph-v1.json`, `phase9_freeze.json`, `phase12_test_policy_results.json`, `phase13_verdict.json` | frozen records | cohort-comparison (G) numbers |

Analysis unit: user × day (501-day full timeline per user); user-level
aggregation where the analysis requires it.

## 3. Allocation and cohorts

Cohorts are the Phase 14 user-disjoint blocks (all users span the full
timeline; temporal position is therefore cohort-invariant by construction):

- TRAIN block: 784 users, 392,784 rows (40 malicious; 1,037 malicious days).
- CAL block: 108 users, 54,108 rows (15 malicious; 379 malicious days).
- TEST block: 108 users, 54,108 rows (15 malicious; 476 malicious days).

Malicious-user reference set (strata analysis): the 70 malicious users of the
allocation (40+15+15).

## 4. Reproduction gates (before any Phase 15 statistic is accepted)

1. **Local scoring gate**: locally reproduce the 54,108 TEST scores from
   `phase14_model.txt`; require `max |local − artifact| ≤ 1e-12` (observed
   `5.55e-17` with lightgbm 4.7.0 local vs 4.6.0 kernel). If the gate fails,
   CAL scoring is not performed and F/C rely on TEST scores only (recorded).
2. **CAL AUC gate**: AUC-ROC of locally produced CAL scores vs CAL labels must
   equal the recorded `0.7355124228200639` within `1e-9`.
3. Input md5s re-verified at run start (merged parquet, predictions parquet,
   model file) and recorded in `phase15_experiment.json`.

CAL per-row scores are produced locally from the frozen auxiliary model
(diagnostic reproduction; deterministic). TRAIN block is NOT scored
(in-sample scores would be overfit-biased); TRAIN contributes feature
distributions and reference statistics only.

## 5. Analysis A — User activity strata

- Strata of malicious users by number of malicious days (`is_malicious==1`
  rows): fixed pre-registered buckets `{1}`, `{2..5}`, `{6..12}`, `{13..44}`,
  `{45..100}`, `{101+}` (mirrors the short/long incident structure of r4.2).
- For each stratum (over the 70 malicious users, and separately over the 15
  TEST-block malicious users): n_users, malicious days, detected users,
  detection rate, missed malicious days, share of TEST alerts issued to
  stratum members (alert concentration), mean/max score of malicious days.
- Detection uses `phase14_user_diagnostics.json` per-user flags and
  `alert_primary` rows.
- Empty strata are reported as empty (not omitted); zero-positive groups
  reported with `n_users=0`.
- Output: `phase15_activity_strata.json`.

## 6. Analysis B — Feature-shift (TRAIN → CAL → TEST cohorts)

- Features: the 12 frozen features — 8 behavioral
  (`login_count`, `after_hours_login_count`, `usb_connection_count`,
  `file_access_count`, `sensitive_file_access_count`, `http_activity_count`,
  `unique_device_count`, `unusual_access_count`) and 4 graph
  (`device_consistency_score`, `rare_device_usage_count`,
  `file_type_consistency_score`, `rare_file_type_access_count`).
- User-level statistics per cohort: per-user median of each feature over the
  user's 501-day timeline, per-user zero-fraction, per-user total activity
  (`log1p(sum)`). Counts are `log1p`-transformed before effect-size and KS
  computation (skew control); consistency scores (`[0,1]`) are used raw.
- Drift measures (all pre-registered, computed on user-level values):
  1. Cohen's `d` (on `log1p` user-medians): TRAIN↔TEST, CAL↔TEST.
  2. Two-sample KS distance on the same values (with exact p via scipy;
     p is descriptive, not a decision rule).
  3. Zero-fraction shift (cohort difference in fraction of users whose
     median is 0).
  4. Row-level PSI (Population Stability Index): 10 equal-frequency bins
     fitted on TRAIN-block rows only (per feature, `log1p` for counts), then
     applied to CAL- and TEST-block rows. PSI is a report-only heuristic
     (no threshold decision).
- "Largest shift" ranking: features ranked by the mean of the standardized
  |d| and PSI ranks; top 3 reported.
- Output: `phase15_feature_shift.json` (per-feature cohort user-level
  quantiles, d, KS, zero-fraction, PSI, ranking).

## 7. Analysis C — Detected vs missed malicious users

- TEST block: 13 detected vs 2 missed users (JJM0203, WDD0366).
- Per-user comparison: n_malicious_days, n_active_days, total activity
  (`log1p`), activity tercile, length class, scenario, onset, temporal half,
  user-history length at onset (active days before onset), max score, n
  alerts, graph feature user-level means (4 graph features), consistency
  score mean and std.
- Reported as: full per-user table (both missed users individually), medians
  and ranges for the detected cohort, and the missed-vs-detected deltas.
  Because n=2, ALL comparative statements are descriptive; no significance
  test is run or claimed.
- Output: `phase15_detected_vs_missed.json`.

## 8. Analysis D — Graph/trust diagnosis

- (i) Cohort profiles: user-level distribution (median, zero-fraction, tail)
  of the 4 graph features per cohort; quantify whether unseen users show
  systematically different graph profiles (d and KS as in B, graph subset).
- (ii) Discriminative value: per-feature row-level AUC-ROC (feature vs
  `is_malicious`) computed separately inside the TRAIN block and inside the
  TEST block (log1p for counts). This measures whether each frozen graph
  feature separates malicious activity for unseen users as well as for
  TRAIN-block users. No model is fitted (single-feature ranking only).
- (iii) Stability: user-level std of `device_consistency_score` and
  `file_type_consistency_score` vs user activity (n_active_days) in the TEST
  block (Pearson r, descriptive); fraction of rows at consistency extremes
  (==1.0 or ==0.0) per cohort. Tests the HYPOTHESIS that consistency
  features are unstable for low-activity unseen users.
- No graph-feature redesign. Output: `phase15_graph_diagnosis.json`.

## 9. Analysis E — Temporal analysis (TEST block)

- Onset (first malicious day) per malicious user: timeline position
  (2010 H1 / 2010 H2 / 2011), onset ordinal among the user's active days
  (history length before onset), detection by onset half.
- Detection delay: days between first `alert_primary` day and onset (only for
  the 13 detected users; delay is negative when the first alert precedes the
  onset — alerts also fire on benign days; measured and labeled OBSERVED).
- Early vs late TEST cohort: users with onset before vs on/after the median
  TEST-block onset; detection rate and score behavior per group.
- First-week / first-tercile detection reuse from Phase 14 diagnostics
  (OBSERVED records) for completeness.
- Output: `phase15_temporal_analysis.json`.

## 10. Analysis F — Threshold diagnostic (descriptive only)

- Score distributions around the frozen threshold `0.9186015432508062`:
  percentiles of malicious-day scores and benign-day scores in the TEST
  block; missed-positive gap (threshold − max score per missed malicious
  user; and row-level distance for all 476 missed-positives... all 476 are
  missed by definition of per-user detection; instead report row-level
  distribution of malicious-day scores of undetected users); near-miss
  fraction (missed malicious days with score within 0.1 / 0.2 of the
  threshold); alerted-benign margin (score − threshold for benign alerted
  rows, concentration of false-positive mass just above the threshold).
- CAL block (locally reproduced scores): same near-threshold statistics and
  the score CDF comparison CAL vs TEST (same model, both unseen-user
  cohorts) — descriptive only.
- Explicitly: NO threshold selection, no operating-point recommendation, no
  TEST-based tuning. Output: `phase15_threshold_diagnostic.json`.

## 11. Analysis G — Cohort comparison (synthesis from frozen records + A–F)

- Table of OBSERVED records: chronological CAL (P7: AUC-ROC 0.888±0.016;
  P9: 244 alerts, precision 0.684, recall 0.517, F1 0.589, coverage 8/10),
  chronological TEST (P12: AUC-ROC 0.9392, AUC-PR 0.2678, 49 alerts,
  precision 0.286, recall 0.467, F1 0.352, coverage 7/10, P@10 1.0),
  Phase 13 stability verdict (FAIL, 2/7 consistent under 2011 conditions),
  user-disjoint CAL (P14: AUC-ROC 0.7355, recorded), user-disjoint TEST
  (P14: AUC-ROC 0.7865, 1,026 alerts, precision 0.2018, recall 0.4349,
  F1 0.2756, coverage 13/15).
- Attribution: statements distinguishing what CAN be attributed to unseen
  users (entity-level shift — user-disjoint CAL is also degraded vs
  chronological CAL under the same model family; alert explosion without
  threshold change is consistent with score-scale/calibration shift) from
  what CANNOT (different model instances; early-stopping optimism in the
  recorded CAL number; single-entity-block chance). Every attribution is
  labeled OBSERVED / INFERENCE / HYPOTHESIS.
- Output: `phase15_cohort_comparison.json`.

## 12. Determinism and environment

- All computations deterministic by construction (no RNG, no resampling).
- Determinism test: run the full `phase15.py` main twice on a small fixture
  and require byte-identical JSON artifacts.
- Environment recorded in `phase15_experiment.json` (Python, lightgbm,
  pandas, numpy, scipy versions; lightgbm 4.7.0 local vs 4.6.0 kernel noted).

## 13. Artifacts

`reports/artifacts/phase15_experiment.json`, `phase15_activity_strata.json`,
`phase15_feature_shift.json`, `phase15_detected_vs_missed.json`,
`phase15_graph_diagnosis.json`, `phase15_temporal_analysis.json`,
`phase15_threshold_diagnostic.json`, `phase15_cohort_comparison.json`,
`phase15_cal_scores.parquet` (locally reproduced CAL scores; TEST scores are
NOT re-saved — they remain the Phase 14 artifact), `phase15_manifest.json`
(md5s of all Phase 15 artifacts).

## 14. Tests

`tests/test_phase15.py`:
- unit tests for every diagnostic calculation (strata, d/KS/PSI,
  detected-vs-missed, graph AUC, temporal stats, threshold stats, cohort
  table);
- edge cases: empty strata, zero-positive groups, single-user cohort,
  all-missed cohort, constant feature (guards for d/KS/PSI), empty frame;
- determinism test (double-run byte-identical);
- artifact reloadability test (all JSON/parquet artifacts load, schema
  checked);
- local score-reproduction gate test on a small model fixture;
- the full local suite must pass; exact counts reported.

## 15. Evidence labels

Every artifact value and every report statement is labeled
OBSERVED (computed from verified artifacts), INFERENCE (reasoned from
OBSERVED), or HYPOTHESIS (expected, not demonstrated). No fabrication; no
unmeasured claims.

## 16. Acceptance criteria

- Gates 1–3 pass (or documented fallback per §4).
- All analyses A–G produced with pre-registered definitions.
- Report `reports/phase15_unseen_user_diagnosis_report.md` written.
- Master report + knowledge layer updated; journal entry added.
- Verification gate (final): artifacts independent md5 verification; no
  frozen artifact changed; no production model/threshold changed; no
  chronological TEST rerun; all reported numbers trace to artifacts.

## 17. Execution

Fully local (lightgbm 4.7.0 available; merged table local). No Kaggle
execution required. If a gate fails, execution stops and the failure is
reported for authorization.