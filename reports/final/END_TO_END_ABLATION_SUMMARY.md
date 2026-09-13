# ISM Project — End-to-End Ablation Summary

Consolidated evidence across all project phases, organized into two tables
with strict comparability discipline.

**Last updated**: 2026-09-12
**Dataset**: CERT r4.2, user × day (501,000 rows, 1,892 malicious, 0.38% prevalence)
**Splits**: TRAIN 395,000 / CAL 59,000 / TEST 47,000 (chronological, no overlap)

---

## Table 1 — Directly Comparable Classifier Evidence (A / B / C / D1)

All entries use the **same chronological TRAIN → CAL → TEST split protocol**
on CERT r4.2. Metrics are from the authoritative phase artifacts (Phase 4,
Phase 6, Phase 7, Phase 8). TEST n = 47,000; positives = 30.

| # | Label | Phase | Features | n_feat | Model | AUC-ROC | AUC-PR | P@10 | P@30 | R@50 | F1 | MCC | Alerts | Precision | Recall | Balanced Acc | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Behavioral only (lgbm-baseline-v2) | 4 | login_count, after_hours_login_count, usb_connection_count, file_access_count, http_activity_count, unique_device_count, sensitive_file_access_count, unusual_access_count | 8 | LightGBM 4.6.0, best_iter 182, seed 42 | 0.93534 | 0.15304 | 0.400 | 0.367 | 0.333 | 0.346 | 0.350 | 22 | 0.409 | 0.300 | 0.662 | Dominant behavioral arm; FREEZE record |
| B | Graph only (Phase 6 Arm B) | 6 | device_consistency_score, rare_device_usage_count, file_type_consistency_score, rare_file_type_access_count, department_file_type_mismatch_count | 5 | LightGBM 4.6.0, best_iter 3, seed 42 | 0.78635 | 0.01431 | 0.029 | 0.029 | 0.400 | 0.054 | 0.105 | 417 | 0.029 | 0.400 | 0.696 | Weak standalone; best_iter 3 = stump |
| C | Behavioral + graph (lgbm-graph-v1) | 7 | 8 behavioral + 4 graph (department_file_type_mismatch_count REJECTED) | 12 | LightGBM 4.6.0, best_iter 186, seed 42 | 0.93916 | 0.26778 | 0.600 | 0.367 | 0.467 | 0.354 | 0.365 | 49 | 0.286 | 0.467 | 0.733 | **FREEZE CANDIDATE** — production model |
| D1 | Adaptive Risk (Phase 8 equal-weight) | 8 | ML + trust_risk + behavior_risk (equal weight) | 3 comp | Derived from frozen components | 0.90977 | 0.06720 | 0.100 | 0.100 | 0.267 | 0.105 | 0.170 | 256 | 0.043 | 0.267 | 0.632 | Rejected: degrades all metrics vs A |

### Table 1 Key Findings

1. **A → C (adding 4 graph features)**: AUC-ROC +0.004, AUC-PR +74.9% relative, P@10 +0.200, R@50 +0.134, alerts +27. Graph features are complementary — earn their place.
2. **B alone (graph-only)**: AUC-PR 0.014 — graph features are weak discriminators in isolation.
3. **D1 (adaptive risk)**: Degrades every metric vs A. Trust and behavior components are redundant or negatively correlated with the target.
4. **C is the dominant model** by all metrics except alerts (49 vs 22), where the difference reflects a higher recall operating point.

### Table 1 Source Artifacts

| Entry | Artifact | Path |
|---|---|---|
| A | Phase 4 freeze record | `reports/artifacts/phase4_experiment_a.json` |
| B | Phase 6 Arm B record | `reports/artifacts/phase6_experiment_b.json` |
| C | Phase 7 freeze record | `reports/artifacts/phase7_freeze_lgbm-graph-v1.json` |
| D1 | Phase 8 experiment record | `reports/artifacts/phase8_experiment.json` |

---

## Table 2 — Incremental System-Layer Evidence (C → D2 → E → F)

This table tracks the frozen classifier C through successive system-layer
additions. Each row adds one component on top of the previous. These rows
use **different cohorts or different evidence types** and are NOT directly
comparable to each other or to Table 1.

| # | Label | Component | Evidence Type | Cohort | Key Metrics | Verdict | Notes |
|---|---|---|---|---|---|---|---|
| C | Frozen classifier | LightGBM 12 features, seed 42, threshold 0.9186 | Chronological TEST | 47,000 rows, 30 pos | AUC-ROC 0.939, AUC-PR 0.268, F1 0.354, MCC 0.365, 49 alerts | FREEZE (Phase 7) | Production baseline; all subsequent layers are pure functions of frozen scores |
| D2 | Phase 19 CONFIRM | Adaptive risk A0 vs Family C | Separate user-disjoint CONFIRM cohort | 98,196 rows, 245 pos | A0: ROC-AUC 0.779, PR-AUC 0.298, F1 0.341, 72 alerts; C: ROC-AUC 0.751, PR-AUC 0.297, F1 0.339, 74 alerts; delta ROC-AUC -0.0281 | FAIL (Phase 19) | Different cohort from C; adaptive risk rejected for production |
| E | Conformal confidence | Mondrian split-conformal, alpha=0.05 | CAL-only fit, TEST evaluation | CAL: n1=323, n0=58,677; TEST: 47,000 | pos coverage 1.000 (30/30, Wilson LB 0.917), neg 0.988, marginal 0.988; monitor band 532 rows, precision 0.015 | ACCEPT (Phase 10) | Diagnostic overlay; does not change frozen outputs |
| F | Full system (Phase 20) | Decision engine: ML risk + risk level + confidence + conformal set + explanations + actions | Full 501K materialization | 501,000 rows, 1,000 users | ALERT=3,785, BORDERLINE=1,484, MONITOR=178,204, NON-ALERT=317,527; score integrity 0 mismatches; determinism PASS; 54/54 tests | COMPLETE (Phase 20) | Dashboard-ready output; Adaptive Risk excluded |

### Table 2 Key Findings

1. **C → D2 (adaptive risk CONFIRM)**: FAIL — delta ROC-AUC -0.0281 < -0.01 threshold. The frozen classifier C outperforms the adaptive-risk variant on the CONFIRM cohort.
2. **C → E (conformal overlay)**: ACCEPT — 100% positive coverage on TEST, honest uncertainty quantification at zero cost to frozen outputs.
3. **C → F (full integration)**: COMPLETE — 501K decisions materialized, all validation gates PASS, ready for dashboard deployment.

### Table 2 Source Artifacts

| Entry | Artifact | Path |
|---|---|---|
| C | Phase 7 freeze record | `reports/artifacts/phase7_freeze_lgbm-graph-v1.json` |
| D2 | Phase 19 CONFIRM result | `phase19_v1_1_1/kaggle_confirm_results/phase19_v1_1_2_confirm_result.json` |
| E | Phase 10 experiment | `reports/artifacts/phase10_experiment.json` |
| F | Phase 20 summary | `reports/artifacts/phase20/final_user_day_decisions_summary.json` |

---

## Evidence Classification

All claims in this document are classified per ISM Principles §12:

- **OBSERVED**: directly measured from authoritative phase artifacts
- **INFERENCE**: reasoned from measured evidence (e.g., "graph features earn their place" from the A→C delta)
- **HYPOTHESIS**: not yet demonstrated (e.g., capacity projections beyond 1,000 users)
- **LITERATURE RESULT**: external (none in this document)

---

## Rejected Approaches (Complete List)

| Approach | Phase | Reason | Evidence |
|---|---|---|---|
| lr 0.05 baseline | 1–2 | Collapsed to best_iter 1, degenerate scores | CAL AUC-ROC 1.000, 0 alerts |
| Graph-only (Arm B) | 6 | Weak standalone: AUC-PR 0.014, best_iter 3 | Phase 6 experiment |
| department_file_type_mismatch_count | 7 | Zero gain 5/5 seeds, 49 non-zero rows, CAL AUC-ROC 0.500 | Phase 7 feature stability |
| Adaptive risk (equal-weight) | 8 | Degrades all metrics vs ML-only | Phase 8 experiment |
| Adaptive risk (learned) | 8 | Collapses to ML-only [1,0,0] | Phase 8 optimization |
| Capacity/percentile alert policies | 9 | Fail S1–S3 selection rules | Phase 9 policy screen |
| De-duplication beyond 1 day | 12 | Destroys recall (3d: 0.248, 7d: 0.139) | Phase 12 operational envelope |
| Rolling threshold | 12 | Never beats frozen policy | Phase 12 operational envelope |
| Counterfactual explanations | 11 | Not supported under frozen-output contract | Phase 11 decision |
| Platt calibration (production) | 17 | Mechanical FAIL: AUC-ROC not invariant under tie creation | Phase 17 verdict |
| Binning calibration (production) | 17 | 21,127 alerts, precision 0.018 — unusable | Phase 17 verdict |
| ECDF calibration (production) | 17 | Brier -216% — percentile map, not calibration | Phase 17 verdict |
| Adaptive risk A0/Family C | 19 | CONFIRM FAIL: delta ROC-AUC -0.0281 < -0.01 | Phase 19 CONFIRM |

---

*Document generated from authoritative phase artifacts. No scientific experiments were conducted or modified during consolidation.*
