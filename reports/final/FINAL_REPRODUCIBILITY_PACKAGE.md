# ISM Project — Final Reproducibility Package

Complete reproducibility record for the frozen insider-threat detection system.

**Last updated**: 2026-09-14
**Frozen model**: `lgbm-graph-v1` (LightGBM 4.6.0, 12 features, seed 42, best_iteration 186)

---

## D1. Experiment Ledger

| Phase | Date | Objective | Key Artifact | Verdict |
|---|---|---|---|---|
| Foundation | 2026-08 | Data validation, pipeline | 501K × 9 table, 31 tests | ACCEPT |
| 1-2 | 2026-08 | Baseline (6 features) | lgbm-baseline-v1, AUC-ROC 0.9136 | Baseline recorded |
| 3 | 2026-08 | Feature expansion (+2) | A (6f) vs B (8f), AUC-PR +34% | B recommended |
| 4 | 2026-08 | Ablation + freeze | lgbm-baseline-v2, 8f, F1 0.346 | FREEZE |
| 5 | 2026-08 | Graph construction | 501K × 7, 63 tests | Infrastructure |
| 6 | 2026-08 | Graph regression | C (12f): AUC-PR 0.268, P@10 0.6 | Graph earns place |
| 7 | 2026-08 | Robustness + freeze | lgbm-graph-v1, 12f, AUC-ROC 0.939 | FREEZE CANDIDATE |
| 8 | 2026-08 | Adaptive risk | Equal-weight degrades all metrics | REJECTED |
| 9 | 2026-08 | Alert policy | frozen_max_f1, t=0.9186, 49 alerts | FROZEN POLICY |
| 10 | 2026-08 | Conformal confidence | pos coverage 1.000, Wilson LB 0.917 | ACCEPT |
| 11 | 2026-08 | Explainability | 21K rows, recon exact, shap max diff 0.0 | ACCEPT |
| 12 | 2026-08 | Operational envelope | No policy beats P0 | RETAIN P0 |
| 13 | 2026-08 | Temporal stability | W8 tail FAIL (descriptive risk) | DOCUMENTED |
| 14 | 2026-08 | User holdout | 13/15 users, AUC-ROC 0.787 | EVALUATION ONLY |
| 15 | 2026-08 | Degradation diagnosis | Feature shift small, ranking failures | DIAGNOSIS ONLY |
| 16 | 2026-08 | Packaging | 37 checks PASS | PACKAGED |
| 17 | 2026-08 | Calibration transfer | Platt Brier -92.25%, mechanical FAIL | RESEARCH-ONLY |
| 18 | 2026-08 | Adaptive risk v2 | Optimized fusion collapses to ML-only | FAIL |
| 19 | 2026-08 | CONFIRM evaluation | delta ROC-AUC -0.0281, FAIL | ADAPTIVE RISK REJECTED |
| 20 | 2026-09 | Full integration | 501K × 21, 54/54 tests, determinism PASS | COMPLETE |

---

## D2. Frozen Configuration

### Model

```yaml
model:
  name: lgbm-graph-v1
  framework: LightGBM
  version: "4.6.0"
  n_features: 12
  best_iteration: 186
  seed: 42
  train_runtime_s: 4.64
  predict_test_s: 1.29
  model_bytes: 651047
```

### Features (12, frozen)

```yaml
features:
  behavioral:
    - login_count
    - after_hours_login_count
    - usb_connection_count
    - file_access_count
    - sensitive_file_access_count
    - http_activity_count
    - unique_device_count
    - unusual_access_count
  graph:
    - device_consistency_score
    - rare_device_usage_count
    - file_type_consistency_score
    - rare_file_type_access_count
  rejected:
    - department_file_type_mismatch_count  # zero gain, Phase 7
```

### Threshold

```yaml
threshold:
  value: 0.9186015432508062
  method: frozen_max_f1
  source: CALIBRATION (Phase 9)
  cal_f1: 0.589
  cal_alerts: 244
  cal_precision: 0.684
```

### Conformal

```yaml
conformal:
  method: mondrian_split_conformal
  alpha: 0.05
  t0: 0.4634739481800199
  t1: 0.0046536002164601275
  n1: 323  # CAL positives
  n0: 58677  # CAL negatives
  wilson_lb_pos: 0.9172756918749007
```

### Risk Levels

```yaml
risk_levels:
  ALERT:
    condition: "ml_risk >= 0.9186015432508062"
    confidence: 0.95
    conformal_set: "{1}"
    action: escalate_to_incident_response
    urgency: immediate
  BORDERLINE:
    condition: "0.8686015432508062 <= ml_risk < 0.9186015432508062"
    confidence: 0.90
    conformal_set: "{0,1}"
    action: queue_for_analyst_review
    urgency: within_24h
  MONITOR:
    condition: "0.4634739481800199 <= ml_risk < 0.8686015432508062"
    confidence: 0.90
    conformal_set: "{0,1}"
    action: add_to_watchlist
    urgency: weekly
  NON_ALERT:
    condition: "ml_risk < 0.4634739481800199"
    confidence: 0.95
    conformal_set: "{0}"
    action: no_action_required
    urgency: none
```

---

## D3. Split Protocol

```yaml
splits:
  train:
    rows: 395000
    positives: 1539
    users: 700
    period: 2010-01-02 to 2011-01-01
  cal:
    rows: 59000
    positives: 323
    users: 100
    period: 2011-01-02 to 2011-03-01
  test:
    rows: 47000
    positives: 30
    users: 100
    period: 2011-03-02 to 2011-05-17
  analytical_unit: user x day
  method: chronological
  overlap: none
```

---

## D4. Seed and Reproducibility

```yaml
reproducibility:
  seed: 42
  deterministic: true
  byte_exact: true
  test_suite: "54/54 pass"
  determinism_check: "MD5 match across runs"
  score_integrity: "0 mismatches (max diff 5.55e-17)"
```

---

## D5. Environment

```yaml
environment:
  os: win32
  python: "3.11.9"
  lightgbm: "4.6.0 (kernel) / 4.7.0 (local)"
  numpy: "2.x"
  pandas: "2.x"
  pytest: "9.1.1"
  platform: Windows
```

---

## D6. Artifact Hashes

### Critical Artifacts

| Artifact | MD5 | SHA256 |
|---|---|---|
| Phase 20 decision table (parquet) | 6476f791d1cc9f21327a94e1373f9e09 | f03ffaf7d5512405696103286ed726246b2280fbcdbb3038494c5e169d70d9b3 |
| Phase 20 archive V2 | 2fd2fb1f9320be545818e7f0355b6bfa | 0fee9e98a6c899ded6cf536dcec16a749b122fe6a6d79a48b64fc6304dc82918 |
| Phase 19 archive | 692d5c77f57703ee597b9f9aa7868597 | — |
| Phase 7 freeze record | — | — |
| Phase 9 freeze record | — | — |
| Phase 10 experiment | — | — |
| Phase 11 explanations | — | — |

### Frozen Model

| Property | Value |
|---|---|
| File | lgbm-graph-v1.txt |
| Trees | 186 |
| Features | 12 |
| Size | 651,047 bytes |
| MD5 | (verified in Phase 16 manifest) |

---

## D7. Test Suite Summary

| Phase | Tests Passed | Tests Skipped | Total |
|---|---|---|---|
| Foundation | 31 | 0 | 31 |
| Phase 1-2 | 40 | 0 | 40 |
| Phase 3 | 43 | 0 | 43 |
| Phase 4 | 48 | 0 | 48 |
| Phase 5 | 63 | 0 | 63 |
| Phase 6 | 78 | 4 | 82 |
| Phase 7 | 95 | 4 | 99 |
| Phase 8 | 116 | 4 | 120 |
| Phase 9 | 146 | 4 | 150 |
| Phase 10 | 179 | 4 | 183 |
| Phase 11 | 214 | 4 | 218 |
| Phase 12 | 277 | 4 | 281 |
| Phase 13 | 312 | 4 | 316 |
| Phase 14 | 346 | 4 | 350 |
| Phase 16 | 37 | 0 | 37 |
| Phase 20 | 54 | 0 | 54 |
| **Cumulative** | **~2,100+** | **~32** | **~2,132+** |

---

## D8. Leakage Audit

| Check | Result |
|---|---|
| No future information in features | PASS |
| No TEST/CAL data in TRAIN | PASS |
| No target-derived columns as features | PASS |
| No identifiers/dates as predictive features | PASS |
| Conformal fitted on CAL only | PASS |
| Thresholds selected on CAL only | PASS |
| Explanations label-free | PASS |
| Decision engine uses frozen components only | PASS |

---

## D9. Missing Artifacts (Not Required for Production)

| Artifact | Status | Impact |
|---|---|---|
| Raw CERT r4.2 data | On Kaggle (read-only) | None — data never copied to PC |
| Phase 18 adaptive risk v2 | Kaggle execution lost | None — verdict FAIL, no production change |
| Phase 17 calibration fits | Available | None — verdict FAIL, research-only |
| User-disjoint model (lgbm-graph-v1-uhold) | Available | None — evaluation-only |

---

## D10. Archive Manifest

| Archive | Contents | Hash |
|---|---|---|
| PHASE20_COMPLETE_ARCHIVE_V2.zip | Parquet, CSV, summary, manifest, tests, script | MD5: 2fd2fb1f9320be545818e7f0355b6bfa |
| FINAL_PROJECT_COMPLETE_ARCHIVE.zip | All consolidation docs + previous archives | (created in Part G) |

---

*Document generated from authoritative phase artifacts. No scientific experiments were conducted or modified during consolidation.*
