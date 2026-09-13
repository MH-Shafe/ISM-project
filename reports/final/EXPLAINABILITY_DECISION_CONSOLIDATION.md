# ISM Project — Explainability and Decision Consolidation

Consolidated record of the explainability layer (Phase 11) and the full
decision table (Phase 20), including coverage, distribution, and integration
evidence.

**Last updated**: 2026-09-12
**Frozen classifier**: `lgbm-graph-v1` (LightGBM 4.6.0, 12 features, seed 42, best_iteration 186)

---

## C1. SHAP Validation (Phase 11)

### Reconstruction Identity

LightGBM native `pred_contrib` (path-dependent TreeSHAP) in margin space:
`margin = bias + Σ contribution`. Reconstruction error across all splits:

| Split | Max Error | Mean Error | Rows |
|---|---|---|---|
| TRAIN | 4.9e-14 | ~1e-15 | 395,000 |
| CAL | 4.8e-14 | ~1e-15 | 59,000 |
| TEST | 3.6e-14 | ~1e-15 | 47,000 |

Tolerance: 1e-6. All splits pass.

### Cross-Validation (shap TreeExplainer)

| Metric | Value |
|---|---|
| Max absolute difference | 0.0 |
| Reconstruction error | ~5e-14 |
| Rows cross-checked | 500 (non-alert sample) |

shap TreeExplainer confirms native `pred_contrib` output is exact.

### Explanation Coverage

| Metric | Value |
|---|---|
| Total TEST rows explained | 21,043 |
| Direct SHAP explanations | 21,043 (4.2%) |
| Fallback explanations | 479,957 (95.8%) |
| Alerts explained | 49 (100%) |
| Monitor band rows | 20,494 |
| Non-alert sample | 500 |
| All alerts confident-positive | 49/49 (100%) |
| Trust diagnostics present | 476,200 (95.1%) |

### Feature Importance (Alert Rows)

| Feature | Top-1 Count (of 49 alerts) | Role |
|---|---|---|
| usb_connection_count | 36 | Dominant alert driver |
| http_activity_count | 26 | Bipolar (≈49% of |contribution|, signed alert mean ≈ 0) |
| file_access_count | 8 | Secondary behavioral |
| device_consistency_score | 9 | Top graph feature |
| file_type_consistency_score | 5 | Graph contributor |
| login_count | 2 | Minor |
| after_hours_login_count | 4 | Minor |
| unique_device_count | 3 | Minor |
| unusual_access_count | 2 | Minor |
| rare_file_type_access_count | 1 | Minor |
| sensitive_file_access_count | 0 | Never top-1 |
| rare_device_usage_count | 0 | Never top-1 |

### Graph Feature Contribution

| Metric | Value |
|---|---|
| Graph share of alert-row |contribution| | ~20.5% |
| device_consistency_score top-3 in CAL alerts | 72% |
| Graph features top-1 in TEST alerts | 9/49 (device + file_type) |

### Stability Under Perturbation

| Metric | Value |
|---|---|
| Decision flips per feature (unit perturbation) | ≤ 3.5% |
| Top-3 rank churn per feature | ≈ 95% |

**Interpretation**: Reason rank is fragile; decisions are not. The top-3
reason template provides a stable attribution narrative even when individual
rank positions are unstable.

### Source Artifact

`reports/artifacts/phase11_experiment.json`

---

## C2. Full Decision Table (Phase 20)

### Schema (21 columns)

| # | Column | Type | Description |
|---|---|---|---|
| 1 | user | string | Employee user ID |
| 2 | day | string | Calendar date YYYY-MM-DD |
| 3 | ml_risk | float | Frozen LightGBM risk score [0,1] |
| 4 | risk_level | string | ALERT / BORDERLINE / MONITOR / NON-ALERT |
| 5 | confidence | float | 0.95 (singleton) / 0.90 (ambiguous) |
| 6 | conformal_set | string | {1}, {0}, {0,1}, {} |
| 7 | p1 | float | Conformal p-value for class 1 |
| 8 | p0 | float | Conformal p-value for class 0 |
| 9 | alert_flag | boolean | True if ALERT or BORDERLINE |
| 10 | recommended_action | string | escalate_to_incident_response / queue_for_analyst_review / add_to_watchlist / no_action_required |
| 11 | urgency | string | immediate / within_24h / weekly / none |
| 12 | is_malicious | integer | 0 or 1 (EVALUATION ONLY) |
| 13 | top_1_reason | string | Highest |contribution| feature |
| 14 | top_2_reason | string | Second highest |contribution| feature |
| 15 | top_3_reason | string | Third highest |contribution| feature |
| 16 | contribution_top_1 | float | Absolute contribution of top-1 feature |
| 17 | contribution_top_2 | float | Absolute contribution of top-2 feature |
| 18 | contribution_top_3 | float | Absolute contribution of top-3 feature |
| 19 | trust_flag | boolean | True if SHAP diagnostics pass |
| 20 | shap_method | string | direct / fallback |
| 21 | graph_contribution_pct | float | Graph feature share of total |contribution| |

### Decision Distribution (501,000 rows)

| Risk Level | Count | % | Action | Urgency |
|---|---|---|---|---|
| ALERT | 3,785 | 0.76% | escalate_to_incident_response | immediate |
| BORDERLINE | 1,484 | 0.30% | queue_for_analyst_review | within_24h |
| MONITOR | 178,204 | 35.57% | add_to_watchlist | weekly |
| NON-ALERT | 317,527 | 63.38% | no_action_required | none |

| Alert Flag | Count | % |
|---|---|---|
| True (ALERT + BORDERLINE) | 5,269 | 1.05% |
| False (MONITOR + NON-ALERT) | 495,731 | 98.95% |

### Confidence Distribution

| Confidence | Count | % | Reason |
|---|---|---|---|
| 0.95 (high) | 61,017 | 12.2% | ALERT or NON-ALERT (conformal set singleton) |
| 0.90 (ambiguous) | 439,983 | 87.8% | BORDERLINE or MONITOR (conformal set {0,1}) |

### Explanation Coverage

| Metric | Count | % |
|---|---|---|
| Rows with top_3_reasons | 501,000 | 100% |
| Rows with trust_flag | 476,200 | 95.1% |
| Rows without trust_flag | 24,800 | 4.9% |

### Score Integrity

| Metric | Value |
|---|---|
| Predictions compared (TEST overlap) | 47,000 |
| Score mismatches | 0 |
| Max absolute difference | 5.55e-17 |
| Alert-only mismatches | 0 |
| BORDERLINE additions (expected) | 58 |

### Determinism

| Metric | Value |
|---|---|
| Run A MD5 | ad017e59168a2a211d0ba74422132d46 |
| Run B MD5 | ad017e59168a2a211d0ba74422132d46 |
| Verdict | PASS |

### Performance

| Metric | Value |
|---|---|
| Runtime | 180.34 seconds |
| Peak memory | 1,744.2 MB |
| Output size (parquet) | 2.76 MB |
| Output size (CSV) | 166 MB |

### Risk Level Thresholds

| Risk Level | Condition | Confidence | Conformal Set |
|---|---|---|---|
| ALERT | ml_risk >= 0.9186015432508062 | 0.95 | {1} |
| BORDERLINE | 0.8686015432508062 <= ml_risk < 0.9186015432508062 | 0.90 | {0,1} |
| MONITOR | 0.4634739481800199 <= ml_risk < 0.8686015432508062 | 0.90 | {0,1} |
| NON-ALERT | ml_risk < 0.4634739481800199 | 0.95 | {0} |

### Test Suite

| Metric | Value |
|---|---|
| Total tests | 54 |
| Passed | 54 |
| Failed | 0 |
| Unit tests | 39 |
| Full-table integration tests | 15 |

### Source Artifacts

| Artifact | Path |
|---|---|
| Decision table | `reports/artifacts/phase20/final_user_day_decisions.parquet` |
| Summary | `reports/artifacts/phase20/final_user_day_decisions_summary.json` |
| CSV | `reports/artifacts/phase20/final_user_day_decisions.csv` |
| Manifest | `reports/artifacts/phase20/phase20_reproducibility_manifest.json` |
| Script | `scripts/phase20_materialize_decisions.py` |
| Tests | `tests/test_phase20.py` |

---

## C3. Integration Summary

### Layer Stack (Bottom to Top)

```
 CERT r4.2 logs (read-only)
 → User-day aggregation (Phase 0)
 → 8 behavioral features (Phase 3-4, frozen)
 → 4 graph features (Phase 5-7, frozen)
 → LightGBM lgbm-graph-v1 (Phase 7, frozen)
 → Alert threshold 0.9186 (Phase 9, frozen)
 → Risk levels: ALERT/BORDERLINE/MONITOR/NON-ALERT (Phase 11)
 → Conformal confidence overlay (Phase 10, accepted)
 → Explanations (Phase 11, accepted)
 → Full decision table (Phase 20, complete)
```

### What Changed at Each Layer

| Layer | Adds | Changes Frozen Outputs? |
|---|---|---|
| Features (frozen) | 12 numeric columns | No (input) |
| Model (frozen) | ml_risk score | No (input) |
| Threshold (frozen) | Binary alert decision | No (input) |
| Risk levels | risk_level, alert_flag, recommended_action, urgency | No (derived from frozen threshold) |
| Conformal | conformal_set, p0, p1, confidence | No (overlay) |
| Explanations | top_3_reasons, contributions, trust_flag, graph_contribution_pct | No (overlay) |
| Decision table | All 21 columns materialized | No (integration) |

### Adaptive Risk Status

Adaptive Risk is **excluded from the production path**:
- Phase 8: Rejected (equal-weight degrades all metrics)
- Phase 19: CONFIRM FAIL (delta ROC-AUC -0.0281 < -0.01)
- Phase 20: Explicitly excluded from decision table schema

---

*Document generated from authoritative phase artifacts. No scientific experiments were conducted or modified during consolidation.*
