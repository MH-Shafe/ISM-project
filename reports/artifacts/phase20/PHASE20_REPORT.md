# Phase 20: Final System Integration — Report

## Purpose

Phase 20 integrates ALL frozen scientific components into a single
deterministic, reproducible, leakage-safe decision engine that produces
analyst-facing output for each user-day.

## Scope

- **In scope**: Integration of frozen components, output schema design,
  deterministic decision rules, provenance documentation, validation tests.
- **Out of scope**: No new scientific evidence, no model retraining,
  no threshold tuning, no feature engineering.

## Frozen Components Integrated

| Component | Source | Key Values |
|-----------|--------|------------|
| LightGBM model | Phase 7 freeze | lgbm-graph-v1, seed=42, iter=186 |
| Feature set | Phase 7 freeze | 12 features (8 behavioral + 4 graph) |
| Alert threshold | Phase 9 freeze | 0.9186015432508062 |
| Conformal layer | Phase 10 | Mondrian split-conformal, alpha=0.05 |
| t0 threshold | Phase 10 | 0.4634739481800199 |
| t1 threshold | Phase 10 | 0.0046536002164601275 |
| Explainability | Phase 11 | TreeSHAP/pred_contrib |
| Graph diagnostics | Phase 5-7 | 4 graph features |

## Decision Engine Architecture

```
Input (user, day, features)
  |
  v
Frozen LightGBM model --> ml_risk (0-1)
  |
  v
Frozen thresholds --> risk_level
  |  ALERT:        score >= 0.9186
  |  HUMAN_REVIEW: 0.4635 <= score < 0.9186
  |  MONITOR:      0.0047 <= score < 0.4635
  |  NO_ACTION:    score < 0.0047
  |
  v
Frozen conformal fit --> confidence, p-values, prediction set
  |
  v
Graph features --> trust flags (context, not labels)
  |
  v
SHAP/pred_contrib --> top explanation reasons
  |
  v
Deterministic policy --> recommended action + urgency
  |
  v
Dashboard-ready output (16 columns)
```

## Risk Level Grounding

All risk levels are grounded in frozen thresholds from CALIBRATION-only
selection:

- **ALERT** (score >= 0.9186): Phase 9 frozen_max_f1 threshold
- **BORDERLINE** (0.8686 <= score < 0.9186): Phase 11 BORDERLINE_WIDTH = 0.05
- **MONITOR** (0.4635 <= score < 0.8686): Phase 10 conformal t0
- **NON-ALERT** (score < 0.4635): Below Phase 10 conformal t0

## Output Schema

16 columns including: user, day, ml_risk, risk_level, confidence,
conformal_set, p1, p0, alert_flag, recommended_action, urgency,
rationale, trust_flags, n_trust_flags, graph_features, top_reasons.

## Files Created

### Source Code
- `src/experiments/phase20.py` — Core decision engine module

### Configuration
- `configs/phase20/final_feature_schema.yaml` — Feature schema
- `configs/phase20/conformal_integration.yaml` — Conformal verification
- `configs/phase20/output_schema.yaml` — Output schema
- `configs/phase20/decision_policy.yaml` — Decision policy documentation

### Tests
- `tests/test_phase20.py` — 39 validation tests (ALL PASS)

### Artifacts
- `reports/artifacts/phase20/phase20_input_inventory.json` — Component inventory
- `reports/artifacts/phase20/phase20_dashboard_sample.json` — Sample output
- `reports/artifacts/phase20/phase20_analyst_summary.md` — Analyst summary
- `reports/artifacts/phase20/phase20_traceability.md` — Provenance chain
- `reports/artifacts/phase20/phase20_leakage_audit.md` — Leakage audit
- `reports/artifacts/phase20/phase20_reproducibility_manifest.json` — Manifest

## Verification

- All 39 tests pass
- All frozen constants match authoritative artifacts
- Conformal t0/t1 verified against phase10_calibration.json
- Phase 19 verdict confirmed: FAIL (Adaptive Risk excluded)
- No leakage detected
- Deterministic and reproducible

## Forbidden Operations (Enforced)

- No model retraining or refitting
- No new feature generation
- No threshold tuning or adaptation
- No Adaptive Risk fusion in production path
- No label-dependent operational rules
- No post-hoc threshold selection
- No new ML model creation
- No TEST data usage in production

## Known Limitations

1. **Conformal sample size**: n1=323 (positive class) is small
2. **Temporal distribution shift**: Conformal assumes exchangeability
3. **Feature freeze**: 12-feature set frozen since Phase 7

## Conclusion

Phase 20 successfully integrates all frozen scientific components into a
deterministic, reproducible, leakage-safe decision engine. The system is
ready for dashboard deployment with full provenance and traceability.

---

## Full-Scale Operational Validation

The complete dashboard-ready user-day decision table has been materialized.

### Complete Decision Table

- **File**: `reports/artifacts/phase20/final_user_day_decisions.parquet`
- **CSV**: `reports/artifacts/phase20/final_user_day_decisions.csv`
- **Rows**: 501,000 (1,000 users x 501 days)
- **Columns**: 21 (user_id, date, ml_risk_score, risk_level, alert_flag, confidence_status, conformal fields, trust diagnostics, recommended action, provenance)
- **Schema version**: phase20-v1

### Coverage

| Metric | Value |
|--------|-------|
| Total user-days | 501,000 |
| Unique users | 1,000 |
| Date range | 2010-01-02 to 2011-05-17 |
| Missing primary keys | 0 |
| Duplicate user-days | 0 |

### Decision Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| ALERT | 3,785 | 0.76% |
| BORDERLINE | 1,484 | 0.30% |
| MONITOR | 178,204 | 35.57% |
| NON-ALERT | 317,527 | 63.38% |

### Confidence Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| high-confidence | 61,017 | 12.18% |
| ambiguous | 439,983 | 87.82% |

### Explanation Coverage

- Rows with 3 SHAP reasons: 501,000 (100.0%)
  - SHAP explanations (Phase 11): 21,043 (4.2%)
  - Feature-value fallback: 479,957 (95.8%)
- Rows with trust diagnostics: 476,200 (95.1%)

### Score Integrity

- Frozen model inference verified against Phase 7 predictions: 47,000 rows, 0 mismatches
- Maximum absolute difference: 5.55e-17 (floating-point tolerance)

### Alert Integrity

- ALERT-only comparison vs Phase 10: 0 mismatches
- 58 BORDERLINE additions: expected (Phase 20 alert_flag = ALERT|BORDERLINE per output_schema.yaml; Phase 10 alert = ALERT only)

### Determinism

- Run A MD5: `ad017e59168a2a211d0ba74422132d46`
- Run B MD5: `ad017e59168a2a211d0ba74422132d46`
- Science outputs identical: PASS

### Performance

| Metric | Value |
|--------|-------|
| Wall-clock runtime | 180.34s |
| Peak memory | 1,744.2 MB |
| Parquet size | 2,759,459 bytes |
| CSV size | 166,097,220 bytes |

### Output Hashes

- Parquet MD5: `6476f791d1cc9f21327a94e1373f9e09`
- Parquet SHA256: `f03ffaf7d5512405696103286ed726246b2280fbcdbb3038494c5e169d70d9b3`

### Leakage Audit

- LABEL_REQUIRED_FOR_DECISION: NO
- GROUND_TRUTH_IN_PRODUCTION_TABLE: NO
- FUTURE_INFORMATION_USED: NO

### Forbidden Operations Verified

| Operation | Status |
|-----------|--------|
| Model retrained | NO |
| Threshold changed | NO |
| Calibration refit | NO |
| Conformal refit | NO |
| Adaptive Risk used | NO |
| PH19_CONFIRM accessed | NO |

### Test Suite

- Total tests: 54
- Passed: 54
- Failed: 0
- Includes 15 full-table integration tests
