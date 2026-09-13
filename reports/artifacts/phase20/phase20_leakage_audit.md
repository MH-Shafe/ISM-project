# Phase 20: Leakage Audit

## Purpose

Verify that the Phase 20 decision engine introduces no data leakage.
All components are frozen from previous phases; no new information is used.

## Audit Checklist

### 1. Model Training
- [x] Model was trained on TRAIN only (Phase 7)
- [x] Early stopping used CALIBRATION only (Phase 7)
- [x] No TEST data was used in training
- [x] No labels were used as features
- [x] No future information was used

### 2. Threshold Selection
- [x] Alert threshold selected on CALIBRATION only (Phase 9)
- [x] Screens S1-S3 used CALIBRATION only (Phase 9)
- [x] TEST evaluated exactly once after freeze (Phase 9)

### 3. Conformal Layer
- [x] Fit on CALIBRATION only (Phase 10)
- [x] No TEST labels or scores used in fit
- [x] D1-D5 decision rule applied after fit (Phase 10)

### 4. Explainability
- [x] SHAP/pred_contrib from frozen model only (Phase 11)
- [x] No label information in explanations (Phase 11)
- [x] Feature values are from input data (Phase 11)

### 5. Graph Diagnostics
- [x] Graph features computed from frozen graph layer (Phase 5-7)
- [x] No future information in graph features (Phase 5)
- [x] Department feature was REJECTED (Phase 7)

### 6. Decision Engine
- [x] No new features generated
- [x] No thresholds tuned or adapted
- [x] No TEST data used in production path
- [x] No Adaptive Risk fusion
- [x] No label-dependent operational rules
- [x] Deterministic and reproducible

### 7. Output Schema
- [x] is_malicious column is evaluation-only
- [x] No identifiers used as predictive features
- [x] No dates used as predictive features
- [x] All features are from frozen registry

## Verdict

**PASS**: No leakage detected. All components are frozen from previous phases
and follow the project's leakage prevention protocol.

## Known Limitations

1. **Conformal sample size**: n1=323 (positive class) is small. The conformal
   guarantee is finite-sample but the positive-class coverage may be less
   precise than with larger samples.

2. **Temporal distribution shift**: The conformal layer assumes exchangeability
   within CALIBRATION. Temporal drift is monitored by Phase 13 but not
   corrected in the production path.

3. **Feature freeze**: The 12-feature set was frozen in Phase 7. No new
   features can be added without unfreezing the model.
