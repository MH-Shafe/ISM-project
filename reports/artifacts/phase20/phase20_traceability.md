# Phase 20: Traceability and Provenance

## Purpose

Every output from the Phase 20 decision engine is fully traceable to frozen
artifacts. This document establishes the provenance chain.

## Artifact Chain

### Model Source
- **File**: `reports/artifacts/phase7_freeze_lgbm-graph-v1.json`
- **Model**: lgbm-graph-v1 (LightGBM 4.6.0)
- **Seed**: 42
- **Best iteration**: 186
- **Features**: 12 (8 behavioral + 4 graph)
- **Freeze date**: Phase 7
- **Freeze record**: FROZEN_CONFIG in `src/experiments/phase7.py`

### Threshold Source
- **File**: `reports/artifacts/phase9_freeze.json`
- **Policy**: frozen_max_f1
- **Threshold**: 0.9186015432508062
- **Selection rule**: Screens S1 (CI width), S2 (window alert-rate ratio), S3 (window recall > 0)
- **Selection evidence**: CALIBRATION-only, best eligible F1 with fewer-alerts tie-break

### Conformal Layer Source
- **Fit file**: `reports/artifacts/phase10_fit.json`
- **Calibration file**: `reports/artifacts/phase10_calibration.json`
- **Decision file**: `reports/artifacts/phase10_decision.json`
- **Method**: Mondrian split-conformal
- **Alpha**: 0.05
- **t0**: 0.4634739481800199
- **t1**: 0.0046536002164601275
- **Decision**: ACCEPT (D1-D5 all pass)

### Explainability Source
- **File**: `reports/artifacts/phase11_experiment.json`
- **Method**: LightGBM native pred_contrib (TreeSHAP)
- **Output**: margin-space contributions, deterministic and bit-reproducible

### Phase 19 Verdict
- **File**: `phase19_v1_1_1/kaggle_confirm_results/phase19_v1_1_2_confirm_result.json`
- **Verdict**: FAIL (Adaptive Risk rejected for production)
- **Confirm opened**: true
- **Confirm completed**: true
- **Rerun allowed**: NO

## Hash Verification

| Artifact | MD5 Hash |
|----------|----------|
| phase19.py (source) | 6cd6f089e1f7277f3a3b79c52f5ba37f |
| confirm result | fb835ce71b0596e9988e2e31bd03319d |
| confirm report | 75a40c1e0802aab0970d54e5691f841e |
| confirm manifest | 0c8bc8df3d125288ce5510a3366af5f6 |
| threshold finalization | 985b11b4cdc3f778e536e9021e93a131 |
| recovery runner | 3ac43b1249b4db933d89646cea62f7fb |
| archive ZIP | 692d5c77f57703ee597b9f9aa7868597 |
| archive TAR.GZ | 9ab29208dd5d0d817278c8c3dc615675 |

## Decision Engine Provenance Record

The `build_provenance_record()` function in `src/experiments/phase20.py`
returns a complete provenance record including:
- All frozen model parameters
- All frozen thresholds
- All source artifact paths
- Forbidden operation flags
- Deterministic guarantee

## Output Provenance

Every row in the dashboard-ready output includes:
- `ml_risk`: from frozen model (traceable to phase7 freeze record)
- `risk_level`: from deterministic classification (traceable to frozen thresholds)
- `confidence`: from frozen conformal fit (traceable to phase10 fit)
- `conformal_set`: from frozen conformal fit
- `alert_flag`: deterministic from risk_level
- `recommended_action`: deterministic from risk_level + trust_flags
- `trust_flags`: from frozen graph features
- `top_reasons`: from frozen SHAP/pred_contrib or feature values
- `is_malicious`: ground truth (evaluation only, never used in production)
