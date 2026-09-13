# Reproducibility Guide

## Dataset

- **Dataset:** CERT Insider Threat Dataset r4.2
- **Source:** SEI CERT (https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
- **Primary analytical unit:** user x day

## Frozen System Constants

| Parameter | Value |
|---|---|
| Production model | lgbm-graph-v1 |
| LightGBM version | 4.6.0 |
| Feature count | 12 (8 behavioral + 4 graph) |
| Seed | 42 |
| Best iteration | 186 |
| Alert threshold | 0.9186015432508062 |
| Adaptive Risk | REJECTED for production |

## Split Protocol

| Split | Period | Rows | Positives |
|---|---|---|---|
| TRAIN | through 2011-01-31 | 395,000 | 1,539 |
| CALIBRATION | 2011-02-01 to 2011-03-31 | 59,000 | 323 |
| TEST | from 2011-04-01 | 47,000 | 30 |

## Conformal Prediction

- Method: Mondrian split conformal
- Alpha: 0.05 (95% coverage target)
- t0 (negative): 0.4634739481800199
- t1 (positive): 0.0046536002164601275
- n1 (CAL positives): 323
- n0 (CAL negatives): 58,677

## Critical Artifact Hashes

| Artifact | MD5 |
|---|---|
| Final parquet (501K x 21) | 6476f791d1cc9f21327a94e1373f9e09 |
| Frozen model (lgbm-graph-v1) | See phase7_freeze_lgbm-graph-v1.json |

## Production Architecture

```
CERT r4.2
→ validation/preprocessing
→ leakage-safe user-day aggregation
→ behavioral features (8)
→ graph/trust features (4)
→ frozen LightGBM (lgbm-graph-v1)
→ frozen decision policy (4 risk levels)
→ conformal uncertainty (prediction sets)
→ SHAP explanations
→ trust/context diagnostics
→ deterministic Phase 20 decision engine
→ analyst-facing output (21 columns)
```

## Known Software Versions

### Frozen Training Environment (Kaggle)
- Python 3.11
- LightGBM 4.6.0
- pandas 2.x
- numpy 1.x
- scikit-learn 1.x

### Local Verification Environment (Windows PC)
- Python 3.11.9
- LightGBM 4.7.0 (reproduces frozen scores to max abs diff 5.6e-17)
- pandas 3.0.5
- numpy 2.4.6
- scikit-learn 1.9.0
- pyarrow 25.0.1
- pytest 9.1.1

See `requirements-local-lock.txt` for exact Windows versions. Use `requirements.txt` for portable installation.

If your development environment differs from the validated environment, document the differences here and verify that results are still reproducible.

## Test Tiers

The test suite is designed so that `pytest` produces **0 failures** from a clean clone. Tests are organized into tiers:

### Tier 1: Core (always pass)
~448 tests using synthetic fixtures. No dataset or external artifacts required. These cover feature computation, model loading, evaluation logic, configuration, and all algorithmic paths.

### Tier 2: Dataset-dependent (skipped without CERT r4.2)
~16 tests requiring the raw CERT r4.2 dataset in `data/raw/cert_r4.2/`. These validate data ingestion, label generation, and end-to-end pipeline behavior on real data.

### Tier 3: Kaggle-artifact-dependent (skipped without frozen intermediates)
~16 tests requiring frozen intermediate artifacts produced during Kaggle execution:

| Test group | Required artifact | Location |
|---|---|---|
| Frozen input MD5 checks | `phase6_merged_features.parquet`, `phase14_split.json`, `phase15_cal_scores.parquet`, `phase14_test_predictions.parquet`, `phase14_user_diagnostics.json`, `phase18_role_department.parquet` | `artifacts/` |
| Model MD5 check | `phase14_model.txt` | `artifacts/` |
| Gate reproduction | `phase14_test_predictions.parquet` | `artifacts/` |
| Role table schema | `phase18_role_department.parquet` | `artifacts/` |
| Diagnostic user checks | `phase14_split.json`, `phase14_user_diagnostics.json` | `artifacts/` |
| Engine determinism | `run_phase18.py` | `kaggle_scripts/` |
| Phase 19 allocation | `phase19_user_allocation.json` | `reports/artifacts/` |

These artifacts are produced by the full Kaggle pipeline and are intentionally excluded from the Git repository (too large, environment-specific).

### Tier 4: Optional-dependency (skipped without torch/PyG)
~16 tests for Families H1, H2, I, J. Skipped when PyTorch or PyG is not installed.

### Running the full validation suite
To run all tests including artifact-dependent ones:
1. Execute the full pipeline on Kaggle (all 24 notebooks pass).
2. The frozen artifacts will be in `artifacts/` and `reports/artifacts/`.
3. The Kaggle engine scripts will be in `kaggle_scripts/`.
4. Run `pytest` locally with all artifacts present — all 562 tests will execute.
