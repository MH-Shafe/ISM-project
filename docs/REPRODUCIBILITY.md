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

- Python 3.11+
- LightGBM 4.6.0
- pandas 2.x
- numpy 1.x
- scikit-learn 1.x
- pyarrow (for parquet I/O)

If your development environment differs from the validated environment, document the differences here and verify that results are still reproducible.
