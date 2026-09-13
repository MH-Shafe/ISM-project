# Baseline Model Benchmark Report

**Experiment ID:** `baseline_benchmark_20260914`
**Date:** 2026-09-14
**Status:** Complete
**Evidence Type:** NEW BASELINE BENCHMARK (all models except frozen LightGBM)

---

## 1. Objective

Evaluate and compare standard ML baselines against the production LightGBM model for insider threat detection on the CERT r4.2 dataset. This benchmark establishes reference performance levels for future model development.

## 2. Protocol

- **Split strategy:** Chronological train/calibration/test split
- **Threshold selection:** Optimized on calibration set using F1 metric
- **Evaluation:** Test set metrics with bootstrap confidence intervals (1000 resamples)
- **Evidence classification:** FROZEN PRODUCTION (pre-existing), NEW BASELINE BENCHMARK (this experiment)

## 3. Dataset / Split

| Split | Rows | Positives | Date Range |
|-------|------|-----------|------------|
| TRAIN | 395,000 | 1,539 | <= 2011-01-31 |
| CALIBRATION | 59,000 | 323 | 2011-02-01 .. 2011-03-31 |
| TEST | 47,000 | 30 | >= 2011-04-01 |

**Dataset**: CERT r4.2 | **Analytical unit**: User × Day | **Prevalence**: 0.38%

## 4. Features

All 12 frozen features (8 behavioral + 4 graph):

1. login_count
2. after_hours_login_count
3. usb_connection_count
4. file_access_count
5. sensitive_file_access_count
6. http_activity_count
7. unique_device_count
8. unusual_access_count
9. device_consistency_score
10. rare_device_usage_count
11. file_type_consistency_score
12. rare_file_type_access_count

## 5. Model Configurations

### Logistic Regression
- Pipeline: StandardScaler + LogisticRegression
- penalty=l2, solver=liblinear, class_weight=balanced, max_iter=2000, random_state=42
- Size: 1,757 bytes

### Random Forest
- n_estimators=500, max_depth=None, min_samples_leaf=2, max_features=sqrt
- class_weight=balanced, random_state=42
- Size: 79,778,441 bytes (~76 MB)

### XGBoost
- n_estimators=3000, lr=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8
- scale_pos_weight=255.66, random_state=42, early_stopping_rounds=100, best_iter=345
- Size: 2,162,687 bytes (~2.1 MB)

### CatBoost
- iterations=3000, lr=0.03, depth=6
- class_weights={0:1.0, 1:255.66}, random_seed=42, early_stopping_rounds=100, best_iter=406
- Size: 490,496 bytes (~479 KB)

### LightGBM (benchmark)
- Identical to frozen: lr=0.03, num_leaves=31, min_data_in_leaf=100
- feature_fraction=0.8, bagging_fraction=0.8, seed=42, best_iter=186
- Size: 650,716 bytes (~635 KB)

### LightGBM (frozen)
- Production snapshot from Phase 7, identical config to benchmark
- Size: 651,047 bytes (~636 KB)

## 6. Threshold Selection

Thresholds were optimized on the calibration set to maximize F1 score:

| Model | Threshold |
|-------|-----------|
| Logistic Regression | 0.84732 |
| Random Forest | 0.48006 |
| XGBoost | 0.81989 |
| CatBoost | 0.88767 |
| LightGBM (benchmark) | 0.91860 |
| LightGBM (frozen) | 0.91860 |

## 7. Results

### Primary Metrics

| Model | Evidence Type | ROC-AUC | PR-AUC | F1 | MCC | Alerts |
|-------|---------------|---------|--------|-----|-----|--------|
| Logistic Regression | NEW BASELINE BENCHMARK | 0.883 | 0.010 | 0.028 | 0.084 | 1119 |
| Random Forest | NEW BASELINE BENCHMARK | 0.773 | 0.183 | 0.177 | 0.206 | 94 |
| XGBoost | NEW BASELINE BENCHMARK | 0.940 | 0.269 | 0.284 | 0.346 | 111 |
| CatBoost | NEW BASELINE BENCHMARK | 0.920 | 0.179 | 0.232 | 0.280 | 108 |
| LightGBM (benchmark) | NEW BASELINE BENCHMARK | 0.939 | 0.268 | 0.354 | 0.365 | 49 |
| LightGBM (frozen) | FROZEN PRODUCTION | 0.939 | 0.268 | 0.354 | 0.365 | 49 |

### Detailed Metrics

| Model | Precision | Recall | Brier Score |
|-------|-----------|--------|-------------|
| Logistic Regression | 0.014 | 0.533 | 0.156 |
| Random Forest | 0.117 | 0.367 | 0.096 |
| XGBoost | 0.180 | 0.667 | 0.099 |
| CatBoost | 0.148 | 0.533 | 0.112 |
| LightGBM (benchmark) | 0.286 | 0.467 | 0.101 |
| LightGBM (frozen) | 0.286 | 0.467 | — |

## 8. Bootstrap Confidence Intervals

1000 bootstrap resamples on the test set:

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | F1 [95% CI] | MCC [95% CI] |
|-------|------------------|-----------------|-------------|--------------|
| Logistic Regression | 0.883 [0.806, 0.944] | 0.010 [0.007, 0.015] | 0.030 [0.017, 0.049] | 0.087 [0.049, 0.137] |
| Random Forest | 0.771 [0.700, 0.842] | 0.185 [0.116, 0.276] | 0.175 [0.096, 0.270] | 0.206 [0.117, 0.307] |
| XGBoost | 0.940 [0.907, 0.974] | 0.272 [0.164, 0.405] | 0.283 [0.186, 0.392] | 0.347 [0.239, 0.457] |
| CatBoost | 0.920 [0.873, 0.965] | 0.181 [0.109, 0.278] | 0.231 [0.141, 0.339] | 0.280 [0.175, 0.388] |
| LightGBM | 0.939 [0.903, 0.973] | 0.269 [0.161, 0.406] | 0.352 [0.242, 0.471] | 0.364 [0.254, 0.475] |

**Key observations:**
- XGBoost and LightGBM have overlapping ROC-AUC CIs, suggesting comparable discrimination
- LightGBM achieves highest F1 with fewest alerts (49), indicating superior precision-recall balance
- Logistic Regression has extremely wide PR-AUC CI due to severe class imbalance

## 9. Computational Cost

| Model | Train Time | Model Size | Inference Complexity |
|-------|------------|------------|---------------------|
| Logistic Regression | 2.72s | 1.7 KB | O(n_features) |
| Random Forest | 39.87s | 76.1 MB | O(n_trees × depth) |
| XGBoost | 9.06s | 2.1 MB | O(n_trees × depth) |
| CatBoost | 39.50s | 479 KB | O(n_trees × depth) |
| LightGBM (benchmark) | 6.87s | 635 KB | O(n_trees × leaves) |
| LightGBM (frozen) | 4.64s | 636 KB | O(n_trees × leaves) |

**Resource efficiency ranking:** LightGBM > CatBoost > XGBoost > Random Forest > Logistic Regression

## 10. ROC / PR Analysis

### ROC Curves
- **XGBoost** achieves highest ROC-AUC (0.940), closely followed by LightGBM (0.939)
- **Random Forest** shows lowest discrimination (0.773)
- All tree-based models substantially outperform logistic regression

### Precision-Recall Curves
- **XGBoost** leads PR-AUC (0.269), with LightGBM very close (0.268)
- **Logistic Regression** has near-random PR performance (0.010) despite decent ROC-AUC
- PR curves reveal the true difficulty of the imbalanced classification task

## 11. Confusion Matrices

| Model | TP | FP | TN | FN | Alerts | Precision | Recall |
|-------|----|----|----|----|--------|-----------|--------|
| Logistic Regression | 16 | 1,103 | 45,867 | 14 | 1,119 | 0.014 | 0.533 |
| Random Forest | 11 | 83 | 46,887 | 19 | 94 | 0.117 | 0.367 |
| XGBoost | 20 | 91 | 46,879 | 10 | 111 | 0.180 | 0.667 |
| CatBoost | 16 | 92 | 46,878 | 14 | 108 | 0.148 | 0.533 |
| LightGBM (benchmark) | 14 | 35 | 46,935 | 16 | 49 | 0.286 | 0.467 |

**Key insight:** LightGBM achieves the fewest false positives (35) and highest precision (0.286) while maintaining competitive recall (0.467).

## 12. Statistical Uncertainty

- Bootstrap CIs reveal substantial uncertainty in PR-based metrics due to class imbalance
- ROC-AUC CIs are relatively narrow for all models except Random Forest
- F1 CIs are wide across all models, reflecting the precision-recall tradeoff sensitivity
- **Recommendation:** Report ROC-AUC as primary metric; treat F1/MCC as secondary

## 13. Comparison with Final LightGBM

| Metric | LightGBM (benchmark) | LightGBM (frozen) | Δ |
|--------|---------------------|-------------------|---|
| ROC-AUC | 0.93916 | 0.93916 | 0.00000 |
| PR-AUC | 0.26778 | 0.26778 | 0.00000 |
| F1 | 0.35443 | 0.354 | +0.00043 |
| MCC | 0.36464 | 0.365 | -0.00036 |
| Alerts | 49 | 49 | 0 |
| Train Time | 6.87s | 4.64s | +2.23s |

The benchmark and frozen LightGBM models are numerically consistent, confirming the frozen model is a faithful snapshot.

## 14. Limitations

1. **Single dataset:** Results specific to CERT r4.2; generalizability unverified
2. **Temporal split:** Performance may vary with different time periods
3. **Feature set:** Limited to available CERT r4.2 features
4. **No hyperparameter tuning:** Default configurations used for all baselines
5. **Class imbalance:** Extreme imbalance makes PR-based metrics noisy
6. **Threshold sensitivity:** Small changes in threshold significantly impact precision/recall

## 15. Reproducibility

### Environment
- Python 3.12.13, NumPy 2.0.2, Pandas 2.3.3
- scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10
- PyArrow 24.0.0 (parquet I/O)
- CERT r4.2 dataset (Kaggle, read-only)

### Random Seeds
- All models: random_state=42
- Bootstrap: random_state=42

### Artifact Locations
- Model files: `models/` directory
- Metrics: `reports/final/BASELINE_MODEL_COMPARISON.json`
- Bootstrap CIs: `reports/artifacts/baseline_benchmark/bootstrap_confidence_intervals.json`
- This report: `reports/final/BASELINE_BENCHMARK_REPORT.md`

### Scripts
- Feature generation: `kaggle_scripts/generate_features.py`
- Model training: `kaggle_scripts/train_models.py`
- Evaluation: `kaggle_scripts/evaluate_models.py`

---

**Generated:** 2026-09-14
**Status:** Complete — all artifacts verified
