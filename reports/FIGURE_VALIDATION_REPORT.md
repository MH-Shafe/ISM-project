# Figure Validation Report

**Date**: 2026-09-14
**Purpose**: Document the source artifacts and verification status for each figure
in `reports/figures/`.

All figures are generated from verified project artifacts (OBSERVED). No raw
dataset is accessed; no model inference is performed. Data values are taken
directly from JSON records in `reports/artifacts/`.

---

## Figure Inventory

| # | File | Source Artifacts | Data Values Verified |
|---|---|---|---|
| 1 | `fig1_system_architecture.png` | Master report pipeline description (§4) | Pipeline stages; no numerical data |
| 2 | `fig2_chronological_split.png` | `phase7_freeze_lgbm-graph-v1.json` §split | Date ranges, row counts, positive counts |
| 3 | `fig3_class_distribution.png` | `phase7_freeze_lgbm-graph-v1.json` §split | Row counts, positive counts, prevalence |
| 4 | `fig4_ablation_comparison.png` | Master report §31 Table 1 (from Phase 6/7 records) | AUC-ROC, AUC-PR, F1, alert counts |
| 5 | `fig5_feature_importance.png` | `phase11_global_importance.json` §calibration.importance | Mean \|SHAP\| per feature |
| 6 | `fig6_decision_distribution.png` | `phase20/final_user_day_decisions_summary.json` | Risk levels, confidence counts |
| 7 | `fig7_runtime_modelsize.png` | `phase6_cost.json`, `phase7_freeze_lgbm-graph-v1.json`, `phase20/final_user_day_decisions_summary.json` | Train/predict times, model sizes |

---

## Source Artifact Verification

### Figure 1 — System Architecture
- **Source**: Pipeline description in master report §4 and Section 8.
- **Nature**: Schematic diagram; no measured data.
- **Status**: Conceptual; verified against the documented pipeline.

### Figure 2 — Chronological Split
- **Source**: `reports/artifacts/phase7_freeze_lgbm-graph-v1.json` lines 23–41.
- **Values**: TRAIN 2010-01-02..2011-01-31 (395,000 rows, 1,539 pos);
  CAL 2011-02-01..2011-03-31 (59,000 rows, 323 pos);
  TEST 2011-04-01..2011-05-17 (47,000 rows, 30 pos).
- **Status**: Exact match to freeze record.

### Figure 3 — Class Distribution
- **Source**: Same as Figure 2.
- **Values**: Same row/positive counts; prevalence 0.38% overall.
- **Status**: Exact match.

### Figure 4 — Ablation Comparison
- **Source**: Master report §31 Table 1, drawn from Phase 6/7 records.
- **Values**: Arm A AUC-ROC 0.93534, Arm B 0.78635, Arm C 0.93916; AUC-PR
  0.15304/0.01431/0.26778; F1 0.346/0.054/0.354; alerts 22/417/49.
- **Status**: Matches the published ablation table. No RF/XGBoost/CatBoost/LR
  data included (these were never evaluated).

### Figure 5 — Feature Importance
- **Source**: `reports/artifacts/phase11_global_importance.json` lines 9–141.
- **Values**: Mean |SHAP| per feature (calibration set). Ranked:
  http_activity_count (1.421), device_consistency_score (0.406),
  usb_connection_count (0.222), file_access_count (0.221), etc.
- **Status**: Exact match to artifact.

### Figure 6 — Decision Distribution
- **Source**: `reports/artifacts/phase20/final_user_day_decisions_summary.json`
  lines 8–16.
- **Values**: ALERT=3785, BORDERLINE=1484, MONITOR=178204, NON-ALERT=317527;
  high-confidence=61017, ambiguous=439983.
- **Status**: Exact match to Phase 20 summary.

### Figure 7 — Runtime and Model Size
- **Source**: `phase6_cost.json` (Arm A/B/C), `phase7_freeze_lgbm-graph-v1.json`
  (frozen model), `phase20/final_user_day_decisions_summary.json` (full engine).
- **Values**: Train times 4.2/1.38/4.76/4.64 s; predict times 1.08/0.048/1.21/1.29 s;
  model sizes 559/13/651/651 KB; full engine 180.3 s.
- **Status**: Exact match to cost artifacts.

---

## Generation Script

- **Script**: `reports/figures/generate_figures.py`
- **Dependencies**: matplotlib, numpy (both in `requirements.txt`)
- **Reproducibility**: `python reports/figures/generate_figures.py` regenerates
  all 7 figures from the same artifact data.
- **No raw data access**: the script reads only JSON summary records, never
  parquet predictions or the CERT dataset.

---

## Constraints

- All numerical claims in figures are classified **OBSERVED** (directly read
  from verified artifacts).
- No figure contains fabricated, interpolated, or projected data.
- The ablation chart (Figure 4) includes only LightGBM arms A/B/C plus
  adaptive risk — no other algorithms were evaluated.
- CPU utilization % is NOT shown in any figure (NOT MEASURED).
