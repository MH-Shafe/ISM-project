# Artifact Registry

This document catalogs all significant project artifacts.

## Included in Repository

### Frozen Model

| Artifact | Path | Size | Purpose |
|---|---|---|---|
| lgbm-graph-v1.txt | `artifacts/model/` | ~636 KB | Serialized LightGBM model (best_iteration=186, seed=42) |

### Demo Decision Table

| Artifact | Path | Size | Purpose |
|---|---|---|---|
| final_user_day_decisions.parquet | `demo_artifacts/` | 2.76 MB | 501,000 x 21 Phase 20 decision table |

**Columns:** user_id, date, ml_risk_score, risk_level, alert_flag, confidence_status, conformal_prediction_set, conformal_p1, conformal_p0, confidence, top_reason_1, top_reason_2, top_reason_3, trust_diagnostic_summary, n_trust_flags, recommended_action, urgency, rationale, model_version, policy_version, explanation_version

### Frozen Metadata

| Artifact | Path | Purpose |
|---|---|---|
| phase7_freeze_lgbm-graph-v1.json | `artifacts/frozen/` | Model freeze record (thresholds, metrics, splits) |
| phase10_experiment.json | `artifacts/frozen/` | Conformal prediction configuration |
| phase10_decision.json | `artifacts/frozen/` | Conformal thresholds |
| phase19_preconfirm_freeze.json | `artifacts/frozen/` | Phase 19 pre-confirm freeze |
| phase19_v1_1_1_final_preconfirm_freeze.json | `artifacts/frozen/` | Phase 19 v1.1.1 final freeze |

### Summaries

| Artifact | Path | Purpose |
|---|---|---|
| phase11_global_importance.json | `artifacts/summaries/` | SHAP global feature importance |

## Not in Repository (Regeneratable)

These artifacts can be regenerated from the pipeline but are excluded from Git due to size:

| Artifact | Size | How to Regenerate |
|---|---|---|
| final_user_day_decisions.csv | 158 MB | Run `src/experiments/phase20.py` with full CERT data |
| phase11_explanations.parquet | 4.4 MB | Run `src/experiments/phase11.py` |
| graph_features.parquet | 543 KB | Run `src/graph/features.py` |
| phase10_fit.json | 1.4 MB | Run `src/experiments/phase10.py` |
| Various phase prediction parquets | ~10 MB total | Run respective phase scripts |

## Not in Repository (External)

| Artifact | Purpose | How to Obtain |
|---|---|---|
| CERT r4.2 raw data | Training/evaluation data | Download from SEI CERT |
