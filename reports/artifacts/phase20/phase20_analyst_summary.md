# Phase 20: Analyst Summary — Dashboard-Ready Output

## Overview

Phase 20 integrates all frozen scientific components into a single
deterministic, reproducible, leakage-safe decision engine that produces
analyst-facing output for each user-day.

## Production System Components

| Component | Source | Status |
|-----------|--------|--------|
| LightGBM model | lgbm-graph-v1 (Phase 7) | FROZEN |
| 12 features | 8 behavioral + 4 graph (Phase 7) | FROZEN |
| Alert threshold | 0.9186 (Phase 9 frozen_max_f1) | FROZEN |
| Conformal layer | Mondrian split-conformal (Phase 10) | FROZEN, ACCEPT |
| Explainability | TreeSHAP/pred_contrib (Phase 11) | FROZEN |
| Graph diagnostics | 4 graph features (Phase 5-7) | FROZEN |
| Adaptive Risk | Candidate (Phase 18-19) | REJECTED FOR PRODUCTION |

## Decision Pipeline

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

## Output Schema (16 columns)

| Column | Type | Description |
|--------|------|-------------|
| user | string | Employee user ID |
| day | string | Calendar date |
| ml_risk | float | Frozen LightGBM risk score (0-1) |
| risk_level | string | ALERT / BORDERLINE / MONITOR / NON-ALERT |
| confidence | float | Conformal confidence (0.90 or 0.95) |
| conformal_set | string | {1}, {0}, {0,1}, or {} |
| p1 | float | Conformal p-value for class 1 |
| p0 | float | Conformal p-value for class 0 |
| alert_flag | boolean | True if ALERT or HUMAN_REVIEW |
| recommended_action | string | Analyst action |
| urgency | string | immediate / within_24h / weekly / none |
| rationale | string | Human-readable explanation |
| trust_flags | array | Active graph/trust diagnostic flags |
| n_trust_flags | integer | Count of active trust flags |
| graph_features | object | Raw graph feature values |
| top_reasons | array | Top-N explanation reasons |

## Risk Level Definitions

- **ALERT** (score >= 0.9186): Confident positive. Escalate to incident response.
  - Grounded in: Phase 9 frozen_max_f1 threshold (CALIBRATION-selected)
  - Confidence: 0.95 (conformal singleton set {1})

- **BORDERLINE** (0.8686 <= score < 0.9186): Borderline alert. Queue for analyst review within 24h.
  - Grounded in: Phase 11 BORDERLINE_WIDTH = 0.05 (within 0.05 of alert threshold)
  - Confidence: 0.90 (conformal ambiguous set {0,1})

- **MONITOR** (0.4635 <= score < 0.8686): Monitor zone. Add to watchlist, review weekly.
  - Grounded in: Phase 10 conformal t0 threshold (alpha=0.05)
  - Confidence: 0.90 (conformal ambiguous set {0,1})

- **NON-ALERT** (score < 0.4635): Confident negative. No action required.
  - Grounded in: Phase 10 conformal t0 threshold (alpha=0.05)
  - Confidence: 0.95 (conformal singleton set {0})

## Forbidden Operations

The following operations are explicitly forbidden in the production path:
- Model retraining or refitting
- New feature generation
- Threshold tuning or adaptation
- Adaptive Risk fusion
- Label-dependent operational rules
- Post-hoc threshold selection
- New ML model creation
- TEST data usage in production

## Provenance

Every output row is traceable to frozen artifacts:
- Model: `reports/artifacts/phase7_freeze_lgbm-graph-v1.json`
- Threshold: `reports/artifacts/phase9_freeze.json`
- Conformal: `reports/artifacts/phase10_calibration.json`, `phase10_fit.json`
- Explainability: `reports/artifacts/phase11_experiment.json`
- Phase 19 verdict: `phase19_v1_1_1/kaggle_confirm_results/phase19_v1_1_2_confirm_result.json`
