# Phase 11 Gap Analysis — Explainability Layer

Date: 2026-08-16
Scope: current `phase11.py` / `run_phase11.py` / `test_phase11.py` vs the
authorized Phase 11 explainability specification for the frozen
`lgbm-graph-v1` detector. Analysis is done BEFORE implementation; the diff is
kept minimal and the frozen model / Phase 9 threshold / Phase 10 conformal
layer are never modified.

## 1. Satisfied by the current implementation (verified by reading the code)

| Spec item | Evidence |
|---|---|
| Primary method = LightGBM native `pred_contrib` (path-dependent TreeSHAP) in margin space | `phase11.margin_contributions`; rationale frozen in `phase11_config.json` |
| `shap` used as validation cross-check only, never a runtime dependency | `run_phase11.py` calibration stage, `SHAP_SAMPLE_N = 1000` |
| Additive identity `margin == bias + sum(contrib)` | `check_reconstruction`, tol `1e-6`, hard gate on TRAIN/CAL/TEST |
| Determinism + row-order invariance | calibration-stage stability checks, `phase11_stability.json` |
| Global importance + gain-vs-explanation comparison (Kendall tau) | `global_importance`, `compare_gain_ranking` |
| Alert contribution patterns | `alert_patterns` (CALIBRATION alert rows, label-free) |
| Label-free deterministic selection policy (alerts / monitor / borderline / decile-stratified non-alert) | `select_explanation_rows`, `_decile_stratified` |
| Deterministic reason templates, no causal verbs | `top_reasons`, `_sentence`, `CAUSAL_VERBS` check |
| Conformal overlay at alpha 0.05 from frozen Phase 10 fit | `load_conformal_fit`, `t0_005` gates |
| Score gates (TEST bit-equal to Phase 7; CAL/TEST alert counts equal to Phase 9 record) | `verify_score_gates` |
| TEST explained exactly once; rerun gate bit-reproduces the record | test-stage `phase11_experiment.json` comparison |
| 12-feature frozen registry enforcement | `_check_features`, `test_explanation_rejects_department_feature` |

## 2. Gaps vs the specification (to implement)

| # | Gap | Current state | Action |
|---|---|---|---|
| G1 | `contribution_detail` lacks the feature `definition` from the registry | detail = feature/value/contribution/abs/rank/direction/n_features | add `definition` (from `config.BEHAVIORAL_FEATURES` / `config.GRAPH_FEATURES`); update hand-computed test |
| G2 | Missing `alert_contribution_stats` aggregate function | only `alert_patterns` exists | new function: per-feature mean/median abs, signed mean, frac positive/negative/zero, mean rank, top-1/2/3 frequency over alert rows; output into the patterns artifact |
| G3 | Missing explicit counterfactual feasibility analysis | no counterfactual step | new `counterfactual_feasibility()` -> decision `NOT_SUPPORTED` (documented reasons + future criteria); artifact `phase11_counterfactual.json` |
| G4 | Missing local perturbation stability | only row-order/determinism checks | new `perturbation_stability()`: per-feature deterministic perturbations (counts +1, ratios +0.1), sample = explained set (CALIBRATION); report mean/max abs margin delta, top-1/top-3 rank flip rate, decision flip rate; written into `phase11_stability.json` |
| G5 | Explanations lack a `risk_level` column | only `explanation_set` tag added in the test stage | add `risk_level` (ALERT / BORDERLINE / MONITOR / NON-ALERT) to `phase11_explanations.parquet` via a small mapped helper |
| G6 | Cost artifact measures time only | `phase11_cost.json` = seconds | add peak RSS (Linux `ru_maxrss`) and per-artifact byte sizes in both stages |
| G7 | Missing artifacts: model integrity, reproduction, counterfactual, alert-only explanations | not present | new `phase11_model_integrity.json` (calibration), `phase11_reproduction.json` (test), `phase11_counterfactual.json` (calibration), `phase11_alert_explanations.parquet` (test, alert rows only) |

## 3. Explicit non-changes (frozen contract)

- The frozen model file, `FROZEN_THRESHOLD`, `EXPLAIN_ALPHA`, selection-policy
  constants, reason templates, and conformal fit are NOT changed.
- Perturbation stability evaluates the frozen model on perturbed inputs for
  internal sensitivity analysis only; the frozen system's reported scores and
  decisions for the original rows are never altered or re-reported.
- Counterfactual explanations would report altered-risk outputs for edited
  rows and are therefore NOT supported under the frozen contract (see G3).
- No new runtime dependencies; `shap` stays validation-only.

## 4. Test additions (Step 3 + 4)

- Update `test_contribution_detail_hand_computed` for the `definition` key.
- New unit tests: `alert_contribution_stats` hand-computed; perturbation
  stability determinism + bounds (tiny real LightGBM); counterfactual
  feasibility decision constant; `risk_level` mapping; reloadability of the
  four new artifacts (skip-guarded until a full run exists).