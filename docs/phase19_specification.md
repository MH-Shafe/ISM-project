# Phase 19 Specification: Comprehensive Adaptive Risk Fusion Benchmark

**Project**: ISM Project Shafe — CERT Insider Threat r4.2
**Date**: 2026-08-21 (v1.1 clarification 2026-09-07)
**Status**: FROZEN — Design freeze complete, v1.1 clarification applied

---

## 1. Source Artifacts Verified

| Artifact | Path | MD5 | Notes |
|----------|------|-----|-------|
| Phase 6 merged features | `reports/artifacts/phase6_merged_features.parquet` | `9a3b188573bb953416981dfea3379def` | 501,000 rows × 16 cols |
| Phase 14 split | `reports/artifacts/phase14_split.json` | `da013f825246d568bcfdaf19dcf2c7e7` | 784 TRAIN / 108 CAL / 108 TEST users |
| Phase 14 CAL scores | `reports/artifacts/phase15_cal_scores.parquet` | `c955a4ec6ecaf7e9baff80abeb361330` | Frozen LightGBM scores |
| Phase 14 TEST predictions | `reports/artifacts/phase14_test_predictions.parquet` | `71eeb3f1948e518518a53e062d5a213d` | Frozen, never used for selection |
| Phase 14 user diagnostics | `reports/artifacts/phase14_user_diagnostics.json` | `e3b0c44298fc1c149afbf4c8996fb924` | Onset dates, per-user metrics |
| Phase 17 calibration fit | `reports/artifacts/phase17_calibration_fit.json` | — | ECE/Brier reference |
| Phase 17 operating points | `reports/artifacts/phase17_operating_points.json` | — | Threshold reference |
| Phase 18 role/department | `reports/artifacts/phase18_role_department.parquet` | — | 1000 users, role + department |
| Phase 18 role build record | `reports/artifacts/phase18_role_build_record.json` | — | Build provenance |

---

## 2. Phase 14 TRAIN User Counts (Verified)

| Cohort | Total | Malicious | Benign |
|--------|-------|-----------|--------|
| Phase 14 TRAIN | 784 | 40 | 744 |

Verified from `phase14_split.json` blocks and `phase6_merged_features.parquet`.

---

## 3. Phase 19 User Allocation (FROZEN)

**Allocation file**: `reports/artifacts/phase19_user_allocation.json`

### 3.1 Split Rule (Deterministic, Seed=42)

- **Malicious users**: Sorted by `(onset_date, user_id)`, every 4th → CONFIRM
- **Benign users**: Sorted by `user_id`, every 4th → CONFIRM

### 3.2 Exact Allocation

| Cohort | Total | Malicious | Benign |
|--------|-------|-----------|--------|
| PH19_DEV | 588 | 30 | 558 |
| PH19_CONFIRM | 196 | 10 | 186 |

### 3.3 Malicious CONFIRM Users (10)

```
GTD0219, RAR0725, DIB0285, PSF0133, MCF0600,
EHD0584, ABC0174, HJB0742, JRG0207, MAR0955
```

### 3.4 Benign CONFIRM Users (186)

First 10: `AAN0823, ABC0253, ADC0391, AGB0186, AIB0948, AIP0982, AJL0462, AKC0924, ALC0788, AMD0077`

---

## 4. Algorithms to Run

| Family | Arms | Category | Description |
|--------|------|----------|-------------|
| A | A0 | Lightweight | ML-only control (`R_final = P_ML`) |
| B | B | Lightweight | Convex fusion (SLSQP, log-loss, sum=1, w≥0) |
| C | C | Lightweight | Logistic stacking (L2, C∈{0.1,1.0,10}) |
| D | D | Lightweight | Residual logistic (logit backbone, θ≥0, λ∈{0.01,0.1,1.0}) |
| E | E | Lightweight | Interaction residual (pairwise B*T, B*C, T*C) |
| F | F1, F2 | Lightweight | Temporal persistence (ρ∈{0.0,0.25,0.5,0.75,0.90}) |
| G | G | Lightweight | Bayesian evidence fusion (10 quantile bins, Laplace a=1, LR cap [-5,5]) |
| H | H1, H2 | Lightweight/Conditional | Dynamic gating (linear + tiny MLP 4→8→4, BCE, L2, entropy reg) |
| I | I | Research | Temporal Transformer (seq=14, d_model=32, heads=4, 1 layer) |
| J | J | Research | GNN+Temporal (feasibility gate first) |

---

## 5. Equations Implemented

### 5.1 R_ML (Frozen 12-feature LightGBM)

- Features: 8 behavioral (F_B) + 4 graph (F_G)
- F_B: `login_count, after_hours_login_count, usb_connection_count, file_access_count, sensitive_file_access_count, http_activity_count, unique_device_count, unusual_access_count`
- F_G: `device_consistency_score, rare_device_usage_count, file_type_consistency_score, rare_file_type_access_count`
- TRAIN: 5-fold entity-disjoint OOF (round-robin, scale_pos_weight per fold)
- CAL/TEST: Frozen scores from `phase15_cal_scores.parquet` / `phase14_test_predictions.parquet`
- Logit: `logit(P) = log(clip(P, 1e-6, 1-1e-6) / (1 - clip(P, 1e-6, 1-1e-6)))`

### 5.2 R_behavior (Strictly-Past Expanding Deviation)

Per feature per user: expanding median/MAD over strictly-past days
```
z_{u,d} = (x_{u,d} - median(H)) / (MAD(H) + 1e-6)
H = {x_{u,d'} : d' < d, |H| ≥ 14}
R_behavior = mean(phi(clip(z, 3))) over 8 features
```
Cold start (`|H| < 14`): z = 0 → R_behavior = 0.5

### 5.3 R_trust (TRAIN-block ECDF Percentiles)

Per graph feature: ECDF on TRAIN-block rows only
```
p(v) = (rank_mid(v) - 0.5) / N_TRAIN, clamped to [0,1]
R_trust = mean(p) over 4 features
```
Degenerate feature (< 2 distinct TRAIN values): constant 0.5

### 5.4 R_context (Role/Department Peer Deviation)

Block-local, strictly-past peer pools, weekday/weekend strata
- Same role first (min 5 peers), else same department (min 5)
- Per day d: medians/MADs over peer rows with day < d, stratified by day-type of d
- Magnitude-only, clip |z| ≤ 3, phi map
- Fallbacks: neutral 0.5 for missing role/dept, small peer group, insufficient support

### 5.5 Fusion Formulas

| Family | Formula | Constraints |
|--------|---------|-------------|
| A0 | `R = P_ML` | — |
| B | `R = αP + βB + γT + δC` | α,β,γ,δ ≥ 0, sum=1 |
| C | `R = sigmoid(b + θ_M·logit(P) + θ_B·B + θ_T·T + θ_C·C)` | L2 reg C∈{0.1,1,10} |
| D | `R = sigmoid(logit(P) + θ_B·B + θ_T·T + θ_C·C)` | θ≥0, L2 λ∈{0.01,0.1,1} |
| E | `R = sigmoid(logit(P) + θ_B·B + θ_T·T + θ_C·C + θ_BT·B·T + θ_BC·B·C + θ_TC·T·C)` | L2 reg |
| F1 | `R_t = (1-ρ)·R_inst,t + ρ·R_{t-1}` on Family D | ρ∈{0.0,0.25,0.5,0.75,0.9} |
| F2 | Same on Family E | ρ∈{0.0,0.25,0.5,0.75,0.9} |
| G | `R = sigmoid(logit(π) + Σ log(LR_k))` | 10 bins, LR cap [-5,5], a=1 |
| H1 | `w_t = softmax(W·z_t + b)` | BCE + L2 + entropy |
| H2 | `w_t = softmax(MLP_4→8→4(z_t))` | BCE + L2 + entropy |
| I | Tiny Transformer (seq=14, d=32, h=4, 1 layer) | BCE, early stopping |
| J | GraphSAGE (d≤16) → GRU (d≤32) | Feasibility gate |

---

## 6. Hyperparameter Grids (FROZEN)

| Family | Parameters | Grid |
|--------|------------|------|
| B | — | None (deterministic SLSQP) |
| C | C (L2 inverse) | {0.1, 1.0, 10.0} |
| D | λ (L2) | {0.01, 0.1, 1.0} |
| E | λ (L2) | {0.01, 0.1, 1.0} |
| F1/F2 | ρ | {0.00, 0.25, 0.50, 0.75, 0.90} |
| G | bins, a, LR_cap | {10}, {1}, {[-5,5]} |
| H1 | L2, entropy_reg | {0.01, 0.1, 1.0} × {0.0, 0.01} |
| H2 | L2, entropy_reg, hidden | {0.01, 0.1} × {0.0, 0.01} × {8} |
| I | d_model, heads, layers, ff | {32} × {4} × {1} × {64} |
| J | graph_dim, temp_dim, layers | {16} × {32} × {1,2} |

**Selection**: Inner CV (user-disjoint, 3-fold) on OUTER_TRAIN only. Seed=42.

---

## 7. Outer/Inner CV Protocol

### 7.1 Outer CV (DEV only)
- 5-fold USER-DISJOINT on PH19_DEV (588 users)
- Deterministic round-robin: sort users by `user_id`, fold = index % 5
- Each fold: OUTER_TRAIN (4/5) → train; OUTER_VALID (1/5) → evaluate

### 7.2 Inner CV (within each OUTER_TRAIN)
- 3-fold USER-DISJOINT on OUTER_TRAIN users
- Used for: C/λ/ρ/entropy_reg selection
- Never uses OUTER_VALID or PH19_CONFIRM

### 7.3 Fold Indices (Deterministic)
```
PH19_DEV users (sorted by user_id):
Fold 0: indices 0, 5, 10, ...
Fold 1: indices 1, 6, 11, ...
Fold 2: indices 2, 7, 12, ...
Fold 3: indices 3, 8, 13, ...
Fold 4: indices 4, 9, 14, ...
```

---

## 8. Leakage Controls

| Control | Implementation |
|---------|----------------|
| No chronological TEST | Never read `phase14_test_predictions.parquet` for selection |
| No Phase 14/17 TEST reuse | PH19_CONFIRM is new internal holdout |
| Strictly-past behavior | Expanding windows only; `|H| < 14` → neutral |
| Fold-safe ML | OOF on TRAIN only; CAL/TEST scores frozen |
| TRAIN-only normalization | ECDF, peer stats, Bayesian bins fitted on fold TRAIN |
| Threshold isolation | Best F1 on inner CAL; never on OUTER_VALID or CONFIRM |
| Deterministic RNG | Seed=42 for all resampling; PyTorch seeds fixed |

---

## 9. Hard Gates

| Gate | Check | Failure Action |
|------|-------|----------------|
| G1 | Allocation exact/reproducible | STOP |
| G2 | No Phase14 TEST or chronological TEST accessed | STOP |
| G3 | No user overlap between DEV outer folds | STOP |
| G4 | No CONFIRM data used during candidate selection | STOP |
| G5 | OOF ML predictions verified (deterministic) | STOP |
| G6 | Strictly-past behavior/context tests pass | STOP |
| G7 | TRAIN-only normalization/fitting verified | STOP |
| G8 | Threshold isolation verified | STOP |
| G9 | Deterministic rerun of selected DEV candidate | STOP |
| G10 | Preconfirm freeze complete before CONFIRM | STOP |
| G11 | CONFIRM evaluated exactly once | STOP |
| G12 | Input artifacts unchanged after execution | STOP |

---

## 10. Estimated Runtime

| Arm | Device | Time (est.) | Notes |
|-----|--------|-------------|-------|
| A0–H2 (lightweight) | CPU | 2–4 hrs | 5 outer × 3 inner × 9 arms ≈ 135 model fits |
| I (Transformer) | GPU | 1–2 hrs | Tiny architecture, early stopping |
| J (GNN+Temporal) | GPU | 2–4 hrs | If feasibility gate passes |
| **Total Kaggle** | — | **6–10 hrs** | With checkpoints every fold |

---

## 11. Deep-Arm Feasibility (ARM I & J)

### ARM I (Transformer): FEASIBLE
- PyTorch available on Kaggle
- Tiny architecture fits in memory
- Inductive (no user embeddings)
- No graph leakage

### ARM J (GNN+Temporal): FEASIBILITY GATE REQUIRED
**Checks**:
1. Strictly-past graph construction from merged features
2. Inductive GraphSAGE for unseen users
3. Fold-specific graph (no future edges)
4. Kaggle GPU memory (embed ≤16, hidden ≤32)
5. PyTorch Geometric available or pure PyTorch implementation

**If any check fails**: ARM J = NOT EXECUTED, recorded with reason.

### DEEP_ROLE = OPTION_B (FROZEN)

I/J are **descriptive upper bounds only**. They are excluded from production candidate selection.

Rationale:
- Transformer I and GNN J are research explorations requiring heavy compute (30–120 min each)
- Their inclusion would require both DEV runs to include I/J, making determinism checks fragile
- A0–H define the full adaptive-layer selection space

**I/J CANNOT:**
- Become the Phase 19 selected candidate
- Be sent to PH19_CONFIRM
- Change ADOPT_LIGHTWEIGHT_ADAPTIVE_LAYER vs ADAPTIVE_LAYER_NOT_SUPPORTED verdict

**I/J ARE:**
- Evaluated separately for research reference
- Reported as upper-bound benchmarks
- Evaluated on DEV data only (not CONFIRM)

If I/J feasibility paths pass, they run once on DEV data. Their results are stored
in `dev_upper_bounds` but do NOT influence candidate selection or the preconfirm freeze.

---

## 12. Expected Kaggle Bundle Size

| Component | Est. Size |
|-----------|-----------|
| `src/` + `tests/` + `kaggle_scripts/` | ~2 MB |
| `docs/phase19_specification.md` | ~50 KB |
| Frozen artifacts (Phase 6, 14, 15, 17, 18) | ~50 MB |
| **Total ZIP** | **~55 MB** |
| MD5 | Computed at build time |

---

## 13. Confirmation of Constraints

✅ **Chronological TEST will not be accessed** — Only Phase 14 TRAIN users used.

✅ **Phase 14/17 TEST will not be used for selection** — PH19_CONFIRM is the sole confirmation holdout.

✅ **PH19_CONFIRM remains unopened** — Allocation file created but no metrics computed on CONFIRM.

---

## 14. Preconfirm Freeze (Template)

```json
{
  "phase": "19",
  "candidate": "<selected_arm>",
  "source_hashes": {
    "phase19_module": "<sha256>",
    "phase19_runner": "<sha256>",
    "phase19_tests": "<sha256>"
  },
  "user_allocation_hash": "<sha256 of phase19_user_allocation.json>",
  "component_definitions_hash": "<sha256 of phase18.py component functions>",
  "base_lgbm_config_hash": "<sha256 of ALL_FEATURES + params>",
  "hyperparameters": { "<frozen values>" },
  "outer_folds": "<fold user lists>",
  "inner_cv_protocol": "3-fold user-disjoint on OUTER_TRAIN",
  "threshold_protocol": "best_f1 on inner CAL",
  "bootstrap": {"n": 1000, "seed": 42, "alpha": 0.10},
  "software": {"python": "3.12", "numpy": "2.0.2", "pandas": "2.3.3", "sklearn": "1.6.1", "lightgbm": "4.6.0", "torch": "2.x"},
  "confirm_opened": false
}
```

---

## 15. Checkpoints (Kaggle Execution)

| Checkpoint | Trigger |
|------------|---------|
| `bundle_verified` | ZIP uploaded, extracted, manifest MD5 match |
| `inputs_verified` | All frozen artifacts present, MD5 match |
| `phase19_split_frozen` | `phase19_user_allocation.json` written |
| `dev_fold{1..5}_complete` | Each outer fold evaluated |
| `dev_comparison_complete` | All arms evaluated on all folds |
| `candidate_selected` | DEV rule applied, at most one candidate |
| `preconfirm_freeze` | Freeze JSON written |
| `confirm_complete` | A0 + candidate evaluated on CONFIRM |
| `bootstrap_complete` | 1000 user-block resamples done |
| `verdict_complete` | Final verdict recorded |
| `artifacts_verified` | All outputs present, manifest MD5 |
| `tests_complete` | Unit tests pass |

---

## 16. Specification Changelog

### v1.1 (2026-09-07) — Pre-CONFIRM Clarification

**Status**: FROZEN clarification (no protocol change)

**Defect discovered**: F1/F2 base instantaneous model L2 regularization was undefined in v1.0.
The implementation conflated the temporal persistence parameter ρ with the base model L2,
causing ZeroDivisionError at ρ=0. Discovered before any valid DEV Run 1 completion.
PH19_CONFIRM has never been opened.

**Clarification**:
- F1 instantaneous base: Family D with fixed l2 = 0.1
- F2 instantaneous base: Family E with fixed l2 = 0.1
- 0.1 is the geometric midpoint / central value of the frozen D/E regularization grid {0.01, 0.1, 1.0}
- Chosen data-independently; not justified by observed DEV performance
- ρ remains the ONLY F1/F2 tuned hyperparameter: {0.00, 0.25, 0.50, 0.75, 0.90}
- Candidate count remains exactly 32

**Also corrected**: Family C penalty changed from `None` to `"l2"` (discovered in same audit).
C values {0.1, 1.0, 10.0} are now effective regularization strengths.

**Implementation changes**:
- F1/F2 now fit base D/E with fixed l2=0.1, then apply temporal persistence viaρ
- Temporal recurrence: out_t = (1-ρ)·instant_t + ρ·out_{t-1}, per user, chronologically
- Inner-fold cache: PML/B/T/C computed once per inner split, reused across 32 candidates
- Progress logging added to inner CV and outer CV

---

*End of Specification — FROZEN for Phase 19 execution*