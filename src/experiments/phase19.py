"""Phase 19: Comprehensive Adaptive Risk Fusion Benchmark.

Implements the approved protocol in docs/phase19_specification.md:

  - Ten algorithm families (A0, B, C, D, E, F, G, H, I, J) benchmarking
    fusion strategies over the four Phase 18 risk components:
    R_ML, R_behavior, R_trust, R_context.
  - User-disjoint Phase 14 TRAIN split into PH19_DEV (588 users) and
    PH19_CONFIRM (196 users) via deterministic 75/25 allocation.
  - 5-fold outer CV on DEV (user-disjoint), 3-fold inner CV for selection.
  - Inner CAL best_f1_threshold; CONFIRM evaluated once per selected candidate.
  - Bootstraps (n=1000, seed=42, 90% CIs); mechanical verdict.

R_ML provenance (fold-safe OOF):
  Every R_ML score is produced by a LightGBM model trained ONLY on users
  disjoint from the prediction target. Frozen 12-feature config from Phase 18.

Execution protocol (CONFIRM-once guard):
  1. run_phase19_dev()     → DEV pipeline only (for determinism double-run)
  2. run_phase19_dev()     → DEV pipeline again (determinism check)
  3. If bit_identical: write preconfirm_freeze
  4. run_phase19_confirm() → CONFIRM exactly once (A0 + selected candidate)
  5. Bootstrap from saved confirm predictions

The module is IO-free except for per-fold checkpoint writes when checkpoint_dir
is explicitly provided. The runner controls all artifact writing.

Persistent checkpoint protocol:
  - After each outer fold: write dev_run{N}_fold{K}.json
  - On resume: verify source hashes, allocation hash, skip completed folds
  - Never overwrite a completed valid fold silently
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase14 as p14
from src.experiments import phase17 as p17
from src.experiments import phase18 as p18
from src.experiments.phase14 import canonical_json, to_jsonable

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------
F_B = p18.F_B
F_G = p18.F_G
ALL_FEATURES = p18.ALL_FEATURES
MIN_HISTORY = p18.MIN_HISTORY
EPS = p18.EPS
ZERO_MAD = p18.ZERO_MAD
CLIP_Z = p18.CLIP_Z
MIN_PEERS = p18.MIN_PEERS
MIN_SUPPORT = p18.MIN_SUPPORT
P_CLIP_LO, P_CLIP_HI = p18.P_CLIP_LO, p18.P_CLIP_HI
PH19_SPEC_VERSION = "1.1"
SEED = 42
N_BOOT = 1000
ALPHA = 0.10
SLSQP_TOL = 1e-12
SLSQP_MAXITER = 1000

RECORDED_TEST_AUC_ROC = 0.786513147174144
RECORDED_TEST_AUC_PR = 0.3757395445913874
RECORDED_PRIMARY = p18.RECORDED_PRIMARY
FROZEN_THRESHOLD = p18.FROZEN_THRESHOLD

LGBM_PARAMS = {"objective": "binary", "metric": "auc",
               "learning_rate": 0.03, "num_leaves": 31,
               "min_data_in_leaf": 100, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1,
               "verbose": -1, "seed": 42}

FAMILIES = {
    "A0": {"type": "lightweight", "desc": "ML-only control"},
    "B":  {"type": "lightweight", "desc": "Convex fusion (SLSQP)"},
    "C":  {"type": "lightweight", "desc": "Logistic stacking"},
    "D":  {"type": "lightweight", "desc": "Residual logistic"},
    "E":  {"type": "lightweight", "desc": "Interaction residual"},
    "F1": {"type": "lightweight", "desc": "Temporal persistence on D"},
    "F2": {"type": "lightweight", "desc": "Temporal persistence on E"},
    "G":  {"type": "lightweight", "desc": "Bayesian evidence fusion"},
    "H1": {"type": "lightweight", "desc": "Dynamic gating (linear)"},
    "H2": {"type": "conditional", "desc": "Dynamic gating (MLP)"},
    "I":  {"type": "research", "desc": "Temporal Transformer"},
    "J":  {"type": "research", "desc": "GNN+Temporal"},
}

HP_GRID = {
    "C": [0.1, 1.0, 10.0],
    "D": [0.01, 0.1, 1.0],
    "E": [0.01, 0.1, 1.0],
    "F1": [0.00, 0.25, 0.50, 0.75, 0.90],
    "F2": [0.00, 0.25, 0.50, 0.75, 0.90],
    "H1": {"l2": [0.01, 0.1, 1.0], "entropy_reg": [0.0, 0.01]},
    "H2": {"l2": [0.01, 0.1], "entropy_reg": [0.0, 0.01], "hidden": [8]},
}


# ---------------------------------------------------------------------------
# Persistent per-fold checkpoints
# ---------------------------------------------------------------------------
def _write_fold_checkpoint(ckpt_dir, run_id, fold_k, fold_result, meta):
    """Write a single fold checkpoint to disk."""
    os.makedirs(ckpt_dir, exist_ok=True)
    rec = {"run_id": run_id, "fold_k": fold_k, "timestamp": time.time(),
           "fold_result": fold_result, "meta": meta}
    path = os.path.join(ckpt_dir, f"dev_run{run_id}_fold{fold_k}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, default=_json_safe)


def load_dev_checkpoints(ckpt_dir, run_id, expected_source_hash=None,
                         expected_alloc_hash=None):
    """Load completed fold checkpoints for a given run_id.

    Returns dict: {fold_k: fold_result} for completed folds.
    Verifies source and allocation hashes if provided.
    Never returns partially-written checkpoints (validates structure).
    """
    if not os.path.isdir(ckpt_dir):
        return {}
    loaded = {}
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.startswith(f"dev_run{run_id}_fold") or not fn.endswith(".json"):
            continue
        fp = os.path.join(ckpt_dir, fn)
        try:
            with open(fp, encoding="utf-8") as fh:
                rec = json.load(fh)
            fold_k = rec["fold_k"]
            meta = rec.get("meta", {})
            if expected_source_hash and meta.get("source_hash") != expected_source_hash:
                continue
            if expected_alloc_hash and meta.get("alloc_hash") != expected_alloc_hash:
                continue
            loaded[fold_k] = rec["fold_result"]
        except (json.JSONDecodeError, KeyError):
            continue
    return loaded


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


expanding_strictly_past_z = p18.expanding_strictly_past_z
r_behavior_block = p18.r_behavior_block
ecdf_percentile = p18.ecdf_percentile
r_trust_block = p18.r_trust_block
r_context_block = p18.r_context_block
fit_fusion = p18.fit_fusion
tie_audit = p18.tie_audit
_day_indices = p18._day_indices
_phi = p18._phi
_close = p18._close


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------
def assign_outer_folds(user_ids: list[str], n_folds: int = 5) -> dict[str, int]:
    sorted_users = sorted(user_ids)
    return {u: i % n_folds for i, u in enumerate(sorted_users)}


def assign_inner_folds(user_ids: list[str], n_folds: int = 3) -> dict[str, int]:
    sorted_users = sorted(user_ids)
    return {u: i % n_folds for i, u in enumerate(sorted_users)}


# ---------------------------------------------------------------------------
# Fold-safe LightGBM OOF
# ---------------------------------------------------------------------------
def fold_safe_lgbm_oof(train_frame: pd.DataFrame, val_frame: pd.DataFrame,
                       features: list[str] = ALL_FEATURES,
                       params: dict | None = None,
                       num_boost_round: int = 3000,
                       early_stopping_rounds: int = 100) -> tuple[np.ndarray, dict]:
    """Train LightGBM on train_frame users, predict on val_frame users."""
    import lightgbm as lgb
    if params is None:
        params = dict(LGBM_PARAMS)
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    X_train = train_frame[features].astype(np.float32)
    y_train = train_frame["is_malicious"].to_numpy().astype(int)
    X_val = val_frame[features].astype(np.float32)
    y_val = val_frame["is_malicious"].to_numpy().astype(int)
    p = dict(params)
    p["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(p, dtrain, num_boost_round=num_boost_round, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                 lgb.log_evaluation(0)])
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rec = {"train_users": int(train_frame["user"].nunique()),
           "val_users": int(val_frame["user"].nunique()),
           "train_rows": len(train_frame), "val_rows": len(val_frame),
           "train_positives": int(y_train.sum()), "val_positives": int(y_val.sum()),
           "best_iteration": int(model.best_iteration)}
    return preds, rec


def fold_safe_lgbm_predict(train_frame: pd.DataFrame, pred_frame: pd.DataFrame,
                           features: list[str] = ALL_FEATURES,
                           params: dict | None = None) -> np.ndarray:
    """Train LightGBM on train_frame, predict on pred_frame (no early stopping)."""
    import lightgbm as lgb
    if params is None:
        params = dict(LGBM_PARAMS)
    train_frame = train_frame.reset_index(drop=True)
    pred_frame = pred_frame.reset_index(drop=True)
    X_train = train_frame[features].astype(np.float32)
    y_train = train_frame["is_malicious"].to_numpy().astype(int)
    X_pred = pred_frame[features].astype(np.float32)
    p = dict(params)
    p["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(p, dtrain, num_boost_round=300)
    return model.predict(X_pred)


# ---------------------------------------------------------------------------
# Component computation
# ---------------------------------------------------------------------------
def compute_components_block(block: pd.DataFrame, train_frame: pd.DataFrame,
                             block_users: list[str], role_map: dict) -> tuple[pd.DataFrame, dict]:
    rb = p18.r_behavior_block(block)
    rt, trust_flags = p18.r_trust_block(block, train_frame)
    rc, ctx_diag = p18.r_context_block(block, block_users, role_map)
    result = pd.DataFrame({
        "user": block["user"].to_numpy(), "day": block["day"].to_numpy(),
        "r_trust": rt, "r_context": rc, "r_behavior": rb,
    })
    return result, {"trust_flags": trust_flags, "context_diag": ctx_diag}


# ---------------------------------------------------------------------------
# Inner CV: candidate evaluation
# ---------------------------------------------------------------------------
def inner_cv_evaluate(outer_train_users: list[str], outer_train_frame: pd.DataFrame,
                      role_map: dict, family: str, hp: Any = None,
                      fold_cache: dict | None = None,
                      progress_prefix: str = "") -> dict:
    """Inner CV for a single candidate. Uses precomputed fold_cache when provided.

    fold_cache: dict {k: {"R_train", "R_val", "y_train", "y_val"}} for k in 0..2.
    When provided, skips LightGBM + component computation entirely.
    """
    inner_folds = assign_inner_folds(outer_train_users, 3)
    fold_results = []
    for k in range(3):
        if fold_cache and k in fold_cache:
            fc = fold_cache[k]
            R_train, R_val = fc["R_train"], fc["R_val"]
            y_train, y_val = fc["y_train"], fc["y_val"]
            n_train, n_val = fc["n_train_users"], fc["n_val_users"]
            val_users_arr = fc.get("val_users")
            val_days_arr = fc.get("val_days")
        else:
            inner_train_users = [u for u in outer_train_users if inner_folds[u] != k]
            inner_val_users = [u for u in outer_train_users if inner_folds[u] == k]
            inner_train = outer_train_frame[outer_train_frame["user"].isin(inner_train_users)].copy()
            inner_val = outer_train_frame[outer_train_frame["user"].isin(inner_val_users)].copy()
            inner_train = inner_train.sort_values(["user", "day"]).reset_index(drop=True)
            inner_val = inner_val.sort_values(["user", "day"]).reset_index(drop=True)
            train_ml, _ = fold_safe_lgbm_oof(inner_train, inner_train)
            val_ml, _ = fold_safe_lgbm_oof(inner_train, inner_val)
            inner_train_ul = sorted(inner_train["user"].unique())
            inner_val_ul = sorted(inner_val["user"].unique())
            train_comp, _ = compute_components_block(inner_train, inner_train, inner_train_ul, role_map)
            val_comp, _ = compute_components_block(inner_val, inner_train, inner_val_ul, role_map)
            R_train = np.column_stack([train_ml, train_comp["r_trust"].to_numpy(),
                                       train_comp["r_context"].to_numpy(), train_comp["r_behavior"].to_numpy()])
            R_val = np.column_stack([val_ml, val_comp["r_trust"].to_numpy(),
                                     val_comp["r_context"].to_numpy(), val_comp["r_behavior"].to_numpy()])
            y_train = inner_train["is_malicious"].to_numpy().astype(int)
            y_val = inner_val["is_malicious"].to_numpy().astype(int)
            n_train, n_val = len(inner_train_users), len(inner_val_users)
            val_users_arr = inner_val["user"].to_numpy()
            val_days_arr = inner_val["day"].to_numpy()
        scores_val = _apply_family(R_train, y_train, R_val, family, hp,
                                   val_users=val_users_arr, val_days=val_days_arr)
        threshold, f1 = th.best_f1_threshold(y_val, scores_val)
        fold_results.append({"fold": k, "n_train_users": n_train,
                             "n_val_users": n_val,
                             "threshold": threshold, "f1": f1})
    mean_f1 = np.mean([r["f1"] for r in fold_results])
    if progress_prefix:
        hp_str = "" if hp is None else f" hp={hp}"
        print(f"{progress_prefix} {family}{hp_str} mean_f1={mean_f1:.4f}", flush=True)
    return {"family": family, "hp": hp, "folds": fold_results, "mean_f1": float(mean_f1)}


def _apply_family(R_train, y_train, R_val, family, hp=None,
                  val_users=None, val_days=None):
    """Apply a family's scoring function.

    For F1/F2: val_users and val_days are required for temporal persistence.
    val_users: array-like of user IDs aligned with R_val rows
    val_days: array-like of day values aligned with R_val rows
    """
    if family == "A0":
        return R_val[:, 0]
    elif family == "B":
        fit = family_b_fit(R_train, y_train)
        return family_b_score(R_val, fit["weights"])
    elif family == "C":
        C_val = hp if hp is not None else 1.0
        logit_ml = np.clip(np.log(R_train[:, 0] / (1 - R_train[:, 0] + EPS) + EPS), -20, 20)
        X_tr = np.column_stack([logit_ml, R_train[:, 1], R_train[:, 2], R_train[:, 3]])
        fit = family_c_fit(X_tr, y_train, C_val)
        return family_c_score(R_val, fit)
    elif family == "D":
        l2 = hp if hp is not None else 0.1
        logit_ml = np.clip(np.log(R_train[:, 0] / (1 - R_train[:, 0] + EPS) + EPS), -20, 20)
        X_tr = np.column_stack([logit_ml, R_train[:, 1], R_train[:, 2], R_train[:, 3]])
        fit = family_d_fit(X_tr, y_train, l2)
        return family_d_score(R_val, fit)
    elif family == "E":
        l2 = hp if hp is not None else 0.1
        logit_ml = np.clip(np.log(R_train[:, 0] / (1 - R_train[:, 0] + EPS) + EPS), -20, 20)
        X_tr = np.column_stack([logit_ml, R_train[:, 1], R_train[:, 2], R_train[:, 3]])
        fit = family_e_fit(X_tr, y_train, l2)
        return family_e_score(R_val, fit)
    elif family in ("F1", "F2"):
        F1_F2_BASE_L2 = 0.1
        rho = hp if hp is not None else 0.0
        base = "D" if family == "F1" else "E"
        instant_scores = _apply_family(R_train, y_train, R_val, base, F1_F2_BASE_L2)
        if rho == 0.0 or val_users is None or val_days is None:
            return instant_scores
        users_arr = np.asarray(val_users, dtype=object)
        days_arr = np.asarray(val_days)
        order = np.lexsort((days_arr, users_arr))
        sorted_users = users_arr[order]
        sorted_days = days_arr[order]
        sorted_instant = instant_scores[order]
        n = len(order)
        result = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_users[j] == sorted_users[i]:
                j += 1
            result[i] = sorted_instant[i]
            for t in range(i + 1, j):
                result[t] = (1.0 - rho) * sorted_instant[t] + rho * result[t - 1]
            i = j
        inv_order = np.empty(n, dtype=np.intp)
        inv_order[order] = np.arange(n)
        return result[inv_order]
    elif family == "G":
        fit = family_g_fit(R_train, y_train)
        return family_g_score(R_val, fit)
    elif family in ("H1", "H2"):
        hidden = None if family == "H1" else 8
        l2, ent = 0.01, 0.0
        if isinstance(hp, dict):
            l2, ent = hp.get("l2", 0.01), hp.get("entropy_reg", 0.0)
        logit_ml = np.clip(np.log(R_train[:, 0] / (1 - R_train[:, 0] + EPS) + EPS), -20, 20)
        X_tr = np.column_stack([logit_ml, R_train[:, 1], R_train[:, 2], R_train[:, 3]])
        fit = family_h_fit(X_tr, y_train, hidden=hidden, l2=l2, entropy_reg=ent)
        return family_h_score(R_val, fit)
    return R_val[:, 0]


# ---------------------------------------------------------------------------
# Outer CV: full DEV evaluation
# ---------------------------------------------------------------------------
def outer_cv_evaluate(dev_users, merged, role_map, candidates,
                      checkpoint_dir=None, run_id=1, source_hash=None, alloc_hash=None):
    """Outer CV: full DEV evaluation with optional per-fold checkpointing.

    When checkpoint_dir is provided, writes dev_run{run_id}_fold{K}.json
    after each completed fold. On resume, completed folds are skipped.
    """
    completed = {}
    if checkpoint_dir:
        completed = load_dev_checkpoints(
            checkpoint_dir, run_id,
            expected_source_hash=source_hash,
            expected_alloc_hash=alloc_hash)
        if completed:
            return completed  # all folds done or partially done

    outer_folds = assign_outer_folds(dev_users, 5)
    results = {}
    for fold_k in range(5):
        if fold_k in completed:
            results[fold_k] = completed[fold_k]
            continue
        outer_train_users = [u for u in dev_users if outer_folds[u] != fold_k]
        outer_val_users = [u for u in dev_users if outer_folds[u] == fold_k]
        print(f"[outer {fold_k+1}/5] START ({len(outer_train_users)} train, {len(outer_val_users)} val)", flush=True)
        outer_train = merged[merged["user"].isin(outer_train_users)].copy()
        outer_val = merged[merged["user"].isin(outer_val_users)].copy()
        outer_train = outer_train.sort_values(["user", "day"]).reset_index(drop=True)
        outer_val = outer_val.sort_values(["user", "day"]).reset_index(drop=True)
        train_ml, _ = fold_safe_lgbm_oof(outer_train, outer_train)
        val_ml, _ = fold_safe_lgbm_oof(outer_train, outer_val)
        ot_ul = sorted(outer_train["user"].unique())
        ov_ul = sorted(outer_val["user"].unique())
        train_comp, _ = compute_components_block(outer_train, outer_train, ot_ul, role_map)
        val_comp, _ = compute_components_block(outer_val, outer_train, ov_ul, role_map)
        R_train = np.column_stack([train_ml, train_comp["r_trust"].to_numpy(),
                                   train_comp["r_context"].to_numpy(), train_comp["r_behavior"].to_numpy()])
        R_val = np.column_stack([val_ml, val_comp["r_trust"].to_numpy(),
                                 val_comp["r_context"].to_numpy(), val_comp["r_behavior"].to_numpy()])
        y_train = outer_train["is_malicious"].to_numpy().astype(int)
        y_val = outer_val["is_malicious"].to_numpy().astype(int)
        fold_scores = {}
        outer_val_users_arr = outer_val["user"].to_numpy()
        outer_val_days_arr = outer_val["day"].to_numpy()
        for cand in candidates:
            family, hp = cand["family"], cand.get("hp")
            scores_val = _apply_family(R_train, y_train, R_val, family, hp,
                                       val_users=outer_val_users_arr,
                                       val_days=outer_val_days_arr)
            threshold, f1 = th.best_f1_threshold(y_val, scores_val)
            auc_roc = float(roc_auc_score(y_val, scores_val))
            auc_pr = float(average_precision_score(y_val, scores_val))
            op = m.binary_decision_metrics(y_val, scores_val, threshold)
            fold_scores[family] = {"threshold": threshold, "f1": f1, "auc_roc": auc_roc,
                                   "auc_pr": auc_pr, "n_alerts": op["n_alerts"],
                                   "precision": op["precision"], "recall": op["recall"],
                                   "mcc": op["mcc"]}
        fold_result = {"outer_train_users": len(outer_train_users),
                       "outer_val_users": len(outer_val_users),
                       "y_val_positive": int(y_val.sum()),
                       "arm_scores": fold_scores}
        results[fold_k] = fold_result
        if checkpoint_dir:
            _write_fold_checkpoint(checkpoint_dir, run_id, fold_k, fold_result,
                                   meta={"source_hash": source_hash,
                                         "alloc_hash": alloc_hash,
                                         "n_candidates": len(candidates)})
        print(f"[outer {fold_k+1}/5] done — {len(candidates)} candidates evaluated", flush=True)
    return results


# ---------------------------------------------------------------------------
# Bootstrap from saved predictions
# ---------------------------------------------------------------------------
def user_block_bootstrap(users, y, score, threshold, n_boot=N_BOOT, seed=SEED):
    users = np.asarray(users, dtype=object)
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    unique = np.unique(users)
    idx_by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("auc_roc", "auc_pr", "n_alerts", "precision", "recall", "f1")}
    skipped = 0
    for rep in range(1, n_boot + 1):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        t, sc = y[idx], s[idx]
        if t.sum() == 0 or t.sum() == len(t):
            skipped += 1
            continue
        rows["auc_roc"].append(roc_auc_score(t, sc))
        rows["auc_pr"].append(average_precision_score(t, sc))
        op = m.binary_decision_metrics(t, sc, threshold)
        rows["n_alerts"].append(op["n_alerts"])
        rows["precision"].append(op["precision"])
        rows["recall"].append(op["recall"])
        rows["f1"].append(op["f1"])
    out = {"n_users": int(len(unique)), "n_boot": n_boot, "seed": seed,
           "n_skipped": skipped, "threshold": threshold, "alpha": ALPHA}
    for metric, values in rows.items():
        v = np.asarray(values)
        out[metric] = {"mean": float(v.mean()),
                       "ci_low": float(np.percentile(v, 100 * ALPHA / 2)),
                       "ci_high": float(np.percentile(v, 100 * (1 - ALPHA / 2)))}
    return out


# ---------------------------------------------------------------------------
# Family scoring functions
# ---------------------------------------------------------------------------
def family_a0_score(R_ML): return R_ML.copy()

def family_b_fit(R, y):
    return fit_fusion(R, y, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25])

def family_b_score(R, weights):
    return R @ np.asarray(weights, dtype=np.float64)

def family_c_fit(X, y, C_val=1.0):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C_val, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(X, y)
    return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "C": C_val}

def family_c_score(R, fit):
    logit_ml = np.clip(np.log(R[:, 0] / (1 - R[:, 0] + EPS) + EPS), -20, 20)
    X = np.column_stack([logit_ml, R[:, 1], R[:, 2], R[:, 3]])
    z = X @ np.asarray(fit["coef"]) + fit["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

def family_d_fit(X, y, l2=0.1):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0/l2, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(X[:, 1:], y)
    return {"coef": np.clip(lr.coef_[0], 0.0, None).tolist(), "l2": l2}

def family_d_score(R, fit):
    logit_ml = np.clip(np.log(R[:, 0] / (1 - R[:, 0] + EPS) + EPS), -20, 20)
    z = logit_ml + R[:, 1:] @ np.asarray(fit["coef"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

def family_e_fit(X, y, l2=0.1):
    B, T, C = X[:, 1], X[:, 2], X[:, 3]
    X_ext = np.column_stack([B, T, C, B*T, B*C, T*C])
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0/l2, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(X_ext, y)
    return {"coef": np.clip(lr.coef_[0], 0.0, None).tolist(), "l2": l2}

def family_e_score(R, fit):
    logit_ml = np.clip(np.log(R[:, 0] / (1 - R[:, 0] + EPS) + EPS), -20, 20)
    B, T, C = R[:, 1], R[:, 2], R[:, 3]
    X_ext = np.column_stack([B, T, C, B*T, B*C, T*C])
    z = logit_ml + X_ext @ np.asarray(fit["coef"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

def family_f_score_user(scores, rho):
    n = len(scores)
    out = np.empty(n, dtype=np.float64)
    if n == 0: return out
    out[0] = scores[0]
    for t in range(1, n): out[t] = (1.0 - rho) * scores[t] + rho * out[t-1]
    return out

def family_g_fit(train_R, train_y, n_bins=10, a_prior=1.0, lr_cap=5.0):
    R_pos, R_neg = train_R[train_y == 1], train_R[train_y == 0]
    bins = {}
    for ci in range(train_R.shape[1]):
        fv = train_R[:, ci]
        lo, hi = float(np.min(fv)), float(np.max(fv))
        if lo >= hi:
            bins[ci] = {"edges": [lo, hi], "lr": [0.0]*n_bins, "n_pos": [0]*n_bins, "n_neg": [0]*n_bins}
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        lr_b, np_b, nn_b = [], [], []
        for b in range(n_bins):
            mp = (R_pos[:, ci] >= edges[b]) & (R_pos[:, ci] < edges[b+1])
            mn = (R_neg[:, ci] >= edges[b]) & (R_neg[:, ci] < edges[b+1])
            lr_b.append(float(np.clip(np.log((int(mp.sum())+a_prior)/(int(mn.sum())+a_prior)), -lr_cap, lr_cap)))
            np_b.append(int(mp.sum())); nn_b.append(int(mn.sum()))
        bins[ci] = {"edges": edges.tolist(), "lr": lr_b, "n_pos": np_b, "n_neg": nn_b}
    return {"bins": bins, "n_bins": n_bins, "a_prior": a_prior, "lr_cap": lr_cap}

def family_g_score(R, fit):
    scores = np.full(R.shape[0], 0.0, dtype=np.float64)
    for ci in range(R.shape[1]):
        bi = fit["bins"].get(ci)
        if bi is None: continue
        edges, lr = np.asarray(bi["edges"]), np.asarray(bi["lr"])
        idx = np.clip(np.searchsorted(edges, R[:, ci], side="right") - 1, 0, len(lr)-1)
        scores += lr[idx]
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))

def family_h_fit(X_train, y_train, hidden=None, l2=0.01, entropy_reg=0.0, lr=0.01, epochs=50, batch_size=256):
    try:
        import torch, torch.nn as nn, torch.optim as optim
    except ImportError:
        return {"error": "torch not available", "fallback": True}
    torch.manual_seed(SEED)
    nf = 4
    net = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Linear(hidden, nf)) if hidden else nn.Linear(nf, nf)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    crit = nn.BCELoss()
    for _ in range(epochs):
        net.train()
        for start in range(0, len(X_t), batch_size):
            idx = torch.randperm(len(X_t))[start:start+batch_size]
            X_b, y_b = X_t[idx], y_t[idx]
            opt.zero_grad()
            w = torch.softmax(net(X_b), dim=1)
            p = torch.clamp(torch.sigmoid((w * X_b).sum(dim=1)), 1e-6, 1-1e-6)
            loss = crit(p, y_b)
            if entropy_reg > 0: loss -= entropy_reg * -(w * torch.log(w+1e-8)).sum(dim=1).mean()
            loss.backward(); opt.step()
    return {"state": net.state_dict().copy(), "hidden": hidden, "l2": l2, "entropy_reg": entropy_reg}

def family_h_score(R, fit):
    if fit.get("error"): return family_b_score(R, [0.25]*4)
    try:
        import torch, torch.nn as nn
    except ImportError:
        return family_b_score(R, [0.25]*4)
    h = fit.get("hidden")
    net = nn.Sequential(nn.Linear(4,h), nn.ReLU(), nn.Linear(h,4)) if h else nn.Linear(4,4)
    net.load_state_dict(fit["state"]); net.eval()
    with torch.no_grad():
        w = torch.softmax(net(torch.tensor(R, dtype=torch.float32)), dim=1).numpy()
    return (w * R).sum(axis=1)

def family_i_fit(train_R, train_y, seq_len=14, d_model=32, n_heads=4, n_layers=1, ff_dim=64):
    try:
        import torch, torch.nn as nn, torch.optim as optim
    except ImportError:
        return {"error": "torch not available", "fallback": True}
    torch.manual_seed(SEED)
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, d_model)
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, batch_first=True), n_layers)
            self.head = nn.Linear(d_model, 1)
        def forward(self, x): return self.head(self.tf(self.proj(x))[:, -1, :]).squeeze(-1)
    model = T(); opt = optim.Adam(model.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    n = len(train_R)
    if n < seq_len: return {"error": "insufficient data", "fallback": True}
    XSeq = torch.tensor(np.array([train_R[i-seq_len:i] for i in range(seq_len, n)]), dtype=torch.float32)
    ySeq = torch.tensor(train_y[seq_len:], dtype=torch.float32)
    model.train()
    for _ in range(20):
        for start in range(0, len(XSeq), 64):
            idx = torch.randperm(len(XSeq))[start:start+64]
            opt.zero_grad(); crit(model(XSeq[idx]), ySeq[idx]).backward(); opt.step()
    return {"state": model.state_dict().copy(), "d_model": d_model, "n_heads": n_heads,
            "n_layers": n_layers, "ff_dim": ff_dim, "seq_len": seq_len}

def family_i_score(R, fit):
    if fit.get("error") or fit.get("fallback"): return family_b_score(R, [0.25]*4)
    try:
        import torch, torch.nn as nn
    except ImportError:
        return family_b_score(R, [0.25]*4)
    sl = fit["seq_len"]
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, fit["d_model"])
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(fit["d_model"], fit["n_heads"], fit["ff_dim"], batch_first=True), fit["n_layers"])
            self.head = nn.Linear(fit["d_model"], 1)
        def forward(self, x): return self.head(self.tf(self.proj(x))[:, -1, :]).squeeze(-1)
    model = T(); model.load_state_dict(fit["state"]); model.eval()
    scores = np.full(len(R), 0.5, dtype=np.float64)
    if len(R) < sl: return scores
    with torch.no_grad():
        XSeq = torch.tensor(np.array([R[i-sl:i] for i in range(sl, len(R))]), dtype=torch.float32)
        scores[sl:] = torch.sigmoid(model(XSeq)).numpy()
    return scores

def family_j_feasibility_check(merged):
    checks = {"graph_features_available": sum(1 for f in F_G if f in merged.columns) >= 2}
    try:
        import torch_geometric; checks["pyg_available"] = True
    except ImportError:
        checks["pyg_available"] = False
    checks["feasible"] = checks["graph_features_available"] and checks["pyg_available"]
    return checks


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def compute_verdict(dev_agg, confirm_results, candidates, selected):
    if selected is None:
        return {"verdict": "CAUTION", "reason": "no candidate selected", "selected": None}
    if selected not in confirm_results:
        return {"verdict": "FAIL", "reason": f"{selected} not in CONFIRM", "selected": selected}
    cr, a0 = confirm_results[selected], confirm_results.get("A0", {})
    dap = cr.get("auc_pr", 0) - a0.get("auc_pr", 0)
    dar = cr.get("auc_roc", 0) - a0.get("auc_roc", 0)
    f1, na = cr.get("f1", 0), cr.get("n_alerts", 0)
    if dar < -0.01 or dap < -0.01 or na > 2000:
        v, r = "FAIL", f"degradation: dr={dar:.4f} dp={dap:.4f}"
    elif dap > 0 and dar > -0.005 and f1 >= 0.30 and na <= 1500:
        v, r = "PASS", f"improvement: dr={dar:.4f} dp={dap:.4f}"
    else:
        v, r = "CAUTION", f"no improvement: dr={dar:.4f} dp={dap:.4f}"
    return {"verdict": v, "reason": r, "selected": selected,
            "delta_auc_roc": dar, "delta_auc_pr": dap, "f1": f1,
            "precision": cr.get("precision", 0), "mcc": cr.get("mcc", 0), "n_alerts": na}


# ---------------------------------------------------------------------------
# Preconfirm freeze (written AFTER DEV determinism passes)
# ---------------------------------------------------------------------------
def build_preconfirm_freeze(selected, candidates, allocation, dev_outer,
                            lgbm_config, source_hashes):
    hp = next((c for c in candidates if c["family"] == selected), {})
    return {
        "phase": "19",
        "version": "1.0",
        "selected_candidate": selected,
        "selected_hyperparameters": hp.get("hp"),
        "dev_families_evaluated": [c["family"] for c in candidates],
        "dev_outer_cv_mean_auc_pr": {c["family"]: dev_outer.get(c["family"], {}).get("mean_auc_pr")
                                     for c in candidates if c["family"] in dev_outer},
        "threshold_policy": "best_f1 on inner CAL, applied once to CONFIRM",
        "frozen_lgbm_config": lgbm_config,
        "component_definitions": "Phase 18 source code (r_behavior_block, r_trust_block, r_context_block)",
        "acceptance_criteria": {"pass_delta_auc_pr": 0.0, "pass_f1": 0.30, "pass_alerts": 1500,
                                "fail_delta_auc_roc": -0.01, "fail_delta_auc_pr": -0.01, "fail_alerts": 2000},
        "bootstrap": {"n": N_BOOT, "seed": SEED, "alpha": ALPHA},
        "allocation": {"dev_users": len(allocation["dev"]["users"]),
                       "confirm_users": len(allocation["confirm"]["users"])},
        "source_hashes": source_hashes,
        "confirm_opened": False,
        "confirm_completed": False,
    }


# ---------------------------------------------------------------------------
# DEV-only assembler (for determinism double-run)
# ---------------------------------------------------------------------------
def run_phase19_dev(merged, role_df, allocation, gates_env=None,
                    checkpoint_dir=None, run_id=1, source_hash=None, alloc_hash=None):
    """DEV pipeline only. No CONFIRM access. For determinism double-run.

    When checkpoint_dir is provided, per-fold outer CV results are written
    to disk after each completed fold. On resume, completed folds are skipped.
    """
    dev_users = allocation["dev"]["users"]
    role_map = {str(r["user_id"]): (r["role"], r["department"]) for _, r in role_df.iterrows()}
    dev_merged = merged[merged["user"].isin(dev_users)].copy()
    dev_merged = dev_merged.sort_values(["user", "day"]).reset_index(drop=True)

    # ── Precompute inner-fold cache (PML + components once per split) ──
    _inner_folds = assign_inner_folds(dev_users, 3)
    inner_fold_cache = {}
    _pf = "[dev] building inner-fold cache"
    for k in range(3):
        print(f"{_pf} {k+1}/3 ...", flush=True)
        i_train_users = [u for u in dev_users if _inner_folds[u] != k]
        i_val_users = [u for u in dev_users if _inner_folds[u] == k]
        i_train = dev_merged[dev_merged["user"].isin(i_train_users)].copy()
        i_val = dev_merged[dev_merged["user"].isin(i_val_users)].copy()
        i_train = i_train.sort_values(["user", "day"]).reset_index(drop=True)
        i_val = i_val.sort_values(["user", "day"]).reset_index(drop=True)
        train_ml, _ = fold_safe_lgbm_oof(i_train, i_train)
        val_ml, _ = fold_safe_lgbm_oof(i_train, i_val)
        i_train_ul = sorted(i_train["user"].unique())
        i_val_ul = sorted(i_val["user"].unique())
        train_comp, _ = compute_components_block(i_train, i_train, i_train_ul, role_map)
        val_comp, _ = compute_components_block(i_val, i_train, i_val_ul, role_map)
        R_train = np.column_stack([train_ml, train_comp["r_trust"].to_numpy(),
                                   train_comp["r_context"].to_numpy(), train_comp["r_behavior"].to_numpy()])
        R_val = np.column_stack([val_ml, val_comp["r_trust"].to_numpy(),
                                 val_comp["r_context"].to_numpy(), val_comp["r_behavior"].to_numpy()])
        inner_fold_cache[k] = {
            "R_train": R_train, "R_val": R_val,
            "y_train": i_train["is_malicious"].to_numpy().astype(int),
            "y_val": i_val["is_malicious"].to_numpy().astype(int),
            "n_train_users": len(i_train_users),
            "n_val_users": len(i_val_users),
            "val_users": i_val["user"].to_numpy(),
            "val_days": i_val["day"].to_numpy(),
        }
    print(f"{_pf} done", flush=True)

    # ── Candidate inner CV ──
    candidates = []
    for family in FAMILIES:
        if family in ("I", "J"): continue
        hp_list = []
        if family in ("F1", "F2", "C", "D", "E"):
            hp_list = [(hp, hp) for hp in HP_GRID[family]]
        elif family in ("H1", "H2"):
            hg = HP_GRID[family]
            hp_list = [((l2, ent), {"l2": l2, "entropy_reg": ent}) for l2 in hg["l2"] for ent in hg["entropy_reg"]]
        else:
            hp_list = [(None, None)]
        for hk, hv in hp_list:
            r = inner_cv_evaluate(dev_users, dev_merged, role_map, family, hv,
                                  fold_cache=inner_fold_cache,
                                  progress_prefix="[dev]")
            candidates.append({"family": family, "hp": hk, "hp_val": hv, "mean_f1": r["mean_f1"]})

    candidates.sort(key=lambda c: c["mean_f1"], reverse=True)
    selected = candidates[0]["family"] if candidates else None

    dev_outer = outer_cv_evaluate(dev_users, dev_merged, role_map, candidates,
                                  checkpoint_dir=checkpoint_dir, run_id=run_id,
                                  source_hash=source_hash, alloc_hash=alloc_hash)
    dev_agg = {}
    for cand in candidates:
        f = cand["family"]
        fa = [dev_outer[k]["arm_scores"][f]["auc_pr"] for k in dev_outer if f in dev_outer[k]["arm_scores"]]
        if fa: dev_agg[f] = {"mean_auc_pr": float(np.mean(fa)), "std_auc_pr": float(np.std(fa))}

    return {
        "candidates": candidates, "selected": selected,
        "dev_outer_cv": dev_outer, "dev_aggregated": dev_agg,
        "confirm_opened": False, "confirm_completed": False,
    }


# ---------------------------------------------------------------------------
# CONFIRM assembler (called ONCE after DEV determinism passes)
# ---------------------------------------------------------------------------
def run_phase19_confirm(merged, role_df, allocation, dev_result, lgbm_config):
    """CONFIRM evaluation: A0 + ONE selected candidate. Exactly once."""
    dev_users = allocation["dev"]["users"]
    confirm_users = allocation["confirm"]["users"]
    role_map = {str(r["user_id"]): (r["role"], r["department"]) for _, r in role_df.iterrows()}
    selected = dev_result["selected"]
    candidates = dev_result["candidates"]

    dev_merged = merged[merged["user"].isin(dev_users)].copy()
    dev_merged = dev_merged.sort_values(["user", "day"]).reset_index(drop=True)
    confirm_merged = merged[merged["user"].isin(confirm_users)].copy()
    confirm_merged = confirm_merged.sort_values(["user", "day"]).reset_index(drop=True)

    # R_ML: train on ALL DEV, predict CONFIRM
    confirm_ml = fold_safe_lgbm_predict(dev_merged, confirm_merged)
    confirm_comp, _ = compute_components_block(confirm_merged, dev_merged,
                                               sorted(confirm_merged["user"].unique()), role_map)
    R_confirm = np.column_stack([confirm_ml, confirm_comp["r_trust"].to_numpy(),
                                 confirm_comp["r_context"].to_numpy(), confirm_comp["r_behavior"].to_numpy()])
    y_confirm = confirm_merged["is_malicious"].to_numpy().astype(int)

    # A0 + selected
    confirm_results = {}
    confirm_scores = {}
    for arm in ("A0", selected):
        if arm == "A0":
            scores = R_confirm[:, 0]
        else:
            dev_ml = fold_safe_lgbm_oof(dev_merged, dev_merged)[0]
            dev_comp, _ = compute_components_block(dev_merged, dev_merged, dev_users, role_map)
            R_dev = np.column_stack([dev_ml, dev_comp["r_trust"].to_numpy(),
                                     dev_comp["r_context"].to_numpy(), dev_comp["r_behavior"].to_numpy()])
            hp_val = candidates[0]["hp_val"] if candidates else None
            scores = _apply_family(R_dev, dev_merged["is_malicious"].to_numpy().astype(int),
                                   R_confirm, arm, hp_val,
                                   val_users=confirm_merged["user"].to_numpy(),
                                   val_days=confirm_merged["day"].to_numpy())
        threshold, f1 = th.best_f1_threshold(y_confirm, scores)
        op = m.binary_decision_metrics(y_confirm, scores, threshold)
        ties = tie_audit(scores, len(y_confirm))
        confirm_results[arm] = {
            "threshold": threshold, "f1": f1,
            "auc_roc": float(roc_auc_score(y_confirm, scores)),
            "auc_pr": float(average_precision_score(y_confirm, scores)),
            "n_alerts": op["n_alerts"], "precision": op["precision"],
            "recall": op["recall"], "mcc": op["mcc"],
            "balanced_accuracy": op["balanced_accuracy"], "tie_audit": ties,
        }
        confirm_scores[arm] = scores

    # Bootstrap from saved predictions
    boot = {}
    for arm in ("A0", selected):
        scores = confirm_scores[arm]
        threshold = confirm_results[arm]["threshold"]
        boot[arm] = user_block_bootstrap(confirm_merged["user"].to_numpy(),
                                          y_confirm, scores, threshold)

    verdict = compute_verdict(dev_result["dev_aggregated"], confirm_results, candidates, selected)

    return {
        "confirm_results": confirm_results,
        "confirm_scores": {k: v.tolist() for k, v in confirm_scores.items()},
        "confirm_users": sorted(confirm_merged["user"].tolist()),
        "y_confirm": y_confirm.tolist(),
        "bootstrap": boot,
        "verdict": verdict,
    }


def json_safe(obj):
    return to_jsonable(obj)

def canonical(obj):
    return canonical_json(obj)

def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
