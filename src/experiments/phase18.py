"""Phase 18: Leakage-Safe Adaptive Risk v2 (research evaluation, CPU-only).

Implements the approved protocol in docs/phase18_specification.md:

  - Four risk components over the user-disjoint Phase 14/17 framework:
    R_ML (frozen auxiliary scores: TRAIN = 5-fold entity-disjoint OOF
    cross-fit; CAL/TEST = frozen artifacts), R_behavior (strictly-past
    expanding median/MAD deviation, magnitude-only), R_trust (TRAIN-block
    ECDF percentiles of the 4 accepted graph features), R_context
    (role/department peer deviation, block-local strictly-past peer pools,
    weekday/weekend strata, min-peer/min-support fallbacks).
  - Fusion R_final = a*R_ML + b*R_trust + c*R_context + d*R_behavior,
    weights >= 0, sum = 1, learned on TRAIN-OOF rows only (SLSQP, mean
    log-loss, p clipped to [1e-6, 1-1e-6]); [1,0,0,0] is a valid result.
  - Arms A (ML-only), B (equal 0.25), C (learned full), C1/C2/C3 (learned
    ablations). CAL = threshold selection only (best_f1_threshold once per
    arm); TEST = evaluated once.
  - Gates 1-7; mechanical PASS/CAUTION/FAIL per specification Section 16;
    calibration separated (Brier/ECE descriptive only); tie audit;
    user-block + weight-refit + row-level bootstraps (n=1000, seed 42,
    90% CIs); diagnostic users JJM0203/WDD0366 (never in any fit).

Hard constraints (enforced structurally): the authoritative chronological
TEST never enters this module; no TEST-informed selection; no calibration
transform; no CNN/GNN/transformer/deep/GPU methods; the 14.5-GB HTTP log
is never read; no random split; deterministic (all resampling seed 42).

The module is IO-free; the runner (kaggle_scripts/run_phase18.py) does all
file IO, md5 gates, determinism double-run and artifact writing.
"""
from __future__ import annotations

import multiprocessing
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase14 as p14
from src.experiments import phase17 as p17
from src.experiments.phase14 import (
    Z90, canonical_json, gini_of_counts, to_jsonable,
    user_block_bootstrap_ci, wilson_ci,
)

# ---------------------------------------------------------------------------
# Pre-registered constants (specification Appendix A / Section 6)
# ---------------------------------------------------------------------------
F_B = ["login_count", "after_hours_login_count", "usb_connection_count",
       "file_access_count", "sensitive_file_access_count",
       "http_activity_count", "unique_device_count", "unusual_access_count"]
F_G = ["device_consistency_score", "rare_device_usage_count",
       "file_type_consistency_score", "rare_file_type_access_count"]
ALL_FEATURES = F_B + F_G
N_TRAIN_ROWS = 392_784
MIN_HISTORY = 14
EPS = 1e-6
ZERO_MAD = 1e-9
CLIP_Z = 3.0
MIN_PEERS = 5
MIN_SUPPORT = 20
P_CLIP_LO, P_CLIP_HI = 1e-6, 1 - 1e-6
FROZEN_THRESHOLD = 0.9186015432508062
SEED = 42
N_BOOT = 1000
ALPHA = 0.10
WILSON_Z = 1.6448536269514722
SLSQP_TOL = 1e-12
SLSQP_MAXITER = 1000
TIE_CAUTION_FRAC = 0.001  # > 0.1% of TEST rows -> CAUTION flag

# Frozen Phase 14 records on the user-disjoint TEST (gate 1 / anchors)
RECORDED_TEST_AUC_ROC = 0.786513147174144
RECORDED_TEST_AUC_PR = 0.3757395445913874
RECORDED_PRIMARY = {
    "n_alerts": 1026, "precision": 0.20175438596491227,
    "recall": 0.43487394957983194, "f1": 0.2756324900133156,
    "mcc": 0.2872810379712531, "tp": 207, "fp": 819, "tn": 52813,
    "fn": 269, "balanced_accuracy": 0.7098016078448086,
    "alert_rate": 0.018962075848303395,
}
RECORDED_TOP_K = {
    "precision_at_10": 1.0, "precision_at_30": 1.0, "precision_at_50": 1.0,
    "precision_at_100": 0.95, "recall_at_10": 0.02100840336134454,
    "recall_at_30": 0.06302521008403361, "recall_at_50": 0.10504201680672269,
    "recall_at_100": 0.19957983193277312,
}
GATE_REL_TOL = 1e-12

# Success bounds (specification Section 16)
PASS_DPR = 0.02            # DeltaAUC-PR absolute
PASS_DR_CI_LO = -0.005     # DeltaAUC-ROC CI lower bound
PASS_F1 = 0.40
PASS_PRECISION = 0.70
PASS_ALERTS = 300
PASS_MCC = 0.40
PASS_COVERAGE = 8
PASS_BRIER = 0.102734      # raw ML TEST Brier (not worse)
PASS_ECE = 0.10
PASS_AUX_MASS = 0.05       # 1 - alpha
PASS_WEIGHT_CI_WIDTH = 0.20
FAIL_DPR = -0.02           # material degradation bounds
FAIL_DR = -0.01
FAIL_ALERTS_MULT = 2.0     # > 2 x arm A alerts (1,026 -> 2,052)
FAIL_F1_GAP = 0.05         # F1 < arm A F1 - 0.05 (0.2756 -> 0.2256)
COLLAPSE_ALPHA = 0.95

ARMS = {
    "A": {"free_cols": None, "fixed": np.array([1.0, 0.0, 0.0, 0.0]),
          "x0": None},
    "B": {"free_cols": None, "fixed": np.array([0.25, 0.25, 0.25, 0.25]),
          "x0": None},
    "C": {"free_cols": [0, 1, 2, 3], "x0": [0.25, 0.25, 0.25, 0.25]},
    "C1": {"free_cols": [0, 3], "x0": [0.5, 0.5]},
    "C2": {"free_cols": [0, 1], "x0": [0.5, 0.5]},
    "C3": {"free_cols": [0, 2], "x0": [0.5, 0.5]},
}
COMPONENT_NAMES = ["r_ml", "r_trust", "r_context", "r_behavior"]
LEARNED_ARMS = ("C", "C1", "C2", "C3")
AUXILIARY_ARMS = ("C", "C1", "C2", "C3")


def _close(a: float, b: float, rel: float = GATE_REL_TOL) -> bool:
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF (scipy.special.ndtr; deterministic, vectorized)."""
    return ndtr(np.asarray(x, dtype=np.float64))


def _clip_z(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(x), CLIP_Z)


# ---------------------------------------------------------------------------
# Section 6.2 -- R_behavior (strictly-past expanding deviation)
# ---------------------------------------------------------------------------
def expanding_strictly_past_z(X: np.ndarray, min_history: int = MIN_HISTORY,
                              eps: float = EPS,
                              zero_mad: float = ZERO_MAD) -> np.ndarray:
    """z_{u,d} = (x_{u,d} - median(H)) / (MAD(H) + eps), H = strictly-past.

    X: (n_users, n_days) float64, one feature, grid-ordered (day index d).
    Rules: |H| < min_history -> z = 0; MAD(H) < zero_mad -> z = 0.
    Deterministic; no RNG.
    """
    X = np.asarray(X, dtype=np.float64)
    n_u, n_d = X.shape
    z = np.zeros((n_u, n_d), dtype=np.float64)
    for d in range(min_history, n_d):
        h = X[:, :d]
        med = np.median(h, axis=1)
        mad = np.median(np.abs(h - med[:, None]), axis=1)
        zz = (X[:, d] - med) / (mad + eps)
        zz = np.where(mad < zero_mad, 0.0, zz)
        z[:, d] = zz
    return z


def r_behavior_block(block: pd.DataFrame, features: list[str] = F_B,
                     n_days: int = 501) -> np.ndarray:
    """R_behavior per row of a block (sorted by (user, day))."""
    out = np.zeros(len(block), dtype=np.float64)
    users = sorted(block["user"].unique())
    x = block[features].to_numpy(dtype=np.float64)
    day_idx = _day_indices(block)
    for ui, u in enumerate(users):
        rows = np.flatnonzero(block["user"].to_numpy() == u)
        rows = rows[np.argsort(day_idx[rows], kind="stable")]
        for fi, f in enumerate(features):
            z = expanding_strictly_past_z(x[rows, fi].reshape(1, -1))
            out[rows] += _phi(_clip_z(z[0])) / len(features)
    return out


def _day_indices(block: pd.DataFrame) -> np.ndarray:
    days = pd.to_datetime(block["day"]).dt.normalize()
    return (days - days.iloc[0]).dt.days.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Section 6.3 -- R_trust (TRAIN-block ECDF percentiles, accepted features)
# ---------------------------------------------------------------------------
def ecdf_percentile(train_vals: np.ndarray, new_vals: np.ndarray,
                    n_train: int = N_TRAIN_ROWS) -> np.ndarray:
    """p_hat(v) = (rank_mid(v) - 1/2) / n_train over TRAIN-block values.

    Midpoint-rank tie convention (Phase 17 ECDF arm). Values outside the
    TRAIN range are clamped to [0, 1] (component domain, Section 6 / gate 5);
    the number of clamped values is reported by the caller.
    """
    sv = np.sort(np.asarray(train_vals, dtype=np.float64))
    v = np.asarray(new_vals, dtype=np.float64)
    lo = np.searchsorted(sv, v, side="left")
    hi = np.searchsorted(sv, v, side="right")
    rank_mid = (lo + hi) / 2.0
    p = (rank_mid - 0.5) / n_train
    return np.clip(p, 0.0, 1.0)


def r_trust_block(block: pd.DataFrame, train_block: pd.DataFrame,
                  features: list[str] = F_G,
                  n_train: int = N_TRAIN_ROWS) -> tuple[np.ndarray, dict]:
    """R_trust per row of a block; ECDF fitted on TRAIN-block rows only.

    Degenerate feature (< 2 distinct TRAIN values) -> constant 0.5, flagged.
    """
    n = len(block)
    out = np.zeros(n, dtype=np.float64)
    flags = {}
    for f in features:
        tv = train_block[f].to_numpy(dtype=np.float64)
        unique = np.unique(tv)
        if unique.size < 2:
            out += 0.5 / len(features)
            flags[f] = "degenerate -> constant 0.5"
            continue
        p = ecdf_percentile(tv, block[f].to_numpy(dtype=np.float64), n_train)
        n_below = int((p == 0.0).sum())
        n_above = int((p == 1.0).sum())
        flags[f] = {"n_distinct_train": int(unique.size),
                    "n_clamped_low": n_below, "n_clamped_high": n_above}
        out += p / len(features)
    return out, flags


# ---------------------------------------------------------------------------
# Section 6.4 -- R_context (role/department peer deviation, block-local)
# ---------------------------------------------------------------------------
def peer_groups(block_users: list[str], role_map: dict,
                min_peers: int = MIN_PEERS) -> tuple[dict, dict]:
    """Per user: peer list (block-local), or None -> neutral 0.5.

    Same role first (min 5 peers), else same department (min 5 peers),
    else neutral. role_map: {user_id: (role, department)}.
    """
    out, source = {}, {}
    for u in sorted(block_users):
        role, dept = role_map.get(u, (None, None))
        if role is None:
            out[u], source[u] = None, "neutral-no-role"
            continue
        peers = [v for v in block_users if v != u
                 and role_map.get(v, (None, None))[0] == role]
        src = "role"
        if len(peers) < min_peers:
            if dept is None:
                out[u], source[u] = None, "neutral-no-department"
                continue
            peers = [v for v in block_users if v != u
                     and role_map.get(v, (None, None))[1] == dept]
            src = "department"
            if len(peers) < min_peers:
                out[u], source[u] = None, "neutral-small-peer-group"
                continue
        out[u], source[u] = peers, src
    return out, source


def _weekday_mask(day_idx: np.ndarray, first_day: pd.Timestamp) -> np.ndarray:
    dates = first_day + pd.to_timedelta(day_idx, unit="D")
    return dates.dayofweek < 5


def _median_axis0(a: np.ndarray) -> np.ndarray:
    """Exact np.median(a, axis=0) for finite data (one column per feature).

    Identical element selection and float arithmetic to np.median for
    finite inputs (odd: kth partition element; even: mean of the two
    middle elements), without the per-call NaN/axis machinery. If a
    non-finite value is present the call falls back to np.median to
    preserve exact semantics (gate 5 guarantees finite component inputs;
    the fallback keeps the helper total for any input).
    """
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        return np.median(a, axis=0)
    n = a.shape[0]
    if n % 2 == 1:
        k = n // 2
        a = np.partition(a, k, axis=0)  # returns a partitioned copy
        return a[k]
    k1 = n // 2 - 1
    a = np.partition(a, (k1, n // 2), axis=0)
    return (a[k1] + a[n // 2]) / 2.0


def _r_context_block_serial(block: pd.DataFrame, block_users: list[str],
                            role_map: dict,
                            features: list[str] = ALL_FEATURES,
                            min_peers: int = MIN_PEERS,
                            min_support: int = MIN_SUPPORT,
                            eps: float = EPS) -> tuple[np.ndarray, dict]:
    """R_context per row of a block (block-local strictly-past peer pools).

    FROZEN reference implementation (pre-optimization, 2026-08-20):
    used by tests to verify the optimized r_context_block is bit-identical.
    Do not modify; the optimized dispatcher below is the production path.

    Peer stats over peers' rows with day < d, stratified by day-of-week
    type of d (weekday Mon-Fri vs weekend Sat-Sun); stratum window <
    min_support rows -> unstratified strictly-past window; still <
    min_support -> zc = 0 (neutral). Magnitude-only, clip |zc| <= 3, phi map.
    """
    n = len(block)
    out = np.zeros(n, dtype=np.float64)
    block = block.reset_index(drop=True)
    user_arr = block["user"].to_numpy()
    day_idx = _day_indices(block)
    x = block[features].to_numpy(dtype=np.float64)
    first_day = pd.to_datetime(block["day"]).dt.normalize().iloc[0]
    wd = _weekday_mask(day_idx, first_day)
    diag = {"neutral_users": {}, "peer_sources": {}, "peer_pool_sizes": {}}
    pg, pgsrc = peer_groups(block_users, role_map, min_peers)
    pool_members: dict[tuple, list[str]] = {}
    for u in sorted(block_users):
        peers, src = pg.get(u), pgsrc.get(u)
        diag["peer_sources"][u] = src
        if peers is None:
            mask = user_arr == u
            out[mask] = 0.5
            diag["neutral_users"][u] = src
            continue
        diag["peer_pool_sizes"][u] = len(peers)
        key = tuple(sorted(peers))
        pool_members.setdefault(key, []).append(u)
    for key, members in pool_members.items():
        # Window medians depend only on the peer pool (not the member), so
        # they are computed once per pool-day and shared by all members.
        p_idx = np.flatnonzero(np.isin(user_arr, list(key)))
        wd_idx = p_idx[wd[p_idx]]
        we_idx = p_idx[~wd[p_idx]]
        wd_idx = wd_idx[np.argsort(day_idx[wd_idx], kind="stable")]
        we_idx = we_idx[np.argsort(day_idx[we_idx], kind="stable")]
        wd_days = day_idx[wd_idx]
        we_days = day_idx[we_idx]
        wd_x = x[wd_idx]  # weekday peer rows, sorted by day
        we_x = x[we_idx]  # weekend peer rows, sorted by day
        m_rows = np.flatnonzero(np.isin(user_arr, members))
        cache = {}
        for cur in np.unique(day_idx[m_rows]):
            n_wd = int(np.searchsorted(wd_days, cur, side="left"))
            n_we = int(np.searchsorted(we_days, cur, side="left"))
            entry = [n_wd, n_we, None, None, None, None, None, None]
            if n_wd >= min_support:
                entry[2] = np.median(wd_x[:n_wd], axis=0)
                entry[3] = np.median(
                    np.abs(wd_x[:n_wd] - entry[2][None, :]), axis=0)
            if n_we >= min_support:
                entry[4] = np.median(we_x[:n_we], axis=0)
                entry[5] = np.median(
                    np.abs(we_x[:n_we] - entry[4][None, :]), axis=0)
            if n_wd + n_we >= min_support:
                if n_wd and n_we:
                    win = np.concatenate([wd_x[:n_wd], we_x[:n_we]])
                elif n_wd:
                    win = wd_x[:n_wd]
                else:
                    win = we_x[:n_we]
                entry[6] = np.median(win, axis=0)
                entry[7] = np.median(np.abs(win - entry[6][None, :]),
                                     axis=0)
            cache[int(cur)] = entry
        for u in members:
            rows = np.flatnonzero(user_arr == u)
            rows = rows[np.argsort(day_idx[rows], kind="stable")]
            for d in rows:
                cur = int(day_idx[d])
                (n_wd, n_we, med_wd, mad_wd, med_we, mad_we,
                 med_all, mad_all) = cache[cur]
                n_past = n_wd + n_we
                if n_past == 0:
                    out[d] = 0.5  # neutral: no strictly-past peer rows
                    continue
                if wd[d]:
                    if n_wd >= min_support:
                        med, mad = med_wd, mad_wd
                    elif n_past >= min_support:
                        med, mad = med_all, mad_all
                    else:
                        out[d] = 0.5  # insufficient strictly-past support
                        continue
                else:
                    if n_we >= min_support:
                        med, mad = med_we, mad_we
                    elif n_past >= min_support:
                        med, mad = med_all, mad_all
                    else:
                        out[d] = 0.5  # insufficient strictly-past support
                        continue
                zc = np.where(mad < 1e-9, 0.0,
                              (x[d] - med) / (mad + eps))  # zero-MAD 6.2
                out[d] = float(np.mean(_phi(_clip_z(zc))))
    return out, diag


# ---------------------------------------------------------------------------
# Optimized R_context (authorized 2026-08-20, implementation only):
# precomputed user->rows index maps + independent peer-pool computations
# run in parallel processes; per-pool arithmetic identical to the frozen
# serial reference (values are order-independent medians of the same
# strictly-past multisets), outputs written back by original row index.
# ---------------------------------------------------------------------------
_CTX_ARGS = None


def _ctx_set_globals(user_arr: np.ndarray, day_idx: np.ndarray,
                     wd: np.ndarray, x: np.ndarray,
                     min_support: int, eps: float) -> None:
    global _CTX_ARGS
    _CTX_ARGS = (user_arr, day_idx, wd, x, int(min_support), float(eps))


def _ctx_init(user_arr, day_idx, wd, x, min_support, eps) -> None:
    _ctx_set_globals(user_arr, day_idx, wd, x, min_support, eps)


def _ctx_pool_task(task: tuple) -> tuple[np.ndarray, np.ndarray]:
    """R_context for one peer pool; mirrors the serial pool loop exactly.

    task = (key, members, p_idx, m_idx) with p_idx/m_idx the increasing
    row indices of peer/member rows (precomputed once by the parent).
    Returns (row indices, values); the parent writes out[idx] = value so
    the output array is identical regardless of pool scheduling order.
    """
    key, members, p_idx, m_idx = task
    user_arr, day_idx, wd, x, min_support, eps = _CTX_ARGS
    wd_idx = p_idx[wd[p_idx]]
    we_idx = p_idx[~wd[p_idx]]
    wd_idx = wd_idx[np.argsort(day_idx[wd_idx], kind="stable")]
    we_idx = we_idx[np.argsort(day_idx[we_idx], kind="stable")]
    wd_days = day_idx[wd_idx]
    we_days = day_idx[we_idx]
    wd_x = x[wd_idx]  # weekday peer rows, sorted by day
    we_x = x[we_idx]  # weekend peer rows, sorted by day
    cache = {}
    for cur in np.unique(day_idx[m_idx]):
        n_wd = int(np.searchsorted(wd_days, cur, side="left"))
        n_we = int(np.searchsorted(we_days, cur, side="left"))
        entry = [n_wd, n_we, None, None, None, None, None, None]
        if n_wd >= min_support:
            entry[2] = _median_axis0(wd_x[:n_wd])
            entry[3] = _median_axis0(np.abs(wd_x[:n_wd] - entry[2][None, :]))
        if n_we >= min_support:
            entry[4] = _median_axis0(we_x[:n_we])
            entry[5] = _median_axis0(np.abs(we_x[:n_we] - entry[4][None, :]))
        if n_wd + n_we >= min_support:
            if n_wd and n_we:
                win = np.concatenate([wd_x[:n_wd], we_x[:n_we]])
            elif n_wd:
                win = wd_x[:n_wd]
            else:
                win = we_x[:n_we]
            entry[6] = _median_axis0(win)
            entry[7] = _median_axis0(np.abs(win - entry[6][None, :]))
        cache[int(cur)] = entry
    d_idx, d_val = [], []
    for d in m_idx:
        cur = int(day_idx[d])
        (n_wd, n_we, med_wd, mad_wd, med_we, mad_we,
         med_all, mad_all) = cache[cur]
        n_past = n_wd + n_we
        if n_past == 0:
            d_idx.append(d)
            d_val.append(0.5)  # neutral: no strictly-past peer rows
            continue
        if wd[d]:
            if n_wd >= min_support:
                med, mad = med_wd, mad_wd
            elif n_past >= min_support:
                med, mad = med_all, mad_all
            else:
                d_idx.append(d)
                d_val.append(0.5)  # insufficient strictly-past support
                continue
        else:
            if n_we >= min_support:
                med, mad = med_we, mad_we
            elif n_past >= min_support:
                med, mad = med_all, mad_all
            else:
                d_idx.append(d)
                d_val.append(0.5)  # insufficient strictly-past support
                continue
        zc = np.where(mad < 1e-9, 0.0,
                      (x[d] - med) / (mad + eps))  # zero-MAD 6.2
        d_idx.append(d)
        d_val.append(float(np.mean(_phi(_clip_z(zc)))))
    return (np.asarray(d_idx, dtype=np.intp),
            np.asarray(d_val, dtype=np.float64))


def r_context_block(block: pd.DataFrame, block_users: list[str],
                    role_map: dict,
                    features: list[str] = ALL_FEATURES,
                    min_peers: int = MIN_PEERS,
                    min_support: int = MIN_SUPPORT,
                    eps: float = EPS,
                    workers: int | None = None) -> tuple[np.ndarray, dict]:
    """R_context per row of a block (block-local strictly-past peer pools).

    Same definitions as _r_context_block_serial (the frozen reference);
    values are bit-identical (verified by tests). Implementation only:
    user->row index maps are precomputed once (single stable argsort) and
    the independent peer-pool computations run in parallel processes
    (workers: None -> min(4, cpu_count); 1 -> in-process serial).
    """
    n = len(block)
    out = np.zeros(n, dtype=np.float64)
    block = block.reset_index(drop=True)
    user_arr = block["user"].to_numpy()
    day_idx = _day_indices(block)
    x = block[features].to_numpy(dtype=np.float64)
    first_day = pd.to_datetime(block["day"]).dt.normalize().iloc[0]
    wd = _weekday_mask(day_idx, first_day)
    diag = {"neutral_users": {}, "peer_sources": {}, "peer_pool_sizes": {}}
    pg, pgsrc = peer_groups(block_users, role_map, min_peers)
    pool_members: dict[tuple, list[str]] = {}
    for u in sorted(block_users):
        peers, src = pg.get(u), pgsrc.get(u)
        diag["peer_sources"][u] = src
        if peers is None:
            mask = user_arr == u
            out[mask] = 0.5
            diag["neutral_users"][u] = src
            continue
        diag["peer_pool_sizes"][u] = len(peers)
        key = tuple(sorted(peers))
        pool_members.setdefault(key, []).append(u)
    if not pool_members:
        return out, diag
    # Precomputed user -> row indices (one stable argsort over the block).
    order = np.argsort(user_arr, kind="stable")
    u_sorted = user_arr[order]
    uniq, start, count = np.unique(u_sorted, return_index=True,
                                   return_counts=True)
    u_map = {str(uniq[i]): order[start[i]:start[i] + count[i]]
             for i in range(len(uniq))}
    tasks = []
    for key, members in pool_members.items():
        p_idx = np.concatenate([u_map[v] for v in key
                                if v in u_map])
        m_idx = np.concatenate([u_map[v] for v in members
                                if v in u_map])
        p_idx = np.sort(p_idx)  # increasing row order, as the serial isin
        m_idx = np.sort(m_idx)
        tasks.append((key, members, p_idx, m_idx))
    w = workers if workers is not None else min(4, os.cpu_count() or 1)
    if w > 1 and len(tasks) > 1:
        with multiprocessing.Pool(
                w, initializer=_ctx_init,
                initargs=(user_arr, day_idx, wd, x, min_support, eps)) as p:
            results = p.map(_ctx_pool_task, tasks, chunksize=1)
    else:
        _ctx_set_globals(user_arr, day_idx, wd, x, min_support, eps)
        results = [_ctx_pool_task(t) for t in tasks]
    for d_idx, d_val in results:
        out[d_idx] = d_val
    return out, diag


# ---------------------------------------------------------------------------
# Section 9 -- OOF cross-fit for TRAIN R_ML (entity-disjoint round-robin)
# ---------------------------------------------------------------------------
def oof_ml_scores(train_frame: pd.DataFrame, features: list[str],
                  params: dict, n_folds: int = 5,
                  num_boost_round: int = 3000,
                  early_stopping_rounds: int = 100) -> tuple[np.ndarray, dict]:
    """Out-of-fold TRAIN scores via K=5 entity-disjoint round-robin folds.

    Fold k = TRAIN users at sorted index i with i mod 5 == k. For fold k:
    fit on the other 4 folds' rows (frozen auxiliary LightGBM config,
    scale_pos_weight from those rows only), early stopping on fold
    (k+1) mod 5 AUC (patience 100, <= 3000 rounds), predict held-out fold.
    No CAL/TEST row enters any fit.
    """
    import lightgbm as lgb  # local import: Kaggle stack never needs it

    train_frame = train_frame.reset_index(drop=True)
    users = sorted(train_frame["user"].unique())
    fold_of_user = {u: i % n_folds for i, u in enumerate(users)}
    user_arr = train_frame["user"].to_numpy()
    oof = np.zeros(len(train_frame), dtype=np.float64)
    rec = {"n_users": len(users), "folds": {}, "fits": {}}
    for k in range(n_folds):
        hold = [u for u in users if fold_of_user[u] == k]
        fit = [u for u in users if fold_of_user[u] != k]
        val = [u for u in users if fold_of_user[u] == (k + 1) % n_folds]
        hold_mask = np.isin(user_arr, hold)
        fit_mask = np.isin(user_arr, fit)
        val_mask = np.isin(user_arr, val)
        Xf = train_frame.loc[fit_mask, features].astype(np.float32)
        yf = train_frame.loc[fit_mask, "is_malicious"].to_numpy().astype(int)
        Xv = train_frame.loc[val_mask, features].astype(np.float32)
        yv = train_frame.loc[val_mask, "is_malicious"].to_numpy().astype(int)
        Xh = train_frame.loc[hold_mask, features].astype(np.float32)
        p = dict(params)
        p["scale_pos_weight"] = float((yf == 0).sum() / max(1, (yf == 1).sum()))
        dtr = lgb.Dataset(Xf, label=yf)
        dva = lgb.Dataset(Xv, label=yv, reference=dtr)
        model = lgb.train(
            p, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping_rounds,
                                          verbose=False),
                       lgb.log_evaluation(0)])
        oof[hold_mask] = model.predict(Xh, num_iteration=model.best_iteration)
        inner = next(iter(model.best_score.values()))
        best_auc = float(next(iter(inner.values())))
        rec["fits"][str(k)] = {
            "fold_size_users": len(hold), "fit_users": len(fit),
            "val_fold": (k + 1) % n_folds, "val_users": len(val),
            "fit_rows": int(fit_mask.sum()), "val_rows": int(val_mask.sum()),
            "hold_rows": int(hold_mask.sum()),
            "fit_positives": int(yf.sum()), "val_positives": int(yv.sum()),
            "scale_pos_weight": p["scale_pos_weight"],
            "best_iteration": int(model.best_iteration),
            "best_score_auc": best_auc,
        }
        rec["folds"][str(k)] = sorted(hold)
    return oof, rec


# ---------------------------------------------------------------------------
# Section 8 -- fusion (SLSQP, mean log-loss, p clipped)
# ---------------------------------------------------------------------------
def _logloss_grad(w_free: np.ndarray, R: np.ndarray, y: np.ndarray,
                  cols: list[int]) -> tuple[float, np.ndarray]:
    Rf = R[:, cols]
    raw = Rf @ w_free
    p = np.clip(raw, P_CLIP_LO, P_CLIP_HI)
    ll = -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    interior = (raw > P_CLIP_LO) & (raw < P_CLIP_HI)
    g = np.zeros_like(w_free)
    if interior.any():
        denom = p[interior] * (1 - p[interior])
        g = -np.mean(Rf[interior]
                     * ((y[interior] - p[interior]) / denom)[:, None],
                     axis=0)
    return ll, g


def fit_fusion(R: np.ndarray, y: np.ndarray, free_cols: list[int],
               x0: list[float], tol: float = SLSQP_TOL,
               maxiter: int = SLSQP_MAXITER,
               warm_start: np.ndarray | None = None) -> dict:
    """Learned fusion weights for an arm (SLSQP, deterministic).

    Returns full 4-vector weights (12 significant digits), fit record.
    warm_start: optional 4-vector to initialize the solver from (bootstrap
    refits only). Mean log-loss is convex in the weights, so the optimum
    (and the rounded 12-digit weights) is independent of the starting
    point; a warm start only reduces the number of SLSQP iterations.
    """
    if warm_start is None:
        w0 = np.asarray(x0, dtype=np.float64)
    else:
        w0 = np.asarray(warm_start, dtype=np.float64).copy()
    res = minimize(
        lambda w: _logloss_grad(w, R, y, free_cols), w0, jac=True,
        method="SLSQP", bounds=[(0.0, 1.0)] * len(free_cols),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": tol, "maxiter": maxiter, "disp": False})
    w_free = np.asarray(res.x, dtype=np.float64)
    clipped = np.clip(w_free, 0.0, 1.0)
    if not np.allclose(clipped, w_free, atol=0.0, rtol=1e-15):
        clipped = clipped / clipped.sum()  # keep sum = 1 after clipping
    w_full = np.zeros(4, dtype=np.float64)
    w_full[free_cols] = clipped
    return {
        "weights": [round(float(v), 12) for v in w_full],
        "free_cols": free_cols,
        "objective_mean_logloss": float(res.fun),
        "converged": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", -1)),
        "n_eval": int(getattr(res, "nfev", -1)),
        "x0": x0,
        "solver": "scipy.optimize.minimize(method='SLSQP', jac=True, "
                  "bounds=(0,1), sum=1, ftol=1e-12, maxiter=1000)",
        "p_clip": [P_CLIP_LO, P_CLIP_HI],
    }


# ---------------------------------------------------------------------------
# Section 12 -- tie audit
# ---------------------------------------------------------------------------
def tie_audit(score: np.ndarray, n_rows: int,
              frac_caution: float = TIE_CAUTION_FRAC) -> dict:
    """Distinct values + collision rows (rows sharing a value with another)."""
    vals, counts = np.unique(score, return_counts=True)
    collisions = int(counts[counts > 1].sum())
    return {
        "n_distinct_values": int(len(vals)),
        "n_collision_rows": collisions,
        "collision_frac": collisions / max(1, n_rows),
        "flagged": collisions > frac_caution * n_rows,
        "threshold": frac_caution,
    }


# ---------------------------------------------------------------------------
# Section 14 -- per-arm evaluation (CAL threshold once; TEST once)
# ---------------------------------------------------------------------------
def evaluate_arm(arm: str, weights: np.ndarray, R_cal: np.ndarray,
                 y_cal: np.ndarray, R_test: np.ndarray, y_test: np.ndarray,
                 test_frame: pd.DataFrame, user_diag: dict) -> dict:
    s_cal = R_cal @ weights
    s_test = R_test @ weights
    t_arm, f1_cal = th.best_f1_threshold(y_cal, s_cal)
    op_test = m.binary_decision_metrics(y_test, s_test, t_arm)
    alert_test = s_test >= t_arm
    cov = p17.coverage_metrics(test_frame, alert_test, user_diag)
    cal_calib = p17.calibration_metrics(y_cal, s_cal)
    test_calib = p17.calibration_metrics(y_test, s_test)
    ranking = m.classification_metrics(y_test, s_test)
    top_k = m.top_k_metrics(y_test, s_test)
    ties = tie_audit(s_test, len(y_test))
    return {
        "arm": arm,
        "weights": [round(float(w), 12) for w in weights],
        "operating_point": {"threshold": t_arm, "cal_best_f1": f1_cal,
                            "rule": "best_f1_threshold on CAL, once, "
                                    "never TEST-informed"},
        "operating_test": {**op_test,
                           "alert_rate_per_calendar_day": float(
                               op_test["n_alerts"]
                               / test_frame["day"].nunique()),
                           "coverage": cov},
        "cal_calibration": cal_calib,
        "test_calibration": test_calib,
        "ranking": ranking,
        "top_k": top_k,
        "tie_audit": ties,
    }


# ---------------------------------------------------------------------------
# Section 7 -- redundancy / sanity diagnostics (TRAIN + CAL)
# ---------------------------------------------------------------------------
def redundancy_report(train: pd.DataFrame, cal: pd.DataFrame,
                      y_train: np.ndarray, y_cal: np.ndarray) -> dict:
    comps = COMPONENT_NAMES
    out = {}
    for block_name, block, y in (("TRAIN", train, y_train),
                                 ("CAL", cal, y_cal)):
        n = len(block)
        corr = {}
        for i, a in enumerate(comps):
            for bname in comps[i + 1:]:
                corr[f"{a}_vs_{bname}"] = float(
                    pd.Series(block[a]).corr(pd.Series(block[bname]),
                                             method="spearman"))
        pb = {f"{c}_vs_label": float(
            pd.Series(block[c]).corr(pd.Series(y), method="pearson"))
              for c in comps}
        aucs = {}
        for c in comps:
            aucs[c] = {"auc_roc": float(roc_auc_score(y, block[c])),
                       "auc_pr": float(average_precision_score(y, block[c]))}
        calib = {c: p17.calibration_metrics(y, block[c].to_numpy())
                 for c in comps}
        degen = {}
        for c in comps:
            v = block[c].to_numpy()
            within = float((np.abs(v - 0.5) <= 0.01).mean())
            degen[c] = {"frac_within_0.49_0.51": within,
                        "n_distinct": int(len(np.unique(v))),
                        "flagged": (within > 0.999
                                    or len(np.unique(v)) < 3)}
        deciles = {}
        for c in comps:
            v = block[c].to_numpy()
            pos = v[y == 1]
            neg = v[y == 0]
            qs = np.quantile(v, np.linspace(0, 1, 11))
            bin_id = np.clip(np.searchsorted(qs, v, side="right") - 1,
                             0, 9)
            deciles[c] = {
                "edges": [float(q) for q in qs],
                "n_per_decile": [
                    int((bin_id == i).sum()) for i in range(10)],
                "positive_share_per_decile": [
                    (float(y[bin_id == i].mean())
                     if (bin_id == i).any() else None)
                    for i in range(10)],
                "pos_mean": float(pos.mean()) if len(pos) else None,
                "neg_mean": float(neg.mean()) if len(neg) else None,
                "pos_median": float(np.median(pos)) if len(pos) else None,
                "neg_median": float(np.median(neg)) if len(neg) else None,
            }
        out[block_name] = {
            "rows": n, "spearman_components": corr,
            "point_biserial_vs_label": pb, "standalone_auc": aucs,
            "standalone_calibration": calib, "degenerate_flags": degen,
            "class_conditional": deciles,
        }
    return out


# ---------------------------------------------------------------------------
# Section 15 -- bootstraps (n=1000, seed 42, 90% CIs)
# ---------------------------------------------------------------------------
def user_block_bootstrap_arm(users, y, score, threshold, n_boot=N_BOOT,
                             seed=SEED, progress_cb=None) -> dict:
    """User-block bootstrap of ranking + operating metrics at fixed
    threshold (weights fixed; threshold never refit).

    progress_cb: optional callable(done, total) invoked per replicate
    (monitoring only; never changes the estimator).
    """
    users = np.asarray(users, dtype=object)
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    unique = np.unique(users)
    idx_by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rows = {metric: [] for metric in ("auc_roc", "auc_pr", "n_alerts",
                                      "precision", "recall", "f1")}
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
        if progress_cb is not None:
            progress_cb(rep, n_boot)
    out = {"n_users": int(len(unique)), "n_boot": n_boot, "seed": seed,
           "n_skipped": skipped, "threshold": threshold,
           "alpha": ALPHA,
           "method": ("user-block percentile bootstrap (resample TEST users "
                      "with replacement, pool rows, recompute; weights "
                      "fixed, threshold fixed)")}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 100 * ALPHA / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - ALPHA / 2))),
        }
    return out


def paired_user_block_deltas(users, y, score_base, score_new,
                             n_boot=N_BOOT, seed=SEED,
                             progress_cb=None) -> dict:
    """Paired user-block bootstrap of Delta(new - base) for AUC-ROC/PR
    (same resamples; same stream as user_block_bootstrap_arm when called
    with the same seed in the same order).

    progress_cb: optional callable(done, total) per replicate (monitoring
    only; never changes the estimator).
    """
    users = np.asarray(users, dtype=object)
    y = np.asarray(y, dtype=int)
    sb = np.asarray(score_base, dtype=float)
    sn = np.asarray(score_new, dtype=float)
    unique = np.unique(users)
    idx_by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rows = {"auc_roc": [], "auc_pr": []}
    skipped = 0
    for rep in range(1, n_boot + 1):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        t = y[idx]
        if t.sum() == 0 or t.sum() == len(t):
            skipped += 1
            continue
        rows["auc_roc"].append(roc_auc_score(t, sn[idx])
                               - roc_auc_score(t, sb[idx]))
        rows["auc_pr"].append(average_precision_score(t, sn[idx])
                              - average_precision_score(t, sb[idx]))
        if progress_cb is not None:
            progress_cb(rep, n_boot)
    out = {"n_users": int(len(unique)), "n_boot": n_boot, "seed": seed,
           "n_skipped": skipped, "alpha": ALPHA,
           "method": "paired user-block bootstrap of delta(new - base)"}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 100 * ALPHA / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - ALPHA / 2))),
            "frac_gt_0": float((values > 0).mean()),
        }
    return out


def weight_refit_bootstrap(train_users, R_train, y_train, R_cal, y_cal,
                           R_test, y_test, test_frame, user_diag,
                           free_cols, x0, n_boot=N_BOOT, seed=SEED,
                           warm_start: np.ndarray | None = None,
                           n_workers: int | None = None,
                           progress_cb=None) -> dict:
    """Weight-refit bootstrap: resample TRAIN users, refit fusion weights on
    the resampled OOF rows, apply to FIXED CAL/TEST component values, record
    CAL threshold + TEST metrics/Brier/ECE distributions (90% intervals).

    warm_start: optional full 4-vector (e.g. the arm's point-fit weights)
    used to initialize every refit (convex objective: same optimum). Draws
    are pre-generated in one deterministic RNG sequence; replicates are
    executed in order-preserving threads (results identical to serial).
    progress_cb: optional callable(done, total) per replicate (monitoring
    only; never changes the estimator).
    """
    train_users = np.asarray(train_users, dtype=object)
    Rtr = np.asarray(R_train, dtype=float)
    ytr = np.asarray(y_train, dtype=int)
    Rc = np.asarray(R_cal, dtype=float)
    yc = np.asarray(y_cal, dtype=int)
    Rt = np.asarray(R_test, dtype=float)
    yt = np.asarray(y_test, dtype=int)
    unique = np.unique(train_users)
    idx_by_user = {u: np.flatnonzero(train_users == u) for u in unique}
    rng = np.random.default_rng(seed)
    chosen_all = [rng.choice(unique, size=len(unique), replace=True)
                  for _ in range(n_boot)]
    if warm_start is not None:
        warm0 = np.asarray(warm_start, dtype=float)[free_cols].copy()
    else:
        warm0 = np.asarray(x0, dtype=float).copy()
    rows = {"alpha": [], "beta": [], "gamma": [], "delta": [],
            "threshold": [], "brier": [], "ece": [], "n_alerts": [],
            "precision": [], "recall": [], "f1": [], "coverage_detected": []}

    def one_rep(i: int):
        chosen = chosen_all[i]
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        try:
            fit = fit_fusion(Rtr[idx], ytr[idx], free_cols, x0,
                             warm_start=warm0)
        except Exception:
            return None
        w = np.asarray(fit["weights"], dtype=float)
        t, _ = th.best_f1_threshold(yc, Rc @ w)
        s_te = Rt @ w
        op = m.binary_decision_metrics(yt, s_te, t)
        cal = p17.calibration_metrics(yt, s_te)
        cov = p17.coverage_metrics(test_frame, s_te >= t, user_diag)
        return (w, t, cal["brier"], cal["ece"]["ece"], op["n_alerts"],
                op["precision"], op["recall"], op["f1"],
                cov["coverage"]["detected"], bool(fit["converged"]))

    workers = (n_workers if n_workers is not None
               else min(4, max(1, os.cpu_count() or 1)))
    if workers <= 1:
        results = [one_rep(i) for i in range(n_boot)]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one_rep, range(n_boot)))
    skipped = 0
    n_unconverged = 0
    for done, res in enumerate(results, start=1):
        if res is None:
            skipped += 1
            continue
        w, t, brier, ece, n_alerts, precision, recall, f1, cov_det, conv = res
        if not conv:
            n_unconverged += 1
        rows["alpha"].append(w[0])
        rows["beta"].append(w[1])
        rows["gamma"].append(w[2])
        rows["delta"].append(w[3])
        rows["threshold"].append(t)
        rows["brier"].append(brier)
        rows["ece"].append(ece)
        rows["n_alerts"].append(n_alerts)
        rows["precision"].append(precision)
        rows["recall"].append(recall)
        rows["f1"].append(f1)
        rows["coverage_detected"].append(cov_det)
        if progress_cb is not None:
            progress_cb(done, n_boot)
    out = {"n_train_users": int(len(unique)), "n_boot": n_boot,
           "seed": seed, "n_skipped": skipped,
           "n_unconverged": n_unconverged, "alpha": ALPHA,
           "free_cols": free_cols,
           "method": ("weight-refit bootstrap: resample TRAIN users with "
                      "replacement, refit fusion weights on the resampled "
                      "OOF rows (SLSQP, initialized from the arm point-fit "
                      "weights; convex objective so the optimum is "
                      "start-independent), apply to the FIXED CAL and TEST "
                      "component values, record CAL threshold and TEST "
                      "Brier/ECE/operating distributions")}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 100 * ALPHA / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - ALPHA / 2))),
            "ci_width": float(np.percentile(values, 100 * (1 - ALPHA / 2))
                              - np.percentile(values, 100 * ALPHA / 2)),
        }
    return out


def row_level_bootstrap(y, score, n_boot=N_BOOT, seed=SEED,
                        progress_cb=None) -> dict:
    """Row-level bootstrap (convention comparability only; optimistic given
    within-user correlation -- Phase 14 Section 12 convention).

    progress_cb: optional callable(done, total) per replicate (monitoring
    only; never changes the estimator).
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    rows = {"auc_roc": [], "auc_pr": []}
    skipped = 0
    n = len(y)
    idx = np.arange(n)
    for rep in range(1, n_boot + 1):
        b = rng.choice(idx, size=n, replace=True)
        t = y[b]
        if t.sum() == 0 or t.sum() == len(t):
            skipped += 1
            continue
        rows["auc_roc"].append(roc_auc_score(t, s[b]))
        rows["auc_pr"].append(average_precision_score(t, s[b]))
        if progress_cb is not None:
            progress_cb(rep, n_boot)
    out = {"n_boot": n_boot, "seed": seed, "n_skipped": skipped,
           "alpha": ALPHA,
           "note": ("row-level bootstrap reported for convention "
                    "comparability only; flagged optimistic given "
                    "within-user correlation")}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {"mean": float(values.mean()),
                       "ci_low": float(np.percentile(
                           values, 100 * ALPHA / 2)),
                       "ci_high": float(np.percentile(
                           values, 100 * (1 - ALPHA / 2)))}
    return out


# ---------------------------------------------------------------------------
# Bootstrap monitoring helpers (specification Section 15 / STAGE 12)
# ---------------------------------------------------------------------------
def _mk_prog(label: str):
    """Progress callback printing 'BOOTSTRAP <label> done/total' every 50
    replicates (monitoring only; never changes the estimator)."""
    def cb(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"BOOTSTRAP {label} {done}/{total}", flush=True)
    return cb


def _probe_bootstrap_time(arms, R, y_train, y_cal, y_te, train_frame,
                          test_frame, user_diag, fits) -> None:
    """STAGE 12 pre-flight timing probe: run each bootstrap family at a
    small fixed n_boot=10 with seed 9999 (distinct from the real seed-42
    streams), then print estimated full n=1000 runtime per run. Monitoring
    only: changes no protocol, no artifact, no RNG stream. Timings are
    intentionally NOT stored in the result dict (they are wall-clock, so
    they must not enter the byte-identical determinism gate).
    """
    import time
    import tracemalloc
    n_small = 10
    test_users = test_frame["user"].to_numpy()
    y = np.asarray(y_te, dtype=int)
    sA = arms["A"]["scores_test"]
    est = {}
    tracemalloc.start()
    t0 = time.perf_counter()
    for a in arms.values():
        user_block_bootstrap_arm(test_users, y, a["scores_test"],
                                 a["operating_point"]["threshold"],
                                 n_boot=n_small, seed=9999)
    est["user_block"] = (time.perf_counter() - t0) / n_small * 1000 * len(arms)
    t0 = time.perf_counter()
    for arm in ("B", "C", "C1", "C2", "C3"):
        paired_user_block_deltas(test_users, y, sA, arms[arm]["scores_test"],
                                 n_boot=n_small, seed=9999)
    est["paired_deltas"] = (time.perf_counter() - t0) / n_small * 1000 * 5
    t0 = time.perf_counter()
    for arm in AUXILIARY_ARMS + ("A",):
        row_level_bootstrap(y, arms[arm]["scores_test"], n_boot=n_small,
                            seed=9999)
    est["row_level"] = (time.perf_counter() - t0) / n_small * 1000 * (
        len(AUXILIARY_ARMS) + 1)
    t0 = time.perf_counter()
    for arm in LEARNED_ARMS:
        weight_refit_bootstrap(
            train_frame["user"].to_numpy(), R["TRAIN"], y_train,
            R["CAL"], y_cal, R["TEST"], y, test_frame, user_diag,
            ARMS[arm]["free_cols"], ARMS[arm]["x0"], n_boot=n_small,
            seed=9999, warm_start=np.asarray(fits[arm]["weights"],
                                             dtype=float))
    est["weight_refit"] = (time.perf_counter() - t0) / n_small * 1000 * len(
        LEARNED_ARMS)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total = sum(est.values())
    print(f"BOOTSTRAP-EST probe n={n_small} -> estimated n=1000 (min per "
          f"run): user_block={est['user_block']/60:.1f} "
          f"paired={est['paired_deltas']/60:.1f} "
          f"row_level={est['row_level']/60:.1f} "
          f"weight_refit={est['weight_refit']/60:.1f} "
          f"TOTAL={total/60:.1f} (x3 runs ~{total*3/60:.1f}) "
          f"probe_peak_python_mem={peak/1e6:.0f} MiB (tracemalloc)",
          flush=True)


# ---------------------------------------------------------------------------
# Gates (specification Section 17)
# ---------------------------------------------------------------------------
def gate1_arm_a_reproduction(test: pd.DataFrame) -> dict:
    """Gate 1: frozen-reference reproduction on the frozen TEST artifact."""
    y = test["is_malicious"].to_numpy().astype(int)
    s = test["score"].to_numpy().astype(float)
    cls = m.classification_metrics(y, s)
    op = m.binary_decision_metrics(y, s, FROZEN_THRESHOLD)
    topk = m.top_k_metrics(y, s)
    cross = int((test["alert_primary"].to_numpy().astype(bool)
                 != (s >= FROZEN_THRESHOLD)).sum())
    checks = {
        "auc_roc": _close(cls["auc_roc"], RECORDED_TEST_AUC_ROC),
        "auc_pr": _close(cls["auc_pr"], RECORDED_TEST_AUC_PR),
        "n_alerts": int(op["n_alerts"]) == RECORDED_PRIMARY["n_alerts"],
        "precision": _close(op["precision"], RECORDED_PRIMARY["precision"]),
        "recall": _close(op["recall"], RECORDED_PRIMARY["recall"]),
        "f1": _close(op["f1"], RECORDED_PRIMARY["f1"]),
        "mcc": _close(op["mcc"], RECORDED_PRIMARY["mcc"]),
        "balanced_accuracy": _close(op["balanced_accuracy"],
                                    RECORDED_PRIMARY["balanced_accuracy"]),
        "top_k": all(_close(topk[k], RECORDED_TOP_K[k])
                     for k in RECORDED_TOP_K),
    }
    return {
        "threshold": FROZEN_THRESHOLD,
        "recomputed": {"classification": cls, "at_primary_threshold": op,
                       "top_k": topk,
                       "alert_primary_column_mismatches": cross},
        "checks": checks,
        "passed": all(checks.values()) and cross == 0,
        "tolerance_note": "1e-12 relative for floats; exact integers",
        "reference": "phase14_experiment.json test_metrics "
                     "(md5-verified input)",
    }


def gate2_split_integrity(split: dict, train_frame: pd.DataFrame,
                          cal_frame: pd.DataFrame,
                          test_frame: pd.DataFrame,
                          oof_rec: dict) -> dict:
    """Gate 2: allocation verbatim; block counts; user disjointness; folds."""
    alloc = split["allocation"]
    counts = {
        "TRAIN": {"users": len(alloc["TRAIN"]),
                  "rows": len(train_frame),
                  "positives": int(train_frame["is_malicious"].sum())},
        "CAL": {"users": len(alloc["CAL"]),
                "rows": len(cal_frame),
                "positives": int(cal_frame["is_malicious"].sum())},
        "TEST": {"users": len(alloc["TEST"]),
                 "rows": len(test_frame),
                 "positives": int(test_frame["is_malicious"].sum())},
    }
    checks = {
        "train_users": counts["TRAIN"]["users"] == 784,
        "cal_users": counts["CAL"]["users"] == 108,
        "test_users": counts["TEST"]["users"] == 108,
        "train_rows": counts["TRAIN"]["rows"] == 392_784,
        "cal_rows": counts["CAL"]["rows"] == 54_108,
        "test_rows": counts["TEST"]["rows"] == 54_108,
        "train_positives": counts["TRAIN"]["positives"] == 1037,
        "cal_positives": counts["CAL"]["positives"] == 379,
        "test_positives": counts["TEST"]["positives"] == 476,
        "disjoint": (set(alloc["TRAIN"]) & set(alloc["CAL"]) == set()
                     and set(alloc["TRAIN"]) & set(alloc["TEST"]) == set()
                     and set(alloc["CAL"]) & set(alloc["TEST"]) == set()),
        "frame_user_sets_match_allocation": (
            set(train_frame["user"].unique()) == set(alloc["TRAIN"])
            and set(cal_frame["user"].unique()) == set(alloc["CAL"])
            and set(test_frame["user"].unique()) == set(alloc["TEST"])),
        "all_users_present": (set(alloc["TRAIN"]) | set(alloc["CAL"])
                              | set(alloc["TEST"])) == set(
                                  train_frame["user"].unique())
                              | set(cal_frame["user"].unique())
                              | set(test_frame["user"].unique()),
    }
    fold_checks = {}
    all_fold_users = []
    for k in sorted(oof_rec["folds"]):
        fu = set(oof_rec["folds"][k])
        all_fold_users.extend(fu)
        fold_checks[f"fold_{k}_disjoint"] = all(
            not (fu & set(oof_rec["folds"][j]))
            for j in sorted(oof_rec["folds"]) if j != k)
    fold_checks["folds_cover_train"] = set(all_fold_users) == set(
        alloc["TRAIN"])
    checks.update(fold_checks)
    return {"counts": counts, "checks": checks,
            "passed": all(checks.values()),
            "reference": "phase14_split.json (md5-verified input)"}


def gate5_component_sanity(components: dict) -> dict:
    """Gate 5: components finite, in [0,1], null-free; provenance asserted."""
    checks = {}
    for block_name, block in components.items():
        for c in COMPONENT_NAMES:
            v = block[c].to_numpy()
            checks[f"{block_name}_{c}_finite"] = bool(np.isfinite(v).all())
            checks[f"{block_name}_{c}_in_01"] = bool(
                (v >= 0.0).all() and (v <= 1.0).all())
            checks[f"{block_name}_{c}_null_free"] = bool(
                not block[c].isna().any())
    return {"checks": checks, "passed": all(checks.values())}


def gate6_weight_isolation(fits: dict) -> dict:
    """Gate 6: structural isolation record (fit inputs = TRAIN-OOF only;
    CAL used only for thresholds; TEST read only at the final stage)."""
    used = []
    for arm in LEARNED_ARMS:
        used.append(fits[arm]["free_cols"])
    return {
        "weight_fits_on_train_oof_only": True,
        "cal_used_only_for_thresholds": True,
        "test_read_only_at_final_stage": True,
        "checks": {"learned_arms": list(LEARNED_ARMS),
                   "structural": "enforced by run_phase18 stage order and "
                                 "asserted in tests (test_phase18.py)"},
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Section 16 -- mechanical verdict
# ---------------------------------------------------------------------------
def compute_verdict(gates: dict, arms: dict, boot: dict,
                    determinism_ok: bool) -> dict:
    """Mechanical PASS/CAUTION/FAIL from the pre-registered criteria.

    Clause conflict resolution (recorded): specification Section 16 lists
    'any auxiliary arm materially degrades' under FAIL as an absolute
    'any of' clause, so FAIL takes precedence over CAUTION (c); CAUTION (c)
    is then unreachable and is recorded as such.
    """
    gates_ok = all(g["passed"] for g in gates.values())

    def meets_bundle(arm: str) -> bool:
        a = arms[arm]
        d = a["deltas_vs_arm_a"]
        op = a["operating_test"]
        cal = a["test_calibration"]
        ties = a["tie_audit"]
        return (d["auc_pr"]["point"] >= PASS_DPR
                and d["auc_pr"]["ci_low"] > 0.0
                and d["auc_roc"]["point"] >= 0.0
                and d["auc_roc"]["ci_low"] >= PASS_DR_CI_LO
                and op["f1"] >= PASS_F1
                and op["precision"] >= PASS_PRECISION
                and op["n_alerts"] <= PASS_ALERTS
                and op["mcc"] >= PASS_MCC
                and op["coverage"]["coverage"]["detected"] >= PASS_COVERAGE
                and cal["brier"] <= PASS_BRIER
                and cal["ece"] <= PASS_ECE
                and not ties["flagged"])

    def degrades(arm: str) -> bool:
        a = arms[arm]
        d = a["deltas_vs_arm_a"]
        op = a["operating_test"]
        return (d["auc_roc"]["point"] < FAIL_DR
                or d["auc_pr"]["point"] < FAIL_DPR
                or op["n_alerts"] > FAIL_ALERTS_MULT * RECORDED_PRIMARY[
                    "n_alerts"]
                or op["f1"] < RECORDED_PRIMARY["f1"] - FAIL_F1_GAP)

    bundle_arms = [a for a in AUXILIARY_ARMS if meets_bundle(a)]
    degraded_arms = [a for a in AUXILIARY_ARMS if degrades(a)]
    collapse = any(arms[a]["weights"][0] >= COLLAPSE_ALPHA
                   for a in LEARNED_ARMS) and not bundle_arms
    weight_evidence = {}
    for a in bundle_arms:
        w = arms[a]["weights"]
        aux = 1.0 - w[0]
        ci = boot["weight_refit"][a]["one_minus_alpha"]
        learned_idx = set(ARMS[a]["free_cols"] or [])
        stable = all(boot["weight_refit"][a]["ci_width"][k] <=
                     PASS_WEIGHT_CI_WIDTH
                     for k in COMPONENT_NAMES
                     if COMPONENT_NAMES.index(k) in learned_idx)
        weight_evidence[a] = {
            "one_minus_alpha": aux,
            "one_minus_alpha_ci_low": ci["ci_low"],
            "one_minus_alpha_ci_high": ci["ci_high"],
            "aux_mass_ok": aux >= PASS_AUX_MASS and ci["ci_low"] > 0.0,
            "weights_stable": stable,
        }
    pass_arms = [a for a in bundle_arms
                 if weight_evidence.get(a, {}).get("aux_mass_ok", False)
                 and weight_evidence[a]["weights_stable"]]

    criteria = {
        "gates_pass": gates_ok,
        "bundle_arms": bundle_arms,
        "degraded_arms": degraded_arms,
        "auxiliary_weight_evidence": weight_evidence,
        "pass_arms": pass_arms,
        "ml_only_collapse": collapse,
        "determinism_ok": bool(determinism_ok),
        "conflict_resolution_note": (
            "FAIL 'any auxiliary arm materially degrades' clause takes "
            "precedence over CAUTION (c) per Section 16 FAIL 'any of' "
            "semantics; CAUTION (c) unreachable"),
    }

    if not gates_ok or not determinism_ok:
        verdict = "FAIL"
        reason = "hard gate(s) failed or determinism violated"
    elif degraded_arms:
        verdict = "FAIL"
        reason = f"material degradation: {sorted(degraded_arms)}"
    elif collapse:
        verdict = "FAIL"
        reason = ("ML-only collapse: learned alpha >= 0.95 and no arm meets "
                  "the improvement bundle (pre-registered negative result)")
    elif pass_arms:
        verdict = "PASS"
        reason = f"improvement bundle + weight evidence: {sorted(pass_arms)}"
    elif bundle_arms:
        verdict = "CAUTION"
        reason = ("bundle met but improvement not attributable to "
                  "auxiliaries (weight CI includes 0) or weight CI width "
                  "> 0.20")
    else:
        verdict = "CAUTION"
        reason = ("no measurable improvement and no material degradation "
                  "(pre-registered 'no measurable improvement' result)")
    return {"verdict": verdict, "reason": reason, "criteria": criteria,
            "bounds": {
                "pass_delta_auc_pr": PASS_DPR,
                "pass_delta_auc_roc_ci_low": PASS_DR_CI_LO,
                "pass_f1": PASS_F1, "pass_precision": PASS_PRECISION,
                "pass_mcc": PASS_MCC, "pass_alerts": PASS_ALERTS,
                "pass_coverage": PASS_COVERAGE,
                "pass_brier": PASS_BRIER, "pass_ece": PASS_ECE,
                "pass_aux_mass": PASS_AUX_MASS,
                "pass_weight_ci_width": PASS_WEIGHT_CI_WIDTH,
                "fail_delta_auc_roc": FAIL_DR,
                "fail_delta_auc_pr": FAIL_DPR,
                "fail_alerts": FAIL_ALERTS_MULT * RECORDED_PRIMARY[
                    "n_alerts"],
                "fail_f1": RECORDED_PRIMARY["f1"] - FAIL_F1_GAP,
                "collapse_alpha": COLLAPSE_ALPHA,
                "tie_caution_frac": TIE_CAUTION_FRAC,
            },
            "note": ("PASS/CAUTION/FAIL computed mechanically from the "
                     "pre-registered specification Section 16 criteria; "
                     "research evidence only, no production decision")}


# ---------------------------------------------------------------------------
# Section 19 -- diagnostics (post-verdict)
# ---------------------------------------------------------------------------
def diagnostic_users_report(test_frame: pd.DataFrame, components: dict,
                            arms: dict, user_diag: dict) -> dict:
    """JJM0203 / WDD0366 per-arm profiles; zero-alert lists; correlations."""
    test = components["TEST"]
    per = {d["user"]: d for d in user_diag["per_user"]}
    diag_users = [u for u in ("JJM0203", "WDD0366")
                  if u in set(test["user"])]
    out = {"users": {}}
    for u in diag_users:
        mask = test["user"].to_numpy() == u
        prof = {c: float(test.loc[mask, c].max()) for c in COMPONENT_NAMES}
        prof["r_ml_max_phase14"] = per.get(u, {}).get("max_score")
        arms_prof = {}
        for arm, a in arms.items():
            s = arms[arm]["scores_test"]
            arms_prof[arm] = {
                "max_score": float(s[mask].max()),
                "alerted": bool((s[mask] >= arms[arm]["operating_point"][
                    "threshold"]).any()),
            }
        out["users"][u] = {"component_profile": prof, "arms": arms_prof}
    zero_lists = {}
    for arm, a in arms.items():
        alert = a["scores_test"] >= a["operating_point"]["threshold"]
        zero_lists[arm] = sorted(
            u for u in per
            if per[u]["n_positive_days"] > 0
            and not (test["user"].to_numpy()[alert] == u).any())
    # component-vs-score correlations on TEST (per arm)
    corr = {}
    for arm, a in arms.items():
        s = arms[arm]["scores_test"]
        corr[arm] = {f"r_final_vs_{c}": float(
            pd.Series(s).corr(pd.Series(test[c]), method="spearman"))
            for c in COMPONENT_NAMES}
    corr["r_ml_vs_aux"] = {f"r_ml_vs_{c}": float(
        pd.Series(test["r_ml"]).corr(pd.Series(test[c]),
                                     method="spearman"))
        for c in COMPONENT_NAMES[1:]}
    return {"users": out["users"], "zero_alert_users_per_arm": zero_lists,
            "correlations_test": corr}


# ---------------------------------------------------------------------------
# Assembler (IO-free, deterministic)
# ---------------------------------------------------------------------------
def run_phase18(merged: pd.DataFrame, split: dict, cal: pd.DataFrame,
                test: pd.DataFrame, user_diag: dict, role_df: pd.DataFrame,
                gates_env: dict, n_boot: int = N_BOOT) -> dict:
    """Assemble the full Phase 18 result (deterministic).

    gates_env (computed by the runner):
      gate4_input_md5: dict passed/md5s; gate7_determinism: bool;
      gate3_tests: {"passed": bool, "n_passed": int, "n_failed": int};
      dataset_id, environment, kaggle_note, role_build_record: dict.
    """
    # ---------------------------------------------------------------- gates 2/1
    alloc = split["allocation"]
    merged = merged.sort_values(["user", "day"]).reset_index(drop=True)
    train_frame = merged[merged["user"].isin(alloc["TRAIN"])].copy()
    cal_frame = merged[merged["user"].isin(alloc["CAL"])].copy()
    test_frame = merged[merged["user"].isin(alloc["TEST"])].copy()
    cal = cal.sort_values(["user", "day"]).reset_index(drop=True)
    test = test.sort_values(["user", "day"]).reset_index(drop=True)
    for name, frame, src in (("CAL", cal_frame, cal),
                             ("TEST", test_frame, test)):
        key = frame[["user", "day"]].merge(
            src[["user", "day"]], on=["user", "day"], how="left",
            indicator=True)
        assert (key["_merge"] == "both").all(), f"{name} alignment failed"
        assert len(frame) == len(src), f"{name} row count mismatch"
        assert (frame["user"].to_numpy() == src["user"].to_numpy()).all()
        assert (pd.to_datetime(frame["day"]).to_numpy()
                == pd.to_datetime(src["day"]).to_numpy()).all()

    # ------------------------------------------------------- role map (static)
    role_map = {str(r["user_id"]): (r["role"], r["department"])
                for _, r in role_df.iterrows()}
    role_diag = {
        "n_users": len(role_map),
        "n_role_null": int(role_df["role"].isna().sum()),
        "n_department_null": int(role_df["department"].isna().sum()),
        "n_distinct_roles": int(role_df["role"].nunique()),
        "n_distinct_departments": int(role_df["department"].nunique()),
    }

    # ---------------------------------------------- components (one pass, all
    # blocks, before any fit; fixed (user, day) order)
    components = {}
    trust_flags = {}
    context_diag = {}
    for name, frame in (("TRAIN", train_frame), ("CAL", cal_frame),
                        ("TEST", test_frame)):
        users = sorted(frame["user"].unique())
        rb = r_behavior_block(frame)
        rt, flags = r_trust_block(frame, train_frame)
        rc, cdiag = r_context_block(frame, users, role_map)
        trust_flags[name] = flags
        context_diag[name] = cdiag
        components[name] = pd.DataFrame({
            "user": frame["user"].to_numpy(),
            "day": frame["day"].to_numpy(),
            "r_trust": rt, "r_context": rc, "r_behavior": rb,
        })
    # R_ML provenance: TRAIN = OOF (below); CAL/TEST = frozen artifacts
    y_cal = cal["is_malicious"].to_numpy().astype(int)
    y_test = test["is_malicious"].to_numpy().astype(int)
    components["CAL"]["r_ml"] = cal["score"].to_numpy().astype(float)
    components["TEST"]["r_ml"] = test["score"].to_numpy().astype(float)
    assert (components["TEST"]["user"].to_numpy()
            == test["user"].to_numpy()).all()

    # ------------------------------------------------------- OOF (TRAIN only)
    params = {"objective": "binary", "metric": "auc",
              "learning_rate": 0.03, "num_leaves": 31,
              "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1,
              "verbose": -1, "seed": 42}
    oof, oof_rec = oof_ml_scores(train_frame, ALL_FEATURES, params)
    components["TRAIN"]["r_ml"] = oof
    y_train = train_frame["is_malicious"].to_numpy().astype(int)
    gate2 = gate2_split_integrity(split, train_frame, cal_frame, test_frame,
                                  oof_rec)

    # --------------------------------------------------- R matrix per block
    R = {name: components[name][COMPONENT_NAMES].to_numpy(dtype=np.float64)
         for name in ("TRAIN", "CAL", "TEST")}

    # ---------------------------------------------------- redundancy (TRAIN
    # + CAL only; informs the report, never the protocol)
    redundancy = redundancy_report(
        components["TRAIN"], components["CAL"], y_train, y_cal)

    # ------------------------------------------------------- fusion fits
    fits = {}
    for arm in LEARNED_ARMS:
        fits[arm] = fit_fusion(R["TRAIN"], y_train,
                               ARMS[arm]["free_cols"], ARMS[arm]["x0"])

    # ------------------------------------------------------- arms evaluation
    y_te = y_test
    arms = {}
    for arm in ("A", "B", "C", "C1", "C2", "C3"):
        w = (ARMS[arm]["fixed"] if ARMS[arm]["free_cols"] is None
             else np.asarray(fits[arm]["weights"], dtype=float))
        arms[arm] = evaluate_arm(
            arm, w, R["CAL"], y_cal, R["TEST"], y_te, test_frame,
            user_diag)
        arms[arm]["scores_test"] = R["TEST"] @ w
    # deltas vs arm A (threshold-free; paired user-block CIs)
    for arm in ("B", "C", "C1", "C2", "C3"):
        a = arms[arm]
        sA = arms["A"]["scores_test"]
        sX = a["scores_test"]
        pa = paired_user_block_deltas(
            test["user"].to_numpy(), y_te, sA, sX, n_boot=n_boot,
            progress_cb=_mk_prog(f"deltas {arm}"))
        a["deltas_vs_arm_a"] = {
            "auc_roc": {"point": float(a["ranking"]["auc_roc"]
                                       - RECORDED_TEST_AUC_ROC),
                        **pa["auc_roc"]},
            "auc_pr": {"point": float(a["ranking"]["auc_pr"]
                                      - RECORDED_TEST_AUC_PR),
                       **pa["auc_pr"]},
        }
    arms["A"]["deltas_vs_arm_a"] = {
        "auc_roc": {"point": 0.0, "note": "reference arm"},
        "auc_pr": {"point": 0.0, "note": "reference arm"},
    }

    # ------------------------------------------------------- gates 1/5/6
    gate1 = gate1_arm_a_reproduction(test)
    gate5 = gate5_component_sanity(components)
    gate6 = gate6_weight_isolation(fits)
    gate4 = {
        "passed": bool(gates_env["gate4_input_md5"]["passed"]),
        "input_md5s": gates_env["gate4_input_md5"]["files"],
        "role_build_record": gates_env["role_build_record"],
        "role_build_gates_passed": bool(
            gates_env["role_build_record"].get("gates", {}).get("passed",
                                                                False)),
    }
    gate4["passed"] = gate4["passed"] and gate4["role_build_gates_passed"]
    gate3 = {"passed": bool(gates_env["gate3_tests"]["passed"]),
             "tests": gates_env["gate3_tests"]}
    gates = {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4,
             "gate5": gate5, "gate6": gate6}

    # ------------------------------------------------------- calibration
    calibration = {"arms": {}}
    for arm, a in arms.items():
        calibration["arms"][arm] = {
            "threshold": a["operating_point"]["threshold"],
            "cal_best_f1": a["operating_point"]["cal_best_f1"],
            "cal_calibration": a["cal_calibration"],
            "test_calibration": a["test_calibration"],
            "rule": a["operating_point"]["rule"],
        }
    calibration["note"] = ("descriptive Brier/ECE of the raw fused score "
                           "(no calibration transform in the protocol; "
                           "Phase 17 machinery not reused)")

    # ------------------------------------------------------- ranking record
    ranking_verification = {}
    for arm, a in arms.items():
        ranking_verification[arm] = {
            "classification": a["ranking"], "top_k": a["top_k"],
            "deltas_vs_arm_a": a["deltas_vs_arm_a"],
            "tie_audit": a["tie_audit"],
        }

    # ------------------------------------------------------- coverage
    coverage = {}
    for arm, a in arms.items():
        alert = a["scores_test"] >= a["operating_point"]["threshold"]
        cov = a["operating_test"]["coverage"]
        coverage[arm] = {
            "coverage": cov["coverage"],
            "wilson_90": cov["coverage"]["wilson_90"],
            "detected_users": cov["detected_users"],
            "zero_alert_users": cov["zero_alert_users"],
            "fully_detected_users": cov["fully_detected_users"],
            "per_user_alerts": cov["per_user_alerts"],
            "alerts_per_user": cov["alerts_per_user"],
            "alert_concentration": cov["alert_concentration"],
            "by_strata": p17.coverage_by_strata(test_frame, alert,
                                                user_diag),
            "scenario_recall": p17.scenario_recall(test_frame, alert,
                                                   user_diag),
        }

    # ------------------------------------------------------- bootstrap
    boot = {"n_boot": n_boot, "seed": SEED, "alpha": ALPHA}
    # Timing probe (monitoring only; never changes the protocol): a small
    # fixed number of repetitions with a DIFFERENT seed (9999) so the real
    # seed-42 streams are untouched. Estimates are printed to stdout (the
    # notebook live log) and intentionally NOT stored in the result dict,
    # so the byte-identical determinism gate is unaffected.
    if n_boot >= 100:
        try:
            _probe_bootstrap_time(arms, R, y_train, y_cal, y_te,
                                  train_frame, test_frame, user_diag, fits)
        except Exception as exc:  # noqa: BLE001
            print(f"BOOTSTRAP-EST probe skipped: {exc}", flush=True)
    boot["user_block"] = {}
    for arm, a in arms.items():
        boot["user_block"][arm] = user_block_bootstrap_arm(
            test["user"].to_numpy(), y_te, a["scores_test"],
            a["operating_point"]["threshold"], n_boot=n_boot,
            progress_cb=_mk_prog(f"user_block {arm}"))
    boot["weight_refit"] = {}
    for arm in LEARNED_ARMS:
        w_point = np.asarray(fits[arm]["weights"], dtype=float)
        br = weight_refit_bootstrap(
            train_frame["user"].to_numpy(), R["TRAIN"], y_train,
            R["CAL"], y_cal, R["TEST"], y_te, test_frame, user_diag,
            ARMS[arm]["free_cols"], ARMS[arm]["x0"], n_boot=n_boot,
            warm_start=w_point, progress_cb=_mk_prog(f"weight_refit {arm}"))
        w = w_point
        br["one_minus_alpha"] = {
            "point": float(1.0 - w[0]),
            "ci_low": float(1.0 - br["alpha"]["ci_high"]),
            "ci_high": float(1.0 - br["alpha"]["ci_low"]),
        }
        br["ci_width"] = {k: float(br[{"r_ml": "alpha", "r_trust": "beta",
                                       "r_context": "gamma",
                                       "r_behavior": "delta"}[k]]["ci_width"])
                          for k in COMPONENT_NAMES}
        boot["weight_refit"][arm] = br
    boot["row_level"] = {}
    for arm in AUXILIARY_ARMS:
        boot["row_level"][arm] = row_level_bootstrap(
            y_te, arms[arm]["scores_test"], n_boot=n_boot,
            progress_cb=_mk_prog(f"row_level {arm}"))
    boot["row_level"]["A"] = row_level_bootstrap(
        y_te, arms["A"]["scores_test"], n_boot=n_boot,
        progress_cb=_mk_prog("row_level A"))

    # ------------------------------------------------------- verdict
    verdict = compute_verdict(gates, arms, boot,
                              determinism_ok=bool(
                                  gates_env["determinism_ok"]))

    # ------------------------------------------------------- diagnostics
    diagnostics = diagnostic_users_report(test_frame, components, arms,
                                          user_diag)

    return {
        "experiment_id": "phase18-adaptive-risk-v2",
        "phase": "18",
        "dataset": gates_env["dataset_id"],
        "scope": ("research evaluation of a leakage-safe adaptive risk "
                  "formulation on the user-disjoint Phase 14/17 framework; "
                  "no training beyond the pre-registered TRAIN OOF "
                  "cross-fits and fusion weights (evaluation-only); no "
                  "production change"),
        "specification": "docs/phase18_specification.md",
        "cohort": {
            "framework": "user-disjoint Phase 14/17 (never conflated with "
                         "the chronological TEST)",
            "blocks": {"TRAIN": {"users": 784, "rows": 392_784,
                                 "positives": 1037, "malicious": 40},
                       "CAL": {"users": 108, "rows": 54_108,
                               "positives": 379, "malicious": 15},
                       "TEST": {"users": 108, "rows": 54_108,
                                "positives": 476, "malicious": 15}},
        },
        "role_diagnostics": role_diag,
        "context_peer_diagnostics": context_diag,
        "redundancy": redundancy,
        "components": {
            "definitions": ("Section 6 of the specification; component "
                            "values in phase18_components.parquet"),
            "trust_ecdf_flags": trust_flags,
            "degenerate_flags": redundancy["TRAIN"]["degenerate_flags"],
        },
        "components_table": components,
        "oof": oof_rec,
        "fusion_fits": {arm: {k: v for k, v in fits[arm].items()
                              if k != "weights"} | {"weights": fits[arm][
                                  "weights"]}
                        for arm in LEARNED_ARMS},
        "gates": gates,
        "arms": {arm: {k: v for k, v in arms[arm].items()
                       if k != "scores_test"}
                 for arm in arms},
        "calibration": calibration,
        "ranking_verification": ranking_verification,
        "coverage": coverage,
        "bootstrap": boot,
        "verdict": verdict,
        "diagnostics": diagnostics,
        "expectations": {
            "hypotheses": [
                ("R_trust redundant with R_ML (Phase 8 measured +0.63 "
                 "Spearman for a similar component); C2 may not beat A "
                 "(HYPOTHESIS, measured)"),
                ("JJM0203/WDD0366 remain zero-alert in every arm "
                 "(HYPOTHESIS; any alert is an OBSERVED surprise)"),
            ],
        },
        "frozen_system": {
            "model": "lgbm-graph-v1",
            "threshold": FROZEN_THRESHOLD,
            "seed": 42,
            "best_iteration_frozen": 186,
            "untouched": True,
        },
        "test_evaluation_policy": (
            "the authoritative chronological TEST is never read or scored; "
            "the user-disjoint TEST is read once from the frozen Phase 14 "
            "predictions artifact; TEST evaluated once (the deterministic "
            "double run is a determinism check, not a second evaluation)"),
        "evidence_labels": {
            "gates": "OBSERVED (recomputed from md5-verified frozen "
                     "artifacts)",
            "components": "OBSERVED (computed from frozen inputs per "
                          "specification Section 6)",
            "arms": "OBSERVED (CAL thresholds applied once, TEST once)",
            "verdict": "pre-registered criteria applied mechanically",
        },
        "determinism": {"runs": 2, "bit_identical": None,
                        "note": "filled by the runner after the "
                                "byte-identical double-run gate"},
        "environment": gates_env["environment"],
        "kaggle_note": gates_env["kaggle_note"],
    }


def json_safe(obj: Any) -> Any:
    return to_jsonable(obj)


def canonical(obj: Any) -> str:
    return canonical_json(obj)