"""Tests for Phase 18 (leakage-safe adaptive risk v2).

Covers (authorization Section 20 list): strictly-past behavioral deviation
(no future leakage), future-invariance, context peer cutoffs and cold
start, unseen/missing context, trust normalization (ECDF on TRAIN block
only, clamped domain), fusion constraints (non-negative weights, sum = 1,
deterministic SLSQP), ML-only reproduction (arm A == frozen scores; gate 1
reproduces the recorded Phase 14 values on the real artifact), threshold
determinism, evaluation isolation (TEST never enters any fit), frozen
artifact immutability (gate 4 md5s), bootstrap determinism, diagnostic
users absent from TRAIN fits, hand-computed synthetic fixtures, and a
small end-to-end assembler run (structure + determinism).

The authoritative chronological TEST never enters this module; only the
frozen user-disjoint TEST predictions artifact is read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from src import config
from src.experiments import phase18 as p18

ART = Path(config.ARTIFACTS_DIR)

EXPECTED_MD5 = {
    "phase6_merged_features.parquet":
        "9a3b188573bb953416981dfea3379def",
    "phase14_split.json": "da013f825246d568bcfdaf19dcf2c7e7",
    "phase15_cal_scores.parquet": "c955a4ec6ecaf7e9baff80abeb361330",
    "phase14_test_predictions.parquet":
        "71eeb3f1948e518518a53e062d5a213d",
    "phase14_user_diagnostics.json":
        "cd893661a3a0c08f9066e14c0bf8dd07",
    "phase18_role_department.parquet":
        "c1a684ed6152fbef7d1b45bb1291079f",
}
MODEL_TXT_MD5 = "3778a4d869f7231e76ec4c08c6dfa419"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Hand-computed fixtures: strictly-past behavioral z
# ---------------------------------------------------------------------------
def test_expanding_z_min_history_and_zero_mad():
    X = np.array([[1.0, 1.0, 1.0, 10.0]])
    z = p18.expanding_strictly_past_z(X, min_history=2)
    assert z.shape == (1, 4)
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0       # |H| < 2
    assert z[0, 2] == 0.0 and z[0, 3] == 0.0       # zero-MAD -> 0


def test_expanding_z_hand_computed_value():
    X = np.array([[0.0, 4.0, 8.0, 100.0, 0.0]])
    z = p18.expanding_strictly_past_z(X, min_history=1)
    assert z[0, 1] == 0.0  # |H| = 1, MAD = 0 -> 0
    z_val = (8.0 - 2.0) / (2.0 + 1e-6)             # H=[0,4]: med 2, MAD 2
    assert abs(z[0, 2] - z_val) < 1e-12
    # day 3 (idx 3): H=[0,4,8] -> med 4, MAD = median(4,0,4) = 4
    # -> (100-4)/(4+eps); NOT zero-MAD (a genuinely zero-MAD case with
    # min_history=1 and all-equal history is covered in the previous test)
    assert abs(z[0, 3] - (100.0 - 4.0) / (4.0 + 1e-6)) < 1e-12
    # non-zero-MAD: H=[0,8] (min_history=1, day 3 of [0, 8, 20, ...])
    X3 = np.array([[0.0, 8.0, 20.0, 40.0]])
    z3 = p18.expanding_strictly_past_z(X3, min_history=1)
    # day 2 (idx 2): H=[0,8] med 4, MAD 4 -> (20-4)/(4+eps)
    assert abs(z3[0, 2] - (20.0 - 4.0) / (4.0 + 1e-6)) < 1e-12
    # day 3 (idx 3): H=[0,8,20] med 8, MAD 8 -> (40-8)/(8+eps)
    assert abs(z3[0, 3] - (40.0 - 8.0) / (8.0 + 1e-6)) < 1e-12


def test_expanding_z_future_invariance():
    """A spike in the future must not change any past z (strictly-past)."""
    base = np.arange(30, dtype=float).reshape(1, -1)
    z_base = p18.expanding_strictly_past_z(base, min_history=5)
    spike = base.copy()
    spike[0, 29] = 1e6
    z_spike = p18.expanding_strictly_past_z(spike, min_history=5)
    assert np.array_equal(z_base[0, :29], z_spike[0, :29])


# ---------------------------------------------------------------------------
# ECDF trust normalization
# ---------------------------------------------------------------------------
def test_ecdf_midpoint_rank_and_ties():
    tv = np.array([1.0, 2.0, 3.0, 4.0])
    p = p18.ecdf_percentile(tv, np.array([2.5, -100.0, 100.0]), n_train=4)
    assert abs(p[0] - 0.375) < 1e-12        # rank 2 -> (2-0.5)/4
    assert p[1] == 0.0                      # below min -> clamped 0
    assert abs(p[2] - (4 - 0.5) / 4) < 1e-12  # above max -> rank n
    p_tie = p18.ecdf_percentile(tv, np.array([2.0]), n_train=4)
    assert abs(p_tie[0] - 0.25) < 1e-12     # midpoint of lo=1, hi=2


def test_ecdf_training_block_only():
    """Fitting values never change the map (pure function of train_vals)."""
    tv = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([2.0, 3.5, 0.5])
    a = p18.ecdf_percentile(tv, v, n_train=4)
    b = p18.ecdf_percentile(tv, v, n_train=4)
    assert np.array_equal(a, b)


def test_trust_block_domain_and_degenerate_flag():
    n_users, n_days = 8, 20
    rows = [{"user": f"U{u:03d}", "day": pd.Timestamp("2010-01-01")
             + pd.Timedelta(days=d)} for u in range(n_users)
            for d in range(n_days)]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    rt, flags = p18.r_trust_block(blk, blk)
    assert np.isfinite(rt).all()
    assert rt.min() >= 0.0 and rt.max() <= 1.0
    assert all(k in flags for k in p18.F_G)


def test_trust_block_constant_feature_flagged():
    blk = pd.DataFrame({"user": ["U000", "U001"],
                        "day": [pd.Timestamp("2010-01-01")] * 2})
    for f in p18.ALL_FEATURES:
        blk[f] = 7.0
    rt, flags = p18.r_trust_block(blk, blk)
    assert np.allclose(rt, 0.5)  # degenerate -> constant 0.5
    assert any("degenerate" in v for v in flags.values())


# ---------------------------------------------------------------------------
# Context (peer groups, cutoffs, cold start, unseen roles)
# ---------------------------------------------------------------------------
def _context_block(n_users: int = 12, n_days: int = 40,
                   roles: dict | None = None) -> tuple:
    users = [f"U{u:03d}" for u in range(n_users)]
    days = pd.date_range("2010-01-01", periods=n_days)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    if roles is None:
        roles = {u: ("R1", "D1") for u in users}
    return users, blk, roles


def test_context_cold_start_neutral():
    users, blk, roles = _context_block()
    rc, diag = p18.r_context_block(blk, users, roles)
    day0 = blk[blk["user"] == users[0]].index[0]
    assert abs(rc[day0] - 0.5) < 1e-9  # no strictly-past peer rows
    assert np.isfinite(rc).all() and rc.min() >= 0 and rc.max() <= 1


def test_context_peer_group_cutoffs():
    users = [f"U{u:03d}" for u in range(8)]
    days = pd.date_range("2010-01-01", periods=40)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    roles = {u: ("Big", "BD") for u in users[:6]}   # 5 peers -> role pool
    roles.update({u: ("Tiny", "TD") for u in users[6:]})  # 1 peer -> neutral
    rc, diag = p18.r_context_block(blk, users, roles)
    assert diag["peer_sources"][users[0]] == "role"
    assert diag["peer_pool_sizes"][users[0]] == 5
    assert diag["neutral_users"][users[6]] == "neutral-small-peer-group"
    idx6 = blk[blk["user"] == users[6]].index[0]
    assert abs(rc[idx6] - 0.5) < 1e-9


def test_context_department_fallback():
    users = [f"U{u:03d}" for u in range(8)]
    days = pd.date_range("2010-01-01", periods=40)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    # each role has only 2 members (< 5), same department has 6 -> fallback
    roles = {u: ("RA" if i % 2 == 0 else "RB", "DPT") for i, u in
             enumerate(users)}
    rc, diag = p18.r_context_block(blk, users, roles)
    assert diag["peer_sources"][users[0]] == "department"
    assert diag["peer_pool_sizes"][users[0]] >= 5
    assert np.all(np.isfinite(rc))


def test_context_unseen_user_neutral():
    users, blk, roles = _context_block(n_users=12)
    rc, diag = p18.r_context_block(blk, users + ["U999"], roles)
    assert diag["neutral_users"]["U999"] == "neutral-no-role"
    assert np.all(np.isfinite(rc))


def test_context_min_support_window():
    """< 20 past peer rows -> neutral; >= 20 -> computed (zero-MAD 0.5)."""
    users = [f"U{u:03d}" for u in range(8)]
    n_days = 30
    days = pd.date_range("2010-01-01", periods=n_days)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    roles = {u: ("R1", "D1") for u in users}
    rc, diag = p18.r_context_block(blk, users, roles)
    mask0 = blk["user"] == users[0]
    u0 = rc[mask0.to_numpy()]
    assert abs(u0[0] - 0.5) < 1e-9       # cold start
    assert abs(u0[1] - 0.5) < 1e-9       # < 20 past peer rows -> neutral
    # 5 peers x 20 past days = 100 rows >= min_support -> computed;
    # all values 1.0 -> zero-MAD rule -> zc = 0 -> 0.5
    assert abs(u0[20] - 0.5) < 1e-9


def test_peer_groups_role_department_neutral_sources():
    users = [f"U{i:03d}" for i in range(10)]
    role_map = {u: ("R1", "D1") for u in users[:6]}
    role_map.update({u: ("R2", "D2") for u in users[6:8]})
    role_map.update({u: ("R3", "D3") for u in users[8:]})
    out, src = p18.peer_groups(users, role_map, min_peers=5)
    assert src[users[0]] == "role" and len(out[users[0]]) == 5
    assert src[users[6]] == "neutral-small-peer-group"
    assert src[users[8]] == "neutral-small-peer-group"
    out2, src2 = p18.peer_groups(users, {}, min_peers=5)
    assert out2[users[0]] is None and src2[users[0]] == "neutral-no-role"


# ---------------------------------------------------------------------------
# Behavioral block (magnitude-only phi map)
# ---------------------------------------------------------------------------
def test_behavior_block_deviation_and_neutral():
    n_users, n_days = 6, 30
    users = [f"U{u:03d}" for u in range(n_users)]
    days = pd.date_range("2010-01-01", periods=n_days)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        blk[f] = 1.0
    blk.loc[blk["user"] == "U000", "login_count"] = 10.0
    rb = p18.r_behavior_block(blk)
    assert np.isfinite(rb).all() and rb.min() >= 0 and rb.max() <= 1
    rb0 = rb[blk["user"] == "U000"]
    assert rb0.min() >= 0.5                    # deviation never lowers
    rb_other = rb[blk["user"] == "U001"]
    assert abs(rb_other.min() - 0.5) < 1e-9    # constant user -> 0.5
    assert np.allclose(rb_other, 0.5)


# ---------------------------------------------------------------------------
# Fusion (SLSQP constraints and determinism)
# ---------------------------------------------------------------------------
def _syn_fusion_inputs(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    R = np.zeros((n, 4))
    R[:, 0] = rng.uniform(0.05, 0.95, n)
    R[:, 1] = np.clip(1.0 - R[:, 0] + rng.normal(0, 0.2, n), 0.001, 0.999)
    R[:, 2] = rng.uniform(0.0, 1.0, n)
    R[:, 3] = np.clip(R[:, 0] * 0.5 + rng.normal(0, 0.1, n), 0.001, 0.999)
    y = (rng.uniform(0, 1, n) < R[:, 0] * 0.6 + 0.05).astype(int)
    return R, y


def test_fusion_single_parameter_ml_only():
    R, y = _syn_fusion_inputs()
    fit = p18.fit_fusion(R, y, [0], [1.0])
    assert abs(fit["weights"][0] - 1.0) < 1e-6
    assert abs(sum(fit["weights"]) - 1.0) < 1e-12


def test_fusion_constraints_and_determinism():
    R, y = _syn_fusion_inputs()
    f1 = p18.fit_fusion(R, y, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25])
    f2 = p18.fit_fusion(R, y, [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25])
    assert f1["weights"] == f2["weights"]
    assert abs(sum(f1["weights"]) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in f1["weights"])
    assert f1["converged"] is True
    # learned weights must not beat the ML-only objective materially when
    # the label depends only on R0 (no degradation of the fit)
    fit_ml = p18.fit_fusion(R, y, [0], [1.0])
    assert f1["objective_mean_logloss"] <= fit_ml[
        "objective_mean_logloss"] + 1e-8


# ---------------------------------------------------------------------------
# Tie audit
# ---------------------------------------------------------------------------
def test_tie_audit():
    ta = p18.tie_audit(np.arange(1000, dtype=float), n_rows=1000)
    assert ta["n_collision_rows"] == 0 and not ta["flagged"]
    ta2 = p18.tie_audit(np.concatenate([np.arange(95, dtype=float),
                                        np.full(5, 0.5)]), n_rows=100)
    assert ta2["n_collision_rows"] == 5 and ta2["flagged"]


# ---------------------------------------------------------------------------
# Bootstrap determinism (same seed -> identical replicates)
# ---------------------------------------------------------------------------
def test_user_block_bootstrap_deterministic():
    rng = np.random.default_rng(3)
    n_users, n_days = 30, 10
    users = np.repeat([f"U{i:03d}" for i in range(n_users)], n_days)
    y = rng.integers(0, 2, len(users))
    s = rng.uniform(0, 1, len(users))
    a = p18.user_block_bootstrap_arm(users, y, s, 0.5, n_boot=50)
    b = p18.user_block_bootstrap_arm(users, y, s, 0.5, n_boot=50)
    assert a == b
    assert 0.0 <= a["auc_roc"]["ci_low"] <= a["auc_roc"]["ci_high"] <= 1.0


def test_paired_and_row_level_deterministic():
    rng = np.random.default_rng(4)
    n_users, n_days = 30, 10
    users = np.repeat([f"U{i:03d}" for i in range(n_users)], n_days)
    y = rng.integers(0, 2, len(users))
    s1 = rng.uniform(0, 1, len(users))
    s2 = np.clip(s1 + 0.05, 0, 1)
    p1 = p18.paired_user_block_deltas(users, y, s1, s2, n_boot=50)
    p2 = p18.paired_user_block_deltas(users, y, s1, s2, n_boot=50)
    assert p1 == p2
    r1 = p18.row_level_bootstrap(y, s1, n_boot=50)
    r2 = p18.row_level_bootstrap(y, s1, n_boot=50)
    assert r1 == r2


def test_weight_refit_bootstrap_deterministic():
    R, y = _syn_fusion_inputs(n=600, seed=1)
    users = np.repeat([f"U{i:03d}" for i in range(60)], 10)
    y_cal = (y[:200] + 0) % 2
    n_users = 20
    test_frame = pd.DataFrame({
        "user": np.repeat([f"U{i:03d}" for i in range(n_users)], 10),
        "day": [pd.Timestamp("2010-01-01") + pd.Timedelta(days=d % 10)
                for d in range(200)],
        "is_malicious": [1 if i == 3 else 0 for i in range(n_users)
                         for _ in range(10)]})
    diag = {"per_user": [
        {"user": f"U{i:03d}", "n_positive_days": 1 if i == 3 else 0}
        for i in range(n_users)]}
    b1 = p18.weight_refit_bootstrap(users, R, y, R[:200], y_cal,
                                    R[:200], y_cal, test_frame, diag,
                                    [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25],
                                    n_boot=10)
    b2 = p18.weight_refit_bootstrap(users, R, y, R[:200], y_cal,
                                    R[:200], y_cal, test_frame, diag,
                                    [0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25],
                                    n_boot=10)
    assert b1 == b2
    assert b1["alpha"]["ci_width"] >= 0.0
    assert b1["coverage_detected"]["ci_low"] >= 0.0


# ---------------------------------------------------------------------------
# Mechanical verdict paths
# ---------------------------------------------------------------------------
def _mk_arm(ok: bool = True, degrade: bool = False) -> dict:
    if ok:
        op = {"f1": 0.5, "precision": 0.8, "n_alerts": 50, "mcc": 0.45,
              "coverage": {"coverage": {"detected": 10}}}
        cal = {"brier": 0.09, "ece": 0.05}
        d = {"auc_pr": {"point": 0.03, "ci_low": 0.005},
             "auc_roc": {"point": 0.001, "ci_low": -0.001}}
    else:
        op = {"f1": 0.30, "precision": 0.60, "n_alerts": 600, "mcc": 0.20,
              "coverage": {"coverage": {"detected": 4}}}
        cal = {"brier": 0.15, "ece": 0.12}
        d = {"auc_pr": {"point": -0.001, "ci_low": -0.008},
             "auc_roc": {"point": -0.002, "ci_low": -0.004}}
    if degrade:
        d = {"auc_pr": {"point": -0.05, "ci_low": -0.08},
             "auc_roc": {"point": -0.03, "ci_low": -0.05}}
    return {"operating_test": op, "deltas_vs_arm_a": d,
            "test_calibration": cal, "tie_audit": {"flagged": False},
            "weights": [0.7, 0.1, 0.1, 0.1]}


@pytest.fixture()
def _gates_ok():
    return {"g1": {"passed": True}, "g2": {"passed": True},
            "g3": {"passed": True}, "g4": {"passed": True},
            "g5": {"passed": True}, "g6": {"passed": True}}


@pytest.fixture()
def _boot_ok():
    boot = {"weight_refit": {}}
    for a in p18.LEARNED_ARMS:
        boot["weight_refit"][a] = {
            "one_minus_alpha": {"ci_low": 0.05, "ci_high": 0.2},
            "ci_width": {k: 0.05 for k in p18.COMPONENT_NAMES}}
    return boot


def test_verdict_pass(_gates_ok, _boot_ok):
    arms = {a: _mk_arm() for a in ("A", "B", "C", "C1", "C2", "C3")}
    v = p18.compute_verdict(_gates_ok, arms, _boot_ok, True)
    assert v["verdict"] == "PASS"
    assert v["criteria"]["pass_arms"]


def test_verdict_ml_only_collapse(_gates_ok, _boot_ok):
    arms = {a: _mk_arm(ok=False) for a in ("A", "B", "C", "C1", "C2", "C3")}
    for a in p18.LEARNED_ARMS:
        arms[a]["weights"] = [0.98, 0.01, 0.005, 0.005]
    v = p18.compute_verdict(_gates_ok, arms, _boot_ok, True)
    assert v["verdict"] == "FAIL" and "collapse" in v["reason"]
    assert v["criteria"]["ml_only_collapse"] is True


def test_verdict_degradation_precedence(_gates_ok, _boot_ok):
    arms = {a: _mk_arm() for a in ("A", "B", "C", "C1", "C2", "C3")}
    arms["C"]["deltas_vs_arm_a"] = {
        "auc_pr": {"point": -0.03, "ci_low": -0.06},
        "auc_roc": {"point": -0.02, "ci_low": -0.04}}
    arms["C"]["operating_test"]["n_alerts"] = 3000
    v = p18.compute_verdict(_gates_ok, arms, _boot_ok, True)
    assert v["verdict"] == "FAIL" and "degrad" in v["reason"]


def test_verdict_gate_failure_and_determinism(_gates_ok, _boot_ok):
    arms = {a: _mk_arm() for a in ("A", "B", "C", "C1", "C2", "C3")}
    bad = dict(_gates_ok)
    bad["g5"] = {"passed": False}
    assert p18.compute_verdict(bad, arms, _boot_ok, True)["verdict"] == "FAIL"
    assert p18.compute_verdict(_gates_ok, arms, _boot_ok,
                               False)["verdict"] == "FAIL"


def test_verdict_caution_no_improvement(_gates_ok, _boot_ok):
    arms = {a: _mk_arm(ok=False) for a in ("A", "B", "C", "C1", "C2", "C3")}
    for a in p18.LEARNED_ARMS:
        arms[a]["weights"] = [0.6, 0.2, 0.1, 0.1]
    v = p18.compute_verdict(_gates_ok, arms, _boot_ok, True)
    assert v["verdict"] == "CAUTION"


# ---------------------------------------------------------------------------
# OOF folds: entity-disjoint, cover TRAIN, never CAL/TEST users
# ---------------------------------------------------------------------------
def test_oof_fold_structure():
    users = [f"U{i:03d}" for i in range(25)]
    rows = [{"user": u, "day": pd.Timestamp("2010-01-01")
             + pd.Timedelta(days=d), "is_malicious": 0}
            for u in users for d in range(10)]
    frame = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        frame[f] = 1.0
    oof, rec = p18.oof_ml_scores(frame, p18.ALL_FEATURES, {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": 31, "min_data_in_leaf": 5, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
        "seed": 42}, n_folds=5, num_boost_round=50, early_stopping_rounds=10)
    assert len(oof) == len(frame)
    folds = [set(rec["folds"][k]) for k in sorted(rec["folds"])]
    assert all(not (a & b) for i, a in enumerate(folds)
               for b in folds[i + 1:])
    assert set().union(*folds) == set(users)
    assert all(len(f) == 5 for f in folds)
    assert np.isfinite(oof).all() and oof.min() >= 0 and oof.max() <= 1


# ---------------------------------------------------------------------------
# Gate functions (synthetic fail case + real-artifact reproduction)
# ---------------------------------------------------------------------------
def test_gate1_fails_on_synthetic_frame():
    rng = np.random.default_rng(7)
    test = pd.DataFrame({
        "user": [f"T{i:03d}" for i in range(108)],
        "day": [pd.Timestamp("2010-01-01")] * 108,
        "is_malicious": [1 if i % 23 == 0 else 0 for i in range(108)],
        "score": rng.uniform(0.0, 1.0, 108),
        "alert_primary": [False] * 108})
    g1 = p18.gate1_arm_a_reproduction(test)
    assert g1["passed"] is False


def test_gate1_reproduces_recorded_phase14_values_on_real_artifact():
    test = pd.read_parquet(
        ART / "phase14_test_predictions.parquet")
    g1 = p18.gate1_arm_a_reproduction(test)
    assert g1["passed"] is True, g1["checks"]
    assert g1["recomputed"]["alert_primary_column_mismatches"] == 0


def test_gate2_structure_on_synthetic_split():
    """Structural sub-checks hold on any disjoint split; the hardcoded
    Phase 14 count checks are expected to fail on synthetic data."""
    n_train, n_cal, n_test = 12, 6, 6
    users = ([f"TR{i:03d}" for i in range(n_train)]
             + [f"CA{i:03d}" for i in range(n_cal)]
             + [f"TE{i:03d}" for i in range(n_test)])
    n_days = 20
    rows = [{"user": u, "day": pd.Timestamp("2010-01-01")
             + pd.Timedelta(days=d), "is_malicious": 0}
            for u in users for d in range(n_days)]
    merged = pd.DataFrame(rows)
    split = {"allocation": {"TRAIN": users[:n_train],
                            "CAL": users[n_train:n_train + n_cal],
                            "TEST": users[n_train + n_cal:]}}
    tr = merged[merged["user"].isin(split["allocation"]["TRAIN"])]
    ca = merged[merged["user"].isin(split["allocation"]["CAL"])]
    te = merged[merged["user"].isin(split["allocation"]["TEST"])]
    oof_rec = {"folds": {str(k): sorted(
        [u for i, u in enumerate(users[:n_train]) if i % 5 == k])
        for k in range(5)}}
    g2 = p18.gate2_split_integrity(split, tr, ca, te, oof_rec)
    assert g2["checks"]["disjoint"] is True
    assert g2["checks"]["frame_user_sets_match_allocation"] is True
    assert g2["checks"]["all_users_present"] is True
    assert g2["checks"]["folds_cover_train"] is True
    assert all(g2["checks"].get(f"fold_{k}_disjoint")
               for k in range(5))
    assert g2["passed"] is False  # hardcoded Phase 14 counts differ


def test_gate5_and_gate6_structure():
    n = 50
    comp = {name: pd.DataFrame({
        "user": [f"U{i:03d}" for i in range(n)],
        "day": [pd.Timestamp("2010-01-01")] * n,
        **{c: np.linspace(0.0, 1.0, n) for c in p18.COMPONENT_NAMES}})
        for name in ("TRAIN", "CAL", "TEST")}
    g5 = p18.gate5_component_sanity(comp)
    assert g5["passed"] is True
    g6 = p18.gate6_weight_isolation(
        {a: {"free_cols": p18.ARMS[a]["free_cols"]}
         for a in p18.LEARNED_ARMS})
    assert g6["passed"] is True
    assert g6["weight_fits_on_train_oof_only"] is True


# ---------------------------------------------------------------------------
# Frozen artifact immutability (gate 4 semantics) + role table schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", EXPECTED_MD5.items())
def test_frozen_input_md5(name, expected):
    path = ART / name
    assert path.is_file(), f"missing frozen artifact: {name}"
    assert md5(path) == expected


def test_model_txt_md5():
    assert md5(ART / "phase14_model.txt") == MODEL_TXT_MD5


def test_role_table_schema_and_gate_consistency():
    role = pd.read_parquet(ART / "phase18_role_department.parquet")
    assert len(role) == 1000
    assert list(role.columns) == ["user_id", "role", "department",
                                  "value_source_month", "n_snapshots_seen"]
    assert role["role"].notna().all()
    assert int(role["department"].isna().sum()) == 14
    assert role["user_id"].str.match(r"^[A-Z]{3}\d{4}$").all()
    assert role["role"].nunique() == 42
    assert role["department"].nunique() == 22
    assert set(role["value_source_month"].dropna()) <= {
        "2009-12", "2010-01", "2010-02", "2010-03", "2010-04",
        "2010-05", "2010-06", "2010-07", "2010-08", "2010-09",
        "2010-10", "2010-11", "2010-12", "2011-01", "2011-02",
        "2011-03", "2011-04"}


def test_diagnostic_users_never_in_train_allocation():
    with open(ART / "phase14_split.json", encoding="utf-8") as fh:
        split = json.load(fh)
    for u in ("JJM0203", "WDD0366"):
        assert u in split["allocation"]["TEST"]
        assert u not in split["allocation"]["TRAIN"]
        assert u not in split["allocation"]["CAL"]


def test_diagnostic_users_recorded_phase14_profiles():
    with open(ART / "phase14_user_diagnostics.json",
              encoding="utf-8") as fh:
        diag = json.load(fh)
    per = {d["user"]: d for d in diag["per_user"]}
    assert abs(per["JJM0203"]["max_score"] - 0.4981394) < 1e-6
    assert abs(per["WDD0366"]["max_score"] - 0.7264171) < 1e-6
    assert per["JJM0203"]["n_alerts"] == 0
    assert per["WDD0366"]["n_alerts"] == 0


# ---------------------------------------------------------------------------
# End-to-end assembler on a small synthetic panel (structure + determinism
# + evaluation isolation: TEST users never in any OOF fit)
# ---------------------------------------------------------------------------
def _syn_panel() -> tuple:
    n_tr, n_ca, n_te = 12, 6, 6
    n_days = 20
    tr_u = [f"TR{i:03d}" for i in range(n_tr)]
    ca_u = [f"CA{i:03d}" for i in range(n_ca)]
    te_u = [f"TE{i:03d}" for i in range(n_te)]
    rows = []
    for u in tr_u + ca_u + te_u:
        for d in range(n_days):
            rows.append({"user": u,
                         "day": pd.Timestamp("2010-01-01")
                         + pd.Timedelta(days=d),
                         "is_malicious": int(u.startswith("TE")
                                             and d >= 15)})
    merged = pd.DataFrame(rows)
    for f in p18.ALL_FEATURES:
        merged[f] = 1.0
    merged.loc[(merged["user"] == "TE003") & (merged["day"].dt.day >= 16),
               "login_count"] = 9.0
    split = {"allocation": {"TRAIN": tr_u, "CAL": ca_u, "TEST": te_u}}
    cal = merged[merged["user"].isin(ca_u)].copy()
    test = merged[merged["user"].isin(te_u)].copy()
    rng = np.random.default_rng(11)
    cal["score"] = rng.uniform(0.0, 1.0, len(cal))
    test["score"] = rng.uniform(0.0, 1.0, len(test))
    test["alert_primary"] = test["score"] >= 0.9
    user_diag = {"per_user": []}
    for u in te_u:
        user_diag["per_user"].append({
            "user": u, "n_positive_days": 5 if u == "TE003" else 0,
            "activity_tercile": "low", "length_class": "short",
            "temporal_half": "2010 H1", "scenario": 1 if u == "TE003"
            else None, "max_score": 0.5, "n_alerts": 0})
    role_df = pd.DataFrame({
        "user_id": tr_u + ca_u + te_u,
        "role": ["R1"] * (n_tr + n_ca + n_te),
        "department": ["D1"] * (n_tr + n_ca + n_te),
        "value_source_month": ["2011-04"] * (n_tr + n_ca + n_te),
        "n_snapshots_seen": [18] * (n_tr + n_ca + n_te)})
    gates_env = {
        "gate4_input_md5": {"passed": True, "files": {}},
        "gate3_tests": {"passed": True, "n_passed": 1, "n_failed": 0},
        "determinism_ok": False,
        "dataset_id": "synthetic",
        "environment": {"python": "test"},
        "kaggle_note": "synthetic test panel",
        "role_build_record": {"gates": {"passed": True}}}
    return merged, split, cal, test, user_diag, role_df, gates_env


def test_assembler_structure_and_determinism():
    args = _syn_panel()
    r1 = p18.run_phase18(*args, n_boot=5)
    r2 = p18.run_phase18(*args, n_boot=5)

    def _strip(r):
        return {k: v for k, v in r.items() if k != "components_table"}

    def _comp_canonical(r):
        return p18.canonical({name: v.to_dict("records")
                              for name, v in
                              r["components_table"].items()})

    assert p18.canonical(_strip(r1)) == p18.canonical(_strip(r2))
    assert _comp_canonical(r1) == _comp_canonical(r2)
    assert r1["experiment_id"] == "phase18-adaptive-risk-v2"
    assert set(r1["gates"]) == {"gate1", "gate2", "gate3", "gate4",
                                "gate5", "gate6"}
    assert set(r1["arms"]) == {"A", "B", "C", "C1", "C2", "C3"}
    assert "scores_test" not in r1["arms"]["C"]
    assert set(r1["verdict"]) == {"verdict", "reason", "criteria",
                                  "bounds", "note"}
    assert "redundancy" in r1 and set(r1["redundancy"]) == {"TRAIN", "CAL"}
    for name in ("TRAIN", "CAL", "TEST"):
        comp = r1["components_table"][name]
        assert list(comp.columns) == ["user", "day", "r_trust",
                                      "r_context", "r_behavior", "r_ml"]
        for c in p18.COMPONENT_NAMES:
            v = comp[c].to_numpy()
            assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
    assert r1["bootstrap"]["n_boot"] == 5
    assert r1["bootstrap"]["seed"] == 42
    for arm in ("A", "B", "C", "C1", "C2", "C3"):
        assert "user_block" in r1["bootstrap"]
        assert arm in r1["bootstrap"]["user_block"]
    for arm in p18.LEARNED_ARMS:
        assert arm in r1["bootstrap"]["weight_refit"]
        assert "one_minus_alpha" in r1["bootstrap"]["weight_refit"][arm]
    assert r1["gates"]["gate2"]["passed"] is False  # synthetic counts
    assert r1["verdict"]["verdict"] == "FAIL"       # gate failure, by design


def test_assembler_test_evaluation_isolation():
    """TEST users appear in no OOF fit fold (fold membership = TRAIN only)."""
    args = _syn_panel()
    r = p18.run_phase18(*args, n_boot=3)
    tr_users = set(args[1]["allocation"]["TRAIN"])
    for k, fold in r["oof"]["folds"].items():
        assert set(fold) <= tr_users
    assert all(u not in tr_users for u in args[1]["allocation"]["TEST"])


def test_assembler_threshold_never_touch_test():
    """CAL thresholds are chosen once on CAL only (structural check)."""
    args = _syn_panel()
    r = p18.run_phase18(*args, n_boot=3)
    for arm in ("A", "B", "C", "C1", "C2", "C3"):
        op = r["arms"][arm]["operating_point"]
        assert op["rule"] == ("best_f1_threshold on CAL, once, "
                              "never TEST-informed")


def test_engine_components_parquet_bytes():
    """Engine determinism helper: scalar block broadcast, canonical
    parquet bytes (regression: scalar dict constructor raised on
    pandas 2.x, discovered on the real Kaggle run after run 1)."""
    import importlib.util
    import io
    import sys

    root = Path(__file__).resolve().parents[1]
    engine_path = root / "kaggle_scripts" / "run_phase18.py"
    spec = importlib.util.spec_from_file_location(
        "run_phase18_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    comp = {}
    for name, n in (("TRAIN", 3), ("CAL", 2), ("TEST", 4)):
        comp[name] = pd.DataFrame({
            "user": np.arange(n),
            "day": np.arange(n) + 1,
            "r_ml": np.full(n, 0.1),
            "r_trust": np.full(n, 0.2),
            "r_context": np.full(n, 0.3),
            "r_behavior": np.full(n, 0.4),
        })
    b1 = mod.components_parquet_bytes(comp)
    b2 = mod.components_parquet_bytes(comp)
    assert b1 == b2 and len(b1) > 0
    out = pd.read_parquet(io.BytesIO(b1))
    assert list(out.columns) == ["block", "user", "day", "r_ml",
                                 "r_trust", "r_context", "r_behavior"]
    assert out["block"].tolist() == (["TRAIN"] * 3 + ["CAL"] * 2
                                     + ["TEST"] * 4)
    assert len(out) == 9


# ---------------------------------------------------------------------------
# Optimized R_context (authorized 2026-08-20, implementation only):
# bit-identical to the frozen serial reference on synthetic fixtures and
# edge cases; real-block equivalence is covered by the standalone
# benchmark (benchmark_phase18_context.py, PC) and a parallel-consistency
# smoke on the real CAL block below.
# ---------------------------------------------------------------------------
def _context_block_var(n_users: int = 12, n_days: int = 45, seed: int = 0,
                       roles: dict | None = None) -> tuple:
    users = [f"U{u:03d}" for u in range(n_users)]
    days = pd.date_range("2010-01-01", periods=n_days)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    for f in p18.ALL_FEATURES:
        blk[f] = rng.uniform(0.0, 5.0, len(blk))
    if roles is None:
        roles = {u: ("R1", "D1") for u in users}
    return users, blk, roles


def _assert_context_bit_identical(blk, users, roles, **kw):
    ref, rdiag = p18._r_context_block_serial(blk, users, roles, **kw)
    opt, odiag = p18.r_context_block(blk, users, roles, workers=1, **kw)
    assert np.array_equal(ref, opt)
    assert rdiag == odiag


def test_context_optimized_equals_serial_synthetic():
    users, blk, roles = _context_block_var(n_users=12, n_days=45, seed=3)
    _assert_context_bit_identical(blk, users, roles)


def test_context_optimized_equals_serial_department_fallback():
    users = [f"U{u:03d}" for u in range(8)]
    days = pd.date_range("2010-01-01", periods=40)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    rng = np.random.default_rng(7)
    for f in p18.ALL_FEATURES:
        blk[f] = rng.uniform(0.0, 5.0, len(blk))
    roles = {u: ("RA" if i % 2 == 0 else "RB", "DPT") for i, u in
             enumerate(users)}
    _assert_context_bit_identical(blk, users, roles)
    rc, diag = p18.r_context_block(blk, users, roles)
    assert diag["peer_sources"][users[0]] == "department"
    assert np.isfinite(rc).all()


def test_context_optimized_null_department_neutral():
    """Role present but < 5 peers and no department -> neutral-no-department."""
    users = [f"U{u:03d}" for u in range(4)]
    days = pd.date_range("2010-01-01", periods=30)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    rng = np.random.default_rng(11)
    for f in p18.ALL_FEATURES:
        blk[f] = rng.uniform(0.0, 5.0, len(blk))
    roles = {u: ("Tiny", None) for u in users}
    _assert_context_bit_identical(blk, users, roles)
    rc, diag = p18.r_context_block(blk, users, roles)
    assert diag["neutral_users"][users[0]] == "neutral-no-department"
    assert np.all(rc[blk["user"].isin(users).to_numpy()] == 0.5)


def test_context_optimized_insufficient_peer_group():
    """Roles with < 5 members and a department fallback that is also
    < 5 members -> neutral-small-peer-group."""
    users = [f"U{u:03d}" for u in range(5)]  # 4 role peers < MIN_PEERS
    days = pd.date_range("2010-01-01", periods=30)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    rng = np.random.default_rng(13)
    for f in p18.ALL_FEATURES:
        blk[f] = rng.uniform(0.0, 5.0, len(blk))
    roles = {u: ("Solo", "SoloDep") for u in users}
    _assert_context_bit_identical(blk, users, roles)
    rc, diag = p18.r_context_block(blk, users, roles)
    assert diag["neutral_users"][users[0]] == "neutral-small-peer-group"
    assert np.all(rc[blk["user"].isin(users).to_numpy()] == 0.5)


def test_context_optimized_weekday_weekend_strata():
    """Weekend rows get a different feature level so strata contribute;
    values are non-trivial and bit-identical to the serial reference."""
    users = [f"U{u:03d}" for u in range(8)]
    n_days = 60
    days = pd.date_range("2010-01-01", periods=n_days)
    rows = [{"user": u, "day": d} for u in users for d in days]
    blk = pd.DataFrame(rows)
    rng = np.random.default_rng(17)
    for f in p18.ALL_FEATURES:
        blk[f] = rng.uniform(0.0, 5.0, len(blk))
    weekend = pd.to_datetime(blk["day"]).dt.dayofweek >= 5
    for f in ("login_count", "usb_connection_count"):
        blk.loc[weekend, f] = 9.0
    roles = {u: ("R1", "D1") for u in users}
    _assert_context_bit_identical(blk, users, roles)
    rc, diag = p18.r_context_block(blk, users, roles)
    assert not np.allclose(rc, 0.5)  # strata actually contribute
    assert np.isfinite(rc).all() and rc.min() >= 0 and rc.max() <= 1


def test_context_optimized_parallel_matches_serial():
    users, blk, roles = _context_block_var(n_users=12, n_days=45, seed=19)
    ref, rdiag = p18._r_context_block_serial(blk, users, roles)
    opt, odiag = p18.r_context_block(blk, users, roles, workers=2)
    assert np.array_equal(ref, opt)
    assert rdiag == odiag


def test_context_optimized_repeatable():
    users, blk, roles = _context_block_var(n_users=12, n_days=45, seed=23)
    a, da = p18.r_context_block(blk, users, roles)
    b, db = p18.r_context_block(blk, users, roles)
    assert np.array_equal(a, b) and da == db


@pytest.mark.skipif(not os.path.isdir(ART),
                    reason="frozen artifacts not present locally")
def test_context_optimized_real_cal_parallel_consistency():
    """Real CAL block: optimized workers=1 vs workers=4 bit-identical
    (full serial-vs-optimized equivalence runs in the standalone
    benchmark benchmark_phase18_context.py)."""
    merged = pd.read_parquet(ART / "phase6_merged_features.parquet")
    with open(ART / "phase14_split.json", encoding="utf-8") as fh:
        split = json.load(fh)
    role_df = pd.read_parquet(ART / "phase18_role_department.parquet")
    role_map = {str(r["user_id"]): (r["role"], r["department"])
                for _, r in role_df.iterrows()}
    cal = merged[merged["user"].isin(split["allocation"]["CAL"])].copy()
    users = sorted(cal["user"].unique())[:24]
    cal = cal[cal["user"].isin(users)].copy()
    a, da = p18.r_context_block(cal, users, role_map, workers=1)
    b, db = p18.r_context_block(cal, users, role_map, workers=4)
    assert np.array_equal(a, b) and da == db


def test_engine_double_determinism_runs_exactly_twice():
    """Gate 7 (spec Section 17.7): the engine runs the full pipeline
    EXACTLY twice; run 2 carries determinism_ok=True into the mechanical
    verdict; only the verdict block may differ; any other difference
    raises SystemExit."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    engine_path = root / "kaggle_scripts" / "run_phase18.py"
    spec = importlib.util.spec_from_file_location(
        "run_phase18_engine2", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    def _comp():
        comp = {}
        for name, n in (("TRAIN", 2), ("CAL", 1), ("TEST", 2)):
            comp[name] = pd.DataFrame({
                "user": np.arange(n),
                "day": np.arange(n) + 1,
                "r_ml": np.full(n, 0.1),
                "r_trust": np.full(n, 0.2),
                "r_context": np.full(n, 0.3),
                "r_behavior": np.full(n, 0.4)})
        return comp

    calls = {"n": 0, "det_flags": []}

    def fake_analysis(*args):
        gates_env = args[-1]
        calls["n"] += 1
        calls["det_flags"].append(gates_env.get("determinism_ok"))
        return {"components_table": _comp(),
                "verdict": {"det": gates_env.get("determinism_ok")}}

    gates_env = {"determinism_ok": False}
    res = mod.run_double_determinism(fake_analysis, (1,), gates_env)
    assert calls["n"] == 2
    assert calls["det_flags"] == [False, True]  # run 2 carries the flag
    assert res["verdict"]["det"] is True        # run-2 result is final
    assert res["determinism"]["runs_required"] == 2
    assert res["determinism"]["runs_completed"] == 2
    assert res["determinism"]["bit_identical"] is True
    assert res["determinism"]["components_byte_identical"] is True
    assert "Section 17.7" in res["determinism"]["note"]

    broken = {"n": 0}

    def divergent_analysis(*args):
        gates_env = args[-1]
        broken["n"] += 1
        if broken["n"] == 2:
            return {"components_table": _comp(), "verdict": {"det": True},
                    "x": 2}
        return {"components_table": _comp(),
                "verdict": {"det": gates_env.get("determinism_ok")}}

    with pytest.raises(SystemExit):
        mod.run_double_determinism(divergent_analysis, (1,),
                                   {"determinism_ok": False})
    assert broken["n"] == 2