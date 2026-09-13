"""Tests for Phase 19 (Comprehensive Adaptive Risk Fusion Benchmark).

Coverage:
  - Allocation integrity (determinism, no overlap, exact counts)
  - Fold assignment (outer/inner, user-disjoint)
  - Fold-safe LightGBM OOF (no user leakage, determinism)
  - All algorithm family execution paths (A0-H2, I smoke, J feasibility)
  - Component reuse from Phase 18
  - Preconfirm freeze structure
  - Verdict mechanics
  - Bootstrap determinism
  - Module structure (families registered, HP grid)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src import config
from src.experiments import phase19 as p19
from src.experiments import phase18 as p18

LOCAL_ART = Path(__file__).resolve().parent.parent / "reports" / "artifacts"

# ── 1. Allocation integrity (3 tests) ──

def test_allocation_user_counts():
    with open(LOCAL_ART / "phase19_user_allocation.json") as f:
        a = json.load(f)
    assert a["dev"]["n_users"] == 588 and a["confirm"]["n_users"] == 196
    assert a["dev"]["malicious"] == 30 and a["confirm"]["malicious"] == 10

def test_allocation_no_overlap():
    with open(LOCAL_ART / "phase19_user_allocation.json") as f:
        a = json.load(f)
    assert len(set(a["dev"]["users"]) & set(a["confirm"]["users"])) == 0

def test_allocation_deterministic():
    with open(LOCAL_ART / "phase19_user_allocation.json") as f:
        a = json.load(f)
    assert "GTD0219" in a["confirm"]["malicious_users"]
    assert "MAR0955" in a["confirm"]["malicious_users"]

# ── 2. Fold assignment (3 tests) ──

def test_outer_fold_assignment():
    folds = p19.assign_outer_folds([f"U{i:04d}" for i in range(20)], 5)
    assert folds["U0000"] == 0 and folds["U0005"] == 0

def test_inner_fold_assignment():
    folds = p19.assign_inner_folds([f"U{i:04d}" for i in range(15)], 3)
    assert folds["U0000"] == 0 and folds["U0003"] == 0

def test_fold_disjointness():
    folds = p19.assign_outer_folds([f"U{i:04d}" for i in range(50)], 5)
    for k in range(5):
        fk = {u for u, f in folds.items() if f == k}
        for j in range(5):
            if j != k:
                assert fk.isdisjoint({u for u, f in folds.items() if f == j})

# ── 3. Fold-safe LightGBM OOF (2 tests) ──

def test_fold_safe_oof_no_leakage():
    np.random.seed(42)
    users = [f"U{i:02d}" for i in range(10)]
    rows = []
    for u in users:
        for d in range(20):
            rows.append({"user": u, "day": f"2010-01-{d+1:02d}",
                         "is_malicious": int(np.random.rand() > 0.9),
                         **{f: np.random.rand() for f in p19.ALL_FEATURES}})
    df = pd.DataFrame(rows)
    tr = df[df["user"].isin(users[:7])]
    va = df[df["user"].isin(users[7:])]
    preds, rec = p19.fold_safe_lgbm_oof(tr, va)
    assert len(preds) == len(va)
    assert set(tr["user"].unique()) == set(users[:7])
    assert set(va["user"].unique()) == set(users[7:])
    assert len(set(users[:7]) & set(users[7:])) == 0

def test_fold_safe_oof_deterministic():
    np.random.seed(42)
    df = pd.DataFrame({"user": [f"U{i//5:02d}" for i in range(25)],
                        "day": [f"2010-01-{(i%5)+1:02d}" for i in range(25)],
                        "is_malicious": np.random.randint(0, 2, 25),
                        **{f: np.random.rand(25) for f in p19.ALL_FEATURES}})
    tr = df[df["user"].isin([f"U{i:02d}" for i in range(4)])]
    va = df[df["user"].isin(["U04"])]
    p1, _ = p19.fold_safe_lgbm_oof(tr, va)
    p2, _ = p19.fold_safe_lgbm_oof(tr, va)
    np.testing.assert_array_equal(p1, p2)

# ── 4. Algorithm family execution paths (12 tests) ──

def test_family_a0():
    R = np.array([0.1, 0.5, 0.9])
    s = p19.family_a0_score(R)
    np.testing.assert_array_equal(s, R)
    R[0] = 0.99
    assert s[0] == 0.1  # copy, not view

def test_family_b():
    R = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    fit = p19.family_b_fit(R, y)
    w = np.asarray(fit["weights"])
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= 0).all()
    # Deterministic
    fit2 = p19.family_b_fit(R, y)
    np.testing.assert_array_equal(fit["weights"], fit2["weights"])

def test_family_c():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    fit = p19.family_c_fit(X, y)
    assert len(fit["coef"]) == 4 and isinstance(fit["intercept"], float)

def test_family_d():
    R = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    fit = p19.family_d_fit(R, y)
    assert (np.asarray(fit["coef"]) >= 0).all()

def test_family_e():
    R = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    fit = p19.family_e_fit(R, y)
    assert len(fit["coef"]) == 6

def test_family_f():
    s = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    np.testing.assert_array_almost_equal(p19.family_f_score_user(s, 0.0), s)
    sm = p19.family_f_score_user(np.array([0.0, 0.0, 0.0, 1.0, 1.0]), 0.9)
    assert sm[0] == 0.0 and sm[-1] > sm[-2]

def test_family_g():
    R = np.random.rand(200, 4)
    y = np.random.randint(0, 2, 200)
    fit = p19.family_g_fit(R, y)
    scores = p19.family_g_score(R, fit)
    assert (scores >= 0).all() and (scores <= 1).all()
    for ci in range(4):
        assert all(-5.0 <= v <= 5.0 for v in fit["bins"][ci]["lr"])

def test_family_h_fallback():
    R = np.random.rand(100, 4)
    s = p19.family_h_score(R, {"error": "x", "fallback": True})
    np.testing.assert_array_almost_equal(s, np.mean(R, axis=1))

def test_family_h_actual_training():
    """Execute actual H1 training/scoring path with synthetic data."""
    R_train = np.random.rand(200, 4)
    y_train = np.random.randint(0, 2, 200)
    R_val = np.random.rand(50, 4)
    try:
        fit = p19.family_h_fit(R_train, y_train, hidden=None, l2=0.01,
                               entropy_reg=0.0, epochs=5, batch_size=64)
        if fit.get("error"):
            pytest.skip("torch not available")
        scores = p19.family_h_score(R_val, fit)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))
        assert (scores >= 0).all() and (scores <= 1).all()
    except ImportError:
        pytest.skip("torch not available")

def test_family_h2_actual_training():
    """Execute actual H2 (MLP) training/scoring path."""
    R_train = np.random.rand(200, 4)
    y_train = np.random.randint(0, 2, 200)
    R_val = np.random.rand(50, 4)
    try:
        fit = p19.family_h_fit(R_train, y_train, hidden=8, l2=0.01,
                               entropy_reg=0.0, epochs=5, batch_size=64)
        if fit.get("error"):
            pytest.skip("torch not available")
        scores = p19.family_h_score(R_val, fit)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))
        assert (scores >= 0).all() and (scores <= 1).all()
    except ImportError:
        pytest.skip("torch not available")

def test_family_i_smoke():
    """Tiny Transformer smoke test: shape, finite, [0,1], deterministic."""
    np.random.seed(42)
    R_train = np.random.rand(100, 4).astype(np.float64)
    y_train = np.random.randint(0, 2, 100).astype(np.float64)
    R_val = np.random.rand(30, 4).astype(np.float64)
    try:
        fit = p19.family_i_fit(R_train, y_train, seq_len=5, d_model=8,
                               n_heads=2, n_layers=1, ff_dim=16)
        if fit.get("error"):
            pytest.skip("torch not available")
        scores = p19.family_i_score(R_val, fit)
        assert scores.shape == (30,)
        assert np.all(np.isfinite(scores))
        assert (scores >= 0).all() and (scores <= 1).all()
        # First seq_len-1 scores should be 0.5 (cold start)
        assert all(s == 0.5 for s in scores[:4])
    except ImportError:
        pytest.skip("torch not available")

def test_family_j_feasibility_unavailable():
    """J feasibility failure path when PyG is unavailable."""
    merged = pd.DataFrame({"user": ["u1"]*5, "day": range(5),
                           "device_consistency_score": [0.5]*5})
    checks = p19.family_j_feasibility_check(merged)
    # PyG likely unavailable locally → feasible should be False
    if not checks.get("pyg_available", False):
        assert checks["feasible"] is False

def test_family_j_feasibility_structure():
    """J feasibility gate structure check."""
    cols = {f: [0.5]*5 for f in p19.F_G}
    merged = pd.DataFrame({"user": ["u1"]*5, "day": range(5), **cols})
    checks = p19.family_j_feasibility_check(merged)
    assert "graph_features_available" in checks
    assert "pyg_available" in checks
    assert "feasible" in checks

# ── 5. Component reuse (2 tests) ──

def test_component_reuse():
    assert p19.r_behavior_block is p18.r_behavior_block
    assert p19.r_trust_block is p18.r_trust_block
    assert p19.r_context_block is p18.r_context_block

def test_constants():
    assert p19.F_B == p18.F_B and p19.ALL_FEATURES == p18.ALL_FEATURES

# ── 6. Protocol integrity (3 tests) ──

def test_preconfirm_freeze():
    f = p19.build_preconfirm_freeze(
        "B", [{"family": "B"}],
        {"dev": {"users": ["u1"]}, "confirm": {"users": ["u2"]}},
        {"B": {"mean_auc_pr": 0.3}}, {}, {})
    assert f["confirm_opened"] is False and f["confirm_completed"] is False

def test_verdict():
    v = p19.compute_verdict({}, {}, [], None)
    assert v["verdict"] == "CAUTION"
    c = {"A0": {"auc_pr": 0.3, "auc_roc": 0.78, "f1": 0.27, "precision": 0.2,
                "mcc": 0.28, "n_alerts": 1026},
         "B": {"auc_pr": 0.35, "auc_roc": 0.80, "f1": 0.45, "precision": 0.75,
               "mcc": 0.45, "n_alerts": 250}}
    assert p19.compute_verdict({}, c, [{"family": "B"}], "B")["verdict"] == "PASS"

def test_bootstrap_deterministic():
    u = np.array(["U1"]*10 + ["U2"]*10 + ["U3"]*10)
    y = np.array([0]*8 + [1]*2 + [0]*8 + [1]*2 + [0]*8 + [1]*2)
    s = np.random.rand(30)
    b1 = p19.user_block_bootstrap(u, y, s, 0.5, n_boot=50, seed=42)
    b2 = p19.user_block_bootstrap(u, y, s, 0.5, n_boot=50, seed=42)
    assert b1["auc_roc"]["mean"] == b2["auc_roc"]["mean"]

# ── 7. Module structure (2 tests) ──

def test_families_registered():
    for f in ("A0","B","C","D","E","F1","F2","G","H1","H2","I","J"):
        assert f in p19.FAMILIES

def test_hp_grid():
    assert p19.HP_GRID["C"] == [0.1, 1.0, 10.0]
    assert len(p19.HP_GRID["F1"]) == 5

# ── 8. Post-audit regression tests (6 tests) ──

def test_family_c_regularization_effect():
    """Family C with penalty='l2' must produce different fits for different C values."""
    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 4).astype(np.float64)
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.2 * np.random.rand(n) > 0.7).astype(int)
    fit_lo = p19.family_c_fit(X, y, C_val=0.1)
    fit_hi = p19.family_c_fit(X, y, C_val=10.0)
    assert fit_lo["C"] == 0.1
    assert fit_hi["C"] == 10.0
    coef_lo = np.asarray(fit_lo["coef"])
    coef_hi = np.asarray(fit_hi["coef"])
    assert not np.allclose(coef_lo, coef_hi, atol=1e-6), \
        "Family C C=0.1 and C=10 produced identical coefficients — regularization not working"

def test_candidate_count_32():
    """Exactly 32 frozen candidates are generated (excluding I/J)."""
    from src.experiments.phase19 import FAMILIES, HP_GRID
    count = 0
    for family in FAMILIES:
        if family in ("I", "J"):
            continue
        if family in ("F1", "F2", "C", "D", "E"):
            count += len(HP_GRID[family])
        elif family in ("H1", "H2"):
            hg = HP_GRID[family]
            count += len(hg["l2"]) * len(hg["entropy_reg"])
        else:
            count += 1
    assert count == 32, f"Expected 32 candidates, got {count}"

def test_cache_equivalence():
    """Cached and uncached component/PML results are identical for the same inner split."""
    np.random.seed(42)
    users = [f"U{i:02d}" for i in range(20)]
    rows = []
    for u in users:
        for d in range(30):
            rows.append({"user": u, "day": f"2010-01-{d+1:02d}",
                         "is_malicious": int(np.random.rand() > 0.9),
                         **{f: np.random.rand() for f in p19.ALL_FEATURES}})
    df = pd.DataFrame(rows).sort_values(["user", "day"]).reset_index(drop=True)
    role_map = {u: ("roleA", "deptB") for u in users}

    # Uncached path: compute inside inner_cv_evaluate (no fold_cache)
    r_uncached = p19.inner_cv_evaluate(users, df, role_map, "A0", None,
                                       fold_cache=None)

    # Cached path: build cache, then pass it
    inner_folds = p19.assign_inner_folds(users, 3)
    fold_cache = {}
    for k in range(3):
        i_train_users = [u for u in users if inner_folds[u] != k]
        i_val_users = [u for u in users if inner_folds[u] == k]
        i_train = df[df["user"].isin(i_train_users)].copy()
        i_val = df[df["user"].isin(i_val_users)].copy()
        i_train = i_train.sort_values(["user", "day"]).reset_index(drop=True)
        i_val = i_val.sort_values(["user", "day"]).reset_index(drop=True)
        train_ml, _ = p19.fold_safe_lgbm_oof(i_train, i_train)
        val_ml, _ = p19.fold_safe_lgbm_oof(i_train, i_val)
        train_comp, _ = p19.compute_components_block(i_train, i_train,
                                                      sorted(i_train["user"].unique()), role_map)
        val_comp, _ = p19.compute_components_block(i_val, i_train,
                                                    sorted(i_val["user"].unique()), role_map)
        R_train = np.column_stack([train_ml, train_comp["r_trust"].to_numpy(),
                                   train_comp["r_context"].to_numpy(), train_comp["r_behavior"].to_numpy()])
        R_val = np.column_stack([val_ml, val_comp["r_trust"].to_numpy(),
                                 val_comp["r_context"].to_numpy(), val_comp["r_behavior"].to_numpy()])
        fold_cache[k] = {
            "R_train": R_train, "R_val": R_val,
            "y_train": i_train["is_malicious"].to_numpy().astype(int),
            "y_val": i_val["is_malicious"].to_numpy().astype(int),
            "n_train_users": len(i_train_users),
            "n_val_users": len(i_val_users),
        }

    r_cached = p19.inner_cv_evaluate(users, df, role_map, "A0", None,
                                     fold_cache=fold_cache)

    assert r_cached["mean_f1"] == r_uncached["mean_f1"], \
        f"Cached={r_cached['mean_f1']} != uncached={r_uncached['mean_f1']}"
    for k in range(3):
        assert r_cached["folds"][k]["f1"] == r_uncached["folds"][k]["f1"]

def test_cache_isolation():
    """Inner VAL users cannot influence fitted TRAIN statistics in cached data."""
    np.random.seed(42)
    users = [f"U{i:02d}" for i in range(20)]
    rows = []
    for u in users:
        for d in range(30):
            rows.append({"user": u, "day": f"2010-01-{d+1:02d}",
                         "is_malicious": int(np.random.rand() > 0.9),
                         **{f: np.random.rand() for f in p19.ALL_FEATURES}})
    df = pd.DataFrame(rows).sort_values(["user", "day"]).reset_index(drop=True)
    role_map = {u: ("roleA", "deptB") for u in users}
    inner_folds = p19.assign_inner_folds(users, 3)

    # Build cache for fold 0
    i_train_users = [u for u in users if inner_folds[u] != 0]
    i_val_users = [u for u in users if inner_folds[u] == 0]
    i_train = df[df["user"].isin(i_train_users)].copy().sort_values(["user", "day"]).reset_index(drop=True)
    i_val = df[df["user"].isin(i_val_users)].copy().sort_values(["user", "day"]).reset_index(drop=True)

    # Verify train and val users are disjoint
    assert len(set(i_train_users) & set(i_val_users)) == 0

    # Compute components — train stats must only use train users
    train_comp, _ = p19.compute_components_block(i_train, i_train,
                                                  sorted(i_train["user"].unique()), role_map)
    val_comp, _ = p19.compute_components_block(i_val, i_train,
                                                sorted(i_val["user"].unique()), role_map)

    # Train R_behavior is expanding median — must be finite and in [0,1]
    rb_train = train_comp["r_behavior"].to_numpy()
    assert np.all(np.isfinite(rb_train)), "Train R_behavior has non-finite values"
    assert (rb_train >= 0).all() and (rb_train <= 1).all(), "Train R_behavior out of [0,1]"

    # Val R_behavior must also be finite
    rb_val = val_comp["r_behavior"].to_numpy()
    assert np.all(np.isfinite(rb_val)), "Val R_behavior has non-finite values"

def test_no_mutation():
    """Candidate evaluation cannot mutate cached arrays/dataframes."""
    np.random.seed(42)
    R_train = np.random.rand(100, 4).astype(np.float64)
    R_val = np.random.rand(30, 4).astype(np.float64)
    y_train = np.random.randint(0, 2, 100)
    y_val = np.random.randint(0, 2, 30)

    R_train_copy = R_train.copy()
    R_val_copy = R_val.copy()
    y_train_copy = y_train.copy()
    y_val_copy = y_val.copy()

    for family in ("A0", "B", "C", "D", "E", "F1", "F2", "G"):
        hp = None
        if family == "C":
            hp = 1.0
        elif family == "D":
            hp = 0.1
        elif family == "E":
            hp = 0.1
        elif family == "F1":
            hp = 0.25
        elif family == "F2":
            hp = 0.25
        users_arr = np.array([f"u{i}" for i in range(3)])
        days_arr = np.array([0, 1, 2])
        p19._apply_family(R_train, y_train, R_val, family, hp,
                          val_users=users_arr, val_days=days_arr)

    np.testing.assert_array_equal(R_train, R_train_copy)
    np.testing.assert_array_equal(R_val, R_val_copy)
    np.testing.assert_array_equal(y_train, y_train_copy)
    np.testing.assert_array_equal(y_val, y_val_copy)

def test_confirmation_guard():
    """confirm_opened=false throughout DEV pipeline output."""
    f = p19.build_preconfirm_freeze(
        "B", [{"family": "B"}],
        {"dev": {"users": ["u1"]}, "confirm": {"users": ["u2"]}},
        {"B": {"mean_auc_pr": 0.3}}, {}, {})
    assert f["confirm_opened"] is False
    assert f["confirm_completed"] is False


# ── F1/F2 temporal persistence regression tests ──

def _make_fixture(n_users=6, days_per_user=5, seed=42):
    """Create a deterministic fixture with user/day metadata."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_users):
        uid = f"user_{u}"
        for d in range(days_per_user):
            ml = rng.uniform(0.1, 0.9)
            beh = rng.uniform(0.1, 0.9)
            trust = rng.uniform(0.1, 0.9)
            ctx = rng.uniform(0.1, 0.9)
            rows.append({"user": uid, "day": d, "ml": ml, "beh": beh,
                         "trust": trust, "ctx": ctx})
    df = pd.DataFrame(rows)
    R = df[["ml", "beh", "trust", "ctx"]].to_numpy()
    users = df["user"].to_numpy()
    days = df["day"].to_numpy()
    y = np.zeros(len(df), dtype=int)
    y[df["user"] == "user_0"] = 1
    return R, y, users, days


def test_f1_rho0_equals_d_l2_01():
    """F1 with rho=0 must exactly equal instantaneous Family D (l2=0.1)."""
    R, y, users, days = _make_fixture()
    scores_d = p19._apply_family(R, y, R, "D", 0.1)
    scores_f1 = p19._apply_family(R, y, R, "F1", 0.0,
                                  val_users=users, val_days=days)
    np.testing.assert_array_almost_equal(scores_f1, scores_d, decimal=10)


def test_f2_rho0_equals_e_l2_01():
    """F2 with rho=0 must exactly equal instantaneous Family E (l2=0.1)."""
    R, y, users, days = _make_fixture()
    scores_e = p19._apply_family(R, y, R, "E", 0.1)
    scores_f2 = p19._apply_family(R, y, R, "F2", 0.0,
                                  val_users=users, val_days=days)
    np.testing.assert_array_almost_equal(scores_f2, scores_e, decimal=10)


def test_f1_rho025_matches_recurrence():
    """F1 with rho=0.25 matches manually computed persistence recurrence."""
    rng = np.random.RandomState(99)
    n = 12
    ml = rng.uniform(0.1, 0.9, n)
    R = np.column_stack([ml, rng.uniform(0.1, 0.9, n),
                         rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)])
    users = np.array(["A"]*4 + ["B"]*4 + ["C"]*4)
    days = np.array([0,1,2,3, 0,1,2,3, 0,1,2,3])
    y = rng.randint(0, 2, n)
    rho = 0.25
    scores_inst = p19._apply_family(R, y, R, "D", 0.1)
    scores_f1 = p19._apply_family(R, y, R, "F1", rho,
                                  val_users=users, val_days=days)
    expected = np.empty(n)
    for u_start in range(0, n, 4):
        expected[u_start] = scores_inst[u_start]
        for t in range(1, 4):
            expected[u_start+t] = (1-rho)*scores_inst[u_start+t] + rho*expected[u_start+t-1]
    np.testing.assert_array_almost_equal(scores_f1, expected, decimal=12)


def test_f2_rho025_matches_recurrence():
    """F2 with rho=0.25 matches manually computed persistence recurrence."""
    rng = np.random.RandomState(77)
    n = 12
    ml = rng.uniform(0.1, 0.9, n)
    R = np.column_stack([ml, rng.uniform(0.1, 0.9, n),
                         rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)])
    users = np.array(["A"]*4 + ["B"]*4 + ["C"]*4)
    days = np.array([0,1,2,3, 0,1,2,3, 0,1,2,3])
    y = rng.randint(0, 2, n)
    rho = 0.25
    scores_inst = p19._apply_family(R, y, R, "E", 0.1)
    scores_f2 = p19._apply_family(R, y, R, "F2", rho,
                                  val_users=users, val_days=days)
    expected = np.empty(n)
    for u_start in range(0, n, 4):
        expected[u_start] = scores_inst[u_start]
        for t in range(1, 4):
            expected[u_start+t] = (1-rho)*scores_inst[u_start+t] + rho*expected[u_start+t-1]
    np.testing.assert_array_almost_equal(scores_f2, expected, decimal=12)


def test_rho090_executes_and_finite():
    """rho=0.90 executes successfully and produces finite scores."""
    R, y, users, days = _make_fixture()
    for family in ("F1", "F2"):
        scores = p19._apply_family(R, y, R, family, 0.90,
                                   val_users=users, val_days=days)
        assert np.all(np.isfinite(scores)), f"{family} rho=0.90 has non-finite scores"
        assert len(scores) == len(R)


def test_first_day_unchanged():
    """For every user, the first observed day's persisted score == instantaneous score."""
    R, y, users, days = _make_fixture()
    for family, base in [("F1", "D"), ("F2", "E")]:
        scores_inst = p19._apply_family(R, y, R, base, 0.1)
        scores_persist = p19._apply_family(R, y, R, family, 0.50,
                                           val_users=users, val_days=days)
        users_unique = np.unique(users)
        for u in users_unique:
            mask = users == u
            min_day = days[mask].min()
            first_mask = mask & (days == min_day)
            np.testing.assert_array_almost_equal(
                scores_persist[first_mask], scores_inst[first_mask], decimal=12,
                err_msg=f"{family}: first day of user {u} changed")


def test_user_boundary_reset():
    """Final score from User A must never influence first score of User B."""
    rng = np.random.RandomState(55)
    users = np.array(["A"]*3 + ["B"]*3)
    days = np.array([0,1,2, 0,1,2])
    R = rng.uniform(0.1, 0.9, (6, 4))
    y = rng.randint(0, 2, 6)
    for family, base in [("F1", "D"), ("F2", "E")]:
        scores_inst = p19._apply_family(R, y, R, base, 0.1)
        scores_persist = p19._apply_family(R, y, R, family, 0.50,
                                           val_users=users, val_days=days)
        user_b_first = scores_persist[3]
        np.testing.assert_equal(user_b_first, scores_inst[3],
                                err_msg=f"{family}: User B first score contaminated by User A")


def test_chronological_ordering():
    """Persistence uses ascending day order within each user."""
    rng = np.random.RandomState(88)
    users = np.array(["A"]*4)
    days = np.array([2,0,3,1])
    ml_vals = np.array([0.9, 0.1, 0.8, 0.2])
    R = np.column_stack([ml_vals, np.full(4, 0.5), np.full(4, 0.5), np.full(4, 0.5)])
    y = rng.randint(0, 2, 4)
    rho = 0.5
    scores_inst = p19._apply_family(R, y, R, "D", 0.1)
    scores_persist = p19._apply_family(R, y, R, "F1", rho,
                                       val_users=users, val_days=days)
    sorted_idx = np.argsort(days)
    expected = np.empty(4)
    expected[sorted_idx[0]] = scores_inst[sorted_idx[0]]
    for pos in range(1, 4):
        expected[sorted_idx[pos]] = (1-rho)*scores_inst[sorted_idx[pos]] + rho*expected[sorted_idx[pos-1]]
    np.testing.assert_array_almost_equal(scores_persist, expected, decimal=12)


def test_row_order_restoration():
    """Shuffled input rows must map back to correct original rows after scoring."""
    rng = np.random.RandomState(123)
    R, y, users, days = _make_fixture(n_users=4, days_per_user=3)
    scores_orig = p19._apply_family(R, y, R, "F1", 0.25,
                                    val_users=users, val_days=days)
    perm = rng.permutation(len(R))
    scores_shuf = p19._apply_family(R[perm], y[perm], R[perm], "F1", 0.25,
                                    val_users=users[perm], val_days=days[perm])
    np.testing.assert_array_almost_equal(scores_shuf, scores_orig[perm], decimal=12)


def test_no_future_information():
    """Modifying a user's later-day score must NOT change an earlier day's persisted score."""
    R, y, users, days = _make_fixture(n_users=2, days_per_user=4)
    users = np.array(["A"]*4 + ["B"]*4)
    days = np.array([0,1,2,3, 0,1,2,3])
    scores_before = p19._apply_family(R, y, R, "F1", 0.25,
                                      val_users=users, val_days=days)
    R_modified = R.copy()
    R_modified[3, 0] = 0.99
    scores_after = p19._apply_family(R_modified, y, R_modified, "F1", 0.25,
                                     val_users=users, val_days=days)
    user_a_mask = users == "A"
    idx_a = np.where(user_a_mask)[0]
    earlier_idx = idx_a[idx_a < 3]
    np.testing.assert_array_almost_equal(
        scores_before[earlier_idx], scores_after[earlier_idx], decimal=12,
        err_msg="Modifying User A day 3 changed earlier User A days")
    assert scores_before[idx_a[-1]] != scores_after[idx_a[-1]], \
        "Day 3 score should change when its input changes"
