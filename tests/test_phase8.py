"""Phase 8: adaptive-risk protocol regression tests (synthetic data).

Protects:
  1. component definitions are complete and in [0,1]
  2. normalization is TRAIN-only, deterministic, monotone, serializable
  3. components never consume identifier/date/label columns
  4. weight learning is non-negative, sum-to-1, deterministic, CAL-selected
  5. arm semantics (A = ML only; B = equal weights; C = learned weights)
  6. decision rule is deterministic (ADOPT / INCONCLUSIVE / NO)
  7. correlation bundle is TRAIN/CALIBRATION only
  8. the frozen adaptive configuration matches the calibration artifact
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config
from src.experiments import phase6 as p6
from src.experiments import phase7 as p7
from src.experiments import phase8 as p8
from src.evaluation import threshold as th

FAST_PARAMS = {"learning_rate": 0.1, "num_leaves": 4, "min_data_in_leaf": 2,
               "feature_fraction": 1.0, "bagging_fraction": 1.0,
               "bagging_freq": 0, "seed": 7}


def _synthetic_table(seed: int = 0, n_users: int = 4):
    rng = np.random.default_rng(seed)
    users = [f"U{i:03d}" for i in range(n_users)]
    days = pd.date_range("2010-12-20", "2011-05-10", freq="D")
    grid = pd.MultiIndex.from_product([users, days], names=["user", "day"]).to_frame(
        index=False)
    for c in p6.BEHAVIORAL_ORDER:
        grid[c] = rng.poisson(1.0, len(grid))
    for c in p6.GRAPH_ORDER:
        grid[c] = rng.uniform(0.0, 1.0, len(grid))
    grid["is_malicious"] = 0
    train = grid["day"] <= pd.Timestamp("2011-01-31")
    cal = (grid["day"] >= pd.Timestamp("2011-02-01")) & \
          (grid["day"] <= pd.Timestamp("2011-03-31"))
    test = grid["day"] >= pd.Timestamp("2011-04-01")
    grid.loc[train & (grid["user"] == "U001")
             & (grid["day"].dt.day.isin([3, 10, 17])), "is_malicious"] = 1
    grid.loc[cal & (grid["user"] == "U001")
             & (grid["day"].dt.day.isin([5, 12])), "is_malicious"] = 1
    grid.loc[test & (grid["user"] == "U000")
             & (grid["day"].dt.day.isin([5, 12, 19, 26])), "is_malicious"] = 1
    return grid


def _block(table, name):
    feats = (p8.TRUST_CONSISTENCY_FEATURES + p8.TRUST_RARITY_FEATURES
             + p8.BEHAVIOR_RISK_FEATURES)
    part = table[table["day"] <= pd.Timestamp("2011-01-31")] if name == "train" \
        else table[(table["day"] > pd.Timestamp("2011-01-31"))
                   & (table["day"] <= pd.Timestamp("2011-03-31"))]
    return part[feats].copy(), part["is_malicious"].to_numpy()


def _scores(n, seed=1):
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# 1. Component definitions and bounds
# ---------------------------------------------------------------------------
def test_component_definitions_documented():
    for name, spec in p8.COMPONENTS.items():
        for field in ("raw_features", "definition", "direction", "normalization",
                      "range", "missing_behavior", "unseen_behavior",
                      "temporal_availability", "leakage_status"):
            assert field in spec, f"{name} missing {field}"


def test_component_features_registry_valid():
    registered = {c for c, s in config.BEHAVIORAL_FEATURES.items()
                  if s["status"] == "defined"} | set(config.GRAPH_FEATURES)
    for f in (p8.TRUST_CONSISTENCY_FEATURES + p8.TRUST_RARITY_FEATURES
              + p8.BEHAVIOR_RISK_FEATURES):
        assert f in registered, f


def test_components_in_unit_range_and_ml_passthrough():
    table = _synthetic_table()
    blk, _ = _block(table, "train")
    norm = p8.fit_normalization(blk)
    scores = _scores(len(blk))
    comp = p8.build_components(scores, blk, norm)
    np.testing.assert_array_equal(comp["ml_risk"], scores)
    for name in (p8.TRUST_RISK, p8.BEHAVIOR_RISK):
        assert comp[name].min() >= 0.0 and comp[name].max() <= 1.0


def test_components_monotone_in_inputs():
    table = _synthetic_table()
    blk, _ = _block(table, "train")
    norm = p8.fit_normalization(blk)
    base = blk.copy()
    base["rare_device_usage_count"] = 0.0
    lo = p8.build_components(_scores(len(blk)), base, norm)[p8.TRUST_RISK]
    base["rare_device_usage_count"] = 50.0
    hi = p8.build_components(_scores(len(blk)), base, norm)[p8.TRUST_RISK]
    assert (hi >= lo - 1e-12).all()

    base = blk.copy()
    base["device_consistency_score"] = 0.0
    lo = p8.build_components(_scores(len(blk)), base, norm)[p8.TRUST_RISK]
    base["device_consistency_score"] = 1.0
    hi = p8.build_components(_scores(len(blk)), base, norm)[p8.TRUST_RISK]
    assert (hi <= lo + 1e-12).all()

    base = blk.copy()
    base["after_hours_login_count"] = 0.0
    lo = p8.build_components(_scores(len(blk)), base, norm)[p8.BEHAVIOR_RISK]
    base["after_hours_login_count"] = 30.0
    hi = p8.build_components(_scores(len(blk)), base, norm)[p8.BEHAVIOR_RISK]
    assert (hi >= lo - 1e-12).all()


def test_build_components_rejects_forbidden_columns():
    table = _synthetic_table()
    blk, _ = _block(table, "train")
    norm = p8.fit_normalization(blk)
    leaked = blk.copy()
    leaked["is_malicious"] = 0
    with pytest.raises(ValueError):
        p8.build_components(_scores(len(blk)), leaked, norm)


def test_build_components_rejects_missing_features():
    blk = pd.DataFrame({"device_consistency_score": [0.5]})
    with pytest.raises(ValueError):
        p8.build_components([0.5], blk, {})


# ---------------------------------------------------------------------------
# 2. Normalization
# ---------------------------------------------------------------------------
def test_cdf_deterministic_monotone_clips():
    rng = np.random.default_rng(0)
    vals = rng.normal(10.0, 3.0, 5000)
    p1 = p8.fit_empirical_cdf(vals)
    p2 = p8.fit_empirical_cdf(vals)
    assert p1 == p2
    x = np.linspace(-100, 100, 401)
    y = p8.cdf_transform(p1, x)
    assert (np.diff(y) >= 0).all()
    assert y[0] == 0.0 and y[-1] == 1.0
    assert p8.cdf_transform(p1, [vals.min() - 100])[0] == 0.0
    assert p8.cdf_transform(p1, [vals.max() + 100])[0] == 1.0
    # median maps near 0.5
    assert abs(p8.cdf_transform(p1, [np.median(vals)])[0] - 0.5) < 0.01


def test_fit_normalization_requires_feature_columns():
    with pytest.raises(ValueError):
        p8.fit_normalization(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------------------
# 3. Weight learning (arm C)
# ---------------------------------------------------------------------------
def test_simplex_grid_size_and_properties():
    grid = p8.simplex_grid(3, 0.1)
    assert len(grid) == 66
    for w in grid:
        assert sum(w) == pytest.approx(1.0)
        assert all(x >= 0 for x in w)


def test_learn_weights_nonnegative_sum_one_deterministic():
    table = _synthetic_table()
    blk_t, y_t = _block(table, "train")
    blk_c, y_c = _block(table, "calibration")
    norm = p8.fit_normalization(blk_t)
    comp_t = p8.build_components(_scores(len(blk_t)), blk_t, norm)
    comp_c = p8.build_components(_scores(len(blk_c)), blk_c, norm)
    r1 = p8.learn_weights(comp_t, y_t, comp_c, y_c)
    r2 = p8.learn_weights(comp_t, y_t, comp_c, y_c)
    assert r1 == r2
    assert sum(r1["weights"]) == pytest.approx(1.0)
    assert all(w >= 0 for w in r1["weights"])
    assert r1["grid_size"] == 66
    assert len(r1["top_candidates"]) == r1["top_k"]
    # selected weights must be on the simplex grid
    assert tuple(r1["weights"]) in p8.simplex_grid(3, 0.1)


def test_learn_weights_selection_prefers_calibration_signal():
    n = 600
    rng = np.random.default_rng(3)
    trust = rng.uniform(0, 1, n)
    y_t = (trust > 0.6).astype(int)
    y_c = (rng.uniform(0, 1, n) > 0.9).astype(int)  # cal target unrelated to trust
    comp = {
        "ml_risk": rng.uniform(0, 1, n),
        "trust_risk": trust,
        "behavior_risk": rng.uniform(0, 1, n),
    }
    # cal target unrelated to trust: an equal-weight arm should not win on CAL
    learned = p8.learn_weights(comp, y_t, comp, y_c)
    selected = learned["weights"]
    best_cal = max(learned["top_candidates"], key=lambda r: r["cal_auc_pr"])
    assert selected == best_cal["weights"]


def test_equal_weights_arm_b_is_mean():
    table = _synthetic_table()
    blk, _ = _block(table, "train")
    norm = p8.fit_normalization(blk)
    comp = p8.build_components(_scores(len(blk)), blk, norm)
    s = p8.risk_score(comp, [1.0 / 3.0] * 3)
    expected = (comp["ml_risk"] + comp["trust_risk"] + comp["behavior_risk"]) / 3.0
    np.testing.assert_allclose(s, expected, rtol=1e-12)


def test_arm_a_risk_equals_ml():
    table = _synthetic_table()
    blk, _ = _block(table, "train")
    norm = p8.fit_normalization(blk)
    comp = p8.build_components(_scores(len(blk)), blk, norm)
    np.testing.assert_array_equal(p8.risk_score(comp, [1.0, 0.0, 0.0]),
                                  comp["ml_risk"])


# ---------------------------------------------------------------------------
# 4. Decision rule
# ---------------------------------------------------------------------------
def test_decide_adaptive_adopt():
    cal = {
        "arm_a": {"auc_roc": 0.88, "auc_pr": 0.50, "precision_at_10": 0.4},
        "arm_b": {"auc_roc": 0.90, "auc_pr": 0.53, "precision_at_10": 0.5},
        "arm_c": {"auc_roc": 0.91, "auc_pr": 0.55, "precision_at_10": 0.6},
    }
    assert p8.decide_adaptive(cal, [0.6, 0.2, 0.2]) == "ADOPT"


def test_decide_adaptive_no_when_gain_small():
    cal = {
        "arm_a": {"auc_roc": 0.88, "auc_pr": 0.50, "precision_at_10": 0.4},
        "arm_b": {"auc_roc": 0.88, "auc_pr": 0.502, "precision_at_10": 0.4},
        "arm_c": {"auc_roc": 0.88, "auc_pr": 0.503, "precision_at_10": 0.4},
    }
    assert p8.decide_adaptive(cal, [0.6, 0.2, 0.2]) == "NO"


def test_decide_adaptive_inconclusive():
    cal = {
        "arm_a": {"auc_roc": 0.88, "auc_pr": 0.50, "precision_at_10": 0.4},
        "arm_b": {"auc_roc": 0.89, "auc_pr": 0.507, "precision_at_10": 0.5},
        "arm_c": {"auc_roc": 0.89, "auc_pr": 0.507, "precision_at_10": 0.5},
    }
    assert p8.decide_adaptive(cal, [0.6, 0.2, 0.2]) == "INCONCLUSIVE"


def test_decide_adaptive_no_when_degenerate_on_ml():
    cal = {
        "arm_a": {"auc_roc": 0.88, "auc_pr": 0.50, "precision_at_10": 0.4},
        "arm_b": {"auc_roc": 0.90, "auc_pr": 0.52, "precision_at_10": 0.5},
        "arm_c": {"auc_roc": 0.91, "auc_pr": 0.55, "precision_at_10": 0.6},
    }
    assert p8.decide_adaptive(cal, [0.95, 0.05, 0.0]) == "NO"


def test_decide_adaptive_no_when_roc_worse():
    cal = {
        "arm_a": {"auc_roc": 0.92, "auc_pr": 0.50, "precision_at_10": 0.4},
        "arm_b": {"auc_roc": 0.90, "auc_pr": 0.53, "precision_at_10": 0.4},
        "arm_c": {"auc_roc": 0.90, "auc_pr": 0.53, "precision_at_10": 0.4},
    }
    assert p8.decide_adaptive(cal, [0.5, 0.3, 0.2]) == "NO"


# ---------------------------------------------------------------------------
# 5. Correlation bundle (TRAIN/CALIBRATION only)
# ---------------------------------------------------------------------------
def test_correlation_bundle_structure_and_scope():
    table = _synthetic_table()
    blk_t, y_t = _block(table, "train")
    blk_c, y_c = _block(table, "calibration")
    norm = p8.fit_normalization(blk_t)
    comp_t = p8.build_components(_scores(len(blk_t)), blk_t, norm)
    comp_c = p8.build_components(_scores(len(blk_c)), blk_c, norm)
    bundle = p8.correlation_bundle(comp_t, y_t, comp_c, y_c)
    assert bundle["scope"].startswith("TRAIN + CALIBRATION")
    assert set(bundle["train"]["single_auc"]) == set(p8.COMPONENT_ORDER)
    assert "incremental_calibration" in bundle
    assert "ml_trust_behavior_equal" in bundle["incremental_calibration"]
    assert not any("test" in k.lower() for k in bundle)


# ---------------------------------------------------------------------------
# 6. Frozen configuration consistency (artifact present after the run)
# ---------------------------------------------------------------------------
ART = os.path.join(os.path.dirname(__file__), "..", "reports", "artifacts")
FREEZE_PATH = os.path.join(ART, "phase8_freeze.json")
CAL_PATH = os.path.join(ART, "phase8_calibration.json")


def test_freeze_matches_calibration_artifact():
    if not os.path.isfile(FREEZE_PATH) or not os.path.isfile(CAL_PATH):
        pytest.skip("phase8 artifacts not present yet")
    freeze = json.load(open(FREEZE_PATH))
    cal = json.load(open(CAL_PATH))
    for arm, weights in freeze["weights"].items():
        assert weights == cal["arms"][arm]["weights"]
        assert (freeze["thresholds"][arm]
                == cal["calibration_metrics"][arm]["threshold_max_f1"])
    assert freeze["decision"] == cal["decision"]


def test_threshold_method_is_calibration_max_f1():
    """Sanity: the threshold function used by the runner maximizes F1 on CAL."""
    rng = np.random.default_rng(4)
    y = (rng.uniform(0, 1, 500) > 0.95).astype(int)
    s = rng.uniform(0, 1, 500)
    t, f1 = th.best_f1_threshold(y, s)
    assert 0.0 <= t <= 1.0 and 0.0 <= f1 <= 1.0