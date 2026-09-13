"""Phase 7: robustness/freeze protocol regression tests (synthetic data).

Protects the frozen candidate configuration and the calibration-only
robustness protocol:
  1. seed sweep evaluates TRAIN/CALIBRATION only (never TEST)
  2. deterministic reproduction with the frozen seed
  3. no identifier/date/target leakage in the candidate features
  4. chronological split remains the frozen defaults
  5. department-mismatch decision rule is deterministic and evidence-based
  6. the frozen configuration matches the freeze record artifact
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config
from src.experiments import phase6 as p6
from src.experiments import phase7 as p7
from src.models.lightgbm_baseline import split_datasets, train_baseline

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
    cal = (grid["day"] >= pd.Timestamp("2011-02-01")) & \
          (grid["day"] <= pd.Timestamp("2011-03-31"))
    grid.loc[cal & (grid["user"] == "U001")
             & (grid["day"].dt.day.isin([5, 12])), "is_malicious"] = 1
    test = grid["day"] >= pd.Timestamp("2011-04-01")
    grid.loc[test & (grid["user"] == "U000")
             & (grid["day"].dt.day.isin([5, 12, 19, 26])), "is_malicious"] = 1
    return grid


# ---------------------------------------------------------------------------
# 1. Seed sweep is CALIBRATION-only
# ---------------------------------------------------------------------------
def test_seed_sweep_produces_calibration_only_fields():
    table = _synthetic_table()
    sweep = p7.seed_sweep(table, p6.ARMS["c"],
                          seeds=[7, 42])  # tiny sweep; all 13 features
    for run in sweep:
        for key in ("seed", "best_iteration", "cal_auc_roc", "cal_auc_pr",
                    "threshold_max_f1", "cal_alert_rate", "cal_precision",
                    "cal_recall", "cal_f1", "cal_mcc", "train_runtime_s"):
            assert key in run, key
        # structural: no TEST-derived field exists in the sweep record
        assert not any("test" in k.lower() for k in run)
        assert 0.0 <= run["cal_auc_roc"] <= 1.0
        assert 0.0 <= run["cal_auc_pr"] <= 1.0


def test_seed_sweep_threshold_consistent_with_calibration():
    """Threshold recorded per seed equals best_f1 on that seed's CAL scores."""
    table = _synthetic_table()
    sweep = p7.seed_sweep(table, p6.ARMS["c"], seeds=[7])
    run = sweep[0]
    model, record = train_baseline(table, features=p6.ARMS["c"],
                                   params={"seed": 7})
    data = split_datasets(table, p6.ARMS["c"])
    s_cal = model.predict(data["calibration"]["X"],
                          num_iteration=model.best_iteration)
    from src.evaluation import threshold as th
    t, _ = th.best_f1_threshold(data["calibration"]["y"], s_cal)
    assert run["threshold_max_f1"] == t


# ---------------------------------------------------------------------------
# 2. Deterministic reproduction with the frozen seed
# ---------------------------------------------------------------------------
def test_frozen_seed_reproduction_is_deterministic():
    table = _synthetic_table()
    feats = list(p6.ARMS["c"])
    m1, r1 = train_baseline(table, features=feats, params=dict(FAST_PARAMS))
    m2, r2 = train_baseline(table, features=feats, params=dict(FAST_PARAMS))
    d1 = split_datasets(table, feats)
    d2 = split_datasets(table, feats)
    s1 = m1.predict(d1["calibration"]["X"], num_iteration=m1.best_iteration)
    s2 = m2.predict(d2["calibration"]["X"], num_iteration=m2.best_iteration)
    np.testing.assert_array_equal(s1, s2)
    assert r1["best_iteration"] == r2["best_iteration"]


def test_summarize_statistics():
    res = p7.summarize([1, 2, 3, 4])
    assert res["mean"] == 2.5
    assert res["min"] == 1.0 and res["max"] == 4.0
    assert res["cv"] == pytest.approx(0.516397779, rel=1e-3)
    assert p7.summarize([5, 5])["cv"] == 0.0


# ---------------------------------------------------------------------------
# 3. No identifier/date/target leakage in the candidate
# ---------------------------------------------------------------------------
def test_candidate_features_have_no_forbidden_columns():
    for feats in (p6.ARMS["c"], list(p7.FROZEN_CONFIG["features"])):
        assert not (set(feats) & set(p6.FORBIDDEN))


# ---------------------------------------------------------------------------
# 4. Chronological split frozen
# ---------------------------------------------------------------------------
def test_split_defaults_frozen():
    assert config.SPLIT_DEFAULTS == {"train_end": "2011-01-31",
                                     "calibration_end": "2011-03-31"}


# ---------------------------------------------------------------------------
# 5. Department-mismatch decision rule (deterministic, calibration-based)
# ---------------------------------------------------------------------------
def test_department_decision_reject_when_all_conditions_met():
    ev = {
        "phase6_zero_gain": True,
        "single_feature_cal": {"auc_pr": 0.005},
        "drop_delta_12_vs_13_cal": {"auc_roc": 0.0001, "auc_pr": 0.0002},
        "distribution": {"nonzero_frac": 0.02},
    }
    assert p7.decide_department(ev) == "REJECT"


def test_department_decision_retain_when_signal_exists():
    ev = {
        "phase6_zero_gain": True,
        "single_feature_cal": {"auc_pr": 0.07},
        "drop_delta_12_vs_13_cal": {"auc_roc": 0.0001, "auc_pr": 0.0002},
        "distribution": {"nonzero_frac": 0.5},
    }
    assert p7.decide_department(ev) == "RETAIN"


def test_department_decision_retain_when_removal_hurts():
    ev = {
        "phase6_zero_gain": True,
        "single_feature_cal": {"auc_pr": 0.003},
        "drop_delta_12_vs_13_cal": {"auc_roc": -0.002, "auc_pr": -0.003},
        "distribution": {"nonzero_frac": 0.02},
    }
    assert p7.decide_department(ev) == "RETAIN"


def test_department_decision_inconclusive_when_ambiguous():
    ev = {
        "phase6_zero_gain": True,
        "single_feature_cal": {"auc_pr": 0.02},
        "drop_delta_12_vs_13_cal": {"auc_roc": 0.0005, "auc_pr": 0.0005},
        "distribution": {"nonzero_frac": 0.05},
    }
    assert p7.decide_department(ev) == "INCONCLUSIVE"


def test_department_decision_inconclusive_without_zero_gain():
    ev = {
        "phase6_zero_gain": False,
        "single_feature_cal": {"auc_pr": 0.0},
        "drop_delta_12_vs_13_cal": {"auc_roc": 0.0, "auc_pr": 0.0},
        "distribution": {"nonzero_frac": 0.0},
    }
    assert p7.decide_department(ev) == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# 6. Frozen configuration matches the freeze record artifact
# ---------------------------------------------------------------------------
def test_frozen_config_is_finalized():
    feats = list(p7.FROZEN_CONFIG["features"])
    if not feats:
        pytest.skip("FROZEN_CONFIG not finalized yet (pre-evidence)")
    assert feats, "FROZEN_CONFIG['features'] must be finalized"
    assert len(set(feats)) == len(feats)
    assert p7.FROZEN_CONFIG["seed"] == 42
    assert p7.FROZEN_CONFIG["params"]["seed"] == 42
    assert p7.FROZEN_CONFIG["split"] == dict(config.SPLIT_DEFAULTS)


def test_frozen_config_features_registry_valid():
    feats = list(p7.FROZEN_CONFIG["features"])
    if not feats:
        pytest.skip("FROZEN_CONFIG not finalized yet (pre-evidence)")
    registered = {c for c, s in config.BEHAVIORAL_FEATURES.items()
                  if s["status"] == "defined"} | set(config.GRAPH_FEATURES)
    assert set(feats) <= registered
    assert feats[:8] == p6.BEHAVIORAL_ORDER  # behavioral block unchanged, in order
    assert set(feats[8:]) <= set(p6.GRAPH_ORDER)  # graph block, registry subset


RECORD_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "artifacts",
                           "phase7_freeze_lgbm-graph-v1.json")


def test_freeze_record_matches_config():
    if not os.path.isfile(RECORD_PATH):
        pytest.skip("freeze record not present yet")
    with open(RECORD_PATH) as fh:
        record = json.load(fh)
    assert record["model_name"] == p7.FROZEN_NAME
    assert record["features"] == list(p7.FROZEN_CONFIG["features"])
    assert record["model"]["seed"] == p7.FROZEN_CONFIG["seed"]
    assert record["model"]["params"] == p7.FROZEN_CONFIG["params"]
    assert record["model"]["early_stopping"] == p7.FROZEN_CONFIG["early_stopping"]
    assert record["calibration"]["method"] == p7.FROZEN_CONFIG["calibration_method"]
    assert record["split"]["strategy"] == p7.FROZEN_CONFIG["split_strategy"]
    # TEST evaluated exactly once, after freeze: record must carry TEST metrics
    assert record["evaluation"]["classification"]["positives"] >= 0
    assert record["evaluation"]["scope"].startswith("TEST only")


# ---------------------------------------------------------------------------
# Graph alignment remains exact for the candidate
# ---------------------------------------------------------------------------
def test_graph_alignment_exact_for_candidate():
    table = _synthetic_table()
    from src.preprocessing import alignment as al
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = table[["user", "day"] + p6.GRAPH_ORDER]
    report = al.verify_alignment(base, extra)
    assert report["common_keys"] == len(table)
    joined = al.merge_preserving_order(base, extra, p6.GRAPH_ORDER)
    assert (joined[["user", "day"]].values == base[["user", "day"]].values).all()


def test_gain_shares_normalize_to_one():
    imp = {"a": 3.0, "b": 1.0, "c": 0.0}
    shares = p7.gain_shares(imp)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["c"] == 0.0


def test_kendall_tau_known():
    assert p7.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert p7.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    assert p7.kendall_tau(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == pytest.approx(2 / 3)