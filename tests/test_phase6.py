"""Phase 6: controlled graph-vs-behavioral regression protocol tests.

Regression protection for the experiment protocol (synthetic data; the
real-data gate runs inside kaggle_scripts/run_phase6.py on Kaggle):
  1. exact feature lists (and their order) are enforced
  2. no identifier/date/target columns can enter a model input
  3. graph feature table aligns exactly with the user-day keys
  4. arms use the intended feature sets
  5. chronological split boundaries remain the frozen defaults
  6. thresholds come from CALIBRATION only (TEST not used for decisions)
  7. the reproducibility gate against the frozen phase4-b record works
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase6 as p6
from src.models.lightgbm_baseline import train_baseline
from src.preprocessing import alignment as al

FAST_PARAMS = {"learning_rate": 0.1, "num_leaves": 4, "min_data_in_leaf": 2,
               "feature_fraction": 1.0, "bagging_fraction": 1.0,
               "bagging_freq": 0, "seed": 7}

EXPECTED_A = ["login_count", "after_hours_login_count", "usb_connection_count",
              "file_access_count", "sensitive_file_access_count",
              "http_activity_count", "unique_device_count", "unusual_access_count"]
EXPECTED_B = ["device_consistency_score", "rare_device_usage_count",
              "file_type_consistency_score", "rare_file_type_access_count",
              "department_file_type_mismatch_count"]


def _synthetic_table(seed: int = 0, n_users: int = 4):
    """13-feature user-day table crossing all three chronological splits."""
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
    test_mask = grid["day"] >= pd.Timestamp("2011-04-01")
    grid.loc[test_mask & (grid["user"] == "U000")
             & (grid["day"].dt.day.isin([5, 12, 19, 26])), "is_malicious"] = 1
    return grid


def _split_table(table, feats=None):
    from src.models.lightgbm_baseline import split_datasets
    return split_datasets(table, feats or p6.ARMS["c"])


# ---------------------------------------------------------------------------
# 1. Exact feature lists and order
# ---------------------------------------------------------------------------
def test_arm_a_is_exact_frozen_behavioral_list():
    assert p6.ARMS["a"] == EXPECTED_A


def test_arm_b_is_exact_graph_list():
    assert p6.ARMS["b"] == EXPECTED_B


def test_arm_c_is_a_plus_b_in_explicit_order():
    assert p6.ARMS["c"] == p6.ARMS["a"] + p6.ARMS["b"]
    assert len(p6.ARMS["c"]) == len(set(p6.ARMS["c"])) == 13


def test_arms_derive_from_registry_order():
    assert p6.ARMS["a"] == [c for c, s in config.BEHAVIORAL_FEATURES.items()
                            if s["status"] == "defined"]
    assert p6.ARMS["b"] == list(config.GRAPH_FEATURES)


# ---------------------------------------------------------------------------
# 2. No identifier/date/target columns can enter a model
# ---------------------------------------------------------------------------
def test_arms_contain_no_identifier_or_label_columns():
    for tag in ("a", "b", "c"):
        assert not (set(p6.ARMS[tag]) & set(p6.FORBIDDEN)), tag


def test_train_baseline_rejects_identifiers_as_features():
    table = _synthetic_table()
    for bad in ("user", "day", "is_malicious"):
        with pytest.raises(ValueError, match="not registry-defined"):
            train_baseline(table, features=[bad], params=FAST_PARAMS,
                           num_boost_round=10, early_stopping_rounds=5)


def test_train_baseline_accepts_graph_features():
    table = _synthetic_table()
    model, record = train_baseline(table, features=p6.ARMS["b"], params=FAST_PARAMS,
                                   num_boost_round=30, early_stopping_rounds=10)
    assert record["features"] == p6.ARMS["b"]
    assert set(record["feature_importance_gain"]) == set(p6.ARMS["b"])


# ---------------------------------------------------------------------------
# 3. Graph table aligns exactly with user-day keys
# ---------------------------------------------------------------------------
def test_verify_alignment_identical_tables():
    table = _synthetic_table()
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = table[["user", "day"] + p6.GRAPH_ORDER]
    report = al.verify_alignment(base, extra)
    assert report["base_rows"] == report["extra_rows"] == len(table)
    assert report["common_keys"] == len(table)
    assert not report["base_only_keys"] and not report["extra_only_keys"]


def test_verify_alignment_detects_key_mismatch():
    table = _synthetic_table()
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = table[["user", "day"] + p6.GRAPH_ORDER]
    extra = extra[extra["day"] != pd.Timestamp("2011-01-05")]  # one day missing
    with pytest.raises(ValueError, match="key sets differ"):
        al.verify_alignment(base, extra)


def test_verify_alignment_detects_duplicates():
    table = _synthetic_table()
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = pd.concat([table[["user", "day"] + p6.GRAPH_ORDER]] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        al.verify_alignment(base, extra)


def test_merge_preserves_row_order_and_values():
    table = _synthetic_table()
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = table[["user", "day"] + p6.GRAPH_ORDER]
    joined = al.merge_preserving_order(base, extra, p6.GRAPH_ORDER)
    assert len(joined) == len(base)
    assert (joined[["user", "day"]].values == base[["user", "day"]].values).all()
    for c in p6.GRAPH_ORDER:
        assert (joined[c].values == table[c].values).all()


def test_merge_detects_missing_graph_values():
    table = _synthetic_table()
    base = table[["user", "day"] + p6.BEHAVIORAL_ORDER + ["is_malicious"]]
    extra = table[["user", "day"] + p6.GRAPH_ORDER]
    extra.loc[0, "device_consistency_score"] = np.nan
    with pytest.raises(ValueError, match="missing values after merge"):
        al.merge_preserving_order(base, extra, p6.GRAPH_ORDER)


# ---------------------------------------------------------------------------
# 5. Chronological split remains the frozen defaults
# ---------------------------------------------------------------------------
def test_split_defaults_frozen():
    assert config.SPLIT_DEFAULTS == {"train_end": "2011-01-31",
                                     "calibration_end": "2011-03-31"}


# ---------------------------------------------------------------------------
# 6. Thresholds from CALIBRATION only + 4. arms use intended sets
# ---------------------------------------------------------------------------
def test_protocol_end_to_end_synthetic():
    """All three arms run the identical protocol; TEST evaluated once."""
    table = _synthetic_table()
    for tag in ("a", "b", "c"):
        exp, pred, model = p6.run_arm(table, tag, params=FAST_PARAMS,
                                      num_boost_round=60, early_stopping_rounds=10)
        assert exp["features"] == p6.ARMS[tag]
        assert exp["n_features"] == len(p6.ARMS[tag])
        assert exp["split"]["strategy"] == "chronological"
        assert exp["evaluation"]["scope"].startswith("TEST only")
        # TEST predictions are exactly the TEST split rows, once
        data = _split_table(table, exp["features"])
        assert len(pred) == len(data["test"]["X"])
        assert pred["score"].isna().sum() == 0
        # threshold consistency: selected on CALIBRATION scores only
        t_recomputed, _ = th.best_f1_threshold(
            data["calibration"]["y"], model.predict(data["calibration"]["X"],
                                                    num_iteration=model.best_iteration))
        assert exp["calibration"]["threshold_max_f1"] == t_recomputed
        # full protocol metric set present
        row = p6.comparison_row(exp)
        for key in ("auc_roc", "auc_pr", "precision_at_10", "precision_at_30",
                    "recall_at_50", "precision", "recall", "f1", "mcc",
                    "balanced_accuracy", "fpr", "fnr", "alert_rate",
                    "best_iteration", "n_features"):
            assert key in row, key


# ---------------------------------------------------------------------------
# 7. Reproducibility gate vs the frozen phase4-b record
# ---------------------------------------------------------------------------
def test_repro_gate_passes_exact_match():
    measured = {"auc_roc": p6.REF_A["auc_roc"], "auc_pr": p6.REF_A["auc_pr"],
                "best_iteration": p6.REF_A["best_iteration"]}
    assert p6.check_repro_gate(measured) == []


def test_repro_gate_fails_on_auc_drift():
    measured = {"auc_roc": p6.REF_A["auc_roc"] + 2e-6,
                "auc_pr": p6.REF_A["auc_pr"],
                "best_iteration": p6.REF_A["best_iteration"]}
    problems = p6.check_repro_gate(measured)
    assert any("auc_roc" in p for p in problems)


def test_repro_gate_fails_on_iteration_drift():
    measured = {"auc_roc": p6.REF_A["auc_roc"], "auc_pr": p6.REF_A["auc_pr"],
                "best_iteration": p6.REF_A["best_iteration"] + 1}
    problems = p6.check_repro_gate(measured)
    assert any("best_iteration" in p for p in problems)


# ---------------------------------------------------------------------------
# Paired bootstrap delta (evaluation-only uncertainty)
# ---------------------------------------------------------------------------
def test_paired_bootstrap_delta_identical_scores():
    rng = np.random.default_rng(0)
    y = np.array([1] * 5 + [0] * 95)
    s = rng.uniform(0, 1, 100)
    res = m.paired_bootstrap_delta(y, s, s, n_boot=200, seed=42)
    for metric in ("auc_roc", "auc_pr"):
        assert abs(res[metric]["mean"]) < 1e-6
        assert res[metric]["frac_gt_0"] <= 0.05  # every delta is exactly 0
        assert res[metric]["ci_low"] <= res[metric]["mean"] <= res[metric]["ci_high"]


def test_paired_bootstrap_delta_better_scores_positive():
    rng = np.random.default_rng(1)
    y = np.array([1] * 5 + [0] * 95)
    s_base = rng.uniform(0, 0.7, 100)
    s_new = s_base.copy()
    s_new[y == 1] += 0.25
    res = m.paired_bootstrap_delta(y, s_base, s_new, n_boot=300, seed=42)
    assert res["auc_roc"]["mean"] > 0
    assert res["auc_roc"]["frac_gt_0"] > 0.9