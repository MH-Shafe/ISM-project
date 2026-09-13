"""Baseline model + evaluation tests (synthetic data, no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from src.data.aggregation import assemble_user_day_table
from src.data.labels import build_user_day_labels
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.models.lightgbm_baseline import (
    model_features,
    predict_proba,
    split_datasets,
    train_baseline,
)


def _labeled_table(synthetic_logs, users, days=None):
    if days is None:
        days = pd.date_range("2010-01-04", "2010-01-05", freq="D")
    table = assemble_user_day_table(synthetic_logs, users, days)
    insiders = pd.DataFrame([
        # AAA0001 malicious in TRAIN (2010-01-04..05)
        {"dataset": 4.2, "scenario": 1, "details": "r4.2-1-x.csv", "user": "AAA0001",
         "start_dt": pd.Timestamp("2010-01-04 00:00:00"),
         "end_dt": pd.Timestamp("2010-01-05 23:59:59")},
        # BBB0002 malicious in CALIBRATION and TEST (chronological protocol)
        {"dataset": 4.2, "scenario": 2, "details": "r4.2-2-x.csv", "user": "BBB0002",
         "start_dt": pd.Timestamp("2011-02-10 00:00:00"),
         "end_dt": pd.Timestamp("2011-02-10 23:59:59")},
        {"dataset": 4.2, "scenario": 3, "details": "r4.2-3-x.csv", "user": "BBB0002",
         "start_dt": pd.Timestamp("2011-04-15 00:00:00"),
         "end_dt": pd.Timestamp("2011-04-15 23:59:59")},
        {"dataset": 3.1, "scenario": 2, "details": "other.csv", "user": "ZZZ0009",
         "start_dt": pd.Timestamp("2010-01-01 00:00:00"),
         "end_dt": pd.Timestamp("2010-01-01 23:59:59")},
    ])
    lab = build_user_day_labels(users, days, insiders=insiders)
    return table.merge(lab, on=["user", "day"])


# ---------------------------------------------------------------------------
# Feature discipline (leakage rule: explicit feature list only)
# ---------------------------------------------------------------------------
def test_model_features_exact_registry_list(synthetic_logs, users):
    table = _labeled_table(synthetic_logs, users)
    feats = model_features(table)
    assert feats == ["login_count", "after_hours_login_count",
                     "usb_connection_count", "file_access_count",
                     "sensitive_file_access_count", "http_activity_count",
                     "unique_device_count", "unusual_access_count"]


def test_model_features_rejects_extra_columns(synthetic_logs, users):
    table = _labeled_table(synthetic_logs, users)
    table["sneaky_future_stat"] = 1
    with pytest.raises(ValueError, match="possible leakage"):
        model_features(table)


def test_model_features_rejects_missing_feature(synthetic_logs, users):
    table = _labeled_table(synthetic_logs, users).drop(columns=["login_count"])
    with pytest.raises(ValueError, match="missing from table"):
        model_features(table)


# ---------------------------------------------------------------------------
# Metrics (known values)
# ---------------------------------------------------------------------------
def test_classification_metrics_known():
    y = np.array([1, 1, 0, 0])          # positives ranked first
    s = np.array([0.9, 0.6, 0.4, 0.1])
    res = m.classification_metrics(y, s)
    assert res["positives"] == 2
    assert res["auc_roc"] == pytest.approx(1.0)
    assert res["auc_pr"] == pytest.approx(1.0)  # AP: precisions 1.0, 1.0


def test_top_k_metrics_known():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.4, 0.1])
    res = m.top_k_metrics(y, s, ks=(1, 2))
    assert res["precision_at_1"] == pytest.approx(1.0)
    assert res["precision_at_2"] == pytest.approx(1.0)
    assert res["recall_at_2"] == pytest.approx(1.0)


def test_confusion_at_threshold_known():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.4, 0.1])
    res = m.confusion_at_threshold(y, s, 0.5)
    assert res["tp"] == 2 and res["fp"] == 0 and res["fn"] == 0
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["n_alerts"] == 2


def test_binary_decision_metrics_perfect():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.6, 0.4, 0.1])
    res = m.binary_decision_metrics(y, s, 0.5)
    assert res["tp"] == 2 and res["tn"] == 2
    assert res["fp"] == 0 and res["fn"] == 0
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(1.0)
    assert res["mcc"] == pytest.approx(1.0)
    assert res["balanced_accuracy"] == pytest.approx(1.0)
    assert res["fpr"] == pytest.approx(0.0)
    assert res["fnr"] == pytest.approx(0.0)
    assert res["alert_rate"] == pytest.approx(0.5)


def test_binary_decision_metrics_imperfect():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    res = m.binary_decision_metrics(y, s, 0.5)
    assert res["tp"] == 1 and res["fp"] == 1
    assert res["fn"] == 1 and res["tn"] == 1
    assert res["precision"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(0.5)
    assert res["f1"] == pytest.approx(0.5)
    assert res["mcc"] == pytest.approx(0.0)          # 1*1 - 1*1 over balanced denom
    assert res["balanced_accuracy"] == pytest.approx(0.5)
    assert res["fpr"] == pytest.approx(0.5)
    assert res["fnr"] == pytest.approx(0.5)
    assert res["alert_rate"] == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 47000)
    s = np.concatenate([rng.uniform(0.5, 1.0, 30), rng.uniform(0.0, 0.6, 47000)])
    point = m.classification_metrics(y, s)["auc_roc"]
    res = m.bootstrap_ci(y, s, metric="auc_roc", n_boot=200, seed=42)
    assert res["n_boot"] == 200
    assert 0 < res["n_valid"] <= 200
    assert res["ci_low"] <= res["mean"] <= res["ci_high"]
    assert res["ci_low"] <= point <= res["ci_high"]


# ---------------------------------------------------------------------------
# Threshold selection (calibration-only contract)
# ---------------------------------------------------------------------------
def test_best_f1_threshold_known():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.7, 0.3, 0.2])
    t, f1 = th.best_f1_threshold(y, s)
    assert t == pytest.approx(0.7)  # prec 1.0, rec 1.0
    assert f1 == pytest.approx(1.0)


def test_threshold_at_precision_known():
    y = np.array([0, 1, 1])
    s = np.array([0.9, 0.7, 0.5])
    t = th.threshold_at_precision(y, s, min_precision=0.5)
    assert t == pytest.approx(0.7)  # top1: 0/1 < 0.5; top2: 1/2 >= 0.5


# ---------------------------------------------------------------------------
# End-to-end smoke (tiny synthetic train across all three splits)
# ---------------------------------------------------------------------------
def test_train_baseline_smoke(synthetic_logs, users):
    days = pd.date_range("2010-01-01", "2011-05-20", freq="D")
    table = _labeled_table(synthetic_logs, users, days)
    params = {"learning_rate": 0.1, "num_leaves": 4, "min_data_in_leaf": 2,
              "feature_fraction": 1.0, "bagging_fraction": 1.0,
              "bagging_freq": 0, "seed": 7}
    model, record = train_baseline(table, params=params,
                                   num_boost_round=50, early_stopping_rounds=10)
    assert record["features"] == model_features(table)
    assert record["params"]["scale_pos_weight"] > 0
    assert 0 < record["best_iteration"] <= 50
    assert record["calibration"]["malicious"] == 1  # BBB0002 on 2011-02-10
    probs = predict_proba(model, table[record["features"]].astype(np.float32))
    assert ((probs >= 0) & (probs <= 1)).all()

    data = split_datasets(table, record["features"])
    s_cal = predict_proba(model, data["calibration"]["X"])
    t, _ = th.best_f1_threshold(data["calibration"]["y"], s_cal)
    s_test = predict_proba(model, data["test"]["X"])
    res = m.classification_metrics(data["test"]["y"], s_test)
    assert 0.0 <= res["auc_roc"] <= 1.0
    assert 0.0 <= res["auc_pr"] <= 1.0
    conf = m.confusion_at_threshold(data["test"]["y"], s_test, t)
    assert conf["n_alerts"] >= 0
    assert conf["threshold"] == t


def test_train_baseline_feature_subsets(synthetic_logs, users):
    """Phase 4: ablation arms must train on explicit registry subsets."""
    days = pd.date_range("2010-01-01", "2011-05-20", freq="D")
    table = _labeled_table(synthetic_logs, users, days)
    params = {"learning_rate": 0.1, "num_leaves": 4, "min_data_in_leaf": 2,
              "feature_fraction": 1.0, "bagging_fraction": 1.0,
              "bagging_freq": 0, "seed": 7}
    base6 = ["login_count", "after_hours_login_count", "usb_connection_count",
             "file_access_count", "http_activity_count", "unique_device_count"]
    subsets = [
        base6,
        base6 + ["sensitive_file_access_count"],
        base6 + ["unusual_access_count"],
        model_features(table),
    ]
    for feats in subsets:
        model, record = train_baseline(table, features=feats, params=params,
                                       num_boost_round=50, early_stopping_rounds=10)
        assert record["features"] == feats
        assert record["feature_importance_gain"].keys() == set(feats)


def test_train_baseline_rejects_undefined_feature(synthetic_logs, users):
    days = pd.date_range("2010-01-01", "2011-05-20", freq="D")
    table = _labeled_table(synthetic_logs, users, days)
    with pytest.raises(ValueError, match="not registry-defined"):
        train_baseline(table, features=["login_count", "made_up_feature"],
                       num_boost_round=10, early_stopping_rounds=5)