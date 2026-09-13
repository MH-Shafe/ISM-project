"""Phase 9: alert-prioritization + threshold-stability protocol regression tests
(synthetic data).

Protects:
  1. candidate policy definitions are complete, a priori, and deterministic
  2. threshold / daily-top-k / percentile masks are correct (hand-computed)
  3. mask-based decision metrics match the shared metric family
  4. per-scenario recall aggregation is correct
  5. chronological windows are disjoint, ordered, and cover all CALIBRATION rows
  6. threshold bootstrap mirrors th.best_f1_threshold / threshold_at_precision
     and is deterministic
  7. user alert concentration (shares + Gini) is correct
  8. user-level aggregation is diagnostic-only and never returns labels as
     inputs to anything
  9. selection rule: screens S1/S2/S3 + F1 selection are deterministic and
     correctly exclude unstable candidates
  10. frozen policy artifacts are internally consistent (present after a run)
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef

from src.evaluation import threshold as th
from src.experiments import phase9 as p9

FROZEN = 0.9186015432508062


# ---------------------------------------------------------------------------
# 1. Candidate definitions
# ---------------------------------------------------------------------------
def test_build_candidates_complete_and_a_priori():
    cands = p9.build_candidates(frozen_threshold=FROZEN, t_p50=0.7)
    ids = [c["id"] for c in cands]
    assert len(cands) == 13
    assert len(set(ids)) == 13
    assert cands[0] == {"id": "frozen_max_f1", "kind": "threshold",
                        "param": FROZEN, "name": p9.build_candidates(
                            frozen_threshold=FROZEN, t_p50=0.7)[0]["name"]}
    assert ids[1] == "cal_50p_precision"
    assert [c["param"] for c in cands[2:8]] == [5, 10, 20, 30, 50, 100]
    assert [c["param"] for c in cands[8:]] == [0.0005, 0.001, 0.0025, 0.005, 0.01]
    assert all(c["kind"] == "capacity" for c in cands[2:8])
    assert all(c["kind"] == "percentile" for c in cands[8:])


def test_candidate_names_documented():
    for c in p9.build_candidates(FROZEN, 0.7):
        assert c["name"]
    assert p9.build_candidates(FROZEN) == [
        c for c in p9.build_candidates(FROZEN, None)
        if c["id"] != "cal_50p_precision"]


# ---------------------------------------------------------------------------
# 2. Policy masks (hand-computed)
# ---------------------------------------------------------------------------
def test_threshold_mask_hand_computed():
    s = np.array([0.99, 0.5, 0.5, 0.1])
    m = p9.threshold_mask(s, 0.5)
    np.testing.assert_array_equal(m, [True, True, True, False])


def test_daily_top_k_mask_hand_computed():
    days = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2)
    keys = pd.DataFrame({"user": ["u3", "u1", "u2", "u1", "u2"], "day": days})
    s = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    np.testing.assert_array_equal(
        p9.daily_top_k_mask(keys, s, 2), [False, True, True, True, True])


def test_daily_top_k_mask_tie_break_and_capacity_overflow():
    days = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2)
    keys = pd.DataFrame({"user": ["u2", "u3", "u1", "u1", "u2"], "day": days})
    s = np.array([0.7, 0.7, 0.7, 0.4, 0.9])
    # day 1: tie at 0.7 -> user asc wins (u1); day 2: top-1 = u2's 0.9
    np.testing.assert_array_equal(
        p9.daily_top_k_mask(keys, s, 1), [False, False, True, False, True])
    # capacity above rows-per-day alerts every row
    np.testing.assert_array_equal(
        p9.daily_top_k_mask(keys, s, 5), [True] * 5)


def test_percentile_mask_hand_computed():
    keys = pd.DataFrame({"user": [f"u{i}" for i in range(10)],
                         "day": pd.to_datetime(["2020-01-01"] * 10)})
    s = np.array([0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.1, 0.6, 0.4, 0.95])
    m = p9.percentile_mask(keys, s, 0.15)  # ceil(1.5) = 2 alerts
    assert int(m.sum()) == 2
    assert m[9] and m[1]
    m1 = p9.percentile_mask(keys, s, 0.05)  # ceil(0.5) = 1 alert
    assert int(m1.sum()) == 1 and m1[9]


def test_percentile_mask_tie_break():
    keys = pd.DataFrame({"user": ["u2", "u1", "u3"],
                         "day": pd.to_datetime(["2020-01-01"] * 3)})
    s = np.array([0.8, 0.8, 0.3])
    np.testing.assert_array_equal(p9.percentile_mask(keys, s, 0.25),
                                  [False, True, False])


def test_policy_mask_dispatch_unknown_kind():
    with pytest.raises(ValueError):
        p9.policy_mask(pd.DataFrame({"user": ["u1"], "day": ["2020-01-01"]}),
                       np.array([0.5]), {"kind": "bogus", "param": 1})


# ---------------------------------------------------------------------------
# 3. Mask-based metrics
# ---------------------------------------------------------------------------
def test_metrics_from_mask_hand_computed():
    y = np.array([1, 0, 1, 0, 0])
    mask = np.array([True, True, False, False, True])
    r = p9.metrics_from_mask(y, mask)
    assert r["tp"] == 1 and r["fp"] == 2 and r["fn"] == 1 and r["tn"] == 1
    assert r["n_alerts"] == 3 and r["alert_rate"] == 0.6
    assert r["precision"] == pytest.approx(1 / 3)
    assert r["recall"] == pytest.approx(1 / 2)
    assert r["f1"] == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))
    assert r["mcc"] == float(matthews_corrcoef(y, mask))
    assert r["balanced_accuracy"] == pytest.approx(0.5 * (1 / 2 + 1 / 3))
    assert r["fpr"] == pytest.approx(2 / 3)
    assert r["fnr"] == pytest.approx(1 / 2)
    assert r["coverage"] == r["recall"]


def test_metrics_from_mask_perfect():
    y = np.array([1, 0, 1])
    mask = np.array([True, False, True])
    r = p9.metrics_from_mask(y, mask)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["mcc"] == 1.0 and r["n_alerts"] == 2


def test_scenario_recall_from_mask_grouping():
    keys = pd.DataFrame({"user": ["u1", "u1", "u2", "u2", "u3"],
                         "day": pd.to_datetime(["2020-01-01"] * 5)})
    scenario = {"u1": 1, "u2": 1, "u3": 2}
    y = np.array([1, 1, 1, 0, 0])
    mask = np.array([True, False, True, False, False])
    out = p9.scenario_recall_from_mask(y, mask, scenario, keys)
    assert set(out) == {"1"}
    assert out["1"]["n_malicious_rows"] == 3
    assert out["1"]["n_rows"] == 4
    assert out["1"]["recall"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 4. Chronological windows
# ---------------------------------------------------------------------------
def test_partition_windows_chronological_disjoint_complete():
    days = pd.date_range("2020-01-01", periods=8, freq="D")
    keys = pd.DataFrame({"user": [f"u{i % 3}" for i in range(16)],
                         "day": np.repeat(days.to_numpy(), 2)})
    masks = p9.partition_windows(keys, 4)
    assert len(masks) == 4
    union = np.zeros(16, dtype=bool)
    for m in masks:
        assert m.sum() == 4
        assert not (union & m).any()
        union |= m
    assert union.all()
    bounds = p9.window_bounds(keys, masks)
    assert [b["first_day"] for b in bounds] == [
        d.strftime("%Y-%m-%d") for d in days[[0, 2, 4, 6]]]
    assert [b["last_day"] for b in bounds] == [
        d.strftime("%Y-%m-%d") for d in days[[1, 3, 5, 7]]]


def test_partition_windows_uneven_tail():
    days = pd.date_range("2020-01-01", periods=7, freq="D")
    keys = pd.DataFrame({"user": ["u1"] * 7, "day": days.to_numpy()})
    masks = p9.partition_windows(keys, 4)
    assert [int(m.sum()) for m in masks] == [2, 2, 2, 1]
    assert sum(int(m.sum()) for m in masks) == 7


# ---------------------------------------------------------------------------
# 5. Threshold bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_objective_equivalence_single_row():
    y = np.array([1])
    s = np.array([0.9])
    t1, f1 = th.best_f1_threshold(y, s)
    assert p9._bootstrap_objective(y, s, "max_f1", 1, 0)[0] == t1 == 0.9
    assert f1 == 1.0
    t2 = th.threshold_at_precision(y, s, 0.5)
    assert p9._bootstrap_objective(y, s, "min_precision", 1, 0)[0] == t2 == 0.9


def test_bootstrap_max_f1_mirrors_higher_threshold_tie_break():
    # two thresholds tie on F1 -> th picks the HIGHER; bootstrap must mirror
    y = np.array([1, 0, 0, 1])
    s = np.array([0.9, 0.7, 0.7, 0.9])
    t, f1 = th.best_f1_threshold(y, s)  # 0.9 (both give F1 1.0, tie -> higher)
    assert t == 0.9
    got = p9._bootstrap_objective(y, s, "max_f1", 1, 0)
    # any bootstrap resample of this 2-class-perfect data keeps F1 1.0 at 0.9
    assert (got == 0.9).all()


def test_threshold_bootstrap_separated_case_and_determinism():
    y = np.array([1] * 10 + [0] * 90)
    s = np.where(y == 1, 1.0, 0.0)
    r1 = p9.threshold_bootstrap(y, s, n_boot=50, seed=0)
    r2 = p9.threshold_bootstrap(y, s, n_boot=50, seed=0)
    assert r1 == r2
    for k in ("mean", "median", "std", "p10", "p90", "ci_low", "ci_high",
              "ci_width", "min", "max"):
        assert k in r1
    assert r1["mean"] == 1.0 and r1["ci_width"] == 0.0 and r1["std"] == 0.0
    assert 0.05 <= r1["alert_rate"]["mean"] <= 0.15
    assert r1["recall"]["mean"] > 0.9
    assert r1["precision"]["mean"] > 0.98
    assert 0.0 <= r1["alert_rate"]["p10"] <= r1["alert_rate"]["p90"] <= 1.0
    # min_precision objective: rank-1 precision is 1.0 whenever a positive is
    # sampled, else the max-score fallback -- every bootstrap picks 1.0.
    rp = p9.threshold_bootstrap(y, s, objective="min_precision",
                                n_boot=50, seed=0)
    assert rp["mean"] == 1.0 and rp["ci_width"] == 0.0
    assert rp["precision"]["mean"] > 0.98


def test_threshold_bootstrap_ci_width_sanity():
    # unidentifiable threshold on noisy data -> width cannot be negative
    rng = np.random.default_rng(5)
    y = (rng.uniform(0, 1, 300) > 0.95).astype(int)
    s = rng.uniform(0, 1, 300)
    r = p9.threshold_bootstrap(y, s, n_boot=30, seed=1)
    assert r["ci_width"] >= 0.0
    assert r["min"] <= r["median"] <= r["max"]


# ---------------------------------------------------------------------------
# 6. User alert concentration
# ---------------------------------------------------------------------------
def test_user_concentration_hand_computed():
    users = [f"u{i:03d}" for i in range(100)]
    days = pd.date_range("2020-01-01", periods=20, freq="D")
    keys = pd.DataFrame({"user": np.repeat(users, 20),
                         "day": np.tile(days.to_numpy(), 100)})
    mask = np.zeros(len(keys), dtype=bool)
    mask[:20] = True                          # u000: all 20 days
    for i in range(1, 20):                    # u001..u019: 2 days each
        mask[20 * i:20 * i + 2] = True
    for i in range(20, 50):                   # u020..u049: 1 day each
        mask[20 * i] = True
    r = p9.user_concentration(keys, mask)
    assert r["n_alerts"] == 88 and r["n_alerted_users"] == 50
    assert r["alerts_per_user"]["mean"] == pytest.approx(88 / 50)
    assert r["alerts_per_user"]["max"] == 20
    assert r["shares"]["top_0.01"] == pytest.approx(20 / 88)
    assert r["shares"]["top_0.05"] == pytest.approx(24 / 88)
    assert r["shares"]["top_0.10"] == pytest.approx(28 / 88)
    assert 0.0 <= r["alerted_user_gini"] <= 1.0


def test_gini_hand_computed():
    assert p9._gini(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0
    assert p9._gini(np.array([10.0, 0.0, 0.0, 0.0])) == 0.75


def test_user_concentration_empty():
    keys = pd.DataFrame({"user": ["u1", "u2"],
                         "day": pd.to_datetime(["2020-01-01"] * 2)})
    r = p9.user_concentration(keys, np.zeros(2, dtype=bool))
    assert r["n_alerts"] == 0 and r["n_alerted_users"] == 0


# ---------------------------------------------------------------------------
# 7. User-level aggregation diagnostic (diagnostic only)
# ---------------------------------------------------------------------------
def test_user_diagnostics_hand_computed():
    keys = pd.DataFrame({
        "user": ["u1", "u1", "u1", "u2", "u2", "u2"],
        "day": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03",
                               "2020-01-04", "2020-01-05", "2020-01-06"]),
    })
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.95, 0.9, 0.4, 0.2, 0.1, 0.3])
    mask = np.array([True, True, False, False, False, True])
    r = p9.user_diagnostics(keys, y, s, mask, high_risk_threshold=0.5)
    assert r["n_users"] == 2 and r["n_malicious_users"] == 1
    u1 = r["top5_users"][0]
    assert u1["user"] == "u1" and u1["malicious"] is True
    assert u1["max_risk"] == 0.95
    assert all(v is not None for v in r["user_level_auc"].values())
    assert r["user_level_auc"]["max_risk"] == 1.0
    assert r["user_level_auc"]["n_alerts"] == 1.0
    assert r["precision_at_5_users_by_max_risk"] == 0.5  # both users in top-5
    assert r["note"].startswith("diagnostic only")


# ---------------------------------------------------------------------------
# 8. Selection rule
# ---------------------------------------------------------------------------
def _win_stats(alert_rate, recall):
    return [{"alert_rate": alert_rate, "recall": recall}] * 4


def _res(cid, kind="capacity", f1=0.5, alert_rate=0.1):
    return {"id": cid, "kind": kind, "param": 1,
            "alert_rate": alert_rate, "f1": f1, "precision": 0.6,
            "recall": 0.5, "mcc": 0.3, "coverage": 0.5}


def test_select_policy_picks_best_eligible_f1_then_fewer_alerts():
    cal = {"c1": _res("c1", f1=0.40, alert_rate=0.05),
           "c2": _res("c2", f1=0.45, alert_rate=0.02),
           "c3": _res("c3", f1=0.45, alert_rate=0.01)}
    wins = {c: _win_stats(cal[c]["alert_rate"], 0.5) for c in cal}
    r = p9.select_policy(cal, {}, wins)
    assert r["selection"] == "c3"
    assert r["eligible"] == ["c1", "c2", "c3"]
    assert "S2 pass" in r["reasons"]["c1"][0]


def test_select_policy_rejects_wide_threshold_ci():
    cal = {"t1": _res("t1", kind="threshold", f1=0.9, alert_rate=0.05)}
    boots = {"t1": {"ci_width": 0.5}}
    wins = {"t1": _win_stats(0.05, 0.8)}
    r = p9.select_policy(cal, boots, wins)
    assert r["selection"] == "NO PRODUCTION POLICY SELECTED"
    assert "S1 fail" in r["reasons"]["t1"][0]
    assert r["eligible"] == []


def test_select_policy_rejects_temporally_unstable_volume():
    cal = {"c1": _res("c1", f1=0.9, alert_rate=0.1)}
    wins = {"c1": [{"alert_rate": 0.1, "recall": 0.5},
                   {"alert_rate": 0.1, "recall": 0.5},
                   {"alert_rate": 0.3, "recall": 0.5},  # 3x global -> fail
                   {"alert_rate": 0.1, "recall": 0.5}]}
    r = p9.select_policy(cal, {}, wins)
    assert r["selection"] == "NO PRODUCTION POLICY SELECTED"
    assert "S2 fail: 1/4 windows" in r["reasons"]["c1"][0]


def test_select_policy_rejects_zero_recall_window():
    cal = {"c1": _res("c1", f1=0.9, alert_rate=0.1)}
    wins = {"c1": [{"alert_rate": 0.1, "recall": 0.5},
                   {"alert_rate": 0.1, "recall": 0.0},  # no detection -> fail
                   {"alert_rate": 0.1, "recall": 0.5},
                   {"alert_rate": 0.1, "recall": 0.5}]}
    r = p9.select_policy(cal, {}, wins)
    assert r["selection"] == "NO PRODUCTION POLICY SELECTED"
    assert "S3 fail: 1/4 windows" in r["reasons"]["c1"][1]


def test_select_policy_rule_documented_in_result():
    cal = {"c1": _res("c1", f1=0.5)}
    r = p9.select_policy(cal, {}, {"c1": _win_stats(0.1, 0.5)})
    assert "screens S1" in r["rule"] and "F1" in r["rule"]
    assert r["selected_calibration_metrics"]["alert_rate"] == 0.1


# ---------------------------------------------------------------------------
# 9. Frozen policy artifacts (present after a full run)
# ---------------------------------------------------------------------------
ART = os.path.join(os.path.dirname(__file__), "..", "reports", "artifacts")
CAL_PATH = os.path.join(ART, "phase9_calibration.json")
FREEZE_PATH = os.path.join(ART, "phase9_freeze.json")
EXP_PATH = os.path.join(ART, "phase9_experiment.json")
PRED_PATH = os.path.join(ART, "phase9_predictions.parquet")


def test_freeze_policy_matches_calibration_selection():
    if not (os.path.isfile(FREEZE_PATH) and os.path.isfile(CAL_PATH)):
        pytest.skip("phase9 artifacts not present yet")
    freeze = json.load(open(FREEZE_PATH))
    cal = json.load(open(CAL_PATH))
    sel = cal["selection"]["selection"]
    assert sel != "NO PRODUCTION POLICY SELECTED"
    cand = next(c for c in cal["candidates"] if c["id"] == sel)
    assert freeze["policy"]["id"] == cand["id"]
    assert freeze["policy"]["kind"] == cand["kind"]
    assert freeze["policy"]["param"] == cand["param"]
    assert freeze["decision_basis"].startswith("CALIBRATION-only")


def test_predictions_parquet_shape():
    if not os.path.isfile(PRED_PATH):
        pytest.skip("phase9 predictions not present yet")
    pred = pd.read_parquet(PRED_PATH)
    assert pred.shape == (47000, 5)
    assert list(pred.columns) == ["user", "day", "is_malicious", "score", "alert"]
    assert int(pred["is_malicious"].sum()) == 30
    assert not pred.isna().any().any()
    assert pred["alert"].dtype == bool


def test_predictions_policy_mask_reproducible():
    if not (os.path.isfile(PRED_PATH) and os.path.isfile(FREEZE_PATH)):
        pytest.skip("phase9 artifacts not present yet")
    pred = pd.read_parquet(PRED_PATH)
    freeze = json.load(open(FREEZE_PATH))
    mask = p9.policy_mask(pred[["user", "day"]], pred["score"].to_numpy(),
                          {"kind": freeze["policy"]["kind"],
                           "param": freeze["policy"]["param"]})
    np.testing.assert_array_equal(mask, pred["alert"].to_numpy())


def test_experiment_record_consistent_with_freeze_and_pins():
    if not (os.path.isfile(EXP_PATH) and os.path.isfile(FREEZE_PATH)):
        pytest.skip("phase9 artifacts not present yet")
    exp = json.load(open(EXP_PATH))
    freeze = json.load(open(FREEZE_PATH))
    assert exp["policy"] == freeze["policy"]
    assert exp["evaluation"]["scope"].startswith("TEST only")
    assert abs(exp["evaluation"]["classification"]["auc_roc"] - 0.939157) < 1e-4
    pol = exp["evaluation"]["policy"]
    assert pol["n_alerts"] > 0
    assert 0.0 <= pol["alert_rate"] <= 1.0
    assert 0.0 <= pol["precision"] <= 1.0
    assert 0.0 <= pol["recall"] <= 1.0