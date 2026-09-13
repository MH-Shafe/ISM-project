"""Phase 12: operational-envelope regression tests.

Protects:
  1. frozen threshold exactness (0.9186015432508062; matches the Phase 9 record)
  2. raw threshold mask (hand-computed)
  3. de-duplication: B1 == B0 by construction; 3-day and 7-day windows
     hand-computed; per-user independence; row-order invariance
  4. chronological ordering of de-duplication (days ascending)
  5. rolling policy: strictly-past references only (no future), first
     window falls back to the frozen threshold, deterministic
  6. zero-alert window counting (hand-computed)
  7. policy metrics hand-computed (confusion matrix, Gini)
  8. capacity projection arithmetic + scenario labels
  9. deterministic selection rule (eligible / RETAIN P0) + determinism
 10. TEST-use guard: policy-decision functions reject non-CALIBRATION rows
 11. artifact reloadability (skip-guarded until artifacts exist):
     frozen-score equality vs Phase 7, schema, calibration-policy
     consistency with the recorded selection rule
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import threshold as th
from src.experiments import phase12 as p12

ART = config.ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _df(rows):
    """Build a user x day table from (user, day, score, is_malicious) tuples."""
    return pd.DataFrame(
        [{"user": u, "day": pd.Timestamp(d), "score": float(s), "is_malicious": int(y)}
         for u, d, s, y in rows])


CAL_DAYS = pd.date_range("2011-02-01", "2011-02-08", freq="D")


def _cal_df(rows):
    out = _df(rows)
    out["day"] = pd.to_datetime(out["day"])
    return out


# ---------------------------------------------------------------------------
# 1. Frozen threshold exactness
# ---------------------------------------------------------------------------
def test_frozen_threshold_exactness():
    assert p12.FROZEN_THRESHOLD == 0.9186015432508062
    assert 0.0 < p12.FROZEN_THRESHOLD < 1.0


def test_frozen_threshold_matches_phase9_record():
    path = os.path.join(ART, "phase9_freeze.json")
    if not os.path.isfile(path):
        pytest.skip("phase9 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    assert rec["policy"]["param"] == p12.FROZEN_THRESHOLD


# ---------------------------------------------------------------------------
# 2. Raw threshold mask
# ---------------------------------------------------------------------------
def test_raw_mask_threshold_hand_computed():
    df = _df([("A", "2011-02-01", 0.90, 0),
              ("A", "2011-02-02", 0.95, 1),
              ("B", "2011-02-01", 0.9186015432508062, 0)])
    mask = p12.raw_mask(df)
    assert mask.tolist() == [False, True, True]  # >= threshold, inclusive


# ---------------------------------------------------------------------------
# 3-4. De-duplication correctness
# ---------------------------------------------------------------------------
def test_dedup_b1_equals_b0_by_construction():
    df = _df([("A", "2011-02-01", 0.99, 1), ("A", "2011-02-02", 0.99, 1)])
    assert np.array_equal(p12.dedup_mask(df, window_days=1),
                          p12.raw_mask(df))


def test_dedup_window_3_hand_computed():
    # crossings on days 1,2,3,4 -> emitted on days 1 and 4 only (gap >= 3)
    df = _df([("A", "2011-02-01", 0.99, 1), ("A", "2011-02-02", 0.99, 0),
              ("A", "2011-02-03", 0.99, 0), ("A", "2011-02-04", 0.99, 1)])
    mask = p12.dedup_mask(df, window_days=3)
    assert mask.tolist() == [True, False, False, True]


def test_dedup_window_7_hand_computed():
    # crossings on 9 consecutive days -> emitted on days 1 and 8 only
    rows = [("A", f"2011-02-{d:02d}", 0.99, 1) for d in range(1, 10)]
    mask = p12.dedup_mask(_df(rows), window_days=7)
    assert mask.tolist() == [True] + [False] * 6 + [True] + [False]


def test_dedup_row_order_invariant():
    rows = [("A", "2011-02-01", 0.99, 1), ("A", "2011-02-02", 0.99, 0),
            ("A", "2011-02-03", 0.99, 1), ("A", "2011-02-05", 0.99, 1)]
    df = _df(rows)
    shuffled = df.sample(frac=1.0, random_state=3)
    m1 = p12.dedup_mask(df, window_days=3)
    m2 = p12.dedup_mask(shuffled, window_days=3)

    def flagged(df, mask):
        return set(zip(df["user"].to_numpy()[mask],
                       pd.to_datetime(df["day"].to_numpy())[mask]))

    assert flagged(df, m1) == flagged(shuffled, m2)
    assert m1.tolist() == [True, False, False, True]  # days 1, 3, 5 -> emit 1, 5


def test_dedup_per_user_independent():
    df = _df([("A", "2011-02-01", 0.99, 1), ("A", "2011-02-02", 0.99, 0),
              ("B", "2011-02-01", 0.99, 1), ("B", "2011-02-02", 0.99, 1),
              ("B", "2011-02-03", 0.99, 1)])
    mask = p12.dedup_mask(df, window_days=3)
    # A: emit 02-01 only. B: emit 02-01, then next emit would be 02-04 (gap 3).
    assert mask.tolist() == [True, False, True, False, False]


def test_dedup_chronological_within_user():
    # input ordered so that user A's rows are NOT sorted by day
    df = _df([("A", "2011-02-04", 0.99, 1), ("A", "2011-02-01", 0.99, 1),
              ("A", "2011-02-02", 0.99, 0), ("A", "2011-02-07", 0.99, 1)])
    mask = p12.dedup_mask(df, window_days=3)
    # chronological: emits 02-01, then next at 02-04 (gap 3), then 02-07 (gap 3)
    assert mask.tolist() == [True, True, False, True]


# ---------------------------------------------------------------------------
# 5. Rolling policy: strictly-past references, first-window fallback
# ---------------------------------------------------------------------------
def test_rolling_apply_no_future_reference():
    # window 0: d1 (y=1, s=0.90), d2 (y=0, s=0.30)
    # window 1: d3 (y=1, s=0.85), d4 (y=0, s=0.40)
    # window 2: d5 (y=1, s=0.54 - a weak positive), d6 (y=0, s=0.20)
    rows = [("A", "2011-02-01", 0.90, 1), ("A", "2011-02-02", 0.30, 0),
            ("A", "2011-02-03", 0.85, 1), ("A", "2011-02-04", 0.40, 0),
            ("A", "2011-02-05", 0.54, 1), ("A", "2011-02-06", 0.20, 0)]
    df = _cal_df(rows)
    mask = p12.rolling_apply(df, window_days=2)

    # window 0: no reference -> frozen threshold fallback (0.90 < 0.9186): no alerts
    # window 1 reference = window 0 only: t = best_f1({0.90/1, 0.30/0}) = 0.90
    #   -> 0.85 < 0.90, no alert
    t_w1 = th.best_f1_threshold(np.array([1, 0]), np.array([0.90, 0.30]))[0]
    assert t_w1 == 0.90
    # window 2 reference = windows 0+1: scores {0.90/1, 0.30/0, 0.85/1, 0.40/0};
    #   first ascending threshold with max F1 = 0.85 (0.40 >= 0.40 catches it
    #   and adds a false positive) -> 0.54 < 0.85, weak positive missed
    t_w2 = th.best_f1_threshold(np.array([1, 0, 1, 0]),
                                np.array([0.90, 0.30, 0.85, 0.40]))[0]
    assert t_w2 == 0.85
    # leak discrimination: had d5 (0.54/1) and d6 (0.20/0) leaked into their own
    #   reference, t would be 0.54 and d5 WOULD be alerted
    leaked = th.best_f1_threshold(np.array([1, 0, 1, 0, 1, 0]),
                                  np.array([0.90, 0.30, 0.85, 0.40, 0.54, 0.20]))[0]
    assert leaked == 0.54

    assert mask.tolist() == [False, False, False, False, False, False]
    assert not mask[4]  # the weak positive is missed: the policy never saw the future


def test_rolling_apply_first_window_falls_back_to_frozen():
    # 14 consecutive days -> two 7-day windows
    rows = [("A", f"2011-02-{d:02d}",
             0.92 if d in (1, 8) else (0.90 if d in (2, 9) else 0.10),
             1 if d in (1, 8) else 0) for d in range(1, 15)]
    df = _cal_df(rows)
    mask = p12.rolling_apply(df, window_days=7)
    # window 0 (days 1-7) uses the frozen threshold 0.918601... -> 0.92 alert,
    # 0.90 no
    assert mask.tolist()[0:2] == [True, False]
    # window 1 (days 8-14) reference = window 0 (1 positive):
    #   t = best_f1({0.92/1, 0.90/0}) = 0.92 -> 0.92 alert, 0.90 no
    assert mask.tolist()[7:9] == [True, False]

    analysis = p12.rolling_threshold_analysis(df)
    w0 = analysis["by_window_size"]["7"]["windows_rolling"][0]
    assert w0["rolling_defined"] is False
    assert w0["fallback_to_frozen"] is True
    assert w0["threshold"] is None
    w1 = analysis["by_window_size"]["7"]["windows_rolling"][1]
    assert w1["rolling_defined"] is True
    assert w1["ref_windows"] == [0]
    assert w1["ref_positives"] == 1


def test_rolling_threshold_analysis_deterministic():
    rows = [("A", f"2011-02-{d:02d}", 0.5 + 0.01 * d, 1 if d % 3 == 0 else 0)
            for d in range(1, 15)]
    df = _cal_df(rows)
    a1 = p12.rolling_threshold_analysis(df)
    a2 = p12.rolling_threshold_analysis(df.sample(frac=1.0, random_state=1))
    assert a1["by_window_size"]["7"]["windows_rolling"] == \
        a2["by_window_size"]["7"]["windows_rolling"]


def test_cal_windows_chunks_unique_days_not_rows():
    # Regression: windows must be computed over unique days. A per-user-day
    # table (multi-user) previously chunked the full sorted column as rows,
    # producing one window per day instead of window_days-day windows.
    days = ["2011-02-01", "2011-02-02", "2011-02-03", "2011-02-04"]
    rows = [(u, d, 0.5, 0) for d in days for u in ("A", "B", "C")]
    df = _cal_df(rows)
    windows = p12.cal_windows(df, window_days=2)
    assert len(windows) == 2            # 4 unique days / 2 = 2 windows
    assert [w["n_days"] for w in windows] == [2, 2]
    assert all(len(w["df"]) == 6 for w in windows)  # 2 days x 3 users each
    assert df["day"].isin(windows[0]["df"]["day"]).sum() == 6
    assert not windows[0]["df"]["day"].isin(windows[1]["df"]["day"]).any()


# ---------------------------------------------------------------------------
# 6. Zero-alert window counting
# ---------------------------------------------------------------------------
def test_zero_alert_chunks_hand_computed():
    # days 02-01..02-08 (two 7-day chunks: 7 days + 1 remainder day)
    rows = [("A", f"2011-02-{d:02d}", 0.99, 1) for d in range(1, 8)]
    rows += [("A", "2011-02-08", 0.99, 1)]
    df = _cal_df(rows)
    mask = p12.raw_mask(df)
    # first chunk (02-01..02-07) has alerts, second chunk (02-08) has an alert too
    assert p12.zero_alert_chunks(df, mask) == 0
    mask2 = np.array([False] * 7 + [True])
    assert p12.zero_alert_chunks(df, mask2) == 1


# ---------------------------------------------------------------------------
# 7. Policy metrics hand-computed
# ---------------------------------------------------------------------------
def test_policy_metrics_hand_computed():
    df = _cal_df([("A", "2011-02-01", 0.99, 1), ("A", "2011-02-02", 0.99, 0),
                  ("A", "2011-02-03", 0.99, 1), ("B", "2011-02-01", 0.10, 0),
                  ("B", "2011-02-02", 0.10, 1)])
    mask = np.array([True, True, False, False, True])
    pm = p12.policy_metrics(df, mask)
    assert (pm["tp"], pm["fp"], pm["fn"], pm["tn"]) == (2, 1, 1, 1)
    assert pm["precision"] == pytest.approx(2 / 3)
    assert pm["recall"] == pytest.approx(2 / 3)
    assert pm["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)))
    assert pm["balanced_accuracy"] == pytest.approx(0.5 * (2 / 3 + 1 / 2))
    assert pm["fpr"] == pytest.approx(1 / 2)
    assert pm["fnr"] == pytest.approx(1 / 3)
    assert pm["n_alerts"] == 3
    assert pm["n_alerted_users"] == 2
    assert pm["alerts_per_user_max"] == 2
    assert pm["zero_alert_windows_7d"] == 0  # one 7-day chunk; alerts exist


def test_gini_hand_computed():
    assert p12.gini_of_counts(np.array([2, 2, 2])) == pytest.approx(0.0)
    n = 4
    assert p12.gini_of_counts(np.array([0, 0, 0, 8])) == pytest.approx((n - 1) / n)
    assert p12.gini_of_counts(np.array([0, 0, 0, 0])) == 0.0  # undefined -> 0


def test_scenario_recall_in_policy_metrics():
    df = _cal_df([("A", "2011-02-01", 0.99, 1), ("B", "2011-02-01", 0.99, 1),
                  ("C", "2011-02-01", 0.10, 1)])
    mask = np.array([True, False, True])
    pm = p12.policy_metrics(df, mask, scenario_of_user={"A": 2, "B": 3})
    assert pm["scenario_recall"]["2"] == {"n_malicious_rows": 1, "recall": 1.0}
    assert pm["scenario_recall"]["3"] == {"n_malicious_rows": 1, "recall": 0.0}


# ---------------------------------------------------------------------------
# 8. Capacity projection
# ---------------------------------------------------------------------------
def test_capacity_projection_arithmetic():
    rows = [("A", f"2011-02-{d:02d}", 0.99 if d <= 3 else 0.10, 1 if d == 1 else 0,
             1.0, 2.0) for d in range(1, 8)]
    rows += [("B", f"2011-02-{d:02d}", 0.10, 0, 1.0, 2.0) for d in range(1, 8)]
    df = pd.DataFrame(rows, columns=["user", "day", "score", "is_malicious",
                                     "f1", "f2"])
    df["day"] = pd.to_datetime(df["day"])
    cap = p12.capacity_projection(df, p12.raw_mask(df))
    obs = cap["observed"]
    assert obs["alerts"] == 3
    assert obs["users_represented"] == 2
    assert obs["user_days"] == 14
    assert obs["alerts_per_day"] == pytest.approx(3 / 7)
    assert obs["top_user_share"] == pytest.approx(1.0)
    assert cap["scenarios"]["S1_uniform_scaling"]["alerts_59d"] == 30
    assert cap["scenarios"]["S2_optimistic_lower_bound"]["alerts_59d"] == 3
    assert cap["scenarios"]["S3_concentration_stress"]["alerts_59d"] == 30
    assert cap["scenarios"]["S3_concentration_stress"]["top_user_alerts_59d"] == 30
    assert "HYPOTHESIS" in cap["scenarios"]["S2_optimistic_lower_bound"]["label"]


# ---------------------------------------------------------------------------
# 9. Deterministic selection rule
# ---------------------------------------------------------------------------
def _policy_dict(n_alerts, recall, f1, gini, z7):
    return {"n_alerts": n_alerts, "recall": recall, "f1": f1, "gini": gini,
            "zero_alert_windows_7d": z7}


def test_selection_rule_eligible():
    policies = {
        "P0": _policy_dict(100, 0.50, 0.30, 0.30, 1),
        "P2": _policy_dict(80, 0.49, 0.31, 0.31, 1),
        "P3": _policy_dict(85, 0.47, 0.32, 0.32, 1),
    }
    selected, info = p12.select_operational_policy(policies)
    assert selected == "P2"  # highest recall among the eligible
    assert info["eligible"] == ["P2", "P3"]
    assert info["rationale"]["P2"] == {"R1_recall_retention": True,
                                       "R2_alert_reduction": True,
                                       "R3_concentration": True,
                                       "R4_zero_alert_windows": True}


def test_selection_rule_retains_p0_when_nothing_qualifies():
    policies = {
        "P0": _policy_dict(100, 0.50, 0.30, 0.30, 1),
        "P3": _policy_dict(60, 0.40, 0.30, 0.30, 0),  # recall retention fails
        "P2": _policy_dict(95, 0.50, 0.31, 0.40, 1),  # Gini fails
    }
    selected, info = p12.select_operational_policy(policies)
    assert selected == "P0"
    assert info["eligible"] == []
    assert "RETAIN" in info["reason"]


def test_selection_rule_deterministic():
    policies = {
        "P0": _policy_dict(100, 0.50, 0.30, 0.30, 1),
        "P2": _policy_dict(80, 0.49, 0.31, 0.31, 1),
        "P3": _policy_dict(85, 0.47, 0.32, 0.32, 1),
    }
    assert p12.select_operational_policy(policies) == \
        p12.select_operational_policy(policies)


# ---------------------------------------------------------------------------
# 10. TEST-use guard
# ---------------------------------------------------------------------------
def test_policy_decisions_reject_test_rows():
    bad = _cal_df([("A", "2011-04-01", 0.99, 1), ("A", "2011-02-02", 0.99, 0)])
    with pytest.raises(ValueError):
        p12.dedup_experiment(bad)
    with pytest.raises(ValueError):
        p12.rolling_threshold_analysis(bad)
    with pytest.raises(ValueError):
        p12.combined_policies(bad)
    with pytest.raises(ValueError):
        p12.capacity_projection(bad, np.zeros(len(bad), dtype=bool))


def test_policy_decisions_reject_train_rows():
    bad = _cal_df([("A", "2011-01-15", 0.99, 1)])
    with pytest.raises(ValueError):
        p12.dedup_experiment(bad)


def test_combined_policies_and_selection_end_to_end():
    rows = []
    d = 1
    for u in ("A", "B", "C"):
        for k in range(7):
            rows.append(("A" if u == "A" else u, f"2011-02-{d:02d}",
                         0.99 if (u == "A" and k % 2 == 0) else 0.10,
                         1 if (u == "A" and k == 0) else 0))
            d += 1
    df = _cal_df(rows)
    combined = p12.combined_policies(df)
    selected, info = p12.select_operational_policy(combined["policies"])
    assert selected in ("P0", "P1", "P2", "P3", "P4")
    assert combined["rolling_window_choice"]["chosen"] in (7, 14, 30)
    assert set(info["rationale"]) == {"P1", "P2", "P3", "P4"}


# ---------------------------------------------------------------------------
# 11. Artifact reloadability (skip-guarded until artifacts exist)
# ---------------------------------------------------------------------------
def test_artifact_frozen_score_equality():
    path = os.path.join(ART, "phase12_test_predictions.parquet")
    ref_path = os.path.join(ART, "phase7_predictions_lgbm-graph-v1.parquet")
    if not os.path.isfile(path) or not os.path.isfile(ref_path):
        pytest.skip("phase12 artifacts not present yet")
    pred = pd.read_parquet(path)
    ref = pd.read_parquet(ref_path)
    merged = pred.merge(ref, on=["user", "day"], suffixes=("", "_ref"))
    assert len(merged) == len(ref)
    assert np.array_equal(merged["score"].to_numpy(),
                          merged["score_ref"].to_numpy())
    assert np.array_equal(merged["is_malicious"].to_numpy(),
                          merged["is_malicious_ref"].to_numpy())


def test_artifact_test_predictions_schema():
    path = os.path.join(ART, "phase12_test_predictions.parquet")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    pred = pd.read_parquet(path)
    assert set(pred.columns) == {"user", "day", "is_malicious", "score",
                                 "alert_p0", "alert_selected", "selected_policy"}
    assert not pred.duplicated(subset=["user", "day"]).any()
    assert not pred.isna().any().any()
    assert pred["alert_p0"].dtype == bool
    assert pred["alert_selected"].dtype == bool
    assert pred["selected_policy"].isin(["P0", "P1", "P2", "P3", "P4"]).all()
    assert pred["score"].between(0.0, 1.0).all()


def test_artifact_calibration_policies_consistency():
    path = os.path.join(ART, "phase12_calibration_policies.json")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    sel = rec["selected_policy"]
    assert sel in rec["policies"]
    assert rec["selection"]["selected"] == sel
    assert "CALIBRATION only" in rec["scope"]
    assert set(rec["selection"]["rationale"]) == {"P1", "P2", "P3", "P4"}
    p0 = rec["policies"]["P0"]
    for pid in ("P1", "P2", "P3", "P4"):
        p = rec["policies"][pid]
        r = rec["selection"]["rationale"][pid]
        assert r["R1_recall_retention"] == (p["recall"] >= 0.90 * p0["recall"])
        assert r["R2_alert_reduction"] == (p["n_alerts"] <= 0.90 * p0["n_alerts"])
        assert r["R3_concentration"] == (p["gini"] <= p0["gini"] + 0.05)
        assert r["R4_zero_alert_windows"] == \
            (p["zero_alert_windows_7d"] <= p0["zero_alert_windows_7d"])


def test_artifact_experiment_record_complete():
    path = os.path.join(ART, "phase12_experiment.json")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    assert rec["experiment_id"] == "phase12-operational-envelope"
    assert rec["model"]["frozen"] is True
    assert rec["model"]["best_iteration"] == 186
    assert rec["model"]["threshold"] == p12.FROZEN_THRESHOLD
    assert rec["verification"]["frozen_test_score_equality"] == "PASS"
    assert rec["protocol"]["test_evaluation"] == "once, after the decision was frozen"
    expected = ["phase12_experiment.json", "phase12_calibration_policies.json",
                "phase12_deduplication.json", "phase12_rolling_threshold.json",
                "phase12_capacity_projection.json", "phase12_comparison.json",
                "phase12_test_predictions.parquet",
                "phase12_test_policy_results.json", "phase12_cost.json"]
    assert rec["artifacts"] == expected