"""Phase 11: explainability-layer regression tests.

Protects:
  1. margin-space decomposition: bias + sum(contrib) == raw margin (real model)
  2. probability-space consistency: sigmoid(margin) == predict proba
  3. determinism and row-order invariance of contributions
  4. explanation schema completeness (user/day/score/alert/conformal/base/…)
  5. hand-computed contribution detail, ranks, directions
  6. deterministic reason templates without causal verbs
  7. deterministic label-free selection policy (alerts/monitor/borderline/sample)
  8. global importance hand-computed; gain-vs-explanation ranking comparison
  9. alert contribution patterns hand-computed
  10. frozen 12-feature registry enforcement (13-feature input raises)
  11. label-free structural guarantee (no labels in explanation/selection APIs)
  12. artifact reloadability (skip-guarded until artifacts exist)
"""
import inspect
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.experiments import phase11 as p11

N_FEATURES = len(p11.FEATURES)


# ---------------------------------------------------------------------------
# Fixtures: tiny real LightGBM (deterministic, fast)
# ---------------------------------------------------------------------------
def _tiny_model_and_data(n_rows: int = 300, n_trees: int = 25):
    import lightgbm as lgb
    rng = np.random.default_rng(7)
    X = rng.integers(0, 6, size=(n_rows, N_FEATURES)).astype(np.float32)
    X[:, 8] = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)  # device_consistency_score
    X[:, 10] = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)  # file_type_consistency_score
    raw = (1.5 * (X[:, 2] >= 3) + 1.0 * (X[:, 8] < 0.3)
           + 0.8 * (X[:, 1] >= 4) - 0.5 * (X[:, 6] >= 3))
    y = (raw > 0.6).astype(int)
    d = lgb.Dataset(X, label=y, feature_name=p11.FEATURES)
    model = lgb.train({"objective": "binary", "learning_rate": 0.1,
                       "num_leaves": 15, "min_data_in_leaf": 10, "seed": 42,
                       "verbose": -1}, d, num_boost_round=n_trees)
    df = pd.DataFrame(X, columns=p11.FEATURES)
    return model, df, y


@pytest.fixture(scope="module")
def tiny():
    model, df, y = _tiny_model_and_data()
    n_iter = model.num_trees()
    X = df[p11.FEATURES].astype(np.float32)
    probs = model.predict(X, num_iteration=n_iter)
    margins = model.predict(X, num_iteration=n_iter, raw_score=True)
    contrib, bias = p11.margin_contributions(model, X, n_iter)
    return {"model": model, "df": df, "y": y, "n_iter": n_iter,
            "probs": probs, "margins": margins,
            "contrib": contrib, "bias": bias}


# ---------------------------------------------------------------------------
# 1/2. Reconstruction gates (real LightGBM)
# ---------------------------------------------------------------------------
def test_decomposition_shape_and_bias(tiny):
    c, b = tiny["contrib"], tiny["bias"]
    assert c.shape == (tiny["df"].shape[0], N_FEATURES)
    assert b.shape == (tiny["df"].shape[0],)
    assert np.allclose(b, b[0]), "LightGBM bias is a row-independent constant"


def test_margin_reconstruction(tiny):
    err = np.abs(tiny["margins"] - (tiny["bias"] + tiny["contrib"].sum(axis=1)))
    assert err.max() <= p11.RECON_TOL
    r = p11.check_reconstruction(tiny["margins"], tiny["contrib"], tiny["bias"])
    assert r["ok"] and r["max_abs_error"] <= p11.RECON_TOL


def test_probability_space_consistency(tiny):
    sigmoid = 1.0 / (1.0 + np.exp(-tiny["margins"]))
    assert np.abs(sigmoid - tiny["probs"]).max() <= 1e-6


# ---------------------------------------------------------------------------
# 3. Determinism + row-order invariance
# ---------------------------------------------------------------------------
def test_contributions_deterministic(tiny):
    c2, b2 = p11.margin_contributions(tiny["model"],
                                      tiny["df"][p11.FEATURES].astype(np.float32),
                                      tiny["n_iter"])
    assert np.array_equal(tiny["contrib"], c2)
    assert np.array_equal(tiny["bias"], b2)


def test_contributions_order_invariant(tiny):
    rev = np.arange(len(tiny["df"]))[::-1]
    c_rev, _ = p11.margin_contributions(
        tiny["model"], tiny["df"][p11.FEATURES].astype(np.float32).iloc[rev],
        tiny["n_iter"])
    assert np.array_equal(tiny["contrib"][rev], c_rev)


# ---------------------------------------------------------------------------
# 4/5. Explanation schema + hand-computed details
# ---------------------------------------------------------------------------
def test_explain_rows_schema(tiny):
    n = 20
    alert = tiny["margins"][:n] >= 0.0
    sets = np.where(np.arange(n) % 4 == 0, "01", "1")
    p1 = np.full(n, 0.9)
    p0 = np.full(n, 0.1)
    keys = pd.DataFrame({"user": [f"AAA{i:04d}" for i in range(n)],
                         "day": ["2011-04-01"] * n})
    frame = p11.explain_rows(keys, tiny["df"][p11.FEATURES].to_numpy()[:n],
                             tiny["probs"][:n], tiny["margins"][:n],
                             tiny["contrib"][:n], tiny["bias"][:n],
                             alert, sets, p1, p0)
    expected = {"user", "day", "model_score", "margin", "alert",
                "conformal_set", "conformal_p1", "conformal_p0",
                "base_value", "top_reason_1", "top_reason_2", "top_reason_3",
                "top_reason_details"} \
        | {f"value_{f}" for f in p11.FEATURES} \
        | {f"contribution_{f}" for f in p11.FEATURES}
    assert set(frame.columns) == expected
    assert frame["alert"].dtype == bool
    assert frame["model_score"].between(0.0, 1.0).all()
    assert frame["top_reason_1"].str.len().gt(0).all()
    assert frame["top_reason_details"].str.startswith("[").all()


def test_explain_rows_alert_flag_matches_threshold(tiny):
    keys = pd.DataFrame({"user": [f"AAA{i:04d}" for i in range(30)],
                         "day": ["2011-04-01"] * 30})
    alert = tiny["margins"][:30] >= 0.0
    frame = p11.explain_rows(keys, tiny["df"][p11.FEATURES].to_numpy()[:30],
                             tiny["probs"][:30], tiny["margins"][:30],
                             tiny["contrib"][:30], tiny["bias"][:30],
                             alert, np.full(30, "1"), np.full(30, 0.9),
                             np.full(30, 0.1))
    assert np.array_equal(frame["alert"].to_numpy(), alert)


def test_contribution_detail_hand_computed():
    d = p11.contribution_detail("usb_connection_count", 3.0, 0.5, 1, 12)
    assert d["feature"] == "usb_connection_count"
    assert d["value"] == 3.0
    assert d["contribution"] == 0.5
    assert d["abs_contribution"] == 0.5
    assert d["rank"] == 1
    assert d["direction"] == "increases risk"
    assert d["n_features"] == 12
    assert d["definition"].startswith(
        "Number of rows with activity='Connect' per user-day")
    assert p11.contribution_detail("x", 0.0, -0.2, 2, 12)["direction"] == \
        "decreases risk"
    assert p11.contribution_detail("x", 0.0, 0.0, 3, 12)["direction"] == \
        "neutral (zero contribution)"
    assert p11.contribution_detail("x", 0.0, 0.0, 3, 12)["definition"] == ""


def test_contribution_ranks_hand_computed():
    c = np.array([[0.1, -0.5, 0.2], [0.0, 0.3, -0.1]])
    ranks = p11.contribution_ranks(c)
    np.testing.assert_array_equal(ranks, [[3, 1, 2], [3, 1, 2]])


def test_contribution_ranks_ties_by_feature_index():
    c = np.array([[0.2, -0.1, 0.2]])
    ranks = p11.contribution_ranks(c)
    np.testing.assert_array_equal(ranks, [[1, 3, 2]])


# ---------------------------------------------------------------------------
# 6. Reason templates
# ---------------------------------------------------------------------------
def _hand_explanation(alert=False):
    det = [p11.contribution_detail("login_count", 5.0, 0.1, 3, 12),
           p11.contribution_detail("usb_connection_count", 3.0, 0.5, 1, 12),
           p11.contribution_detail("http_activity_count", 2.0, -0.3, 2, 12),
           p11.contribution_detail("login_count_dup", 0.0, 0.0, 4, 12)]
    return {"model_score": 0.95 if alert else 0.4,
            "base_value": -2.0, "alert": alert,
            "top_reason_details": det}


def test_top_reasons_deterministic_and_ordered():
    e = _hand_explanation()
    r1 = p11.top_reasons(e)
    r2 = p11.top_reasons(e)
    assert r1 == r2
    assert r1[0] == ("usb_connection_count (value 3) increased the model's "
                     "risk score (contribution +0.5000, rank 1 of 12)")
    assert r1[1] == ("http_activity_count (value 2) decreased the model's "
                     "risk score (contribution -0.3000, rank 2 of 12)")
    assert len(r1) == 3


def test_top_reasons_alert_variant():
    e = _hand_explanation(alert=True)
    r = p11.top_reasons(e)
    assert r[0].startswith("Alert: model risk score 0.9500 (base -2.0000); "
                           "strongest positive contribution usb_connection_count "
                           "(+0.5000); strongest negative http_activity_count "
                           "(-0.3000)")
    assert len(r) == 3


def test_reasons_contain_no_causal_verbs():
    e = _hand_explanation(alert=True)
    for r in p11.top_reasons(e):
        low = r.lower()
        for v in p11.CAUSAL_VERBS:
            assert v not in low, f"causal verb '{v}' in reason: {r}"


# ---------------------------------------------------------------------------
# 7. Selection policy (deterministic, label-free)
# ---------------------------------------------------------------------------
def _selection_fixture(n=2000):
    rng = np.random.default_rng(3)
    scores = np.sort(rng.uniform(0.0, 0.95, size=n))
    keys = pd.DataFrame({"user": [f"U{i:04d}" for i in range(n)],
                         "day": pd.to_datetime("2011-04-01") + pd.to_timedelta(
                             np.arange(n) % 30, unit="D")})
    alert = scores >= p11.FROZEN_THRESHOLD
    return scores, alert, keys


def test_selection_policy_masks():
    scores, alert, keys = _selection_fixture()
    t0 = 0.4
    sel = p11.select_explanation_rows(scores, alert, t0, keys)
    assert np.array_equal(sel["alerts"], scores >= p11.FROZEN_THRESHOLD)
    assert np.array_equal(sel["monitor"],
                          (~alert) & (scores >= t0))
    assert np.array_equal(sel["borderline"],
                          sel["monitor"]
                          & (scores >= p11.FROZEN_THRESHOLD - p11.BORDERLINE_WIDTH))
    assert sel["non_alert_sample"].sum() == 10 * p11.SAMPLE_PER_DECILE


def test_selection_policy_deterministic():
    scores, alert, keys = _selection_fixture()
    s1 = p11.select_explanation_rows(scores, alert, 0.4, keys)
    s2 = p11.select_explanation_rows(scores, alert, 0.4, keys)
    for k in ("alerts", "monitor", "borderline", "non_alert_sample"):
        assert np.array_equal(s1[k], s2[k])


def test_selection_policy_is_label_free():
    sig = inspect.signature(p11.select_explanation_rows)
    assert "y" not in sig.parameters and "label" not in sig.parameters
    sig2 = inspect.signature(p11.explain_rows)
    assert "y" not in sig2.parameters and "label" not in sig2.parameters


def test_decile_stratified_empty_pool():
    scores = np.zeros(10)
    eligible = np.zeros(10, dtype=bool)
    mask = p11._decile_stratified(scores, eligible,
                                  pd.DataFrame({"user": list("abcdefghij"),
                                                "day": ["d"] * 10}), 50)
    assert mask.sum() == 0


def test_decile_stratified_small_pool():
    scores = np.linspace(0.0, 1.0, 40)
    eligible = np.ones(40, dtype=bool)
    keys = pd.DataFrame({"user": [f"U{i}" for i in range(40)],
                         "day": ["d"] * 40})
    mask = p11._decile_stratified(scores, eligible, keys, 50)
    assert mask.sum() == 40
    assert np.array_equal(
        mask, p11._decile_stratified(scores, eligible, keys, 50))


# ---------------------------------------------------------------------------
# 8. Global importance + gain comparison (hand-computed)
# ---------------------------------------------------------------------------
def test_global_importance_hand_computed():
    n = N_FEATURES
    contrib = np.zeros((3, n))
    contrib[:, 0] = [1.0, 2.0, 3.0]   # mean abs 2.0
    contrib[:, 1] = [-1.0, 1.0, -1.0]  # mean abs 1.0, mean signed -1/3
    contrib[:, 2] = [0.0, 0.0, 0.0]    # mean abs 0.0
    imp = p11.global_importance(contrib)
    assert imp["n"] == 3
    assert imp["per_feature"][p11.FEATURES[0]]["mean_abs"] == pytest.approx(2.0)
    assert imp["per_feature"][p11.FEATURES[1]]["mean_signed"] == pytest.approx(-1 / 3)
    assert imp["per_feature"][p11.FEATURES[1]]["frac_nonzero"] == 1.0
    assert imp["per_feature"][p11.FEATURES[2]]["frac_nonzero"] == 0.0
    assert imp["ranking_mean_abs"][:3] == [p11.FEATURES[0], p11.FEATURES[1],
                                           p11.FEATURES[2]]
    assert imp["per_feature"][p11.FEATURES[0]]["rank"] == 0


def test_compare_gain_ranking_hand_computed():
    gain = {f: 1.0 / (i + 1) for i, f in enumerate(p11.FEATURES)}
    contrib = np.zeros((1, N_FEATURES))
    for i, f in enumerate(p11.FEATURES):
        contrib[0, i] = 1.0 / (i + 1)
    imp = p11.global_importance(contrib)
    same = p11.compare_gain_ranking(gain, imp)
    assert same["kendall_tau"] == pytest.approx(1.0)
    assert same["delta_rank_expl_vs_gain"] == {f: 0 for f in p11.FEATURES}
    rev = {f: 1.0 / (N_FEATURES - i) for i, f in enumerate(p11.FEATURES)}
    opp = p11.compare_gain_ranking(rev, imp)
    assert opp["kendall_tau"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 9. Alert patterns (hand-computed)
# ---------------------------------------------------------------------------
def test_alert_patterns_hand_computed():
    n = N_FEATURES
    a = np.zeros((3, n))
    a[0, 0] = 2.0
    a[0, 1] = -1.0
    a[1, 0] = 1.0
    a[1, 2] = -0.5
    a[2, 1] = 3.0
    a[2, 2] = 0.1
    pat = p11.alert_patterns(a)
    assert pat["n_alerts"] == 3
    assert pat["per_feature"][p11.FEATURES[0]]["frac_top_positive"] == pytest.approx(2 / 3)
    assert pat["per_feature"][p11.FEATURES[1]]["frac_top_positive"] == pytest.approx(1 / 3)
    assert pat["per_feature"][p11.FEATURES[1]]["frac_top_negative"] == pytest.approx(1 / 3)
    assert pat["per_feature"][p11.FEATURES[2]]["frac_top_negative"] == pytest.approx(1 / 3)
    assert pat["per_feature"][p11.FEATURES[0]]["frac_top3_abs"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 9b. Alert contribution statistics (hand-computed)
# ---------------------------------------------------------------------------
def test_alert_contribution_stats_hand_computed():
    n = N_FEATURES
    a = np.zeros((3, n))
    a[0, 0] = 2.0
    a[0, 1] = -1.0
    a[1, 0] = 1.0
    a[1, 1] = -3.0
    a[2, 0] = 0.5
    a[2, 1] = 0.2
    stats = p11.alert_contribution_stats(a)
    assert stats["n_alerts"] == 3
    f0 = stats["per_feature"][p11.FEATURES[0]]
    assert f0["mean"] == pytest.approx((2.0 + 1.0 + 0.5) / 3)
    assert f0["mean_abs"] == pytest.approx((2.0 + 1.0 + 0.5) / 3)
    assert f0["frac_positive"] == pytest.approx(1.0)
    assert f0["frac_negative"] == pytest.approx(0.0)
    assert f0["frac_zero"] == pytest.approx(0.0)
    assert f0["mean_rank"] == pytest.approx(4 / 3)
    assert f0["frac_top1"] == pytest.approx(2 / 3)
    assert f0["frac_top3"] == pytest.approx(1.0)
    f1 = stats["per_feature"][p11.FEATURES[1]]
    assert f1["mean"] == pytest.approx((-1.0 - 3.0 + 0.2) / 3)
    assert f1["frac_positive"] == pytest.approx(1 / 3)
    assert f1["frac_negative"] == pytest.approx(2 / 3)
    assert f1["frac_top1"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 9c. Perturbation stability (deterministic, inference-only)
# ---------------------------------------------------------------------------
def test_perturbation_stability_deterministic_and_bounded(tiny):
    X = tiny["df"][p11.FEATURES].to_numpy()
    m, c = tiny["margins"], tiny["contrib"]
    p1 = p11.perturbation_stability(tiny["model"], X, m, c, tiny["n_iter"])
    p2 = p11.perturbation_stability(tiny["model"], X, m, c, tiny["n_iter"])
    assert p1 == p2
    assert p1["n_rows"] == len(X)
    for f in p11.FEATURES:
        per = p1["per_feature"][f]
        assert per["delta"] == (1.0 if f in p11.COUNT_FEATURES else 0.1)
        assert per["mean_abs_margin_delta"] >= 0.0
        assert 0.0 <= per["frac_top1_flip"] <= 1.0
        assert 0.0 <= per["frac_top3_flip"] <= 1.0
        assert 0.0 <= per["frac_decision_flip"] <= 1.0
    total = sum(p1["per_feature"][f]["mean_abs_margin_delta"]
                for f in p11.FEATURES)
    assert total > 0.0


def test_perturbation_stability_sample_mask(tiny):
    mask = np.zeros(len(tiny["df"]), dtype=bool)
    mask[:10] = True
    p = p11.perturbation_stability(tiny["model"],
                                   tiny["df"][p11.FEATURES].to_numpy(),
                                   tiny["margins"], tiny["contrib"],
                                   tiny["n_iter"], sample_mask=mask)
    assert p["n_rows"] == 10


# ---------------------------------------------------------------------------
# 9d. Counterfactual feasibility + risk-level mapping
# ---------------------------------------------------------------------------
def test_counterfactual_feasibility_not_supported():
    a = p11.counterfactual_feasibility()
    assert a["decision"] == "NOT_SUPPORTED"
    assert len(a["reasons"]) >= 3
    assert "future_criteria" in a


def test_risk_level_mapping():
    assert p11.risk_level(True, "non_alert_sample") == "ALERT"
    assert p11.risk_level(False, "borderline") == "BORDERLINE"
    assert p11.risk_level(False, "monitor") == "MONITOR"
    assert p11.risk_level(False, "non_alert_sample") == "NON-ALERT"
    assert p11.risk_level(False, "alerts") == "NON-ALERT"
    assert set(p11.RISK_LEVELS) == {"ALERT", "BORDERLINE", "MONITOR",
                                    "NON-ALERT"}


# ---------------------------------------------------------------------------
# 10. Frozen feature registry enforcement
# ---------------------------------------------------------------------------
def test_explanation_rejects_department_feature():
    bad = list(p11.FEATURES) + ["department_file_type_mismatch_count"]
    with pytest.raises(ValueError):
        p11._check_features(bad)
    with pytest.raises(ValueError):
        p11.explain_rows(pd.DataFrame({"user": ["u"], "day": ["d"]}),
                         np.zeros((1, len(bad))), np.zeros(1), np.zeros(1),
                         np.zeros((1, len(bad))), np.zeros(1), np.zeros(1),
                         np.array(["1"]), np.zeros(1), np.zeros(1),
                         features=bad)
    with pytest.raises(ValueError):
        p11.global_importance(np.zeros((1, len(bad))), features=bad)


def test_frozen_features_are_the_12_frozen_ones():
    assert p11.FEATURES == [
        "login_count", "after_hours_login_count", "usb_connection_count",
        "file_access_count", "sensitive_file_access_count",
        "http_activity_count", "unique_device_count", "unusual_access_count",
        "device_consistency_score", "rare_device_usage_count",
        "file_type_consistency_score", "rare_file_type_access_count"]
    assert "department_file_type_mismatch_count" not in p11.FEATURES


# ---------------------------------------------------------------------------
# 11. Artifact reloadability (present after a full run)
# ---------------------------------------------------------------------------
ART = os.path.join(os.path.dirname(__file__), "..", "reports", "artifacts")
EXP_PATH = os.path.join(ART, "phase11_experiment.json")
GLOB_PATH = os.path.join(ART, "phase11_global_importance.json")
EXPL_PATH = os.path.join(ART, "phase11_explanations.parquet")
CF_PATH = os.path.join(ART, "phase11_counterfactual.json")
INTEG_PATH = os.path.join(ART, "phase11_model_integrity.json")
REPRO_PATH = os.path.join(ART, "phase11_reproduction.json")
ALERT_EXPL_PATH = os.path.join(ART, "phase11_alert_explanations.parquet")


def test_global_importance_artifact_reloads():
    if not os.path.isfile(GLOB_PATH):
        pytest.skip("phase11 artifacts not present yet")
    art = json.load(open(GLOB_PATH))
    assert art["scope"].startswith("TRAIN + CALIBRATION")
    for block in ("calibration", "train"):
        imp = art[block]["importance"]
        assert set(imp["per_feature"]) == set(p11.FEATURES)
        assert len(imp["ranking_mean_abs"]) == N_FEATURES
        assert "kendall_tau" in art[block]["gain_vs_explanation"]


def test_explanations_artifact_reconstructs_and_is_clean():
    if not os.path.isfile(EXPL_PATH):
        pytest.skip("phase11 explanations not present yet")
    frame = pd.read_parquet(EXPL_PATH)
    assert {f"value_{f}" for f in p11.FEATURES}.issubset(set(frame.columns))
    assert {f"contribution_{f}" for f in p11.FEATURES} \
        .issubset(set(frame.columns))
    contrib_sum = sum(frame[f"contribution_{f}"].to_numpy() for f in p11.FEATURES)
    err = np.abs(frame["margin"].to_numpy() - (frame["base_value"].to_numpy() + contrib_sum))
    assert err.max() <= p11.RECON_TOL
    assert frame["alert"].dtype == bool
    assert int(frame["alert"].sum()) == 49
    assert set(frame["conformal_set"].unique()) <= {"0", "1", "01", "empty"}
    assert set(frame["risk_level"].unique()) <= set(p11.RISK_LEVELS)
    assert (frame.loc[frame["alert"], "risk_level"] == "ALERT").all()
    assert not frame.isna().any().any()
    assert not frame.duplicated(subset=["user", "day"]).any()
    for col in ("top_reason_1", "top_reason_2", "top_reason_3"):
        text = " ".join(frame[col].dropna().tolist()).lower()
        for v in p11.CAUSAL_VERBS:
            assert v not in text, f"causal verb '{v}' found in reasons"


def test_counterfactual_artifact_reloads():
    if not os.path.isfile(CF_PATH):
        pytest.skip("phase11 artifacts not present yet")
    art = json.load(open(CF_PATH))
    assert art["analysis"]["decision"] == "NOT_SUPPORTED"


def test_model_integrity_artifact_reloads():
    if not os.path.isfile(INTEG_PATH):
        pytest.skip("phase11 artifacts not present yet")
    art = json.load(open(INTEG_PATH))
    assert art["model"]["num_trees"] == 186
    assert art["feature_names_match_frozen_12"]
    assert art["alert_policy"]["threshold"] == p11.FROZEN_THRESHOLD


def test_reproduction_artifact_reloads():
    if not os.path.isfile(REPRO_PATH):
        pytest.skip("phase11 artifacts not present yet")
    art = json.load(open(REPRO_PATH))
    assert art["calibration_stage"]["determinism_recomputed_identical"]
    assert art["test_stage"]["had_previous_record"]
    assert art["test_stage"]["reproduced_bit_identical"]
    assert art["frozen_spec"]["model_trees"] == 186


def test_alert_explanations_artifact_reloads():
    if not os.path.isfile(ALERT_EXPL_PATH):
        pytest.skip("phase11 artifacts not present yet")
    frame = pd.read_parquet(ALERT_EXPL_PATH)
    assert set(frame.columns) == set(pd.read_parquet(EXPL_PATH).columns)
    assert frame["alert"].all()
    assert (frame["risk_level"] == "ALERT").all()
    assert (frame["conformal_set"] == "1").all()


def test_experiment_artifact_matches_frozen_records():
    if not (os.path.isfile(EXP_PATH) and os.path.isfile(EXPL_PATH)):
        pytest.skip("phase11 artifacts not present yet")
    exp = json.load(open(EXP_PATH))
    assert exp["explained"]["alerts"] == 49
    assert exp["reconstruction_test"]["ok"]
    assert exp["conformal_overlay"]["alpha"] == 0.05
    with open(os.path.join(ART, "phase7_freeze_lgbm-graph-v1.json")) as fh:
        p7rec = json.load(fh)
    ev = p7rec["evaluation"]["classification"]
    assert abs(ev["auc_roc"] - 0.9391565538286849) < 1e-6
    assert abs(ev["auc_pr"] - 0.2677761042396373) < 1e-6
    assert p7rec["model"]["best_iteration"] == 186