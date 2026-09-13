"""Phase 10: conformal confidence-layer protocol regression tests (synthetic).

Protects:
  1. conformal p-value formulas (class 1 and class 0) are hand-computed correct
  2. inclusion thresholds exactly mirror the p-value rule (incl. ties)
  3. prediction-set construction: {0}, {1}, {0,1}, empty
  4. alpha handling: sets shrink monotonically with alpha; invalid alpha raises
  5. class handling: fit requires both classes; labels in {0, 1}
  6. deterministic output (fit + sets)
  7. chronological calibration isolation / TEST exclusion: the layer never
     consumes inference-block labels; sets depend only on the fit + scores
  8. frozen-model score preservation (artifact gate)
  9. edge cases (single-point classes, ties, alpha bounds)
  10. empty/ambiguous set handling (middle band, overlap band)
  11. artifact reloadability (fit JSON -> sets reproduce saved predictions)
  12. reproducibility (fit twice -> identical)
  13. decision rule unit cases (ACCEPT / CONDITIONAL / REJECT)
"""
import json
import os

import numpy as np
import pytest

from src.experiments import phase10 as p10

FROZEN = 0.9186015432508062


def _fit():
    """6-row calibration: 4 negatives [0.1..0.4], 2 positives [0.9, 0.95]."""
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    y = np.array([0, 0, 0, 0, 1, 1])
    return p10.fit_conformal(s, y)


# ---------------------------------------------------------------------------
# 1. Conformal p-values (hand-computed)
# ---------------------------------------------------------------------------
def test_p1_formula_hand_computed():
    fit = _fit()  # n1 = 2, pos_sorted = [0.9, 0.95]
    s = np.array([0.0, 0.89, 0.9, 0.95, 0.99])
    np.testing.assert_allclose(p10.p1_values(fit, s),
                               [(0 + 1) / 3, (0 + 1) / 3, (1 + 1) / 3,
                                (2 + 1) / 3, (2 + 1) / 3])


def test_p0_formula_hand_computed():
    fit = _fit()  # n0 = 4, neg_sorted = [0.1, 0.2, 0.3, 0.4]
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(p10.p0_values(fit, s),
                               [(4 + 1) / 5, (4 + 1) / 5, (3 + 1) / 5,
                                (2 + 1) / 5, (1 + 1) / 5, (0 + 1) / 5])


def test_p_values_ties_conservative():
    # ties count toward the p-value (>= counts in the nonconforming direction)
    fit = p10.fit_conformal(np.array([0.2, 0.5, 0.5, 0.9]), np.array([0, 1, 1, 1]))
    assert p10.p1_values(fit, np.array([0.5]))[0] == (2 + 1) / 4
    assert p10.p1_values(fit, np.array([0.49]))[0] == 1 / 4


# ---------------------------------------------------------------------------
# 2. Inclusion thresholds mirror the p-value rule exactly (incl. ties)
# ---------------------------------------------------------------------------
def test_inclusion_thresholds_mirror_pvalues():
    fit = _fit()
    thr = p10.inclusion_thresholds(fit, 0.5)
    assert thr == {"alpha": 0.5, "k1": 1, "k0": 2, "t1": 0.9, "t0": 0.3}
    grid = np.linspace(0.0, 1.0, 501)
    np.testing.assert_array_equal(p10.p1_values(fit, grid) > 0.5,
                                  grid >= thr["t1"])
    np.testing.assert_array_equal(p10.p0_values(fit, grid) > 0.5,
                                  grid <= thr["t0"])


def test_inclusion_thresholds_ties():
    fit = p10.fit_conformal(np.array([0.2, 0.5, 0.5, 0.9]), np.array([0, 1, 1, 1]))
    thr = p10.inclusion_thresholds(fit, 0.5)  # k1 = floor(0.5 * 4) = 2
    assert thr["t1"] == 0.5
    assert (p10.p1_values(fit, np.array([0.5])) > 0.5)[0]
    assert not (p10.p1_values(fit, np.array([0.49])) > 0.5)[0]


def test_inclusion_thresholds_extreme_alphas():
    fit = _fit()
    assert p10.inclusion_thresholds(fit, 0.1)["t1"] == -np.inf  # k1 = 0
    assert p10.inclusion_thresholds(fit, 0.1)["t0"] == np.inf   # k0 = 0
    thr = p10.inclusion_thresholds(fit, 0.99)
    assert thr["k1"] == 2 and thr["k0"] == 4
    assert thr["t1"] == 0.95 and thr["t0"] == 0.1


# ---------------------------------------------------------------------------
# 3. Prediction-set construction (hand-computed)
# ---------------------------------------------------------------------------
def test_set_category_hand_computed():
    fit = _fit()
    s = np.array([0.2, 0.25, 0.3, 0.5, 0.9, 0.95, 0.99])
    np.testing.assert_array_equal(
        p10.set_category(fit, s, 0.5),
        ["0", "0", "0", "empty", "1", "1", "1"])


def test_set_category_ambiguous_band():
    # pos_sorted = [0.1, 0.5], neg_sorted = [0.2, 0.3, 0.4, 0.9]
    fit = p10.fit_conformal(np.array([0.2, 0.3, 0.4, 0.9, 0.1, 0.5]),
                            np.array([0, 0, 0, 0, 1, 1]))
    s = np.array([0.05, 0.35, 0.45])
    np.testing.assert_array_equal(p10.set_category(fit, s, 0.5),
                                  ["0", "01", "1"])


# ---------------------------------------------------------------------------
# 4. Alpha handling
# ---------------------------------------------------------------------------
def test_sets_shrink_monotonically_with_alpha():
    fit = _fit()
    grid = np.linspace(0.0, 1.0, 101)
    prev1 = np.ones(len(grid), dtype=bool)
    prev0 = np.ones(len(grid), dtype=bool)
    for alpha in sorted(p10.ALPHAS):
        cats = p10.set_category(fit, grid, alpha)
        c1 = (cats == "1") | (cats == "01")
        c0 = (cats == "0") | (cats == "01")
        assert not (c1 & ~prev1).any(), "class-1 inclusion must not grow with alpha"
        assert not (c0 & ~prev0).any(), "class-0 inclusion must not grow with alpha"
        prev1, prev0 = c1, c0


def test_alpha_bounds_raise():
    fit = _fit()
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            p10.set_category(fit, np.array([0.5]), bad)
        with pytest.raises(ValueError):
            p10.inclusion_thresholds(fit, bad)
        with pytest.raises(ValueError):
            p10.coverage_stats(fit, np.array([0.5]), np.array([0]), bad)


# ---------------------------------------------------------------------------
# 5. Class handling
# ---------------------------------------------------------------------------
def test_fit_requires_both_classes():
    with pytest.raises(ValueError):
        p10.fit_conformal(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError):
        p10.fit_conformal(np.array([0.1, 0.2]), np.array([1, 1]))


def test_fit_rejects_invalid_labels_and_shapes():
    with pytest.raises(ValueError):
        p10.fit_conformal(np.array([0.1, 0.2]), np.array([0, 2]))
    with pytest.raises(ValueError):
        p10.fit_conformal(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# 6. Determinism + reproducibility
# ---------------------------------------------------------------------------
def test_fit_deterministic():
    f1, f2 = _fit(), _fit()
    for key in ("method", "n1", "n0", "alphas"):
        assert f1[key] == f2[key]
    np.testing.assert_array_equal(f1["pos_sorted"], f2["pos_sorted"])
    np.testing.assert_array_equal(f1["neg_sorted"], f2["neg_sorted"])


def test_sets_reproducible_across_fits():
    f1, f2 = _fit(), _fit()
    s = np.linspace(0.0, 1.0, 50)
    for alpha in p10.ALPHAS:
        np.testing.assert_array_equal(p10.set_category(f1, s, alpha),
                                      p10.set_category(f2, s, alpha))


# ---------------------------------------------------------------------------
# 7. Calibration isolation / TEST exclusion
# ---------------------------------------------------------------------------
def test_inference_block_labels_never_consumed():
    # The layer's prediction path takes only (fit, scores, alpha); the fit
    # object contains only CALIBRATION arrays, so inference-block labels can
    # never influence any prediction set.
    fit = _fit()
    s = np.array([0.2, 0.5, 0.9])
    cats = p10.set_category(fit, s, 0.5)
    for alpha in p10.ALPHAS:
        np.testing.assert_array_equal(cats, p10.set_category(fit, s, alpha)
                                      if alpha == 0.5 else cats)
    assert "y" not in fit and "scores" not in fit
    assert set(fit) == {"method", "n1", "n0", "pos_sorted", "neg_sorted", "alphas"}


def test_no_self_calibration_within_inference_block():
    # p-values are computed per-row against the CALIBRATION sample only;
    # growing the inference block (duplicating rows) cannot change them.
    fit = _fit()
    s = np.array([0.2, 0.5, 0.9])
    s2 = np.concatenate([s, s])
    for alpha in p10.ALPHAS:
        np.testing.assert_array_equal(p10.set_category(fit, s2, alpha)[:3],
                                      p10.set_category(fit, s, alpha))


def test_p_values_bounded_and_monotone():
    fit = _fit()
    grid = np.linspace(0.0, 1.0, 101)
    p1 = p10.p1_values(fit, grid)
    p0 = p10.p0_values(fit, grid)
    assert p1.min() == pytest.approx(1 / (fit["n1"] + 1))
    assert p1.max() == 1.0
    assert p0.min() == pytest.approx(1 / (fit["n0"] + 1))
    assert p0.max() == 1.0
    assert (np.diff(p1) >= 0).all(), "p1 non-decreasing in score"
    assert (np.diff(p0) <= 0).all(), "p0 non-increasing in score"


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------
def test_single_point_classes():
    fit = p10.fit_conformal(np.array([0.5, 0.7]), np.array([0, 1]))
    thr = p10.inclusion_thresholds(fit, 0.5)
    assert thr["t1"] == 0.7 and thr["t0"] == 0.5
    assert p10.set_category(fit, np.array([0.7]), 0.5)[0] == "1"
    assert p10.set_category(fit, np.array([0.5]), 0.5)[0] == "0"
    assert p10.set_category(fit, np.array([0.6]), 0.5)[0] == "empty"


def test_all_identical_scores():
    fit = p10.fit_conformal(np.array([0.5, 0.5, 0.5]), np.array([0, 0, 1]))
    # every calibration score equals the boundary; ties resolved consistently:
    # p1 = (1+1)/2 = 1.0 and p0 = (2+1)/3 = 1.0 at s = 0.5 -> "01"
    cats = p10.set_category(fit, np.array([0.5, 0.49, 0.51]), 0.5)
    np.testing.assert_array_equal(cats, ["01", "0", "1"])


# ---------------------------------------------------------------------------
# 9. Coverage diagnostics (hand-computed) + Wilson interval
# ---------------------------------------------------------------------------
def test_coverage_stats_hand_computed():
    fit = _fit()
    s = np.array([0.2, 0.5, 0.9, 0.95])
    y = np.array([0, 0, 1, 1])
    r = p10.coverage_stats(fit, s, y, 0.5)
    assert r["marginal_coverage"] == pytest.approx(3 / 4)
    assert r["pos_coverage"] == 1.0
    assert r["neg_coverage"] == 0.5
    d = r["set_size_dist"]
    assert d["singleton1"] == 0.5 and d["singleton0"] == 0.25
    assert d["ambiguous"] == 0.0 and d["empty"] == 0.25
    assert d["mean_size"] == 0.75
    assert r["positive_prediction_rate"] == 0.5


def test_wilson_ci_sanity():
    lo, hi = p10.wilson_ci(30, 30, confidence=0.90)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.9173, abs=1e-3)
    lo, hi = p10.wilson_ci(0, 30, confidence=0.90)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0827, abs=1e-3)
    for k, n in ((28, 30), (24, 30), (15, 30)):
        lo, hi = p10.wilson_ci(k, n, confidence=0.90)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
    assert p10.wilson_ci(5, 0, confidence=0.90) is None


def test_policy_crosstab_hand_computed():
    fit = _fit()
    s = np.array([0.2, 0.9, 0.95, 0.5])
    y = np.array([0, 1, 1, 0])
    alert = s >= 0.85
    r = p10.policy_crosstab(fit, s, alert, y, 0.5)
    assert r["n_alerts"] == 2 and r["n_uncertain_alerts"] == 0
    assert r["cells"]["alert"]["1"] == {"n": 2, "precision": 1.0}
    assert r["cells"]["no_alert"]["0"] == {"n": 1, "precision": 0.0}
    assert r["cells"]["no_alert"]["empty"] == {"n": 1, "precision": 0.0}
    assert r["n_monitor"] == 0


# ---------------------------------------------------------------------------
# 10. Decision rule unit cases
# ---------------------------------------------------------------------------
def _evidence(**over):
    base = {
        "test_005": {"pos_coverage": 0.90, "pos_wilson_lb": 0.80,
                     "neg_coverage": 0.95},
        "cal_005": {"n_uncertain_alerts": 8, "precision_gap": 0.10,
                    "n_monitor": 3, "window_pos_coverage": [0.90] * 4},
        "inference_s": 0.01,
    }
    for k, v in over.items():
        if k in base:
            base[k].update(v) if isinstance(base[k], dict) else base.update({k: v})
        else:
            base[k] = v
    return base


def test_decide_accept_all_pass():
    r = p10.decide_conformal(_evidence())
    assert r["decision"] == "ACCEPT"
    assert all(k in r["reasons"] for k in ("D1", "D2", "D3", "D4", "D5"))


def test_decide_reject_when_positive_coverage_fails():
    r = p10.decide_conformal(_evidence(test_005={"pos_coverage": 0.80,
                                                 "pos_wilson_lb": 0.65,
                                                 "neg_coverage": 0.95}))
    assert r["decision"] == "REJECT"
    assert r["reasons"]["D1"].startswith("FAIL")


def test_decide_reject_when_negative_coverage_fails():
    r = p10.decide_conformal(_evidence(test_005={"pos_coverage": 0.90,
                                                 "pos_wilson_lb": 0.80,
                                                 "neg_coverage": 0.85}))
    assert r["decision"] == "REJECT"
    assert r["reasons"]["D2"].startswith("FAIL")


def test_decide_conditional_when_usefulness_only_fails():
    r = p10.decide_conformal(_evidence(cal_005={"n_uncertain_alerts": 1,
                                                "precision_gap": 0.01,
                                                "n_monitor": 2,
                                                "window_pos_coverage": [0.9] * 4}))
    assert r["decision"] == "CONDITIONAL / DIAGNOSTIC ONLY"
    assert r["reasons"]["D3"].startswith("FAIL")


def test_decide_reject_when_window_coverage_collapses():
    r = p10.decide_conformal(_evidence(cal_005={"n_uncertain_alerts": 8,
                                                "precision_gap": 0.10,
                                                "n_monitor": 3,
                                                "window_pos_coverage": [0.9, 0.5, 0.9, 0.9]}))
    assert r["decision"] == "REJECT"
    assert r["reasons"]["D4"].startswith("FAIL")


def test_decide_reject_when_inference_cost_unjustified():
    r = p10.decide_conformal(_evidence(inference_s=5.0))
    assert r["decision"] == "REJECT"
    assert r["reasons"]["D5"].startswith("FAIL")


def test_decide_deterministic():
    e = _evidence()
    assert p10.decide_conformal(e) == p10.decide_conformal(e)


# ---------------------------------------------------------------------------
# 11. Artifact consistency (present after a full run)
# ---------------------------------------------------------------------------
ART = os.path.join(os.path.dirname(__file__), "..", "reports", "artifacts")
CFG_PATH = os.path.join(ART, "phase10_config.json")
FIT_PATH = os.path.join(ART, "phase10_fit.json")
CAL_PATH = os.path.join(ART, "phase10_calibration.json")
EXP_PATH = os.path.join(ART, "phase10_experiment.json")
DEC_PATH = os.path.join(ART, "phase10_decision.json")
PRED_PATH = os.path.join(ART, "phase10_predictions.parquet")
P7_PRED = os.path.join(ART, "phase7_predictions_lgbm-graph-v1.parquet")


def test_frozen_model_score_preservation():
    if not (os.path.isfile(PRED_PATH) and os.path.isfile(P7_PRED)):
        pytest.skip("phase10 artifacts not present yet")
    import pandas as pd
    pred = pd.read_parquet(PRED_PATH)
    p7 = pd.read_parquet(P7_PRED)
    assert np.array_equal(pred["score"].to_numpy(dtype=float),
                          p7["score"].to_numpy(dtype=float))
    assert np.array_equal(pred["alert"].to_numpy(),
                          pred["score"].to_numpy(dtype=float) >= FROZEN)


def test_fit_artifact_reloads_and_reproduces_sets():
    if not (os.path.isfile(FIT_PATH) and os.path.isfile(PRED_PATH)):
        pytest.skip("phase10 artifacts not present yet")
    import pandas as pd
    fit_art = json.load(open(FIT_PATH))
    fit = {"method": fit_art["method"], "n1": fit_art["n1"],
           "n0": fit_art["n0"],
           "pos_sorted": np.asarray(fit_art["pos_sorted"], dtype=float),
           "neg_sorted": np.asarray(fit_art["neg_sorted"], dtype=float),
           "alphas": tuple(fit_art["alphas"])}
    pred = pd.read_parquet(PRED_PATH)
    s = pred["score"].to_numpy(dtype=float)
    np.testing.assert_allclose(pred["p1"].to_numpy(), p10.p1_values(fit, s))
    np.testing.assert_allclose(pred["p0"].to_numpy(), p10.p0_values(fit, s))
    for a in p10.ALPHAS:
        key = f"set_{int(a * 100):03d}"
        np.testing.assert_array_equal(pred[key].to_numpy(),
                                      p10.set_category(fit, s, a))


def test_experiment_record_and_decision_consistent():
    if not (os.path.isfile(EXP_PATH) and os.path.isfile(DEC_PATH)
            and os.path.isfile(CFG_PATH)):
        pytest.skip("phase10 artifacts not present yet")
    exp = json.load(open(EXP_PATH))
    dec = json.load(open(DEC_PATH))
    cfg = json.load(open(CFG_PATH))
    assert exp["method"] == cfg["method"] == "mondrian_split_conformal"
    assert exp["calibration_sample"] == {"n1": 323, "n0": 58677}
    assert exp["frozen_alert_policy"]["threshold"] == FROZEN
    assert exp["frozen_alert_policy"]["n_alerts_test"] == 49
    assert exp["decision"]["decision"] == dec["decision"]["decision"]
    assert abs(exp["evaluation"]["classification"]["auc_roc"] - 0.939157) < 1e-4
    assert abs(exp["evaluation"]["classification"]["auc_pr"] - 0.267776) < 1e-4


def test_predictions_parquet_shape():
    if not os.path.isfile(PRED_PATH):
        pytest.skip("phase10 predictions not present yet")
    import pandas as pd
    pred = pd.read_parquet(PRED_PATH)
    assert pred.shape == (47000, 11)
    assert list(pred.columns) == ["user", "day", "is_malicious", "score",
                                  "alert", "p1", "p0",
                                  "set_001", "set_005", "set_010", "set_020"]
    assert int(pred["is_malicious"].sum()) == 30
    assert not pred.isna().any().any()
    assert set(pred["set_005"].unique()) <= {"0", "1", "01", "empty"}