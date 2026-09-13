"""Phase 10: conformal prediction / confidence layer for the frozen model (CERT r4.2).

Central question: can a leakage-safe conformal method provide useful, honest
uncertainty information for the frozen `lgbm-graph-v1` detector under severe
class imbalance and the chronological TRAIN/CALIBRATION/TEST protocol?

Method (selected in Step 2, documented a priori):

  Mondrian (label-conditional) split conformal prediction on the frozen
  model's scores, with conformal p-values and prediction sets in
  {0}, {1}, {0, 1}, or empty. Calibration sample = the CALIBRATION block
  (323 positive / 58,677 negative user-days), which was never used for
  training the model.

Definitions (exact):

  Nonconformity scores (higher = more nonconforming):
    class 1: nc_1(x) = 1 - s(x)   (a low score is nonconforming for class 1)
    class 0: nc_0(x) = s(x)       (a high score is nonconforming for class 0)

  Conformal p-values (tie-inclusive / conservative convention, >= counts):
    p_1(x) = (#{i in cal_1: s_i <= s(x)} + 1) / (n_1 + 1)
    p_0(x) = (#{i in cal_0: s_i >= s(x)} + 1) / (n_0 + 1)

  Prediction set at level alpha:
    include class y iff p_y(x) > alpha.
    Sets: {1} confident positive, {0} confident negative,
          {0,1} ambiguous, {} empty (insufficient evidence for either label).

  Finite-sample rule: with n_y calibration points of class y, the class-y
  coverage guarantee P(y in set(x) | y) >= 1 - alpha holds under
  exchangeability of the calibration and test rows within each class
  (Mondrian validity). p-values are valid p-values under exchangeability;
  they are NOT posterior probabilities and the set is NOT a probability of
  correctness.

  Effective class-conditional calibration sample sizes:
    n_1 = 323 (positive class)  -- documented, small; the positive-class
    threshold t_1(alpha) is the floor(alpha * (n_1 + 1))-th smallest positive
    calibration score (16th at alpha = 0.05).

  Inclusion thresholds (exact mirrors of the p-value rule):
    t_1(alpha) = k_1-th smallest positive score, k_1 = floor(alpha * (n_1 + 1))
                 (include class 1 iff s(x) >= t_1)
    t_0(alpha) = k_0-th largest negative score,  k_0 = floor(alpha * (n_0 + 1))
                 (include class 0 iff s(x) <= t_0)

Decision rule (deterministic, applied after the single authorized TEST
evaluation; method/alpha/interpretation are frozen before TEST):

  D1 positive-class empirical TEST coverage >= 0.85 at alpha = 0.05 with
     Wilson 90% lower bound >= 0.75
  D2 negative-class empirical TEST coverage >= 0.90 at alpha = 0.05
  D3 usefulness at alpha = 0.05 on CALIBRATION: >= 5 alerts in the uncertain
     categories ({0,1} or empty) with confident-alert precision minus
     uncertain-alert precision >= 0.05, OR >= 10 monitor rows (set {1},
     score below the frozen alert threshold)
  D4 temporal: every 2-week CALIBRATION window's positive-class coverage at
     alpha = 0.05 >= 0.70
  D5 cost: conformal inference for CAL + TEST <= 2.0 s (vectorized CPU)

  D1-D5 all pass -> ACCEPT
  D1,D2,D4,D5 pass but D3 fails -> CONDITIONAL / DIAGNOSTIC ONLY
  otherwise -> REJECT
"""
from __future__ import annotations

import math

import numpy as np

ALPHAS = (0.01, 0.05, 0.10, 0.20)
FROZEN_ALERT_THRESHOLD = 0.9186015432508062  # Phase 9 frozen_max_f1 policy
Z_90 = 1.6448536269514722  # norm.ppf(0.95) for Wilson 90% intervals

# Decision-rule constants (documented a priori)
D1_POS_COVERAGE_MIN = 0.85
D1_WILSON_LB_MIN = 0.75
D2_NEG_COVERAGE_MIN = 0.90
D3_N_UNCERTAIN_ALERTS_MIN = 5
D3_PRECISION_GAP_MIN = 0.05
D3_N_MONITOR_MIN = 10
D4_WINDOW_POS_COVERAGE_MIN = 0.70
D5_MAX_INFERENCE_S = 2.0


# ---------------------------------------------------------------------------
# Fit + conformal p-values
# ---------------------------------------------------------------------------
def fit_conformal(scores_cal, y_cal, alphas: tuple = ALPHAS) -> dict:
    """Fit the Mondrian split-conformal layer on CALIBRATION scores/labels.

    Pure function of CALIBRATION data; never receives TEST rows or labels.
    Raises if either class is empty or labels are not {0, 1}.
    """
    scores_cal = np.asarray(scores_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    if scores_cal.shape != y_cal.shape:
        raise ValueError("scores and labels must be same-length arrays")
    if not set(np.unique(y_cal)) <= {0, 1}:
        raise ValueError("labels must be in {0, 1}")
    pos = np.sort(scores_cal[y_cal == 1])
    neg = np.sort(scores_cal[y_cal == 0])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("calibration requires at least one row of each class")
    return {
        "method": "mondrian_split_conformal",
        "n1": int(len(pos)),
        "n0": int(len(neg)),
        "pos_sorted": pos,
        "neg_sorted": neg,
        "alphas": tuple(float(a) for a in alphas),
    }


def p1_values(fit: dict, s) -> np.ndarray:
    """Conformal p-value for class 1: (#{cal_1 scores <= s} + 1) / (n_1 + 1)."""
    s = np.asarray(s, dtype=float)
    c = np.searchsorted(fit["pos_sorted"], s, side="right")
    return (c + 1) / (fit["n1"] + 1)


def p0_values(fit: dict, s) -> np.ndarray:
    """Conformal p-value for class 0: (#{cal_0 scores >= s} + 1) / (n_0 + 1)."""
    s = np.asarray(s, dtype=float)
    c = fit["n0"] - np.searchsorted(fit["neg_sorted"], s, side="left")
    return (c + 1) / (fit["n0"] + 1)


# ---------------------------------------------------------------------------
# Prediction sets
# ---------------------------------------------------------------------------
def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return alpha


def set_category(fit: dict, s, alpha: float) -> np.ndarray:
    """Prediction-set category per row: '1' | '0' | '01' | 'empty'."""
    alpha = _check_alpha(alpha)
    s = np.asarray(s, dtype=float)
    p1 = p1_values(fit, s) > alpha
    p0 = p0_values(fit, s) > alpha
    out = np.empty(len(s), dtype="<U5")
    out[p1 & p0] = "01"
    out[p1 & ~p0] = "1"
    out[~p1 & p0] = "0"
    out[~p1 & ~p0] = "empty"
    return out


def inclusion_thresholds(fit: dict, alpha: float) -> dict:
    """Exact inclusion boundaries (mirrors of the p-value rule; see docstring)."""
    alpha = _check_alpha(alpha)
    k1 = int(np.floor(alpha * (fit["n1"] + 1)))
    k0 = int(np.floor(alpha * (fit["n0"] + 1)))
    t1 = float(fit["pos_sorted"][k1 - 1]) if k1 >= 1 else -math.inf
    t0 = float(fit["neg_sorted"][fit["n0"] - k0]) if k0 >= 1 else math.inf
    return {"alpha": alpha, "k1": k1, "k0": k0, "t1": t1, "t0": t0}


# ---------------------------------------------------------------------------
# Diagnostics (CALIBRATION diagnostics; TEST evaluated once)
# ---------------------------------------------------------------------------
def coverage_stats(fit: dict, s, y, alpha: float) -> dict:
    """Empirical coverage + set-size distribution for one block (diagnostic).

    Marginal and class-conditional empirical coverage are diagnostics, not
    guarantees; guarantees require exchangeability of calibration + block.
    """
    alpha = _check_alpha(alpha)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=int)
    cats = set_category(fit, s, alpha)
    contains1 = (cats == "1") | (cats == "01")
    contains0 = (cats == "0") | (cats == "01")
    is_pos = y == 1
    is_neg = y == 0
    n = len(y)
    return {
        "n": int(n),
        "n_pos": int(is_pos.sum()),
        "n_neg": int(is_neg.sum()),
        "marginal_coverage": float(np.mean((is_pos & contains1) | (is_neg & contains0))),
        "pos_coverage": float(np.mean(contains1[is_pos]) if is_pos.any() else float("nan")),
        "neg_coverage": float(np.mean(contains0[is_neg]) if is_neg.any() else float("nan")),
        "set_size_dist": {
            "singleton1": float(np.mean(cats == "1")),
            "singleton0": float(np.mean(cats == "0")),
            "ambiguous": float(np.mean(cats == "01")),
            "empty": float(np.mean(cats == "empty")),
            "mean_size": float(np.mean((cats == "1") + (cats == "0") + 2 * (cats == "01"))),
        },
        "positive_prediction_rate": float(np.mean(contains1)),
        "positive_inclusion_count": int(contains1.sum()),
    }


def policy_crosstab(fit: dict, s, alert_mask, y, alpha: float) -> dict:
    """Alert status (frozen Phase 9 threshold) x conformal category table.

    Analysis only; the frozen alert policy is not modified.
    """
    alpha = _check_alpha(alpha)
    s = np.asarray(s, dtype=float)
    alert = np.asarray(alert_mask, dtype=bool)
    y = np.asarray(y, dtype=int)
    cats = set_category(fit, s, alpha)
    cats_al = cats[alert]
    cats_no = cats[~alert]
    y_al = y[alert]
    y_no = y[~alert]
    cells = {}
    for name, cc, yy in (("alert", cats_al, y_al), ("no_alert", cats_no, y_no)):
        cells[name] = {}
        for cat in ("1", "0", "01", "empty"):
            m = cc == cat
            n_c = int(m.sum())
            cells[name][cat] = {
                "n": n_c,
                "precision": float(yy[m].mean()) if n_c else 0.0,
            }
    n_al = int(alert.sum())
    n_unc_al = int(((cats_al == "01") | (cats_al == "empty")).sum())
    return {
        "alpha": alpha,
        "cells": cells,
        "n_alerts": n_al,
        "n_uncertain_alerts": n_unc_al,
        "n_monitor": int(((cats_no == "1")).sum()),
        "uncertain_alert_rate": float(n_unc_al / n_al) if n_al else 0.0,
    }


def wilson_ci(k: int, n: int, confidence: float = 0.90) -> tuple | None:
    """Wilson score interval for a binomial proportion; None for n == 0."""
    if n == 0:
        return None
    p = k / n
    z = Z_90 if confidence == 0.90 else 1.959963984540054  # 0.95
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------------------
# Decision rule (deterministic; applied after the single TEST evaluation)
# ---------------------------------------------------------------------------
def decide_conformal(evidence: dict) -> dict:
    """Apply the documented D1-D5 rule; returns the verdict + reasons."""
    reasons = {}
    t = evidence["test_005"]
    c = evidence["cal_005"]
    reasons["D1"] = (f"TEST positive coverage {t['pos_coverage']:.3f} >= "
                     f"{D1_POS_COVERAGE_MIN} and Wilson LB "
                     f"{t['pos_wilson_lb']:.3f} >= {D1_WILSON_LB_MIN}"
                     if (t["pos_coverage"] >= D1_POS_COVERAGE_MIN
                         and t["pos_wilson_lb"] >= D1_WILSON_LB_MIN)
                     else f"FAIL: TEST positive coverage {t['pos_coverage']:.3f}"
                          f" (Wilson LB {t['pos_wilson_lb']:.3f})")
    reasons["D2"] = (f"TEST negative coverage {t['neg_coverage']:.3f} >= "
                     f"{D2_NEG_COVERAGE_MIN}"
                     if t["neg_coverage"] >= D2_NEG_COVERAGE_MIN
                     else f"FAIL: TEST negative coverage {t['neg_coverage']:.3f}")
    useful = ((c["n_uncertain_alerts"] >= D3_N_UNCERTAIN_ALERTS_MIN
               and c["precision_gap"] >= D3_PRECISION_GAP_MIN)
              or c["n_monitor"] >= D3_N_MONITOR_MIN)
    reasons["D3"] = (f"usefulness: uncertain alerts {c['n_uncertain_alerts']}"
                     f" (gap {c['precision_gap']:.3f}), monitor rows "
                     f"{c['n_monitor']}" if useful
                     else f"FAIL: uncertain alerts {c['n_uncertain_alerts']}"
                          f" (gap {c['precision_gap']:.3f}), monitor rows "
                          f"{c['n_monitor']}")
    reasons["D4"] = (f"all {len(c['window_pos_coverage'])} windows >= "
                     f"{D4_WINDOW_POS_COVERAGE_MIN}"
                     if min(c["window_pos_coverage"]) >= D4_WINDOW_POS_COVERAGE_MIN
                     else f"FAIL: window coverages {c['window_pos_coverage']}")
    reasons["D5"] = (f"inference {evidence['inference_s']:.4f} s <= "
                     f"{D5_MAX_INFERENCE_S}"
                     if evidence["inference_s"] <= D5_MAX_INFERENCE_S
                     else f"FAIL: inference {evidence['inference_s']:.4f} s")
    d1 = reasons["D1"].startswith("TEST")
    d2 = reasons["D2"].startswith("TEST")
    d3 = reasons["D3"].startswith("usefulness")
    d4 = reasons["D4"].startswith("all")
    d5 = reasons["D5"].startswith("inference")
    if d1 and d2 and d4 and d5:
        verdict = "ACCEPT" if d3 else "CONDITIONAL / DIAGNOSTIC ONLY"
    else:
        verdict = "REJECT"
    return {"decision": verdict, "reasons": reasons,
            "rule": ("D1-D5 documented rule; D3 failure alone downgrades "
                     "ACCEPT to CONDITIONAL / DIAGNOSTIC ONLY")}