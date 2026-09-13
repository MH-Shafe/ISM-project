"""Phase 17: Targeted Calibration/Generalization Intervention on Unseen Users.

Implements the approved protocol in docs/phase17_specification.md:

  - Gates 1/1b/2/3/4 (arm-A reproduction, operating-point equality, CAL
    artifact reproduction, input md5s, frozen-system immutability);
  - Arms A-D: A frozen reference (no transform, frozen threshold);
    B Platt scaling; C binning recalibration (K=20, n<30 merge, PAV);
    D rank-preserving ECDF calibrated probability mapping;
  - Calibration fitted on the user-disjoint CAL block ONLY (sorted by
    (user, day)); applied in memory to the frozen user-disjoint TEST
    score artifact;
  - Metrics: Brier/ECE/reliability (CAL and TEST), operating-point
    metrics + user-level coverage (Wilson 90%), ranking verification vs
    recorded Phase 14 values, user-block bootstrap (n=1000, seed 42),
    CAL-refit calibration-transfer bootstrap;
  - PASS/CAUTION/FAIL verdict computed mechanically from the
    pre-registered criteria (specification Section 14).

Hard constraints (enforced structurally): the authoritative chronological
TEST never enters this module; no training, no threshold hunting, no
TEST-informed selection; the Phase 14 allocation is preserved; TEST rows
are only read from the frozen artifact and never re-scored.

The module is IO-free and deterministic (all resampling uses seed 42;
Platt fits are order-fixed by sorting CAL rows on (user, day)). Evidence
labels: statistics computed here on verified inputs are OBSERVED;
interpretation is done by the report writer per specification Section 12.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments.phase14 import (
    Z90, canonical_json, gini_of_counts, to_jsonable,
    user_block_bootstrap_ci, wilson_ci,
)

# ---------------------------------------------------------------------------
# Pre-registered constants (specification Appendix A)
# ---------------------------------------------------------------------------
FROZEN_THRESHOLD = 0.9186015432508062
RECORDED_CAL_AUC_ROC = 0.7355124228200639
RECORDED_SECONDARY_THRESHOLD = 0.9837865316173778
RECORDED_TEST_AUC_ROC = 0.786513147174144
RECORDED_TEST_AUC_PR = 0.3757395445913874
RECORDED_PRIMARY = {
    "n_alerts": 1026, "precision": 0.20175438596491227,
    "recall": 0.43487394957983194, "f1": 0.2756324900133156,
    "mcc": 0.2872810379712531, "tp": 207, "fp": 819, "tn": 52813,
    "fn": 269, "balanced_accuracy": 0.7098016078448086,
    "alert_rate": 0.018962075848303395,
}
RECORDED_TOP_K = {
    "precision_at_10": 1.0, "precision_at_30": 1.0, "precision_at_50": 1.0,
    "precision_at_100": 0.95, "recall_at_10": 0.02100840336134454,
    "recall_at_30": 0.06302521008403361, "recall_at_50": 0.10504201680672269,
    "recall_at_100": 0.19957983193277312,
}
SEED = 42
N_BOOT = 1000
ALPHA = 0.10
BINNING_K = 20
MIN_BIN_N = 30
MIN_BINS = 5
ECE_N_BINS = 10
PLATT_C = 1.0
PLATT_MAX_ITER = 1000
PLATT_TOL = 1e-4
# Gate tolerances (specification Appendix A / Section 9.5)
GATE_REL_TOL = 1e-12          # arm-A reproduction, relative
GATE_AUC_TOL = 1e-9           # CAL AUC reproduction
GATE_THRESHOLD_TOL = 1e-12    # gate 1b operating-point equality
GATE_RANKING_TOL_MONO = 1e-12  # strictly monotone arms (B, D)
GATE_RANKING_TOL = 1e-9       # all arms (order-preserving C included)
CAUTION_TIE_TOL = 1e-3        # arm-C tie threshold for CAUTION

# Success bounds (specification Section 14)
PASS_BRIER_IMPROVEMENT = 0.20
PASS_ECE = 0.05
PASS_F1 = 0.40
PASS_PRECISION = 0.70
PASS_ALERTS = 300
PASS_COVERAGE = 8


def _close(a: float, b: float, rel: float = GATE_REL_TOL) -> bool:
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))


# ---------------------------------------------------------------------------
# Arm B — Platt scaling (specification Section 9)
# ---------------------------------------------------------------------------
def platt_fit(s: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression of y on s (single score feature); returns fit."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=int)
    clf = LogisticRegression(C=PLATT_C, fit_intercept=True,
                             solver="lbfgs", max_iter=PLATT_MAX_ITER,
                             tol=PLATT_TOL)
    clf.fit(s.reshape(-1, 1), y)
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return {
        "method": "Platt scaling (LogisticRegression, C=1.0, lbfgs, "
                  "max_iter=1000, tol=1e-4) on the single score feature",
        "a": a, "b": b,
        "monotone_increasing": a > 0,
        "n_fit_rows": int(len(s)),
        "n_positives": int(y.sum()),
        "score_range": [float(s.min()), float(s.max())],
        "aborted": a <= 0,
        "abort_reason": ("non-monotone fit (a <= 0); pre-registered abort "
                         "per specification Section 9") if a <= 0 else None,
    }


def platt_apply(s: np.ndarray, fit: dict) -> np.ndarray:
    """p_hat = sigmoid(a*s + b); strictly increasing if a > 0."""
    if fit["aborted"]:
        raise ValueError("cannot apply an aborted Platt fit")
    s = np.asarray(s, dtype=float)
    return expit(fit["a"] * s + fit["b"]).astype(np.float64)


# ---------------------------------------------------------------------------
# Arm C — binning recalibration (specification Section 9)
# ---------------------------------------------------------------------------
def _merge_small_bins(ns: list[int], tps: list[int],
                      min_n: int = MIN_BIN_N):
    """Merge bins with n < min_n into the smaller neighbor (tie -> left).

    Returns (groups, merged_ns, merged_tps, merge_log); groups maps each
    final group to the contiguous original bin indices it absorbed.
    """
    groups = [[i] for i in range(len(ns))]
    ns = list(ns)
    tps = list(tps)
    merge_log = []
    while True:
        small = [i for i, n in enumerate(ns) if n < min_n]
        if not small:
            break
        i = small[0]
        if i == 0:
            j = 1
        elif i == len(ns) - 1:
            j = i - 1
        else:
            j = i - 1 if ns[i - 1] <= ns[i + 1] else i + 1
        merge_log.append({"absorbed_group_position": i,
                          "into_group_position": j})
        ns[j] += ns[i]
        tps[j] += tps[i]
        groups[j] += groups[i]
        del ns[i], tps[i], groups[i]
    return groups, ns, tps, merge_log


def _pav_pools(ns: list[int], tps: list[int]):
    """Pool-adjacent-violators over group rates (weights = n).

    Returns (rates, pool_of_group): rate per PAV pool, and the pool index
    of each pre-PAV group. PAV pools are contiguous spans of groups; the
    span structure is preserved under neighbor merges.
    """
    n = [float(x) for x in ns]
    t = [float(x) for x in tps]
    rates = [ti / ni for ti, ni in zip(t, n)]
    spans = [[i] for i in range(len(rates))]
    i = 0
    while i < len(rates) - 1:
        if rates[i] <= rates[i + 1]:
            i += 1
            continue
        n[i] += n[i + 1]
        t[i] += t[i + 1]
        rates[i] = t[i] / n[i]
        spans[i] += spans[i + 1]
        del n[i + 1], t[i + 1], rates[i + 1], spans[i + 1]
        i = max(0, i - 1)
    pool_of_group = [0] * sum(len(sp) for sp in spans)
    for pi, span in enumerate(spans):
        for g in span:
            pool_of_group[g] = pi
    return rates, pool_of_group


def binning_fit(s: np.ndarray, y: np.ndarray, k: int = BINNING_K,
                min_n: int = MIN_BIN_N, min_bins: int = MIN_BINS) -> dict:
    """K equal-frequency bins, n<min_n merge, PAV smoothing; returns fit."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.quantile(s, np.linspace(0.0, 1.0, k + 1))
    bin_id = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, k - 1)
    ns = [int((bin_id == i).sum()) for i in range(k)]
    tps = [int(y[bin_id == i].sum()) for i in range(k)]
    groups, ns_m, tps_m, merge_log = _merge_small_bins(ns, tps, min_n)
    rates, pool_of_group = _pav_pools(ns_m, tps_m)
    n_groups = len(ns_m)
    group_of_bin = np.zeros(k, dtype=int)
    for gi, g in enumerate(groups):
        for b in g:
            group_of_bin[b] = gi
    bin_rate = np.asarray(rates, dtype=float)[
        np.asarray(pool_of_group, dtype=int)[group_of_bin]].tolist()
    fit = {
        "method": f"binning recalibration: K={k} equal-frequency bins on CAL "
                  f"scores, bins with n < {min_n} merged with the smaller "
                  f"neighbor (tie -> left), monotone PAV smoothing of "
                  f"within-bin rates",
        "k": k, "min_bin_n": min_n, "n_merged_groups": n_groups,
        "n_pav_pools": len(rates),
        "edges": [float(e) for e in edges],
        "original_bin_n": ns, "original_bin_tp": tps,
        "merged_bin_n": ns_m, "merged_bin_tp": tps_m,
        "pav_rates": rates,
        "bin_rate": bin_rate,
        "merge_log": merge_log,
        "aborted": n_groups < min_bins,
        "abort_reason": (f"fewer than {min_bins} bins remain after merging "
                         f"(got {n_groups}); pre-registered abort per "
                         "specification Section 9") if n_groups < min_bins
        else None,
    }
    return fit


def binning_apply(s: np.ndarray, fit: dict) -> np.ndarray:
    """Step-function mapping: p_hat(s) = PAV rate of the bin containing s."""
    if fit["aborted"]:
        raise ValueError("cannot apply an aborted binning fit")
    s = np.asarray(s, dtype=float)
    edges = np.asarray(fit["edges"], dtype=float)
    k = int(fit["k"])
    bin_id = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, k - 1)
    bin_rate = np.asarray(fit["bin_rate"], dtype=float)
    return bin_rate[bin_id].astype(np.float64)


# ---------------------------------------------------------------------------
# Arm D — ECDF calibrated probability mapping (specification Section 9)
# ---------------------------------------------------------------------------
def ecdf_apply(s_cal: np.ndarray, s_new: np.ndarray) -> np.ndarray:
    """p_hat(s) = (rank_mid(s) − 1/2)/n over CAL rows (specification
    Section 9 / line 388 formula; midpoint-rank tie convention; strictly
    monotone in s except at exact ties; denominator n = 54,108 CAL rows).

    TEST scores outside the CAL range map marginally outside [0, 1]
    (reported OBSERVED in the fit record; no clamping per the
    pre-registered formula).
    """
    s_cal = np.sort(np.asarray(s_cal, dtype=float))
    s_new = np.asarray(s_new, dtype=float)
    n = len(s_cal)
    lo = np.searchsorted(s_cal, s_new, side="left")
    hi = np.searchsorted(s_cal, s_new, side="right")
    rank_mid = (lo + hi) / 2.0
    return (rank_mid - 0.5) / n


# ---------------------------------------------------------------------------
# Calibration-quality metrics (specification Section 10.4)
# ---------------------------------------------------------------------------
def brier_score(y: np.ndarray, p_hat: np.ndarray) -> float:
    """Brier = mean((p_hat - y)^2)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece_score(y: np.ndarray, p_hat: np.ndarray,
              n_bins: int = ECE_N_BINS) -> dict:
    """ECE with K=10 equal-frequency bins over the evaluated p_hat.

    Returns ECE plus the per-bin reliability data (fixed K, no tuning).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p_hat, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ys = y[order]
    bins = np.array_split(np.arange(n), n_bins)
    rel = []
    ece = 0.0
    for i, idx in enumerate(bins):
        pk = float(ps[idx].mean())
        rk = float(ys[idx].mean())
        nk = int(len(idx))
        rel.append({"bin": i, "n": nk, "mean_p": pk,
                    "empirical_rate": rk,
                    "abs_gap": abs(pk - rk)})
        ece += (nk / n) * abs(pk - rk)
    return {
        "n_bins": n_bins,
        "binning": "equal-frequency over the evaluated p_hat (descriptive "
                   "summarization; fixed K, no tuning)",
        "ece": float(ece),
        "reliability": rel,
    }


def calibration_metrics(y: np.ndarray, p_hat: np.ndarray) -> dict:
    return {"brier": brier_score(y, p_hat), "ece": ece_score(y, p_hat)}


# ---------------------------------------------------------------------------
# Operating point (single pre-registered rule, CAL only; spec Section 10.3)
# ---------------------------------------------------------------------------
def best_f1_operating_point(y_cal: np.ndarray, p_cal: np.ndarray) -> dict:
    t, f1 = th.best_f1_threshold(y_cal, p_cal)
    return {"threshold": t, "cal_best_f1": f1,
            "rule": "best-F1 threshold on CAL scores (src/evaluation/"
                    "threshold.py best_f1_threshold), applied once, never "
                    "TEST-informed"}


# ---------------------------------------------------------------------------
# Gates (specification Section 9.5)
# ---------------------------------------------------------------------------
def gate1_test_reproduction(test: pd.DataFrame) -> dict:
    """Arm-A reference metrics vs the recorded Phase 14 values."""
    y = test["is_malicious"].to_numpy().astype(int)
    s = test["score"].to_numpy().astype(float)
    cls = m.classification_metrics(y, s)
    op = m.binary_decision_metrics(y, s, FROZEN_THRESHOLD)
    cross = int((test["alert_primary"].to_numpy().astype(bool)
                 != (s >= FROZEN_THRESHOLD)).sum())
    checks = {
        "auc_roc": _close(cls["auc_roc"], RECORDED_TEST_AUC_ROC),
        "auc_pr": _close(cls["auc_pr"], RECORDED_TEST_AUC_PR),
        "n_alerts": int(op["n_alerts"]) == RECORDED_PRIMARY["n_alerts"],
        "precision": _close(op["precision"], RECORDED_PRIMARY["precision"]),
        "recall": _close(op["recall"], RECORDED_PRIMARY["recall"]),
        "f1": _close(op["f1"], RECORDED_PRIMARY["f1"]),
        "mcc": _close(op["mcc"], RECORDED_PRIMARY["mcc"]),
        "tp_fp_tn_fn": (op["tp"], op["fp"], op["tn"], op["fn"]) == (
            RECORDED_PRIMARY["tp"], RECORDED_PRIMARY["fp"],
            RECORDED_PRIMARY["tn"], RECORDED_PRIMARY["fn"]),
        "balanced_accuracy": _close(op["balanced_accuracy"],
                                    RECORDED_PRIMARY["balanced_accuracy"]),
        "alert_rate": _close(op["alert_rate"],
                             RECORDED_PRIMARY["alert_rate"]),
    }
    return {
        "threshold": FROZEN_THRESHOLD,
        "recomputed": {
            "classification": cls,
            "at_primary_threshold": op,
            "alert_primary_column_mismatches": cross,
        },
        "checks": checks,
        "passed": all(checks.values()) and cross == 0,
        "tolerance_note": "1e-12 relative for floats; exact integers",
        "reference": "phase14_experiment.json test_metrics "
                     "(md5-verified input)",
    }


def gate2_cal_reproduction(cal: pd.DataFrame) -> dict:
    """CAL artifact reproduction vs recorded Phase 15/14 values."""
    y = cal["is_malicious"].to_numpy().astype(int)
    s = cal["score"].to_numpy().astype(float)
    n = len(cal)
    pos = int(y.sum())
    ge = s >= FROZEN_THRESHOLD
    n_ge = int(ge.sum())
    tp_ge = int((ge & (y == 1)).sum())
    auc = float(roc_auc_score(y, s))
    checks = {
        "rows": n == 54108,
        "positives": pos == 379,
        "rows_ge_threshold": n_ge == 580,
        "true_positives_ge_threshold": tp_ge == 135,
        "cal_auc_roc": abs(auc - RECORDED_CAL_AUC_ROC) <= GATE_AUC_TOL,
    }
    return {
        "n_rows": n, "n_positives": pos,
        "rows_ge_threshold": n_ge, "true_positives_ge_threshold": tp_ge,
        "cal_auc_roc": auc,
        "recorded_cal_auc_roc": RECORDED_CAL_AUC_ROC,
        "checks": checks,
        "passed": all(checks.values()),
        "tolerance_note": "AUC within 1e-9 (Phase 14 gate 2 convention)",
        "reference": ("phase15_cal_scores.parquet (md5-verified); Phase 15 "
                      "threshold diagnostic + Phase 14 calibration records"),
    }


def gate1b_operating_equality(y_cal: np.ndarray, s_cal: np.ndarray,
                              s_te: np.ndarray, fits: dict) -> dict:
    """B and D operating thresholds must equal the Phase 14 secondary
    threshold in raw space (monotone invariance); C reported OBSERVED."""
    t_raw, f1_raw = th.best_f1_threshold(y_cal, s_cal)
    raw_close = _close(t_raw, RECORDED_SECONDARY_THRESHOLD,
                       GATE_THRESHOLD_TOL)
    out = {
        "raw_cal_best_f1_threshold": t_raw,
        "recorded_secondary_threshold": RECORDED_SECONDARY_THRESHOLD,
        "raw_threshold_close": raw_close,
        "arms": {},
    }
    for arm in ("B", "C", "D"):
        rec = fits[arm]
        if rec["p_cal"] is None:
            out["arms"][arm] = {"gate_applies": False,
                                "status": "aborted",
                                "note": "arm aborted; gate not applicable"}
            continue
        p_cal = rec["p_cal"]
        t_arm, _ = th.best_f1_threshold(y_cal, p_cal)
        s_arm = _raw_threshold_of(p_cal, s_cal, t_arm)
        alert_te = rec["apply"](s_te) >= t_arm
        alert_raw_te = s_te >= t_raw
        out["arms"][arm] = {
            "gate_applies": arm in ("B", "D"),
            "transformed_threshold": t_arm,
            "raw_space_threshold": s_arm,
            "raw_threshold_close": _close(s_arm, t_raw,
                                          GATE_THRESHOLD_TOL),
            "alert_set_rows_identical_to_raw": bool(
                np.array_equal(alert_te, alert_raw_te)),
            "n_alert_rows_test": int(alert_te.sum()),
            "n_alert_rows_test_raw": int(alert_raw_te.sum()),
            "n_rows_differing_from_raw": int(
                (alert_te != alert_raw_te).sum()),
            "status": "evaluated",
        }
    out["arms"]["C"].setdefault(
        "note", ("bin-level alerting can legitimately differ from the raw "
                 "point; reported as an OBSERVED discretization/tie effect "
                 "(specification Section 12.4)"))
    out["passed"] = (raw_close
                     and out["arms"]["B"].get("raw_threshold_close", False)
                     and out["arms"]["D"].get("raw_threshold_close", False)
                     and out["arms"]["B"].get(
                         "alert_set_rows_identical_to_raw", False)
                     and out["arms"]["D"].get(
                         "alert_set_rows_identical_to_raw", False))
    return out


def _raw_threshold_of(p_cal: np.ndarray, s_cal: np.ndarray,
                      t_arm: float) -> float:
    """Smallest raw CAL score whose transformed value >= t_arm."""
    return float(s_cal[p_cal >= t_arm].min())


# ---------------------------------------------------------------------------
# Coverage, per-user, scenario (specification Sections 10.5/10.6/12.5)
# ---------------------------------------------------------------------------
def coverage_metrics(test: pd.DataFrame, alert: np.ndarray,
                     user_diag: dict) -> dict:
    """User-level detection at an operating point (Wilson 90% CI)."""
    df = pd.DataFrame({
        "user": test["user"].to_numpy(),
        "mal": test["is_malicious"].to_numpy().astype(int),
        "alert": np.asarray(alert, dtype=bool),
    })
    per = {d["user"]: d for d in user_diag["per_user"]}
    mal_users = sorted(u for u, d in per.items() if d["n_positive_days"] > 0)
    n_alerts_user = df.groupby("user")["alert"].sum()
    detected = [u for u in mal_users if int(n_alerts_user.get(u, 0)) > 0]
    zero = [u for u in mal_users if u not in detected]
    alerted_any = df[df["alert"] & (df["mal"] == 1)]
    full = [u for u in detected
            if int(alerted_any[alerted_any["user"] == u]["alert"].sum())
            == per[u]["n_positive_days"]]
    counts = n_alerts_user.reindex(sorted(df["user"].unique()),
                                   fill_value=0).to_numpy()
    alerted = df[df["alert"]]["user"].nunique()
    gini = gini_of_counts(counts[counts > 0]) if alerted else 0.0
    return {
        "n_malicious_test_users": len(mal_users),
        "coverage": {
            "detected": len(detected), "total": len(mal_users),
            "frac": len(detected) / len(mal_users),
            "wilson_90": wilson_ci(len(detected), len(mal_users)),
        },
        "detected_users": detected,
        "zero_alert_users": zero,
        "fully_detected_users": full,
        "per_user_alerts": {u: int(n_alerts_user.get(u, 0))
                            for u in mal_users},
        "alerts_per_user": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": float(counts.max()),
        },
        "alert_concentration": {
            "n_alerts_total": int(df["alert"].sum()),
            "n_alerted_users": int(alerted),
            "gini": gini,
        },
    }


def coverage_by_strata(test: pd.DataFrame, alert: np.ndarray,
                       user_diag: dict) -> dict:
    """Coverage sub-analysis by tercile / length class / onset half."""
    df = pd.DataFrame({
        "user": test["user"].to_numpy(),
        "alert": np.asarray(alert, dtype=bool),
    })
    per = {d["user"]: d for d in user_diag["per_user"]}
    out = {}
    for field in ("activity_tercile", "length_class", "temporal_half",
                  "scenario"):
        groups = {}
        for u, d in per.items():
            if d["n_positive_days"] <= 0:
                continue
            level = str(d.get(field))
            groups.setdefault(level, []).append(u)
        out[field] = {
            level: {
                "n_users": len(users),
                "detected": int(sum(1 for u in users
                                    if df[df["user"] == u]["alert"].any())),
                "users": sorted(users),
            }
            for level, users in sorted(groups.items())}
    return out


def scenario_recall(test: pd.DataFrame, alert: np.ndarray,
                    user_diag: dict) -> dict:
    """Per-scenario recall (evaluation-only; scenario mapping reused from
    the Phase 14 per-user diagnostics record, NOT VERIFIED independently)."""
    per = {d["user"]: d for d in user_diag["per_user"]}
    df = pd.DataFrame({
        "user": test["user"].to_numpy(),
        "mal": test["is_malicious"].to_numpy().astype(int),
        "alert": np.asarray(alert, dtype=bool),
    })
    out = {}
    for scn in sorted({d["scenario"] for d in per.values()
                       if d["scenario"] is not None}):
        users = [u for u, d in per.items() if d["scenario"] == scn]
        mask = df["user"].isin(users)
        n_pos = int((mask & (df["mal"] == 1)).sum())
        if n_pos == 0:
            out[str(scn)] = {"n_malicious_rows": 0,
                             "recall": None,
                             "note": "NOT VERIFIED (zero positive rows)"}
            continue
        out[str(scn)] = {
            "n_malicious_rows": n_pos,
            "recall": float((mask & (df["mal"] == 1) & df["alert"]).sum())
            / n_pos,
        }
    return out


# ---------------------------------------------------------------------------
# Ranking verification (specification Section 12.2)
# ---------------------------------------------------------------------------
def ranking_verification(y: np.ndarray, p_hat: np.ndarray,
                         arm_name: str) -> dict:
    """Ranking metrics + deltas vs the recorded Phase 14 values."""
    cls = m.classification_metrics(y, p_hat)
    top_k = m.top_k_metrics(y, p_hat)
    deltas = {
        "auc_roc": cls["auc_roc"] - RECORDED_TEST_AUC_ROC,
        "auc_pr": cls["auc_pr"] - RECORDED_TEST_AUC_PR,
    }
    for k in RECORDED_TOP_K:
        deltas[k] = top_k[k] - RECORDED_TOP_K[k]
    return {
        "arm": arm_name,
        "classification": cls,
        "top_k": top_k,
        "recorded_phase14": {
            "auc_roc": RECORDED_TEST_AUC_ROC,
            "auc_pr": RECORDED_TEST_AUC_PR,
            "top_k": RECORDED_TOP_K,
        },
        "deltas": deltas,
        "auc_roc_delta_abs": abs(deltas["auc_roc"]),
        "note": ("monotone arms cannot change ranking metrics (mathematical "
                 "identity, specification Section 10.1); arm C may change "
                 "AUC-PR/P@k only through bin-level tie effects (reported "
                 "OBSERVED, no tolerance claimed)"),
    }


# ---------------------------------------------------------------------------
# Bootstrap (specification Section 13)
# ---------------------------------------------------------------------------
def user_block_bootstrap_operating(users, y, p_hat, threshold: float,
                                   n_boot: int = N_BOOT,
                                   seed: int = SEED) -> dict:
    """User-block percentile bootstrap of operating metrics at a fixed
    threshold (evaluation-only; threshold never refit)."""
    users = np.asarray(users, dtype=object)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p_hat, dtype=float)
    unique = np.unique(users)
    idx_by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rows = {metric: [] for metric in ("n_alerts", "precision",
                                      "recall", "f1")}
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        op = m.binary_decision_metrics(y[idx], p[idx], threshold)
        rows["n_alerts"].append(op["n_alerts"])
        rows["precision"].append(op["precision"])
        rows["recall"].append(op["recall"])
        rows["f1"].append(op["f1"])
    out = {"n_users": int(len(unique)), "n_boot": n_boot,
           "n_skipped": 0, "threshold": threshold}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 100 * ALPHA / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - ALPHA / 2))),
        }
    out["method"] = ("user-block percentile bootstrap (resample TEST users "
                     "with replacement, pool rows, recompute operating "
                     "metrics at the fixed pre-registered threshold)")
    return out


def calibration_transfer_bootstrap(cal_users, cal_s, cal_y, te_s, te_y,
                                   method: str, n_boot: int = N_BOOT,
                                   seed: int = SEED) -> dict:
    """CAL-refit transfer bootstrap: resample CAL users, refit the arm's
    calibration, apply to the FIXED frozen TEST scores, record TEST
    Brier/ECE (specification Section 13)."""
    cal_users = np.asarray(cal_users, dtype=object)
    cal_s = np.asarray(cal_s, dtype=float)
    cal_y = np.asarray(cal_y, dtype=int)
    te_s = np.asarray(te_s, dtype=float)
    te_y = np.asarray(te_y, dtype=int)
    unique = np.unique(cal_users)
    idx_by_user = {u: np.flatnonzero(cal_users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rows = {"brier": [], "ece": []}
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        idx = np.sort(idx)  # deterministic (user, day) order within resample
        sc, yc = cal_s[idx], cal_y[idx]
        try:
            p_te = _fit_and_apply(method, sc, yc, te_s)
        except (ValueError, AssertionError):
            skipped += 1
            continue
        rows["brier"].append(brier_score(te_y, p_te))
        rows["ece"].append(ece_score(te_y, p_te)["ece"])
    out = {"method": method, "n_cal_users": int(len(unique)),
           "n_boot": n_boot, "n_skipped": skipped,
           "note": ("CAL users resampled with replacement, calibration "
                    "refit inside each resample, applied to the FIXED "
                    "frozen TEST score artifact")}
    for metric, values in rows.items():
        values = np.asarray(values)
        out[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 100 * ALPHA / 2)),
            "ci_high": float(np.percentile(values, 100 * (1 - ALPHA / 2))),
        }
    return out


def _fit_and_apply(method: str, sc: np.ndarray, yc: np.ndarray,
                   s_new: np.ndarray) -> np.ndarray:
    if method == "platt":
        fit = platt_fit(sc, yc)
        if fit["aborted"]:
            raise ValueError("aborted Platt fit in bootstrap resample")
        return platt_apply(s_new, fit)
    if method == "binning":
        fit = binning_fit(sc, yc)
        if fit["aborted"]:
            raise ValueError("aborted binning fit in bootstrap resample")
        return binning_apply(s_new, fit)
    if method == "ecdf":
        return ecdf_apply(sc, s_new)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Pre-registered verdict (specification Section 14)
# ---------------------------------------------------------------------------
def compute_verdict(arms: dict, gates: dict, determinism_ok: bool,
                    tie_effect_max: float) -> dict:
    """Mechanical PASS / CAUTION / FAIL from the pre-registered criteria."""
    gates_ok = (gates["gate1"]["passed"] and gates["gate1b"]["passed"]
                and gates["gate2"]["passed"] and gates["gate3"]["passed"]
                and gates["gate4"]["passed"] and determinism_ok)

    def transfers(arm: dict) -> bool:
        return (arm["test_calibration"]["brier_improvement"] >=
                PASS_BRIER_IMPROVEMENT
                and arm["test_calibration"]["ece"] <= PASS_ECE)

    def operates(arm: dict) -> bool:
        op = arm["operating_test"]
        return (op["f1"] >= PASS_F1 and op["precision"] >= PASS_PRECISION
                and op["n_alerts"] <= PASS_ALERTS
                and op["coverage"]["coverage"]["detected"] >= PASS_COVERAGE)

    ranking_ok = all(a["ranking"]["auc_roc_delta_abs"] <= GATE_RANKING_TOL
                     for a in arms.values() if a["status"] == "evaluated")
    ranking_mono_ok = all(
        arms[name]["ranking"]["auc_roc_delta_abs"] <= GATE_RANKING_TOL_MONO
        for name in ("B", "D") if arms[name]["status"] == "evaluated")
    coverage_all_ok = all(
        a["operating_test"]["coverage"]["coverage"]["detected"]
        >= PASS_COVERAGE
        for name, a in arms.items()
        if name in ("B", "C", "D") and a["status"] == "evaluated")

    evaluated = {n: a for n, a in arms.items() if a["status"] == "evaluated"}
    transfer_arm = [name for name, a in evaluated.items() if transfers(a)]
    operate_arm = [name for name, a in evaluated.items() if operates(a)]

    crit = {
        "gates_pass": gates_ok,
        "ranking_preserved_all_arms": ranking_ok,
        "ranking_preserved_strictly_monotone": ranking_mono_ok,
        "coverage_at_least_8_of_15_every_calibrated_arm": coverage_all_ok,
        "some_arm_calibration_transfers": bool(transfer_arm),
        "some_arm_meets_operating_targets": bool(operate_arm),
        "same_arm_transfers_and_operates": bool(
            set(transfer_arm) & set(operate_arm)),
        "arm_c_tie_effects_within_caution_tol": tie_effect_max
        <= CAUTION_TIE_TOL,
    }

    if not gates_ok or not ranking_ok or not ranking_mono_ok:
        verdict = "FAIL"
    elif tie_effect_max > CAUTION_TIE_TOL:
        verdict = "CAUTION"
    elif coverage_all_ok and set(transfer_arm) & set(operate_arm):
        verdict = "PASS"
    elif transfer_arm or operate_arm:
        verdict = "CAUTION"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "criteria": crit,
        "transfer_arms": transfer_arm,
        "operating_target_arms": operate_arm,
        "bounds": {
            "brier_improvement_min": PASS_BRIER_IMPROVEMENT,
            "ece_max": PASS_ECE, "f1_min": PASS_F1,
            "precision_min": PASS_PRECISION, "alerts_max": PASS_ALERTS,
            "coverage_min": PASS_COVERAGE,
            "auc_roc_delta_max": GATE_RANKING_TOL,
            "auc_roc_delta_max_monotone": GATE_RANKING_TOL_MONO,
        },
        "note": ("PASS/CAUTION/FAIL computed mechanically from the "
                 "pre-registered specification Section 14 criteria; "
                 "research evidence only, no production decision"),
    }


# ---------------------------------------------------------------------------
# Assembler (IO-free, deterministic)
# ---------------------------------------------------------------------------
def run_phase17(cal: pd.DataFrame, test: pd.DataFrame, user_diag: dict,
                gates_env: dict, n_boot: int = N_BOOT) -> dict:
    """Assemble the full Phase 17 result (deterministic).

    gates_env: {"gate3_md5": {"passed": bool, "files": {...}},
                "gate4_baselines_match": bool,
                "determinism_ok": bool,
                "dataset_id": str, "environment": dict,
                "kaggle_note": str} -- computed by the runner.

    n_boot: bootstrap resamples (default 1000 per the pre-registered
    protocol; small values allowed only for tests).
    """
    # structural guards (frozen artifacts, TEST-once)
    if set(test.columns) != {"user", "day", "is_malicious", "score",
                             "alert_primary", "alert_secondary"}:
        raise AssertionError(f"test columns: {list(test.columns)}")
    if set(cal.columns) != {"user", "day", "is_malicious", "score"}:
        raise AssertionError(f"cal columns: {list(cal.columns)}")
    if len(test) != 54108 or test["is_malicious"].sum() != 476:
        raise AssertionError("TEST frame must be the frozen 54,108-row "
                             "artifact with 476 positives")
    if len(cal) != 54108 or cal["is_malicious"].sum() != 379:
        raise AssertionError("CAL frame must be the frozen 54,108-row "
                             "artifact with 379 positives")
    for frame in (cal, test):
        if frame.isnull().any().any():
            raise AssertionError("null values in frozen score frame")
        if frame.duplicated(["user", "day"]).any():
            raise AssertionError("duplicate (user, day) keys in score frame")

    cal = cal.sort_values(["user", "day"]).reset_index(drop=True)
    s_cal = cal["score"].to_numpy().astype(float)
    y_cal = cal["is_malicious"].to_numpy().astype(int)
    s_te = test["score"].to_numpy().astype(float)
    y_te = test["is_malicious"].to_numpy().astype(int)
    te_users = test["user"].to_numpy()

    # ------------------------------------------------------------------
    # gates
    # ------------------------------------------------------------------
    gate1 = gate1_test_reproduction(test)
    gate2 = gate2_cal_reproduction(cal)

    # fits (CAL only, fixed (user, day) order)
    platt_f = platt_fit(s_cal, y_cal)
    binning_f = binning_fit(s_cal, y_cal)
    ecdf_notes = {
        "method": "empirical CDF midpoint-rank mapping over CAL scores "
                  "(specification Section 9)",
        "n_cal_rows": int(len(s_cal)),
        "applied_range": [float(ecdf_apply(s_cal, s_te).min()),
                          float(ecdf_apply(s_cal, s_te).max())],
        "note": ("TEST scores outside the CAL range map marginally outside "
                 "[0, 1] by the pre-registered formula; no clamping "
                 "(specification Section 9 / line 388)"),
        "aborted": False, "abort_reason": None,
    }

    fits = {"B": {"name": "Platt scaling", "fit": platt_f,
                  "apply": lambda s: platt_apply(s, platt_f)},
            "C": {"name": "binning recalibration", "fit": binning_f,
                  "apply": lambda s: binning_apply(s, binning_f)},
            "D": {"name": "ECDF calibrated probability mapping",
                  "fit": ecdf_notes,
                  "apply": lambda s: ecdf_apply(s_cal, s)}}

    # per-arm transformed CAL/TEST scores
    p_cal_arms, p_te_arms = {}, {}
    for arm in ("B", "C", "D"):
        f = fits[arm]
        if f["fit"].get("aborted"):
            p_cal_arms[arm] = None
            p_te_arms[arm] = None
            continue
        p_cal_arms[arm] = f["apply"](s_cal)
        p_te_arms[arm] = f["apply"](s_te)

    gate1b = gate1b_operating_equality(
        y_cal, s_cal, s_te,
        {"B": {"p_cal": p_cal_arms["B"], "apply": fits["B"]["apply"]},
         "C": {"p_cal": p_cal_arms["C"], "apply": fits["C"]["apply"]},
         "D": {"p_cal": p_cal_arms["D"], "apply": fits["D"]["apply"]}})

    # ------------------------------------------------------------------
    # arms
    # ------------------------------------------------------------------
    arms = {}
    # arm A: frozen reference
    op_A = m.binary_decision_metrics(y_te, s_te, FROZEN_THRESHOLD)
    cov_A = coverage_metrics(test, s_te >= FROZEN_THRESHOLD, user_diag)
    te_ece_A = ece_score(y_te, s_te)
    arms["A"] = {
        "status": "reference",
        "transform": "none (frozen scores)",
        "operating_point": {"rule": "frozen production threshold applied "
                                    "as an immutable reference point",
                            "threshold": FROZEN_THRESHOLD},
        "operating_test": {
            **op_A,
            "alert_rate_per_calendar_day": float(
                op_A["n_alerts"] / test["day"].nunique()),
            "coverage": cov_A,
        },
        "test_calibration": {
            "brier": brier_score(y_te, s_te),
            "brier_improvement": 0.0,
            "ece": te_ece_A["ece"],
            "reliability": te_ece_A["reliability"],
        },
        "cal_calibration": calibration_metrics(y_cal, s_cal),
        "ranking": ranking_verification(y_te, s_te, "A"),
        "coverage_by_strata": coverage_by_strata(
            test, s_te >= FROZEN_THRESHOLD, user_diag),
        "scenario_recall": scenario_recall(
            test, s_te >= FROZEN_THRESHOLD, user_diag),
    }

    for arm in ("B", "C", "D"):
        f = fits[arm]
        if f["fit"].get("aborted"):
            arms[arm] = {"status": "aborted", "transform": f["name"],
                         "fit": f["fit"],
                         "note": f["fit"]["abort_reason"]}
            continue
        p_cal = p_cal_arms[arm]
        p_te = p_te_arms[arm]
        op_cal = best_f1_operating_point(y_cal, p_cal)
        t_arm = op_cal["threshold"]
        op_te = m.binary_decision_metrics(y_te, p_te, t_arm)
        alert_te = p_te >= t_arm
        cov = coverage_metrics(test, alert_te, user_diag)
        cal_met = calibration_metrics(y_cal, p_cal)
        te_met = calibration_metrics(y_te, p_te)
        brier_raw = brier_score(y_te, s_te)
        arms[arm] = {
            "status": "evaluated",
            "transform": f["name"],
            "fit": f["fit"],
            "operating_point": {**op_cal,
                                "raw_space_threshold": _raw_threshold_of(
                                    p_cal, s_cal, t_arm)},
            "operating_test": {
                **op_te,
                "alert_rate_per_calendar_day": float(
                    op_te["n_alerts"] / test["day"].nunique()),
                "coverage": cov,
            },
            "test_calibration": {
                "brier": te_met["brier"],
                "brier_improvement": (brier_raw - te_met["brier"])
                / brier_raw,
                "ece": te_met["ece"]["ece"],
                "reliability": te_met["ece"]["reliability"],
            },
            "cal_calibration": cal_met,
            "ranking": ranking_verification(y_te, p_te, arm),
            "coverage_by_strata": coverage_by_strata(test, alert_te,
                                                     user_diag),
            "scenario_recall": scenario_recall(test, alert_te, user_diag),
        }

    # ------------------------------------------------------------------
    # bootstrap (seed 42, n=1000)
    # ------------------------------------------------------------------
    boot = {"n_boot": n_boot, "seed": SEED, "alpha": ALPHA}
    boot["user_block_ranking"] = {}
    boot["user_block_operating"] = {}
    for arm, a in arms.items():
        if a["status"] == "aborted":
            boot["user_block_ranking"][arm] = None
            boot["user_block_operating"][arm] = None
            continue
        p_te = s_te if arm == "A" else p_te_arms[arm]
        boot["user_block_ranking"][arm] = {
            "auc_roc": user_block_bootstrap_ci(te_users, y_te, p_te,
                                               "auc_roc", n_boot=n_boot),
            "auc_pr": user_block_bootstrap_ci(te_users, y_te, p_te,
                                              "auc_pr", n_boot=n_boot),
        }
        threshold = (FROZEN_THRESHOLD if arm == "A"
                     else a["operating_point"]["threshold"])
        boot["user_block_operating"][arm] = user_block_bootstrap_operating(
            te_users, y_te, p_te, threshold, n_boot=n_boot)
    boot["calibration_transfer"] = {}
    for method, arm in (("platt", "B"), ("binning", "C"), ("ecdf", "D")):
        if arms[arm]["status"] == "aborted":
            boot["calibration_transfer"][arm] = None
            continue
        boot["calibration_transfer"][arm] = calibration_transfer_bootstrap(
            cal["user"].to_numpy(), s_cal, y_cal, s_te, y_te, method,
            n_boot=n_boot)

    # ------------------------------------------------------------------
    # tie-effect magnitude for the verdict (arm C, max |delta| over
    # AUC-PR and P@k/R@k)
    # ------------------------------------------------------------------
    tie_max = 0.0
    if arms["C"]["status"] == "evaluated":
        d = arms["C"]["ranking"]["deltas"]
        tie_max = max(abs(d["auc_pr"]),
                      *[abs(d[k]) for k in RECORDED_TOP_K])

    verdict = compute_verdict(
        arms,
        {"gate1": gate1, "gate1b": gate1b, "gate2": gate2,
         "gate3": {"passed": bool(gates_env["gate3_md5"]["passed"])},
         "gate4": {"passed": bool(
             gates_env["gate4_baselines_match"])}},
        determinism_ok=bool(gates_env["determinism_ok"]),
        tie_effect_max=tie_max)

    return {
        "experiment_id": "phase17-calibration-transfer",
        "phase": "17",
        "dataset": gates_env["dataset_id"],
        "scope": "evaluation-only calibration/generalization intervention; "
                 "no training, no threshold hunting, no production change",
        "specification": "docs/phase17_specification.md",
        "monotone_invariance": {
            "statement": ("strictly monotone score transforms preserve the "
                          "ranking; AUC-ROC/AUC-PR/P@k/R@k are unchanged for "
                          "arms B and D (mathematical identity, "
                          "specification Section 10.1); arm C preserves "
                          "order, so AUC-ROC is unchanged while AUC-PR/P@k "
                          "can move only through bin-level tie effects"),
            "verified": {"B": gate1b["arms"]["B"].get(
                             "alert_set_rows_identical_to_raw", False),
                         "D": gate1b["arms"]["D"].get(
                             "alert_set_rows_identical_to_raw", False)},
        },
        "gates": {"gate1": gate1, "gate1b": gate1b, "gate2": gate2,
                  "gate3": gates_env["gate3_md5"],
                  "gate4": {"baselines_match": bool(
                      gates_env["gate4_baselines_match"]),
                      "source": ("final_verification_report.json "
                                 "new_baselines (Phase 16 record, 152 "
                                 "files)"),
                      "passed": bool(gates_env["gate4_baselines_match"])}},
        "arms": arms,
        "bootstrap": boot,
        "tie_effect_max_arm_c": tie_max,
        "verdict": verdict,
        "expectations": {
            "zero_alert_users_persist": {
                "expected": ["JJM0203", "WDD0366"],
                "note": ("ranking failures are not fixable by calibration "
                         "(R15-2); if any arm alerts them, that is an "
                         "OBSERVED surprise (specification Section 12.5)"),
            },
            "b_d_operating_point_coincidence": (
                "arms B and D expected to produce identical alert sets "
                "(monotone invariance); verified in gate 1b"),
        },
        "frozen_system": {
            "model": "lgbm-graph-v1",
            "threshold": FROZEN_THRESHOLD,
            "seed": 42,
            "best_iteration_frozen": 186,
            "untouched": True,
        },
        "test_evaluation_policy": ("the authoritative chronological TEST is "
                                   "never read or scored; the user-disjoint "
                                   "TEST is read once from the frozen Phase "
                                   "14 predictions artifact; transforms are "
                                   "applied in memory only; the double-run "
                                   "gate is a determinism check, not a "
                                   "second evaluation"),
        "evidence_labels": {
            "gates": "OBSERVED (recomputed from md5-verified frozen "
                     "artifacts)",
            "arm metrics": "OBSERVED (computed once from the frozen score "
                           "artifacts under the pre-registered protocol)",
            "calibration transfer": "OBSERVED (CAL-fitted, TEST-applied)",
            "verdict": "pre-registered criteria applied mechanically",
        },
        "determinism": {"runs": 3, "bit_identical": None,
                        "note": ("filled by the runner after the "
                                 "byte-identical triple-run gate "
                                 "(specification Sections 14/16)")},
        "environment": gates_env["environment"],
        "kaggle_note": gates_env["kaggle_note"],
    }


def json_safe(obj: Any) -> Any:
    """JSON-safe conversion (numpy/date aware)."""
    return to_jsonable(obj)


def canonical(obj: Any) -> str:
    """Deterministic canonical JSON (sorted keys) for bit-comparison."""
    return canonical_json(obj)