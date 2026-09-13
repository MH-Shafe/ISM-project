"""Phase 9: production-style alert prioritization + threshold stability (CERT r4.2).

Central question: given the frozen `lgbm-graph-v1` model and its severe class
imbalance, how should predicted user-day risks be prioritized and converted
into alerts under realistic analyst capacity constraints?

Scope: analysis only -- the frozen LightGBM model is never retrained or
modified; no threshold adaptation is implemented. Every decision (candidate
policies, selection rule, stability screens) uses TRAIN/CALIBRATION evidence
only; TEST is evaluated exactly once after the policy is frozen.

Operational unit: user x day. "Daily capacity N" = on each calendar day,
alert the top-N user-days by model score that day (deterministic tie-break by
user id). Percentile policy p = alert the top ceil(p * rows) user-days
globally (deterministic tie-break by user id).

Candidate policies:
  - threshold: frozen max-F1 (0.9186015432508062, Phase 7 record) and
    CALIBRATION 50%-precision threshold (both evaluated as candidates;
    thresholds are NOT re-selected in this phase)
  - capacity: daily top-N for N in DAILY_CAPACITIES
  - percentile: global top-p for p in PERCENTILE_POLICIES

Selection rule (documented, deterministic, CALIBRATION-only):
  Screen S1 (threshold candidates only): bootstrap (n=200) CI width of the
    candidate threshold must be <= THRESHOLD_CI_WIDTH_CAP (the threshold must
    be identifiable on CALIBRATION).
  Screen S2 (all candidates): every 2-week CALIBRATION window's alert rate
    must lie within [WINDOW_RATE_LOW, WINDOW_RATE_HIGH] x the candidate's
    global CALIBRATION alert rate (alert volume must be temporally stable).
  Screen S3 (all candidates): every window must capture >= 1 malicious
    user-day (no window with zero detection).
  Among eligible candidates: pick the highest CALIBRATION F1; tie-break by
  fewer alerts, then by candidate id (deterministic).
  If no candidate is eligible: NO PRODUCTION POLICY SELECTED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Declared policy candidates and rule constants (a priori, documented)
# ---------------------------------------------------------------------------
DAILY_CAPACITIES = [5, 10, 20, 30, 50, 100]
PERCENTILE_POLICIES = [0.0005, 0.001, 0.0025, 0.005, 0.01]

FROZEN_THRESHOLD = 0.9186015432508062  # Phase 7 freeze record
WINDOWS_N = 4                          # chronological CALIBRATION windows
N_BOOT_THRESHOLD = 200
BOOT_SEED = 42

THRESHOLD_CI_WIDTH_CAP = 0.2
WINDOW_RATE_LOW = 0.5
WINDOW_RATE_HIGH = 2.0

HIGH_RISK_THRESHOLD = FROZEN_THRESHOLD  # user-diagnostic definition

CONCENTRATION_SHARES = [0.01, 0.05, 0.10]


# ---------------------------------------------------------------------------
# Policy masks (deterministic; original row order preserved)
# ---------------------------------------------------------------------------
def daily_top_k_mask(keys, scores, k: int) -> np.ndarray:
    """Alert the top-k scored rows of each calendar day (tie-break: user id).

    A day with fewer than k rows alerts all its rows. Deterministic.
    """
    df = pd.DataFrame({
        "idx": np.arange(len(scores)),
        "day": keys["day"].to_numpy(),
        "user": keys["user"].to_numpy(),
        "score": np.asarray(scores, dtype=float),
    })
    df = df.sort_values(["day", "score", "user"],
                        ascending=[True, False, True], kind="mergesort")
    df["rank"] = df.groupby("day").cumcount()
    mask = np.zeros(len(scores), dtype=bool)
    mask[df.loc[df["rank"] < k, "idx"].to_numpy()] = True
    return mask


def percentile_mask(keys, scores, p: float) -> np.ndarray:
    """Alert the top ceil(p * rows) user-days globally (tie-break: user id)."""
    n = len(scores)
    k = int(np.ceil(p * n))
    df = pd.DataFrame({
        "idx": np.arange(n),
        "user": keys["user"].to_numpy(),
        "score": np.asarray(scores, dtype=float),
    })
    df = df.sort_values(["score", "user"], ascending=[False, True],
                        kind="mergesort")
    top = df.head(k)["idx"].to_numpy()
    mask = np.zeros(n, dtype=bool)
    mask[top] = True
    return mask


def threshold_mask(scores, t: float) -> np.ndarray:
    """Alert every user-day with score >= t (frozen threshold policy)."""
    return np.asarray(scores, dtype=float) >= t


def policy_mask(keys, scores, candidate: dict) -> np.ndarray:
    """Dispatch a candidate spec {'kind','param'} to its alert mask."""
    kind = candidate["kind"]
    if kind == "threshold":
        return threshold_mask(scores, candidate["param"])
    if kind == "capacity":
        return daily_top_k_mask(keys, scores, candidate["param"])
    if kind == "percentile":
        return percentile_mask(keys, scores, candidate["param"])
    raise ValueError(f"unknown policy kind: {kind}")


# ---------------------------------------------------------------------------
# Metrics from an alert mask (capacity policies have no single threshold)
# ---------------------------------------------------------------------------
def metrics_from_mask(y, mask) -> dict:
    """Full decision-metric set for an alert mask (same family as
    m.binary_decision_metrics; malicious-user-day coverage == recall)."""
    y = np.asarray(y, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    tp = int((mask & (y == 1)).sum())
    fp = int((mask & (y == 0)).sum())
    fn = int((~mask & (y == 1)).sum())
    tn = int((~mask & (y == 0)).sum())
    n = max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "n_alerts": int(mask.sum()),
        "alert_rate": int(mask.sum()) / n,
        "precision": precision,
        "recall": recall,
        "coverage": recall,  # malicious-user-day coverage
        "f1": f1,
        "mcc": float(matthews_corrcoef(y, mask)),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "fpr": fp / max(1, fp + tn),
        "fnr": fn / max(1, fn + tp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def scenario_recall_from_mask(y, mask, scenario_of_user, keys) -> dict:
    """Per-CERT-scenario recall for a mask (same semantics as
    m.per_scenario_recall, but mask-based)."""
    y = np.asarray(y, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    users = np.asarray(keys["user"])
    out = {}
    for scenario in sorted({v for v in scenario_of_user.values()}):
        m = np.array([scenario_of_user.get(u, -1) == scenario for u in users])
        n_pos = int((m & (y == 1)).sum())
        if n_pos == 0:
            continue
        out[str(scenario)] = {
            "n_malicious_rows": n_pos,
            "n_rows": int(m.sum()),
            "recall": float((m & mask & (y == 1)).sum()) / n_pos,
        }
    return out


# ---------------------------------------------------------------------------
# Candidate construction + evaluation
# ---------------------------------------------------------------------------
def build_candidates(frozen_threshold: float = FROZEN_THRESHOLD,
                     t_p50: float | None = None) -> list[dict]:
    """The full candidate set (2 thresholds + 6 capacities + 5 percentiles)."""
    cands = [
        {"id": "frozen_max_f1", "kind": "threshold", "param": frozen_threshold,
         "name": "frozen max-F1 threshold (Phase 7)"},
    ]
    if t_p50 is not None:
        cands.append({"id": "cal_50p_precision", "kind": "threshold",
                      "param": t_p50, "name": "CALIBRATION 50%-precision threshold"})
    for n in DAILY_CAPACITIES:
        cands.append({"id": f"daily_top_{n}", "kind": "capacity", "param": n,
                      "name": f"top-{n} alerts per day"})
    for p in PERCENTILE_POLICIES:
        cands.append({"id": f"pct_{p:.4f}", "kind": "percentile", "param": p,
                      "name": f"global top-{100 * p:.2f}%"})
    return cands


def evaluate_candidates(keys, y, scores, candidates,
                        scenario_of_user=None) -> dict[str, dict]:
    """Evaluate every candidate on one block; returns {candidate_id: metrics}."""
    scenario_of_user = scenario_of_user or {}
    out = {}
    for cand in candidates:
        mask = policy_mask(keys, scores, cand)
        res = metrics_from_mask(y, mask)
        res["scenario_recall"] = scenario_recall_from_mask(
            y, mask, scenario_of_user, keys)
        res["kind"] = cand["kind"]
        res["param"] = cand["param"]
        res["name"] = cand["name"]
        out[cand["id"]] = res
    return out


# ---------------------------------------------------------------------------
# Chronological windows (CALIBRATION only)
# ---------------------------------------------------------------------------
def partition_windows(keys, n_windows: int = WINDOWS_N) -> list[np.ndarray]:
    """Split rows into n_windows contiguous chronological day-blocks.

    Returns boolean masks over rows: disjoint, chronological, union = all.
    """
    days = np.sort(pd.unique(keys["day"]).astype("datetime64[ns]"))
    edges = np.array_split(np.arange(len(days)), n_windows)
    day_of = keys["day"].to_numpy().astype("datetime64[ns]")
    masks = []
    for idx in edges:
        masks.append(np.isin(day_of, days[idx]))
    return masks


def window_bounds(keys, masks) -> list[dict]:
    """First..last day of each window (for reporting)."""
    days = keys["day"].to_numpy()
    bounds = []
    for m in masks:
        w = np.sort(np.unique(days[m]))
        bounds.append({"days": int(m.sum() and len(w)),
                       "first_day": pd.Timestamp(w[0]).strftime("%Y-%m-%d"),
                       "last_day": pd.Timestamp(w[-1]).strftime("%Y-%m-%d")})
    return bounds


# ---------------------------------------------------------------------------
# Threshold stability (CALIBRATION bootstrap, vectorized)
# ---------------------------------------------------------------------------
def _bootstrap_objective(y, s, objective: str, n_boot: int, seed: int,
                         min_precision: float = 0.5) -> np.ndarray:
    """Vectorized bootstrap of a threshold objective over resampled
    (score, label) pairs -- the pairings are preserved within each sample.

    objective "max_f1": mirrors th.best_f1_threshold semantics per sample
    (thresholds at unique scores, pred = score >= t, F1 ties broken toward
    the higher threshold). objective "min_precision": mirrors
    th.threshold_at_precision (first rank with cumulative precision >=
    min_precision; fallback: score of the single highest-ranked positive).
    """
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ss = s[idx]                                  # (n_boot, n) score pairs
    yy = y[idx].astype(np.int64)
    order = np.argsort(-ss, axis=1)              # per-sample descending scores
    ys = np.take_along_axis(yy, order, axis=1)
    cum_tp = np.cumsum(ys, axis=1)               # (n_boot, n)
    ranks = np.arange(1, n + 1)
    if objective == "max_f1":
        total = cum_tp[:, -1:]
        precision = cum_tp / np.maximum(1, ranks)
        recall = cum_tp / np.maximum(1, total)
        f1 = 2 * precision * recall / np.maximum(1e-9, precision + recall)
        # within a score-tie block F1 peaks at the block's last rank, so the
        # rank sweep covers every unique threshold; ties across thresholds are
        # broken toward the highest one (smallest rank), mirroring
        # best_f1_threshold's strict-greater update with 1e-12 tolerance.
        best_rank = np.argmax(f1 >= f1.max(axis=1, keepdims=True) - 1e-12,
                              axis=1)
        return ss[np.arange(n_boot), order[np.arange(n_boot), best_rank]]
    if objective == "min_precision":
        prec_rank = cum_tp / ranks
        ok = np.where(cum_tp[:, -1:] > 0,
                      prec_rank >= min_precision, False)
        any_ok = ok.any(axis=1)
        hit = np.argmax(ok, axis=1)              # first rank meeting precision
        first_pos = np.argmax(ys > 0, axis=1)    # highest-ranked positive
        rank_idx = np.where(any_ok, hit, first_pos)
        return ss[np.arange(n_boot), order[np.arange(n_boot), rank_idx]]
    raise ValueError(f"unknown objective: {objective}")


def threshold_bootstrap(y, s, objective: str = "max_f1",
                        n_boot: int = N_BOOT_THRESHOLD,
                        seed: int = BOOT_SEED) -> dict:
    """Bootstrap distribution of a CALIBRATION-selected threshold + resulting
    CALIBRATION alert rate / recall / precision at each bootstrapped value."""
    ts = _bootstrap_objective(y, s, objective, n_boot, seed)
    rates = np.zeros(n_boot)
    recs = np.zeros(n_boot)
    precs = np.zeros(n_boot)
    yf = np.asarray(y, dtype=int)
    for i, t in enumerate(ts):
        mm = metrics_from_mask(yf, threshold_mask(s, t))
        rates[i] = mm["alert_rate"]
        recs[i] = mm["recall"]
        precs[i] = mm["precision"]
    q = lambda a: float(np.percentile(ts, a))
    return {
        "objective": objective,
        "n_boot": n_boot,
        "mean": float(np.mean(ts)),
        "median": float(np.median(ts)),
        "std": float(np.std(ts, ddof=1)),
        "p10": q(10), "p90": q(90),
        "ci_low": q(2.5), "ci_high": q(97.5),
        "ci_width": q(97.5) - q(2.5),
        "min": float(np.min(ts)), "max": float(np.max(ts)),
        "alert_rate": {"mean": float(np.mean(rates)),
                       "p10": float(np.percentile(rates, 10)),
                       "p90": float(np.percentile(rates, 90))},
        "recall": {"mean": float(np.mean(recs)),
                   "p10": float(np.percentile(recs, 10)),
                   "p90": float(np.percentile(recs, 90))},
        "precision": {"mean": float(np.mean(precs)),
                      "p10": float(np.percentile(precs, 10)),
                      "p90": float(np.percentile(precs, 90))},
    }


# ---------------------------------------------------------------------------
# Temporal / concentration / user-level diagnostics
# ---------------------------------------------------------------------------
def temporal_stability(keys, y, scores, candidates, masks_windows) -> dict:
    """Per-window alert volume/rate/precision/recall + score distribution."""
    rows_all = np.arange(len(scores))
    out = {"windows": [], "per_policy": {}}
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    for wi, wm in enumerate(masks_windows):
        w_s = s_arr[wm]
        out["windows"].append({
            "window": wi,
            "rows": int(wm.sum()),
            "positives": int(y_arr[wm].sum()),
            "score": {"min": float(w_s.min()), "p10": float(np.percentile(w_s, 10)),
                      "median": float(np.median(w_s)),
                      "p90": float(np.percentile(w_s, 90)), "max": float(w_s.max()),
                      "mean": float(w_s.mean())},
        })
    for cand in candidates:
        mask = policy_mask(keys, scores, cand)
        per_w = []
        for wm in masks_windows:
            mm = metrics_from_mask(y_arr[wm], mask[wm])
            per_w.append({"alert_count": mm["n_alerts"],
                          "alert_rate": mm["alert_rate"],
                          "precision": mm["precision"], "recall": mm["recall"]})
        out["per_policy"][cand["id"]] = per_w
    return out


def user_concentration(keys, mask) -> dict:
    """Alert concentration among alerted users (top-1/5/10% share of alerts)."""
    mask = np.asarray(mask, dtype=bool)
    users = np.asarray(keys["user"])[mask]
    counts = pd.Series(users).value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    n_users = len(counts)
    if total == 0:
        return {"n_alerts": 0, "n_alerted_users": 0, "alerts_per_user": None,
                "shares": None, "max_alerts_per_user": 0}
    shares = {}
    for q in CONCENTRATION_SHARES:
        k = max(1, int(np.ceil(q * n_users)))
        shares[f"top_{q:.2f}"] = float(counts.head(k).sum() / total)
    return {
        "n_alerts": total,
        "n_alerted_users": n_users,
        "alerts_per_user": {"mean": float(counts.mean()),
                            "median": float(counts.median()),
                            "max": float(counts.max())},
        "shares": shares,
        "max_alerts_per_user": int(counts.max()),
        "alerted_user_gini": _gini(counts.to_numpy(dtype=float)),
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / x.sum()) / n)


def user_diagnostics(keys, y, scores, mask_selected,
                     high_risk_threshold: float = HIGH_RISK_THRESHOLD) -> dict:
    """CALIBRATION-only user-level aggregation diagnostic.

    Per user: n days, max/mean risk, mean of top-3 daily risks, n high-risk
    days (score >= high_risk_threshold), n alerts under the selected policy.
    User-level labels: any malicious user-day in the block. Diagnostics only;
    the user x day unit remains the analytical unit and nothing here enters
    the model or the policy selection.
    """
    keys = keys.reset_index(drop=True)
    df = pd.DataFrame({
        "user": keys["user"].to_numpy(),
        "score": np.asarray(scores, dtype=float),
        "y": np.asarray(y, dtype=int),
        "alert": np.asarray(mask_selected, dtype=bool),
    })
    grp = df.groupby("user")
    agg = pd.DataFrame({
        "n_days": grp.size(),
        "max_risk": grp["score"].max(),
        "mean_risk": grp["score"].mean(),
        "top3_risk": grp["score"].apply(
            lambda s: float(np.sort(s.to_numpy())[-3:].mean())),
        "n_high_risk_days": grp["score"].apply(
            lambda s: int((np.asarray(s) >= high_risk_threshold).sum())),
        "n_alerts": grp["alert"].sum(),
        "malicious_user": grp["y"].max(),
    }).reset_index()

    n_mal = int(agg["malicious_user"].sum())
    auc = {}
    for stat in ("max_risk", "mean_risk", "top3_risk", "n_high_risk_days", "n_alerts"):
        s = agg[stat].to_numpy(dtype=float)
        if n_mal == 0 or n_mal == len(agg):
            auc[stat] = None
        else:
            auc[stat] = float(roc_auc_score(agg["malicious_user"].to_numpy(), s))

    top_by_max = agg.sort_values("max_risk", ascending=False).head(5)
    return {
        "n_users": int(len(agg)),
        "n_malicious_users": n_mal,
        "user_level_auc": auc,
        "precision_at_5_users_by_max_risk": float(
            top_by_max["malicious_user"].mean()),
        "top5_users": [{"user": str(u), "max_risk": float(r),
                        "malicious": bool(m)}
                       for u, r, m in top_by_max[["user", "max_risk", "malicious_user"]]
                       .itertuples(index=False, name=None)],
        "high_risk_threshold": high_risk_threshold,
        "note": ("diagnostic only; user-day unit unchanged; period-end "
                 "summaries over the CALIBRATION block, never model inputs"),
    }


# ---------------------------------------------------------------------------
# Policy selection (documented deterministic rule, CALIBRATION only)
# ---------------------------------------------------------------------------
def select_policy(cal_results: dict, threshold_boots: dict,
                  window_stats: dict) -> dict:
    """Apply screens S1/S2/S3 and select by CALIBRATION F1 (see module doc).

    Returns the decision bundle: selected id (or 'NO PRODUCTION POLICY
    SELECTED'), eligibility reasons per candidate, and the chosen metrics.
    """
    reasons = {}
    eligible = []
    for cid, res in cal_results.items():
        notes = []
        if res["kind"] == "threshold":
            width = threshold_boots[cid]["ci_width"]
            if width > THRESHOLD_CI_WIDTH_CAP:
                notes.append(f"S1 fail: bootstrap CI width {width:.4f} > {THRESHOLD_CI_WIDTH_CAP}")
            else:
                notes.append(f"S1 pass: CI width {width:.4f}")
        if res["alert_rate"] <= 0:
            notes.append("S2 fail: zero alert rate")
        else:
            bad = [w for w in window_stats[cid]
                   if not (WINDOW_RATE_LOW * res["alert_rate"]
                           <= w["alert_rate"]
                           <= WINDOW_RATE_HIGH * res["alert_rate"])]
            notes.append(f"S2 {'pass' if not bad else 'fail'}: "
                         f"{len(bad)}/4 windows outside [{WINDOW_RATE_LOW}, "
                         f"{WINDOW_RATE_HIGH}]x global rate")
        zero = [w for w in window_stats[cid] if w["recall"] == 0]
        notes.append(f"S3 {'pass' if not zero else 'fail'}: "
                     f"{len(zero)}/4 windows with zero recall")
        reasons[cid] = notes
        failed = any(n.startswith(("S1 fail", "S2 fail", "S3 fail")) for n in notes)
        if not failed:
            eligible.append(cid)

    if not eligible:
        return {
            "selection": "NO PRODUCTION POLICY SELECTED",
            "rule": ("screens S1 (threshold CI width), S2 (window alert-rate "
                     "ratio), S3 (window recall > 0); best eligible CALIBRATION "
                     "F1 with fewer-alerts tie-break"),
            "eligible": [], "reasons": reasons,
        }
    best = max(eligible, key=lambda c: (cal_results[c]["f1"],
                                        -cal_results[c]["alert_rate"], c))
    return {
        "selection": best,
        "rule": ("screens S1 (threshold CI width), S2 (window alert-rate "
                 "ratio), S3 (window recall > 0); best eligible CALIBRATION "
                 "F1 with fewer-alerts tie-break"),
        "eligible": eligible,
        "reasons": reasons,
        "selected_calibration_metrics": {k: cal_results[best][k]
                                         for k in ("alert_rate", "precision",
                                                   "recall", "f1", "mcc",
                                                   "coverage")},
    }