"""Phase 12: operational envelope, alert capacity, de-duplication,
rolling threshold stability (CERT r4.2).

Evaluation/operational-policy study ONLY. The frozen model
(lgbm-graph-v1), the frozen Phase 9 alert policy (threshold
0.9186015432508062), and all Phase 7-11 artifacts are never modified.

Protocol:
- ALL policy decisions and robustness analysis happen on CALIBRATION.
- TEST is evaluated exactly once, after the operational decision is frozen.
- No TEST-derived threshold, window, policy, or parameter is ever used.

Every policy is deterministic and hand-checkable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.evaluation import threshold as th
from src.experiments import phase9 as p9

FROZEN_THRESHOLD = p9.FROZEN_THRESHOLD  # 0.9186015432508062 (Phase 9 policy)
DEDUP_WINDOWS = (1, 3, 7)               # B1/B2/B3 de-duplication windows (days)
ROLLING_WINDOWS = (7, 14, 30)           # chronological CALIBRATION windows (days)
WINDOW_CHUNK_DAYS = 7                   # zero-alert window granularity
CAL_START = pd.Timestamp("2011-02-01")
CAL_END = pd.Timestamp("2011-03-31")


# ---------------------------------------------------------------------------
# Alert masks (deterministic)
# ---------------------------------------------------------------------------
def raw_mask(df: pd.DataFrame, threshold: float = FROZEN_THRESHOLD) -> np.ndarray:
    """B0 / P0 mask: score >= threshold (no de-duplication)."""
    return df["score"].to_numpy() >= threshold


def dedup_mask(df: pd.DataFrame, window_days: int = 1,
               mask: np.ndarray | None = None) -> np.ndarray:
    """Per-user sequential de-duplication mask (deterministic).

    For each user, days are processed in ascending order. An alert is
    emitted on day d iff the crossing condition holds (score >=
    FROZEN_THRESHOLD by default, or the supplied mask) AND no alert was
    emitted in the previous (window_days - 1) days
    (d - last_emitted >= window_days). window_days=1 suppresses only
    same-day duplicates, which cannot exist in a user x day table
    (B1 == B0 exactly).
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    crossing = raw_mask(df) if mask is None else np.asarray(mask, dtype=bool)
    n = len(df)
    out = np.zeros(n, dtype=bool)
    if not crossing.any():
        return out
    users = df["user"].to_numpy()
    days = pd.to_datetime(df["day"]).values.astype("datetime64[s]").astype(np.int64)
    for u in pd.unique(users):
        idx = np.flatnonzero(users == u)
        idx = idx[np.argsort(days[idx], kind="stable")]
        last_emit = None
        for i in idx:
            if not crossing[i]:
                continue
            if last_emit is None or days[i] - last_emit >= window_days * 86400:
                out[i] = True
                last_emit = days[i]
    return out


# ---------------------------------------------------------------------------
# Policy metrics (arbitrary deterministic alert mask)
# ---------------------------------------------------------------------------
def gini_of_counts(counts: np.ndarray) -> float:
    """Gini coefficient over a per-user alert-count distribution (0 = equal)."""
    c = np.asarray(counts, dtype=float)
    if c.sum() <= 0:
        return 0.0
    c = np.sort(c)
    n = len(c)
    return float((2 * np.sum(np.arange(1, n + 1) * c) / (n * c.sum()))
                 - (n + 1) / n)


def zero_alert_chunks(df: pd.DataFrame, mask: np.ndarray,
                      chunk_days: int = WINDOW_CHUNK_DAYS) -> int:
    """Number of chronological chunk_days windows with zero alerts.

    Chunks are consecutive non-overlapping blocks of the sorted unique
    days in df (the last chunk may be shorter; it still counts).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(df["day"])).normalize()
    per_day = pd.Series(np.asarray(mask, dtype=int)).groupby(idx).sum()
    days = np.sort(idx.unique())
    n_zero = 0
    for i in range(0, len(days), chunk_days):
        chunk = days[i:i + chunk_days]
        if int(per_day.reindex(chunk, fill_value=0).sum()) == 0:
            n_zero += 1
    return int(n_zero)


def policy_metrics(df: pd.DataFrame, mask: np.ndarray,
                   scenario_of_user: dict | None = None) -> dict:
    """Full metric set for an arbitrary alert mask (evaluation only)."""
    y = df["is_malicious"].to_numpy().astype(int)
    pred = np.asarray(mask, dtype=bool)
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    n = len(df)
    alerts = int(pred.sum())
    alerted_users = int(df.loc[pred, "user"].nunique())
    counts = df.loc[pred].groupby("user").size().reindex(
        df["user"].unique(), fill_value=0).to_numpy()
    out = {
        "n_rows": n,
        "n_positives": int(y.sum()),
        "n_alerts": alerts,
        "alert_rate": alerts / max(1, n),
        "n_alerted_users": alerted_users,
        "alerts_per_user_mean_all": alerts / max(1, df["user"].nunique()),
        "alerts_per_user_mean_alerted": alerts / max(1, alerted_users),
        "alerts_per_user_max": int(counts.max()) if alerts else 0,
        "gini": gini_of_counts(counts),
        "zero_alert_windows_7d": zero_alert_chunks(df, pred),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1e-9, 2 * tp + fp + fn),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "fpr": fp / max(1, fp + tn),
        "fnr": fn / max(1, fn + tp),
    }
    if scenario_of_user is not None:
        users = df["user"].to_numpy()
        sc = {}
        for scenario in sorted({v for v in scenario_of_user.values()}):
            mask_s = np.array([scenario_of_user.get(u, -1) == scenario for u in users])
            n_pos = int((mask_s & (y == 1)).sum())
            if n_pos == 0:
                continue
            sc[str(scenario)] = {
                "n_malicious_rows": n_pos,
                "recall": float((mask_s & pred & (y == 1)).sum()) / n_pos,
            }
        out["scenario_recall"] = sc
    return out


# ---------------------------------------------------------------------------
# Experiment B: de-duplication (CALIBRATION only)
# ---------------------------------------------------------------------------
def dedup_experiment(cal: pd.DataFrame,
                     scenario_of_user: dict | None = None) -> dict:
    """B0 (none) / B1 (1 per day) / B2 (1 per 3 days) / B3 (1 per 7 days)."""
    assert_calibration_only(cal)
    raw_crossings = int(raw_mask(cal).sum())
    rows = {"B0": {"dedup_window_days": 0,
                   **policy_metrics(cal, raw_mask(cal), scenario_of_user)}}
    for key, w in (("B1", 1), ("B2", 3), ("B3", 7)):
        mask = raw_mask(cal) if w == 1 else dedup_mask(cal, window_days=w)
        pm = policy_metrics(cal, mask, scenario_of_user)
        rows[key] = {
            "dedup_window_days": w,
            **pm,
            "alerts_suppressed": raw_crossings - pm["n_alerts"],
        }
    return {
        "scope": "CALIBRATION only (2011-02-01..2011-03-31); policy selection never on TEST",
        "raw_threshold_crossings": raw_crossings,
        "policies": rows,
        "note": ("B1 (1 alert/user/day) is exactly B0: a user x day table has at "
                 "most one row per user-day, so same-day duplicates cannot exist."),
    }


def select_dedup_window(cal: pd.DataFrame,
                        scenario_of_user: dict | None = None) -> tuple[int, dict]:
    """Deterministic CALIBRATION-only de-duplication selection for P4.

    Same rule family as the operational selection (R1 recall retention
    >= 0.90, R2 alert reduction >= 10%, R3 Gini +0.05, R4 zero-alert
    windows), applied to B1/B2/B3 vs B0. Returns the chosen window in
    days (0 = no de-duplication) and the experiment record.
    """
    assert_calibration_only(cal)
    res = dedup_experiment(cal, scenario_of_user)
    b0 = res["policies"]["B0"]
    eligible = []
    for key in ("B1", "B2", "B3"):
        p = res["policies"][key]
        ok = (p["recall"] >= 0.90 * b0["recall"]
              and p["n_alerts"] <= 0.90 * b0["n_alerts"]
              and p["gini"] <= b0["gini"] + 0.05
              and p["zero_alert_windows_7d"] <= b0["zero_alert_windows_7d"])
        if ok:
            eligible.append(key)
    if not eligible:
        return 0, res
    best = max(eligible, key=lambda k: (res["policies"][k]["recall"],
                                        -res["policies"][k]["n_alerts"],
                                        res["policies"][k]["f1"]))
    return {"B1": 1, "B2": 3, "B3": 7}[best], res


# ---------------------------------------------------------------------------
# Experiment C: rolling threshold stability (CALIBRATION only)
# ---------------------------------------------------------------------------
def cal_windows(cal: pd.DataFrame, window_days: int) -> list[dict]:
    """Consecutive non-overlapping chronological windows over CALIBRATION.

    Windows are computed from the sorted unique days in the table; the
    last window may be shorter (remainder days are recorded).
    """
    days = sorted(pd.DatetimeIndex(pd.to_datetime(cal["day"])).normalize().unique())
    windows = []
    for i in range(0, len(days), window_days):
        chunk = days[i:i + window_days]
        windows.append({
            "index": len(windows),
            "start": str(chunk[0].date()),
            "end": str(chunk[-1].date()),
            "n_days": len(chunk),
            "df": cal[pd.to_datetime(cal["day"]).isin(chunk)].copy(),
        })
    return windows


def _window_metrics(w: dict, threshold: float) -> dict:
    df = w["df"]
    mask = df["score"].to_numpy() >= threshold
    pm = policy_metrics(df, mask)
    scores = df["score"].to_numpy()
    return {
        "n_days": w["n_days"],
        "n_rows": pm["n_rows"],
        "n_positives": pm["n_positives"],
        "n_alerts": pm["n_alerts"],
        "alert_rate": pm["alert_rate"],
        "precision": pm["precision"],
        "recall": pm["recall"],
        "f1": pm["f1"],
        "mcc": pm["mcc"],
        "zero_alert_window": pm["n_alerts"] == 0,
        "zero_recall_window": pm["n_positives"] > 0 and pm["recall"] == 0,
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_median": float(np.median(scores)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
    }


def rolling_threshold_analysis(cal: pd.DataFrame) -> dict:
    """Static frozen threshold + rolling calibration threshold per window.

    Rolling: window i uses threshold = max-F1 on the union of windows
    strictly before i (no future observations). A window whose reference
    contains no positive has an undefined rolling threshold, recorded as
    such (frozen threshold fallback, flagged).
    """
    assert_calibration_only(cal)
    out = {}
    for wd in ROLLING_WINDOWS:
        windows = cal_windows(cal, wd)
        static, rolling, ref_dfs = [], [], []
        for i, w in enumerate(windows):
            static.append({"window": i, **_window_metrics(w, FROZEN_THRESHOLD)})
            ref = pd.concat(ref_dfs, ignore_index=True) if ref_dfs else None
            if ref is not None and ref["is_malicious"].sum() >= 1:
                t = th.best_f1_threshold(ref["is_malicious"].to_numpy(),
                                         ref["score"].to_numpy())[0]
                defined = True
            else:
                t, defined = None, False
            wm = _window_metrics(w, t if defined else FROZEN_THRESHOLD)
            rolling.append({
                "window": i,
                "threshold": t,
                "rolling_defined": defined,
                "fallback_to_frozen": not defined,
                "ref_windows": [j["index"] for j in windows[:i]],
                "ref_positives": int(ref["is_malicious"].sum()) if ref is not None else 0,
                **{k: wm[k] for k in ("n_alerts", "n_positives", "alert_rate",
                                      "precision", "recall", "f1", "mcc",
                                      "zero_alert_window", "zero_recall_window")},
            })
            ref_dfs.append(w["df"])
        out[str(wd)] = {
            "window_days": wd,
            "n_windows": len(windows),
            "windows_static": static,
            "windows_rolling": rolling,
        }
    return {
        "scope": ("CALIBRATION only; rolling references strictly before each "
                  "window; no future information"),
        "frozen_threshold": FROZEN_THRESHOLD,
        "by_window_size": out,
    }


def rolling_apply(df: pd.DataFrame, window_days: int) -> np.ndarray:
    """Alert mask of the rolling calibration policy applied chronologically.

    Each window uses the max-F1 threshold of the union of strictly
    earlier windows (no future observations); the first window has no
    reference and falls back to the frozen threshold (flagged in the
    analysis artifacts). Deterministic and reorder-safe (windows follow
    the sorted unique days of df).
    """
    masks = np.zeros(len(df), dtype=bool)
    ref_dfs = []
    for w in cal_windows(df, window_days):
        win = w["df"]
        ref = pd.concat(ref_dfs, ignore_index=True) if ref_dfs else None
        if ref is not None and ref["is_malicious"].sum() >= 1:
            t = th.best_f1_threshold(ref["is_malicious"].to_numpy(),
                                     ref["score"].to_numpy())[0]
        else:
            t = FROZEN_THRESHOLD
        in_win = df.index.isin(win.index)
        masks[in_win] = win["score"].to_numpy() >= t
        ref_dfs.append(win)
    return masks


def assert_calibration_only(df: pd.DataFrame) -> None:
    """TEST-use guard: policy decisions require a CALIBRATION-only table."""
    days = pd.DatetimeIndex(pd.to_datetime(df["day"])).normalize()
    if (days > CAL_END).any() or (days < CAL_START).any():
        raise ValueError(
            "policy decisions require CALIBRATION-only rows "
            "(2011-02-01..2011-03-31); found days outside it")


def _aggregate_policy(rows: list[dict]) -> dict:
    """Aggregate CALIBRATION metrics over the rolling-defined windows only."""
    defined = [r for r in rows if r["rolling_defined"]]
    if not defined:
        return {"n_defined_windows": 0, "recall": 0.0, "f1": 0.0,
                "n_alerts": 0, "n_positives": 0, "defined": False}
    tp = sum(r["recall"] * r["n_positives"] for r in defined)
    n_pos = sum(r["n_positives"] for r in defined)
    alerts = sum(r["n_alerts"] for r in defined)
    f1 = sum(r["f1"] * r["n_alerts"] for r in defined) / max(1, alerts)
    return {
        "n_defined_windows": len(defined),
        "defined": True,
        "recall": tp / max(1, n_pos),
        "n_positives": n_pos,
        "n_alerts": alerts,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Experiment D: combined policies + deterministic selection rule
# ---------------------------------------------------------------------------
def combined_policies(cal: pd.DataFrame,
                      scenario_of_user: dict | None = None,
                      rolling: dict | None = None) -> dict:
    """P0..P4 combined policies on CALIBRATION.

    P0 = frozen threshold, no de-duplication (Phase 9 policy).
    P1/P2/P3 = frozen threshold + 1/3/7-day de-duplication.
    P4 = calibration-only rolling threshold (window size chosen by the
    documented CAL rule) + the de-duplication window selected on CAL.

    rolling: optional precomputed rolling_threshold_analysis(cal) result
    to reuse (its per-window thresholds are exactly what rolling_apply
    would compute; avoids duplicate expensive fits).
    """
    assert_calibration_only(cal)
    rows = {}
    rows["P0"] = {"frozen_threshold": FROZEN_THRESHOLD, "dedup_window_days": 0,
                  **policy_metrics(cal, raw_mask(cal), scenario_of_user)}
    rows["P1"] = {"frozen_threshold": FROZEN_THRESHOLD, "dedup_window_days": 1,
                  **rows["P0"]}
    for key, w in (("P2", 3), ("P3", 7)):
        mask = dedup_mask(cal, window_days=w)
        rows[key] = {"frozen_threshold": FROZEN_THRESHOLD, "dedup_window_days": w,
                     **policy_metrics(cal, mask, scenario_of_user)}

    if rolling is None:
        rolling = rolling_threshold_analysis(cal)
    agg = {str(wd): _aggregate_policy(
        rolling["by_window_size"][str(wd)]["windows_rolling"])
        for wd in ROLLING_WINDOWS}
    best_wd = max(ROLLING_WINDOWS, key=lambda w: (agg[str(w)]["f1"],
                                                  agg[str(w)]["recall"]))

    dedup_w, dedup_res = select_dedup_window(cal, scenario_of_user)
    p4_mask = np.zeros(len(cal), dtype=bool)
    for w in rolling["by_window_size"][str(best_wd)]["windows_rolling"]:
        t = w["threshold"] if w["rolling_defined"] else FROZEN_THRESHOLD
        win = cal_windows(cal, best_wd)[w["window"]]["df"]
        in_win = cal.index.isin(win.index)
        p4_mask[in_win] = win["score"].to_numpy() >= t
    if dedup_w > 0:
        p4_mask = dedup_mask(cal, window_days=dedup_w, mask=p4_mask)
    rows["P4"] = {
        "rolling_window_days": best_wd,
        "rolling_aggregate": agg[str(best_wd)],
        "dedup_window_days": dedup_w,
        **policy_metrics(cal, p4_mask, scenario_of_user),
    }
    return {"policies": rows, "dedup": dedup_res,
            "rolling_window_choice": {
                "rule": ("rolling window size with highest CALIBRATION aggregate F1 "
                         "(tie-break recall); CALIBRATION only"),
                "aggregates": agg, "chosen": best_wd},
            "p4_mask": p4_mask}


def select_operational_policy(policies: dict) -> tuple[str, dict]:
    """Deterministic CALIBRATION-only selection rule (recorded before TEST).

    Criteria vs P0 (frozen threshold, no de-duplication):
      R1 recall retention >= 0.90 x recall(P0)
      R2 alert reduction >= 10% (n_alerts <= 0.90 x P0)
      R3 concentration not worse than +0.05 Gini
      R4 zero-alert 7-day windows not more than P0
    Among eligible policies: highest recall, then fewest alerts, then
    highest F1. If no policy qualifies: RETAIN P0 (frozen policy).
    """
    p0 = policies["P0"]
    rationale, eligible = {}, []
    for pid, p in policies.items():
        if pid == "P0":
            continue
        r = {
            "R1_recall_retention": p["recall"] >= 0.90 * p0["recall"],
            "R2_alert_reduction": p["n_alerts"] <= 0.90 * p0["n_alerts"],
            "R3_concentration": p["gini"] <= p0["gini"] + 0.05,
            "R4_zero_alert_windows": p["zero_alert_windows_7d"] <= p0["zero_alert_windows_7d"],
        }
        rationale[pid] = r
        if all(r.values()):
            eligible.append(pid)
    if not eligible:
        return "P0", {
            "rule": "R1-R4 vs P0; highest recall, fewest alerts, highest F1",
            "rationale": rationale, "eligible": [],
            "reason": "no policy satisfies all of R1-R4 on CALIBRATION; RETAIN frozen policy",
            "selected": "P0",
        }
    best = max(eligible, key=lambda pid: (policies[pid]["recall"],
                                          -policies[pid]["n_alerts"],
                                          policies[pid]["f1"]))
    return best, {
        "rule": "R1-R4 vs P0; highest recall, fewest alerts, highest F1",
        "rationale": rationale, "eligible": eligible, "selected": best,
        "reason": f"selected on CALIBRATION: {best}",
    }


# ---------------------------------------------------------------------------
# Experiment A: capacity projection (CALIBRATION-based, multiple scenarios)
# ---------------------------------------------------------------------------
def capacity_projection(cal: pd.DataFrame, p0_mask: np.ndarray) -> dict:
    """Observed operating descriptors + 10,000-user scenarios.

    Observed values are OBSERVED (CALIBRATION block of the frozen
    system). Projections are INFERENCE/HYPOTHESIS arithmetic from those
    observables; the model has never been run on 10,000 real users.
    """
    assert_calibration_only(cal)
    feats = [c for c in cal.columns
             if c not in ("user", "day", "is_malicious", "score")]
    active_per_day = (cal[feats].gt(0).any(axis=1)
                      .groupby(pd.DatetimeIndex(pd.to_datetime(cal["day"])).normalize()).sum().mean())
    alerts = int(p0_mask.sum())
    n_days = cal["day"].nunique()
    counts = cal.loc[p0_mask].groupby("user").size()
    gini = gini_of_counts(counts.reindex(cal["user"].unique(), fill_value=0).to_numpy())
    n_users = cal["user"].nunique()
    top_user_share = float(counts.max() / max(1, alerts))
    top1pct_share = float(counts.nlargest(max(1, int(0.01 * n_users))).sum()
                          / max(1, alerts))
    observed = {
        "scope": ("CALIBRATION block (2011-02-01..2011-03-31); frozen "
                  "threshold; P0 (no de-duplication)"),
        "users_represented": int(n_users),
        "user_days": int(len(cal)),
        "n_days": int(n_days),
        "alerts": alerts,
        "alerts_per_day": alerts / max(1, n_days),
        "alerts_per_user_mean": alerts / max(1, n_users),
        "alerted_users": int(counts.size),
        "active_users_per_day_mean": float(active_per_day),
        "gini": gini,
        "top_user_share": float(top_user_share),
        "top_1pct_user_share": float(top1pct_share),
    }
    scenarios = {
        "S1_uniform_scaling": {
            "assumption": ("9,000 additional users behave identically to the "
                           "observed 1,000 (same activity and alert propensity)"),
            "label": "INFERENCE (arithmetic on OBSERVED)",
            "users": 10000,
            "alerts_59d": int(alerts * 10),
            "alerts_per_day": round(alerts / max(1, n_days) * 10, 2),
            "alerts_per_user_mean": round(alerts / max(1, n_users), 4),
        },
        "S2_optimistic_lower_bound": {
            "assumption": "additional users never trigger the threshold (clean population)",
            "label": "HYPOTHESIS (untested)",
            "users": 10000,
            "alerts_59d": int(alerts),
            "alerts_per_day": round(alerts / max(1, n_days), 2),
            "alerts_per_user_mean": round(alerts / 10000, 4),
        },
        "S3_concentration_stress": {
            "assumption": ("volume scales x10 AND the observed top-user "
                           "concentration is preserved per user"),
            "label": "INFERENCE + HYPOTHESIS",
            "users": 10000,
            "alerts_59d": int(alerts * 10),
            "alerts_per_day": round(alerts / max(1, n_days) * 10, 2),
            "top_user_alerts_59d": int(counts.max() * 10),
            "top_user_alerts_per_day": round(counts.max() * 10 / max(1, n_days), 2),
            "top_1pct_user_share": round(top1pct_share, 4),
        },
    }
    return {
        "note": ("No synthetic users and no fabricated labels; the model was "
                 "never run on 10,000 real users. Projections extrapolate "
                 "OBSERVED CALIBRATION behavior only."),
        "observed": observed,
        "scenarios": scenarios,
    }