"""Phase 15: Unseen-User Generalization Diagnosis.

Implements the approved protocol in docs/phase15_specification.md:

  - A: user activity strata (malicious users by number of malicious days);
  - B: feature-shift TRAIN -> CAL -> TEST cohorts (12 frozen features);
  - C: detected vs missed malicious users (TEST block, descriptive);
  - D: graph/trust diagnosis (cohort profiles, discriminative value,
       stability for low-activity users; no graph redesign);
  - E: temporal analysis (onset position, detection delay, early/late);
  - F: threshold diagnostic (descriptive score distributions around the
       frozen threshold 0.9186015432508062; no threshold selection);
  - G: cohort comparison (frozen records + attribution, OBSERVED/INFERENCE/
       HYPOTHESIS labels).

Hard constraints (enforced structurally): the authoritative chronological
TEST (2011-04-01..2011-05-17) never enters this module; no training, no
threshold choice, no production change; the Phase 14 user-disjoint
allocation is preserved and never modified; the TRAIN block contributes
reference distributions/statistics only.

The module is IO-free and deterministic by construction (no RNG, no
resampling). Evidence labels: statistics computed here on verified inputs
are OBSERVED; attribution reasoning is done by the report writer per
specification Section 15.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.experiments.phase14 import (  # noqa: F401  (re-exported for runners)
    T, C, X, FROZEN_FEATURES, canonical_json, to_jsonable,
)

FROZEN_THRESHOLD = 0.9186015432508062
RECORDED_CAL_AUC_ROC = 0.7355124228200639
SCORE_GATE_TOL = 1e-12
CAL_AUC_GATE_TOL = 1e-9

COUNT_FEATURES = [f for f in FROZEN_FEATURES if f not in (
    "device_consistency_score", "file_type_consistency_score")]
CONSISTENCY_FEATURES = ["device_consistency_score",
                        "file_type_consistency_score"]
GRAPH_FEATURES = ["device_consistency_score", "rare_device_usage_count",
                  "file_type_consistency_score", "rare_file_type_access_count"]
BEHAVIORAL_FEATURES = [f for f in FROZEN_FEATURES if f not in GRAPH_FEATURES]

# Fixed malicious-day-count buckets (specification Section 5): mirrors the
# short (<=12) / long (>=45) incident structure of r4.2.
ACTIVITY_BUCKETS = [(1, 1), (2, 5), (6, 12), (13, 44), (45, 100), (101, None)]
BUCKET_LABELS = ["1", "2-5", "6-12", "13-44", "45-100", "101+"]
PSI_N_BINS = 10
PSI_EPS = 1e-6  # zero-share clip for PSI (documented; not a decision rule)


def _bucket_of(n: int) -> str:
    for (lo, hi), label in zip(ACTIVITY_BUCKETS, BUCKET_LABELS):
        if n >= lo and (hi is None or n <= hi):
            return label
    raise AssertionError(f"n={n} outside pre-registered buckets")


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(x, dtype=float))


def _transform(feature: str, values: np.ndarray) -> np.ndarray:
    """Pre-registered per-feature transform (specification Section 6)."""
    if feature in CONSISTENCY_FEATURES:
        return np.asarray(values, dtype=float)
    return _log1p(values)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's d on transformed user-level values; None if degenerate."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    if sa == 0 and sb == 0:
        return 0.0 if a.mean() == b.mean() else None
    pooled = np.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2)
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return None
    return float((a.mean() - b.mean()) / pooled)


def _ks(a: np.ndarray, b: np.ndarray) -> dict | None:
    """Two-sample KS on transformed user-level values (descriptive p)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    res = stats.ks_2samp(a, b)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def _psi(reference: np.ndarray, actual: np.ndarray,
         feature: str, n_bins: int = PSI_N_BINS) -> float | None:
    """PSI of `actual` vs `reference` rows; bins fitted on reference only.

    n_bins equal-frequency bins on the transformed reference; zero shares
    clipped to PSI_EPS (documented); None if < 2 usable bin boundaries
    (constant feature).
    """
    ref = _transform(feature, reference)
    act = _transform(feature, actual)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    if len(np.unique(edges)) < 2:
        return None
    exp, _ = np.histogram(ref, bins=edges)
    actv, _ = np.histogram(act, bins=edges)
    n_exp, n_act = len(ref), len(act)
    if n_exp == 0 or n_act == 0:
        return None
    e = np.clip(exp / n_exp, PSI_EPS, 1.0)
    a = np.clip(actv / n_act, PSI_EPS, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------------
# A: user activity strata
# ---------------------------------------------------------------------------
def activity_strata(table: pd.DataFrame, test_pred: pd.DataFrame,
                    user_diag: dict) -> dict:
    """Malicious users by number of malicious days; detection by stratum.

    Strata over the 70 malicious users of the allocation (reference) and
    over the 15 TEST-block malicious users (detection, alerts). Detection
    flags come from the verified Phase 14 per-user diagnostics.
    """
    mal_days = (table[table["is_malicious"] == 1]
                .groupby("user")["day"].count().to_dict())
    diag_users = {d["user"]: d for d in user_diag["per_user"]}
    zero_det = set(user_diag["zero_detection_users"])

    mal_rows = test_pred[test_pred["is_malicious"] == 1]
    alerts = test_pred[test_pred["alert_primary"].to_numpy().astype(bool)]
    n_alerts_total = len(alerts)

    all_ref = []
    test_strata = []
    for label in BUCKET_LABELS:
        users = sorted(u for u, n in mal_days.items() if _bucket_of(n) == label)
        all_ref.append({
            "bucket": label, "n_users": len(users),
            "malicious_days": int(sum(mal_days[u] for u in users)),
        })
        tu = sorted(u for u in users if u in diag_users)
        detected = [u for u in tu if u not in zero_det]
        undetected = [u for u in tu if u in zero_det]
        undet_rows = mal_rows[mal_rows["user"].isin(undetected)]
        bucket_alerts = alerts[alerts["user"].isin(tu)]
        test_strata.append({
            "bucket": label,
            "n_users": len(tu),
            "detected_users": detected,
            "detection_rate": (len(detected) / len(tu)
                               if tu else None),
            "missed_malicious_days": int(len(undet_rows)),
            "n_alerts": int(len(bucket_alerts)),
            "alert_share": (float(len(bucket_alerts) / n_alerts_total)
                            if n_alerts_total else None),
            "malicious_day_score_mean": (float(undet_rows["score"].mean())
                                         if len(undet_rows) else None),
            "malicious_day_score_max": (float(undet_rows["score"].max())
                                        if len(undet_rows) else None),
        })
    return {
        "buckets": BUCKET_LABELS,
        "bucket_definition": "malicious days per user: 1 / 2-5 / 6-12 / "
                             "13-44 / 45-100 / 101+ (specification Section 5)",
        "all_70_malicious_users": all_ref,
        "test_block_15_users": test_strata,
        "n_malicious_users_total": len(mal_days),
        "n_alerts_total_test": n_alerts_total,
        "evidence_label": "OBSERVED (frozen label column + verified Phase 14 "
                          "per-user diagnostics)",
        "note": "descriptive only; empty strata are reported as n_users=0",
    }


# ---------------------------------------------------------------------------
# B: feature-shift (TRAIN -> CAL -> TEST cohorts)
# ---------------------------------------------------------------------------
def _user_level(table: pd.DataFrame, users: list[str]) -> pd.DataFrame:
    """Per-user medians over the user's full 501-day timeline."""
    part = table[table["user"].isin(users)]
    return part.groupby("user")[FROZEN_FEATURES].median()


def _feature_shift_one(feature: str, tr: pd.DataFrame,
                       cal: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Per-feature cohort drift on user-level values (spec Section 6)."""
    a, b, c = tr[feature].to_numpy(), cal[feature].to_numpy(), \
        te[feature].to_numpy()
    ta, tb, tc = _transform(feature, a), _transform(feature, b), \
        _transform(feature, c)

    def q(x: np.ndarray) -> list[float] | None:
        if len(x) == 0:
            return None
        return [float(v) for v in np.quantile(x, [0.0, .25, .5, .75, 1.0])]

    def mad(x: np.ndarray) -> float | None:
        if len(x) == 0:
            return None
        return float(np.median(np.abs(x - np.median(x))))

    return {
        "feature": feature,
        "feature_kind": ("graph" if feature in GRAPH_FEATURES
                         else "behavioral"),
        "transform": "log1p" if feature in COUNT_FEATURES else "raw",
        "cohort_user_median_quantiles": {
            "TRAIN": q(a), "CAL": q(b), "TEST": q(c)},
        "cohort_user_median_quantiles_are": "raw user-level medians "
                                            "(501-day timeline)",
        "cohort_mean_transformed": {
            "TRAIN": float(ta.mean()) if len(ta) else None,
            "CAL": float(tb.mean()) if len(tb) else None,
            "TEST": float(tc.mean()) if len(tc) else None},
        "cohort_mad_transformed": {
            "TRAIN": mad(ta), "CAL": mad(tb), "TEST": mad(tc)},
        "cohort_zero_median_fraction": {
            "TRAIN": float((a == 0).mean()) if len(a) else None,
            "CAL": float((b == 0).mean()) if len(b) else None,
            "TEST": float((c == 0).mean()) if len(c) else None},
        "d_train_test": _cohen_d(ta, tc),
        "d_cal_test": _cohen_d(tb, tc),
        "ks_train_test": _ks(ta, tc),
        "ks_cal_test": _ks(tb, tc),
        "zero_median_shift_train_test": (
            float((c == 0).mean() - (a == 0).mean())
            if len(a) and len(c) else None),
        "psi_cal": None, "psi_test": None,
    }


def feature_shift(table: pd.DataFrame, blocks: dict) -> dict:
    """Cohort feature-shift over the 12 frozen features (specification B)."""
    cohorts = {s: _user_level(table, blocks[s]["users"])
               for s in (T, C, X)}
    out = []
    for f in FROZEN_FEATURES:
        rec = _feature_shift_one(f, cohorts[T], cohorts[C], cohorts[X])
        ref_rows = table.loc[table["user"].isin(blocks[T]["users"]), f]
        for split, name in ((C, "psi_cal"), (X, "psi_test")):
            rows = table.loc[table["user"].isin(blocks[split]["users"]), f]
            rec[name] = _psi(ref_rows.to_numpy(), rows.to_numpy(), f)
        out.append(rec)

    def rank_key(r: dict) -> float:
        ranks = []
        for key in ("d_train_test", "ks_train_test"):
            v = r[key]
            if v is None:
                continue
            v = abs(v["statistic"]) if isinstance(v, dict) else abs(v)
            ranks.append(v)
        for key in ("psi_cal", "psi_test"):
            v = r[key]
            if v is not None:
                ranks.append(v)
        return float(np.mean(ranks)) if ranks else float("inf")

    ranked = sorted(out, key=rank_key)
    return {
        "features": out,
        "ranking_by_combined_drift": {
            "method": "mean of |d|, KS statistic, and PSI (CAL, TEST); "
                      "larger = larger drift",
            "top_3": [r["feature"] for r in ranked[-3:]][::-1],
            "bottom_3": [r["feature"] for r in ranked[:3]],
            "full_rank": [r["feature"] for r in ranked][::-1],
        },
        "cohort_n_users": {s: int(len(blocks[s]["users"])) for s in (T, C, X)},
        "reference_fit": "PSI bins fitted on TRAIN-block rows only "
                         "(specification Section 6); zero shares clipped "
                         "at 1e-6",
        "evidence_label": "OBSERVED (computed on verified frozen table + "
                          "Phase 14 allocation)",
    }


# ---------------------------------------------------------------------------
# C: detected vs missed malicious users (TEST block)
# ---------------------------------------------------------------------------
def detected_vs_missed(table: pd.DataFrame, test_pred: pd.DataFrame,
                       user_diag: dict) -> dict:
    """Per-user profile for the 15 TEST malicious users; descriptive."""
    diag_users = {d["user"]: d for d in user_diag["per_user"]}
    zero_det = set(user_diag["zero_detection_users"])
    mal_test = sorted(u for u, d in diag_users.items()
                      if d["n_positive_days"] > 0)

    part = table[table["user"].isin(mal_test)]
    g = part.groupby("user")
    login_pos = (part["login_count"] > 0).groupby(part["user"]).sum()
    active_before = {}
    onset_of = {}
    for u, d in part.sort_values(["user", "day"]).groupby("user"):
        active_days = d.loc[d["login_count"] > 0, "day"]
        onset = d.loc[d["is_malicious"] == 1, "day"].min()
        onset_of[u] = onset
        active_before[u] = int((active_days < onset).sum())
    graph_means = g[GRAPH_FEATURES].mean()
    graph_stds = g[CONSISTENCY_FEATURES].std().rename(
        columns={c: c + "_std" for c in CONSISTENCY_FEATURES})

    rows = []
    for u in mal_test:
        d = diag_users[u]
        p = test_pred[test_pred["user"] == u]
        mal_scores = p.loc[p["is_malicious"] == 1, "score"]
        rows.append({
            "user": u,
            "detected": u not in zero_det,
            "n_malicious_days": int(d["n_positive_days"]),
            "n_active_days": int(login_pos[u]),
            "total_activity_log1p": float(np.log1p(
                int(part.loc[part["user"] == u, "login_count"].sum()))),
            "activity_tercile": d["activity_tercile"],
            "length_class": d["length_class"],
            "scenario": d["scenario"],
            "onset": d["onset"],
            "temporal_half": d["temporal_half"],
            "active_days_before_onset": int(active_before[u]),
            "max_score": float(p["score"].max()),
            "n_alerts": int(p["alert_primary"].sum()),
            "malicious_day_score_median": (float(mal_scores.median())
                                           if len(mal_scores) else None),
            "graph_feature_user_mean": {f: float(graph_means.loc[u, f])
                                        for f in GRAPH_FEATURES},
            "consistency_user_std": {f + "_std": (
                float(graph_stds.loc[u, f + "_std"])
                if f + "_std" in graph_stds else None)
                for f in CONSISTENCY_FEATURES},
        })

    det = [r for r in rows if r["detected"]]
    mis = [r for r in rows if not r["detected"]]

    def summarize(rs: list[dict], key: str) -> dict:
        vals = [r[key] for r in rs if r[key] is not None]
        return {"n": len(vals),
                "median": float(np.median(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None}

    def summarize_nested(rs: list[dict], dict_key: str, field: str) -> dict:
        vals = [r[dict_key][field] for r in rs
                if r[dict_key].get(field) is not None]
        return {"n": len(vals),
                "median": float(np.median(vals)) if vals else None,
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None}

    compare = {}
    for key in ("n_malicious_days", "n_active_days", "total_activity_log1p",
                "active_days_before_onset", "max_score", "n_alerts"):
        compare[key] = {"detected": summarize(det, key),
                        "missed": summarize(mis, key)}
    for f in GRAPH_FEATURES:
        compare["graph_" + f] = {
            "detected": summarize_nested(det, "graph_feature_user_mean", f),
            "missed": summarize_nested(mis, "graph_feature_user_mean", f)}
    for f in CONSISTENCY_FEATURES:
        compare["consistency_std_" + f] = {
            "detected": summarize_nested(det, "consistency_user_std",
                                         f + "_std"),
            "missed": summarize_nested(mis, "consistency_user_std",
                                       f + "_std")}

    return {
        "n_detected": len(det), "n_missed": len(mis),
        "missed_users": [r["user"] for r in mis],
        "users": rows,
        "comparison": compare,
        "evidence_label": "OBSERVED (computed on verified inputs)",
        "note": "descriptive only; n_missed=2, no significance test is "
                "run or claimed (specification Section 7)",
    }


# ---------------------------------------------------------------------------
# D: graph/trust diagnosis
# ---------------------------------------------------------------------------
def _row_auc(y: np.ndarray, x: np.ndarray, feature: str) -> float | None:
    """Single-feature ranking AUC (no model fitted). None if degenerate."""
    x = _transform(feature, np.asarray(x, dtype=float))
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, x))
    except ValueError:
        return None


def graph_diagnosis(table: pd.DataFrame, blocks: dict) -> dict:
    """Graph-feature cohort profiles, discrimination, stability."""
    tr = table[table["user"].isin(blocks[T]["users"])]
    te = table[table["user"].isin(blocks[X]["users"])]
    cal = table[table["user"].isin(blocks[C]["users"])]

    profiles = {}
    for f in GRAPH_FEATURES:
        a, b, c = tr[f].to_numpy(), cal[f].to_numpy(), te[f].to_numpy()
        ta, tb, tc = _transform(f, a), _transform(f, b), _transform(f, c)
        profiles[f] = {
            "cohort_user_median_quantiles": {
                s: [float(v) for v in np.quantile(
                    _user_level(table, blocks[s]["users"])[f],
                    [0.0, .25, .5, .75, 1.0])] for s in (T, C, X)},
            "d_train_test": _cohen_d(ta, tc),
            "ks_train_test": _ks(ta, tc),
            "row_zero_fraction": {"TRAIN": float((a == 0).mean()),
                                  "CAL": float((b == 0).mean()),
                                  "TEST": float((c == 0).mean())},
            "row_one_fraction": {"TRAIN": float((a == 1).mean()),
                                 "CAL": float((b == 1).mean()),
                                 "TEST": float((c == 1).mean())},
        }

    discrimination = {}
    for f in GRAPH_FEATURES:
        discrimination[f] = {
            "auc_roc_in_TRAIN_block": _row_auc(
                tr["is_malicious"].to_numpy(), tr[f].to_numpy(), f),
            "auc_roc_in_TEST_block": _row_auc(
                te["is_malicious"].to_numpy(), te[f].to_numpy(), f),
            "note": "single-feature row-level ranking AUC; no model fitted",
        }

    stability = {}
    for f in CONSISTENCY_FEATURES:
        te_users = sorted(te["user"].unique())
        stds, acts = [], []
        for u in te_users:
            d = te[te["user"] == u]
            v = d[f].std(ddof=1)
            if not np.isnan(v):
                stds.append(float(v))
                acts.append(int((d["login_count"] > 0).sum()))
        r = None
        if len(stds) >= 3 and len(set(stds)) > 1 and len(set(acts)) > 1:
            r = stats.pearsonr(np.asarray(stds), np.asarray(acts))
        stability[f] = {
            "test_block_users_with_variance": len(stds),
            "pearson_r_std_vs_active_days": (
                float(r.statistic) if r is not None else None),
            "pearson_pvalue": (float(r.pvalue) if r is not None else None),
        }

    return {
        "cohort_profiles": profiles,
        "discrimination": discrimination,
        "stability": stability,
        "hypotheses_tested": {
            "consistency_unstable_for_low_activity": (
                "HYPOTHESIS: consistency scores are volatile for "
                "low-activity users; tested via std-vs-activity correlation "
                "and one-fraction/zero-fraction by cohort"),
            "graph_features_less_discriminative_for_unseen_users": (
                "HYPOTHESIS: graph features separate malicious activity "
                "less well for TEST-block users; tested via within-block "
                "single-feature AUC"),
        },
        "evidence_label": "OBSERVED statistics; interpretation is INFERENCE "
                          "in the report",
        "no_redesign": True,
    }


# ---------------------------------------------------------------------------
# E: temporal analysis (TEST block)
# ---------------------------------------------------------------------------
def temporal_analysis(table: pd.DataFrame, test_pred: pd.DataFrame,
                      user_diag: dict) -> dict:
    """Onset position, history length, detection delay, early vs late."""
    diag_users = {d["user"]: d for d in user_diag["per_user"]}
    zero_det = set(user_diag["zero_detection_users"])
    mal_test = sorted(u for u, d in diag_users.items()
                      if d["n_positive_days"] > 0)

    part = table[table["user"].isin(mal_test)].sort_values(["user", "day"])
    records = []
    for u, d in part.groupby("user"):
        active_days = d.loc[d["login_count"] > 0, "day"].tolist()
        onset = d.loc[d["is_malicious"] == 1, "day"].min()
        onset_ord = active_days.index(onset) + 1 if onset in active_days \
            else int((d["day"] < onset).sum()) + 1
        diag = diag_users[u]
        half = ("2011" if onset.year == 2011 else
                "2010 H1" if onset.month <= 6 else "2010 H2")
        records.append({
            "user": u,
            "detected": u not in zero_det,
            "onset": str(onset.date()),
            "onset_half": half,
            "onset_ordinal_active_day": int(onset_ord),
            "active_days_before_onset": int(onset_ord - 1),
            "first_alert_day": diag["first_alert_day"],
            "detection_delay_days": diag["detection_delay_days"],
            "first_week_detection": diag["detected_first_week"],
            "first_tercile_detection": diag["detected_first_tercile"],
        })

    onsets = sorted(r["onset"] for r in records)
    median_onset = onsets[len(onsets) // 2]
    early = [r for r in records if r["onset"] <= median_onset]
    late = [r for r in records if r["onset"] > median_onset]

    def group_summary(rs: list[dict]) -> dict:
        detected = [r for r in rs if r["detected"]]
        delays = [r["detection_delay_days"] for r in detected
                  if r["detection_delay_days"] is not None]
        return {
            "n_users": len(rs),
            "detected": len(detected),
            "detection_rate": (len(detected) / len(rs) if rs else None),
            "detection_delay_days": {
                "n": len(delays),
                "median": float(np.median(delays)) if delays else None,
                "min": float(np.min(delays)) if delays else None,
                "max": float(np.max(delays)) if delays else None},
        }

    by_half = {}
    for h in ("2010 H1", "2010 H2", "2011"):
        rs = [r for r in records if r["onset_half"] == h]
        by_half[h] = group_summary(rs)

    return {
        "users": records,
        "median_onset": median_onset,
        "early_group": {"definition": f"onset <= {median_onset}",
                        **group_summary(early)},
        "late_group": {"definition": f"onset > {median_onset}",
                       **group_summary(late)},
        "by_onset_half": by_half,
        "evidence_label": "OBSERVED (computed on verified inputs; "
                          "delay/first-week flags reused from verified "
                          "Phase 14 diagnostics)",
        "note": "detection delay is days from first alert_primary day to "
                "onset; alerts also fire on benign days, so delay can be "
                "negative (alert precedes onset)",
    }


# ---------------------------------------------------------------------------
# F: threshold diagnostic (descriptive; no threshold selection)
# ---------------------------------------------------------------------------
def _near_threshold_stats(score: np.ndarray, y: np.ndarray,
                          alert: np.ndarray) -> dict:
    """Near-threshold descriptive statistics at the frozen threshold."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    a = np.asarray(alert, dtype=bool)
    t = FROZEN_THRESHOLD
    mal, ben = s[y == 1], s[y == 0]
    fp = s[a & (y == 0)]
    tp = s[a & (y == 1)]
    near = {}
    for d in (0.05, 0.1, 0.2):
        near[str(d)] = {
            "missed_positives_within": int(
                ((mal < t) & (mal >= t - d)).sum()),
            "false_positives_within": int(
                ((fp >= t) & (fp < t + d)).sum()),
        }

    def pct(x: np.ndarray) -> list[float] | None:
        if len(x) == 0:
            return None
        return [float(v) for v in np.quantile(
            x, [0.0, .25, .5, .75, .9, .95, .99, 1.0])]

    return {
        "threshold": t,
        "score_percentiles": {
            "malicious_rows": pct(mal),
            "benign_rows": pct(ben),
        },
        "n_malicious_rows": int(len(mal)),
        "n_benign_rows": int(len(ben)),
        "near_threshold": near,
        "max_missed_positive_score": float(mal.max()) if len(mal) else None,
        "alerted_benign_margin": {
            "min": float(fp.min() - t) if len(fp) else None,
            "median": float(np.median(fp) - t) if len(fp) else None,
            "max": float(fp.max() - t) if len(fp) else None,
        },
        "alerted_true_positive_margin": {
            "median": float(np.median(tp) - t) if len(tp) else None,
        },
    }


def threshold_diagnostic(test_pred: pd.DataFrame,
                         cal_scores: pd.DataFrame) -> dict:
    """Score distributions around the frozen threshold (TEST + CAL)."""
    te = _near_threshold_stats(
        test_pred["score"].to_numpy(),
        test_pred["is_malicious"].to_numpy(),
        test_pred["alert_primary"].to_numpy())

    cal = _near_threshold_stats(
        cal_scores["score"].to_numpy(),
        cal_scores["is_malicious"].to_numpy(),
        (cal_scores["score"].to_numpy() >= FROZEN_THRESHOLD))

    grid = [0.1, 0.25, 0.5, 0.75, 0.9]
    cdf = {
        "score_quantiles_at_probability_grid": {
            "CAL": ([float(v) for v in np.quantile(
                cal_scores["score"].to_numpy(), grid)]
                    if len(cal_scores) else None),
            "TEST": ([float(v) for v in np.quantile(
                test_pred["score"].to_numpy(), grid)]
                     if len(test_pred) else None),
        },
        "grid": grid,
    }
    return {
        "test_block": te,
        "cal_block": cal,
        "cdf_comparison": cdf,
        "evidence_label": "OBSERVED (TEST scores = verified Phase 14 "
                          "artifact; CAL scores = locally reproduced from "
                          "the frozen auxiliary model under the "
                          "reproduction gate)",
        "note": "descriptive only; no threshold selection, no operating-point "
                "recommendation (specification Section 10)",
    }


# ---------------------------------------------------------------------------
# G: cohort comparison (frozen records + attribution)
# ---------------------------------------------------------------------------
def cohort_comparison(frozen: dict) -> dict:
    """OBSERVED records table + pre-registered attribution statements."""
    rows = [
        {"cohort": "chronological CAL (2011-02..03)",
         "model": "lgbm-graph-v1 (P7/P9, frozen)",
         "auc_roc": frozen["p7_cal_auc_roc"],
         "auc_roc_note": "seed sweep mean +/- std (P7)",
         "alerts_at_threshold": frozen["p9_cal_alerts"],
         "precision": frozen["p9_cal_precision"],
         "recall": frozen["p9_cal_recall"],
         "coverage": frozen["p9_cal_coverage"]},
        {"cohort": "chronological TEST (2011-04..05-17)",
         "model": "lgbm-graph-v1 (frozen)",
         "auc_roc": frozen["p12_test_auc_roc"],
         "auc_roc_note": "evaluated once (P12 record)",
         "alerts_at_threshold": frozen["p12_test_alerts"],
         "precision": frozen["p12_test_precision"],
         "recall": frozen["p12_test_recall"],
         "coverage": frozen["p12_test_coverage"],
         "coverage_note": ("detected / malicious users "
                           f"{frozen['p12_test_coverage']} / "
                           f"{frozen['p12_test_n_malicious_users']} "
                           "(computed from frozen P12 predictions "
                           "parquet, alert_selected)"),
         "n_chronological_test_malicious_users": frozen[
             "p12_test_n_malicious_users"]},
        {"cohort": "user-disjoint CAL (full 501 days)",
         "model": "lgbm-graph-v1-uhold (P14 aux)",
         "auc_roc": RECORDED_CAL_AUC_ROC,
         "auc_roc_note": "early-stopping objective (mildly optimistic)",
         "alerts_at_threshold": None,
         "precision": None, "recall": None, "coverage": None},
        {"cohort": "user-disjoint TEST (full 501 days)",
         "model": "lgbm-graph-v1-uhold (P14 aux)",
         "auc_roc": frozen["p14_test_auc_roc"],
         "auc_roc_note": "evaluated once per finalized protocol",
         "alerts_at_threshold": frozen["p14_test_alerts"],
         "precision": frozen["p14_test_precision"],
         "recall": frozen["p14_test_recall"],
         "coverage": frozen["p14_test_coverage"]},
    ]
    attribution = [
        {"claim": "Both user-disjoint cohorts (CAL and TEST) degrade vs the "
                  "chronological cohorts: CAL 0.7355 vs 0.888+/-0.016, TEST "
                  "0.7865 vs 0.9392, while the alert count rises (1,026 vs "
                  "49) at the same frozen threshold.",
         "label": "OBSERVED",
         "basis": "frozen P7/P9/P12 records + verified Phase 14 artifacts"},
        {"claim": "Because the user-disjoint blocks span the same 501-day "
                  "timeline, a pure calendar-time effect cannot explain the "
                  "drop; the drop is associated with entity novelty.",
         "label": "INFERENCE",
         "basis": "allocation structure (users partitioned, days shared)"},
        {"claim": "The alert explosion at an unchanged threshold implies an "
                  "absolute score-scale shift for unseen users (calibration "
                  "transport failure), beyond the AUC-ROC discrimination "
                  "drop.",
         "label": "INFERENCE",
         "basis": "1026 vs 49 alerts; tested in F via score distributions"},
        {"claim": "The recorded user-disjoint CAL AUC-ROC 0.7355 is the "
                  "early-stopping objective and is mildly optimistic; the "
                  "TEST 0.7865 is the honest out-of-sample number.",
         "label": "INFERENCE",
         "basis": "early-stopping protocol (patience 100 on CAL)"},
        {"claim": "Low-activity unseen users dominate the degradation: all "
                  "15 unseen malicious users fall in the low-activity "
                  "tercile of their 108-user block.",
         "label": "OBSERVED",
         "basis": "verified Phase 14 user diagnostics (strata)"},
        {"claim": "Cross-phase comparisons mix two model instances (186 "
                  "trees chronological vs 214 trees user-disjoint) and two "
                  "time scopes (47-day tail vs full 501 days); the "
                  "unseen-user effect can therefore not be isolated "
                  "numerically across phases.",
         "label": "INFERENCE",
         "basis": "model records P7 vs P14"},
        {"claim": "Within Phase 14, only in-sample (TRAIN) vs out-of-sample "
                  "unseen-user (CAL/TEST) contrasts exist; feature-shift "
                  "(B), graph (D), and threshold (F) evidence is used to "
                  "rank candidate causes: feature shift, sparse-user "
                  "behavior, graph-feature weakness, operating-point "
                  "effects.",
         "label": "HYPOTHESIS",
         "basis": "no counterfactual allocation exists inside r4.2"},
    ]
    return {
        "rows": rows,
        "phase13_stability": {"verdict": frozen["p13_verdict"],
                              "note": "frozen Phase 13 record; 2/7 "
                                      "consistent TEST windows under 2011 "
                                      "conditions (NOT VERIFIED here)"},
        "attribution": attribution,
        "evidence_label": "OBSERVED rows from frozen records; attribution "
                          "statements labeled individually",
    }

# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
def run_phase15(table: pd.DataFrame, blocks: dict, test_pred: pd.DataFrame,
                user_diag: dict, cal_scores: pd.DataFrame,
                frozen: dict, reproduction: dict) -> dict:
    """Assemble the full Phase 15 diagnostic result (deterministic)."""
    if set(test_pred.columns) != {"user", "day", "is_malicious", "score",
                                  "alert_primary", "alert_secondary"}:
        raise AssertionError(f"test_pred columns: {list(test_pred.columns)}")
    if set(cal_scores.columns) != {"user", "day", "is_malicious", "score"}:
        raise AssertionError(f"cal_scores columns: {list(cal_scores.columns)}")
    if not set(blocks).issubset({T, C, X}):
        raise AssertionError("blocks must cover TRAIN/CAL/TEST only")

    return {
        "experiment_id": "phase15-unseen-user-diagnosis",
        "phase": "15",
        "scope": "descriptive diagnosis only; no training, no threshold "
                 "selection, no production change",
        "activity_strata": activity_strata(table, test_pred, user_diag),
        "feature_shift": feature_shift(table, blocks),
        "detected_vs_missed": detected_vs_missed(table, test_pred, user_diag),
        "graph_diagnosis": graph_diagnosis(table, blocks),
        "temporal_analysis": temporal_analysis(table, test_pred, user_diag),
        "threshold_diagnostic": threshold_diagnostic(test_pred, cal_scores),
        "cohort_comparison": cohort_comparison(frozen),
        "reproduction_gates": reproduction,
        "frozen_system": {
            "model": "lgbm-graph-v1",
            "threshold": FROZEN_THRESHOLD,
            "seed": 42,
            "best_iteration_frozen": 186,
            "untouched": True,
        },
        "test_evaluation_policy": ("the authoritative chronological TEST is "
                                   "never read or scored; the Phase 14 "
                                   "auxiliary TEST is analyzed descriptively "
                                   "only"),
    }


def json_safe(obj: Any) -> Any:
    """JSON-safe conversion (numpy/date aware)."""
    return to_jsonable(obj)


def canonical(obj: Any) -> str:
    """Deterministic canonical JSON (sorted keys) for bit-comparison."""
    return canonical_json(obj)