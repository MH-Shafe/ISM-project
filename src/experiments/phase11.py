"""Phase 11: explainability layer for the frozen `lgbm-graph-v1` detector.

Explanation-only phase: the frozen model, the Phase 9 alert threshold, and the
Phase 10 conformal layer are never modified; no predictive model is retrained.

Method (selected in Step 2, evidence-based, frozen before TEST explanations):

  PRIMARY (runtime): LightGBM native `pred_contrib` -- the path-dependent
  TreeSHAP algorithm as implemented by LightGBM. Zero new dependencies
  (`shap` is not installed in the local environment; on Kaggle it exists but
  is deliberately NOT a runtime dependency), exact additive decomposition in
  margin space, deterministic and bit-reproducible for the frozen Booster,
  and vectorized (negligible cost vs score computation).

  CROSS-CHECK ONLY: the `shap` package's TreeExplainer computes the same
  path-dependent values for this Booster; if available it is used on a small
  sample to validate agreement, never as a dependency of the explanation
  path, and never for selection.

Output space (exact, per Step 4):
  contributions and the bias operate in MARGIN space (raw score / logit):
    margin(x) = bias + sum_f contrib_f(x)
    prob(x)   = sigmoid(margin(x))
  Contributions are never added to probabilities, and probabilities are never
  subtracted from contributions.

Selection policy (deterministic, score/policy-based, label-free; frozen in
the calibration stage before TEST explanations are generated):
  alerts      : all rows with score >= FROZEN_THRESHOLD
  monitor     : rows with t0(0.05) <= score < FROZEN_THRESHOLD
                (t0 from the frozen Phase 10 conformal fit, alpha = 0.05)
  borderline  : monitor rows with score >= FROZEN_THRESHOLD - BORDERLINE_WIDTH
                (reported as a named subset of monitor)
  non_alert   : decile-stratified sample (SAMPLE_PER_DECILE per decile) of
                rows with score < t0(0.05); decile boundaries from the
                eligible pool's score quantiles (deterministic); within a
                decile rows are ordered by (user, day) and the first
                SAMPLE_PER_DECILE are taken (no RNG anywhere).

Reason templates (Step 9; deterministic, no causal verbs):
  Per-feature sentence: "{feature} (value {v}) increased/decreased the model's
  risk score (contribution {c:+.4f}, rank {r} of {nf})".  A contribution is a
  measured model-output attribution, not a causal claim about the user or the
  world; the templates never say "because", "cause", "drives" or "due to".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.experiments import phase6 as p6
from src.experiments import phase7 as p7
from src.experiments import phase9 as p9

FROZEN_THRESHOLD = p9.FROZEN_THRESHOLD  # 0.9186015432508062 (Phase 9 policy)
EXPLAIN_ALPHA = 0.05                    # conformal set reported in explanations
RECON_TOL = 1e-6                        # absolute margin reconstruction tolerance
N_REASONS = 3
SAMPLE_PER_DECILE = 50
BORDERLINE_WIDTH = 0.05
FEATURES = list(p7.FROZEN_CONFIG["features"])  # exact 12-feature frozen order

CAUSAL_VERBS = ("because", "cause", "drives", "due to", "causes", "caused",
                "driven")

# Registry definitions for every frozen feature (single source of truth in
# src/config.py: BEHAVIORAL_FEATURES / GRAPH_FEATURES). Included in each
# contribution detail so explanations are self-describing.
FEATURE_DEFINITIONS = {
    f: (config.BEHAVIORAL_FEATURES[f].get("definition", "")
        if f in config.BEHAVIORAL_FEATURES
        else config.GRAPH_FEATURES[f].get("definition", ""))
    for f in FEATURES
}

# Deterministic per-feature perturbation deltas for local sensitivity
# analysis: count features shift by 1 (their unit of meaning), ratio
# features (device/file consistency scores, [0, 1]) by 0.1.
COUNT_FEATURES = {
    "login_count", "after_hours_login_count", "usb_connection_count",
    "file_access_count", "sensitive_file_access_count",
    "http_activity_count", "unique_device_count", "unusual_access_count",
    "rare_device_usage_count", "rare_file_type_access_count",
}
RATIO_FEATURES = {"device_consistency_score", "file_type_consistency_score"}
PERTURBATION_DELTAS = {f: (1.0 if f in COUNT_FEATURES else 0.1)
                       for f in FEATURES}

# Risk-level labels attached to each explained row (operational triage).
RISK_LEVELS = ("ALERT", "BORDERLINE", "MONITOR", "NON-ALERT")


def margin_contributions(model, X, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Additive margin-space decomposition via LightGBM pred_contrib.

    Returns (contrib, bias): contrib.shape == (n, 12), bias.shape == (n,),
    with margin == bias + contrib.sum(axis=1) exactly (see
    check_reconstruction for the verified tolerance).
    """
    X = np.asarray(X, dtype=np.float32)
    out = np.asarray(model.predict(X, num_iteration=n_iter, pred_contrib=True),
                     dtype=float)
    return out[:, :-1], out[:, -1]


def check_reconstruction(margin: np.ndarray, contrib: np.ndarray,
                         bias: np.ndarray, tol: float = RECON_TOL) -> dict:
    """Assert the additive identity margin == bias + sum(contrib) within tol."""
    err = np.abs(margin - (bias + contrib.sum(axis=1)))
    return {
        "n": int(len(margin)),
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "tol": tol,
        "ok": bool(err.max() <= tol),
    }


def contribution_detail(feature: str, value: float, c: float,
                        rank: int, n_features: int,
                        definition: str | None = None) -> dict:
    if c > 0:
        direction = "increases risk"
    elif c < 0:
        direction = "decreases risk"
    else:
        direction = "neutral (zero contribution)"
    if definition is None:
        definition = FEATURE_DEFINITIONS.get(feature, "")
    return {
        "feature": feature,
        "definition": definition,
        "value": float(value),
        "contribution": float(c),
        "abs_contribution": float(abs(c)),
        "rank": int(rank),
        "direction": direction,
        "n_features": int(n_features),
    }


def contribution_ranks(contrib: np.ndarray) -> np.ndarray:
    """Rank per row by |contribution| descending; ties by feature index."""
    order = np.argsort(-np.abs(contrib), axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=int)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(order.shape[1])
    return ranks + 1


def _sentence(feature: str, value: float, c: float, rank: int, nf: int) -> str:
    if c > 0:
        verb = "increased"
    elif c < 0:
        verb = "decreased"
    else:
        verb = "left unchanged"
    return (f"{feature} (value {value:g}) {verb} the model's risk score "
            f"(contribution {c:+.4f}, rank {rank} of {nf})")


def top_reasons(explanation: dict, n: int = N_REASONS) -> list[str]:
    """Deterministic reason strings for one explained row (no causal verbs).

    Details are listed in decreasing |contribution| order (rank order).
    """
    details = sorted(explanation["top_reason_details"],
                     key=lambda d: d["rank"])
    out = []
    if explanation["alert"]:
        strongest_pos = max(details, key=lambda d: d["contribution"])
        strongest_neg = min(details, key=lambda d: d["contribution"])
        head = (f"Alert: model risk score {explanation['model_score']:.4f} "
                f"(base {explanation['base_value']:.4f}); strongest positive "
                f"contribution {strongest_pos['feature']} "
                f"(+{strongest_pos['contribution']:.4f})")
        if strongest_neg["contribution"] < 0:
            head += (f"; strongest negative {strongest_neg['feature']} "
                     f"({strongest_neg['contribution']:.4f})")
        out.append(head)
    n_feat = max(0, n - (1 if explanation["alert"] else 0))
    for d in details[:n_feat]:
        out.append(_sentence(d["feature"], d["value"], d["contribution"],
                             d["rank"], d["n_features"]))
    return out[:n]


def explain_rows(keys: pd.DataFrame, values: np.ndarray, probs: np.ndarray,
                 margins: np.ndarray, contrib: np.ndarray, bias: np.ndarray,
                 alert: np.ndarray, conformal_set: np.ndarray,
                 p1: np.ndarray, p0: np.ndarray,
                 features: list[str] | None = None,
                 n_reasons: int = N_REASONS) -> pd.DataFrame:
    """Wide explanation frame (one row per explained user-day).

    keys: user/day frame (preserved per row); values: (n, 12) feature matrix;
    all arrays share the row order.
    """
    features = list(features) if features is not None else list(FEATURES)
    _check_features(features)
    nf = len(features)
    n = len(probs)
    ranks = contribution_ranks(contrib)
    values = np.asarray(values, dtype=float)
    rows = []
    for i in range(n):
        detail = [contribution_detail(features[j], float(values[i, j]),
                                      float(contrib[i, j]), int(ranks[i, j]), nf)
                  for j in range(nf)]
        row = {
            "user": keys.iloc[i]["user"],
            "day": keys.iloc[i]["day"],
            "model_score": float(probs[i]),
            "margin": float(margins[i]),
            "alert": bool(alert[i]),
            "conformal_set": str(conformal_set[i]),
            "conformal_p1": float(p1[i]),
            "conformal_p0": float(p0[i]),
            "base_value": float(bias[i]),
            "top_reason_1": "",
            "top_reason_2": "",
            "top_reason_3": "",
            "top_reason_details": "",
        }
        expl = {"model_score": row["model_score"], "base_value": row["base_value"],
                "alert": row["alert"], "top_reason_details": detail}
        reasons = top_reasons(expl, n=n_reasons)
        for k, r in enumerate(reasons):
            row[f"top_reason_{k + 1}"] = r
        row["top_reason_details"] = json.dumps(detail)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for j, f in enumerate(features):
        frame[f"value_{f}"] = values[:, j]
        frame[f"contribution_{f}"] = contrib[:, j]
    return frame


def _check_features(features: list[str]) -> None:
    if features != FEATURES:
        raise ValueError(
            f"explanation feature order must be the frozen 12-feature list, "
            f"got {features}")


def select_explanation_rows(scores: np.ndarray, alert: np.ndarray,
                            t0: float, keys: pd.DataFrame,
                            n_per_decile: int = SAMPLE_PER_DECILE) -> dict:
    """Deterministic, label-free selection masks (documented policy)."""
    scores = np.asarray(scores, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    monitor = (~alert) & (scores >= t0)
    borderline = monitor & (scores >= FROZEN_THRESHOLD - BORDERLINE_WIDTH)
    low_pool = (~alert) & (scores < t0)
    low_sample = _decile_stratified(scores, low_pool, keys, n_per_decile)
    return {
        "alerts": alert,
        "monitor": monitor,
        "borderline": borderline,
        "non_alert_sample": low_sample,
    }


def _decile_stratified(scores: np.ndarray, eligible: np.ndarray,
                       keys: pd.DataFrame, n_per_decile: int) -> np.ndarray:
    """Deterministic decile-stratified sample (user, day) order within bucket."""
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    pool_idx = np.flatnonzero(eligible)
    if len(pool_idx) == 0:
        return np.zeros(len(scores), dtype=bool)
    pool_s = scores[pool_idx]
    bounds = np.quantile(pool_s, np.linspace(0.0, 1.0, 11))
    order = np.lexsort((keys["user"].to_numpy(), keys["day"].to_numpy()))
    picked = []
    for b in range(10):
        lo, hi = bounds[b], bounds[b + 1]
        in_b = []
        for i in order:
            if not eligible[i]:
                continue
            s = scores[i]
            if lo <= s <= hi if b == 9 else lo <= s < hi:
                in_b.append(i)
        picked.extend(in_b[:n_per_decile])
    mask = np.zeros(len(scores), dtype=bool)
    mask[picked] = True
    return mask


def global_importance(contrib: np.ndarray,
                      features: list[str] | None = None) -> dict:
    """Mean absolute / signed contribution, rank and distribution per feature."""
    features = list(features) if features is not None else list(FEATURES)
    _check_features(features)
    a = np.abs(contrib)
    total_abs = float(a.sum())
    per = {}
    for j, f in enumerate(features):
        col = contrib[:, j]
        per[f] = {
            "mean_abs": float(a[:, j].mean()),
            "share_abs": float(a[:, j].sum() / total_abs) if total_abs else 0.0,
            "mean_signed": float(col.mean()),
            "median": float(np.median(col)),
            "q25": float(np.quantile(col, 0.25)),
            "q75": float(np.quantile(col, 0.75)),
            "frac_nonzero": float((col != 0).mean()),
            "frac_positive": float((col > 0).mean()),
        }
    order = sorted(features, key=lambda f: per[f]["mean_abs"], reverse=True)
    for r, f in enumerate(order):
        per[f]["rank"] = r
    return {"n": int(contrib.shape[0]), "per_feature": per,
            "ranking_mean_abs": order}


def compare_gain_ranking(gain_importance: dict, imp: dict) -> dict:
    """Explanation-vs-gain ranking comparison (Kendall tau + delta ranks).

    Agreement between the two rankings is descriptive evidence, not proof of
    correctness of either.
    """
    gain_order = sorted(gain_importance, key=lambda f: gain_importance[f],
                        reverse=True)
    exp_order = imp["ranking_mean_abs"]
    tau = p7.kendall_tau(gain_order, exp_order)
    delta = {}
    for r, f in enumerate(exp_order):
        gain_rank = gain_order.index(f)
        delta[f] = int(r - gain_rank)
    return {"kendall_tau": float(tau),
            "gain_ranking": gain_order,
            "explanation_ranking": exp_order,
            "delta_rank_expl_vs_gain": delta}


def alert_patterns(contrib_alert: np.ndarray,
                   features: list[str] | None = None) -> dict:
    """Contribution patterns across alert rows (exploratory, label-free)."""
    features = list(features) if features is not None else list(FEATURES)
    _check_features(features)
    nf = len(features)
    n = contrib_alert.shape[0]
    a = contrib_alert
    with np.errstate(invalid="ignore"):
        top_pos = np.argmax(np.where(a > 0, a, -np.inf), axis=1)
        top_neg = np.argmin(np.where(a < 0, a, np.inf), axis=1)
    nonzero = np.abs(a) > 0
    top3 = np.full((n, 3), -1, dtype=int)
    for i in range(n):
        cand = np.nonzero(nonzero[i])[0]
        if cand.size:
            top3[i, :min(3, cand.size)] = cand[
                np.argsort(-np.abs(a[i, cand]), kind="stable")][:min(3, cand.size)]
    per = {}
    for j, f in enumerate(features):
        col = a[:, j]
        per[f] = {
            "frac_top_positive": float((top_pos == j).mean()) if n else 0.0,
            "frac_top_negative": float((top_neg == j).mean()) if n else 0.0,
            "frac_top3_abs": float((top3 == j).any(axis=1).mean()) if n else 0.0,
            "frac_positive": float((col > 0).mean()),
            "frac_negative": float((col < 0).mean()),
            "frac_zero": float((col == 0).mean()),
            "mean_signed": float(col.mean()),
        }
    graph_feats = [f for f in features if f in p6.GRAPH_ORDER]
    graph_cols = np.abs(a[:, [features.index(f) for f in graph_feats]])
    return {
        "n_alerts": int(n),
        "per_feature": per,
        "graph_vs_behavioral": {
            "graph_mean_abs_share": float(graph_cols.sum() / a.sum())
            if n else 0.0,
            "graph_features": graph_feats,
            "behavioral_features": [f for f in features
                                    if f not in graph_feats],
        },
    }


def alert_contribution_stats(contrib: np.ndarray,
                             features: list[str] | None = None) -> dict:
    """Aggregate contribution statistics across alert rows (label-free).

    Complements `alert_patterns` with distributional stats: per-feature
    mean/median, mean abs, sign fractions, mean rank, and top-1/2/3
    frequencies over the alert rows.
    """
    features = list(features) if features is not None else list(FEATURES)
    _check_features(features)
    n = contrib.shape[0]
    ranks = contribution_ranks(contrib)
    a = contrib
    per = {}
    for j, f in enumerate(features):
        col = a[:, j]
        per[f] = {
            "mean": float(col.mean()),
            "median": float(np.median(col)),
            "mean_abs": float(np.abs(col).mean()),
            "median_abs": float(np.median(np.abs(col))),
            "frac_positive": float((col > 0).mean()) if n else 0.0,
            "frac_negative": float((col < 0).mean()) if n else 0.0,
            "frac_zero": float((col == 0).mean()) if n else 0.0,
            "mean_rank": float(ranks[:, j].mean()) if n else 0.0,
            "frac_top1": float((ranks[:, j] == 1).mean()) if n else 0.0,
            "frac_top2": float((ranks[:, j] <= 2).mean()) if n else 0.0,
            "frac_top3": float((ranks[:, j] <= 3).mean()) if n else 0.0,
        }
    return {"n_alerts": int(n), "per_feature": per}


def perturbation_stability(model, X: np.ndarray, margins: np.ndarray,
                           contrib: np.ndarray, n_iter: int,
                           features: list[str] | None = None,
                           deltas: dict | None = None,
                           sample_mask: np.ndarray | None = None) -> dict:
    """Local sensitivity of margins and contribution ranks (inference-only).

    For each feature, add the deterministic perturbation delta to that
    feature alone, re-score the frozen model on the edited rows, and measure:
      - mean/max absolute margin delta (raw score / logit space),
      - fraction of rows whose top-1 (and top-3) |contribution| features
        change,
      - fraction of rows whose alert flag flips (probability-space policy).

    This is internal sensitivity analysis of the frozen model; the frozen
    system's reported scores and decisions for the original rows are never
    altered or re-reported.
    """
    features = list(features) if features is not None else list(FEATURES)
    _check_features(features)
    n = contrib.shape[0]
    if sample_mask is None:
        sample_mask = np.ones(n, dtype=bool)
    X = np.asarray(X, dtype=np.float32)
    margins = np.asarray(margins, dtype=float)
    contrib = np.asarray(contrib, dtype=float)
    use = np.flatnonzero(sample_mask)
    ranks = contribution_ranks(contrib[use])
    top1 = np.argmax(np.abs(contrib[use]), axis=1)
    top3 = np.argsort(-np.abs(contrib[use]), axis=1, kind="stable")[:, :3]
    probs = 1.0 / (1.0 + np.exp(-margins))
    alert = probs >= FROZEN_THRESHOLD
    d = dict(deltas) if deltas is not None else dict(PERTURBATION_DELTAS)
    per = {}
    for j, f in enumerate(features):
        delta = d.get(f, PERTURBATION_DELTAS.get(f, 0.1))
        Xp = X[use].copy()
        Xp[:, j] = Xp[:, j] + delta
        if f in RATIO_FEATURES:
            Xp[:, j] = np.clip(Xp[:, j], 0.0, 1.0)
        else:
            Xp[:, j] = np.maximum(Xp[:, j], 0.0)
        mp = np.asarray(model.predict(Xp, num_iteration=n_iter, raw_score=True),
                        dtype=float)
        cp, _ = margin_contributions(model, Xp, n_iter)
        d_margin = np.abs(mp - margins[use])
        top1_p = np.argmax(np.abs(cp), axis=1)
        top3_p = np.argsort(-np.abs(cp), axis=1, kind="stable")[:, :3]
        alert_p = (1.0 / (1.0 + np.exp(-mp))) >= FROZEN_THRESHOLD
        per[f] = {
            "delta": float(delta),
            "mean_abs_margin_delta": float(d_margin.mean()),
            "max_abs_margin_delta": float(d_margin.max()),
            "frac_top1_flip": float((top1_p != top1).mean()),
            "frac_top3_flip": float(
                (top3_p != top3).any(axis=1).mean()),
            "frac_decision_flip": float((alert_p != alert[use]).mean()),
        }
    return {
        "n_rows": int(len(use)),
        "note": ("frozen model scored on deterministic single-feature "
                 "perturbations; sensitivity analysis only, reported "
                 "outputs for original rows are unchanged"),
        "per_feature": per,
    }


def counterfactual_feasibility() -> dict:
    """Explicit feasibility analysis for counterfactual explanations.

    Decision: NOT_SUPPORTED under the frozen contract (recorded in
    phase11_counterfactual.json).
    """
    return {
        "decision": "NOT_SUPPORTED",
        "definition": ("counterfactual explanation = an altered-risk output "
                       "for an edited input row ('what if feature X had "
                       "value v')"),
        "reasons": [
            ("a counterfactual necessarily reports a risk score for a "
             "hypothetical row; that score is not the frozen model's output "
             "on the observed row, and re-reporting altered scores would "
             "break the frozen-output contract (every reported score/alert "
             "must be the frozen model's score on the true user-day)"),
            ("editing features implies a new, partially hypothetical input "
             "table outside the frozen user_day_features / graph_features "
             "artifacts"),
            ("retraining or perturbing the model to generate counterfactuals "
             "is forbidden (model frozen in Phase 7)"),
            ("the operational needs counterfactuals would serve are covered "
             "without altered outputs: the Phase 10 monitor band (scores in "
             "(t0, threshold)) and the local perturbation stability analysis "
             "(sensitivity of margins/ranks/decisions to small feature "
             "deltas, computed on the frozen model, never re-reported)"),
        ],
        "future_criteria": (
            "a new authorized phase could support counterfactuals only if: "
            "hypothetical rows are explicitly marked as not the frozen "
            "system's outputs; the frozen model and threshold are unchanged; "
            "the counterfactual generator is fixed before evaluation; TEST "
            "is evaluated once; the outputs carry a documented "
            "non-interpretation warning"),
    }


def risk_level(alert: bool, explanation_set: str) -> str:
    """Operational triage label for an explained row (see RISK_LEVELS)."""
    if alert:
        return "ALERT"
    if explanation_set == "borderline":
        return "BORDERLINE"
    if explanation_set == "monitor":
        return "MONITOR"
    return "NON-ALERT"