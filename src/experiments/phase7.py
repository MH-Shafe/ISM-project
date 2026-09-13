"""Phase 7: robustness + freeze of the Phase 6 candidate graph baseline (CERT r4.2).

Candidate: Phase 6 arm C (13 features) or a clearly documented variant (the
department-mismatch decision). ALL robustness decisions use TRAIN/CALIBRATION
only; TEST is evaluated exactly once after the configuration is frozen.

Diagnostics (calibration-only):
  - seed sweep over a predefined seed set (robustness, NOT seed tuning)
  - per-seed gain importance + rank stability (Kendall tau vs seed 42)
  - department_file_type_mismatch_count evidence bundle + documented
    deterministic decision rule

The frozen configuration (FROZEN_CONFIG) is the single source of truth for
the freeze record; the runner asserts the recorded decision agrees with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase6 as p6
from src.models.lightgbm_baseline import default_params, predict_proba, split_datasets, train_baseline
from src.preprocessing.splits import split_user_day_table

# Predefined seed set (robustness diagnostic only; 42 remains the production
# seed -- seeds are NOT selected by performance).
APPROVED_SEEDS = [7, 42, 123, 2024, 2026]

DEPARTMENT_FEATURE = "department_file_type_mismatch_count"


def summarize(values) -> dict:
    """Mean / std (ddof=1) / min / max / coefficient of variation."""
    a = np.asarray(list(values), dtype=float)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {
        "mean": round(mean, 8),
        "std": round(std, 8),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv": round(std / mean, 6) if mean else None,
    }


def seed_sweep(table: pd.DataFrame, feats: list[str],
               seeds: list[int] | None = None) -> list[dict]:
    """Train the candidate on each seed; record CALIBRATION-only diagnostics.

    Every model uses the fixed protocol (default_params + TRAIN-derived
    scale_pos_weight, early stopping on CALIBRATION auc, thresholds on
    CALIBRATION). TEST is never touched.
    """
    seeds = seeds or APPROVED_SEEDS
    out = []
    for seed in seeds:
        model, record = train_baseline(table, features=feats, params={"seed": seed})
        data = split_datasets(table, feats)
        s_cal = predict_proba(model, data["calibration"]["X"])
        t_f1, f1_cal = th.best_f1_threshold(data["calibration"]["y"], s_cal)
        cal = m.classification_metrics(data["calibration"]["y"], s_cal)
        dec = m.binary_decision_metrics(data["calibration"]["y"], s_cal, t_f1)
        out.append({
            "seed": seed,
            "best_iteration": record["best_iteration"],
            "cal_auc_roc": cal["auc_roc"],
            "cal_auc_pr": cal["auc_pr"],
            "threshold_max_f1": t_f1,
            "cal_alert_rate": dec["alert_rate"],
            "cal_precision": dec["precision"],
            "cal_recall": dec["recall"],
            "cal_f1": dec["f1"],
            "cal_mcc": dec["mcc"],
            "train_runtime_s": record["runtime_s"],
            "importance_gain": record["feature_importance_gain"],
        })
    return out


def gain_shares(importance: dict) -> dict:
    """Gain importance normalized to a share of the total (per model)."""
    total = sum(importance.values())
    return {f: importance[f] / total if total else 0.0 for f in importance}


def kendall_tau(a, b) -> float:
    """Kendall tau between two parallel rank lists (ties broken by position)."""
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    items = list(ra)
    conc, disc = 0, 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            da = ra[items[i]] - ra[items[j]]
            db = rb[items[i]] - rb[items[j]]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    return (conc - disc) / max(1, conc + disc)


def feature_stability(sweep: list[dict]) -> dict:
    """Per-feature gain-share statistics across seeds + rank tau vs seed 42."""
    seed42 = next(s for s in sweep if s["seed"] == 42)
    base_rank = sorted(seed42["importance_gain"],
                       key=lambda f: seed42["importance_gain"][f], reverse=True)
    features = sorted({f for s in sweep for f in s["importance_gain"]})
    per_feature = {}
    for f in features:
        shares = [gain_shares(s["importance_gain"])[f] for s in sweep]
        ranks = [sorted(s["importance_gain"],
                        key=lambda x: s["importance_gain"][x], reverse=True).index(f)
                 for s in sweep]
        per_feature[f] = {
            "gain_share": summarize(shares),
            "rank_mean": round(float(np.mean(ranks)), 2),
            "rank_min": int(min(ranks)),
            "rank_max": int(max(ranks)),
            "n_models_gain_zero": int(sum(1 for s in sweep
                                          if s["importance_gain"][f] == 0.0)),
        }
    tau = {}
    for s in sweep:
        rank = sorted(s["importance_gain"],
                      key=lambda f: s["importance_gain"][f], reverse=True)
        tau[s["seed"]] = kendall_tau(base_rank, rank)
    return {"per_feature": per_feature, "rank_tau_vs_seed42": tau}


def department_evidence(table: pd.DataFrame, phase6_gain_dept: float) -> dict:
    """TRAIN/CALIBRATION-only evidence bundle for the department-mismatch feature."""
    parts = split_user_day_table(table, config.SPLIT_DEFAULTS["train_end"],
                                 config.SPLIT_DEFAULTS["calibration_end"])
    tr_cal = pd.concat([parts["train"], parts["calibration"]], ignore_index=True)
    vals = tr_cal[DEPARTMENT_FEATURE]
    dist = {
        "rows": int(len(tr_cal)),
        "nonzero_rows": int((vals != 0).sum()),
        "nonzero_frac": round(float((vals != 0).mean()), 8),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "p99": float(vals.quantile(0.99)),
    }

    def cal_single(f):
        model, record = train_baseline(table, features=[f])
        data = split_datasets(table, [f])
        s_cal = predict_proba(model, data["calibration"]["X"])
        cal = m.classification_metrics(data["calibration"]["y"], s_cal)
        return {"auc_roc": cal["auc_roc"], "auc_pr": cal["auc_pr"],
                "best_iteration": record["best_iteration"]}

    singles = {f: cal_single(f) for f in p6.GRAPH_ORDER}

    # CALIBRATION-only: does removing the feature change the candidate?
    feats13 = p6.ARMS["c"]
    feats12 = [f for f in feats13 if f != DEPARTMENT_FEATURE]
    model13, rec13 = train_baseline(table, features=feats13)
    model12, rec12 = train_baseline(table, features=feats12)
    data13 = split_datasets(table, feats13)
    data12 = split_datasets(table, feats12)
    cal13 = m.classification_metrics(data13["calibration"]["y"],
                                     predict_proba(model13, data13["calibration"]["X"]))
    cal12 = m.classification_metrics(data12["calibration"]["y"],
                                     predict_proba(model12, data12["calibration"]["X"]))
    drop_delta = {
        "auc_roc": round(cal12["auc_roc"] - cal13["auc_roc"], 8),
        "auc_pr": round(cal12["auc_pr"] - cal13["auc_pr"], 8),
        "note": "positive = 12-feature candidate better on CALIBRATION",
    }

    return {
        "feature": DEPARTMENT_FEATURE,
        "scope": "TRAIN + CALIBRATION only; TEST not consulted",
        "phase6_zero_gain": bool(phase6_gain_dept == 0.0),
        "distribution": dist,
        "single_feature_cal": singles[DEPARTMENT_FEATURE],
        "context_single_feature_cal": singles,
        "drop_delta_12_vs_13_cal": drop_delta,
        "cost": "O(events), negligible (Phase 5 record)",
        "leakage_safety": ("verified in Phase 5: future-invariance, department-month "
                           "rule, cold-start department, unseen users"),
        "conceptual": ("only cross-user feature (department-level strictly-past file "
                       "type history); context signal, never a label"),
    }


def decide_department(evidence: dict) -> str:
    """Documented deterministic decision rule (TRAIN/CALIBRATION evidence only).

    REJECT only if ALL of:
      - Phase 6 gain was exactly zero (one fitted model -- corroborated below)
      - single-feature CALIBRATION PR-AUC < 0.01
      - removing the feature does not harm CALIBRATION metrics
        (|delta AUC-ROC| < 0.001 and |delta AUC-PR| < 0.001)
      - the feature is non-zero on < 10% of TRAIN+CALIBRATION rows
    RETAIN if it shows standalone CALIBRATION signal (PR-AUC >= 0.05) or its
    removal hurts CALIBRATION (AUC-PR delta < -0.001).
    Otherwise INCONCLUSIVE (keep the feature; do not force a decision).
    """
    if not evidence.get("phase6_zero_gain"):
        return "INCONCLUSIVE"
    single = evidence["single_feature_cal"]
    delta = evidence["drop_delta_12_vs_13_cal"]
    frac = evidence["distribution"]["nonzero_frac"]
    if (single["auc_pr"] < 0.01
            and abs(delta["auc_roc"]) < 0.001
            and abs(delta["auc_pr"]) < 0.001
            and frac < 0.10):
        return "REJECT"
    if single["auc_pr"] >= 0.05 or delta["auc_pr"] < -0.001:
        return "RETAIN"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Frozen candidate configuration (source of truth for the freeze record).
# `features` was finalized from the recorded Phase 7 department decision
# (REJECT: zero gain in all 5 seeds, ~0.011% non-zero TRAIN+CAL rows,
# single-feature CAL AUC-ROC = 0.5, removal delta exactly 0.0 on
# CALIBRATION); the freeze stage asserts it agrees with the evidence
# artifacts.
# ---------------------------------------------------------------------------
FROZEN_NAME = "lgbm-graph-v1"

FROZEN_CONFIG = {
    "model_name": FROZEN_NAME,
    "dataset_version": "CERT r4.2",
    "feature_version": ("v2 (8 behavioral + 4 graph; "
                        "department_file_type_mismatch_count REJECTED in Phase 7)"),
    "features": (p6.BEHAVIORAL_ORDER
                 + [f for f in p6.GRAPH_ORDER if f != DEPARTMENT_FEATURE]),
    "seed": 42,
    "params": default_params(),
    "early_stopping": "on calibration (auc), patience 100, up to 3000 rounds",
    "calibration_method": "max-F1 threshold on calibration only",
    "threshold_method": "best_f1_threshold on calibration scores",
    "split_strategy": "chronological",
    "split": dict(config.SPLIT_DEFAULTS),
}