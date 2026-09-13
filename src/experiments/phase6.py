"""Phase 6 protocol: controlled graph-vs-behavioral regression (CERT r4.2).

Research question: do the Phase 5 graph/trust features provide measurable
predictive value beyond the frozen 8-feature behavioral baseline?

Three arms, identical evaluation protocol, only the feature set differs:

  a  behavioral only   8 registry-order features (control; must reproduce
                       the frozen phase4-b record within tolerance)
  b  graph only        5 Phase 5 graph features (registry order)
  c  behavioral+graph  13 features, explicit order a-then-b (primary)

All protocol pieces are reused from existing src modules: chronological
split (split_user_day_table, config.SPLIT_DEFAULTS), fixed hyperparameters
(default_params with TRAIN-derived scale_pos_weight), early stopping on
CALIBRATION auc, thresholds selected on CALIBRATION, TEST evaluated
exactly once per arm, bootstrap uncertainty is evaluation-only.

Leakage guards inside run_arm: the model input must be exactly the
registered feature list in the given order -- identifier/date/target
columns are structurally impossible to enter.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.models.lightgbm_baseline import predict_proba, split_datasets, train_baseline

BEHAVIORAL_ORDER = [c for c in config.BEHAVIORAL_FEATURES
                    if config.BEHAVIORAL_FEATURES[c]["status"] == "defined"]
GRAPH_ORDER = list(config.GRAPH_FEATURES)

ARMS = {
    "a": BEHAVIORAL_ORDER,
    "b": GRAPH_ORDER,
    "c": BEHAVIORAL_ORDER + GRAPH_ORDER,
}

KEY_COLUMNS = ("user", "day")
FORBIDDEN = KEY_COLUMNS + ("is_malicious",)

# Frozen phase4-b record (reproducibility gate for arm a). Values are the
# TEST measurements recorded in reports/artifacts/phase4_experiment_b.json.
REF_A = {
    "auc_roc": 0.9353409978000142,
    "auc_pr": 0.15304127090471,
    "best_iteration": 160,
}
AUC_ATOL = 1e-6

# Protocol metric list (paths into an experiment dict) -- identical to the
# Phase 4 comparison table so the A-vs-phase4 record is directly readable.
METRIC_KEYS = {
    "auc_roc": ("evaluation", "classification", "auc_roc"),
    "auc_pr": ("evaluation", "classification", "auc_pr"),
    "precision_at_10": ("evaluation", "top_k", "precision_at_10"),
    "precision_at_30": ("evaluation", "top_k", "precision_at_30"),
    "recall_at_50": ("evaluation", "top_k", "recall_at_50"),
    "precision": ("evaluation", "at_max_f1_threshold", "precision"),
    "recall": ("evaluation", "at_max_f1_threshold", "recall"),
    "f1": ("evaluation", "at_max_f1_threshold", "f1"),
    "mcc": ("evaluation", "at_max_f1_threshold", "mcc"),
    "balanced_accuracy": ("evaluation", "at_max_f1_threshold", "balanced_accuracy"),
    "fpr": ("evaluation", "at_max_f1_threshold", "fpr"),
    "fnr": ("evaluation", "at_max_f1_threshold", "fnr"),
    "alert_rate": ("evaluation", "at_max_f1_threshold", "alert_rate"),
    "n_alerts": ("evaluation", "at_max_f1_threshold", "n_alerts"),
    "scenario2_recall": ("evaluation", "per_scenario_recall_at_max_f1_threshold",
                         "2", "recall"),
    "scenario3_recall": ("evaluation", "per_scenario_recall_at_max_f1_threshold",
                         "3", "recall"),
    "best_iteration": ("model", "best_iteration"),
    "n_features": ("n_features",),
    "train_runtime_s": ("cost", "train_runtime_s"),
    "test_predict_s": ("cost", "test_predict_s"),
    "model_bytes": ("cost", "model_bytes"),
}


def check_repro_gate(measured: dict, ref: dict | None = None,
                     atol: float = AUC_ATOL) -> list[str]:
    """Return the list of gate violations vs the frozen phase4-b record.

    Empty list means the gate passes. checked: auc_roc, auc_pr (atol),
    best_iteration (exact).
    """
    ref = REF_A if ref is None else ref
    problems = []
    for k in ("auc_roc", "auc_pr"):
        got = measured.get(k)
        diff = abs(float(got) - ref[k]) if got is not None else float("inf")
        if not np.isfinite(diff) or diff > atol:
            problems.append(f"{k}: measured {got} vs recorded {ref[k]} (diff {diff})")
    if measured.get("best_iteration") != ref["best_iteration"]:
        problems.append(f"best_iteration: {measured.get('best_iteration')} "
                        f"!= {ref['best_iteration']}")
    return problems


def run_arm(table: pd.DataFrame, tag: str, scenario_of_user: dict | None = None,
            params: dict | None = None, num_boost_round: int = 3000,
            early_stopping_rounds: int = 100,
            features: list[str] | None = None) -> tuple[dict, pd.DataFrame, lgb.Booster]:
    """Train + calibrate + evaluate one Phase 6 arm; TEST evaluated once.

    Returns (experiment_dict, test_predictions_df, model). `params`
    override is for fast synthetic tests only; production uses the fixed
    defaults. `features` overrides the arm's registry list (used by
    Phase 7 for a documented candidate variant); it must still be a
    registered subset in registry order.
    """
    feats = list(ARMS[tag]) if features is None else list(features)
    forbidden = set(feats) & set(FORBIDDEN)
    if forbidden:
        raise ValueError(f"{tag}: identifier/label columns in feature list: {forbidden}")
    if len(set(feats)) != len(feats):
        raise ValueError(f"{tag}: duplicate features in list")

    model, record = train_baseline(table, features=feats, params=params,
                                   num_boost_round=num_boost_round,
                                   early_stopping_rounds=early_stopping_rounds)
    data = split_datasets(table, feats)
    for name in ("train", "calibration", "test"):
        X = data[name]["X"]
        if list(X.columns) != feats:
            raise ValueError(f"{tag}/{name}: model input columns != registered features")
        if set(X.columns) & set(FORBIDDEN):
            raise ValueError(f"{tag}/{name}: identifier/label column in model input")

    start = time.time()
    scores = {k: predict_proba(model, v["X"]) for k, v in data.items()}
    predict_s = time.time() - start

    t_f1, f1_cal = th.best_f1_threshold(data["calibration"]["y"], scores["calibration"])
    t_p50 = th.threshold_at_precision(data["calibration"]["y"], scores["calibration"],
                                      min_precision=0.5)
    y_t, s_t = data["test"]["y"], scores["test"]
    ev = {
        "classification": m.classification_metrics(y_t, s_t),
        "bootstrap": {
            "auc_roc": m.bootstrap_ci(y_t, s_t, "auc_roc"),
            "auc_pr": m.bootstrap_ci(y_t, s_t, "auc_pr"),
        },
        "top_k": m.top_k_metrics(y_t, s_t),
        "at_max_f1_threshold": m.binary_decision_metrics(y_t, s_t, t_f1),
        "at_50p_precision_threshold": m.binary_decision_metrics(y_t, s_t, t_p50),
        "calibration_at_max_f1_threshold": m.binary_decision_metrics(
            data["calibration"]["y"], scores["calibration"], t_f1),
        "per_scenario_recall_at_max_f1_threshold": m.per_scenario_recall(
            y_t, s_t, scenario_of_user or {}, data["test"]["keys"], t_f1),
    }

    experiment = {
        "experiment_id": f"phase6-{tag}",
        "phase": "6",
        "dataset": "CERT r4.2",
        "feature_version": "v2 (8 behavioral + 5 graph)",
        "features": feats,
        "n_features": len(feats),
        "split": {
            "strategy": "chronological",
            "train": {"end": config.SPLIT_DEFAULTS["train_end"],
                      "rows": record["train"]["rows"],
                      "malicious": record["train"]["malicious"]},
            "calibration": {"end": config.SPLIT_DEFAULTS["calibration_end"],
                            "rows": record["calibration"]["rows"],
                            "malicious": record["calibration"]["malicious"]},
            "test": {"rows": int(len(data["test"]["X"])),
                     "malicious": int(data["test"]["y"].sum())},
        },
        "model": {
            "name": "LightGBM",
            "version": lgb.__version__,
            "params": record["params"],
            "best_iteration": record["best_iteration"],
            "early_stopping": "on calibration (auc), patience 100, up to 3000 rounds",
            "seed": record["params"]["seed"],
        },
        "calibration": {
            "method": "threshold on calibration only",
            "threshold_max_f1": t_f1,
            "threshold_50p_precision": t_p50,
            "f1_at_max_f1_threshold_calibration": f1_cal,
        },
        "evaluation": {
            "scope": "TEST only (2011-04-01..2011-05-17); evaluated once",
            "bootstrap_scope": "evaluation-only uncertainty (resample TEST rows; no refit/tune)",
            **ev,
        },
        "cost": {
            "train_runtime_s": record["runtime_s"],
            "test_predict_s": round(predict_s, 4),
            "model_bytes": None,  # filled by the runner after save_model
        },
        "feature_importance_gain": record["feature_importance_gain"],
        "notes": ("Phase 6 controlled arm: identical protocol/hyperparameters for "
                  "all arms; thresholds on calibration; TEST evaluated once."),
    }

    pred = data["test"]["keys"].copy()
    pred["is_malicious"] = data["test"]["y"]
    pred["score"] = scores["test"]
    return experiment, pred, model


def comparison_row(exp: dict) -> dict:
    """Flatten the protocol metric list for a comparison table.

    Missing entries (e.g. per-scenario recall when no scenario is present
    in the evaluation set) are recorded as None rather than raising.
    """
    row = {}
    for name, path in METRIC_KEYS.items():
        cur = exp
        try:
            for p in path:
                cur = cur[p]
            row[name] = cur
        except KeyError:
            row[name] = None
    return row