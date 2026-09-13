"""Leakage-safe LightGBM baseline trainer (user x day).

Follows the project protocol in .opencode/instructions/ism-principles.md:
  - features come only from the registry (status == "defined")
  - TRAIN fits the model, CALIBRATION is used for early stopping
    (model selection) and threshold selection, TEST is never touched
  - no identifiers/dates/labels are used as predictive features

The feature list is enforced structurally: any extra numeric column in
the table raises an error rather than silently entering the model.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.preprocessing.splits import split_user_day_table


def model_features(table: pd.DataFrame) -> list[str]:
    """The exact model input list: registry features with status 'defined'."""
    feats = [c for c in config.BEHAVIORAL_FEATURES
             if config.BEHAVIORAL_FEATURES[c]["status"] == "defined"]
    missing = [f for f in feats if f not in table.columns]
    if missing:
        raise ValueError(f"defined feature columns missing from table: {missing}")
    extra = [c for c in table.columns
             if c not in ("user", "day", "is_malicious") + tuple(feats)]
    if extra:
        raise ValueError(
            f"unexpected columns in modeling table (possible leakage): {extra}")
    return feats


def default_params() -> dict:
    """Fixed baseline hyperparameters (explicit, reproducible).

    learning_rate 0.03 was selected over 0.05 during development: with
    lr 0.05 + patience 50 the round-1 split dominated the calibration
    AUC, early stopping collapsed to a single tree, and scores saturated
    at 0.0/1.0 making top-k metrics degenerate. lr 0.03 trains a stable
    multi-tree model whose AUC peaks well inside the patience window.
    Selection used TRAIN/CALIBRATION only (never TEST).
    """
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }


def split_datasets(table: pd.DataFrame, feats: list[str]) -> dict:
    """Chronological TRAIN/CALIBRATION/TEST split of the modeling table."""
    parts = split_user_day_table(table, config.SPLIT_DEFAULTS["train_end"],
                                 config.SPLIT_DEFAULTS["calibration_end"])
    out = {}
    for name, part in parts.items():
        out[name] = {
            "X": part[feats].astype(np.float32),
            "y": part["is_malicious"].to_numpy(),
            "keys": part[["user", "day"]].copy(),
        }
    return out


def train_baseline(table: pd.DataFrame, params: dict | None = None,
                   num_boost_round: int = 3000,
                   early_stopping_rounds: int = 100,
                   features: list[str] | None = None) -> tuple[lgb.Booster, dict]:
    """Fit LightGBM on TRAIN with early stopping on CALIBRATION.

    features: explicit feature subset (default: all registry 'defined'
    features). Subsets must be registered -- defined behavioral features
    (config.BEHAVIORAL_FEATURES, status 'defined') or any Phase 5 graph
    feature (config.GRAPH_FEATURES) -- and present in the table.

    scale_pos_weight is derived from TRAIN labels only (leakage-safe).
    Returns (model, record) where record holds the full experiment config.
    """
    feats = model_features(table) if features is None else list(features)
    registered = {c for c, s in config.BEHAVIORAL_FEATURES.items()
                  if s["status"] == "defined"} | set(config.GRAPH_FEATURES)
    unknown = [f for f in feats if f not in registered]
    if unknown:
        raise ValueError(f"features not registry-defined: {unknown}")
    missing = [f for f in feats if f not in table.columns]
    if missing:
        raise ValueError(f"features missing from table: {missing}")
    data = split_datasets(table, feats)
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    Xcal, ycal = data["calibration"]["X"], data["calibration"]["y"]

    params = dict(default_params(), **(params or {}))
    params["scale_pos_weight"] = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    dtr = lgb.Dataset(Xtr, label=ytr)
    dcal = lgb.Dataset(Xcal, label=ycal, reference=dtr)
    start = time.time()
    model = lgb.train(
        params, dtr, num_boost_round=num_boost_round,
        valid_sets=[dcal],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )
    runtime_s = time.time() - start

    importance = dict(zip(feats, model.feature_importance("gain").tolist()))
    record = {
        "features": feats,
        "params": params,
        "best_iteration": int(model.best_iteration),
        "runtime_s": round(runtime_s, 2),
        "train": {"rows": int(len(Xtr)), "malicious": int(ytr.sum())},
        "calibration": {"rows": int(len(Xcal)), "malicious": int(ycal.sum())},
        "feature_importance_gain": importance,
    }
    return model, record


def predict_proba(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities (LightGBM binary)."""
    return model.predict(X, num_iteration=model.best_iteration)