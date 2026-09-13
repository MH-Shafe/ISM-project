"""Phase 8: adaptive risk layer -- controlled, leakage-safe evaluation (CERT r4.2).

Research question: does an adaptive risk score provide useful information
beyond the frozen LightGBM prediction (`lgbm-graph-v1`), especially for
practical alert prioritization and operating-point decisions?

Formulation (only components that can be defined from validated data without
leakage; see COMPONENTS for per-component justification):

    Final Risk = alpha * ML Risk + beta * Trust Risk + gamma * Behavioral Risk

No "context risk" component: the only cross-user context feature
(department_file_type_mismatch_count) was REJECTED in Phase 7, and no other
context signal survived the availability check -- documented in COMPONENTS.

Protocol (identical to Phases 6/7):
  - chronological split (config.SPLIT_DEFAULTS), TRAIN/CALIBRATION only for
    all fitting (normalization, weights, thresholds); TEST exactly once after
    the adaptive configuration is frozen
  - arm A (ML risk only) must reproduce the Phase 7 freeze record bit-for-bit
  - weights: equal (arm B) as the transparent baseline; learned (arm C) via a
    documented simplex grid on TRAIN with selection on CALIBRATION
  - the frozen LightGBM model is never retrained or modified
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th

# ---------------------------------------------------------------------------
# Component registry (single source of truth for Phase 8 definitions)
# ---------------------------------------------------------------------------
ML_RISK = "ml_risk"
TRUST_RISK = "trust_risk"
BEHAVIOR_RISK = "behavior_risk"

COMPONENT_ORDER = [ML_RISK, TRUST_RISK, BEHAVIOR_RISK]

# Trust risk semantics (all four are validated Phase 5 graph features):
#   - device_consistency_score       higher = more consistent   -> lower risk
#   - rare_device_usage_count        higher = more novel PCs    -> higher risk
#   - file_type_consistency_score    higher = more consistent   -> lower risk
#   - rare_file_type_access_count    higher = more novel types  -> higher risk
TRUST_CONSISTENCY_FEATURES = ["device_consistency_score",
                              "file_type_consistency_score"]
TRUST_RARITY_FEATURES = ["rare_device_usage_count",
                         "rare_file_type_access_count"]

# Behavioral risk: only features with a documented, direction-specific risk
# semantics and low redundancy with the frozen model's dominant signal.
# usb_connection_count is the model's top feature (gain share ~0.54) and is
# therefore redundant by construction; pure volume features (login_count,
# file_access_count, http_activity_count, unique_device_count) carry no
# a-priori risk direction. The three selected features:
#   - after_hours_login_count         work outside 8-18h (CERT significance)
#   - unusual_access_count            first-time PC use (28-day past window)
#   - sensitive_file_access_count     declared sensitive document handling
BEHAVIOR_RISK_FEATURES = ["after_hours_login_count", "unusual_access_count",
                          "sensitive_file_access_count"]

# Empirical-CDF normalization grid (TRAIN-only fit; deterministic).
CDF_GRID_STEP = 0.001

# Weight learning (arm C): simplex grid, selection on CALIBRATION.
WEIGHT_GRID_STEP = 0.1
WEIGHT_TOP_K = 5

# Decision rule margins (CALIBRATION only; declared a priori).
ADOPT_MARGIN_AUC_PR = 0.01      # CAL AUC-PR gain required to ADOPT
INCONCLUSIVE_MARGIN_AUC_PR = 0.005
ML_DEGENERATE_WEIGHT = 0.9      # learned ml_risk weight >= this => degenerate

COMPONENTS = {
    ML_RISK: {
        "raw_features": None,
        "definition": ("Probability P(y=1) from the frozen lgbm-graph-v1 model "
                       "(immutable; never retrained in this phase)."),
        "direction": "higher = higher risk",
        "normalization": "none (probability already in [0,1])",
        "range": "[0,1]",
        "missing_behavior": "frozen model always produces a probability",
        "unseen_behavior": "none (model input is per-user-day features)",
        "temporal_availability": "end of day d (all model features are "
                                 "end-of-day aggregates)",
        "leakage_status": "none: model was fit on TRAIN with early stopping on "
                          "CALIBRATION; scores on CALIBRATION are out-of-fit",
    },
    TRUST_RISK: {
        "raw_features": TRUST_CONSISTENCY_FEATURES + TRUST_RARITY_FEATURES,
        "definition": ("1 - Trust Score; Trust Score = 0.25*(device_consistency)"
                       " + 0.25*(1 - CDF(rare_device)) + 0.25*(file_type_consistency)"
                       " + 0.25*(1 - CDF(rare_file_type)); equal semantic weights "
                       "inside the trust block (no evidence for unequal ones)."),
        "direction": "higher = lower trust = higher risk",
        "normalization": "Jaccard scores used as-is; counts via TRAIN empirical "
                         "CDF (quantile grid, step 0.001)",
        "range": "[0,1]",
        "missing_behavior": ("graph features are 0 when the user has no past "
                             "history; treated as low consistency / high trust "
                             "risk (conservative, documented in Phase 5)"),
        "unseen_behavior": ("normalization is a fixed TRAIN-fitted mapping; "
                            "unseen users' values map through the same CDF and "
                            "clip at [0,1]"),
        "temporal_availability": "end of day d (strictly-past feature values)",
        "leakage_status": "none: all four features passed Phase 5 "
                          "future-invariance checks; CDF fitted on TRAIN only",
    },
    BEHAVIOR_RISK: {
        "raw_features": BEHAVIOR_RISK_FEATURES,
        "definition": ("mean of CDF(after_hours_login_count), "
                       "CDF(unusual_access_count), CDF(sensitive_file_access_count); "
                       "equal semantic weights (no evidence for unequal ones)."),
        "direction": "higher = more anomalous work behavior = higher risk",
        "normalization": "TRAIN empirical CDF (quantile grid, step 0.001)",
        "range": "[0,1]",
        "missing_behavior": "counts are 0 when no events (CDF(0) >= 0)",
        "unseen_behavior": "fixed TRAIN-fitted mapping; clips at [0,1]",
        "temporal_availability": "end of day d (per-user-day counts; "
                                 "unusual_access uses the strictly-past 28-day window)",
        "leakage_status": "none: registry features with documented windows; "
                          "CDF fitted on TRAIN only",
    },
}


# ---------------------------------------------------------------------------
# Normalization (empirical CDF on TRAIN only)
# ---------------------------------------------------------------------------
def fit_empirical_cdf(train_values) -> dict:
    """Fit a monotone empirical-CDF map on TRAIN values (quantile grid).

    The fitted object is a deterministic, serializable parameter dict:
    {method, step, quantiles, values} where values[q] = TRAIN quantile(q).
    """
    x = np.asarray(train_values, dtype=float)
    if x.size == 0:
        raise ValueError("cannot fit CDF on an empty population")
    quantiles = np.arange(0.0, 1.0 + 1e-9, CDF_GRID_STEP)
    vals = np.quantile(x, quantiles)
    return {
        "method": "empirical CDF via TRAIN-only quantile grid",
        "step": CDF_GRID_STEP,
        "quantiles": quantiles.tolist(),
        "values": vals.tolist(),
    }


def cdf_transform(params: dict, values) -> np.ndarray:
    """Apply a fitted CDF mapping; monotone, clips to [0,1], deterministic."""
    q = np.asarray(params["quantiles"], dtype=float)
    v = np.asarray(params["values"], dtype=float)
    x = np.asarray(values, dtype=float)
    return np.interp(x, v, q, left=0.0, right=1.0)


def fit_normalization(train: pd.DataFrame) -> dict:
    """Fit all component normalizers on the TRAIN block (nothing else)."""
    fitted = {}
    for feat in TRUST_RARITY_FEATURES + BEHAVIOR_RISK_FEATURES:
        if feat not in train.columns:
            raise ValueError(f"normalization feature missing from TRAIN block: {feat}")
        fitted[feat] = fit_empirical_cdf(train[feat])
    return fitted


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def build_components(ml_scores, block: pd.DataFrame,
                     norm: dict) -> dict[str, np.ndarray]:
    """Build the three risk components for one data block.

    `norm` must come from fit_normalization(TRAIN) -- the runner asserts the
    fitted dict is reused verbatim for every block (never refit).
    Only registered feature columns and the frozen model's scores enter.
    """
    missing = [f for f in (TRUST_CONSISTENCY_FEATURES + TRUST_RARITY_FEATURES
                           + BEHAVIOR_RISK_FEATURES) if f not in block.columns]
    if missing:
        raise ValueError(f"component features missing from block: {missing}")
    forbidden = set(block.columns) & {"user", "day", "is_malicious"}
    if forbidden:
        raise ValueError(f"identifier/label columns in component block: {forbidden}")

    trust = 0.0 * np.ones(len(block))
    for f in TRUST_CONSISTENCY_FEATURES:
        trust += (1.0 - block[f].to_numpy(dtype=float)) * 0.25
    for f in TRUST_RARITY_FEATURES:
        trust += cdf_transform(norm[f], block[f]) * 0.25

    behavior = np.zeros(len(block))
    for f in BEHAVIOR_RISK_FEATURES:
        behavior += cdf_transform(norm[f], block[f])
    behavior = behavior / len(BEHAVIOR_RISK_FEATURES)

    return {
        ML_RISK: np.asarray(ml_scores, dtype=float),
        TRUST_RISK: trust,
        BEHAVIOR_RISK: behavior,
    }


def risk_score(comp: dict[str, np.ndarray], weights) -> np.ndarray:
    """Weighted sum of components (weights in COMPONENT_ORDER)."""
    w = np.asarray(weights, dtype=float)
    if len(w) != len(COMPONENT_ORDER):
        raise ValueError(f"expected {len(COMPONENT_ORDER)} weights, got {len(w)}")
    out = np.zeros(len(comp[COMPONENT_ORDER[0]]))
    for name, wi in zip(COMPONENT_ORDER, w):
        out += wi * comp[name]
    return out


# ---------------------------------------------------------------------------
# Weight learning (arm C): simplex grid on TRAIN, selection on CALIBRATION
# ---------------------------------------------------------------------------
def simplex_grid(n: int = len(COMPONENT_ORDER), step: float = WEIGHT_GRID_STEP):
    """All non-negative weights summing to 1 on a step grid (deterministic)."""
    m = int(round(1.0 / step))
    out = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            k = m - i - j
            out.append((round(i * step, 10), round(j * step, 10),
                        round(k * step, 10)))
    return out


def learn_weights(train_comp, y_train, cal_comp, y_cal,
                  step: float = WEIGHT_GRID_STEP,
                  top_k: int = WEIGHT_TOP_K) -> dict:
    """Learned non-negative sum-to-1 weights (TRAIN grid, CAL selection).

    Selection: of the top-k grid points by TRAIN AUC-PR, pick the one with the
    best CALIBRATION AUC-PR. Deterministic; no TEST involvement.
    """
    grid = simplex_grid(len(COMPONENT_ORDER), step)
    rows = []
    for w in grid:
        st = risk_score(train_comp, w)
        sc = risk_score(cal_comp, w)
        tm = m.classification_metrics(y_train, st)
        cm = m.classification_metrics(y_cal, sc)
        rows.append({"weights": list(w),
                     "train_auc_roc": tm["auc_roc"],
                     "train_auc_pr": tm["auc_pr"],
                     "cal_auc_roc": cm["auc_roc"],
                     "cal_auc_pr": cm["auc_pr"]})
    rows.sort(key=lambda r: (-r["train_auc_pr"], tuple(r["weights"])))
    top = rows[:top_k]
    best = max(top, key=lambda r: (r["cal_auc_pr"], tuple(r["weights"])))
    return {
        "method": ("simplex grid (non-negative, sum-to-1) on TRAIN; selection: "
                   f"best CALIBRATION AUC-PR among top-{top_k} by TRAIN AUC-PR"),
        "grid_step": step,
        "top_k": top_k,
        "grid_size": len(rows),
        "weights": list(best["weights"]),
        "train_auc_roc": best["train_auc_roc"],
        "train_auc_pr": best["train_auc_pr"],
        "cal_auc_roc": best["cal_auc_roc"],
        "cal_auc_pr": best["cal_auc_pr"],
        "top_candidates": top,
    }


# ---------------------------------------------------------------------------
# Decision rule (CALIBRATION only, deterministic, declared margins)
# ---------------------------------------------------------------------------
def decide_adaptive(cal: dict, weights_c: list) -> str:
    """Documented decision rule.

    cal: {"arm_a": {"auc_roc", "auc_pr", "precision_at_10"},
          "arm_b": {...}, "arm_c": {...}} -- CALIBRATION metrics only.
    weights_c: learned arm-C weights (COMPONENT_ORDER).

    Best adaptive arm = B or C by CAL AUC-PR.
    - ADOPT if CAL AUC-PR(best) - CAL AUC-PR(A) >= 0.01 AND
      CAL AUC-ROC(best) >= CAL AUC-ROC(A) AND CAL P@10(best) >= CAL P@10(A)
      AND the learned arm-C ml_risk weight < 0.9 (not degenerate).
    - INCONCLUSIVE if the AUC-PR gain is in [0.005, 0.01) (evidence
      insufficient; do not adopt).
    - NO otherwise (adaptive risk does not earn its complexity).
    """
    best = max(["arm_b", "arm_c"], key=lambda k: cal[k]["auc_pr"])
    gain = cal[best]["auc_pr"] - cal["arm_a"]["auc_pr"]
    degenerate = weights_c[COMPONENT_ORDER.index(ML_RISK)] >= ML_DEGENERATE_WEIGHT
    if gain >= ADOPT_MARGIN_AUC_PR and not degenerate:
        if (cal[best]["auc_roc"] >= cal["arm_a"]["auc_roc"]
                and cal[best]["precision_at_10"] >= cal["arm_a"]["precision_at_10"]):
            return "ADOPT"
        return "NO"
    if gain >= INCONCLUSIVE_MARGIN_AUC_PR and not degenerate:
        return "INCONCLUSIVE"
    return "NO"


# ---------------------------------------------------------------------------
# Correlation / redundancy bundle (TRAIN/CALIBRATION only)
# ---------------------------------------------------------------------------
def correlation_bundle(train_comp, y_train, cal_comp, y_cal) -> dict:
    """Spearman relationships + single-component and incremental AUCs.

    All on TRAIN and CALIBRATION blocks; TEST never enters.
    """
    def block_stats(comp, y):
        df = pd.DataFrame({k: comp[k] for k in COMPONENT_ORDER})
        df["target"] = y
        corr = df.corr(method="spearman")
        single = {}
        for k in COMPONENT_ORDER:
            cm = m.classification_metrics(y, comp[k])
            single[k] = {"auc_roc": cm["auc_roc"], "auc_pr": cm["auc_pr"]}
        return {"spearman": corr.round(6).to_dict(), "single_auc": single}

    def incremental(comp, y):
        ml = comp[ML_RISK]
        trust = comp[TRUST_RISK]
        beh = comp[BEHAVIOR_RISK]
        out = {}
        for name, s in [
            ("ml", ml),
            ("ml_trust_equal", 0.5 * ml + 0.5 * trust),
            ("ml_behavior_equal", 0.5 * ml + 0.5 * beh),
            ("ml_trust_behavior_equal", (ml + trust + beh) / 3.0),
        ]:
            cm = m.classification_metrics(y, s)
            out[name] = {"auc_roc": cm["auc_roc"], "auc_pr": cm["auc_pr"]}
        return out

    return {
        "scope": "TRAIN + CALIBRATION only; TEST not consulted",
        "train": block_stats(train_comp, y_train),
        "calibration": block_stats(cal_comp, y_cal),
        "incremental_calibration": incremental(cal_comp, y_cal),
        "note": ("incremental = equal-weight combinations on CALIBRATION; "
                 "shows the marginal contribution of each component beyond ML"),
    }