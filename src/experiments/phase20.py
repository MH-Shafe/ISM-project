"""Phase 20: Final System Integration — deterministic decision engine (CERT r4.2).

Integrates ALL frozen scientific components into a single reproducible,
leakage-safe, analyst-facing user-day decision pipeline. Produces a
dashboard-ready output schema with ML risk, risk level, confidence,
uncertainty, graph/trust diagnostics, explanation reasons, alert flags,
and recommended actions.

CRITICAL CONSTRAINTS (Phase 20 core rules):
  - NO new scientific performance evidence is generated.
  - NO model retraining, refitting, hyperparameter/threshold tuning,
    new features, new graph stats, new calibration/conformal fitting,
    candidate search, Adaptive Risk experiments, chronological TEST
    optimization, or new ML model.
  - All components are frozen artifacts from Phases 7-19.
  - Adaptive Risk fusion does NOT appear in the production scoring path.
  - The pipeline is deterministic and bit-reproducible given the same inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen constants (single source of truth: authoritative artifacts)
# ---------------------------------------------------------------------------
FROZEN_MODEL_NAME = "lgbm-graph-v1"
FROZEN_MODEL_VERSION = "LightGBM 4.6.0"
FROZEN_SEED = 42
FROZEN_BEST_ITERATION = 186

FROZEN_FEATURES = [
    "login_count", "after_hours_login_count", "usb_connection_count",
    "file_access_count", "sensitive_file_access_count", "http_activity_count",
    "unique_device_count", "unusual_access_count",
    "device_consistency_score", "rare_device_usage_count",
    "file_type_consistency_score", "rare_file_type_access_count",
]
N_FEATURES = 12

# Thresholds (all from CALIBRATION-only selection, frozen before TEST)
FROZEN_ALERT_THRESHOLD = 0.9186015432508062      # Phase 9 frozen_max_f1
CONFORMAL_ALPHA = 0.05
CONFORMAL_T0 = 0.4634739481800199                 # class-0 inclusion threshold
CONFORMAL_T1 = 0.0046536002164601275              # class-1 inclusion threshold
CONFORMAL_N1 = 323                                # positive calibration sample
CONFORMAL_N0 = 58677                              # negative calibration sample

# Decision thresholds (Phase 12 operational envelope)
BORDERLINE_WIDTH = 0.05                           # Phase 11

# Risk level definitions — REUSES the validated Phase 11 risk levels:
#   ALERT      : score >= frozen alert threshold (Phase 9 frozen_max_f1)
#   BORDERLINE : score in monitor zone, within BORDERLINE_WIDTH of threshold
#   MONITOR    : conformal t0 <= score < alert threshold
#   NON-ALERT  : score < conformal t0
#
# These are the SAME levels defined in src/experiments/phase11.py RISK_LEVELS
# and validated in tests/test_phase11.py::test_risk_level_mapping.

# ---------------------------------------------------------------------------
# Feature definitions (from src/config.py registry)
# ---------------------------------------------------------------------------
from src import config

BEHAVIORAL_FEATURES = {
    f: config.BEHAVIORAL_FEATURES[f]
    for f in config.BEHAVIORAL_FEATURES
    if config.BEHAVIORAL_FEATURES[f]["status"] == "defined"
}
GRAPH_FEATURES = dict(config.GRAPH_FEATURES)

ALL_FEATURE_DEFS = {**BEHAVIORAL_FEATURES, **GRAPH_FEATURES}

# Graph features in the frozen set
GRAPH_FEATURES_IN_MODEL = [
    "device_consistency_score", "rare_device_usage_count",
    "file_type_consistency_score", "rare_file_type_access_count",
]

# ---------------------------------------------------------------------------
# Risk level definitions (deterministic, grounded in frozen thresholds)
# ---------------------------------------------------------------------------

def classify_risk_level(score: float) -> str:
    """Map a frozen ML risk score to a risk level using Phase 11 validated levels.

    Reuses the validated risk levels from src/experiments/phase11.py:
      ALERT      : score >= frozen alert threshold (0.9186)
      BORDERLINE : score in monitor zone, within BORDERLINE_WIDTH of threshold
      MONITOR    : conformal t0 <= score < alert threshold (0.4635 .. 0.9186)
      NON-ALERT  : score < conformal t0 (< 0.4635)
    """
    if score >= FROZEN_ALERT_THRESHOLD:
        return "ALERT"
    if score >= FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH:
        return "BORDERLINE"
    if score >= CONFORMAL_T0:
        return "MONITOR"
    return "NON-ALERT"


def risk_level_confidence(score: float, fit_conformal: dict) -> dict:
    """Compute confidence/uncertainty for a risk level assignment.

    Uses conformal p-values as the uncertainty measure:
      - p1: probability that the score belongs to class 1 (malicious)
      - p0: probability that the score belongs to class 0 (benign)
      - set: conformal prediction set at alpha=0.05
      - confidence: 1 - alpha when set is singleton, else 1 - 2*alpha for
        ambiguous set
    """
    s = np.asarray([score], dtype=float)
    # Class-1 p-value
    c1 = np.searchsorted(fit_conformal["pos_sorted"], s, side="right")[0]
    p1 = (c1 + 1) / (fit_conformal["n1"] + 1)
    # Class-0 p-value
    c0 = fit_conformal["n0"] - np.searchsorted(
        fit_conformal["neg_sorted"], s, side="left")[0]
    p0 = (c0 + 1) / (fit_conformal["n0"] + 1)

    # Prediction set
    include_1 = p1 > CONFORMAL_ALPHA
    include_0 = p0 > CONFORMAL_ALPHA
    if include_1 and not include_0:
        pred_set = "{1}"
        confidence = 1.0 - CONFORMAL_ALPHA
    elif include_0 and not include_1:
        pred_set = "{0}"
        confidence = 1.0 - CONFORMAL_ALPHA
    elif include_1 and include_0:
        pred_set = "{0,1}"
        confidence = 1.0 - 2 * CONFORMAL_ALPHA
    else:
        pred_set = "{}"
        confidence = 0.0

    return {
        "p1": round(float(p1), 6),
        "p0": round(float(p0), 6),
        "conformal_set": pred_set,
        "confidence": round(float(confidence), 4),
    }


# ---------------------------------------------------------------------------
# Graph/trust diagnostics
# ---------------------------------------------------------------------------

def compute_graph_diagnostics(row: pd.Series) -> dict:
    """Compute graph-based trust diagnostics for a user-day row.

    Uses the frozen graph features already present in the row. These are
    context signals, never inferred labels.
    """
    graph_vals = {}
    for f in GRAPH_FEATURES_IN_MODEL:
        if f in row.index:
            graph_vals[f] = float(row[f])
        else:
            graph_vals[f] = 0.0

    # Compute a simple trust summary
    n_new_devices = graph_vals.get("rare_device_usage_count", 0)
    n_new_file_types = graph_vals.get("rare_file_type_access_count", 0)
    device_consistency = graph_vals.get("device_consistency_score", 1.0)
    file_type_consistency = graph_vals.get("file_type_consistency_score", 1.0)

    # Trust flags (deterministic rules from graph diagnostics)
    trust_flags = []
    if n_new_devices > 0:
        trust_flags.append(f"new_device_count={int(n_new_devices)}")
    if n_new_file_types > 0:
        trust_flags.append(f"new_file_type_count={int(n_new_file_types)}")
    if device_consistency < 0.5:
        trust_flags.append("low_device_consistency")
    if file_type_consistency < 0.5:
        trust_flags.append("low_file_type_consistency")

    return {
        "graph_features": graph_vals,
        "trust_flags": trust_flags,
        "n_trust_flags": len(trust_flags),
    }


# ---------------------------------------------------------------------------
# Explanation reasons (from frozen SHAP/pred_contrib)
# ---------------------------------------------------------------------------

def extract_top_reasons(row: pd.Series, contributions: dict | None = None,
                        n_reasons: int = 3) -> list[dict]:
    """Extract the top-N explanation reasons for a user-day row.

    If pre-computed SHAP contributions are available, use them directly.
    Otherwise, use the raw feature values and their frozen definitions.
    """
    reasons = []

    if contributions and "contributions" in contributions:
        # Use pre-computed SHAP contributions
        contribs = contributions["contributions"]
        sorted_features = sorted(
            contribs.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feat, contrib in sorted_features[:n_reasons]:
            value = float(row.get(feat, 0))
            reasons.append({
                "feature": feat,
                "value": round(value, 6),
                "contribution": round(float(contrib), 6),
                "direction": "increased" if contrib > 0 else "decreased",
                "definition": ALL_FEATURE_DEFS.get(feat, {}).get("definition", ""),
            })
    else:
        # Use raw feature values (fallback)
        feature_vals = {}
        for f in FROZEN_FEATURES:
            if f in row.index:
                feature_vals[f] = float(row[f])
        # Sort by absolute value (highest = most anomalous)
        sorted_features = sorted(
            feature_vals.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for feat, val in sorted_features[:n_reasons]:
            reasons.append({
                "feature": feat,
                "value": round(val, 6),
                "contribution": None,
                "direction": None,
                "definition": ALL_FEATURE_DEFS.get(feat, {}).get("definition", ""),
            })

    return reasons


# ---------------------------------------------------------------------------
# Recommended action (deterministic, policy-based)
# ---------------------------------------------------------------------------

def recommend_action(risk_level: str, n_trust_flags: int,
                     score: float) -> dict:
    """Map risk level + diagnostics to a recommended analyst action.

    Actions are deterministic and grounded in the frozen policy:
      ALERT      -> escalate to incident response
      BORDERLINE -> queue for analyst review (borderline alert)
      MONITOR    -> add to watchlist, review weekly
      NON-ALERT  -> no action required
    """
    if risk_level == "ALERT":
        urgency = "immediate"
        action = "escalate_to_incident_response"
        rationale = (f"Score {score:.4f} >= frozen alert threshold "
                     f"{FROZEN_ALERT_THRESHOLD:.4f}")
    elif risk_level == "BORDERLINE":
        urgency = "within_24h"
        action = "queue_for_analyst_review"
        rationale = (f"Score {score:.4f} in borderline zone "
                     f"[{FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH:.4f}, "
                     f"{FROZEN_ALERT_THRESHOLD:.4f})")
    elif risk_level == "MONITOR":
        urgency = "weekly"
        action = "add_to_watchlist"
        rationale = (f"Score {score:.4f} in monitor zone "
                     f"[{CONFORMAL_T0:.4f}, "
                     f"{FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH:.4f})")
    else:
        urgency = "none"
        action = "no_action_required"
        rationale = (f"Score {score:.4f} < conformal threshold "
                     f"{CONFORMAL_T0:.4f}")

    if n_trust_flags > 0:
        rationale += f"; {n_trust_flags} trust flag(s) active"

    return {
        "action": action,
        "urgency": urgency,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Main decision engine
# ---------------------------------------------------------------------------

def build_decision_engine_output(
    user_day_row: pd.Series,
    ml_score: float,
    fit_conformal: dict,
    contributions: dict | None = None,
) -> dict:
    """Produce the full dashboard-ready output for a single user-day.

    Pipeline (deterministic, frozen components only):
      1. ML risk score (frozen model output)
      2. Risk level classification (frozen thresholds)
      3. Confidence/uncertainty (frozen conformal fit)
      4. Graph/trust diagnostics (frozen graph features)
      5. Explanation reasons (frozen SHAP/pred_contrib or feature values)
      6. Alert flag (deterministic from risk level)
      7. Recommended action (deterministic from risk level + diagnostics)
    """
    # 1. ML risk score (already provided)
    ml_risk = float(ml_score)

    # 2. Risk level
    risk_level = classify_risk_level(ml_risk)

    # 3. Confidence/uncertainty
    confidence_info = risk_level_confidence(ml_risk, fit_conformal)

    # 4. Graph/trust diagnostics
    graph_diag = compute_graph_diagnostics(user_day_row)

    # 5. Explanation reasons
    reasons = extract_top_reasons(user_day_row, contributions)

    # 6. Alert flag (ALERT or BORDERLINE triggers alert)
    alert_flag = risk_level in ("ALERT", "BORDERLINE")

    # 7. Recommended action
    action_info = recommend_action(
        risk_level, graph_diag["n_trust_flags"], ml_risk
    )

    return {
        "ml_risk": round(ml_risk, 6),
        "risk_level": risk_level,
        "confidence": confidence_info["confidence"],
        "conformal_set": confidence_info["conformal_set"],
        "p1": confidence_info["p1"],
        "p0": confidence_info["p0"],
        "alert_flag": alert_flag,
        "recommended_action": action_info["action"],
        "urgency": action_info["urgency"],
        "rationale": action_info["rationale"],
        "trust_flags": graph_diag["trust_flags"],
        "n_trust_flags": graph_diag["n_trust_flags"],
        "graph_features": graph_diag["graph_features"],
        "top_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_user_day_table(
    table: pd.DataFrame,
    scores: np.ndarray,
    fit_conformal: dict,
    contributions_list: list[dict] | None = None,
) -> pd.DataFrame:
    """Process an entire user-day table through the decision engine.

    Args:
        table: DataFrame with user, day, features, and optional columns
        scores: frozen ML risk scores (same length as table)
        fit_conformal: frozen conformal fit dict
        contributions_list: optional list of SHAP contribution dicts

    Returns:
        DataFrame with the full dashboard-ready output schema
    """
    records = []
    for i in range(len(table)):
        row = table.iloc[i]
        contrib = contributions_list[i] if contributions_list else None
        output = build_decision_engine_output(row, scores[i], fit_conformal, contrib)
        output["user"] = row.get("user", "")
        output["day"] = row.get("day", "")
        output["is_malicious"] = int(row.get("is_malicious", 0))
        records.append(output)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Provenance and traceability
# ---------------------------------------------------------------------------

def build_provenance_record() -> dict:
    """Build the complete provenance record for the Phase 20 output."""
    return {
        "phase": "20",
        "purpose": "Final System Integration - deterministic decision engine",
        "frozen_model": FROZEN_MODEL_NAME,
        "frozen_model_version": FROZEN_MODEL_VERSION,
        "frozen_seed": FROZEN_SEED,
        "frozen_best_iteration": FROZEN_BEST_ITERATION,
        "frozen_features": FROZEN_FEATURES,
        "frozen_n_features": N_FEATURES,
        "frozen_alert_threshold": FROZEN_ALERT_THRESHOLD,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_t0": CONFORMAL_T0,
        "conformal_t1": CONFORMAL_T1,
        "conformal_n1": CONFORMAL_N1,
        "conformal_n0": CONFORMAL_N0,
        "risk_level_thresholds": {
            "ALERT": f">= {FROZEN_ALERT_THRESHOLD}",
            "BORDERLINE": f"[{FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH}, {FROZEN_ALERT_THRESHOLD})",
            "MONITOR": f"[{CONFORMAL_T0}, {FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH})",
            "NON-ALERT": f"< {CONFORMAL_T0}",
        },
        "source_artifacts": [
            "reports/artifacts/phase7_freeze_lgbm-graph-v1.json",
            "reports/artifacts/phase9_freeze.json",
            "reports/artifacts/phase10_calibration.json",
            "reports/artifacts/phase10_fit.json",
            "reports/artifacts/phase10_decision.json",
            "reports/artifacts/phase11_experiment.json",
            "phase19_v1_1_1/kaggle_confirm_results/phase19_v1_1_2_confirm_result.json",
        ],
        "adaptive_risk_excluded": True,
        "no_retraining": True,
        "no_new_features": True,
        "no_threshold_tuning": True,
        "deterministic": True,
    }
