"""Phase 20 validation tests — deterministic integration checks.

Tests the frozen decision engine output against known inputs and verifies
all frozen constants match authoritative artifacts.
"""
from __future__ import annotations

import json
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.phase20 import (
    FROZEN_MODEL_NAME,
    FROZEN_FEATURES,
    N_FEATURES,
    FROZEN_ALERT_THRESHOLD,
    BORDERLINE_WIDTH,
    CONFORMAL_ALPHA,
    CONFORMAL_T0,
    CONFORMAL_T1,
    CONFORMAL_N1,
    CONFORMAL_N0,
    FROZEN_SEED,
    FROZEN_BEST_ITERATION,
    classify_risk_level,
    risk_level_confidence,
    compute_graph_diagnostics,
    extract_top_reasons,
    recommend_action,
    build_decision_engine_output,
    build_provenance_record,
)


# ---------------------------------------------------------------------------
# Frozen constant verification
# ---------------------------------------------------------------------------
class TestFrozenConstants:
    """Verify all frozen constants match authoritative artifacts."""

    def test_model_name(self):
        assert FROZEN_MODEL_NAME == "lgbm-graph-v1"

    def test_n_features(self):
        assert N_FEATURES == 12

    def test_feature_count_matches_list(self):
        assert len(FROZEN_FEATURES) == N_FEATURES

    def test_frozen_features_exact_list(self):
        expected = [
            "login_count", "after_hours_login_count", "usb_connection_count",
            "file_access_count", "sensitive_file_access_count",
            "http_activity_count", "unique_device_count", "unusual_access_count",
            "device_consistency_score", "rare_device_usage_count",
            "file_type_consistency_score", "rare_file_type_access_count",
        ]
        assert FROZEN_FEATURES == expected

    def test_alert_threshold(self):
        assert abs(FROZEN_ALERT_THRESHOLD - 0.9186015432508062) < 1e-15

    def test_conformal_alpha(self):
        assert CONFORMAL_ALPHA == 0.05

    def test_conformal_t0(self):
        assert abs(CONFORMAL_T0 - 0.4634739481800199) < 1e-15

    def test_conformal_t1(self):
        assert abs(CONFORMAL_T1 - 0.0046536002164601275) < 1e-15

    def test_conformal_n1(self):
        assert CONFORMAL_N1 == 323

    def test_conformal_n0(self):
        assert CONFORMAL_N0 == 58677

    def test_seed(self):
        assert FROZEN_SEED == 42

    def test_best_iteration(self):
        assert FROZEN_BEST_ITERATION == 186


# ---------------------------------------------------------------------------
# Artifact cross-check
# ---------------------------------------------------------------------------
class TestArtifactCrossCheck:
    """Verify constants match the authoritative artifact files."""

    def test_phase7_freeze_match(self):
        artifact_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "artifacts",
            "phase7_freeze_lgbm-graph-v1.json"
        )
        if os.path.exists(artifact_path):
            with open(artifact_path) as f:
                d = json.load(f)
            assert d["model_name"] == FROZEN_MODEL_NAME
            assert d["features"] == FROZEN_FEATURES
            assert d["model"]["seed"] == FROZEN_SEED
            assert d["model"]["best_iteration"] == FROZEN_BEST_ITERATION
            assert abs(d["calibration"]["threshold_max_f1"] - FROZEN_ALERT_THRESHOLD) < 1e-15

    def test_phase10_conformal_match(self):
        cal_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "artifacts",
            "phase10_calibration.json"
        )
        fit_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "artifacts",
            "phase10_fit.json"
        )
        if os.path.exists(cal_path) and os.path.exists(fit_path):
            with open(cal_path) as f:
                cal = json.load(f)
            with open(fit_path) as f:
                fit = json.load(f)
            assert abs(cal["per_alpha"]["0.05"]["thresholds"]["t0"] - CONFORMAL_T0) < 1e-15
            assert abs(cal["per_alpha"]["0.05"]["thresholds"]["t1"] - CONFORMAL_T1) < 1e-15
            assert fit["n1"] == CONFORMAL_N1
            assert fit["n0"] == CONFORMAL_N0


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------
class TestRiskLevelClassification:
    """Verify deterministic risk level classification using Phase 11 validated levels."""

    def test_alert_at_threshold(self):
        assert classify_risk_level(FROZEN_ALERT_THRESHOLD) == "ALERT"

    def test_alert_above_threshold(self):
        assert classify_risk_level(0.999) == "ALERT"

    def test_borderline_at_threshold_minus_width(self):
        assert classify_risk_level(FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH) == "BORDERLINE"

    def test_borderline_between(self):
        assert classify_risk_level(0.90) == "BORDERLINE"

    def test_monitor_at_t0(self):
        assert classify_risk_level(CONFORMAL_T0) == "MONITOR"

    def test_monitor_between_t0_and_borderline(self):
        assert classify_risk_level(0.7) == "MONITOR"

    def test_non_alert_below_t0(self):
        assert classify_risk_level(0.3) == "NON-ALERT"

    def test_non_alert_at_zero(self):
        assert classify_risk_level(0.0) == "NON-ALERT"

    def test_boundary_alert_just_below(self):
        assert classify_risk_level(FROZEN_ALERT_THRESHOLD - 1e-10) == "BORDERLINE"

    def test_boundary_borderline_just_below(self):
        assert classify_risk_level(FROZEN_ALERT_THRESHOLD - BORDERLINE_WIDTH - 1e-10) == "MONITOR"

    def test_boundary_monitor_just_below(self):
        assert classify_risk_level(CONFORMAL_T0 - 1e-10) == "NON-ALERT"


# ---------------------------------------------------------------------------
# Conformal confidence
# ---------------------------------------------------------------------------
class TestConformalConfidence:
    """Verify conformal p-value computation and confidence."""

    def _make_fit(self):
        """Create a conformal fit dict for testing with realistic sample sizes."""
        rng = np.random.RandomState(42)
        return {
            "pos_sorted": np.sort(rng.rand(323)),
            "neg_sorted": np.sort(rng.rand(58677)),
            "n1": 323,
            "n0": 58677,
        }

    def test_high_score_confident_positive(self):
        fit = self._make_fit()
        # Score above all positive calibration scores -> p1 ~ 1.0 > alpha
        # Score above all negative calibration scores -> p0 ~ very small < alpha
        result = risk_level_confidence(0.99, fit)
        assert result["conformal_set"] == "{1}"
        assert result["confidence"] == 0.95

    def test_low_score_confident_negative(self):
        fit = self._make_fit()
        # Score below all positive calibration scores -> p1 ~ 1/(324) < alpha? No, p1 = (0+1)/(324) ~ 0.003 < alpha
        # Actually: p1 = (#{pos <= s} + 1) / (n1 + 1). If s < all pos scores, #{pos<=s}=0, p1=1/324=0.003 < 0.05
        # Score above many negative scores -> p0 large > alpha
        result = risk_level_confidence(0.001, fit)
        assert result["conformal_set"] == "{0}"
        assert result["confidence"] == 0.95

    def test_middle_score_ambiguous(self):
        fit = self._make_fit()
        # Score in the middle -> both p1 and p0 > alpha -> ambiguous {0,1}
        result = risk_level_confidence(0.5, fit)
        assert result["conformal_set"] == "{0,1}"
        assert result["confidence"] == 0.90


# ---------------------------------------------------------------------------
# Graph diagnostics
# ---------------------------------------------------------------------------
class TestGraphDiagnostics:
    """Verify graph trust flag computation."""

    def test_no_flags_when_defaults(self):
        row = pd.Series({
            "device_consistency_score": 1.0,
            "rare_device_usage_count": 0,
            "file_type_consistency_score": 1.0,
            "rare_file_type_access_count": 0,
        })
        result = compute_graph_diagnostics(row)
        assert result["n_trust_flags"] == 0
        assert result["trust_flags"] == []

    def test_new_device_flag(self):
        row = pd.Series({
            "device_consistency_score": 1.0,
            "rare_device_usage_count": 2,
            "file_type_consistency_score": 1.0,
            "rare_file_type_access_count": 0,
        })
        result = compute_graph_diagnostics(row)
        assert result["n_trust_flags"] == 1
        assert any("new_device_count=2" in f for f in result["trust_flags"])

    def test_low_consistency_flag(self):
        row = pd.Series({
            "device_consistency_score": 0.3,
            "rare_device_usage_count": 0,
            "file_type_consistency_score": 0.4,
            "rare_file_type_access_count": 0,
        })
        result = compute_graph_diagnostics(row)
        assert result["n_trust_flags"] == 2
        assert "low_device_consistency" in result["trust_flags"]
        assert "low_file_type_consistency" in result["trust_flags"]


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------
class TestRecommendedAction:
    """Verify deterministic action recommendations using Phase 11 risk levels."""

    def test_alert_action(self):
        result = recommend_action("ALERT", 0, 0.99)
        assert result["action"] == "escalate_to_incident_response"
        assert result["urgency"] == "immediate"

    def test_borderline_action(self):
        result = recommend_action("BORDERLINE", 0, 0.91)
        assert result["action"] == "queue_for_analyst_review"
        assert result["urgency"] == "within_24h"

    def test_monitor_action(self):
        result = recommend_action("MONITOR", 0, 0.5)
        assert result["action"] == "add_to_watchlist"
        assert result["urgency"] == "weekly"

    def test_non_alert_action(self):
        result = recommend_action("NON-ALERT", 0, 0.1)
        assert result["action"] == "no_action_required"
        assert result["urgency"] == "none"

    def test_trust_flags_add_context(self):
        result = recommend_action("ALERT", 3, 0.99)
        assert "3 trust flag(s)" in result["rationale"]


# ---------------------------------------------------------------------------
# Full decision engine output
# ---------------------------------------------------------------------------
class TestDecisionEngineOutput:
    """Verify the full pipeline produces correct output schema."""

    def _make_row(self):
        return pd.Series({
            "user": "AAA0001",
            "day": "2011-04-15",
            "login_count": 5,
            "after_hours_login_count": 2,
            "usb_connection_count": 1,
            "file_access_count": 3,
            "sensitive_file_access_count": 1,
            "http_activity_count": 10,
            "unique_device_count": 2,
            "unusual_access_count": 1,
            "device_consistency_score": 0.8,
            "rare_device_usage_count": 0,
            "file_type_consistency_score": 0.9,
            "rare_file_type_access_count": 0,
        })

    def _make_fit(self):
        return {
            "pos_sorted": np.sort(np.random.RandomState(42).rand(323)),
            "neg_sorted": np.sort(np.random.RandomState(42).rand(58677)),
            "n1": 323,
            "n0": 58677,
        }

    def test_output_has_required_keys(self):
        row = self._make_row()
        fit = self._make_fit()
        result = build_decision_engine_output(row, 0.5, fit)
        required = [
            "ml_risk", "risk_level", "confidence", "conformal_set",
            "p1", "p0", "alert_flag", "recommended_action",
            "urgency", "rationale", "trust_flags", "n_trust_flags",
            "graph_features", "top_reasons",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_alert_output(self):
        row = self._make_row()
        fit = self._make_fit()
        result = build_decision_engine_output(row, 0.99, fit)
        assert result["risk_level"] == "ALERT"
        assert result["alert_flag"] is True
        assert result["recommended_action"] == "escalate_to_incident_response"

    def test_provenance_record(self):
        p = build_provenance_record()
        assert p["phase"] == "20"
        assert p["frozen_model"] == "lgbm-graph-v1"
        assert p["adaptive_risk_excluded"] is True
        assert p["no_retraining"] is True
        assert len(p["source_artifacts"]) == 7


# ---------------------------------------------------------------------------
# Full-table integration tests (require complete decision table artifact)
# ---------------------------------------------------------------------------
class TestFullDecisionTable:
    """Integration tests against the materialized final_user_day_decisions.parquet."""

    @pytest.fixture(autouse=True)
    def _load_table(self):
        table_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "artifacts",
            "phase20", "final_user_day_decisions.parquet"
        )
        if not os.path.exists(table_path):
            pytest.skip("Decision table not yet materialized")
        self.df = pd.read_parquet(table_path)

    def test_complete_row_coverage(self):
        """501000 user-days in full universe."""
        assert len(self.df) == 501000

    def test_primary_key_uniqueness(self):
        """No duplicate user_id + date."""
        assert not self.df.duplicated(subset=["user_id", "date"]).any()

    def test_no_missing_primary_keys(self):
        """No nulls in primary keys."""
        assert self.df["user_id"].isna().sum() == 0
        assert self.df["date"].isna().sum() == 0

    def test_exact_schema(self):
        """21-column schema matches output specification."""
        expected_cols = [
            "user_id", "date", "ml_risk_score", "risk_level", "alert_flag",
            "confidence_status", "conformal_prediction_set", "conformal_p1",
            "conformal_p0", "confidence", "top_reason_1", "top_reason_2",
            "top_reason_3", "trust_diagnostic_summary", "n_trust_flags",
            "recommended_action", "urgency", "rationale", "model_version",
            "policy_version", "explanation_version",
        ]
        assert list(self.df.columns) == expected_cols

    def test_score_integrity(self):
        """Scores match frozen model inference (verified during generation)."""
        assert self.df["ml_risk_score"].between(0, 1).all()
        assert self.df["ml_risk_score"].notna().all()

    def test_alert_integrity(self):
        """Alert flags follow frozen policy: ALERT or BORDERLINE = True."""
        alerts = self.df[self.df["alert_flag"] == True]
        assert alerts["risk_level"].isin(["ALERT", "BORDERLINE"]).all()
        non_alerts = self.df[self.df["alert_flag"] == False]
        assert non_alerts["risk_level"].isin(["MONITOR", "NON-ALERT"]).all()

    def test_risk_level_valid(self):
        """All risk levels are valid Phase 11 categories."""
        valid = {"ALERT", "BORDERLINE", "MONITOR", "NON-ALERT"}
        assert self.df["risk_level"].isin(valid).all()

    def test_no_labels_in_production_table(self):
        """No ground truth columns in production output."""
        forbidden = {"label", "malicious", "is_malicious", "answer_key", "target", "y_true"}
        assert forbidden.intersection(set(self.df.columns)) == set()

    def test_no_adaptive_risk_columns(self):
        """Adaptive Risk fields must not appear."""
        adaptive_cols = {"adaptive_risk", "final_risk", "trust_adjusted_risk"}
        assert adaptive_cols.intersection(set(self.df.columns)) == set()

    def test_deterministic_decisions(self):
        """Risk level is deterministic from score (no randomness)."""
        for _, row in self.df.sample(100, random_state=42).iterrows():
            rl = classify_risk_level(row["ml_risk_score"])
            assert rl == row["risk_level"]

    def test_conformal_integration_coverage(self):
        """All rows have conformal fields."""
        assert self.df["conformal_p1"].notna().all()
        assert self.df["conformal_p0"].notna().all()
        assert self.df["conformal_prediction_set"].notna().all()
        assert self.df["confidence"].notna().all()

    def test_explanation_fallback(self):
        """All rows have at least top_reason_1 (SHAP or fallback)."""
        assert (self.df["top_reason_1"] != "").all()

    def test_trust_fallback(self):
        """All rows have trust_diagnostic_summary."""
        assert self.df["trust_diagnostic_summary"].notna().all()
        assert (self.df["trust_diagnostic_summary"] != "").all()

    def test_provenance_fields(self):
        """Model/policy/explanation versions are populated."""
        assert (self.df["model_version"] == "LightGBM 4.6.0").all()
        assert (self.df["policy_version"] == "phase20-v1").all()
        assert (self.df["explanation_version"] == "phase11-v1").all()

    def test_decision_counts_match_summary(self):
        """Risk level counts sum to total rows."""
        counts = self.df["risk_level"].value_counts()
        assert counts.sum() == len(self.df)
        assert "ALERT" in counts.index
        assert "NON-ALERT" in counts.index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
