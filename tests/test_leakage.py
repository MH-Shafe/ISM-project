"""Leakage-related structural tests.

These encode the project's core leakage rules as executable checks:
  L1: features are time-local (per-user-day, no future statistics)
  L2: features never use answer-key / label information
  L3: temporal splits are chronological and leak-proof (see test_splits)
  L4: validation never modifies or drops data
"""
import pandas as pd

from src import config
from src.data.aggregation import assemble_user_day_table


def test_feature_registry_statuses_are_explicit():
    """Every behavioral feature must declare a status; nothing implicit."""
    for name, spec in config.BEHAVIORAL_FEATURES.items():
        assert spec["status"] in ("defined", "pending"), name
        assert "source" in spec and "definition" in spec, name
        assert "definition" in spec and len(spec["definition"]) > 0, name


def test_feature_registry_matches_columns(synthetic_logs, users, days):
    t = assemble_user_day_table(synthetic_logs, users, days)
    defined = [n for n, s in config.BEHAVIORAL_FEATURES.items() if s["status"] == "defined"]
    assert set(defined) <= set(t.columns)


def test_no_label_columns_in_feature_table(synthetic_logs, users, days):
    """L2: feature table must not contain answer-key-derived columns."""
    t = assemble_user_day_table(synthetic_logs, users, days)
    forbidden = {"is_malicious", "scenario", "details", "dataset", "start", "end"}
    assert not (set(t.columns) & forbidden)


def test_validation_is_read_only():
    """L4: validation must not mutate the input frame."""
    df = pd.DataFrame([{"id": "{L1}", "date": "01/04/2010 06:00:00",
                        "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"}])
    snapshot = df.copy(deep=True)
    from src.data import validation as val
    val.validate_schema(df, config.SCHEMAS["logon"])
    val.validate_timestamps(df)
    val.validate_ids(df)
    val.validate_missing(df)
    val.validate_duplicates(df)
    val.validate_allowed_values(df, "activity", {"Logon", "Logoff"})
    pd.testing.assert_frame_equal(df, snapshot)


def test_phase3_pending_features_promoted():
    """Phase 3 outcome: both foundation-pending features are now 'defined'."""
    assert config.BEHAVIORAL_FEATURES["sensitive_file_access_count"]["status"] == "defined"
    assert config.BEHAVIORAL_FEATURES["unusual_access_count"]["status"] == "defined"


def test_pending_features_are_documented():
    """Any remaining PENDING feature must keep the explicit marker."""
    pending = [n for n, s in config.BEHAVIORAL_FEATURES.items() if s["status"] == "pending"]
    for name in pending:
        assert "PENDING" in config.BEHAVIORAL_FEATURES[name]["definition"].upper()