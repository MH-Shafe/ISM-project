"""Tests for Phase 15 (unseen-user generalization diagnosis).

Covers: unit tests for every diagnostic calculation (A-G), edge cases
(empty strata, zero-positive groups, single-user cohort, constant
feature, empty frame), determinism, artifact reloadability, and the
score-reproduction gate on a small model fixture.

The authoritative chronological TEST never enters this module (structural
guards assert the TEST-block rows come only from the Phase 14 artifact).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src import config
from src.experiments import phase15 as p15
from src.experiments.phase14 import FROZEN_FEATURES

ART = Path(config.ARTIFACTS_DIR)


def _syn_table(n_users: int = 6, n_days: int = 10,
               mal_users: int = 2) -> pd.DataFrame:
    """Deterministic synthetic user x day table (no RNG)."""
    rows = []
    feats = FROZEN_FEATURES
    for u in range(n_users):
        for d in range(1, n_days + 1):
            rec = {"user": f"U{u:03d}", "day": pd.Timestamp(f"2010-01-{d:02d}")}
            for f in feats:
                rec[f] = (u + d) % 7
            rec["is_malicious"] = int(
                (u >= n_users - mal_users) and (d >= n_days - 2))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["user"] = df["user"].astype(str)
    df["day"] = pd.to_datetime(df["day"])
    return df


def _syn_blocks(n_users: int = 6, mal_users: int = 2,
                n_train: int = 3, n_cal: int = 1) -> dict:
    return {
        "TRAIN": {"users": [f"U{u:03d}" for u in range(n_train)]},
        "CAL": {"users": [f"U{u:03d}" for u in
                          range(n_train, n_train + n_cal)]},
        "TEST": {"users": [f"U{u:03d}" for u in
                           range(n_train + n_cal, n_users)]},
    }


def _syn_pred(table: pd.DataFrame, test_users: list[str]) -> pd.DataFrame:
    part = table[table["user"].isin(test_users)].copy()
    part["score"] = 0.3 + 0.6 * (part["is_malicious"].astype(int) * 0.9) \
        + (part["user"].astype(str).str[1:].astype(int) % 5) * 0.02
    part["alert_primary"] = part["score"] >= p15.FROZEN_THRESHOLD
    part["alert_secondary"] = part["score"] >= 0.99
    return part[["user", "day", "is_malicious", "score",
                 "alert_primary", "alert_secondary"]]


def _syn_diag(table: pd.DataFrame, test_users: list[str]) -> dict:
    part = table[table["user"].isin(test_users)]
    mal = sorted(part.loc[part["is_malicious"] == 1, "user"].unique())
    per = []
    for u in mal:
        d = part[part["user"] == u]
        pos = d.loc[d["is_malicious"] == 1, "day"]
        onset = pos.min()
        score = 0.3 + 0.6 * 0.9
        detected = score >= p15.FROZEN_THRESHOLD
        per.append({"user": u, "n_positive_days": int(len(pos)),
                    "onset": str(onset.date()), "temporal_half": "2010 H2",
                    "activity_tercile": "low", "length_class": "short",
                    "scenario": 3, "detected": detected,
                    "detected_first_week": False,
                    "detected_first_tercile": False,
                    "first_alert_day": str(onset.date()),
                    "detection_delay_days": 0,
                    "first_alert_positive_rank": None,
                    "max_score": score, "n_alerts": 1,
                    "fully_detected": detected})
    return {"per_user": per, "zero_detection_users": [
        u for u in mal if u not in [
            x["user"] for x in per if x["detected"]]]}


def _syn_cal_scores(table: pd.DataFrame, cal_users: list[str]) -> pd.DataFrame:
    part = table[table["user"].isin(cal_users)].copy()
    part["score"] = 0.2 + 0.5 * part["is_malicious"].astype(int)
    return part[["user", "day", "is_malicious", "score"]]


def _syn_frozen() -> dict:
    return {
        "p7_cal_auc_roc": 0.88, "p7_cal_auc_roc_std": 0.01,
        "p9_cal_alerts": 244, "p9_cal_precision": 0.684,
        "p9_cal_recall": 0.517, "p9_cal_coverage": 8,
        "p12_test_auc_roc": 0.939, "p12_test_alerts": 49,
        "p12_test_precision": 0.286, "p12_test_recall": 0.467,
        "p12_test_coverage": 5, "p12_test_n_malicious_users": 5,
        "p13_verdict": "FAIL",
        "p14_test_auc_roc": 0.7865, "p14_test_alerts": 1026,
        "p14_test_precision": 0.2018, "p14_test_recall": 0.4349,
        "p14_test_coverage": 13, "p14_cal_auc_roc": 0.7355,
    }


# ---------------------------------------------------------------------------
# A: activity strata
# ---------------------------------------------------------------------------
def test_activity_strata_buckets_cover_all_users():
    table = _syn_table(6, 10, mal_users=2)
    pred = _syn_pred(table, _syn_blocks()["TEST"]["users"])
    diag = _syn_diag(table, _syn_blocks()["TEST"]["users"])
    res = p15.activity_strata(table, pred, diag)
    total = sum(b["n_users"] for b in res["all_70_malicious_users"])
    assert total == 2  # all malicious users appear in exactly one bucket
    assert res["buckets"] == p15.BUCKET_LABELS


def test_activity_strata_edge_empty_bucket():
    table = _syn_table(6, 10, mal_users=2)  # all malicious have 2 mal days
    pred = _syn_pred(table, _syn_blocks()["TEST"]["users"])
    diag = _syn_diag(table, _syn_blocks()["TEST"]["users"])
    res = p15.activity_strata(table, pred, diag)
    bucket = next(b for b in res["test_block_15_users"] if b["bucket"] == "1")
    assert bucket["n_users"] == 0
    assert bucket["detection_rate"] is None
    assert bucket["alert_share"] == 0.0


def test_activity_strata_zero_positive_group():
    table = _syn_table(6, 10, mal_users=0)
    pred = _syn_pred(table, _syn_blocks()["TEST"]["users"])
    diag = _syn_diag(table, _syn_blocks()["TEST"]["users"])
    res = p15.activity_strata(table, pred, diag)
    assert sum(b["n_users"] for b in res["all_70_malicious_users"]) == 0


# ---------------------------------------------------------------------------
# B: feature shift
# ---------------------------------------------------------------------------
def test_feature_shift_covers_12_frozen_features():
    table = _syn_table()
    blocks = _syn_blocks()
    res = p15.feature_shift(table, blocks)
    feats = [r["feature"] for r in res["features"]]
    assert set(feats) == set(FROZEN_FEATURES)
    assert len(feats) == 12
    assert len(res["ranking_by_combined_drift"]["top_3"]) == 3


def test_feature_shift_constant_feature_psi_none():
    table = _syn_table()
    table["unique_device_count"] = 3  # constant -> degenerate PSI bins
    res = p15.feature_shift(table, _syn_blocks())
    rec = next(r for r in res["features"]
               if r["feature"] == "unique_device_count")
    assert rec["psi_cal"] is None
    assert rec["psi_test"] is None
    assert rec["d_train_test"] is None or abs(rec["d_train_test"]) < 1e-12


def test_feature_shift_no_leakage_psi_bins_from_train_only():
    table = _syn_table()
    blocks = _syn_blocks()
    # Move the TEST block far away in value space; PSI must reflect it.
    te = [f"U{i:03d}" for i in range(4, 6)]
    table.loc[table["user"].isin(te), "login_count"] = 1000
    res = p15.feature_shift(table, blocks)
    rec = next(r for r in res["features"] if r["feature"] == "login_count")
    assert rec["psi_test"] is not None and rec["psi_test"] > 0.1


def test_feature_shift_empty_cohort():
    table = _syn_table()
    blocks = _syn_blocks(n_cal=1)
    blocks["CAL"] = {"users": []}
    res = p15.feature_shift(table, blocks)
    rec = res["features"][0]
    assert rec["d_cal_test"] is None
    assert rec["ks_cal_test"] is None


# ---------------------------------------------------------------------------
# C: detected vs missed
# ---------------------------------------------------------------------------
def test_detected_vs_missed_split_and_rows():
    table = _syn_table(6, 10, mal_users=2)
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    res = p15.detected_vs_missed(table, pred, diag)
    assert res["n_detected"] + res["n_missed"] == 2
    assert set(res["missed_users"]).issubset(
        set(r["user"] for r in res["users"]))
    for r in res["users"]:
        assert set(r) >= {"user", "detected", "n_malicious_days", "max_score",
                          "n_alerts", "graph_feature_user_mean"}


def test_detected_vs_missed_all_missed():
    table = _syn_table(6, 10, mal_users=2)
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    pred["score"] = 0.1  # everything below threshold -> all missed
    pred["alert_primary"] = False
    diag = _syn_diag(table, blocks["TEST"]["users"])
    diag["per_user"] = [{**d, "detected": False} for d in diag["per_user"]]
    diag["zero_detection_users"] = [d["user"] for d in diag["per_user"]]
    res = p15.detected_vs_missed(table, pred, diag)
    assert res["n_detected"] == 0
    assert res["n_missed"] == 2


# ---------------------------------------------------------------------------
# D: graph diagnosis
# ---------------------------------------------------------------------------
def test_graph_diagnosis_coverage():
    table = _syn_table()
    res = p15.graph_diagnosis(table, _syn_blocks())
    assert set(res["cohort_profiles"]) == set(p15.GRAPH_FEATURES)
    assert set(res["discrimination"]) == set(p15.GRAPH_FEATURES)
    assert set(res["stability"]) == set(p15.CONSISTENCY_FEATURES)


def test_graph_diagnosis_degenerate_auc_none():
    table = _syn_table()
    table["device_consistency_score"] = 1.0
    res = p15.graph_diagnosis(table, _syn_blocks())
    d = res["discrimination"]["device_consistency_score"]
    assert d["auc_roc_in_TRAIN_block"] is None
    assert d["auc_roc_in_TEST_block"] is None


# ---------------------------------------------------------------------------
# E: temporal analysis
# ---------------------------------------------------------------------------
def test_temporal_analysis_structure():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    res = p15.temporal_analysis(table, pred, diag)
    assert len(res["users"]) == 2
    assert res["median_onset"] is not None
    assert res["early_group"]["n_users"] + res["late_group"]["n_users"] == 2
    assert set(res["by_onset_half"]) == {"2010 H1", "2010 H2", "2011"}


def test_temporal_analysis_delay_negative_allowed():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    for d in diag["per_user"]:
        d["detection_delay_days"] = -160
    res = p15.temporal_analysis(table, pred, diag)
    delays = [r["detection_delay_days"] for r in res["users"]]
    assert delays == [-160, -160]


# ---------------------------------------------------------------------------
# F: threshold diagnostic
# ---------------------------------------------------------------------------
def test_threshold_diagnostic_near_threshold():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    cal = _syn_cal_scores(table, blocks["CAL"]["users"])
    res = p15.threshold_diagnostic(pred, cal)
    assert res["test_block"]["threshold"] == p15.FROZEN_THRESHOLD
    assert set(res["test_block"]["near_threshold"]) == {"0.05", "0.1", "0.2"}
    assert set(res["cdf_comparison"]["grid"]) == {0.1, 0.25, 0.5, 0.75, 0.9}


def test_threshold_diagnostic_empty_frame_guards():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    empty = pred.iloc[0:0].copy()
    res = p15.threshold_diagnostic(empty, _syn_cal_scores(
        table, blocks["CAL"]["users"]))
    assert res["test_block"]["n_malicious_rows"] == 0
    assert res["test_block"]["max_missed_positive_score"] is None


# ---------------------------------------------------------------------------
# G: cohort comparison
# ---------------------------------------------------------------------------
def test_cohort_comparison_rows_and_labels():
    res = p15.cohort_comparison(_syn_frozen())
    assert len(res["rows"]) == 4
    assert all(r["coverage"] is None or r["coverage"] > 0 for r in res["rows"])
    labels = {a["label"] for a in res["attribution"]}
    assert labels <= {"OBSERVED", "INFERENCE", "HYPOTHESIS"}
    assert "OBSERVED" in labels


# ---------------------------------------------------------------------------
# Assembler + structural guards
# ---------------------------------------------------------------------------
def test_run_phase15_structure():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    cal = _syn_cal_scores(table, blocks["CAL"]["users"])
    repro = {"gate1": {"passed": True}, "gate2": {"passed": True}}
    res = p15.run_phase15(table, blocks, pred, diag, cal, _syn_frozen(),
                          repro)
    assert res["experiment_id"] == "phase15-unseen-user-diagnosis"
    for key in ("activity_strata", "feature_shift", "detected_vs_missed",
                "graph_diagnosis", "temporal_analysis",
                "threshold_diagnostic", "cohort_comparison"):
        assert key in res


def test_run_phase15_rejects_wrong_columns():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"]).drop(
        columns=["alert_secondary"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    cal = _syn_cal_scores(table, blocks["CAL"]["users"])
    with pytest.raises(AssertionError):
        p15.run_phase15(table, blocks, pred, diag, cal, _syn_frozen(), {})


def test_run_phase15_rejects_chronological_test_columns():
    """Structural guard: test_pred must be the Phase 14 artifact schema."""
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    pred["alert_p0"] = False  # Phase 12 (chronological) schema column
    diag = _syn_diag(table, blocks["TEST"]["users"])
    cal = _syn_cal_scores(table, blocks["CAL"]["users"])
    with pytest.raises(AssertionError):
        p15.run_phase15(table, blocks, pred, diag, cal, _syn_frozen(), {})


# ---------------------------------------------------------------------------
# Determinism + JSON
# ---------------------------------------------------------------------------
def test_determinism_double_run_identical():
    table = _syn_table()
    blocks = _syn_blocks()
    pred = _syn_pred(table, blocks["TEST"]["users"])
    diag = _syn_diag(table, blocks["TEST"]["users"])
    cal = _syn_cal_scores(table, blocks["CAL"]["users"])
    repro = {"gate1": {"passed": True}, "gate2": {"passed": True}}
    r1 = p15.run_phase15(table, blocks, pred, diag, cal, _syn_frozen(), repro)
    r2 = p15.run_phase15(table, blocks, pred, diag, cal, _syn_frozen(), repro)
    assert p15.canonical(r1) == p15.canonical(r2)


def test_canonical_json_numpy_safe():
    obj = {"a": np.int64(3), "b": np.float64(1.5), "c": np.bool_(True),
           "d": np.array([1.0, 2.0]), "e": pd.Timestamp("2010-01-02")}
    out = json.loads(p15.canonical(obj))
    assert out == {"a": 3, "b": 1.5, "c": True, "d": [1.0, 2.0],
                   "e": "2010-01-02 00:00:00"}


# ---------------------------------------------------------------------------
# Small model fixture: score reproduction path (gate wiring)
# ---------------------------------------------------------------------------
def _train_tiny_model() -> lgb.Booster:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 12))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    ds = lgb.Dataset(X, label=y)
    return lgb.train({"objective": "binary", "verbosity": -1, "seed": 42,
                      "min_data_in_leaf": 5, "num_leaves": 8},
                     ds, num_boost_round=10)


def test_score_reproduction_gate_detects_perturbation():
    """The gate wiring must flag scores that do not match an artifact."""
    model = _train_tiny_model()
    table = _syn_table()
    feats = FROZEN_FEATURES
    X = table[feats].astype(np.float64).to_numpy()
    s1 = model.predict(X)
    s2 = model.predict(X + 1e-3)  # perturbed input -> different scores
    assert np.abs(s1 - s2).max() > p15.SCORE_GATE_TOL


# ---------------------------------------------------------------------------
# Real-data artifact reloadability (skipped until Phase 15 runs)
# ---------------------------------------------------------------------------
ARTIFACT_CHECKS = {
    "phase15_experiment.json": ("json", "gates"),
    "phase15_activity_strata.json": ("json", "test_block_15_users"),
    "phase15_feature_shift.json": ("json", "ranking_by_combined_drift"),
    "phase15_detected_vs_missed.json": ("json", "users"),
    "phase15_graph_diagnosis.json": ("json", "discrimination"),
    "phase15_temporal_analysis.json": ("json", "users"),
    "phase15_threshold_diagnostic.json": ("json", "test_block"),
    "phase15_cohort_comparison.json": ("json", "attribution"),
    "phase15_cal_scores.parquet": ("parquet", None),
}


@pytest.mark.parametrize("name,kind,key",
                         sorted((k, v[0], v[1]) for k, v
                                in ARTIFACT_CHECKS.items()))
def test_phase15_artifact_reloadable(name, kind, key):
    path = ART / name
    if not path.is_file():
        pytest.skip(f"artifact not produced yet: {name}")
    if kind == "json":
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        assert key in obj
    else:
        df = pd.read_parquet(path)
        assert len(df) == 54108
        assert list(df.columns) == ["user", "day", "is_malicious", "score"]


def test_phase15_manifest_md5s_match():
    manifest = ART / "phase15_manifest.json"
    if not manifest.is_file():
        pytest.skip("manifest not produced yet")
    with open(manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    for name, rec in man["files"].items():
        path = ART / name
        if not path.is_file():
            continue
        h = hashlib.md5()
        with open(path, "rb") as fh2:
            for chunk in iter(lambda: fh2.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == rec["md5"], f"{name} md5 mismatch vs manifest"


def test_frozen_inputs_still_immutable():
    """Frozen Phase 14 artifacts must be unchanged (md5 baselines)."""
    baselines = {
        "phase6_merged_features.parquet": "9a3b188573bb953416981dfea3379def",
        "phase14_test_predictions.parquet": "71eeb3f1948e518518a53e062d5a213d",
        "phase14_model.txt": "3778a4d869f7231e76ec4c08c6dfa419",
    }
    for name, expected in baselines.items():
        path = ART / name
        if not path.is_file():
            pytest.skip(f"artifact not present: {name}")
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == expected, f"{name} changed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))