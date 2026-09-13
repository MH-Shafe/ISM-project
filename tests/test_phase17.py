"""Tests for Phase 17 (targeted calibration/generalization intervention).

Covers: unit tests for every arm transform (Platt, binning with merge
rule + PAV, ECDF midpoint ranks), calibration metrics (Brier/ECE) on
hand-computed fixtures, best-F1 operating-point semantics, gate machinery
on small fixtures, coverage + Wilson CI, user-block bootstrap on a small
panel, the mechanical PASS/CAUTION/FAIL verdict paths, determinism
(double-run byte-identical), artifact reloadability/schema for all Phase
17 artifacts, and structural guards (the chronological TEST never enters;
frozen TEST/CAL shapes enforced).

The authoritative chronological TEST never enters this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase17 as p17

ART = Path(config.ARTIFACTS_DIR)


def _syn_frames(n_users_cal: int = 6, n_users_test: int = 6,
                n_days: int = 20) -> tuple:
    """Deterministic synthetic CAL/TEST score frames (no RNG)."""
    def frame(user_off: int, n_users: int, n_pos_users: int,
              pos_rows: int) -> pd.DataFrame:
        rows = []
        for u in range(user_off, user_off + n_users):
            for d in range(1, n_days + 1):
                base = (u + 1) * 1e-2 + d * 1e-5
                score = float(np.clip(base + 0.9 if u - user_off
                                      < n_pos_users else base, 0.0, 1.0))
                mal = int((u - user_off < n_pos_users)
                          and (d <= pos_rows))
                rows.append({"user": f"U{u:03d}",
                             "day": pd.Timestamp(f"2010-01-{d:02d}"),
                             "is_malicious": mal, "score": score})
        return pd.DataFrame(rows)

    cal = frame(0, n_users_cal, 2, 3)
    te = frame(n_users_cal, n_users_test, 2, 4)
    te["alert_primary"] = te["score"] >= p17.FROZEN_THRESHOLD
    te["alert_secondary"] = te["score"] >= p17.RECORDED_SECONDARY_THRESHOLD
    user_diag = {"per_user": []}
    for u in range(n_users_cal, n_users_cal + n_users_test):
        mal = u - n_users_cal < 2
        user_diag["per_user"].append({
            "user": f"U{u:03d}",
            "n_positive_days": 4 if mal else 0,
            "activity_tercile": "low",
            "length_class": "short",
            "temporal_half": "2010 H2",
            "scenario": 1 if mal else None,
        })
    gates_env = {"gate3_md5": {"passed": True, "files": {}},
                 "gate4_baselines_match": True,
                 "determinism_ok": False,
                 "dataset_id": "synthetic",
                 "environment": {"python": "test"},
                 "kaggle_note": "none"}
    return cal, te, user_diag, gates_env


# ---------------------------------------------------------------------------
# Arm B — Platt
# ---------------------------------------------------------------------------
def test_platt_fit_and_apply_monotone_increasing():
    rng = np.random.default_rng(7)
    s = np.sort(rng.uniform(0.1, 0.95, 400))
    y = (rng.uniform(size=400) < expit(8 * (s - 0.5))).astype(int)
    fit = p17.platt_fit(s, y)
    assert fit["aborted"] is False
    assert fit["a"] > 0
    p_hat = p17.platt_apply(s, fit)
    assert np.all(np.diff(p_hat) >= 0)
    assert np.all((p_hat > 0) & (p_hat < 1))


def test_platt_abort_on_anti_monotone_fit():
    s = np.linspace(0.05, 0.95, 200)
    y = (s > 0.5).astype(int)[::-1]  # anti-correlated -> a <= 0
    fit = p17.platt_fit(s, y)
    assert fit["aborted"] is True
    assert fit["abort_reason"]
    with pytest.raises(ValueError):
        p17.platt_apply(s, fit)


def test_platt_recovers_logistic_signal():
    s = np.linspace(0.01, 0.99, 500)
    log_odds = logit(s)
    p = expit(2.5 * log_odds + 0.7)
    y = (np.random.default_rng(3).uniform(size=500) < p).astype(int)
    fit = p17.platt_fit(s, y)
    p_hat = p17.platt_apply(s, fit)
    assert np.corrcoef(p, p_hat)[0, 1] > 0.98


# ---------------------------------------------------------------------------
# Arm C — binning
# ---------------------------------------------------------------------------
def test_binning_fit_merge_rule_and_pav():
    rng = np.random.default_rng(11)
    s = np.sort(rng.uniform(0.0, 1.0, 2000))
    p = expit(6 * (s - 0.5))
    y = (rng.uniform(size=2000) < p).astype(int)
    fit = p17.binning_fit(s, y)
    assert fit["aborted"] is False
    assert fit["n_merged_groups"] >= 5
    rates = fit["pav_rates"]
    assert rates == sorted(rates)  # PAV monotone
    assert all(n >= p17.MIN_BIN_N for n in fit["merged_bin_n"])
    p_hat = p17.binning_apply(s, fit)
    assert np.all((p_hat >= 0) & (p_hat <= 1))
    # step-function: within a bin the value is constant
    edges = np.asarray(fit["edges"])
    bin_id = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, 20 - 1)
    for b in np.unique(bin_id):
        assert len(np.unique(p_hat[bin_id == b])) <= 1


def test_binning_constant_scores_aborts():
    s = np.full(300, 0.5)
    y = np.zeros(300, dtype=int)
    y[:3] = 1
    fit = p17.binning_fit(s, y)
    assert fit["aborted"] is True
    assert fit["n_merged_groups"] < 5
    with pytest.raises(ValueError):
        p17.binning_apply(s, fit)


def test_binning_apply_matches_fit_rates_manually():
    rng = np.random.default_rng(5)
    s = np.sort(rng.uniform(0.0, 1.0, 1000))
    y = (s > 0.6).astype(int)
    fit = p17.binning_fit(s, y, k=8, min_n=10)
    p_hat = p17.binning_apply(s, fit)
    bin_rate = np.asarray(fit["bin_rate"])
    edges = np.asarray(fit["edges"])
    bin_id = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, 7)
    assert np.array_equal(p_hat, bin_rate[bin_id])


# ---------------------------------------------------------------------------
# Arm D — ECDF
# ---------------------------------------------------------------------------
def test_ecdf_midpoint_rank_ties_and_strictness():
    s_cal = np.array([0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.9])
    s_new = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.9, 0.95])
    p = p17.ecdf_apply(s_cal, s_new)
    n = len(s_cal)
    # manual midpoint ranks: (lo + (hi - lo)/2 - 0.5) / n
    expected = np.array([(0 - 0.5) / n, (0.5 - 0.5) / n, (2.5 - 0.5) / n,
                         (4 - 0.5) / n, (4.5 - 0.5) / n, (6 - 0.5) / n,
                         (7.5 - 0.5) / n, (8 - 0.5) / n])
    assert np.allclose(p, expected)
    # strictly increasing over distinct raw scores
    assert np.all(np.diff(p[[0, 1, 3, 4, 5, 6, 7]]) > 0)
    # out-of-CAL-range maps marginally outside [0, 1] (pre-registered)
    assert p[0] < 0 and p[-1] < 1


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def test_brier_hand_computed():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.25, 0.75, 0.5, 1.0])
    expected = np.mean((p - y) ** 2)
    assert p17.brier_score(y, p) == pytest.approx(expected)


def test_ece_perfect_calibration():
    rng = np.random.default_rng(1)
    n = 2000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p).astype(int)
    e = p17.ece_score(y, p)
    assert e["ece"] < 0.05
    assert len(e["reliability"]) == 10


def test_ece_extreme_case():
    y = np.zeros(1000, dtype=int)
    p = np.ones(1000)
    e = p17.ece_score(y, p)
    assert e["ece"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
def test_operating_point_matches_best_f1():
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=500) < 0.2).astype(int)
    s = rng.uniform(size=500)
    op = p17.best_f1_operating_point(y, s)
    t, f1 = th.best_f1_threshold(y, s)
    assert op["threshold"] == t
    assert op["cal_best_f1"] == f1


# ---------------------------------------------------------------------------
# Gates (small fixtures)
# ---------------------------------------------------------------------------
def test_gate1b_monotone_arms_identical_alert_sets():
    rng = np.random.default_rng(9)
    s_cal = np.sort(rng.uniform(0.0, 1.0, 300))
    y_cal = (rng.uniform(size=300) < expit(6 * (s_cal - 0.45))).astype(int)
    s_te = rng.uniform(0.0, 1.0, 200)
    fits = {}
    pf = p17.platt_fit(s_cal, y_cal)
    fits["B"] = {"p_cal": p17.platt_apply(s_cal, pf),
                 "apply": lambda s: p17.platt_apply(s, pf)}
    bf = p17.binning_fit(s_cal, y_cal)
    fits["C"] = {"p_cal": p17.binning_apply(s_cal, bf),
                 "apply": lambda s: p17.binning_apply(s, bf)}
    fits["D"] = {"p_cal": p17.ecdf_apply(s_cal, s_cal),
                 "apply": lambda s: p17.ecdf_apply(s_cal, s)}
    g = p17.gate1b_operating_equality(y_cal, s_cal, s_te, fits)
    assert g["arms"]["B"]["gate_applies"] is True
    assert g["arms"]["B"]["alert_set_rows_identical_to_raw"] is True
    assert g["arms"]["D"]["alert_set_rows_identical_to_raw"] is True
    assert g["arms"]["C"]["gate_applies"] is False
    assert g["arms"]["C"]["raw_threshold_close"] in (True, False)


# ---------------------------------------------------------------------------
# Coverage / per-user / scenario
# ---------------------------------------------------------------------------
def test_coverage_metrics_hand_computed():
    test = pd.DataFrame({
        "user": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
        "is_malicious": [1, 1, 1, 0, 0, 0, 1, 0],
    })
    user_diag = {"per_user": [
        {"user": "U1", "n_positive_days": 2, "activity_tercile": "high",
         "length_class": "long", "temporal_half": "2011", "scenario": 2},
        {"user": "U2", "n_positive_days": 1, "activity_tercile": "high",
         "length_class": "long", "temporal_half": "2011", "scenario": 2},
        {"user": "U3", "n_positive_days": 0, "activity_tercile": "high",
         "length_class": "long", "temporal_half": "2011", "scenario": None},
        {"user": "U4", "n_positive_days": 1, "activity_tercile": "high",
         "length_class": "long", "temporal_half": "2011", "scenario": 3},
    ]}
    alert = np.array([True, True, False, False, False, False, True, False])
    cov = p17.coverage_metrics(test, alert, user_diag)
    assert cov["coverage"]["detected"] == 2
    assert cov["coverage"]["total"] == 3
    assert cov["zero_alert_users"] == ["U2"]
    lo, hi = p17.wilson_ci(2, 3, z=p17.Z90)
    assert cov["coverage"]["wilson_90"] == [lo, hi]
    assert cov["alert_concentration"]["n_alerts_total"] == 3


def test_coverage_by_strata_and_scenario():
    test = pd.DataFrame({
        "user": ["U1", "U2", "U3"],
        "is_malicious": [1, 1, 1],
    })
    user_diag = {"per_user": [
        {"user": "U1", "n_positive_days": 1, "activity_tercile": "low",
         "length_class": "short", "temporal_half": "2010 H2",
         "scenario": 1},
        {"user": "U2", "n_positive_days": 1, "activity_tercile": "low",
         "length_class": "short", "temporal_half": "2010 H2",
         "scenario": 1},
        {"user": "U3", "n_positive_days": 1, "activity_tercile": "low",
         "length_class": "short", "temporal_half": "2010 H2",
         "scenario": 2},
    ]}
    alert = np.array([True, False, True])
    strata = p17.coverage_by_strata(test, alert, user_diag)
    assert strata["scenario"]["1"]["n_users"] == 2
    assert strata["scenario"]["1"]["detected"] == 1
    rec = p17.scenario_recall(test, alert, user_diag)
    assert rec["1"]["recall"] == pytest.approx(0.5)
    assert rec["2"]["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap (small panel)
# ---------------------------------------------------------------------------
def test_user_block_bootstrap_operating_small():
    rng = np.random.default_rng(13)
    users = np.repeat([f"U{i}" for i in range(8)], 10)
    y = (rng.uniform(size=80) < 0.2).astype(int)
    s = rng.uniform(size=80)
    out = p17.user_block_bootstrap_operating(users, y, s, 0.8,
                                             n_boot=20)
    assert out["n_users"] == 8
    assert set(out) >= {"n_alerts", "precision", "recall", "f1"}
    assert out["f1"]["ci_low"] <= out["f1"]["mean"] <= out["f1"]["ci_high"]


def test_calibration_transfer_bootstrap_small():
    rng = np.random.default_rng(17)
    cal_users = np.repeat([f"C{i}" for i in range(6)], 10)
    s_cal = rng.uniform(size=60)
    y_cal = (rng.uniform(size=60) < 0.3).astype(int)
    s_te = rng.uniform(size=40)
    y_te = (rng.uniform(size=40) < 0.3).astype(int)
    out = p17.calibration_transfer_bootstrap(
        cal_users, s_cal, y_cal, s_te, y_te, "platt", n_boot=10)
    assert out["n_cal_users"] == 6
    assert out["n_skipped"] >= 0
    assert 0.0 <= out["brier"]["mean"] <= 1.0
    out2 = p17.calibration_transfer_bootstrap(
        cal_users, s_cal, y_cal, s_te, y_te, "ecdf", n_boot=10)
    assert 0.0 <= out2["brier"]["mean"] <= 1.0


# ---------------------------------------------------------------------------
# Verdict paths
# ---------------------------------------------------------------------------
def _fake_arm(brier_improvement=0.0, ece=0.1, f1=0.1, precision=0.1,
              alerts=500, coverage=3, status="evaluated") -> dict:
    if status == "aborted":
        return {"status": "aborted"}
    return {
        "status": status,
        "test_calibration": {"brier_improvement": brier_improvement,
                             "ece": ece},
        "operating_test": {"f1": f1, "precision": precision,
                           "n_alerts": alerts,
                           "coverage": {"coverage": {"detected": coverage}}},
        "ranking": {"auc_roc_delta_abs": 0.0},
    }


def _gates_all_pass() -> dict:
    return {"gate1": {"passed": True}, "gate1b": {"passed": True},
            "gate2": {"passed": True}, "gate3": {"passed": True},
            "gate4": {"passed": True}}


def test_verdict_pass():
    arms = {"A": _fake_arm(),
            "B": _fake_arm(brier_improvement=0.30, ece=0.03, f1=0.6,
                           precision=0.8, alerts=150, coverage=10),
            "C": _fake_arm(coverage=9), "D": _fake_arm(coverage=9)}
    v = p17.compute_verdict(arms, _gates_all_pass(), determinism_ok=True,
                            tie_effect_max=0.0)
    assert v["verdict"] == "PASS"
    assert v["transfer_arms"] == ["B"] and v["operating_target_arms"] == ["B"]


def test_verdict_caution_transfer_without_operating():
    arms = {"A": _fake_arm(),
            "B": _fake_arm(brier_improvement=0.30, ece=0.03, f1=0.2,
                           precision=0.3, alerts=700, coverage=4),
            "C": _fake_arm(), "D": _fake_arm()}
    v = p17.compute_verdict(arms, _gates_all_pass(), determinism_ok=True,
                            tie_effect_max=0.0)
    assert v["verdict"] == "CAUTION"


def test_verdict_caution_tie_effects():
    arms = {"A": _fake_arm(),
            "B": _fake_arm(brier_improvement=0.30, ece=0.03, f1=0.6,
                           precision=0.8, alerts=150, coverage=10),
            "C": _fake_arm(), "D": _fake_arm()}
    v = p17.compute_verdict(arms, _gates_all_pass(), determinism_ok=True,
                            tie_effect_max=0.005)
    assert v["verdict"] == "CAUTION"


def test_verdict_fail_gate_and_no_transfer():
    arms = {"A": _fake_arm(), "B": _fake_arm(), "C": _fake_arm(),
            "D": _fake_arm()}
    v = p17.compute_verdict(arms, _gates_all_pass(), determinism_ok=False,
                            tie_effect_max=0.0)
    assert v["verdict"] == "FAIL"
    gates = _gates_all_pass()
    gates["gate2"]["passed"] = False
    v2 = p17.compute_verdict(arms, gates, determinism_ok=True,
                             tie_effect_max=0.0)
    assert v2["verdict"] == "FAIL"


def test_verdict_skips_aborted_arms():
    arms = {"A": _fake_arm(), "B": _fake_arm(status="aborted"),
            "C": _fake_arm(status="aborted"), "D": _fake_arm()}
    v = p17.compute_verdict(arms, _gates_all_pass(), determinism_ok=True,
                            tie_effect_max=0.0)
    assert v["verdict"] == "FAIL"  # nothing transfers or operates


# ---------------------------------------------------------------------------
# Full-pipeline determinism + structure (synthetic, exact frozen shapes)
# ---------------------------------------------------------------------------
def _syn_full_frames() -> tuple:
    """Synthetic CAL/TEST frames with the FROZEN shapes: 108 users x 501
    days = 54,108 rows each; 379 (CAL) / 476 (TEST) positives."""
    n_users, n_days = 108, 501
    cal_pos = [25] * 14 + [29]   # 15 malicious users, 379 positive days
    test_pos = [32] * 14 + [28]  # 15 malicious users, 476 positive days

    def frame(off: int, pos_counts: list[int]) -> pd.DataFrame:
        rows = []
        first_mal = n_users - len(pos_counts)
        for i in range(n_users):
            u = f"U{off + i:03d}"
            mal_days = (pos_counts[i - first_mal] if i >= first_mal
                        else 0)
            for d in range(n_days):
                score = float(np.clip(0.25 + 0.65 * (i / (n_users - 1))
                                      + 1e-4 * (d % 7), 0.0, 1.0))
                rows.append({
                    "user": u,
                    "day": pd.Timestamp("2010-01-01")
                    + pd.Timedelta(days=d),
                    "is_malicious": int(mal_days > 0 and d < mal_days),
                    "score": score})
        return pd.DataFrame(rows)

    cal = frame(0, cal_pos)
    te = frame(n_users, test_pos)
    te["alert_primary"] = te["score"] >= p17.FROZEN_THRESHOLD
    te["alert_secondary"] = te["score"] >= p17.RECORDED_SECONDARY_THRESHOLD
    user_diag = {"per_user": []}
    for i in range(n_users):
        mal = i >= n_users - len(test_pos)
        user_diag["per_user"].append({
            "user": f"U{n_users + i:03d}",
            "n_positive_days": (test_pos[i - (n_users - len(test_pos))]
                                if mal else 0),
            "activity_tercile": "low",
            "length_class": "short",
            "temporal_half": "2010 H2",
            "scenario": 1 if mal else None,
        })
    gates_env = {"gate3_md5": {"passed": True, "files": {}},
                 "gate4_baselines_match": True,
                 "determinism_ok": False,
                 "dataset_id": "synthetic",
                 "environment": {"python": "test"},
                 "kaggle_note": "none"}
    assert len(cal) == 54108 and int(cal["is_malicious"].sum()) == 379
    assert len(te) == 54108 and int(te["is_malicious"].sum()) == 476
    return cal, te, user_diag, gates_env


def test_full_pipeline_deterministic_and_structured():
    cal, te, ud, env = _syn_full_frames()
    r1 = p17.run_phase17(cal, te, ud, env, n_boot=5)
    r2 = p17.run_phase17(cal, te, ud, env, n_boot=5)
    assert p17.canonical(r1) == p17.canonical(r2)
    for arm in ("A", "B", "C", "D"):
        assert arm in r1["arms"]
        assert r1["arms"][arm]["status"] in ("reference", "evaluated",
                                             "aborted")
    assert set(r1["gates"]) == {"gate1", "gate1b", "gate2", "gate3",
                                "gate4"}
    assert set(r1["bootstrap"]) == {"n_boot", "seed", "alpha",
                                    "user_block_ranking",
                                    "user_block_operating",
                                    "calibration_transfer"}
    assert r1["verdict"]["verdict"] in ("PASS", "CAUTION", "FAIL")


def test_full_pipeline_structural_guards():
    cal, te, ud, env = _syn_full_frames()
    bad = te.drop(columns=["alert_secondary"])
    with pytest.raises(AssertionError):
        p17.run_phase17(cal, bad, ud, env, n_boot=2)
    bad_cal = cal.copy()
    bad_cal.loc[0, "score"] = np.nan
    with pytest.raises(AssertionError):
        p17.run_phase17(bad_cal, te, ud, env, n_boot=2)
    bad_shape = te.iloc[:-1]
    with pytest.raises(AssertionError):
        p17.run_phase17(cal, bad_shape, ud, env, n_boot=2)


def test_ranking_invariant_under_monotone_transform():
    rng = np.random.default_rng(4)
    y = (rng.uniform(size=300) < 0.2).astype(int)
    s = rng.uniform(size=300)
    g = expit(3.0 * (s - 0.5))  # strictly monotone in s
    r_s = p17.ranking_verification(y, s, "X")
    r_g = p17.ranking_verification(y, g, "X")
    assert r_g["classification"]["auc_roc"] == r_s["classification"]["auc_roc"]
    assert r_g["classification"]["auc_pr"] == r_s["classification"]["auc_pr"]
    assert r_g["top_k"] == r_s["top_k"]
    assert r_g["deltas"]["auc_roc"] == r_s["deltas"]["auc_roc"]


# ---------------------------------------------------------------------------
# Artifact reloadability + frozen-input immutability (real artifacts)
# ---------------------------------------------------------------------------
def test_phase17_experiment_artifact_reloadable():
    path = ART / "phase17_experiment.json"
    if not path.is_file():
        pytest.skip("artifact not produced yet")
    with open(path, encoding="utf-8") as fh:
        exp = json.load(fh)
    assert exp["phase"] == "17"
    assert exp["verdict"]["verdict"] in ("PASS", "CAUTION", "FAIL")
    assert exp["determinism"]["bit_identical"] is True
    assert set(exp["gates"]) == {"gate1", "gate1b", "gate2", "gate3",
                                 "gate4"}


def test_phase17_artifacts_schema():
    files = {
        "phase17_calibration_fit.json": ["B", "C", "D"],
        "phase17_calibration_metrics.json": ["arms"],
        "phase17_operating_points.json": ["A", "B", "C", "D"],
        "phase17_ranking_verification.json": ["A", "B", "C", "D"],
        "phase17_cost.json": ["elapsed_s", "artifact_sizes_bytes"],
    }
    for name, keys in files.items():
        path = ART / name
        if not path.is_file():
            pytest.skip(f"artifact not produced yet: {name}")
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        for k in keys:
            assert k in obj


def test_phase17_manifest_md5s_match():
    manifest = ART / "phase17_manifest.json"
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
        assert h.hexdigest() == rec["md5"], f"{name} md5 mismatch"


def test_frozen_phase14_15_inputs_still_immutable():
    baselines = {
        "phase14_test_predictions.parquet": "71eeb3f1948e518518a53e062d5a213d",
        "phase15_cal_scores.parquet": "c955a4ec6ecaf7e9baff80abeb361330",
        "phase14_user_diagnostics.json": "cd893661a3a0c08f9066e14c0bf8dd07",
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