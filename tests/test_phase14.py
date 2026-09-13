"""Phase 14 tests: user-disjoint allocation, block gates, leakage controls,
metrics wiring, user-block bootstrap, determinism, interpretation rules,
and artifact reloadability (specification Section 15).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.experiments import phase14 as p14
from src.experiments.phase14 import (
    AUX_NAME,
    C,
    FROZEN_FEATURES,
    FROZEN_THRESHOLD,
    T,
    X,
    PRE_REGISTERED_TABLE,
    canonical_json,
)

ART = Path(config.ARTIFACTS_DIR)
MERGED = ART / "phase6_merged_features.parquet"

has_merged = MERGED.is_file()

real_table = pytest.mark.skipif(
    not has_merged, reason="frozen merged parquet not available locally")

FEAT_COLS = FROZEN_FEATURES


def synthetic_table(n_users: int, n_days: int = 40) -> pd.DataFrame:
    """Small full-grid table; column order user/day/12-features/is_malicious
    (mirrors the modeling view produced by the Phase 14 runner)."""
    users = [f"AAA{i:04d}" for i in range(n_users)]
    days = pd.date_range("2010-06-01", periods=n_days, freq="D")
    grid = pd.MultiIndex.from_product([users, days], names=["user", "day"])
    df = pd.DataFrame(index=grid).reset_index()
    for c in FEAT_COLS:
        df[c] = 0
    df["is_malicious"] = 0
    return df[["user", "day"] + FEAT_COLS + ["is_malicious"]]


def expected_from_allocation(table: pd.DataFrame) -> dict:
    """Block-count gate derived from the deterministic allocation rule.

    Used by synthetic protocol-wiring tests; the real-data gate values
    (specification Section 6) are tested separately against the frozen
    parquet.
    """
    alloc = p14.allocate_users(table)
    out = {}
    for s in (T, C, X):
        us = sorted(u for u, v in alloc.items() if v == s)
        part = table[table["user"].isin(us)]
        out[s] = {
            "users": len(us),
            "rows": len(part),
            "pos": int(part["is_malicious"].sum()),
            "mal_users": int(part.loc[part["is_malicious"] == 1,
                                      "user"].nunique()),
        }
    return out


# ---------------------------------------------------------------------------
# Allocation rule (deterministic, pre-registered)
# ---------------------------------------------------------------------------
def test_malicious_round_robin_rule():
    """70 users sorted by (first_positive_day, user_id); period 14 -> 40/15/15."""
    users = [f"AAA{i:04d}" for i in range(70)]
    onsets = pd.to_datetime("2010-06-10") + pd.Timedelta(days=1) * (
        np.arange(70) % 31)
    table = synthetic_table(70)
    table["is_malicious"] = 0
    for i, u in enumerate(users):
        mask = table["user"] == u
        table.loc[mask & (table["day"] == onsets[i]), "is_malicious"] = 1
    alloc = p14.allocate_users(table)
    counts = {s: sum(1 for v in alloc.values() if v == s) for s in (T, C, X)}
    assert counts == {T: 40, C: 15, X: 15}


def test_benign_round_robin_rule():
    """930 benign users, period 10 (0-7 TRAIN, 8 CAL, 9 TEST) -> 744/93/93."""
    table = synthetic_table(1000)
    # make 70 users malicious so benign == 930
    mal_users = [f"AAA{i:04d}" for i in range(70)]
    table.loc[table["user"].isin(mal_users), "is_malicious"] = 1
    alloc = p14.allocate_users(table)
    benign = {u: s for u, s in alloc.items() if u not in mal_users}
    counts = {s: sum(1 for v in benign.values() if v == s) for s in (T, C, X)}
    assert counts == {T: 744, C: 93, X: 93}


def test_allocation_deterministic_no_rng():
    table = synthetic_table(1000)
    table.loc[table["user"] == "AAA0000", "is_malicious"] = 1
    a1 = p14.allocate_users(table)
    a2 = p14.allocate_users(table.copy().sample(frac=1.0, random_state=7))
    assert a1 == a2  # row order must not matter


@real_table
def test_pre_registered_table_reproduced_verbatim():
    table = pd.read_parquet(MERGED)
    alloc = p14.verify_allocation_table(table)
    assert len(alloc) == 1000
    mal = table[table["is_malicious"] == 1]
    per = mal.groupby("user")["day"].agg(first="min", count="count")
    for user, (split, pos, onset) in PRE_REGISTERED_TABLE.items():
        assert alloc[user] == split
        assert int(per.loc[user, "count"]) == pos
        assert str(per.loc[user, "first"].date()) == onset


# ---------------------------------------------------------------------------
# Block gates (real frozen table)
# ---------------------------------------------------------------------------
@real_table
def test_block_gates():
    table = pd.read_parquet(MERGED)
    blocks = p14.user_disjoint_blocks(table)
    assert {s: (b["n_users"], b["rows"], b["positives"], b["malicious_users"])
            for s, b in blocks.items()} == {
        T: (784, 392784, 1037, 40),
        C: (108, 54108, 379, 15),
        X: (108, 54108, 476, 15),
    }
    # disjointness
    sets = [set(blocks[s]["users"]) for s in (T, C, X)]
    assert not (sets[0] & sets[1] | sets[0] & sets[2] | sets[1] & sets[2])


@real_table
def test_label_reproduction_gate():
    table = pd.read_parquet(MERGED)
    assert int(table["is_malicious"].sum()) == 1892
    assert table.loc[table["is_malicious"] == 1, "user"].nunique() == 70
    # CDE1846: unparseable answer-key date (foundation report); OBSERVED to
    # be absent from the 1,000-user population entirely (0 rows in the
    # frozen table) -> it can never be labeled; benign by exclusion.
    assert "CDE1846" not in table["user"].unique()


@real_table
def test_full_grid_per_user():
    table = pd.read_parquet(MERGED)
    sizes = table.groupby("user").size()
    assert sizes.min() == sizes.max() == 501
    assert table["day"].nunique() == 501


# ---------------------------------------------------------------------------
# Feature / leakage controls
# ---------------------------------------------------------------------------
def test_frozen_feature_set():
    assert len(FROZEN_FEATURES) == 12
    assert "department_file_type_mismatch_count" not in FROZEN_FEATURES
    assert "user" not in FROZEN_FEATURES and "day" not in FROZEN_FEATURES
    assert "is_malicious" not in FROZEN_FEATURES
    for f in FROZEN_FEATURES:
        assert f in config.BEHAVIORAL_FEATURES or f in config.GRAPH_FEATURES


@real_table
def test_modeling_view_columns():
    table = pd.read_parquet(MERGED)
    view = table[["user", "day"] + FROZEN_FEATURES + ["is_malicious"]]
    assert list(view.columns) == ["user", "day"] + FROZEN_FEATURES + ["is_malicious"]


def test_no_label_in_features_structural():
    table = synthetic_table(10)
    feats = list(p14.FROZEN_FEATURES)
    assert set(feats) & {"user", "day", "is_malicious"} == set()


# ---------------------------------------------------------------------------
# Metric wiring (hand-computed operating points)
# ---------------------------------------------------------------------------
def test_operating_point_application():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.99, 0.92, 0.50, 0.30, 0.10])
    prim = m.binary_decision_metrics(y, s, FROZEN_THRESHOLD)
    # 0.99 and 0.92 are both >= 0.9186015432508062
    assert prim["tp"] == 2 and prim["fp"] == 0
    assert prim["precision"] == 1.0 and prim["recall"] == 1.0
    sec = m.binary_decision_metrics(y, s, 0.40)
    assert sec["tp"] == 2 and sec["fp"] == 1
    assert sec["n_alerts"] == 3


def test_best_f1_threshold_wiring():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    t, f1 = th.best_f1_threshold(y, s)
    assert t == 0.8
    assert f1 == 1.0


# ---------------------------------------------------------------------------
# User-block bootstrap (new estimator; hand-computed semantics)
# ---------------------------------------------------------------------------
def test_user_block_bootstrap_tiny_panel():
    # 3 users; user B has the only positive; resampling users must keep the
    # panel structure (rows pooled per resampled user).
    users = np.array(["u1", "u1", "u2", "u2", "u3", "u3"], dtype=object)
    y = np.array([0, 0, 1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    res = p14.user_block_bootstrap_ci(users, y, s, "auc_roc", n_boot=100,
                                      seed=42, alpha=0.10)
    assert res["n_users"] == 3
    assert res["n_boot"] == 100
    assert res["n_valid"] + res["n_skipped"] == 100
    assert 0.0 <= res["ci_low"] <= res["mean"] <= res["ci_high"] <= 1.0


def test_user_block_bootstrap_deterministic():
    users = np.array(["u1"] * 4 + ["u2"] * 4, dtype=object)
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    s = np.arange(8, dtype=float) / 8
    a = p14.user_block_bootstrap_ci(users, y, s, "auc_roc", n_boot=50, seed=42)
    b = p14.user_block_bootstrap_ci(users, y, s, "auc_roc", n_boot=50, seed=42)
    assert a == b


def test_user_block_bootstrap_degenerate_skipped():
    users = np.array(["u1", "u1", "u2", "u2"], dtype=object)
    y = np.zeros(4, dtype=int)
    s = np.arange(4, dtype=float)
    res = p14.user_block_bootstrap_ci(users, y, s, "auc_roc", n_boot=10, seed=42)
    assert res["n_valid"] == 0 and res["n_skipped"] == 10


# ---------------------------------------------------------------------------
# Wilson CI + Gini (conventions reused from Phases 9/12/13)
# ---------------------------------------------------------------------------
def test_wilson_ci_values():
    # hand-computed values verified in the Phase 13 suite (same Z90 formula)
    assert p14.wilson_ci(14, 49) == pytest.approx(
        (0.19842132118593642, 0.39543267207797805), abs=1e-12)
    assert p14.wilson_ci(0, 10) == pytest.approx(
        (0.022672141311501293, 0.1902698287719057), abs=1e-12)


def test_gini():
    # Phase 9/12 convention (alternative Gini formula)
    assert p14.gini_of_counts(np.array([1, 1, 1])) == pytest.approx(0.0)
    assert p14.gini_of_counts(np.array([0, 0, 3])) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Interpretation rule (pre-registered bands, spec Section 20)
# ---------------------------------------------------------------------------
def test_interpretation_bands():
    chrono = {"auc_roc": 0.90, "auc_pr": 0.30}
    same = p14.interpret(0.91, 0.32, 13, chrono)
    assert same["auc_roc_label"] == "comparable"
    assert same["auc_pr_label"] == "comparable"
    assert same["coverage_label"] == "high user-level coverage"
    worse = p14.interpret(0.80, 0.10, 5, chrono)
    assert worse["auc_roc_label"] == "materially worse"
    assert worse["auc_pr_label"] == "materially worse"
    assert worse["coverage_label"] == "low user-level coverage"
    mid = p14.interpret(0.86, 0.25, 9, chrono)
    assert mid["auc_roc_label"] == "inconclusive"
    assert mid["auc_pr_label"] == "comparable"
    assert mid["coverage_label"] == "moderate user-level coverage"


# ---------------------------------------------------------------------------
# Determinism of the deterministic components (no RNG outside seed 42)
# ---------------------------------------------------------------------------
def _mixed_table() -> pd.DataFrame:
    """28 users (18 malicious, 10 benign) sized so the round-robin rule puts
    malicious users in all three blocks and benign users in CAL and TEST:
    malicious positions 0-17 -> T(12)/C(3)/X(4); benign 10 -> T(8)/C(1)/X(1)."""
    table = synthetic_table(28)
    mal_users = [f"AAA{i:04d}" for i in range(18)]
    table.loc[table["user"].isin(mal_users), "is_malicious"] = 1
    return table, mal_users


def test_determinism_small_table():
    table, mal_users = _mixed_table()
    scen = {u: (i % 3) + 1 for i, u in enumerate(mal_users)}
    chrono = {"auc_roc": 0.9, "auc_pr": 0.3}
    expected = expected_from_allocation(table)
    n_mal = int(table["is_malicious"].sum())
    r1 = p14.run_experiment(table, scen, chrono, expected_blocks=expected,
                            label_gate=(18, n_mal))
    r2 = p14.run_experiment(table.copy(), dict(scen), chrono,
                            expected_blocks=expected, label_gate=(18, n_mal))
    assert canonical_json(r1) == canonical_json(r2)
    # every block must contain positives (wiring sanity)
    for s in (T, C, X):
        assert r1["blocks"][s]["positives"] > 0


def test_canonical_json_roundtrip():
    obj = {"a": np.int64(1), "b": np.float64(2.5), "c": np.array([1, 2]),
           "d": pd.Timestamp("2010-01-02")}
    s = canonical_json(obj)
    assert json.loads(s) == {"a": 1, "b": 2.5, "c": [1, 2],
                             "d": "2010-01-02 00:00:00"}


def test_run_experiment_returns_artifacts_consistently():
    table, mal_users = _mixed_table()
    scen = {u: (i % 3) + 1 for i, u in enumerate(mal_users)}
    chrono = {"auc_roc": 0.9, "auc_pr": 0.3}
    expected = expected_from_allocation(table)
    n_mal = int(table["is_malicious"].sum())
    r_plain = p14.run_experiment(table, scen, chrono, expected_blocks=expected,
                                 label_gate=(18, n_mal))
    r_art, pred, model = p14.run_experiment(table, scen, chrono,
                                            expected_blocks=expected,
                                            label_gate=(18, n_mal),
                                            return_artifacts=True)
    assert canonical_json(r_plain) == canonical_json(r_art)
    assert list(pred.columns) == ["user", "day", "is_malicious", "score",
                                  "alert_primary", "alert_secondary"]
    assert model is not None


# ---------------------------------------------------------------------------
# Artifact reloadability (skip-guarded; after artifacts are pulled to PC)
# ---------------------------------------------------------------------------
ARTIFACT_CHECKS = {
    "phase14_split.json": ("json", "blocks"),
    "phase14_calibration.json": ("json", "secondary_threshold"),
    "phase14_test_metrics.json": ("json", "test_metrics"),
    "phase14_user_diagnostics.json": ("json", "coverage"),
    "phase14_bootstrap.json": ("json", "user_block"),
    "phase14_scenario.json": ("json", T),
    "phase14_cost.json": ("json", "elapsed_s"),
    "phase14_manifest.json": ("json", "files"),
    "phase14_experiment.json": ("json", "interpretation"),
    "phase14_model_record.json": ("json", "best_iteration"),
    "phase14_test_predictions.parquet": ("parquet", None),
}


@pytest.mark.parametrize("name,kind,key",
                         sorted((k, v[0], v[1]) for k, v in ARTIFACT_CHECKS.items()))
def test_artifact_reloadable(name, kind, key):
    path = ART / name
    if not path.is_file():
        pytest.skip(f"artifact not pulled yet: {name}")
    if kind == "json":
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        assert key in obj
    else:
        df = pd.read_parquet(path)
        assert len(df) == 54108
        assert list(df.columns) == ["user", "day", "is_malicious", "score",
                                    "alert_primary", "alert_secondary"]


def test_artifact_manifest_md5s_match():
    manifest = ART / "phase14_manifest.json"
    if not manifest.is_file():
        pytest.skip("manifest not pulled yet")
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


def test_model_record_auxiliary_name():
    path = ART / "phase14_model_record.json"
    if not path.is_file():
        pytest.skip("model record not pulled yet")
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    assert rec["model_name"] == AUX_NAME
    assert rec["best_iteration"] >= 1