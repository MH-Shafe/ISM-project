"""Phase 14: user-level holdout generalization (auxiliary, evaluation-only).

Implements the approved protocol in docs/phase14_specification.md:

  - deterministic user-disjoint TRAIN/CAL/TEST allocation (pre-registered
    rule; the embedded 70-user table must be reproduced verbatim);
  - auxiliary model `lgbm-graph-v1-uhold`: frozen hyperparameters, frozen 12
    features, scale_pos_weight from user-disjoint TRAIN labels, early
    stopping on user-disjoint CAL AUC (patience 100, <= 3000 rounds);
  - primary operating point = frozen production threshold (reference only);
    secondary = best-F1 threshold on user-disjoint CAL (descriptive only);
  - TEST evaluated once per finalized protocol; all metrics, bootstrap CIs,
    and user-level diagnostics pre-registered in the specification.

The module is IO-free and deterministic (no timestamps, no unseeded RNG):
all resampling uses seed 42. Evidence labels: everything computed here is
OBSERVED on the given inputs; interpretation is done by the report writer
per specification Section 20.
"""
from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.evaluation import metrics as m
from src.evaluation import threshold as th
from src.models.lightgbm_baseline import default_params

# ---------------------------------------------------------------------------
# Pre-registered constants (specification Appendix A)
# ---------------------------------------------------------------------------
AUX_NAME = "lgbm-graph-v1-uhold"
FROZEN_THRESHOLD = 0.9186015432508062
SEED = 42
N_BOOT = 1000
ALPHA = 0.10
Z90 = 1.6448536269514722  # 90% Wilson z
EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 3000
SHORT_MAX = 12
LONG_MIN = 45

# Frozen 12-feature input list (Phase 7 FROZEN_CONFIG order; the department
# feature is REJECTED and must never appear).
FROZEN_FEATURES = [c for c in config.BEHAVIORAL_FEATURES
                   if config.BEHAVIORAL_FEATURES[c]["status"] == "defined"]
FROZEN_FEATURES += [g for g in config.GRAPH_FEATURES
                    if g != "department_file_type_mismatch_count"]
assert len(FROZEN_FEATURES) == 12

T = "TRAIN"
C = "CAL"
X = "TEST"

# Expected block counts (specification Section 6; asserted at runtime).
EXPECTED_BLOCKS = {
    T: {"users": 784, "rows": 392784, "pos": 1037, "mal_users": 40},
    C: {"users": 108, "rows": 54108, "pos": 379, "mal_users": 15},
    X: {"users": 108, "rows": 54108, "pos": 476, "mal_users": 15},
}

# Pre-registered allocation of the 70 malicious users (specification
# Section 6, verbatim: user -> (split, positive rows, onset)).
PRE_REGISTERED_TABLE = {
    "CSC0217": (T, 2, "2010-06-10"), "PNL0301": (T, 51, "2010-06-14"),
    "LCC0819": (T, 56, "2010-06-16"), "GTD0219": (T, 2, "2010-06-17"),
    "RMW0542": (T, 59, "2010-06-21"), "AAF0535": (T, 54, "2010-06-28"),
    "IJM0776": (T, 58, "2010-07-06"), "RAR0725": (T, 45, "2010-07-06"),
    "MOS0047": (T, 58, "2010-07-15"), "EHB0824": (T, 8, "2010-07-22"),
    "JTM0223": (T, 2, "2010-07-22"), "DIB0285": (T, 50, "2010-07-26"),
    "BDV0168": (T, 12, "2010-07-30"), "LJR0523": (T, 12, "2010-07-31"),
    "EGD0132": (T, 58, "2010-08-02"), "PSF0133": (T, 59, "2010-08-02"),
    "AJR0932": (T, 9, "2010-09-10"), "RAB0589": (T, 11, "2010-09-13"),
    "LQC0479": (T, 9, "2010-09-14"), "MCF0600": (T, 4, "2010-09-20"),
    "BLS0678": (T, 10, "2010-09-21"), "MAS0025": (T, 2, "2010-09-29"),
    "BSS0369": (T, 2, "2010-09-30"), "EHD0584": (T, 7, "2010-10-02"),
    "RGG0064": (T, 8, "2010-10-20"), "AAM0658": (T, 7, "2010-10-23"),
    "TAP0551": (T, 7, "2010-10-23"), "ABC0174": (T, 59, "2010-10-27"),
    "MPM0220": (T, 2, "2010-11-04"), "GHL0460": (T, 1, "2010-11-09"),
    "DRR0162": (T, 55, "2010-11-11"), "HJB0742": (T, 7, "2010-11-19"),
    "MDH0580": (T, 59, "2011-01-04"), "FMG0527": (T, 8, "2011-01-05"),
    "FSC0601": (T, 59, "2011-01-18"), "JRG0207": (T, 8, "2011-01-19"),
    "DCH0843": (T, 1, "2011-02-04"), "CEJ0109": (T, 54, "2011-02-07"),
    "NWT0098": (T, 58, "2011-02-07"), "MAR0955": (T, 4, "2011-02-08"),
    "KPC0073": (C, 9, "2010-07-07"), "BIH0745": (C, 1, "2010-07-13"),
    "RHL0992": (C, 59, "2010-07-13"), "XHW0498": (C, 59, "2010-08-09"),
    "CAH0936": (C, 2, "2010-08-11"), "BBS0039": (C, 2, "2010-08-12"),
    "AKR0057": (C, 57, "2010-10-04"), "BTL0226": (C, 9, "2010-10-06"),
    "IUB0565": (C, 56, "2010-10-06"), "KRL0501": (C, 59, "2010-11-22"),
    "FTM0406": (C, 8, "2010-11-25"), "MSO0222": (C, 2, "2010-12-09"),
    "PPF0435": (C, 1, "2011-02-09"), "KLH0596": (C, 1, "2011-02-12"),
    "HBO0413": (C, 54, "2011-02-14"),
    "RKD0604": (X, 8, "2010-07-13"), "JMB0308": (X, 8, "2010-07-14"),
    "JGT0221": (X, 2, "2010-07-15"), "HXL0968": (X, 59, "2010-08-31"),
    "JJM0203": (X, 48, "2010-09-02"), "VSS0154": (X, 53, "2010-09-07"),
    "CCA0046": (X, 2, "2010-10-14"), "TNM0961": (X, 56, "2010-10-15"),
    "EDB0714": (X, 58, "2010-10-18"), "MYD0978": (X, 6, "2010-12-13"),
    "CCL0068": (X, 57, "2010-12-27"), "IKR0401": (X, 53, "2010-12-27"),
    "CQW0652": (X, 56, "2011-02-18"), "WDD0366": (X, 8, "2011-02-24"),
    "JLM0364": (X, 2, "2011-04-28"),
}


def wilson_ci(k: int, n: int, z: float = Z90) -> list[float]:
    """Wilson score interval for a proportion k/n (90% by default)."""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, (centre - half) / denom)),
            float(min(1.0, (centre + half) / denom))]


def gini_of_counts(counts: np.ndarray) -> float:
    """Gini over a per-user alert-count distribution (Phase 9/12 convention)."""
    c = np.asarray(counts, dtype=float)
    if c.sum() <= 0:
        return 0.0
    c = np.sort(c)
    n = len(c)
    return float((2 * np.sum(np.arange(1, n + 1) * c) / (n * c.sum()))
                 - (n + 1) / n)


# ---------------------------------------------------------------------------
# Allocation (pre-registered rule; no RNG)
# ---------------------------------------------------------------------------
def allocate_users(table: pd.DataFrame) -> dict[str, dict]:
    """Return {split: {users: [...], ...}} from the pre-registered rule.

    Malicious users: sorted by (first_positive_day, user_id); round-robin
    period 14: positions 0-7 TRAIN, 8-10 CAL, 11-13 TEST.
    Benign users: sorted by user_id; round-robin period 10: 0-7 TRAIN,
    8 CAL, 9 TEST.
    """
    table = table.sort_values(["user", "day"]).reset_index(drop=True)
    mal = table[table["is_malicious"] == 1]
    per = mal.groupby("user")["day"].agg(first="min", count="count")
    mal_users = sorted(per.index.tolist(),
                       key=lambda u: (per.loc[u, "first"], u))
    benign_users = sorted(set(table["user"].unique()) - set(mal_users))

    def rr(users: list[str], period: int, cal_pos: int, test_pos: int) -> dict:
        out = {}
        for i, u in enumerate(users):
            p = i % period
            out[u] = T if p < cal_pos else (C if p < test_pos else X)
        return out

    alloc = rr(mal_users, 14, 8, 11)
    alloc.update(rr(benign_users, 10, 8, 9))
    return alloc


def verify_allocation_table(table: pd.DataFrame) -> dict:
    """Assert computed allocation reproduces the pre-registered table."""
    alloc = allocate_users(table)
    mal = table[table["is_malicious"] == 1]
    per = mal.groupby("user")["day"].agg(first="min", count="count")
    for user, (split, pos, onset) in PRE_REGISTERED_TABLE.items():
        got = alloc[user]
        if got != split:
            raise AssertionError(f"{user}: computed split {got} != pre-registered {split}")
        if int(per.loc[user, "count"]) != pos:
            raise AssertionError(f"{user}: positive rows {per.loc[user, 'count']} != {pos}")
        if str(per.loc[user, "first"].date()) != onset:
            raise AssertionError(f"{user}: onset {per.loc[user, 'first']} != {onset}")
    if len(alloc) != 1000:
        raise AssertionError(f"allocation covers {len(alloc)} users; expected 1000")
    return alloc


def user_disjoint_blocks(table: pd.DataFrame,
                         expected: dict | None = None) -> dict:
    """Partition the full 501-day timelines by user; assert gate counts.

    expected: per-split (users, rows, positives, malicious_users) gate;
    defaults to the pre-registered real-data gate (specification Section 6).
    The real-data default is never weakened; overrides exist only for
    synthetic tests of the protocol wiring.
    """
    expected = expected or EXPECTED_BLOCKS
    alloc = allocate_users(table)
    blocks = {}
    for split in (T, C, X):
        users = sorted(u for u, s in alloc.items() if s == split)
        part = table[table["user"].isin(users)].copy()
        pos = int(part["is_malicious"].sum())
        mal_users = int(part.loc[part["is_malicious"] == 1, "user"].nunique())
        blocks[split] = {
            "users": users, "n_users": len(users),
            "rows": len(part), "positives": pos, "malicious_users": mal_users,
        }
        exp = expected[split]
        if (len(users), len(part), pos, mal_users) != (
                exp["users"], exp["rows"], exp["pos"], exp["mal_users"]):
            raise AssertionError(
                f"{split} block counts mismatch: got "
                f"({len(users)}, {len(part)}, {pos}, {mal_users}) "
                f"expected {tuple(exp.values())}")
    overlap = (set(blocks[T]["users"]) & set(blocks[C]["users"])
               | set(blocks[T]["users"]) & set(blocks[X]["users"])
               | set(blocks[C]["users"]) & set(blocks[X]["users"]))
    if overlap:
        raise AssertionError(f"users in more than one block: {sorted(overlap)}")
    return blocks


# ---------------------------------------------------------------------------
# Auxiliary model training (mirrors the frozen protocol, user-disjoint)
# ---------------------------------------------------------------------------
def train_uhold(table: pd.DataFrame, blocks: dict) -> tuple[lgb.Booster, dict]:
    """Fit lgbm-graph-v1-uhold on user-disjoint TRAIN, early stop on CAL."""
    if list(table.columns) != ["user", "day"] + FROZEN_FEATURES + ["is_malicious"]:
        raise AssertionError(
            f"table columns must be user/day/12-frozen-features/is_malicious; "
            f"got {list(table.columns)}")
    idx = {s: table["user"].isin(blocks[s]["users"]) for s in (T, C)}
    Xtr = table.loc[idx[T], FROZEN_FEATURES].astype(np.float32)
    ytr = table.loc[idx[T], "is_malicious"].to_numpy()
    Xcal = table.loc[idx[C], FROZEN_FEATURES].astype(np.float32)
    ycal = table.loc[idx[C], "is_malicious"].to_numpy()

    params = dict(default_params())
    params["scale_pos_weight"] = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    dtr = lgb.Dataset(Xtr, label=ytr)
    dcal = lgb.Dataset(Xcal, label=ycal, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dcal],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    record = {
        "model_name": AUX_NAME,
        "features": list(FROZEN_FEATURES),
        "params": params,
        "best_iteration": int(model.best_iteration),
        "early_stopping": f"user-disjoint CAL auc, patience {EARLY_STOPPING_ROUNDS}, "
                          f"up to {NUM_BOOST_ROUND} rounds",
        "train": {"rows": int(len(Xtr)), "malicious": int(ytr.sum())},
        "calibration": {"rows": int(len(Xcal)), "malicious": int(ycal.sum())},
    }
    return model, record


# ---------------------------------------------------------------------------
# User-block bootstrap (pre-registered primary CI estimator, spec Section 12)
# ---------------------------------------------------------------------------
def user_block_bootstrap_ci(users, y_true, y_score, metric: str = "auc_roc",
                            n_boot: int = N_BOOT, seed: int = SEED,
                            alpha: float = ALPHA) -> dict:
    """Percentile bootstrap over TEST *users* (rows pooled per resampled user).

    Resamples the 108 TEST users with replacement, pools each user's rows,
    recomputes the metric: honest for the panel structure (within-user row
    correlation). Evaluation-only; no refit.
    """
    users = np.asarray(users, dtype=object)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    unique = np.unique(users)
    idx_by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    values = []
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_user[u] for u in chosen])
        t, sc = y[idx], s[idx]
        if t.sum() == 0 or t.sum() == len(t):
            skipped += 1
            continue
        if metric == "auc_roc":
            values.append(roc_auc_score(t, sc))
        elif metric == "auc_pr":
            values.append(average_precision_score(t, sc))
        else:
            raise ValueError(f"unsupported metric: {metric}")
    values = np.asarray(values)
    if len(values) == 0:
        lo = hi = mean = None
    else:
        lo = float(np.percentile(values, 100 * alpha / 2))
        hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
        mean = float(values.mean())
    return {
        "method": "user-block percentile bootstrap (resample TEST users with "
                  "replacement, pool rows, recompute)",
        "metric": metric,
        "n_users": int(len(unique)),
        "n_boot": n_boot,
        "n_valid": int(len(values)),
        "n_skipped": skipped,
        "ci_low": lo,
        "ci_high": hi,
        "mean": mean,
    }


# ---------------------------------------------------------------------------
# User-level diagnostics (auxiliary TEST only, spec Section 11)
# ---------------------------------------------------------------------------
def user_diagnostics(keys: pd.DataFrame, y: np.ndarray, score: np.ndarray,
                     alert_primary: np.ndarray, table: pd.DataFrame,
                     scenario_of_user: dict) -> dict:
    """Per-user + stratified diagnostics on the auxiliary TEST block."""
    df = pd.DataFrame({
        "user": keys["user"].to_numpy(),
        "day": pd.to_datetime(keys["day"]).dt.normalize(),
        "y": np.asarray(y, dtype=int),
        "score": np.asarray(score, dtype=float),
        "alert": np.asarray(alert_primary, dtype=bool),
    })
    users = sorted(df["user"].unique())
    n_days_user = int(df["user"].nunique())

    # per-user aggregation over the user's full TEST timeline
    agg = {}
    for u in users:
        d = df[df["user"] == u]
        pos_days = sorted(d.loc[d["y"] == 1, "day"].tolist())
        alert_days = sorted(d.loc[d["alert"], "day"].tolist())
        agg[u] = {
            "user": u,
            "n_days": int(len(d)),
            "n_positive_days": len(pos_days),
            "n_alerts": len(alert_days),
            "max_score": float(d["score"].max()),
            "mean_score": float(d["score"].mean()),
        }

    # active-day counts (login_count > 0) over the user's full timeline
    active = (table[table["user"].isin(users)]
              .groupby("user")["login_count"].apply(lambda s: int((s > 0).sum())))
    activity_vals = active.sort_values().to_numpy()
    q1, q2 = (np.percentile(activity_vals, [100 / 3, 200 / 3]))
    def activity_tercile(u: str) -> str:
        a = int(active[u])
        return "low" if a <= q1 else ("mid" if a <= q2 else "high")

    # incident length class + onset half + first-positive day per user
    mal_users = sorted(set(df.loc[df["y"] == 1, "user"]))
    diag_rows = []
    for u in mal_users:
        d = df[df["user"] == u]
        pos_days = sorted(d.loc[d["y"] == 1, "day"].tolist())
        alert_days = sorted(d.loc[d["alert"], "day"].tolist())
        n_pos = len(pos_days)
        length = ("short" if n_pos <= SHORT_MAX else
                  "long" if n_pos >= LONG_MIN else "middle")
        onset = pos_days[0]
        half = "2010 H2" if onset.year == 2010 else "2011"
        first_alert = alert_days[0] if alert_days else None
        delay_days = (first_alert - onset).days if first_alert is not None else None
        pos_rank_of_alert = None
        if first_alert is not None:
            ranks = {d2: i + 1 for i, d2 in enumerate(pos_days)}
            pos_rank_of_alert = ranks.get(first_alert)
        first_week = bool(pos_rank_of_alert is not None and pos_rank_of_alert <= 7)
        first_tercile = bool(pos_rank_of_alert is not None
                             and pos_rank_of_alert <= int(np.ceil(n_pos / 3)))
        diag_rows.append({
            "user": u,
            "n_positive_days": n_pos,
            "length_class": length,
            "onset": str(onset.date()),
            "temporal_half": half,
            "scenario": scenario_of_user.get(u),
            "activity_tercile": activity_tercile(u),
            "n_alerts": len(alert_days),
            "detected": len(alert_days) > 0,
            "first_alert_day": str(first_alert.date()) if first_alert else None,
            "detection_delay_days": delay_days,
            "first_alert_positive_rank": pos_rank_of_alert,
            "detected_first_week": first_week,
            "detected_first_tercile": first_tercile,
            "fully_detected": bool(alert_days and len(set(alert_days) & set(pos_days)) == n_pos),
            "max_score": float(d["score"].max()),
        })

    detected = [r for r in diag_rows if r["detected"]]
    fully = [r["user"] for r in diag_rows if r["fully_detected"]]
    zero = [r["user"] for r in diag_rows if not r["detected"]]
    delays = [r["detection_delay_days"] for r in detected
              if r["detection_delay_days"] is not None]

    # user-level AUC: max score per user vs any-malicious-day label (all TEST
    # users, diagnostic aggregation only -- never an alerting unit)
    user_max = df.groupby("user")["score"].max().reindex(users)
    user_any = df.groupby("user")["y"].max().reindex(users).to_numpy()
    user_auc_roc = None
    if user_any.sum() not in (0, len(user_any)):
        user_auc_roc = float(roc_auc_score(user_any, user_max.to_numpy()))

    # alert concentration over alerted users (Gini, Phase 9 convention)
    alert_counts = (df[df["alert"]].groupby("user").size()
                    .reindex(df["user"].unique(), fill_value=0).to_numpy())
    alerted = df[df["alert"]]["user"].nunique()

    return {
        "n_test_users": n_days_user,
        "n_malicious_test_users": len(mal_users),
        "coverage": {
            "detected": len(detected),
            "total": len(mal_users),
            "frac": len(detected) / len(mal_users),
            "wilson_90": wilson_ci(len(detected), len(mal_users)),
        },
        "zero_detection_users": zero,
        "fully_detected_users": fully,
        "detection_delay_days": {
            "n_users_with_alert": len(delays),
            "min": min(delays) if delays else None,
            "median": float(np.median(delays)) if delays else None,
            "max": max(delays) if delays else None,
        },
        "first_week_detection": {
            "n_users": int(sum(1 for r in diag_rows if r["detected_first_week"])),
            "of_malicious": len(mal_users),
            "users": [r["user"] for r in diag_rows if r["detected_first_week"]],
        },
        "first_tercile_detection": {
            "n_users": int(sum(1 for r in diag_rows if r["detected_first_tercile"])),
            "users": [r["user"] for r in diag_rows if r["detected_first_tercile"]],
        },
        "fully_detected": {"n_users": len(fully), "users": fully},
        "user_level_auc_roc": user_auc_roc,
        "alert_concentration": {
            "n_alerts_total": int(df["alert"].sum()),
            "n_alerted_users": int(alerted),
            "gini": gini_of_counts(alert_counts),
        },
        "per_user": diag_rows,
        "strata": {
            "by_scenario": _stratum(diag_rows, "scenario"),
            "by_activity_tercile": _stratum(diag_rows, "activity_tercile"),
            "by_length_class": _stratum(diag_rows, "length_class"),
            "by_temporal_half": _stratum(diag_rows, "temporal_half"),
        },
    }


def _stratum(rows: list[dict], key: str) -> dict:
    """Detection per stratum level (descriptive, with counts; no tests)."""
    out = {}
    for r in rows:
        level = r[key]
        out.setdefault(str(level), []).append(r)
    return {k: {"n_users": len(v),
                "detected": int(sum(1 for r in v if r["detected"])),
                "users": [r["user"] for r in v],
                "detected_users": [r["user"] for r in v if r["detected"]]}
            for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


# ---------------------------------------------------------------------------
# Interpretation (pre-registered labels, spec Section 20)
# ---------------------------------------------------------------------------
def interpret(auc_roc: float, auc_pr: float, coverage_detected: int,
              chrono_test: dict) -> dict:
    """Descriptive labels vs the frozen chronological TEST record.

    chrono_test: {"auc_roc": ..., "auc_pr": ...} from the frozen record
    (specification Section 20; never a system decision).
    """
    droc = auc_roc - chrono_test["auc_roc"]
    if abs(droc) <= 0.02:
        roc = "comparable"
    elif droc < -0.05:
        roc = "materially worse"
    else:
        roc = "inconclusive"
    dpr = (auc_pr - chrono_test["auc_pr"]) / chrono_test["auc_pr"]
    if abs(dpr) <= 0.20:
        pr = "comparable"
    elif dpr < -0.40:
        pr = "materially worse"
    else:
        pr = "inconclusive"
    cov = ("high user-level coverage" if coverage_detected >= 12 else
           "moderate user-level coverage" if coverage_detected >= 8 else
           "low user-level coverage")
    return {
        "auc_roc_delta_vs_chronological": round(droc, 6),
        "auc_roc_label": roc,
        "auc_pr_delta_rel_vs_chronological": round(dpr, 6),
        "auc_pr_label": pr,
        "coverage_label": cov,
        "band_rule": ("AUC-ROC within +/-0.02 comparable; worse by >0.05 "
                      "materially worse; AUC-PR within +/-20% relative "
                      "comparable, worse by >40% relative materially worse; "
                      "coverage 12+/8-11/<=7 of 15"),
        "no_production_decision": ("Phase 14 makes no production decision; "
                                   "operating-point metrics are cohort-"
                                   "conditional and not directly comparable "
                                   "to the chronological record"),
    }


# ---------------------------------------------------------------------------
# Deterministic full protocol run (IO-free)
# ---------------------------------------------------------------------------
def run_experiment(table: pd.DataFrame, scenario_of_user: dict,
                   chrono_test: dict, input_md5: str | None = None,
                   return_artifacts: bool = False,
                   expected_blocks: dict | None = None,
                   label_gate: tuple[int, int] | None = None):
    """Full pre-registered Phase 14 protocol on the frozen merged table.

    Deterministic: fixed seeds (42) for both bootstrap estimators and the
    seeded LightGBM config; no timestamps. Callers run it twice and require
    bit-identical canonical JSON (rerun gate).

    return_artifacts=True additionally returns (pred_df, model) for artifact
    writing; the result dict is identical either way.

    expected_blocks / label_gate: real-data gates by default (never
    weakened); overrides exist only for synthetic protocol-wiring tests.
    """
    table = table.sort_values(["user", "day"]).reset_index(drop=True)
    blocks = user_disjoint_blocks(table, expected_blocks)

    # label reproduction gates (specification Section 9.4 / 19)
    label_gate = label_gate or (70, 1892)
    n_mal_users = int(table.loc[table["is_malicious"] == 1, "user"].nunique())
    n_mal_rows = int(table["is_malicious"].sum())
    if (n_mal_users, n_mal_rows) != label_gate:
        raise AssertionError(f"label reproduction failed: {n_mal_users} users / "
                             f"{n_mal_rows} rows != {label_gate}")

    # auxiliary model (evaluation-only)
    model, model_record = train_uhold(table, blocks)

    # CAL scores -> secondary operating point (descriptive only)
    Xcal = table.loc[table["user"].isin(blocks[C]["users"]), FROZEN_FEATURES]
    ycal = table.loc[table["user"].isin(blocks[C]["users"]), "is_malicious"].to_numpy()
    s_cal = model.predict(Xcal.astype(np.float32),
                          num_iteration=model.best_iteration)
    sec_threshold, sec_f1_cal = th.best_f1_threshold(ycal, s_cal)
    cal_metrics = {
        "rows": int(len(ycal)),
        "positives": int(ycal.sum()),
        "classification": m.classification_metrics(ycal, s_cal),
        "at_secondary_threshold": m.binary_decision_metrics(ycal, s_cal, sec_threshold),
        "secondary_threshold": sec_threshold,
        "secondary_f1_cal": sec_f1_cal,
        "best_iteration": int(model.best_iteration),
        "early_stopping_curve": {"note": "curve summary not retained; "
                                         "best_iteration recorded as outcome"},
    }

    # TEST scores + alerts (evaluated once per finalized protocol)
    Xte = table.loc[table["user"].isin(blocks[X]["users"]), FROZEN_FEATURES]
    yte = table.loc[table["user"].isin(blocks[X]["users"]), "is_malicious"].to_numpy()
    s_te = model.predict(Xte.astype(np.float32),
                         num_iteration=model.best_iteration)
    pred = table.loc[table["user"].isin(blocks[X]["users"]),
                     ["user", "day", "is_malicious"]].copy()
    pred["score"] = s_te
    pred["alert_primary"] = (s_te >= FROZEN_THRESHOLD).astype(int)
    pred["alert_secondary"] = (s_te >= sec_threshold).astype(int)

    # TEST metrics at both operating points (specification Section 10)
    cls = m.classification_metrics(yte, s_te)
    top_k = m.top_k_metrics(yte, s_te)
    prim = m.binary_decision_metrics(yte, s_te, FROZEN_THRESHOLD)
    sec = m.binary_decision_metrics(yte, s_te, sec_threshold)
    n_days_test = int(pred["day"].nunique())
    alerts_per_day = prim["n_alerts"] / n_days_test
    per_user_alerts = (pred[pred["alert_primary"] == 1].groupby("user").size())
    all_test_users = sorted(pred["user"].unique())
    alert_counts = per_user_alerts.reindex(all_test_users, fill_value=0)
    test_metrics = {
        "n_rows": int(len(pred)),
        "n_days": n_days_test,
        "n_positives": int(yte.sum()),
        "n_malicious_users": int(pred.loc[pred["is_malicious"] == 1, "user"].nunique()),
        "classification": cls,
        "top_k": top_k,
        "at_primary_threshold": prim,
        "at_secondary_threshold": sec,
        "alert_rate": {
            "per_user_day": prim["alert_rate"],
            "per_calendar_day": alerts_per_day,
        },
        "alerts_per_user": {
            "mean": float(alert_counts.mean()),
            "median": float(alert_counts.median()),
            "max": float(alert_counts.max()),
        },
        "gini_over_alerted_users": gini_of_counts(
            alert_counts[alert_counts > 0].to_numpy()),
        "primary_threshold_note": ("frozen production threshold applied as an "
                                   "immutable reference operating point"),
        "secondary_threshold_note": ("user-disjoint CAL best-F1 threshold; "
                                     "descriptive only, selects nothing"),
    }

    # bootstrap uncertainty (both estimators, seed 42)
    boot = {
        "row_level": {
            "auc_roc": m.bootstrap_ci(yte, s_te, "auc_roc", n_boot=N_BOOT, seed=SEED),
            "auc_pr": m.bootstrap_ci(yte, s_te, "auc_pr", n_boot=N_BOOT, seed=SEED),
        },
        "user_block": {
            "auc_roc": user_block_bootstrap_ci(pred["user"], yte, s_te, "auc_roc"),
            "auc_pr": user_block_bootstrap_ci(pred["user"], yte, s_te, "auc_pr"),
        },
        "n_boot": N_BOOT,
        "seed": SEED,
        "alpha": ALPHA,
        "note": ("user-block is the primary uncertainty statement for "
                 "user-generalization claims; row-level kept for "
                 "comparability (optimistic given within-user correlation)"),
    }

    # scenario counts per split + recall where scores exist (answer key,
    # evaluation-only; zero-positive scenarios reported, never imputed)
    def _scenario_counts(part: pd.DataFrame) -> dict:
        users = part["user"].to_numpy()
        yp = part["is_malicious"].to_numpy()
        out = {}
        for scn in sorted({v for v in scenario_of_user.values()}):
            mask = np.array([scenario_of_user.get(u, -1) == scn for u in users])
            n_pos = int((mask & (yp == 1)).sum())
            out[str(scn)] = {"n_malicious_rows": n_pos,
                             "n_users": int(mask.sum())}
        return out

    part_tr = table[table["user"].isin(blocks[T]["users"])]
    part_cal = table[table["user"].isin(blocks[C]["users"])]
    part_te = pred
    scenario = {
        T: _scenario_counts(part_tr),
        C: _scenario_counts(part_cal),
        X: _scenario_counts(part_te),
    }
    scenario[C]["recall_at_primary"] = _scenario_recall(
        part_cal, scenario_of_user, ycal, s_cal, FROZEN_THRESHOLD)
    scenario[C]["recall_at_secondary"] = _scenario_recall(
        part_cal, scenario_of_user, ycal, s_cal, sec_threshold)
    scenario[X]["recall_at_primary"] = _scenario_recall(
        part_te, scenario_of_user, yte, s_te, FROZEN_THRESHOLD)
    scenario[X]["recall_at_secondary"] = _scenario_recall(
        part_te, scenario_of_user, yte, s_te, sec_threshold)

    # diagnostics (auxiliary TEST only)
    diag = user_diagnostics(
        pred[["user", "day"]], yte, s_te,
        (s_te >= FROZEN_THRESHOLD), table, scenario_of_user)

    interpretation = interpret(cls["auc_roc"], cls["auc_pr"],
                               diag["coverage"]["detected"], chrono_test)

    result = {
        "experiment_id": "phase14-user-holdout-generalization",
        "auxiliary_model": AUX_NAME,
        "frozen_system": {
            "model": "lgbm-graph-v1",
            "features": FROZEN_FEATURES,
            "threshold": FROZEN_THRESHOLD,
            "seed": 42,
            "best_iteration_frozen": 186,
            "lightgbm": lgb.__version__,
        },
        "input_md5": input_md5,
        "blocks": {s: {k: v for k, v in b.items() if k != "users"}
                   for s, b in blocks.items()},
        "allocation": {s: sorted(blocks[s]["users"]) for s in (T, C, X)},
        "model": model_record,
        "calibration": cal_metrics,
        "test_metrics": test_metrics,
        "bootstrap": boot,
        "scenario": scenario,
        "user_diagnostics": diag,
        "interpretation": interpretation,
        "evidence_labels": {
            "allocation": "OBSERVED (frozen label column; pre-registered rule)",
            "block counts": "OBSERVED (asserted vs specification Section 6)",
            "model": "OBSERVED (auxiliary lgbm-graph-v1-uhold, seed 42; "
                     "evaluation-only)",
            "TEST metrics": "OBSERVED (auxiliary TEST, evaluated once per "
                            "finalized protocol)",
            "scenario mapping": "OBSERVED at execution from the answer key "
                                "(was NOT VERIFIED before execution)",
        },
        "test_evaluation_policy": ("auxiliary user-holdout TEST evaluated "
                                   "once per finalized protocol; the "
                                   "authoritative chronological TEST is "
                                   "never re-evaluated"),
    }
    if return_artifacts:
        return result, pred, model
    return result


def _scenario_recall(part: pd.DataFrame, scenario_of_user: dict,
                     y: np.ndarray, score: np.ndarray,
                     threshold: float) -> dict:
    """Per-scenario recall on a split at a given operating point.

    Scenario with zero positive rows in the split is reported with
    recall None (NOT VERIFIED, never imputed; specification Section 10).
    """
    users = part["user"].to_numpy()
    y = np.asarray(y, dtype=int)
    pred = np.asarray(score) >= threshold
    out = {}
    for scn in sorted({v for v in scenario_of_user.values()}):
        mask = np.array([scenario_of_user.get(u, -1) == scn for u in users])
        n_pos = int((mask & (y == 1)).sum())
        if n_pos == 0:
            out[str(scn)] = {"n_malicious_rows": 0, "n_users": int(mask.sum()),
                             "recall": None}
            continue
        out[str(scn)] = {
            "n_malicious_rows": n_pos,
            "n_users": int(mask.sum()),
            "recall": float((mask & (y == 1) & pred).sum()) / n_pos,
        }
    return out


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/date objects to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, np.str_)):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys) for bit-comparison."""
    return json.dumps(to_jsonable(obj), sort_keys=True, indent=2,
                      ensure_ascii=False)