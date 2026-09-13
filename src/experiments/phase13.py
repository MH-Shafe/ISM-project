"""Phase 13: temporal-stability analysis of the frozen system.

Reads ONLY frozen outputs and published records (Phase 9/12); computes a
pre-registered descriptive temporal-stability verdict from CALIBRATION
windows (2011-02-01..2011-03-31, 9 x 7-day windows); TEST (2011-04-01..
2011-05-17, 7 windows) is a descriptive annex only and never contributes to
the verdict, any bound, or any system decision.

Safety guarantees enforced here:
  - windows are chunks of sorted UNIQUE CALENDAR DAYS, never row slices;
  - every day/row belongs to exactly one window (asserted);
  - the verdict function is architecturally CAL-only (raises on any non-CAL
    input; test-enforced);
  - TEST predictions are never regenerated (only frozen artifacts are read);
  - analysis output is deterministic (no timestamps/RNG inside run_analysis;
    the only RNG use is the seeded evaluation bootstrap, seed 42).

Reference: docs/phase13_specification.md (approved protocol).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    binary_decision_metrics,
    bootstrap_ci,
    classification_metrics,
)

# ---------------------------------------------------------------------------
# Pre-registered constants (from the approved specification, Section 25)
# ---------------------------------------------------------------------------
FROZEN_THRESHOLD = 0.9186015432508062  # referenced only; never re-scored here

# Published CALIBRATION block record (Phase 9 `frozen_max_f1`, 59 days).
CAL_BLOCK = {
    "split": "CAL",
    "n_days": 59,
    "n_rows": 59000,
    "n_positives": 323,
    "n_alerts": 244,
    "alerts_per_day": 244 / 59,  # 4.135593220338983
    "precision": 0.6844262295081968,
    "recall": 0.5170278637770898,
    "score_median": 0.11595630377584766,
}

# Published TEST block record (Phase 7/9/12; evaluated exactly once).
TEST_BLOCK = {
    "split": "TEST",
    "n_days": 47,
    "n_rows": 47000,
    "n_positives": 30,
    "n_alerts": 49,
    "alerts_per_day": 49 / 47,  # 1.0425531914893618
    "precision": 0.2857142857142857,
    "recall": 0.4666666666666667,
    "auc_roc": 0.9391565538286849,
    "auc_pr": 0.2677761042396373,
}

# Pre-registered bounds (specification Section 25; margin = 10% of the bound).
RECALL_FLOOR = 0.25
PRECISION_FLOOR_FACTOR = 0.5  # >= 0.5 x block precision
RATE_BAND_FACTORS = (0.5, 2.0)  # x block alerts-per-day
MEDIAN_BAND_FACTORS = (0.5, 2.0)  # x CAL block score median (0.11596)
MARGIN = 0.10
Z90 = 1.6448536269514722  # standard normal quantile for a 90% interval

# ---------------------------------------------------------------------------
# Pre-registered 16-window date table (specification Section 5, verbatim)
# ---------------------------------------------------------------------------
PRE_REGISTERED_WINDOWS = [
    {"index": 0, "split": "CAL", "start": "2011-02-01", "end": "2011-02-07", "n_days": 7},
    {"index": 1, "split": "CAL", "start": "2011-02-08", "end": "2011-02-14", "n_days": 7},
    {"index": 2, "split": "CAL", "start": "2011-02-15", "end": "2011-02-21", "n_days": 7},
    {"index": 3, "split": "CAL", "start": "2011-02-22", "end": "2011-02-28", "n_days": 7},
    {"index": 4, "split": "CAL", "start": "2011-03-01", "end": "2011-03-07", "n_days": 7},
    {"index": 5, "split": "CAL", "start": "2011-03-08", "end": "2011-03-14", "n_days": 7},
    {"index": 6, "split": "CAL", "start": "2011-03-15", "end": "2011-03-21", "n_days": 7},
    {"index": 7, "split": "CAL", "start": "2011-03-22", "end": "2011-03-28", "n_days": 7},
    {"index": 8, "split": "CAL", "start": "2011-03-29", "end": "2011-03-31", "n_days": 3},
    {"index": 9, "split": "TEST", "start": "2011-04-01", "end": "2011-04-07", "n_days": 7},
    {"index": 10, "split": "TEST", "start": "2011-04-08", "end": "2011-04-14", "n_days": 7},
    {"index": 11, "split": "TEST", "start": "2011-04-15", "end": "2011-04-21", "n_days": 7},
    {"index": 12, "split": "TEST", "start": "2011-04-22", "end": "2011-04-28", "n_days": 7},
    {"index": 13, "split": "TEST", "start": "2011-04-29", "end": "2011-05-05", "n_days": 7},
    {"index": 14, "split": "TEST", "start": "2011-05-06", "end": "2011-05-12", "n_days": 7},
    {"index": 15, "split": "TEST", "start": "2011-05-13", "end": "2011-05-17", "n_days": 5},
]

CAL_WINDOW_INDEX = range(0, 9)
TEST_WINDOW_INDEX = range(9, 16)


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


# ---------------------------------------------------------------------------
# Window construction (unique calendar days only; never row slices)
# ---------------------------------------------------------------------------
def build_windows() -> list[dict]:
    """Compute the 16 windows from calendar arithmetic and assert they equal
    the pre-registered table verbatim (safety gate 5)."""
    computed = []
    cal_days = [_d("2011-02-01") + dt.timedelta(days=i) for i in range(59)]
    test_days = [_d("2011-04-01") + dt.timedelta(days=i) for i in range(47)]
    for base, offset, split in ((cal_days, 0, "CAL"), (test_days, 9, "TEST")):
        for chunk in range(0, len(base), 7):
            part = base[chunk:chunk + 7]
            computed.append({
                "index": offset + chunk // 7,
                "split": split,
                "start": str(part[0]),
                "end": str(part[-1]),
                "n_days": len(part),
            })
    expected = [{k: w[k] for k in ("index", "split", "start", "end", "n_days")}
                for w in PRE_REGISTERED_WINDOWS]
    if computed != expected:
        raise AssertionError(
            "computed windows differ from the pre-registered table "
            "(specification Section 5)")
    return computed


def normalized_dates(day_col) -> np.ndarray:
    """Normalize a day column (datetime64/Timestamp/ISO str) to dt.date."""
    s = pd.to_datetime(pd.Series(day_col)).dt.normalize()
    return np.array([d.date() for d in s], dtype=object)


def window_masks(dates: np.ndarray, windows: list[dict]) -> np.ndarray:
    """Boolean (n_windows x n_rows) membership matrix over unique-day windows.

    Safety gates 3/4: asserts every day is in exactly one window (no gaps,
    no overlaps) and every window matches the pre-registered table exactly.
    """
    expected = {w["index"]: w for w in build_windows()}
    masks = np.zeros((len(windows), len(dates)), dtype=bool)
    for j, w in enumerate(windows):
        assert w == expected[w["index"]], "window does not match the pre-registered table"
        masks[j] = np.array([_d(w["start"]) <= d <= _d(w["end"]) for d in dates])
    membership = masks.sum(axis=0)
    if (membership != 1).any():
        raise AssertionError("every day must belong to exactly one window "
                             "(no gaps, no overlaps)")
    return masks


def cal_day_gate(cal_n_days_from_record: int) -> dict:
    """Safety gate 2 (CAL side, record-based): 59 unique calendar days."""
    if cal_n_days_from_record != 59:
        raise AssertionError(f"CAL must have exactly 59 unique days; got "
                             f"{cal_n_days_from_record}")
    return {"cal_days": cal_n_days_from_record}


def test_day_gate(test_dates: np.ndarray) -> dict:
    """Safety gate 2 (TEST side, row-based): 47 unique days in range."""
    days = sorted(set(test_dates))
    if len(days) != 47:
        raise AssertionError(f"TEST must have exactly 47 unique days; got {len(days)}")
    if not (_d("2011-04-01") <= days[0] and days[-1] <= _d("2011-05-17")):
        raise AssertionError("TEST days outside 2011-04-01..2011-05-17")
    return {"test_days": len(days)}


def total_day_gate(cal_gate: dict, test_gate: dict) -> dict:
    """Safety gate 2 (combined): 59 + 47 = 106 unique days, disjoint by
    construction (CAL ends 2011-03-31, TEST starts 2011-04-01)."""
    total = cal_gate["cal_days"] + test_gate["test_days"]
    if total != 106:
        raise AssertionError(f"CAL + TEST must total 106 unique days; got {total}")
    return {"cal_days": cal_gate["cal_days"], "test_days": test_gate["test_days"],
            "total_days": total}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n at confidence level z.

    Returns (lo, hi). Degenerate cases: n == 0 -> (0.0, 0.0). Pure
    arithmetic (no RNG), hand-checked in tests.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def _bound_result(status: str, value: Any, reference: Any, deviation: float,
                  margin: float, note: str = "") -> dict:
    return {"status": status, "value": value, "reference": reference,
            "deviation": deviation, "margin": margin, "note": note}


def _check_floor(value: float, floor: float) -> dict:
    if value >= floor:
        return _bound_result("pass", value, floor, 0.0, MARGIN * floor)
    deviation = floor - value
    return _bound_result("caution" if deviation <= MARGIN * floor else "fail",
                         value, floor, deviation, MARGIN * floor)


def _check_band(value: float, lo: float, hi: float) -> dict:
    if lo <= value <= hi:
        return _bound_result("pass", value, [lo, hi], 0.0, MARGIN * min(lo, hi))
    if value < lo:
        deviation, bound = lo - value, lo
    else:
        deviation, bound = value - hi, hi
    return _bound_result("caution" if deviation <= MARGIN * bound else "fail",
                         value, [lo, hi], deviation, MARGIN * bound)


# ---------------------------------------------------------------------------
# Bound checks (pre-registered B1-B6)
# ---------------------------------------------------------------------------
def check_bounds(stats: list[dict], block: dict) -> list[dict]:
    """Apply the pre-registered B1-B6 bounds to per-window statistics.

    B1  every window has >= 1 alert
    B2  per-window alerts-per-day in [0.5, 2.0] x block alerts-per-day
    B3  per-window recall >= 0.25 (NA when no positives in window)
    B4  per-window precision >= 0.5 x block precision
    B5  per-window score median in [0.5, 2.0] x CAL block median (0.11596)
    B6  no window with positives but zero recall

    block: CAL_BLOCK (verdict) or TEST_BLOCK (annex). B5 always uses the CAL
    median (specification Section 25).
    """
    rate_lo, rate_hi = (f * block["alerts_per_day"] for f in RATE_BAND_FACTORS)
    prec_floor = PRECISION_FLOOR_FACTOR * block["precision"]
    med_lo, med_hi = (f * CAL_BLOCK["score_median"] for f in MEDIAN_BAND_FACTORS)
    out = []
    for s in stats:
        b = {}
        b["B1"] = _check_floor(float(s["n_alerts"]), 1.0)
        b["B2"] = _check_band(s["alerts_per_day"], rate_lo, rate_hi)
        if s["n_positives"] > 0:
            b["B3"] = _check_floor(float(s["recall"]), RECALL_FLOOR)
            b["B6"] = _bound_result(
                "pass" if s["recall"] > 0 else "fail",
                s["recall"], 0.0,
                0.0 if s["recall"] > 0 else 1.0,
                MARGIN if s["recall"] > 0 else float("inf"),
                note="" if s["recall"] > 0 else "positives present but zero recall")
        else:
            b["B3"] = _bound_result("pass", None, RECALL_FLOOR, 0.0,
                                    MARGIN * RECALL_FLOOR,
                                    note="NA: no positives in window")
            b["B6"] = _bound_result("pass", None, 0.0, 0.0, 0.0,
                                    note="NA: no positives in window")
        b["B4"] = _check_floor(float(s["precision"]), prec_floor)
        b["B5"] = _check_band(s["score_median"], med_lo, med_hi)
        out.append({"window": s["window"], "split": s["split"], "bounds": b})
    return out


def aggregate_checks(checks: list[dict]) -> dict:
    """Counts of bound statuses across windows (per-bound and total)."""
    per_bound = {f"B{i}": {"pass": 0, "caution": 0, "fail": 0}
                 for i in range(1, 7)}
    for c in checks:
        for name, res in c["bounds"].items():
            per_bound[name][res["status"]] += 1
    return {"per_bound": per_bound,
            "n_violations": sum(v["caution"] + v["fail"] for v in per_bound.values()),
            "n_marginal": sum(v["caution"] for v in per_bound.values()),
            "n_hard": sum(v["fail"] for v in per_bound.values())}


def verdict_cal_only(checks: list[dict]) -> dict:
    """Pre-registered verdict rule over CAL windows ONLY (specification 25).

    PASS    all B1-B6 hold on all 9 CAL windows
    CAUTION <= 2 bound violations, each within 10% margin of the bound
    FAIL    >= 3 violations, or any violation beyond 10% margin

    Architectural isolation: raises ValueError on any non-CAL input (TEST
    windows, wrong count) so TEST can never reach the verdict
    (specification 24; test-enforced).
    """
    for c in checks:
        if c["split"] != "CAL" or c["window"] not in CAL_WINDOW_INDEX:
            raise ValueError(
                f"verdict input must be CAL-only; got window={c.get('window')} "
                f"split={c.get('split')}")
    if len(checks) != 9:
        raise ValueError(f"verdict requires exactly 9 CAL windows; got {len(checks)}")
    agg = aggregate_checks(checks)
    if agg["n_hard"] > 0 or agg["n_violations"] >= 3:
        verdict = "FAIL"
    elif agg["n_violations"] == 0:
        verdict = "PASS"
    else:
        verdict = "CAUTION"
    return {"verdict": verdict, **agg}


def annex_consistency(checks: list[dict]) -> dict:
    """Descriptive TEST annex: same bounds, reported consistent/inconsistent
    per window; never merged into the verdict (specification 25)."""
    rows = []
    for c in checks:
        rows.append({
            "window": c["window"],
            "consistent": all(r["status"] == "pass" for r in c["bounds"].values()),
            "violations": [name for name, r in c["bounds"].items()
                           if r["status"] != "pass"],
        })
    return {"per_window": rows,
            "n_consistent": sum(r["consistent"] for r in rows),
            "n_inconsistent": sum(not r["consistent"] for r in rows)}


# ---------------------------------------------------------------------------
# Window statistics
# ---------------------------------------------------------------------------
def cal_stats_from_record(rec: dict) -> list[dict]:
    """Normalize the Phase 12 windows_static record into window stat dicts.

    Reconciles the frozen record against the published CAL block (59 days,
    59,000 rows, 323 positives, 244 alerts) - fails fast on any mismatch.
    """
    static = rec["by_window_size"]["7"]["windows_static"]
    out = []
    for w in static:
        out.append({
            "window": w["window"], "split": "CAL",
            "n_days": w["n_days"], "n_rows": w["n_rows"],
            "n_positives": w["n_positives"], "n_alerts": w["n_alerts"],
            "alerts_per_day": w["n_alerts"] / w["n_days"],
            "alert_rate": w["alert_rate"],
            "precision": w["precision"], "recall": w["recall"],
            "f1": w["f1"], "mcc": w["mcc"],
            "zero_alert_window": w["zero_alert_window"],
            "zero_recall_window": w["zero_recall_window"],
            "score_median": w["score_median"],
            "score_p90": w["score_p90"], "score_p99": w["score_p99"],
            "balanced_accuracy": None, "fpr": None, "fnr": None,
            "note": "balanced_accuracy/fpr/fnr NOT VERIFIED: no per-row CAL "
                    "scores in frozen records (optional scoring pass not run)",
        })
    sums = {k: sum(w[k] for w in out)
            for k in ("n_days", "n_rows", "n_positives", "n_alerts")}
    expected = {"n_days": CAL_BLOCK["n_days"], "n_rows": CAL_BLOCK["n_rows"],
                "n_positives": CAL_BLOCK["n_positives"],
                "n_alerts": CAL_BLOCK["n_alerts"]}
    if sums != expected:
        raise AssertionError(f"phase12 windows_static sums mismatch: {sums} != {expected}")
    if [w["window"] for w in out] != list(range(9)):
        raise AssertionError("phase12 windows_static must cover windows 0..8")
    return out


def _tp_from_record(s: dict) -> int:
    """True positives from the frozen record (precision x alerts ==
    recall x positives up to rounding); asserts consistency."""
    tp_a = s["precision"] * s["n_alerts"]
    tp_b = s["recall"] * s["n_positives"]
    if abs(tp_a - tp_b) > 1e-6:
        raise AssertionError(
            f"window {s['window']}: precision x alerts ({tp_a}) != "
            f"recall x positives ({tp_b})")
    return int(round(tp_a))


def _row_stats(df: pd.DataFrame, windows: list[dict]) -> list[dict]:
    dates = normalized_dates(df["day"])
    masks = window_masks(dates, windows)
    scores = df["score"].to_numpy(dtype=float)
    y = df["is_malicious"].to_numpy(dtype=int)
    out = []
    for j, w in enumerate(windows):
        idx = masks[j]
        if not idx.any():
            raise AssertionError(f"window {w['index']} has no rows")
        if idx.sum() != 1000 * w["n_days"]:
            raise AssertionError(
                f"window {w['index']} must contain 1000 rows/day "
                f"(user x day grid); got {int(idx.sum())}")
        m = binary_decision_metrics(y[idx], scores[idx], FROZEN_THRESHOLD)
        n_pos = int(m["tp"] + m["fn"])
        out.append({
            "window": w["index"], "split": w["split"],
            "n_days": w["n_days"], "n_rows": int(idx.sum()),
            "n_positives": n_pos,
            "n_alerts": int(m["n_alerts"]),
            "alerts_per_day": m["n_alerts"] / w["n_days"],
            "alert_rate": m["alert_rate"],
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "mcc": m["mcc"],
            "balanced_accuracy": m["balanced_accuracy"],
            "fpr": m["fpr"], "fnr": m["fnr"],
            "zero_alert_window": m["n_alerts"] == 0,
            "zero_recall_window": n_pos > 0 and m["recall"] == 0,
            "precision_90": list(wilson_ci(int(m["tp"]), int(m["n_alerts"]))),
            "recall_90": list(wilson_ci(int(m["tp"]), n_pos)),
            "score_median": float(np.median(scores[idx])),
            "score_p90": float(np.percentile(scores[idx], 90)),
            "score_p99": float(np.percentile(scores[idx], 99)),
        })
    return out


def test_stats_from_rows(df: pd.DataFrame) -> dict:
    """TEST window stats + block reconciliation against the frozen record.

    Safety gates: TEST rows must span exactly the 7 TEST windows; the frozen
    alert columns (alert_p0, alert_selected - P0 was retained) must agree
    with the published 49-alert record; block AUCs must equal the record.
    """
    expected = build_windows()
    test_windows = [w for w in expected if w["split"] == "TEST"]
    if len(df) != 47000:
        raise AssertionError(f"TEST rows must be 47,000; got {len(df)}")
    for col in ("alert_p0", "alert_selected"):
        if int(df[col].sum()) != TEST_BLOCK["n_alerts"]:
            raise AssertionError(f"frozen {col} count {int(df[col].sum())} "
                                 f"!= published {TEST_BLOCK['n_alerts']}")
    if not (df["alert_p0"] == df["alert_selected"]).all():
        raise AssertionError("alert_p0 != alert_selected (P0 was retained in Phase 12)")
    days = normalized_dates(df["day"])
    test_gate = test_day_gate(days)
    masks = window_masks(days, test_windows)
    if not (masks.sum(axis=0) == 1).all():
        raise AssertionError("every TEST row must fall inside exactly one TEST window")
    return {"block": _block_stats(df), "windows": _row_stats(df, test_windows),
            "unique_days_test": test_gate["test_days"]}


def _block_stats(df: pd.DataFrame) -> dict:
    y = df["is_malicious"].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    cls = classification_metrics(y, s)
    pol = binary_decision_metrics(y, s, FROZEN_THRESHOLD)
    ci_roc = bootstrap_ci(y, s, "auc_roc", n_boot=1000, seed=42)
    ci_pr = bootstrap_ci(y, s, "auc_pr", n_boot=1000, seed=42)
    if abs(cls["auc_roc"] - TEST_BLOCK["auc_roc"]) > 1e-12 or \
            abs(cls["auc_pr"] - TEST_BLOCK["auc_pr"]) > 1e-12:
        raise AssertionError("frozen TEST AUCs differ from the published record")
    if pol["n_alerts"] != TEST_BLOCK["n_alerts"] or \
            pol["tp"] + pol["fn"] != TEST_BLOCK["n_positives"]:
        raise AssertionError("frozen TEST policy metrics differ from the published record")
    return {"n_rows": int(len(df)), "n_positives": int(y.sum()),
            "n_alerts": pol["n_alerts"],
            "alerts_per_day": pol["n_alerts"] / 47.0,
            "precision": pol["precision"], "recall": pol["recall"],
            "f1": pol["f1"], "mcc": pol["mcc"],
            "balanced_accuracy": pol["balanced_accuracy"],
            "fpr": pol["fpr"], "fnr": pol["fnr"],
            "auc_roc": cls["auc_roc"], "auc_pr": cls["auc_pr"],
            "bootstrap_ci": {"auc_roc": ci_roc, "auc_pr": ci_pr}}


def cross_check_15d(phase9_rec: dict) -> dict:
    """15-day CAL cross-check from the Phase 9 record (4 windows)."""
    stab = phase9_rec["temporal_stability"]
    per_policy = stab["per_policy"]["frozen_max_f1"]
    bounds = stab["window_bounds"]
    if len(per_policy) != 4 or len(bounds) != 4:
        raise AssertionError("phase9 15-day record must have 4 windows")
    stats = []
    for i, (p, b) in enumerate(zip(per_policy, bounds)):
        stats.append({
            "window": i, "split": "CAL",
            "start": b["first_day"], "end": b["last_day"],
            "n_days": b["days"], "n_alerts": p["alert_count"],
            "alerts_per_day": p["alert_count"] / b["days"],
            "precision": p["precision"], "recall": p["recall"],
            "score_median": None,  # not present in the Phase 9 record
            "note": "score median / n_positives not in Phase 9 record; "
                    "B5 NOT VERIFIED on this grid",
        })
    if sum(s["n_alerts"] for s in stats) != CAL_BLOCK["n_alerts"]:
        raise AssertionError("phase9 15-day alert counts must sum to 244")
    checks = []
    rate_lo, rate_hi = (f * CAL_BLOCK["alerts_per_day"] for f in RATE_BAND_FACTORS)
    for s in stats:
        b = {
            "B1": _check_floor(float(s["n_alerts"]), 1.0),
            "B2": _check_band(s["alerts_per_day"], rate_lo, rate_hi),
            "B3": _check_floor(float(s["recall"]), RECALL_FLOOR),
            "B4": _check_floor(float(s["precision"]),
                               PRECISION_FLOOR_FACTOR * CAL_BLOCK["precision"]),
            "B5": _bound_result("pass", None, None, 0.0, 0.0,
                                note="B5 NOT VERIFIED: no medians in Phase 9 record"),
            "B6": _bound_result("pass", None, 0.0, 0.0, 0.0,
                                note="B6 NA: no zero-recall flag in Phase 9 record"),
        }
        checks.append({"window": s["window"], "split": "CAL", "bounds": b})
    return {"windows": stats, "checks": checks}


# ---------------------------------------------------------------------------
# Deterministic full analysis
# ---------------------------------------------------------------------------
def run_analysis(phase12_rolling: dict, test_df: pd.DataFrame,
                 phase9_rec: dict) -> dict:
    """Full deterministic Phase 13 analysis (no IO, no timestamps, no RNG
    except the seeded evaluation bootstrap; bit-reproducible)."""
    windows = build_windows()

    cal_stats = cal_stats_from_record(phase12_rolling)
    cal_gate = cal_day_gate(sum(s["n_days"] for s in cal_stats))
    cal_checks = check_bounds(cal_stats, CAL_BLOCK)
    cal_agg = aggregate_checks(cal_checks)
    verdict = verdict_cal_only(cal_checks)

    test_stats = test_stats_from_rows(test_df)
    test_checks = check_bounds(test_stats["windows"], TEST_BLOCK)
    test_agg = aggregate_checks(test_checks)
    annex = annex_consistency(test_checks)
    total_gate = total_day_gate(
        cal_gate, {"test_days": test_stats["unique_days_test"]})

    x15 = cross_check_15d(phase9_rec)
    x15_agg = aggregate_checks(x15["checks"])

    wilson_cal = []
    for s in cal_stats:
        tp = _tp_from_record(s)
        wilson_cal.append({
            "window": s["window"], "n_alerts": s["n_alerts"],
            "n_positives": s["n_positives"],
            "precision_90": list(wilson_ci(tp, s["n_alerts"])),
            "recall_90": list(wilson_ci(tp, s["n_positives"])),
        })

    return {
        "experiment_id": "phase13-temporal-stability",
        "scope": "descriptive temporal-stability of the frozen system over all "
                 "model-unseen days (CAL 59 + TEST 47 = 106 days, 353 positives)",
        "inputs": [
            "reports/artifacts/phase12_rolling_threshold.json (CAL 7-day windows)",
            "reports/artifacts/phase12_test_predictions.parquet (frozen TEST rows, reused)",
            "reports/artifacts/phase9_calibration.json (15-day CAL cross-check)",
        ],
        "frozen_system": {"model": "lgbm-graph-v1", "threshold": FROZEN_THRESHOLD,
                          "best_iteration": 186, "seed": 42, "lightgbm": "4.6.0"},
        "windows": {"table": windows,
                    "gate": "computed == pre-registered (verbatim)"},
        "gates": total_gate | {
            "partition": "every day/row in exactly one window (asserted)",
            "boundary": "no window crosses 2011-03-31/2011-04-01",
            "test_regenerated": False,
        },
        "cal_block": CAL_BLOCK,
        "cal_windows": cal_stats,
        "cal_checks": cal_checks,
        "cal_aggregate": cal_agg,
        "verdict": verdict,
        "test_block": test_stats["block"],
        "test_windows": test_stats["windows"],
        "test_checks": test_checks,
        "test_aggregate": test_agg,
        "annex": annex,
        "cross_check_15d": x15,
        "cross_check_15d_aggregate": x15_agg,
        "uncertainty": {
            "wilson_90_cal": wilson_cal,
            "note": "Wilson 90% intervals from frozen counts (CAL) and frozen "
                    "rows (TEST); block bootstrap on TEST only (seed 42, n=1000)",
        },
        "test_evaluation_policy": "TEST evaluated exactly once (Phase 7/9/12 "
                                  "record); Phase 13 annex is descriptive only "
                                  "(specification Section 7/24)",
        "evidence_labels": {
            "CAL stats": "OBSERVED (frozen Phase 12 record, md5-verified)",
            "CAL verdict": "OBSERVED (pre-registered B1-B6 on frozen records)",
            "TEST annex": "OBSERVED (descriptive re-analysis of frozen parquet)",
            "15-day cross-check": "OBSERVED (frozen Phase 9 record)",
            "B5 on 15-day grid": "NOT VERIFIED (no medians in Phase 9 record)",
            "CAL per-row partition": "NOT VERIFIED (no per-row CAL data locally; "
                                     "record-based assertions only)",
            "CAL balanced acc/FPR/FNR": "NOT VERIFIED (not in frozen records)",
        },
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and dates to Python JSON-safe
    values (keeps the analysis output serializable and deterministic)."""
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
    if isinstance(obj, (dt.date, pd.Timestamp, np.str_)):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys) for bit-comparison."""
    return json.dumps(to_jsonable(obj), sort_keys=True, indent=2,
                      ensure_ascii=False)