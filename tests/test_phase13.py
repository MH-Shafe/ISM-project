"""Phase 13: temporal-stability analysis regression tests.

Protects (specification Section 21):
  1. pre-registered 16-window date table (Section 5, verbatim) + determinism
  2. unique-day window partition (hand-computed chunks, uneven tails, no
     gaps/overlaps, boundary integrity)
  3. unique-day gates (CAL 59 / TEST 47 / total 106, disjoint)
  4. pre-registered B1-B6 bound definitions (constants parse; per-bound
     pass/caution/fail with exact margin)
  5. verdict aggregation (PASS / CAUTION <= 2 marginal / FAIL >= 3 or hard)
  6. verdict isolation: CAL-only, raises on any non-CAL input
  7. Wilson 90% CI (hand-computed cases)
  8. annex consistency logic (descriptive, never merged into the verdict)
  9. artifact reloadability (skip-guarded): parquet schema, frozen alert
     counts, record sums, 15-day cross-check record
 10. determinism: analysis run twice -> identical canonical JSON
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config
from src.experiments import phase13 as p13

ART = config.ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stats(window, split="CAL", n_alerts=5, alerts_per_day=4.2, precision=0.7,
           recall=0.5, score_median=0.12, n_positives=10):
    return {"window": window, "split": split, "n_days": 7, "n_rows": 7000,
            "n_positives": n_positives, "n_alerts": n_alerts,
            "alerts_per_day": alerts_per_day, "alert_rate": n_alerts / 7000,
            "precision": precision, "recall": recall, "f1": 0.5, "mcc": 0.5,
            "zero_alert_window": n_alerts == 0,
            "zero_recall_window": n_positives > 0 and recall == 0,
            "score_median": score_median, "score_p90": 0.5, "score_p99": 0.6}


def _syn_dates(n_cal=59, n_test=47):
    cal = np.array([pd.Timestamp("2011-02-01") + pd.Timedelta(days=i)
                    for i in range(n_cal)], dtype=object)
    test = np.array([pd.Timestamp("2011-04-01") + pd.Timedelta(days=i)
                     for i in range(n_test)], dtype=object)
    return p13.normalized_dates(cal), p13.normalized_dates(test)


# ---------------------------------------------------------------------------
# 1. Pre-registered window table
# ---------------------------------------------------------------------------
def test_pre_registered_table_verbatim():
    windows = p13.build_windows()
    assert len(windows) == 16
    assert windows == p13.PRE_REGISTERED_WINDOWS


def test_pre_registered_uneven_tails():
    windows = p13.build_windows()
    cal = [w for w in windows if w["split"] == "CAL"]
    test = [w for w in windows if w["split"] == "TEST"]
    assert [w["n_days"] for w in cal] == [7] * 8 + [3]
    assert [w["n_days"] for w in test] == [7] * 6 + [5]
    assert sum(w["n_days"] for w in cal) == 59
    assert sum(w["n_days"] for w in test) == 47


def test_window_table_deterministic():
    assert p13.build_windows() == p13.build_windows()


def test_boundary_integrity():
    windows = p13.build_windows()
    assert windows[8] == {"index": 8, "split": "CAL", "start": "2011-03-29",
                          "end": "2011-03-31", "n_days": 3}
    assert windows[9] == {"index": 9, "split": "TEST", "start": "2011-04-01",
                          "end": "2011-04-07", "n_days": 7}
    assert all(w["end"] <= "2011-03-31" for w in windows[:9])
    assert all(w["start"] >= "2011-04-01" for w in windows[9:])


# ---------------------------------------------------------------------------
# 2. Unique-day partition
# ---------------------------------------------------------------------------
def test_partition_no_gaps_no_overlaps():
    cal, test = _syn_dates()
    windows = p13.build_windows()
    cal_w = [w for w in windows if w["split"] == "CAL"]
    test_w = [w for w in windows if w["split"] == "TEST"]
    mcal = p13.window_masks(cal, cal_w)
    mtest = p13.window_masks(test, test_w)
    assert (mcal.sum(axis=0) == 1).all()
    assert (mtest.sum(axis=0) == 1).all()
    assert mcal.sum() == 59
    assert mtest.sum() == 47


def test_partition_hand_computed_chunks():
    cal, _ = _syn_dates()
    windows = [w for w in p13.build_windows() if w["split"] == "CAL"]
    masks = p13.window_masks(cal, windows)
    assert masks[0].sum() == 7      # 2011-02-01..02-07
    assert masks[7].sum() == 7      # 2011-03-22..03-28
    assert masks[8].sum() == 3      # 2011-03-29..03-31 tail
    assert masks.sum(axis=1).tolist() == [7] * 8 + [3]


def test_partition_rejects_out_of_range_days():
    dates = np.array([p13._d("2011-03-31"), p13._d("2011-04-01")], dtype=object)
    windows = [w for w in p13.build_windows() if w["split"] == "CAL"]
    with pytest.raises(AssertionError):
        p13.window_masks(dates, windows)  # 04-01 in no CAL window -> gap


# ---------------------------------------------------------------------------
# 3. Unique-day gates
# ---------------------------------------------------------------------------
def test_unique_day_gates():
    cal, test = _syn_dates()
    cg = p13.cal_day_gate(59)
    tg = p13.test_day_gate(test)
    total = p13.total_day_gate(cg, {"test_days": tg["test_days"]})
    assert total == {"cal_days": 59, "test_days": 47, "total_days": 106}
    with pytest.raises(AssertionError):
        p13.cal_day_gate(58)
    with pytest.raises(AssertionError):
        p13.test_day_gate(np.array([], dtype=object))
    with pytest.raises(AssertionError):
        p13.total_day_gate({"cal_days": 59}, {"test_days": 46})


def test_test_day_gate_rejects_out_of_range():
    dates = np.array([p13._d("2011-06-01")] * 47, dtype=object)
    with pytest.raises(AssertionError):
        p13.test_day_gate(dates)


# ---------------------------------------------------------------------------
# 4. Pre-registered bounds
# ---------------------------------------------------------------------------
def test_bound_constants_parse():
    assert p13.RECALL_FLOOR == 0.25
    assert p13.PRECISION_FLOOR_FACTOR == 0.5
    assert p13.RATE_BAND_FACTORS == (0.5, 2.0)
    assert p13.MEDIAN_BAND_FACTORS == (0.5, 2.0)
    assert p13.MARGIN == 0.10
    assert p13.CAL_BLOCK["n_alerts"] == 244
    assert p13.CAL_BLOCK["n_days"] == 59
    assert p13.CAL_BLOCK["precision"] == 0.6844262295081968
    assert p13.TEST_BLOCK["n_alerts"] == 49
    assert abs(p13.TEST_BLOCK["alerts_per_day"] - 49 / 47) < 1e-12


def test_b1_binary_floor():
    c = p13.check_bounds([_stats(0, n_alerts=1)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B1"]["status"] == "pass"
    c = p13.check_bounds([_stats(0, n_alerts=0)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B1"]["status"] == "fail"  # boolean bound: no marginal state


def test_b2_band_hand_computed():
    rate = p13.CAL_BLOCK["alerts_per_day"]
    lo, hi = 0.5 * rate, 2.0 * rate
    c = p13.check_bounds([_stats(0, alerts_per_day=rate)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B2"]["status"] == "pass"
    c = p13.check_bounds([_stats(0, alerts_per_day=lo - 0.05 * lo)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B2"]["status"] == "caution"  # within 10% margin below lo
    c = p13.check_bounds([_stats(0, alerts_per_day=lo / 2)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B2"]["status"] == "fail"


def test_b3_recall_floor():
    c = p13.check_bounds([_stats(0, recall=0.25)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B3"]["status"] == "pass"  # boundary inclusive
    c = p13.check_bounds([_stats(0, recall=0.24)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B3"]["status"] == "caution"  # deviation 0.01 <= 0.025 margin
    c = p13.check_bounds([_stats(0, recall=0.2)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B3"]["status"] == "fail"


def test_b4_precision_floor_w8_case():
    floor = 0.5 * p13.CAL_BLOCK["precision"]  # 0.342213...
    c = p13.check_bounds([_stats(0, precision=floor)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B4"]["status"] == "pass"
    c = p13.check_bounds([_stats(0, precision=0.3333333333333333)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B4"]["status"] == "caution"  # W8 case: deviation 0.0089 < 0.0342
    c = p13.check_bounds([_stats(0, precision=0.2)], p13.CAL_BLOCK)[0]["bounds"]
    assert c["B4"]["status"] == "fail"


def test_b5_median_band_w8_case():
    lo = 0.5 * p13.CAL_BLOCK["score_median"]  # 0.0579782...
    c = p13.check_bounds([_stats(0, score_median=0.11595630377584766)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B5"]["status"] == "pass"
    c = p13.check_bounds([_stats(0, score_median=0.0530)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B5"]["status"] == "caution"  # deviation 0.0050 <= 0.0058 margin
    c = p13.check_bounds([_stats(0, score_median=0.041413463665337505)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B5"]["status"] == "fail"  # W8 case: beyond 10% margin


def test_b6_zero_recall():
    c = p13.check_bounds([_stats(0, recall=0.0, n_positives=5)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B6"]["status"] == "fail"
    c = p13.check_bounds([_stats(0, recall=0.0, n_positives=0)],
                         p13.CAL_BLOCK)[0]["bounds"]
    assert c["B6"]["status"] == "pass"  # NA
    assert "NA" in c["B6"]["note"]


# ---------------------------------------------------------------------------
# 5. Verdict aggregation
# ---------------------------------------------------------------------------
def _checks_9(mutate=None):
    checks = [{"window": i, "split": "CAL",
               "bounds": {f"B{j}": {"status": "pass", "value": 1, "reference": 1,
                                    "deviation": 0.0, "margin": 0.1, "note": ""}
                          for j in range(1, 7)}}
              for i in range(9)]
    if mutate:
        mutate(checks)
    return checks


def test_verdict_pass():
    assert p13.verdict_cal_only(_checks_9())["verdict"] == "PASS"


def test_verdict_caution_two_marginal():
    def m(checks):
        checks[0]["bounds"]["B4"]["status"] = "caution"
        checks[1]["bounds"]["B5"]["status"] = "caution"
    v = p13.verdict_cal_only(_checks_9(m))
    assert v["verdict"] == "CAUTION"
    assert v["n_violations"] == 2 and v["n_marginal"] == 2 and v["n_hard"] == 0


def test_verdict_fail_three_marginal():
    def m(checks):
        for i in range(3):
            checks[i]["bounds"]["B4"]["status"] = "caution"
    v = p13.verdict_cal_only(_checks_9(m))
    assert v["verdict"] == "FAIL"
    assert v["n_violations"] == 3


def test_verdict_fail_one_hard():
    def m(checks):
        checks[0]["bounds"]["B5"]["status"] = "fail"
    v = p13.verdict_cal_only(_checks_9(m))
    assert v["verdict"] == "FAIL"
    assert v["n_hard"] == 1


def test_verdict_rejects_test_windows():
    bad = _checks_9()
    bad[0]["split"] = "TEST"
    with pytest.raises(ValueError):
        p13.verdict_cal_only(bad)


def test_verdict_rejects_test_window_index():
    bad = _checks_9()
    bad[8]["window"] = 9
    with pytest.raises(ValueError):
        p13.verdict_cal_only(bad)


def test_verdict_rejects_wrong_count():
    with pytest.raises(ValueError):
        p13.verdict_cal_only(_checks_9()[:8])


# ---------------------------------------------------------------------------
# 6. Annex consistency (descriptive; separate from the verdict)
# ---------------------------------------------------------------------------
def test_annex_never_merges_into_verdict():
    cal = p13.check_bounds([_stats(i) for i in range(9)], p13.CAL_BLOCK)
    test = p13.check_bounds([_stats(i, split="TEST") for i in range(9, 16)],
                            p13.TEST_BLOCK)
    annex = p13.annex_consistency(test)
    assert len(annex["per_window"]) == 7
    assert annex["n_consistent"] + annex["n_inconsistent"] == 7
    assert all(row["window"] >= 9 for row in annex["per_window"])
    v = p13.verdict_cal_only(cal)
    assert "annex" not in v  # annex output cannot reach the verdict


def test_annex_counts_violations():
    rows = [{"window": 9, "split": "TEST",
             "bounds": {f"B{j}": {"status": "pass", "value": 1, "reference": 1,
                                  "deviation": 0.0, "margin": 0.1, "note": ""}
                        for j in range(1, 7)}}]
    rows[0]["bounds"]["B4"]["status"] = "fail"
    a = p13.annex_consistency(rows)
    assert a["n_consistent"] == 0 and a["n_inconsistent"] == 1
    assert a["per_window"][0]["violations"] == ["B4"]


# ---------------------------------------------------------------------------
# 7. Wilson 90% CI (hand-computed)
# ---------------------------------------------------------------------------
def test_wilson_ci_hand_computed():
    assert p13.wilson_ci(14, 49) == pytest.approx(
        (0.19842132118593642, 0.39543267207797805), abs=1e-12)
    assert p13.wilson_ci(17, 33) == pytest.approx(
        (0.38685137966940397, 0.641155475940724), abs=1e-12)
    assert p13.wilson_ci(3, 9) == pytest.approx(
        (0.19510495802089325, 0.5486062150856923), abs=1e-12)
    assert p13.wilson_ci(0, 10) == pytest.approx(
        (0.022672141311501293, 0.1902698287719057), abs=1e-12)


def test_wilson_ci_degenerate():
    assert p13.wilson_ci(0, 0) == (0.0, 0.0)
    assert p13.wilson_ci(50, 50) == pytest.approx(
        (0.9499843621952138, 0.998682452018516), abs=1e-12)


# ---------------------------------------------------------------------------
# 8. Artifact reloadability (skip-guarded until artifacts exist)
# ---------------------------------------------------------------------------
def test_artifact_test_predictions_schema():
    path = os.path.join(ART, "phase12_test_predictions.parquet")
    if not os.path.isfile(path):
        pytest.skip("phase13 artifacts not present yet")
    pred = pd.read_parquet(path)
    assert set(pred.columns) == {"user", "day", "is_malicious", "score",
                                 "alert_p0", "alert_selected", "selected_policy"}
    assert len(pred) == 47000
    assert int(pred["alert_p0"].sum()) == 49
    assert int(pred["alert_selected"].sum()) == 49
    assert (pred["alert_p0"] == pred["alert_selected"]).all()


def test_artifact_phase12_rolling_record():
    path = os.path.join(ART, "phase12_rolling_threshold.json")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    static = rec["by_window_size"]["7"]["windows_static"]
    assert len(static) == 9
    assert sum(w["n_days"] for w in static) == 59
    assert sum(w["n_rows"] for w in static) == 59000
    assert sum(w["n_positives"] for w in static) == 323
    assert sum(w["n_alerts"] for w in static) == 244


def test_artifact_phase9_15day_record():
    path = os.path.join(ART, "phase9_calibration.json")
    if not os.path.isfile(path):
        pytest.skip("phase9 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    pp = rec["temporal_stability"]["per_policy"]["frozen_max_f1"]
    assert [p["alert_count"] for p in pp] == [79, 70, 56, 39]
    assert [b["days"] for b in rec["temporal_stability"]["window_bounds"]] == \
        [15, 15, 15, 14]


# ---------------------------------------------------------------------------
# 9. cal_stats_from_record reconciliation
# ---------------------------------------------------------------------------
def test_cal_stats_from_record_reconciles():
    path = os.path.join(ART, "phase12_rolling_threshold.json")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    stats = p13.cal_stats_from_record(rec)
    assert [s["window"] for s in stats] == list(range(9))
    assert stats[8]["n_days"] == 3 and stats[8]["n_alerts"] == 9
    assert stats[8]["score_median"] == 0.041413463665337505


def test_cal_stats_rejects_tampered_sums():
    import copy
    path = os.path.join(ART, "phase12_rolling_threshold.json")
    if not os.path.isfile(path):
        pytest.skip("phase12 artifacts not present yet")
    with open(path) as fh:
        rec = json.load(fh)
    tampered = copy.deepcopy(rec)
    tampered["by_window_size"]["7"]["windows_static"][0]["n_alerts"] += 1
    with pytest.raises(AssertionError):
        p13.cal_stats_from_record(tampered)


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------
def test_bounds_and_checks_deterministic():
    a1 = p13.check_bounds([_stats(i) for i in range(9)], p13.CAL_BLOCK)
    a2 = p13.check_bounds([_stats(i) for i in range(9)], p13.CAL_BLOCK)
    assert p13.canonical_json(a1) == p13.canonical_json(a2)


def test_full_analysis_deterministic():
    r_path = os.path.join(ART, "phase12_rolling_threshold.json")
    t_path = os.path.join(ART, "phase12_test_predictions.parquet")
    p9_path = os.path.join(ART, "phase9_calibration.json")
    if not (os.path.isfile(r_path) and os.path.isfile(t_path)
            and os.path.isfile(p9_path)):
        pytest.skip("phase13 artifacts not present yet")
    with open(r_path) as fh:
        rolling = json.load(fh)
    with open(p9_path) as fh:
        phase9 = json.load(fh)
    test_df = pd.read_parquet(t_path)
    r1 = p13.run_analysis(rolling, test_df, phase9)
    r2 = p13.run_analysis(rolling, test_df, phase9)
    assert p13.canonical_json(r1) == p13.canonical_json(r2)
    assert r1["verdict"]["verdict"] in ("PASS", "CAUTION", "FAIL")
    assert r1["gates"]["total_days"] == 106


def test_verdict_cal_only_in_full_analysis():
    r_path = os.path.join(ART, "phase12_rolling_threshold.json")
    t_path = os.path.join(ART, "phase12_test_predictions.parquet")
    p9_path = os.path.join(ART, "phase9_calibration.json")
    if not (os.path.isfile(r_path) and os.path.isfile(t_path)
            and os.path.isfile(p9_path)):
        pytest.skip("phase13 artifacts not present yet")
    with open(r_path) as fh:
        rolling = json.load(fh)
    with open(p9_path) as fh:
        phase9 = json.load(fh)
    test_df = pd.read_parquet(t_path)
    res = p13.run_analysis(rolling, test_df, phase9)
    for c in res["cal_checks"]:
        assert c["split"] == "CAL" and c["window"] in range(9)
    for c in res["test_checks"]:
        assert c["split"] == "TEST" and c["window"] in range(9, 16)
    with pytest.raises(ValueError):
        p13.verdict_cal_only(res["test_checks"])  # TEST cannot reach the verdict