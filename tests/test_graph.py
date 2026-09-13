"""Phase 5: leakage-safe graph feature tests.

Covers: edge construction, temporal filtering, historical-window
correctness, user-day aggregation, unseen entities, cold start, empty
history, duplicate interactions, deterministic output, key alignment,
department month rule, and the mandatory future-invariance / same-day
exclusion leakage tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.graph import features as g

P_OLE2 = "D0-CF-11-E0-A1-B1-1A-E1"   # real declared prefixes
P_PDF = "25-50-44-46-2D"
P_ZIP = "50-4B-03-04"

USERS = ["U1", "U2", "U3"]


def _days(n=10):
    return pd.date_range("2010-01-01", periods=n, freq="D")


def _dt(day, hh=8):
    return f"01/{day:02d}/2010 {hh:02d}:00:00"


def _lon(rows):
    return pd.DataFrame(rows, columns=["id", "date", "user", "pc", "activity"])


def _fil(rows):
    return pd.DataFrame(rows, columns=["id", "date", "user", "pc", "filename", "content"])


def _ldap(*months):
    out = []
    for m, d in months:
        out.append((m, pd.DataFrame({"user_id": list(d), "department": list(d.values())})))
    return out


def build(logon_rows, file_rows, ldap=None, users=None, days=None):
    return g.build_graph_features(
        _lon(logon_rows), _fil(file_rows), ldap or [],
        users if users is not None else USERS,
        days if days is not None else _days())


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------
def test_first_use_days_min_and_dedup():
    ev = pd.DataFrame({"user": ["U1", "U1", "U1", "U2"],
                       "pc": ["PC-0001", "PC-0001", "PC-0002", "PC-0001"],
                       "day": pd.to_datetime(["2010-01-03", "2010-01-02",
                                              "2010-01-05", "2010-01-02"])})
    fu = g.first_use_days(ev, "pc")
    assert len(fu) == 3
    row = fu[(fu["user"] == "U1") & (fu["pc"] == "PC-0001")].iloc[0]
    assert row["first_day"] == pd.Timestamp("2010-01-02")


def test_content_type_prefixes_and_other():
    s = pd.Series([P_OLE2 + "-X", P_PDF + "-Y", "AA-BB", P_ZIP + "-Z", ""])
    t = g.content_type(s)
    assert t.tolist() == [P_OLE2, P_PDF, "other", P_ZIP, "other"]


# ---------------------------------------------------------------------------
# Temporal filtering / historical window
# ---------------------------------------------------------------------------
def test_past_set_size_strictly_before():
    fu = pd.DataFrame({"user": ["U1", "U1", "U1"],
                       "first_day": pd.to_datetime(["2010-01-01", "2010-01-03",
                                                    "2010-01-05"])})
    days = pd.date_range("2010-01-01", "2010-01-08", freq="D")
    past = g.past_set_size(fu, "user", ["U1"], days).set_index("day")["n_past"]
    assert past.loc[pd.Timestamp("2010-01-01")] == 0   # same day excluded
    assert past.loc[pd.Timestamp("2010-01-02")] == 1
    assert past.loc[pd.Timestamp("2010-01-04")] == 2
    assert past.loc[pd.Timestamp("2010-01-06")] == 3


def test_jaccard_known():
    out = g._jaccard([2, 1, 0], [2, 3, 5], [1, 0, 0])
    assert out[0] == pytest.approx(1 / 3)   # |A∩B|/|A∪B| = 1/3
    assert out[1] == 0.0                    # empty intersection
    assert out[2] == 0.0                    # empty day set -> 0


# ---------------------------------------------------------------------------
# Core leakage tests (mandatory)
# ---------------------------------------------------------------------------
def test_future_events_do_not_change_past_features():
    """Add events strictly after day D; (user, D) features must not change."""
    logon_rows = [
        ("L1", _dt(2), "U1", "PC-0001", "Logon"),
        ("L2", _dt(3), "U1", "PC-0001", "Logon"),
        ("L3", _dt(3), "U1", "PC-0002", "Logon"),
    ]
    file_rows = [
        ("F1", _dt(2), "U1", "PC-0001", "f1", P_OLE2),
    ]
    logon_future = logon_rows + [
        ("L4", _dt(6), "U1", "PC-9999", "Logon"),
        ("L5", _dt(9), "U1", "PC-8888", "Logon"),
    ]
    file_future = file_rows + [
        ("F2", _dt(6), "U1", "PC-0001", "f2", P_PDF),
    ]
    days = _days(10)
    before = build(logon_rows, file_rows, days=days)
    after = build(logon_future, file_future, days=days)
    for d in [pd.Timestamp("2010-01-02"), pd.Timestamp("2010-01-03")]:
        b = before[(before["user"] == "U1") & (before["day"] == d)].iloc[0]
        a = after[(after["user"] == "U1") & (after["day"] == d)].iloc[0]
        for c in g_fcols():
            assert b[c] == a[c], f"future leakage in {c} on {d}"


def test_same_day_events_affect_day_terms_but_not_past_terms():
    """Same-day events may change day-set terms; historical terms stay past-only."""
    rows = [
        ("L1", _dt(2), "U1", "PC-0001", "Logon"),
        ("L2", _dt(3), "U1", "PC-0001", "Logon"),
        ("L3", _dt(3), "U1", "PC-0002", "Logon"),   # PC-0002 first used day 3
        ("L4", _dt(4), "U1", "PC-0001", "Logon"),
    ]
    rows_plus_same_day = rows + [
        ("L5", _dt(4), "U1", "PC-0003", "Logon"),   # same-day NEW device
    ]
    days = _days(6)
    t = build(rows, [], days=days)
    t2 = build(rows_plus_same_day, [], days=days)
    # day-3 features unchanged (no events after day 3 touched)
    for c in g_fcols():
        a = t[(t["user"] == "U1") & (t["day"] == pd.Timestamp("2010-01-03"))][c].iloc[0]
        b = t2[(t2["user"] == "U1") & (t2["day"] == pd.Timestamp("2010-01-03"))][c].iloc[0]
        assert a == b
    # day-4: PC-0003 added same-day -> it is NOT in the past set for day 4
    d4 = t2[(t2["user"] == "U1") & (t2["day"] == pd.Timestamp("2010-01-04"))].iloc[0]
    assert d4["rare_device_usage_count"] == 1          # PC-0003 new-to-user
    assert d4["device_consistency_score"] == pytest.approx(1 / 3)  # {PC-0001} ∩ {PC-0001,PC-0003}


def test_department_history_is_strictly_past():
    """A type accessed by the department on day D itself is not dept history for D."""
    ldap = _ldap(("2009-12", {"U1": "DeptA", "U2": "DeptA"}))
    rows = [
        ("F1", _dt(2), "U2", "PC-0001", "f1", P_OLE2),   # DeptA first sees OLE2 on day 2
        ("F2", _dt(2), "U1", "PC-0001", "f2", P_OLE2),   # U1 accesses OLE2 same day 2
        ("F3", _dt(3), "U1", "PC-0001", "f3", P_OLE2),   # day 3: OLE2 now in dept past
        ("F4", _dt(3), "U1", "PC-0001", "f4", P_PDF),    # day 3: PDF never seen by dept
    ]
    t = build([], rows, ldap=ldap)
    u1 = t[t["user"] == "U1"].set_index("day")
    assert u1.loc[pd.Timestamp("2010-01-02"), "department_file_type_mismatch_count"] == 1
    assert u1.loc[pd.Timestamp("2010-01-03"), "department_file_type_mismatch_count"] == 1  # PDF
    assert u1.loc[pd.Timestamp("2010-01-04"), "department_file_type_mismatch_count"] == 0


# ---------------------------------------------------------------------------
# Department month rule
# ---------------------------------------------------------------------------
def test_department_by_day_uses_previous_month_snapshot():
    ldap = _ldap(("2009-12", {"U1": "DeptDec"}), ("2010-01", {"U1": "DeptJan"}))
    days = pd.to_datetime(["2010-01-05", "2010-02-05"])
    dept = g.department_by_day(ldap, ["U1"], days)
    dept = dept.set_index("day")
    assert dept.loc[pd.Timestamp("2010-01-05"), "department"] == "DeptDec"
    assert dept.loc[pd.Timestamp("2010-02-05"), "department"] == "DeptJan"


def test_user_without_ldap_has_no_department_feature():
    ldap = _ldap(("2009-12", {"U1": "DeptA"}))
    rows = [("F1", _dt(3), "U2", "PC-0001", "f1", P_OLE2)]
    t = build([], rows, ldap=ldap)
    u2 = t[(t["user"] == "U2") & (t["day"] == pd.Timestamp("2010-01-03"))].iloc[0]
    assert u2["department_file_type_mismatch_count"] == 0


# ---------------------------------------------------------------------------
# Unseen entities / cold start / empty history
# ---------------------------------------------------------------------------
def test_cold_start_user_and_empty_history():
    """No history -> Jaccard 0; first-ever device on day D is rare."""
    rows = [
        ("L1", _dt(1), "U1", "PC-0001", "Logon"),
        ("L2", _dt(1), "U1", "PC-0002", "Logon"),
    ]
    t = build(rows, [])
    u1 = t[(t["user"] == "U1") & (t["day"] == pd.Timestamp("2010-01-01"))].iloc[0]
    assert u1["device_consistency_score"] == 0.0     # empty past
    assert u1["rare_device_usage_count"] == 2        # both devices new
    # user with no events at all: all features 0
    u3 = t[t["user"] == "U3"]
    for c in g_fcols():
        assert (u3[c] == 0).all()


def test_file_cold_start():
    rows = [("F1", _dt(1), "U1", "PC-0001", "f1", P_OLE2)]
    t = build([], rows)
    u1 = t[(t["user"] == "U1") & (t["day"] == pd.Timestamp("2010-01-01"))].iloc[0]
    assert u1["file_type_consistency_score"] == 0.0
    assert u1["rare_file_type_access_count"] == 1


# ---------------------------------------------------------------------------
# Duplicate interactions
# ---------------------------------------------------------------------------
def test_duplicate_rows_set_features_unchanged():
    base = [
        ("L1", _dt(2), "U1", "PC-0001", "Logon"),
        ("L2", _dt(3), "U1", "PC-0001", "Logon"),
    ]
    dup = base + [("L3", _dt(2), "U1", "PC-0001", "Logon"),
                  ("L4", _dt(3), "U1", "PC-0001", "Logon")]
    days = _days(5)
    t1 = build(base, [], days=days)
    t2 = build(dup, [], days=days)
    for d in [pd.Timestamp("2010-01-02"), pd.Timestamp("2010-01-03")]:
        for c in g_fcols():
            a = t1[(t1["user"] == "U1") & (t1["day"] == d)][c].iloc[0]
            b = t2[(t2["user"] == "U1") & (t2["day"] == d)][c].iloc[0]
            assert a == b, f"{c} on {d} changed under duplicates"


# ---------------------------------------------------------------------------
# Determinism + alignment + aggregation
# ---------------------------------------------------------------------------
def test_deterministic_output():
    logon_rows = [("L1", _dt(2), "U1", "PC-0001", "Logon")]
    file_rows = [("F1", _dt(2), "U1", "PC-0001", "f1", P_OLE2)]
    days = _days(4)
    t1 = build(logon_rows, file_rows, days=days)
    t2 = build(logon_rows, file_rows, days=days)
    pd.testing.assert_frame_equal(t1, t2)


def test_grid_exact_and_key_aligned():
    days = _days(5)
    t = build([("L1", _dt(1), "U1", "PC-0001", "Logon")], [], days=days)
    assert len(t) == len(USERS) * len(days)
    assert not t.duplicated(subset=["user", "day"]).any()
    expected = pd.MultiIndex.from_product([sorted(USERS), pd.to_datetime(days).normalize()],
                                          names=["user", "day"])
    assert set(t.set_index(["user", "day"]).index) == set(expected)


def test_file_type_consistency_known_value():
    """Hand-computed Jaccard over content-type sets (filenames unique by design)."""
    rows = [
        ("F1", _dt(2), "U1", "PC-0001", "f1", P_OLE2),
        ("F2", _dt(3), "U1", "PC-0001", "f2", P_OLE2),
        ("F3", _dt(3), "U1", "PC-0001", "f3", P_PDF),
    ]
    t = build([], rows)
    u1 = t[(t["user"] == "U1") & (t["day"] == pd.Timestamp("2010-01-03"))].iloc[0]
    assert u1["file_type_consistency_score"] == pytest.approx(1 / 2)
    assert u1["rare_file_type_access_count"] == 1     # PDF is new to U1


def g_fcols():
    return list(config.GRAPH_FEATURES)