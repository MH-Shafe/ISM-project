"""User-day aggregation tests (exact values on synthetic logs)."""
import pandas as pd

from src.data.aggregation import (
    aggregate_device,
    aggregate_file,
    aggregate_http,
    aggregate_logon,
    assemble_user_day_table,
)


def test_logon_aggregation_exact(synthetic_logs, users, days):
    g = aggregate_logon(synthetic_logs["logon"])
    row = g.set_index(["user", "day"]).loc[("AAA0001", pd.Timestamp("2010-01-04"))]
    assert row["login_count"] == 2
    assert row["after_hours_login_count"] == 1  # 06:00 only; 09:00 is work hours
    assert row["unique_device_count"] == 1
    row2 = g.set_index(["user", "day"]).loc[("AAA0001", pd.Timestamp("2010-01-05"))]
    assert row2["login_count"] == 1
    assert row2["after_hours_login_count"] == 0
    assert row2["unique_device_count"] == 1  # PC-0002
    row3 = g.set_index(["user", "day"]).loc[("BBB0002", pd.Timestamp("2010-01-04"))]
    assert row3["after_hours_login_count"] == 1  # 22:00


def test_device_aggregation_counts_only_connects(synthetic_logs):
    g = aggregate_device(synthetic_logs["device"])
    row = g.set_index(["user", "day"]).loc[("AAA0001", pd.Timestamp("2010-01-04"))]
    assert row["usb_connection_count"] == 1  # only Connect counted


def test_file_and_http_aggregation(synthetic_logs):
    g = aggregate_file(synthetic_logs["file"])
    idx = g.set_index(["user", "day"])
    row = idx.loc[("AAA0001", pd.Timestamp("2010-01-04"))]
    assert row["file_access_count"] == 2
    assert row["sensitive_file_access_count"] == 2  # OLE2 + PDF magic prefixes
    row2 = idx.loc[("AAA0001", pd.Timestamp("2010-01-05"))]
    assert row2["file_access_count"] == 1
    assert row2["sensitive_file_access_count"] == 0  # JPEG is not sensitive
    g = aggregate_http(synthetic_logs["http"])
    assert g.set_index(["user", "day"]).loc[("BBB0002", pd.Timestamp("2010-01-04"))]["http_activity_count"] == 1


def test_unusual_access_semantics(synthetic_logs):
    """Windowed feature: no same-day/future info; cold start counts."""
    from src.data.aggregation import aggregate_unusual_access

    days = [pd.Timestamp("2010-01-%02d" % d) for d in range(1, 11)]
    rows = []
    for i, day in enumerate(days, start=1):
        rows.append({"id": f"{{{i}}}", "date": day.strftime("%m/%d/%Y") + " 09:00:00",
                     "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"})
    rows.append({"id": "{X}", "date": "01/10/2010 09:05:00", "user": "AAA0001",
                 "pc": "PC-0002", "activity": "Logon"})  # PC-0002: first use
    logon = pd.DataFrame(rows)
    g = aggregate_unusual_access(logon, window_days=7)
    idx = g.set_index(["user", "day"])
    # day 1: cold start -> PC-0001 unusual; days 2..9: PC-0001 seen in window
    # (no unusual events -> no row emitted for those days)
    assert idx.loc[("AAA0001", days[0])]["unusual_access_count"] == 1
    assert idx.loc[("AAA0001", days[9])]["unusual_access_count"] == 1  # PC-0002 (day 10)
    for day in days[1:9]:
        assert ("AAA0001", day) not in idx.index
    # window boundary: PC-0001 used only on day 1; with W=2 the day-4 window
    # is [day 2, day 3] -> day-1 use is outside -> unusual
    boundary = pd.DataFrame([
        {"id": "{b1}", "date": "01/01/2010 09:00:00", "user": "AAA0001",
         "pc": "PC-0001", "activity": "Logon"},
        {"id": "{b2}", "date": "01/04/2010 09:00:00", "user": "AAA0001",
         "pc": "PC-0001", "activity": "Logon"},
    ])
    gb2 = aggregate_unusual_access(boundary, window_days=2).set_index(["user", "day"])
    assert gb2.loc[("AAA0001", days[3])]["unusual_access_count"] == 1
    gb3 = aggregate_unusual_access(boundary, window_days=3).set_index(["user", "day"])
    assert ("AAA0001", days[3]) not in gb3.index  # day 1 inside [day1, day3]


def test_assemble_table_grid_and_zero_fill(synthetic_logs, users, days):
    t = assemble_user_day_table(synthetic_logs, users, days)
    assert len(t) == 4  # 2 users x 2 days
    assert set(t.columns) == {"user", "day", "login_count", "after_hours_login_count",
                              "usb_connection_count", "file_access_count",
                              "sensitive_file_access_count", "http_activity_count",
                              "unique_device_count", "unusual_access_count"}
    # BBB0002 on 01-05 has no events at all -> all zeros
    row = t[(t["user"] == "BBB0002") & (t["day"] == pd.Timestamp("2010-01-05"))].iloc[0]
    assert row[["login_count", "after_hours_login_count", "usb_connection_count",
                "file_access_count", "sensitive_file_access_count",
                "http_activity_count", "unique_device_count",
                "unusual_access_count"]].tolist() == [0] * 8
    # exact assembled values (regression: merge must write INTO the feature columns,
    # not leave the pre-initialized zeros in place)
    row = t[(t["user"] == "AAA0001") & (t["day"] == pd.Timestamp("2010-01-04"))].iloc[0]
    assert row["login_count"] == 2
    assert row["after_hours_login_count"] == 1
    assert row["usb_connection_count"] == 1
    assert row["file_access_count"] == 2
    assert row["sensitive_file_access_count"] == 2
    assert row["http_activity_count"] == 2
    assert row["unique_device_count"] == 1
    assert row["unusual_access_count"] == 2  # both logons on a first-seen pc
    row2 = t[(t["user"] == "AAA0001") & (t["day"] == pd.Timestamp("2010-01-05"))].iloc[0]
    assert row2["unusual_access_count"] == 1  # PC-0002 never used before


def test_day_independence_no_future_leakage(synthetic_logs, users, days):
    """Leakage property: day-local features must depend only on that day's rows.

    Aggregating the full data then filtering to a day must equal aggregating
    the data pre-filtered to that day. This rules out future-derived
    statistics by construction. The windowed feature (unusual_access_count)
    is exempt from this exact equality (it legitimately uses the PAST) and is
    covered by test_windowed_feature_no_future_leakage instead.
    """
    from src.data.aggregation import FEATURE_COLUMNS

    t_full = assemble_user_day_table(synthetic_logs, users, days)
    day = pd.Timestamp("2010-01-04")
    logs_day = {k: v[v["date"].str.startswith("01/04/")] for k, v in synthetic_logs.items()}
    t_single = assemble_user_day_table(logs_day, users, [day])
    a = t_full[t_full["day"] == day].sort_values("user").reset_index(drop=True)
    b = t_single.sort_values("user").reset_index(drop=True)
    cols = [c for c in FEATURE_COLUMNS if c != "unusual_access_count"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_windowed_feature_no_future_leakage():
    """unusual_access_count may use the PAST but never the SAME DAY or FUTURE.

    The scored day's value must be identical whether or not future days are
    present, and it must be influenced by the past (PC used 2 days earlier is
    not unusual).
    """
    from src.data.aggregation import aggregate_unusual_access

    def logon_row(day, pc, hid):
        return {"id": f"{{{hid}}}", "date": f"01/{day:02d}/2010 09:00:00",
                "user": "AAA0001", "pc": pc, "activity": "Logon"}

    past_only = pd.DataFrame([logon_row(2, "PC-0001", "p1"), logon_row(4, "PC-0001", "p2"),
                              logon_row(4, "PC-0002", "p3")])
    with_future = pd.concat([past_only, pd.DataFrame([logon_row(5, "PC-0001", "p4")])],
                            ignore_index=True)
    day4 = pd.Timestamp("2010-01-04")
    g1 = aggregate_unusual_access(past_only, window_days=7).set_index(["user", "day"])
    g2 = aggregate_unusual_access(with_future, window_days=7).set_index(["user", "day"])
    # day 4: PC-0001 seen on day 2 (within window) -> not unusual;
    #        PC-0002 never used before -> unusual. Future day 5 must not matter.
    assert g1.loc[("AAA0001", day4), "unusual_access_count"] == 1
    assert g2.loc[("AAA0001", day4), "unusual_access_count"] == 1