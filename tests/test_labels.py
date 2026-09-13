"""Answer-key -> user-day label mapping tests."""
import pandas as pd

from src.data.labels import build_user_day_labels


def _insiders():
    return pd.DataFrame([
        {"dataset": 4.2, "scenario": 1, "details": "r4.2-1-x.csv", "user": "AAA0001",
         "start_dt": pd.Timestamp("2010-01-04 00:00:00"), "end_dt": pd.Timestamp("2010-01-05 23:59:59")},
        {"dataset": 3.1, "scenario": 2, "details": "other.csv", "user": "ZZZ0009",
         "start_dt": pd.Timestamp("2010-01-01 00:00:00"), "end_dt": pd.Timestamp("2010-01-01 23:59:59")},
    ])


def test_labels_only_from_dataset_4_2():
    users = ["AAA0001", "BBB0002", "ZZZ0009"]
    days = pd.date_range("2010-01-04", "2010-01-06", freq="D")
    lab = build_user_day_labels(users, days, insiders=_insiders())
    assert len(lab) == 3 * 3
    # ZZZ0009 is dataset 3.1 -> never malicious in our frame
    assert lab.loc[lab["user"] == "ZZZ0009", "is_malicious"].sum() == 0
    # AAA0001 malicious on 04 and 05, not on 06
    a = lab[lab["user"] == "AAA0001"]
    assert list(a.sort_values("day")["is_malicious"]) == [1, 1, 0]


def test_labels_grid_exhaustive():
    users = ["AAA0001"]
    days = pd.date_range("2010-01-04", "2010-01-05", freq="D")
    lab = build_user_day_labels(users, days, insiders=_insiders())
    assert len(lab) == 2
    assert set(lab["user"]) == {"AAA0001"}
    assert set(lab["day"]) == set(days)


def test_labels_unknown_user_safe():
    users = ["NOPE0001"]
    days = pd.date_range("2010-01-04", "2010-01-04", freq="D")
    lab = build_user_day_labels(users, days, insiders=_insiders())
    assert lab["is_malicious"].sum() == 0