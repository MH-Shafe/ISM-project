"""Validation module tests (pure functions, synthetic data)."""
import pandas as pd

from src.data import validation as val


def make_logon():
    return pd.DataFrame([
        {"id": "{L1-A1}", "date": "01/04/2010 06:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"},
        {"id": "{L1-A1}", "date": "01/04/2010 09:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"},
        {"id": "{L2-BAD}", "date": "not-a-date", "user": "aaa1", "pc": "PC-01", "activity": "Wat"},
        {"id": None, "date": "01/05/2010 10:00:00", "user": "BBB0002", "pc": "PC-0100", "activity": "Logon"},
    ])


def test_validate_schema_detects_mismatch():
    df = make_logon().rename(columns={"activity": "act"})
    res = val.validate_schema(df, [("id", str), ("date", str), ("user", str), ("pc", str), ("activity", str)])
    assert res["ok"] is False
    assert "columns mismatch" in res["issues"][0]


def test_validate_timestamps_detects_bad():
    res = val.validate_timestamps(make_logon())
    assert res["ok"] is False
    assert res["unparseable"] == 1
    assert res["min"] == "2010-01-04 06:00:00"
    assert res["max"] == "2010-01-05 10:00:00"


def test_validate_ids_detects_duplicates_and_format():
    res = val.validate_ids(make_logon())
    assert res["ok"] is False
    assert res["duplicate_ids"] == 1
    assert res["invalid_format_ids"] == 1


def test_validate_missing():
    res = val.validate_missing(make_logon())
    assert res["missing"]["id"] == 1
    assert res["missing"]["date"] == 0


def test_validate_duplicates():
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 3, "b": 4}])
    res = val.validate_duplicates(df)
    assert res["duplicate_rows"] == 1
    assert res["ok"] is False


def test_validate_allowed_values():
    df = make_logon()
    res = val.validate_allowed_values(df, "activity", {"Logon", "Logoff"})
    assert res["invalid_count"] == 1  # only "Wat"; the None in make_logon is the id, not activity
    assert res["ok"] is False


def test_validate_user_and_pc_format():
    df = make_logon()
    assert val.validate_user_ids(df)["invalid_user_ids"] == 1
    assert val.validate_pc_ids(df)["invalid_pc_ids"] == 1


def test_validate_user_coverage():
    population = {"AAA0001", "BBB0002", "CCC0003"}
    res = val.validate_user_coverage(population, {"logon": {"AAA0001", "BBB0002"}})
    assert res["logon"]["population_missing_from_file"] == 1
    assert res["logon"]["users_not_in_population"] == 0


def test_validate_labels_rejects_unparseable():
    insiders = pd.DataFrame([
        {"dataset": 4.2, "scenario": 1, "details": "x.csv", "user": "AAA0001",
         "start": "/21/2011 11:43:39", "end": "04/25/2011 17:55:00"},
    ])
    insiders = val.load_insiders.__wrapped__ if hasattr(val.load_insiders, "__wrapped__") else insiders
    df = insiders.copy()
    df["start_dt"] = pd.to_datetime(df["start"], format="mixed", errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end"], format="mixed", errors="coerce")
    res = val.validate_labels(df, population={"AAA0001"})
    assert res["unparseable_start"] == 1
    assert res["ok"] is False