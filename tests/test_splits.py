"""Temporal split logic tests: chronological, exhaustive, no shuffle."""
import pandas as pd
import pytest

from src.preprocessing.splits import split_user_day_table, temporal_split_days


def test_temporal_split_days_order_and_partition():
    days = pd.date_range("2010-01-01", "2010-01-31", freq="D")
    train, cal, test = temporal_split_days(days, "2010-01-10", "2010-01-20")
    assert train[0] == pd.Timestamp("2010-01-01")
    assert train[-1] == pd.Timestamp("2010-01-10")
    assert cal[0] == pd.Timestamp("2010-01-11")
    assert cal[-1] == pd.Timestamp("2010-01-20")
    assert test[0] == pd.Timestamp("2010-01-21")
    assert test[-1] == pd.Timestamp("2010-01-31")
    # exhaustive, disjoint
    all_days = set(train) | set(cal) | set(test)
    assert len(all_days) == 31
    assert len(train) + len(cal) + len(test) == 31


def test_temporal_split_unsorted_input():
    days = pd.date_range("2010-01-05", "2010-01-01", freq="-1D")  # reversed
    train, cal, test = temporal_split_days(days, "2010-01-02", "2010-01-04")
    assert sorted(train) == train
    assert sorted(cal) == cal
    assert sorted(test) == test


def test_invalid_boundaries_raise():
    days = pd.date_range("2010-01-01", "2010-01-31", freq="D")
    with pytest.raises(ValueError):
        temporal_split_days(days, "2010-01-20", "2010-01-10")


def test_empty_partition_raises():
    days = pd.date_range("2010-01-01", "2010-01-31", freq="D")
    with pytest.raises(ValueError):
        temporal_split_days(days, "2010-02-01", "2010-02-10")


def test_split_user_day_table():
    grid = pd.DataFrame({
        "user": ["A"] * 6 + ["B"] * 6,
        "day": list(pd.date_range("2010-01-01", "2010-01-06", freq="D")) * 2,
    })
    parts = split_user_day_table(grid, "2010-01-02", "2010-01-04")
    assert len(parts["train"]) == 4   # days 01, 02 -> 2 days x 2 users
    assert len(parts["calibration"]) == 4  # days 03, 04
    assert len(parts["test"]) == 4    # days 05, 06
    # no row may straddle: days are the partition key
    assert set(parts["train"]["day"]) <= set(pd.date_range("2010-01-01", "2010-01-02"))
    assert set(parts["calibration"]["day"]) <= set(pd.date_range("2010-01-03", "2010-01-04"))