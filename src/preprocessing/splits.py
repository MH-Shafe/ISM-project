"""Temporal (chronological) split logic.

The default evaluation design is TRAIN -> CALIBRATION -> TEST by date,
never a random split of temporal behavior. Splitting operates on day
values only; rows never move across boundaries.
"""
from __future__ import annotations

import pandas as pd


def temporal_split_days(days, train_end, calibration_end):
    """Partition unique days chronologically.

    days: iterable of day values (date/Timestamp)
    Returns (train_days, calibration_days, test_days) as sorted lists.
    Boundaries are inclusive: train = days <= train_end,
    calibration = train_end < days <= calibration_end, rest = test.
    """
    train_end = pd.Timestamp(train_end).normalize()
    calibration_end = pd.Timestamp(calibration_end).normalize()
    if calibration_end <= train_end:
        raise ValueError("calibration_end must be after train_end")
    d = sorted({pd.Timestamp(x).normalize() for x in days})
    train = [x for x in d if x <= train_end]
    cal = [x for x in d if train_end < x <= calibration_end]
    test = [x for x in d if x > calibration_end]
    if not train or not cal or not test:
        raise ValueError("split produced an empty partition; adjust boundaries")
    return train, cal, test


def split_user_day_table(table: pd.DataFrame, train_end, calibration_end) -> dict[str, pd.DataFrame]:
    """Split a user-day table into train/calibration/test by day."""
    train_days, cal_days, test_days = temporal_split_days(table["day"], train_end, calibration_end)
    return {
        "train": table[table["day"].isin(train_days)].copy(),
        "calibration": table[table["day"].isin(cal_days)].copy(),
        "test": table[table["day"].isin(test_days)].copy(),
    }


def split_boundaries(table: pd.DataFrame, train_end, calibration_end) -> dict:
    """Report counts per partition (for audit purposes)."""
    parts = split_user_day_table(table, train_end, calibration_end)
    out = {}
    for name, part in parts.items():
        out[name] = {
            "rows": len(part),
            "days": [str(x) for x in sorted(part["day"].unique())],
            "users": part["user"].nunique(),
            "malicious_rows": int(part["is_malicious"].sum()) if "is_malicious" in part else None,
        }
    return out