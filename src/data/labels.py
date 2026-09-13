"""Answer-key -> user-day label mapping (CERT r4.2).

A user-day is malicious if the user appears in the r4.2 subset of
insiders.csv AND the day falls inside the incident window [start, end].

The insiders.csv file contains incidents from many CERT releases; only
dataset == 4.2 rows describe incidents inside this dataset's timeline.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data.validation import load_insiders


def r42_incidents(insiders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the dataset==4.2 incidents with parsed windows."""
    df = load_insiders(config.INSIDERS_PATH) if insiders is None else insiders
    return df[df["dataset"] == 4.2].copy()


def build_user_day_labels(users: list[str], days: pd.DatetimeIndex | list,
                          insiders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a user x day label frame.

    users: population user ids
    days:  iterable of day values (datetime.date or Timestamp)
    insiders: raw insiders DataFrame (optional; loaded from config path)

    Returns DataFrame[user, day, is_malicious] with one row per (user, day).
    """
    incidents = r42_incidents(insiders)
    day_index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in days])
    grid = pd.MultiIndex.from_product([sorted(users), day_index], names=["user", "day"])
    out = pd.DataFrame(index=grid).reset_index()
    out["is_malicious"] = 0

    if len(incidents) == 0:
        return out

    for _, row in incidents.iterrows():
        start = pd.Timestamp(row["start_dt"]).normalize()
        end = pd.Timestamp(row["end_dt"]).normalize()
        if pd.isna(start) or pd.isna(end):
            continue
        mask = (out["user"] == row["user"]) & (out["day"] >= start) & (out["day"] <= end)
        out.loc[mask, "is_malicious"] = 1
    return out