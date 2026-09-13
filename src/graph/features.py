"""Leakage-safe graph-derived features (Phase 5).

Graph model: G_d = (V, E_d) where nodes are users, PCs, files, file content
types and departments, and edges carry event timestamps. For a prediction
on day d ONLY two edge sets are ever consulted:

  - day edges E_d   (events on day d itself; observable at end of day)
  - past edges E_<d (events strictly before d; the historical snapshot)

No edge from day d+1 or later ever enters a feature for day d.

Key simplification: "entity x has been seen by user u strictly before day
d" is equivalent to "first-use day of (u, x) < d". Every historical set
therefore reduces to a first-use-day table, computed once, which makes all
five features cheap and strictly past-only.

File names in CERT r4.2 are globally unique (verified: 445,581 distinct
filenames in 445,581 rows), so file features are defined over CONTENT TYPES
(magic-byte prefixes declared in config, plus config.FILE_TYPE_OTHER) rather
than filenames; filename-based features are rejected (see config).

Departments come from LDAP monthly snapshots; the department of user u on
day d is taken from the LATEST snapshot strictly before month(d), so no
same-month LDAP information is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.aggregation import build_user_day_grid, parse_dates


def content_type(content: pd.Series) -> pd.Series:
    """Map file content to a type label via the declared magic-byte prefixes.

    Type labels are the prefix strings themselves (OLE2/PDF/ZIP declarations
    live in config.SENSITIVE_MAGIC_PREFIXES); anything else is 'other'.
    """
    s = content.astype(str)
    out = pd.Series(config.FILE_TYPE_OTHER, index=s.index, dtype=object)
    for prefix in config.SENSITIVE_MAGIC_PREFIXES:
        out[s.str.startswith(prefix)] = prefix
    return out


def first_use_days(events: pd.DataFrame, entity_col: str,
                   user_col: str = "user", day_col: str = "day") -> pd.DataFrame:
    """(user, entity) -> earliest day of appearance (duplicates collapsed).

    Returns DataFrame[user_col, entity_col, first_day] with one row per
    (user, entity). This is the historical-snapshot representation: entity
    is in user's past set on day d iff first_day < d.
    """
    ev = events[[user_col, entity_col, day_col]].drop_duplicates()
    g = ev.groupby([user_col, entity_col], as_index=False)[day_col].min()
    return g.rename(columns={day_col: "first_day"})


def past_set_size(first_use: pd.DataFrame, key_col: str, users, days) -> pd.DataFrame:
    """For every (key, day): count of entities with first_day < day.

    first_use: DataFrame[key_col, first_day] (one row per entity).
    Returns DataFrame[key_col, day, n_past] over the full user x day grid.
    """
    day_unit = 86_400_000_000_000  # ns per day
    grid = build_user_day_grid(users, days)
    day_num = grid["day"].astype("datetime64[ns]").astype("int64") // day_unit
    fu = first_use.copy()
    fu["first_num"] = fu["first_day"].astype("datetime64[ns]").astype("int64") // day_unit
    out = []
    for key, sub in grid.groupby(key_col):
        f = np.sort(fu.loc[fu[key_col] == key, "first_num"].to_numpy(dtype="int64"))
        counts = np.searchsorted(f, sub["day"].astype("datetime64[ns]").astype("int64") // day_unit,
                                 side="left")
        sub = sub.copy()
        sub["n_past"] = counts
        out.append(sub[[key_col, "day", "n_past"]])
    return pd.concat(out, ignore_index=True)


def department_by_day(ldap_frames: list[tuple[str, pd.DataFrame]],
                      users, days) -> pd.DataFrame:
    """Department of each user per day, from the latest snapshot STRICTLY
    before month(d).

    ldap_frames: [(month_str 'YYYY-MM', DataFrame[user_id, department]), ...]
    Returns DataFrame[user, day, department] over the full grid (department
    may be NaN for users absent from LDAP).
    """
    def month_num(m_str: str) -> int:
        y, m = m_str.split("-")
        return int(y) * 12 + int(m) - 1

    snapshots = {}
    for m_str, df in ldap_frames:
        snapshots[month_num(m_str)] = dict(zip(df["user_id"].astype(str),
                                               df["department"].astype(str)))
    out = []
    for d in pd.to_datetime(list(days)).normalize():
        mapping = snapshots.get(month_num(f"{d.year:04d}-{d.month:02d}") - 1, {})
        rows = pd.DataFrame({"user": sorted(users)})
        rows["department"] = rows["user"].map(mapping)
        rows["day"] = d
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def _jaccard(n_day: np.ndarray, n_past: np.ndarray, n_inter: np.ndarray) -> np.ndarray:
    """|Aâˆ©B| / |AâˆªB| with the documented empty-set rule (0.0)."""
    n_day = np.asarray(n_day, dtype=float)
    n_past = np.asarray(n_past, dtype=float)
    n_inter = np.asarray(n_inter, dtype=float)
    denom = n_day + n_past - n_inter
    return np.where((n_day > 0) & (n_past > 0), n_inter / np.maximum(denom, 1e-12), 0.0)


def build_graph_features(logon: pd.DataFrame, file: pd.DataFrame,
                         ldap_frames: list[tuple[str, pd.DataFrame]],
                         users, days) -> pd.DataFrame:
    """Assemble the Phase 5 graph feature table over the user x day grid.

    Returns DataFrame[user, day, device_consistency_score,
    rare_device_usage_count, file_type_consistency_score,
    rare_file_type_access_count, department_file_type_mismatch_count]
    with exactly len(users) x len(days) rows (zero-filled where no events).
    """
    grid = build_user_day_grid(users, days)
    grid["day"] = pd.to_datetime(grid["day"]).dt.normalize()
    for col in config.GRAPH_FEATURES:
        grid[col] = 0

    # ---------------- device features (logon.csv, activity == 'Logon') -------
    lon = parse_dates(logon)
    lon = lon[lon["activity"] == "Logon"].copy()
    lon = lon[["user", "pc", "day"]].drop_duplicates()
    fu_dev = first_use_days(lon, "pc")
    past_dev = past_set_size(fu_dev, "user", users, days)
    past_dev["day"] = pd.to_datetime(past_dev["day"]).dt.normalize()

    day_dev = (lon.groupby(["user", "day"])["pc"].nunique().reset_index()
               .rename(columns={"pc": "n_day"}))
    inter = lon.merge(fu_dev, on=["user", "pc"], how="left")
    inter["is_past"] = inter["first_day"] < inter["day"]
    n_inter = (inter.groupby(["user", "day"])["is_past"].sum().reset_index()
               .rename(columns={"is_past": "n_inter"}))

    dev = day_dev.merge(n_inter, on=["user", "day"], how="left")
    dev = dev.merge(past_dev, on=["user", "day"], how="left")
    dev["n_inter"] = dev["n_inter"].fillna(0.0)
    dev["n_past"] = dev["n_past"].fillna(0.0)

    # ---------------- file features (file.csv, content types) ---------------
    fil = parse_dates(file)
    fil["type"] = content_type(fil["content"])
    fu_type = first_use_days(fil, "type")
    past_type = past_set_size(fu_type, "user", users, days)
    past_type["day"] = pd.to_datetime(past_type["day"]).dt.normalize()

    day_type = (fil.groupby(["user", "day"])["type"].nunique().reset_index()
                .rename(columns={"type": "n_day_types"}))
    t_inter = fil[["user", "day", "type"]].drop_duplicates().merge(
        fu_type, on=["user", "type"], how="left")
    t_inter["is_past"] = t_inter["first_day"] < t_inter["day"]
    n_t_inter = (t_inter.groupby(["user", "day"])["is_past"].sum().reset_index()
                 .rename(columns={"is_past": "n_inter"}))

    fil_t = day_type.merge(n_t_inter, on=["user", "day"], how="left")
    fil_t = fil_t.merge(past_type, on=["user", "day"], how="left")
    fil_t["n_inter"] = fil_t["n_inter"].fillna(0.0)
    fil_t["n_past"] = fil_t["n_past"].fillna(0.0)

    rare_types = fil.merge(fu_type, on=["user", "type"], how="left")
    rare_types["is_rare"] = rare_types["first_day"].isna() | (rare_types["first_day"] >= rare_types["day"])
    rare_count = (rare_types.groupby(["user", "day"])["is_rare"].sum().reset_index()
                  .rename(columns={"is_rare": "rare_file_type_access_count"}))

    # ---------------- department feature (LDAP monthly snapshots) ------------
    dept_grid = department_by_day(ldap_frames, users, days)
    dept_grid["day"] = pd.to_datetime(dept_grid["day"]).dt.normalize()
    dept_rows = fil.merge(dept_grid, on=["user", "day"], how="left")
    dept_rows = dept_rows.dropna(subset=["department"]).copy()
    dept_fu = first_use_days(dept_rows.rename(columns={"department": "dept"}), "type",
                             user_col="dept")

    mismatch = dept_rows[["user", "day", "department", "type"]].merge(
        dept_fu.rename(columns={"dept": "department"}),
        on=["department", "type"], how="left")
    mismatch["is_mismatch"] = (mismatch["first_day"].isna()
                               | (mismatch["first_day"] >= mismatch["day"]))
    mismatch_count = (mismatch.groupby(["user", "day"])["is_mismatch"].sum().reset_index()
                      .rename(columns={"is_mismatch": "department_file_type_mismatch_count"}))

    # ---------------- assemble onto the grid --------------------------------
    def fill(grid, col, part, val_col):
        part = part.set_index(["user", "day"])[val_col]
        g = grid.set_index(["user", "day"])
        g[col] = part.reindex(g.index).fillna(0)
        return g.reset_index()

    dev["device_consistency_score"] = _jaccard(dev["n_day"], dev["n_past"], dev["n_inter"])
    dev["rare_device_usage_count"] = dev["n_day"] - dev["n_inter"]
    grid = fill(grid, "device_consistency_score", dev, "device_consistency_score")
    grid = fill(grid, "rare_device_usage_count", dev, "rare_device_usage_count")

    fil_t["file_type_consistency_score"] = _jaccard(
        fil_t["n_day_types"], fil_t["n_past"], fil_t["n_inter"])
    grid = fill(grid, "file_type_consistency_score", fil_t, "file_type_consistency_score")
    grid = fill(grid, "rare_file_type_access_count", rare_count, "rare_file_type_access_count")
    grid = fill(grid, "department_file_type_mismatch_count", mismatch_count,
                "department_file_type_mismatch_count")

    int_cols = ["rare_device_usage_count", "rare_file_type_access_count",
                "department_file_type_mismatch_count"]
    for c in int_cols:
        grid[c] = grid[c].astype(int)
    return grid[["user", "day"] + list(config.GRAPH_FEATURES)]