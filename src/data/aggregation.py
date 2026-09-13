"""User-day aggregation (one row = one user x one day).

Pure-pandas path is used for tests and small inputs; the duckdb path is
used for the full CERT r4.2 logs on Kaggle (http.csv is 14.5 GB).

Most features are computed strictly within a single (user, day): no
statistic may use future or other-user data. The one exception is
unusual_access_count, which uses a strictly PAST window (day-W .. day-1)
by design; it never uses same-day or future logons. See
BEHAVIORAL_FEATURES in src/config.py for exact definitions and status.
"""
from __future__ import annotations

import pandas as pd

from src import config

FEATURE_COLUMNS = [
    "login_count",
    "after_hours_login_count",
    "usb_connection_count",
    "file_access_count",
    "sensitive_file_access_count",
    "http_activity_count",
    "unique_device_count",
    "unusual_access_count",
]


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df[col], format=config.DATE_FORMAT, errors="coerce")
    df["day"] = df["ts"].dt.normalize()
    return df


def build_user_day_grid(users, days) -> pd.DataFrame:
    day_index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in days])
    grid = pd.MultiIndex.from_product([sorted(users), day_index], names=["user", "day"])
    return pd.DataFrame(index=grid).reset_index()


def aggregate_logon(df: pd.DataFrame) -> pd.DataFrame:
    """login_count, after_hours_login_count, unique_device_count per user-day."""
    d = parse_dates(df)
    d["is_after_hours"] = (d["ts"].dt.hour < 8) | (d["ts"].dt.hour >= 18)
    d["is_logon"] = d["activity"] == "Logon"
    g = (d[d["is_logon"]]
         .groupby(["user", "day"])
         .agg(login_count=("id", "count"),
              after_hours_login_count=("is_after_hours", "sum"),
              unique_device_count=("pc", "nunique"))
         .reset_index())
    return g


def aggregate_device(df: pd.DataFrame) -> pd.DataFrame:
    """usb_connection_count per user-day (activity == 'Connect')."""
    d = parse_dates(df)
    g = (d[d["activity"] == "Connect"]
         .groupby(["user", "day"])
         .agg(usb_connection_count=("id", "count"))
         .reset_index())
    return g


def aggregate_file(df: pd.DataFrame) -> pd.DataFrame:
    """file_access_count + sensitive_file_access_count per user-day.

    sensitive_file_access_count counts rows whose content starts with one of
    the declared sensitive magic-byte prefixes (see config).
    """
    d = parse_dates(df)
    d["is_sensitive"] = d["content"].str.startswith(tuple(config.SENSITIVE_MAGIC_PREFIXES))
    g = (d.groupby(["user", "day"])
         .agg(file_access_count=("id", "count"),
              sensitive_file_access_count=("is_sensitive", "sum"))
         .reset_index())
    return g


def aggregate_unusual_access(df: pd.DataFrame,
                             window_days: int | None = None) -> pd.DataFrame:
    """unusual_access_count per user-day (strictly past window, W days).

    A Logon event on (user, day, pc) is unusual iff the user has NO Logon on
    that pc in [day-W, day-1]. The current day and all future days are never
    consulted; first-ever use of a pc counts as unusual (cold start).
    """
    if window_days is None:
        window_days = config.UNUSUAL_ACCESS_WINDOW_DAYS
    d = parse_dates(df)
    ev = d[d["activity"] == "Logon"]
    if ev.empty:
        return pd.DataFrame(columns=["user", "day", "unusual_access_count"])
    hist = ev[["user", "pc", "day"]].drop_duplicates()

    m = ev[["user", "pc", "day"]].merge(hist, on=["user", "pc"], suffixes=("", "_h"))
    lo = m["day"] - pd.Timedelta(days=window_days)
    m = m[(m["day_h"] >= lo) & (m["day_h"] < m["day"])]
    seen = m.groupby(["user", "pc", "day"]).size().reset_index(name="prior_days")

    ev = ev.merge(seen, on=["user", "pc", "day"], how="left")
    ev["prior_days"] = ev["prior_days"].fillna(0)
    g = (ev[ev["prior_days"] == 0]
         .groupby(["user", "day"]).size()
         .reset_index(name="unusual_access_count"))
    return g


def aggregate_http(df: pd.DataFrame) -> pd.DataFrame:
    """http_activity_count per user-day."""
    d = parse_dates(df)
    g = (d.groupby(["user", "day"])
         .agg(http_activity_count=("id", "count"))
         .reset_index())
    return g


def assemble_user_day_table(logs: dict[str, pd.DataFrame], users, days) -> pd.DataFrame:
    """Pure-pandas assembly used by tests and small inputs.

    logs: {"logon": df, "device": df, "file": df, "http": df}
    """
    table = build_user_day_grid(users, days)
    for col in FEATURE_COLUMNS:
        table[col] = 0

    logon_agg = aggregate_logon(logs["logon"])
    file_agg = aggregate_file(logs["file"])
    parts = {
        "login_count": logon_agg.drop(columns=["after_hours_login_count", "unique_device_count"]),
        "after_hours_login_count": logon_agg.drop(columns=["login_count", "unique_device_count"]),
        "unique_device_count": logon_agg.drop(columns=["login_count", "after_hours_login_count"]),
        "usb_connection_count": aggregate_device(logs["device"]),
        "file_access_count": file_agg.drop(columns=["sensitive_file_access_count"]),
        "sensitive_file_access_count": file_agg.drop(columns=["file_access_count"]),
        "http_activity_count": aggregate_http(logs["http"]),
        "unusual_access_count": aggregate_unusual_access(logs["logon"]),
    }
    for col, part in parts.items():
        part = part.set_index(["user", "day"])
        table = table.set_index(["user", "day"])
        table[col] = part[col].reindex(table.index).fillna(0).astype(int)
        table = table.reset_index()
    table["day"] = pd.to_datetime(table["day"])
    return table


# ---------------------------------------------------------------------------
# DuckDB (Kaggle production) path for the full-size logs
# ---------------------------------------------------------------------------
def build_user_day_table_duckdb(paths: dict[str, str], users, days, chunk_days: int = 120) -> pd.DataFrame:
    """Stream the full logs through duckdb, then assemble the grid in pandas.

    chunk_days: number of days processed per http scan to cap memory.
    """
    import duckdb

    con = duckdb.connect()
    DATE_FMT = config.DATE_FORMAT.replace("%", "%")
    day_expr = f"CAST(strptime(date, '{DATE_FMT}') AS DATE)"

    def scan(name, cols, extra_sql="", types=None):
        t = f", types={types}" if types else ""
        con.execute(
            f"CREATE OR REPLACE VIEW v_{name} AS "
            f"SELECT {cols} FROM read_csv_auto('{paths[name]}', nullstr=['','null']{t}) {extra_sql}"
        )

    scan("logon", f"user, {day_expr} AS day, activity, pc, id, hour(strptime(date, '{DATE_FMT}')) AS hr")
    scan("device", f"user, {day_expr} AS day, activity, id", types={"date": "VARCHAR"})
    scan("file", f"user, {day_expr} AS day, id, content", types={"date": "VARCHAR"})
    scan("http", f"user, {day_expr} AS day, id", types={"date": "VARCHAR"})

    grid = build_user_day_grid(users, days)
    grid["day"] = pd.to_datetime(grid["day"]).dt.date

    # per-source aggregates
    q_logon = con.execute(
        "SELECT user, day, "
        "count(*) FILTER (WHERE activity = 'Logon') AS login_count, "
        "count(*) FILTER (WHERE activity = 'Logon' AND (hr < 8 OR hr >= 18)) AS after_hours_login_count, "
        "count(DISTINCT pc) AS unique_device_count "
        "FROM v_logon GROUP BY user, day"
    ).fetchdf()
    q_device = con.execute(
        "SELECT user, day, count(*) AS usb_connection_count "
        "FROM v_device WHERE activity = 'Connect' GROUP BY user, day"
    ).fetchdf()
    q_file = con.execute(
        "SELECT user, day, count(*) AS file_access_count, "
        "count(*) FILTER (WHERE content LIKE 'D0-CF-11-E0-A1-B1-1A-E1%' "
        "OR content LIKE '25-50-44-46-2D%' OR content LIKE '50-4B-03-04%') "
        "AS sensitive_file_access_count FROM v_file GROUP BY user, day"
    ).fetchdf()
    q_http = con.execute(
        "SELECT user, day, count(*) AS http_activity_count FROM v_http GROUP BY user, day"
    ).fetchdf()
    q_unusual = con.execute(
        "SELECT l.user, l.day, count(*) AS unusual_access_count "
        "FROM v_logon l "
        "LEFT JOIN (SELECT DISTINCT user, pc, day FROM v_logon WHERE activity = 'Logon') h "
        "ON h.user = l.user AND h.pc = l.pc "
        f"AND h.day >= l.day - INTERVAL '{config.UNUSUAL_ACCESS_WINDOW_DAYS}' DAY "
        "AND h.day < l.day "
        "WHERE l.activity = 'Logon' AND h.day IS NULL "
        "GROUP BY l.user, l.day"
    ).fetchdf()
    con.close()

    table = grid.copy()
    for col in FEATURE_COLUMNS:
        table[col] = 0
    table = table.set_index(["user", "day"])
    for part in [q_logon, q_device, q_file, q_http, q_unusual]:
        if part.empty:
            continue
        part["user"] = part["user"].astype(str)
        part["day"] = pd.to_datetime(part["day"]).dt.date
        feats = [c for c in part.columns if c not in ("user", "day")]
        part = part.set_index(["user", "day"])
        for c in feats:
            table[c] = part[c].reindex(table.index).fillna(0).astype(int)
    table = table.reset_index()
    table["day"] = pd.to_datetime(table["day"])
    return table[["user", "day"] + FEATURE_COLUMNS]