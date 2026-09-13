"""Data validation for CERT r4.2 logs.

All functions are pure (DataFrame -> dict) so they can be unit-tested on
synthetic data and executed against the real dataset on Kaggle.

Validation is read-only: nothing is removed or transformed silently here.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Low-level pure checks
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, expected: list[tuple[str, type]]) -> dict:
    """Check that df has exactly the expected columns in order."""
    actual = list(df.columns)
    expected_names = [c for c, _ in expected]
    return {
        "expected": expected_names,
        "actual": actual,
        "ok": actual == expected_names,
        "issues": [] if actual == expected_names else [f"columns mismatch: expected {expected_names}, got {actual}"],
    }


def validate_timestamps(df: pd.DataFrame, col: str = "date") -> dict:
    """Parse timestamp column; report unparseable count, range, distinct days."""
    if col not in df.columns:
        return {"ok": False, "issues": [f"column {col} missing"]}
    parsed = pd.to_datetime(df[col], format=config.DATE_FORMAT, errors="coerce")
    n_bad = int(parsed.isna().sum())
    good = parsed.dropna()
    return {
        "ok": n_bad == 0,
        "unparseable": n_bad,
        "min": str(good.min()) if len(good) else None,
        "max": str(good.max()) if len(good) else None,
        "n_distinct_days": int(good.dt.date.nunique()) if len(good) else 0,
        "issues": [] if n_bad == 0 else [f"{n_bad} unparseable timestamps"],
    }


def validate_ids(df: pd.DataFrame, col: str = "id") -> dict:
    """IDs must be unique within a file and match CERT format {XXXX-...}."""
    if col not in df.columns:
        return {"ok": False, "issues": [f"column {col} missing"]}
    dup = int(df[col].duplicated().sum())
    bad_fmt = int((~df[col].astype(str).str.match(config.ID_RE)).sum())
    return {
        "ok": dup == 0 and bad_fmt == 0,
        "duplicate_ids": dup,
        "invalid_format_ids": bad_fmt,
        "issues": [],
    }


def validate_missing(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    cols = cols or list(df.columns)
    missing = {c: int(df[c].isna().sum()) for c in cols}
    return {"ok": all(v == 0 for v in missing.values()), "missing": missing}


def validate_duplicates(df: pd.DataFrame) -> dict:
    """Full-row duplicate count."""
    n = int(df.duplicated().sum())
    return {"ok": n == 0, "duplicate_rows": n}


def validate_allowed_values(df: pd.DataFrame, col: str, allowed: set[str]) -> dict:
    if col not in df.columns:
        return {"ok": False, "issues": [f"column {col} missing"]}
    n = int((~df[col].isin(allowed)).sum())
    return {"ok": n == 0, "invalid_count": n, "allowed": sorted(allowed),
            "issues": [] if n == 0 else [f"{n} values outside {sorted(allowed)}"]}


def validate_user_ids(df: pd.DataFrame, col: str = "user") -> dict:
    n = int((~df[col].astype(str).str.match(config.USER_RE)).sum())
    return {"ok": n == 0, "invalid_user_ids": n}


def validate_pc_ids(df: pd.DataFrame, col: str = "pc") -> dict:
    n = int((~df[col].astype(str).str.match(config.PC_RE)).sum())
    return {"ok": n == 0, "invalid_pc_ids": n}


def validate_numeric_range(df: pd.DataFrame, col: str, lo: float, hi: float) -> dict:
    if col not in df.columns:
        return {"ok": False, "issues": [f"column {col} missing"]}
    n = int(((df[col] < lo) | (df[col] > hi)).sum())
    return {"ok": n == 0, "out_of_range": n, "range": [lo, hi]}


def validate_user_coverage(population: set[str], per_file: dict[str, set[str]]) -> dict:
    """Every population user must appear in logon/http/email; other files are subsets."""
    out = {}
    for fname, users in per_file.items():
        missing = sorted(population - users)
        out[fname] = {"n_users": len(users), "users_not_in_population": len(users - population),
                      "population_missing_from_file": len(missing)}
    return out


# ---------------------------------------------------------------------------
# Label (answer key) validation
# ---------------------------------------------------------------------------
def load_insiders(path: str | None = None) -> pd.DataFrame:
    path = path or config.INSIDERS_PATH
    df = pd.read_csv(path)
    df["start_dt"] = pd.to_datetime(df["start"], format="mixed", errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end"], format="mixed", errors="coerce")
    return df


def validate_labels(df: pd.DataFrame, population: set[str]) -> dict:
    r42 = df[df["dataset"] == 4.2]
    return {
        "n_rows": len(df),
        "n_dataset_4_2": len(r42),
        "n_dataset_4_2_users": r42["user"].nunique(),
        "unparseable_start": int(df["start_dt"].isna().sum()),
        "unparseable_end": int(df["end_dt"].isna().sum()),
        "malicious_users_not_in_population": sorted(set(r42["user"]) - population),
        "scenarios": sorted(r42["scenario"].unique().tolist()),
        "by_scenario": r42["scenario"].value_counts().sort_index().to_dict(),
        "ok": bool(
            len(set(r42["user"]) - population) == 0
            and df["start_dt"].isna().sum() == 0
            and df["end_dt"].isna().sum() == 0
        ),
    }


# ---------------------------------------------------------------------------
# SQL (duckdb) path — used for the full-size logs (http.csv is 14.5 GB).
# Same report structure as the pandas path, but streams via duckdb so peak
# memory stays low.
# ---------------------------------------------------------------------------
def validate_file_sql(con, name: str, path: str, types: dict | None = None) -> dict:
    import duckdb  # noqa: F401

    extra = f", types={types}" if types else ""
    con.execute(
        f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{path}', nullstr=['','null']{extra})"
    )
    cols = con.execute("DESCRIBE SELECT * FROM v").fetchall()
    col_names = [c[0] for c in cols]
    expected = [c for c, _ in config.SCHEMAS[name]]
    qname = lambda c: f'"{c}"'

    missing_sql = ", ".join(f"count({qname(c)}) AS n_{c}" for c in col_names)
    agg_row = con.execute(f"SELECT count(*) AS n, {missing_sql} FROM v").fetchone()
    r = dict(zip([x[0] for x in con.description], agg_row))
    n = r.pop("n")
    missing = {c: n - r[f"n_{c}"] for c in col_names}

    out = {
        "schema": {"expected": expected, "actual": col_names,
                   "ok": col_names == expected,
                   "issues": [] if col_names == expected else [f"columns mismatch"]},
        "missing": {"ok": all(v == 0 for v in missing.values()), "missing": missing},
    }

    if name in config.LOG_FILES:
        fmt = config.DATE_FORMAT.replace("%", "%")
        ts = con.execute(
            f"SELECT count(*) FILTER (WHERE try_strptime({qname('date')}, '{fmt}') IS NULL) AS bad, "
            f"min(try_strptime({qname('date')}, '{fmt}')) AS mn, "
            f"max(try_strptime({qname('date')}, '{fmt}')) AS mx, "
            f"count(DISTINCT date_trunc('day', try_strptime({qname('date')}, '{fmt}'))) AS ndays "
            f"FROM v"
        ).fetchone()
        out["timestamps"] = {"ok": ts[0] == 0, "unparseable": ts[0], "min": str(ts[1]),
                             "max": str(ts[2]), "n_distinct_days": ts[3]}
        dup_ids = con.execute(f"SELECT count(*) - count(DISTINCT {qname('id')}) FROM v").fetchone()[0]
        bad_ids = con.execute(
            f"SELECT count(*) FILTER (WHERE NOT regexp_matches({qname('id')}, '^\\{{[A-Z0-9-]+\\}}$')) FROM v"
        ).fetchone()[0]
        out["ids"] = {"ok": dup_ids == 0 and bad_ids == 0, "duplicate_ids": dup_ids,
                      "invalid_format_ids": bad_ids}
        bad_users = con.execute(
            f"SELECT count(*) FILTER (WHERE NOT regexp_matches({qname('user')}, '^[A-Z]{{3}}\\d{{4}}$')) FROM v"
        ).fetchone()[0]
        bad_pcs = con.execute(
            f"SELECT count(*) FILTER (WHERE NOT regexp_matches({qname('pc')}, '^PC-\\d{{4}}$')) FROM v"
        ).fetchone()[0]
        out["user_ids"] = {"ok": bad_users == 0, "invalid_user_ids": bad_users}
        out["pc_ids"] = {"ok": bad_pcs == 0, "invalid_pc_ids": bad_pcs}
        if name == "logon":
            bad_act = con.execute(
                f"SELECT count(*) FILTER (WHERE {qname('activity')} NOT IN ('Logon','Logoff')) FROM v"
            ).fetchone()[0]
            out["activities"] = {"ok": bad_act == 0, "invalid_count": bad_act,
                                 "allowed": sorted(config.LOGON_ACTIVITIES)}
        if name == "device":
            bad_act = con.execute(
                f"SELECT count(*) FILTER (WHERE {qname('activity')} NOT IN ('Connect','Disconnect')) FROM v"
            ).fetchone()[0]
            out["activities"] = {"ok": bad_act == 0, "invalid_count": bad_act,
                                 "allowed": sorted(config.DEVICE_ACTIVITIES)}
        if name == "email":
            for c, lo, hi in [("size", 0, 10 ** 12), ("attachments", 0, 10 ** 6)]:
                bad = con.execute(
                    f"SELECT count(*) FILTER (WHERE {qname(c)} < {lo} OR {qname(c)} > {hi}) FROM v"
                ).fetchone()[0]
                out[f"numeric_{c}"] = {"ok": bad == 0, "out_of_range": bad, "range": [lo, hi]}
    else:
        for c in ["O", "C", "E", "A", "N"]:
            bad = con.execute(f"SELECT count(*) FILTER (WHERE {qname(c)} < 0 OR {qname(c)} > 100) FROM v").fetchone()[0]
            out[f"range_{c}"] = {"ok": bad == 0, "out_of_range": bad, "range": [0, 100]}

    dup_rows = con.execute(
        f"SELECT count(*) - count(DISTINCT row({', '.join(qname(c) for c in col_names)})) FROM v"
    ).fetchone()[0]
    out["duplicates"] = {"ok": dup_rows == 0, "duplicate_rows": dup_rows}
    return out


def distinct_users_sql(con, path: str, user_col: str = "user", types: dict | None = None) -> set[str]:
    extra = f", types={types}" if types else ""
    return set(con.execute(
        f"SELECT DISTINCT {user_col} FROM read_csv_auto('{path}', nullstr=['','null']{extra})"
    ).fetchdf()[user_col].astype(str))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_validation(population: set[str],
                   logon: pd.DataFrame, device: pd.DataFrame, file: pd.DataFrame,
                   http: pd.DataFrame, email: pd.DataFrame, psychometric: pd.DataFrame,
                   insiders: pd.DataFrame | None = None) -> dict:
    """Run the full validation battery over already-loaded DataFrames."""
    report: dict = {"files": {}}
    for name, df in [("logon", logon), ("device", device), ("file", file),
                     ("http", http), ("email", email), ("psychometric", psychometric)]:
        f = {"schema": validate_schema(df, config.SCHEMAS[name])}
        if name in config.LOG_FILES:
            f["timestamps"] = validate_timestamps(df)
            f["ids"] = validate_ids(df)
            f["user_ids"] = validate_user_ids(df)
            f["pc_ids"] = validate_pc_ids(df)
            if name == "logon":
                f["activities"] = validate_allowed_values(df, "activity", config.LOGON_ACTIVITIES)
            if name == "device":
                f["activities"] = validate_allowed_values(df, "activity", config.DEVICE_ACTIVITIES)
            if name == "email":
                f["numeric_size"] = validate_numeric_range(df, "size", 0, 1e12)
                f["numeric_attachments"] = validate_numeric_range(df, "attachments", 0, 1e6)
        else:
            for c in ["O", "C", "E", "A", "N"]:
                f[f"range_{c}"] = validate_numeric_range(df, c, 0, 100)
        f["missing"] = validate_missing(df)
        f["duplicates"] = validate_duplicates(df)
        report["files"][name] = f

    per_file = {
        "logon": set(logon["user"]),
        "device": set(device["user"]) if "user" in device.columns else set(),
        "file": set(file["user"]) if "user" in file.columns else set(),
        "http": set(http["user"]),
        "email": set(email["user"]),
        "psychometric": set(psychometric["user_id"]) if "user_id" in psychometric.columns else set(),
    }
    report["user_coverage"] = validate_user_coverage(population, per_file)

    if insiders is not None:
        report["labels"] = validate_labels(insiders, population)

    report["all_ok"] = _all_ok(report)
    return report


def _all_ok(report: dict) -> bool:
    for f in report["files"].values():
        for k, v in f.items():
            if isinstance(v, dict) and v.get("ok") is False:
                return False
    if report.get("labels") and not report["labels"].get("ok", True):
        return False
    return True


def save_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path