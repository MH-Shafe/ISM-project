"""Strict key-alignment verification for user-day feature tables (Phase 6).

Feature tables are joined on the validated analytical key (user, day) --
never by row position. Alignment is checked structurally before any merge,
and the merge itself preserves the base table's row order so that the
frozen behavioral table (and therefore the Phase 4 training input) is
bit-identical after joining graph features.
"""
from __future__ import annotations

import pandas as pd


def verify_alignment(base: pd.DataFrame, extra: pd.DataFrame,
                     keys=("user", "day")) -> dict:
    """Assert identical unique key sets with no duplicates; return a report.

    Raises ValueError on any mismatch: duplicate keys, base-only keys or
    extra-only keys. The report is useful for the artifact record.
    """
    ks = list(keys)
    for name, df in (("base", base), ("extra", extra)):
        if df.duplicated(subset=ks).any():
            raise ValueError(f"duplicate {ks} keys in {name} table")
    b = set(map(tuple, base[ks].itertuples(index=False, name=None)))
    e = set(map(tuple, extra[ks].itertuples(index=False, name=None)))
    report = {
        "keys": ks,
        "base_rows": int(len(base)),
        "extra_rows": int(len(extra)),
        "common_keys": int(len(b & e)),
        "base_only_keys": sorted(b - e),
        "extra_only_keys": sorted(e - b),
    }
    if b != e:
        raise ValueError(
            f"key sets differ: {len(b - e)} base-only, {len(e - b)} extra-only")
    return report


def merge_preserving_order(base: pd.DataFrame, extra: pd.DataFrame,
                           columns: list[str], keys=("user", "day")) -> pd.DataFrame:
    """Left-join `extra` onto `base` by key, preserving base row order.

    Strict one-to-one contract: base row count and key sequence are
    unchanged, and every requested `columns` value must be present after
    the join (missing graph features are a hard error).
    """
    ks = list(keys)
    if base.duplicated(subset=ks).any() or extra.duplicated(subset=ks).any():
        raise ValueError("duplicate keys before merge")
    joined = base.merge(extra, on=ks, how="left", sort=False)
    if len(joined) != len(base):
        raise ValueError("merge changed row count")
    if not (joined[ks].values == base[ks].values).all():
        raise ValueError("merge did not preserve base row order")
    missing = [c for c in columns if joined[c].isna().any()]
    if missing:
        raise ValueError(f"missing values after merge: {missing}")
    return joined