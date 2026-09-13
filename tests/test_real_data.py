"""Integration tests against the real mounted dataset.

Skipped automatically when the CERT data is not mounted (e.g. local PC),
run on Kaggle where /kaggle/input exists.
"""
import os

import pytest

from src import config

REAL_DATA = os.path.isdir(config.CERT_DATA_ROOT)

pytestmark = pytest.mark.skipif(not REAL_DATA, reason="CERT dataset not mounted")


def test_dataset_discovery():
    """Dataset discovery: expected top-level structure must exist."""
    assert os.path.isdir(config.R4_2_DIR)
    assert os.path.isdir(config.ANSWERS_DIR)
    for name in config.LOG_FILES:
        assert os.path.isfile(config.LOG_PATHS[name]), config.LOG_PATHS[name]
    assert os.path.isfile(config.PSYCHOMETRIC_PATH)
    assert os.path.isfile(config.INSIDERS_PATH)


def test_real_schemas_on_first_rows():
    """Schema check on a sample of the real files."""
    import pandas as pd

    for name in ["logon", "device", "file", "http"]:
        df = pd.read_csv(config.LOG_PATHS[name], nrows=5)
        expected = [c for c, _ in config.SCHEMAS[name]]
        assert list(df.columns) == expected, name
        assert df["date"].str.match(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}").all()


def test_real_label_users_present_in_population():
    """All r4.2 malicious users must exist in the LDAP population."""
    import duckdb

    from src.data.validation import load_insiders

    insiders = load_insiders()
    r42 = insiders[insiders["dataset"] == 4.2]
    pop = set(duckdb.connect().execute(
        f"SELECT DISTINCT user_id FROM read_csv_auto('{config.LDAP_GLOB}')"
    ).fetchdf()["user_id"])
    assert set(r42["user"]) <= pop


def test_real_user_day_aggregation_small_window():
    """Aggregate 3 real days for the full population and sanity-check output."""
    import duckdb

    from src.data.aggregation import build_user_day_table_duckdb

    pop = sorted(set(duckdb.connect().execute(
        f"SELECT DISTINCT user_id FROM read_csv_auto('{config.LDAP_GLOB}')"
    ).fetchdf()["user_id"]))
    days = __import__("pandas").date_range("2010-01-02", "2010-01-04", freq="D")
    table = build_user_day_table_duckdb(config.LOG_PATHS, pop, days)
    assert len(table) == len(pop) * 3
    assert (table[["login_count", "http_activity_count"]] >= 0).all().all()
    # spot check: someone must have logged on in those 3 days
    assert table["login_count"].sum() > 0
    assert table["http_activity_count"].sum() > 0