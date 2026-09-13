"""Shared fixtures: synthetic CERT-like logs with known ground truth."""
import pandas as pd
import pytest

from src import config


@pytest.fixture
def synthetic_logs():
    """Small, hand-computed logs so aggregation results are exact.

    Users: AAA0001, BBB0002.
    Days: 2010-01-04 (Mon), 2010-01-05 (Tue).
    """
    logon = pd.DataFrame([
        # user AAA0001 on 2010-01-04: 2 logons (06:00 after-hours, 09:00 work), 1 logoff
        {"id": "{L1}", "date": "01/04/2010 06:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"},
        {"id": "{L2}", "date": "01/04/2010 09:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Logon"},
        {"id": "{L3}", "date": "01/04/2010 17:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Logoff"},
        # user AAA0001 on 2010-01-05: 1 logon work hours, on a second pc
        {"id": "{L4}", "date": "01/05/2010 10:00:00", "user": "AAA0001", "pc": "PC-0002", "activity": "Logon"},
        # user BBB0002 on 2010-01-04: 1 logon after hours, 1 logoff
        {"id": "{L5}", "date": "01/04/2010 22:00:00", "user": "BBB0002", "pc": "PC-0100", "activity": "Logon"},
        {"id": "{L6}", "date": "01/04/2010 23:00:00", "user": "BBB0002", "pc": "PC-0100", "activity": "Logoff"},
    ])
    device = pd.DataFrame([
        {"id": "{D1}", "date": "01/04/2010 08:00:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Connect"},
        {"id": "{D2}", "date": "01/04/2010 08:05:00", "user": "AAA0001", "pc": "PC-0001", "activity": "Disconnect"},
        {"id": "{D3}", "date": "01/05/2010 11:00:00", "user": "AAA0001", "pc": "PC-0002", "activity": "Connect"},
        {"id": "{D4}", "date": "01/04/2010 22:05:00", "user": "BBB0002", "pc": "PC-0100", "activity": "Connect"},
    ])
    file = pd.DataFrame([
        # content starts with magic-byte prefixes: F1 = OLE2 (sensitive),
        # F2 = PDF (sensitive), F3 = JPEG (NOT sensitive)
        {"id": "{F1}", "date": "01/04/2010 08:10:00", "user": "AAA0001", "pc": "PC-0001",
         "filename": "a.doc", "content": "D0-CF-11-E0-A1-B1-1A-E1 xx"},
        {"id": "{F2}", "date": "01/04/2010 08:11:00", "user": "AAA0001", "pc": "PC-0001",
         "filename": "b.pdf", "content": "25-50-44-46-2D xx"},
        {"id": "{F3}", "date": "01/05/2010 11:05:00", "user": "AAA0001", "pc": "PC-0002",
         "filename": "c.jpg", "content": "FF-D8 xx"},
    ])
    http = pd.DataFrame([
        {"id": "{H1}", "date": "01/04/2010 06:05:00", "user": "AAA0001", "pc": "PC-0001",
         "url": "http://a.com", "content": "kw"},
        {"id": "{H2}", "date": "01/04/2010 06:10:00", "user": "AAA0001", "pc": "PC-0001",
         "url": "http://b.com", "content": "kw"},
        {"id": "{H3}", "date": "01/04/2010 09:30:00", "user": "BBB0002", "pc": "PC-0100",
         "url": "http://c.com", "content": "kw"},
    ])
    return {"logon": logon, "device": device, "file": file, "http": http}


@pytest.fixture
def users():
    return ["AAA0001", "BBB0002"]


@pytest.fixture
def days():
    return pd.date_range("2010-01-04", "2010-01-05", freq="D")