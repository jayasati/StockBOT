"""Smoke tests for the deterministic health checks (db, disk, bars_fresh).

Network checks (nse_api_responsive, telegram_reachable) are excluded —
they hit live endpoints; they should be exercised manually via
``python -m health --once``."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from health.checks import bars_fresh, db_writable, disk_space, ticks_fresh

IST = ZoneInfo("Asia/Kolkata")

# DDL — copied from bot/db.py to keep tests independent of import order.
HEALTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS health_log (
    ts          TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    latency_ms  INTEGER,
    detail      TEXT
);
"""
BARS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_5m (
    symbol  TEXT    NOT NULL,
    ts      INTEGER NOT NULL,
    open    REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)
);
"""


def _ts_ms(dt: datetime) -> int:
    """Convert a tz-aware datetime to UTC epoch milliseconds."""
    return int(dt.timestamp() * 1000)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(HEALTH_SCHEMA + BARS_SCHEMA)
    return db


def test_disk_space_returns_positive_free_space():
    r = disk_space(".", min_gb=0.0)
    assert r.ok
    assert "GB free" in r.detail
    assert r.latency_ms >= 0


def test_disk_space_fails_when_threshold_unreachable():
    r = disk_space(".", min_gb=10**9)  # demand 1 EB free
    assert not r.ok
    assert "GB free" in r.detail


def test_db_writable_roundtrip(tmp_db: Path):
    r = db_writable(tmp_db)
    assert r.ok, r.detail
    # Verify the canary row was deleted (not left behind)
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM health_log WHERE check_name = '_canary'"
        ).fetchone()
    assert rows[0] == 0


def test_db_writable_fails_on_missing_file(tmp_path: Path):
    missing = tmp_path / "no-such-db.db"
    r = db_writable(missing)
    # SQLite will create the file but the table won't exist → INSERT fails
    assert not r.ok
    assert "sqlite error" in r.detail.lower()


def test_bars_fresh_no_symbols(tmp_db: Path):
    r = bars_fresh(tmp_db, symbols=[])
    assert not r.ok
    assert "no symbols" in r.detail.lower()


def test_bars_fresh_no_bars(tmp_db: Path):
    r = bars_fresh(tmp_db, symbols=["NSE:RELIANCE-EQ"])
    assert not r.ok
    assert "no bars" in r.detail.lower()


def test_bars_fresh_recent_bar_passes(tmp_db: Path):
    now = datetime.now(IST)
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("NSE:X-EQ", _ts_ms(now), 1.0, 1.0, 1.0, 1.0, 100.0),
        )
    r = bars_fresh(tmp_db, symbols=["NSE:X-EQ"], max_age_s=600)
    assert r.ok, r.detail


def test_bars_fresh_stale_bar_fails(tmp_db: Path):
    stale = datetime.now(IST) - timedelta(minutes=20)
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("NSE:X-EQ", _ts_ms(stale), 1.0, 1.0, 1.0, 1.0, 100.0),
        )
    r = bars_fresh(tmp_db, symbols=["NSE:X-EQ"], max_age_s=360)
    assert not r.ok
    assert "old" in r.detail


def test_bars_fresh_no_flap_when_bar_just_persisted(tmp_db: Path):
    """Regression for the 13:33→13:35→13:38 flap pattern.

    A bar that was just persisted has ``ts`` = its slot open time, which is
    ~5 minutes in the past relative to wall clock (the bar only lands in
    SQLite when the *next* slot's first tick arrives). The freshness check
    must measure age against the bar's close (open + 300s), or this is
    guaranteed to flap every cycle."""
    just_persisted_open = datetime.now(IST) - timedelta(minutes=5, seconds=2)
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("NSE:X-EQ", _ts_ms(just_persisted_open), 1.0, 1.0, 1.0, 1.0, 100.0),
        )
    r = bars_fresh(tmp_db, symbols=["NSE:X-EQ"], max_age_s=360)
    assert r.ok, r.detail


def test_bars_fresh_one_fresh_among_many_passes(tmp_db: Path):
    now = datetime.now(IST)
    stale = now - timedelta(hours=5)
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("STALE.NS", _ts_ms(stale), 1, 1, 1, 1, 0),
        )
        conn.execute(
            "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("FRESH.NS", _ts_ms(now), 1, 1, 1, 1, 0),
        )
    r = bars_fresh(tmp_db, symbols=["STALE.NS", "FRESH.NS"], max_age_s=300)
    assert r.ok, "ANY fresh symbol = ok"
    assert "FRESH.NS" in r.detail, "detail names the freshest symbol"


# ---------------------------------------------------------------------------
# ticks_fresh — independent of the bar boundary; reads the aggregator
# ---------------------------------------------------------------------------

def test_ticks_fresh_no_symbols():
    r = ticks_fresh(symbols=[])
    assert not r.ok
    assert "no symbols" in r.detail.lower()


def test_ticks_fresh_no_ticks_received(monkeypatch):
    """Aggregator has zero entries — the check fails with a clear message."""
    class _EmptyAgg:
        def latest_overall_tick_ts(self, symbols):
            return None, None

    from data import realtime_feed
    monkeypatch.setattr(realtime_feed, "get_aggregator", lambda: _EmptyAgg())
    r = ticks_fresh(symbols=["NSE:X-EQ"])
    assert not r.ok
    assert "no ticks" in r.detail.lower()


def test_ticks_fresh_recent_tick_passes(monkeypatch):
    """A tick 10s ago, budget 60s → OK, and the detail names the symbol."""
    recent = datetime.now(IST) - timedelta(seconds=10)

    class _Agg:
        def latest_overall_tick_ts(self, symbols):
            return "NSE:RELIANCE-EQ", recent

    from data import realtime_feed
    monkeypatch.setattr(realtime_feed, "get_aggregator", lambda: _Agg())
    r = ticks_fresh(symbols=["NSE:RELIANCE-EQ"], max_age_s=60)
    assert r.ok, r.detail
    assert "RELIANCE" in r.detail


def test_ticks_fresh_stale_tick_fails(monkeypatch):
    """A tick 120s ago, budget 60s → FAIL."""
    stale = datetime.now(IST) - timedelta(seconds=120)

    class _Agg:
        def latest_overall_tick_ts(self, symbols):
            return "NSE:RELIANCE-EQ", stale

    from data import realtime_feed
    monkeypatch.setattr(realtime_feed, "get_aggregator", lambda: _Agg())
    r = ticks_fresh(symbols=["NSE:RELIANCE-EQ"], max_age_s=60)
    assert not r.ok
    assert "120s" in r.detail
    assert "RELIANCE" in r.detail
