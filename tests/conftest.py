"""Shared pytest fixtures.

Also keeps the project root importable when pytest is invoked from any
working directory.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation seed data
# ---------------------------------------------------------------------------

SESSION_DATE = "2026-05-04"
AFTER_CLOSE = pd.Timestamp(f"{SESSION_DATE} 16:00:00", tz="Asia/Kolkata")

# Fixed seed — these arrays must be stable across test runs because tests
# assert exact OHLCV equality against them.
rng = np.random.default_rng(seed=42)

BASE_OPEN = 100.0
opens = BASE_OPEN + np.cumsum(rng.uniform(-1, 1, 75))
highs = opens + rng.uniform(0, 2, 75)
lows = opens - rng.uniform(0, 2, 75)
closes = opens + rng.uniform(-1, 1, 75)
volumes = rng.integers(500, 2000, 75).astype(float)

session_start_ist = pd.Timestamp(f"{SESSION_DATE} 09:15:00", tz="Asia/Kolkata")
timestamps_utc_ms = [
    int((session_start_ist + pd.Timedelta(minutes=5 * i)).value / 1e6)
    for i in range(75)
]


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    """Create an in-memory SQLite DB seeded with the given rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE bars_5m (
            symbol TEXT, ts INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )
    """)
    conn.executemany("INSERT INTO bars_5m VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def _patch_timeframes_db(monkeypatch, conn: sqlite3.Connection) -> None:
    import data.timeframes as tf_mod
    monkeypatch.setattr(tf_mod, "_get_connection", lambda: conn)
    # Cache could carry stale results between tests — clear before & after.
    tf_mod._get_resampled_full.cache_clear()


@pytest.fixture
def full_session_db(monkeypatch):
    """75-bar complete session for symbol 'X'. Patches the module DB connection."""
    rows = [
        ("X", timestamps_utc_ms[i],
         float(opens[i]), float(highs[i]), float(lows[i]),
         float(closes[i]), float(volumes[i]))
        for i in range(75)
    ]
    conn = _make_db(rows)
    _patch_timeframes_db(monkeypatch, conn)
    yield conn
    import data.timeframes as tf_mod
    tf_mod._get_resampled_full.cache_clear()
    conn.close()


@pytest.fixture
def partial_session_db(monkeypatch):
    """12-bar partial session for symbol 'X' (bars 0–11, 09:15–10:10).

    Bar 11 opens at 10:10 IST and closes at 10:15 IST.
    With ``_now = 10:12 IST``, bar 11 is in-progress.
    Bars 0–10 (09:15–10:05) are complete at ``_now = 10:12``.
    """
    rows = [
        ("X", timestamps_utc_ms[i],
         float(opens[i]), float(highs[i]), float(lows[i]),
         float(closes[i]), float(volumes[i]))
        for i in range(12)
    ]
    conn = _make_db(rows)
    _patch_timeframes_db(monkeypatch, conn)
    yield conn
    import data.timeframes as tf_mod
    tf_mod._get_resampled_full.cache_clear()
    conn.close()
