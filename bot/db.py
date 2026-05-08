"""Master schema + init for ``alerts.db``.

Single point that owns every CREATE TABLE the bot persists to. Replaces the
three separate ``init_db()`` functions that used to live in ``bot/storage.py``,
``filings.py``, and ``suppression.py``.

The ``bars_5m`` table is owned by ``data.realtime_feed.BarAggregator`` and
initialised on first use; it's not part of this master init."""
from __future__ import annotations

import sqlite3

from .config import DB_PATH

# alerts_sent — sent-alert log used by the cooldown check.
ALERTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts_sent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    score INTEGER NOT NULL,
    reasons TEXT,
    price REAL,
    sent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_sym ON alerts_sent(symbol, sent_at);
"""

# filings_seen — BSE-corporate-announcements dedupe + classification log.
FILINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings_seen (
    filing_id      TEXT PRIMARY KEY,
    symbol         TEXT NOT NULL,
    title          TEXT NOT NULL,
    classification TEXT NOT NULL,
    seen_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filings_sym_class_time
    ON filings_seen(symbol, classification, seen_at);
"""

# risk_flags — ASM/GSM/pledge/PE flags consulted by the suppression layer.
RISK_FLAGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_flags (
    symbol     TEXT NOT NULL,
    flag_type  TEXT NOT NULL CHECK (flag_type IN ('asm', 'gsm', 'pledge_pct', 'high_pe')),
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, flag_type)
);
CREATE INDEX IF NOT EXISTS idx_risk_flags_type ON risk_flags(flag_type);
"""

# health_log — per-tick result of every health check; consumed by the
# /status command and by FailureTracker for consecutive-failure detection.
HEALTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS health_log (
    ts          TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    latency_ms  INTEGER,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_health_check_ts
    ON health_log(check_name, ts DESC);
"""

MASTER_SCHEMA = (
    ALERTS_SCHEMA + FILINGS_SCHEMA + RISK_FLAGS_SCHEMA + HEALTH_SCHEMA
)


def init_db() -> None:
    """Create every persistent table the bot writes to. Idempotent."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(MASTER_SCHEMA)
