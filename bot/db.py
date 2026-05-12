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

# Phase-6 filter audit. Every signal that reaches the filter chain
# leaves a row here, killed or not, with the chain's decisions
# captured as searchable text. ``kill_reasons`` is a comma-joined
# string for cheap "grep killed by adx" queries; soft adjustments
# go as JSON because they're tuples.
FILTER_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS filter_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT,
    score INTEGER NOT NULL,
    kill_reasons TEXT,
    soft_adjustments_json TEXT,
    final_confidence REAL NOT NULL,
    alerted INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filter_audit_symbol_ts
    ON filter_audit(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_filter_audit_kill
    ON filter_audit(kill_reasons);
"""

MASTER_SCHEMA = (
    ALERTS_SCHEMA + FILINGS_SCHEMA + RISK_FLAGS_SCHEMA + HEALTH_SCHEMA
    + FILTER_AUDIT_SCHEMA
)


def init_db() -> None:
    """Create every persistent table the bot writes to. Idempotent."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(MASTER_SCHEMA)
    # Phase-5 paper-tracker tables live in a sibling package so the SQL
    # stays next to its writers. Local import avoids a circular path
    # through ``bot/__init__.py``.
    from paper.schema import ensure_paper_schema
    ensure_paper_schema()
