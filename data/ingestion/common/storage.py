"""SQLite schema for the ingestion pipeline.

The schema lives here so every source/extractor/worker imports the
same DDL. `migrate()` is idempotent (CREATE TABLE IF NOT EXISTS) and
safe to call from any process at startup. It does NOT drop or alter
existing tables — additive only.

The legacy `filings_seen` table (created by bot.db) stays as the
short-key dedup index used by the BSE headline path. New code writes
to `filings_v2` in addition; eventual readers (extractor router, ML
feature builder) consume `filings_v2` + `filing_metrics`.

Sub-phase 1.1 only DEFINES the schema. `migrate()` is wired into
the worker entrypoint in 1.2 / 1.10 — not auto-run on import.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("alerts.db")


# ---------------------------------------------------------------------------
# DDL — all CREATE TABLE IF NOT EXISTS, additive-only.
# ---------------------------------------------------------------------------

_FILINGS_V2 = """
CREATE TABLE IF NOT EXISTS filings_v2 (
    filing_id              TEXT PRIMARY KEY,
    source                 TEXT NOT NULL,           -- 'bse' | 'nse'
    symbol                 TEXT,                    -- NSE ticker (e.g. RELIANCE.NS)
    filing_type            TEXT,                    -- routed type, e.g. 'quarterly_results'
    title                  TEXT NOT NULL,
    classification         TEXT,                    -- binary_high / event_unknown / binary_med / fluff
    pdf_url                TEXT,
    pdf_sha1               TEXT,                    -- set after download
    pdf_status             TEXT NOT NULL DEFAULT 'pending',
                                                     -- pending | downloaded | extracted | failed
                                                     -- ocr_required | no_pdf
    raw_text_path          TEXT,                    -- relative path under data/_artifacts/text/
    structured_json        TEXT,                    -- JSON blob of ExtractedFiling.metrics+
    posted_at_ist          TEXT NOT NULL,
    market_session_at_post TEXT,                    -- 'pre' | 'live' | 'post' | 'closed'
    parsed_at              TEXT,
    error                  TEXT
);
"""

_FILING_METRICS = """
CREATE TABLE IF NOT EXISTS filing_metrics (
    filing_id    TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL,
    unit         TEXT,
    PRIMARY KEY (filing_id, metric_name),
    FOREIGN KEY (filing_id) REFERENCES filings_v2(filing_id)
);
"""

_BULK_DEALS = """
CREATE TABLE IF NOT EXISTS bulk_deals (
    trade_date   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    client_name  TEXT NOT NULL,
    buy_sell     TEXT NOT NULL,                    -- 'BUY' | 'SELL'
    qty          INTEGER NOT NULL,
    price        REAL NOT NULL,
    PRIMARY KEY (trade_date, symbol, client_name, buy_sell, qty, price)
);
"""

_BLOCK_DEALS = """
CREATE TABLE IF NOT EXISTS block_deals (
    trade_date   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    client_name  TEXT NOT NULL,
    buy_sell     TEXT NOT NULL,
    qty          INTEGER NOT NULL,
    price        REAL NOT NULL,
    PRIMARY KEY (trade_date, symbol, client_name, buy_sell, qty, price)
);
"""

_CORPORATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol            TEXT NOT NULL,
    action_type       TEXT NOT NULL,               -- 'dividend' | 'split' | 'bonus' | 'rights'
    ex_date           TEXT NOT NULL,
    record_date       TEXT,
    ratio_or_amount   TEXT,                        -- '1:5', '₹12', etc.
    raw_purpose       TEXT,
    PRIMARY KEY (symbol, action_type, ex_date)
);
"""

_MACRO_SIGNALS = """
CREATE TABLE IF NOT EXISTS macro_signals (
    source         TEXT NOT NULL,                  -- 'rbi' | 'sebi'
    signal_type    TEXT NOT NULL,                  -- 'repo' | 'crr' | 'slr' | 'policy_release'
    effective_date TEXT NOT NULL,
    value          REAL,
    raw_text       TEXT,
    PRIMARY KEY (source, signal_type, effective_date)
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_filings_v2_symbol     ON filings_v2(symbol);",
    "CREATE INDEX IF NOT EXISTS ix_filings_v2_status     ON filings_v2(pdf_status);",
    "CREATE INDEX IF NOT EXISTS ix_filings_v2_posted     ON filings_v2(posted_at_ist);",
    "CREATE INDEX IF NOT EXISTS ix_bulk_deals_symbol     ON bulk_deals(symbol);",
    "CREATE INDEX IF NOT EXISTS ix_block_deals_symbol    ON block_deals(symbol);",
    "CREATE INDEX IF NOT EXISTS ix_corp_actions_exdate   ON corporate_actions(ex_date);",
]


def migrate(db_path: Path = DB_PATH) -> None:
    """Idempotent schema setup. Safe to call multiple times / from multiple processes."""
    with sqlite3.connect(db_path) as conn:
        for ddl in (
            _FILINGS_V2,
            _FILING_METRICS,
            _BULK_DEALS,
            _BLOCK_DEALS,
            _CORPORATE_ACTIONS,
            _MACRO_SIGNALS,
        ):
            conn.execute(ddl)
        for idx in _INDEXES:
            conn.execute(idx)
        conn.commit()
