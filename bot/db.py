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
    alerted INTEGER NOT NULL,
    components_json TEXT,
    final_score REAL
);
CREATE INDEX IF NOT EXISTS idx_filter_audit_symbol_ts
    ON filter_audit(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_filter_audit_kill
    ON filter_audit(kill_reasons);
"""

# Phase-8 daily structure-level cache. Precomputed once at 09:00 IST
# by ``data.precompute.run_morning_compute``; read by the scoring
# engine during the day. ``levels_json`` carries the full bundle
# (pivot/CPR/fib/PDH/PDL/PDC) so adding new level types doesn't need
# a schema change.
DAILY_LEVELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_levels (
    symbol       TEXT NOT NULL,
    session_date TEXT NOT NULL,
    levels_json  TEXT NOT NULL,
    computed_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, session_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_levels_date
    ON daily_levels(session_date);
"""

# Phase-9 NSE replay cache. Every successful response from
# ``data.nse._get_json`` lands here as a JSON blob so a missed scan
# can be re-played offline (and so test fixtures can be captured from
# real responses). ``kind`` is the endpoint key plus a param signature
# (e.g. ``option_chain:symbol=NIFTY``).
NSE_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS nse_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nse_snapshots_kind_ts
    ON nse_snapshots(kind, ts DESC);
"""

# Phase-10 news + sentiment cache.
# ``news_items``  — one row per unique headline (id = sha1 of
#                   source+url+title). FinBERT score is signed
#                   [-1.0, +1.0]; label is the dominant class
#                   ('positive'/'negative'/'neutral'/'stub'/'error').
# ``news_scores`` — many-to-many between items and symbols. A single
#                   item ("Reliance and Tata in JV") fans out into
#                   one row per matched symbol with the relevance
#                   from ``news.symbol_match``.
NEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_items (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT,
    url             TEXT,
    published_at    TEXT,
    fetched_at      TEXT NOT NULL,
    finbert_score   REAL,
    finbert_label   TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_items_published
    ON news_items(published_at DESC);

CREATE TABLE IF NOT EXISTS news_scores (
    symbol     TEXT NOT NULL,
    news_id    TEXT NOT NULL REFERENCES news_items(id),
    relevance  REAL NOT NULL,
    PRIMARY KEY (symbol, news_id)
);
CREATE INDEX IF NOT EXISTS idx_news_scores_symbol
    ON news_scores(symbol);
"""

MASTER_SCHEMA = (
    ALERTS_SCHEMA + FILINGS_SCHEMA + RISK_FLAGS_SCHEMA + HEALTH_SCHEMA
    + FILTER_AUDIT_SCHEMA + DAILY_LEVELS_SCHEMA + NSE_SNAPSHOTS_SCHEMA
    + NEWS_SCHEMA
)


def init_db() -> None:
    """Create every persistent table the bot writes to. Idempotent."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(MASTER_SCHEMA)
        _migrate_filter_audit(conn)
    # Phase-5 paper-tracker tables live in a sibling package so the SQL
    # stays next to its writers. Local import avoids a circular path
    # through ``bot/__init__.py``.
    from paper.schema import ensure_paper_schema
    ensure_paper_schema()


def _migrate_filter_audit(conn: sqlite3.Connection) -> None:
    """Phase-7 schema-extension migration. ``filter_audit`` predates
    the components/final_score columns; add them in place when the
    table already exists from a pre-Phase-7 init. CREATE TABLE IF
    NOT EXISTS does not retrofit columns to an existing table, so
    we ALTER on a per-column basis with PRAGMA introspection. Newly
    added columns default to NULL — rows written before Phase-7
    keep their NULL semantics."""
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(filter_audit)")
    }
    if "components_json" not in existing_cols:
        conn.execute("ALTER TABLE filter_audit ADD COLUMN components_json TEXT")
    if "final_score" not in existing_cols:
        conn.execute("ALTER TABLE filter_audit ADD COLUMN final_score REAL")
