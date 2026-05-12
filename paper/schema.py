"""DDL + connection helper for the Phase-5 paper-tracker tables.

Two tables sit in ``alerts.db``:

  paper_trades        — one row per simulated trade. Status lifecycles
                        through OPEN → (TP1|TP2|SL|TIMEOUT|MANUAL).
  signal_indicators   — N rows per trade, captures every indicator
                        value that contributed to the alert score at
                        the moment the trade was opened. Reads from
                        ``paper.journal.win_rate_by_indicator()``.

DDL lives here (not in ``bot/db.py``) so the SQL stays next to the
package that writes it. ``bot.db.init_db()`` calls
``ensure_paper_schema()`` at production startup; tests can call it
directly with a custom ``db_path``.

Everyone in ``paper/*`` uses :func:`connect` rather than raw
``sqlite3.connect`` so foreign-key enforcement is on for every
connection — SQLite ships with ``PRAGMA foreign_keys = OFF`` by
default and the pragma is per-connection, not per-database."""
from __future__ import annotations

import sqlite3


PAPER_TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG','SHORT')),
    entry_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    qty INTEGER NOT NULL,
    stop_loss REAL NOT NULL,
    target_1 REAL NOT NULL,
    target_2 REAL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN','TP1','TP2','SL','TIMEOUT','MANUAL')),
    exit_ts TEXT,
    exit_price REAL,
    pnl_gross REAL,
    pnl_net REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_open_symbol
    ON paper_trades(symbol, status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_ts
    ON paper_trades(entry_ts);
"""

SIGNAL_INDICATORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_indicators (
    paper_trade_id INTEGER NOT NULL REFERENCES paper_trades(id),
    indicator TEXT NOT NULL,
    value REAL NOT NULL,
    timeframe TEXT NOT NULL,
    PRIMARY KEY (paper_trade_id, indicator, timeframe)
);
"""

PAPER_SCHEMA = PAPER_TRADES_SCHEMA + SIGNAL_INDICATORS_SCHEMA


def _default_db_path() -> str:
    """Resolve ``DB_PATH`` lazily to avoid a circular import — ``paper``
    is imported from ``bot.db``, so module-level ``from bot.config
    import DB_PATH`` would re-enter ``bot/__init__.py`` mid-load."""
    from bot.config import DB_PATH
    return DB_PATH


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection to ``alerts.db`` with foreign-key enforcement
    enabled. Use this everywhere in ``paper/*`` so the FK constraint
    on ``signal_indicators.paper_trade_id`` actually fires."""
    path = db_path if db_path is not None else _default_db_path()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_paper_schema(db_path: str | None = None) -> None:
    """Idempotent CREATE for ``paper_trades`` + ``signal_indicators``.

    Safe to call on every process start and on every test. ``CREATE
    TABLE IF NOT EXISTS`` is the whole migration strategy here — no
    PRAGMA user_version bump needed."""
    with connect(db_path) as conn:
        conn.executescript(PAPER_SCHEMA)
