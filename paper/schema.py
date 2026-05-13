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
        CHECK (status IN ('OPEN','TP1','TP2','SL','TRAIL','TIMEOUT','MANUAL')),
    exit_ts TEXT,
    exit_price REAL,
    pnl_gross REAL,
    pnl_net REAL,
    notes TEXT,
    -- Phase-7b: 50% partial at TP1 + trailing stop on the runner.
    -- ``tp1_filled=1`` flips a trade into the runner phase: half the
    -- size is already booked at ``tp1_exit_price``; the remaining
    -- ``runner_qty`` floats until ``trailing_stop`` is hit (status
    -- 'TRAIL') or the session times out. ``running_high`` /
    -- ``running_low`` track the favourable extreme since TP1 so the
    -- trailing stop only moves one way.
    tp1_filled INTEGER NOT NULL DEFAULT 0,
    tp1_exit_ts TEXT,
    tp1_exit_price REAL,
    tp1_qty INTEGER,
    tp1_pnl_gross REAL,
    tp1_pnl_net REAL,
    runner_qty INTEGER,
    trailing_stop REAL,
    running_high REAL,
    running_low REAL
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
    """Idempotent CREATE for ``paper_trades`` + ``signal_indicators``,
    plus the Phase-7b column-and-CHECK migration for databases that
    were created before partial-fill + trailing-stop support.

    Safe to call on every process start and on every test."""
    with connect(db_path) as conn:
        conn.executescript(PAPER_SCHEMA)
        _migrate_paper_trades_for_trailing(conn)


def _migrate_paper_trades_for_trailing(conn: sqlite3.Connection) -> None:
    """Upgrade a pre-Phase-7b ``paper_trades`` table in place.

    Two problems to solve:

      1. SQLite ``ALTER TABLE`` can add columns but cannot modify a
         CHECK constraint. The original table's status CHECK rejects
         the new ``TRAIL`` value, so we rebuild the table when the
         migration sentinel (``tp1_filled`` column) is missing.
      2. The new columns need to coexist with the old data — every
         pre-existing row migrates with ``tp1_filled=0``, NULL on the
         trailing columns. That puts legacy trades on the new code
         path naturally: their next monitor tick evaluates them under
         the partial-fill + trail rules."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(paper_trades)")}
    if "tp1_filled" in cols:
        return  # already migrated

    # Disable FK enforcement during the swap — signal_indicators has
    # an FK on paper_trades(id) that would otherwise refuse the DROP.
    # The new table reuses the same id values via INSERT … SELECT, so
    # the FK pointers remain valid after the rename.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE paper_trades_new (
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
                    CHECK (status IN ('OPEN','TP1','TP2','SL','TRAIL','TIMEOUT','MANUAL')),
                exit_ts TEXT,
                exit_price REAL,
                pnl_gross REAL,
                pnl_net REAL,
                notes TEXT,
                tp1_filled INTEGER NOT NULL DEFAULT 0,
                tp1_exit_ts TEXT,
                tp1_exit_price REAL,
                tp1_qty INTEGER,
                tp1_pnl_gross REAL,
                tp1_pnl_net REAL,
                runner_qty INTEGER,
                trailing_stop REAL,
                running_high REAL,
                running_low REAL
            )
        """)
        conn.execute("""
            INSERT INTO paper_trades_new (
                id, symbol, side, entry_ts, entry_price, qty,
                stop_loss, target_1, target_2, confidence, status,
                exit_ts, exit_price, pnl_gross, pnl_net, notes
            )
            SELECT id, symbol, side, entry_ts, entry_price, qty,
                stop_loss, target_1, target_2, confidence, status,
                exit_ts, exit_price, pnl_gross, pnl_net, notes
            FROM paper_trades
        """)
        conn.execute("DROP TABLE paper_trades")
        conn.execute("ALTER TABLE paper_trades_new RENAME TO paper_trades")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_open_symbol
                ON paper_trades(symbol, status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_ts
                ON paper_trades(entry_ts)
        """)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
