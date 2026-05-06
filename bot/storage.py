"""SQLite storage for sent alerts (used by the cooldown check)."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from .config import DB_PATH

SCHEMA = """
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


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


def record_alert(symbol: str, score: int, reasons: str, price: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO alerts_sent (symbol, score, reasons, price, sent_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, score, reasons, price, datetime.now().isoformat()),
        )
