"""SQLite write helpers for the alerts_sent table.

The schema and ``init_db()`` live in :mod:`bot.db`; this module only does
inserts (the cooldown read happens in :mod:`suppression`)."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from .config import DB_PATH


def record_alert(symbol: str, score: int, reasons: str, price: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO alerts_sent (symbol, score, reasons, price, sent_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, score, reasons, price, datetime.now().isoformat()),
        )
