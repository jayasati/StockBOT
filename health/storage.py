"""Read/write helpers for the ``health_log`` table.

Schema lives in :mod:`bot.db` (master init). This module only writes
new rows and queries by check name."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .checks import CheckResult

DB_PATH = Path("alerts.db")


def log_check_result(
    check_name: str, r: CheckResult, db_path: Path = DB_PATH
) -> None:
    """Append a row to ``health_log``. Best-effort — caller decides what to
    do on failure (we re-raise so the monitor's exception handler logs it)."""
    with sqlite3.connect(db_path, timeout=5) as conn:
        conn.execute(
            "INSERT INTO health_log (ts, check_name, ok, latency_ms, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                check_name,
                1 if r.ok else 0,
                r.latency_ms,
                r.detail,
            ),
        )


def last_failure_ts(
    check_name: str, db_path: Path = DB_PATH
) -> datetime | None:
    """Timestamp of the most recent failure for ``check_name``."""
    with sqlite3.connect(db_path, timeout=5) as conn:
        row = conn.execute(
            "SELECT ts FROM health_log WHERE check_name = ? AND ok = 0 "
            "ORDER BY ts DESC LIMIT 1",
            (check_name,),
        ).fetchone()
    if not row:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except ValueError:
        return None


def latest_per_check(
    db_path: Path = DB_PATH,
) -> dict[str, tuple[bool, str, datetime]]:
    """Return ``{check_name: (ok, detail, ts)}`` for the latest row of
    each check. Excludes the ``_canary`` rows used by ``db_writable``."""
    with sqlite3.connect(db_path, timeout=5) as conn:
        rows = conn.execute(
            """
            SELECT h1.check_name, h1.ok, h1.detail, h1.ts
            FROM health_log h1
            WHERE h1.check_name != '_canary'
              AND h1.ts = (
                  SELECT MAX(ts) FROM health_log h2
                  WHERE h2.check_name = h1.check_name
              )
            """
        ).fetchall()
    out: dict[str, tuple[bool, str, datetime]] = {}
    for name, ok, detail, ts in rows:
        try:
            ts_dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        out[name] = (bool(ok), detail or "", ts_dt)
    return out
