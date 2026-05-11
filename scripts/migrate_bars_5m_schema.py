"""One-shot migration: bars_5m old schema → spec schema.

OLD                                       NEW
---                                       ---
ts_open  TEXT (ISO-8601 IST tz-aware)  →  ts      INTEGER (UTC epoch ms)
o, h, l, c, v REAL                     →  open, high, low, close, volume REAL

Everything else is preserved (symbol column, primary key shape).

Behaviour
---------
* If ``bars_5m`` is already on the new schema, exit cleanly with a notice.
* Otherwise:
    1. Copy the DB to ``<db>.pre-migration.<timestamp>.bak``.
    2. Build ``bars_5m_new`` with the spec schema.
    3. INSERT ... SELECT, parsing each ``ts_open`` ISO string into epoch ms.
    4. Verify row count parity. Abort (rolling back the new table) on mismatch.
    5. Drop ``bars_5m``, rename ``bars_5m_new`` to ``bars_5m``.

Idempotent: running the script twice is safe — the second run sees the new
schema already in place and exits without touching data.

Usage
-----
    python -m scripts.migrate_bars_5m_schema [path/to/alerts.db]

Default path is ``alerts.db`` in the current working directory.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

NEW_SCHEMA_SQL = """
CREATE TABLE bars_5m_new (
    symbol  TEXT    NOT NULL,
    ts      INTEGER NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  REAL    NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _is_already_migrated(conn: sqlite3.Connection) -> bool:
    cols = _table_columns(conn, "bars_5m")
    return "ts" in cols and "open" in cols and "ts_open" not in cols


def _iso_to_epoch_ms(iso: str) -> int:
    """Convert a stored ts_open ISO-8601 IST string to UTC epoch milliseconds.

    Stored values look like ``'2026-05-04T09:15:00+05:30'``. We parse with
    pandas which handles the offset; ``.value`` gives ns since UTC epoch.
    """
    ts = pd.Timestamp(iso)
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    return int(ts.value // 1_000_000)


def migrate(db_path: Path) -> None:
    if not db_path.exists():
        print(f"[migrate] {db_path} does not exist — nothing to migrate.")
        return

    with sqlite3.connect(db_path) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "bars_5m" not in tables:
            print(f"[migrate] {db_path}: no bars_5m table — nothing to migrate.")
            return
        if _is_already_migrated(conn):
            print(f"[migrate] {db_path}: bars_5m is already on the new schema.")
            return

        cols = _table_columns(conn, "bars_5m")
        expected_old = {"symbol", "ts_open", "o", "h", "l", "c", "v"}
        if not expected_old.issubset(cols):
            raise RuntimeError(
                f"bars_5m has unexpected columns {cols!r}; "
                f"refusing to migrate (expected superset of {sorted(expected_old)})"
            )

    backup = db_path.with_suffix(
        db_path.suffix + f".pre-migration.{datetime.now():%Y%m%d_%H%M%S}.bak"
    )
    shutil.copy2(db_path, backup)
    print(f"[migrate] backup written to {backup}")

    with sqlite3.connect(db_path) as conn:
        old_count = conn.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        print(f"[migrate] migrating {old_count} rows ...")

        conn.execute("DROP TABLE IF EXISTS bars_5m_new")
        conn.executescript(NEW_SCHEMA_SQL)

        # Stream rows in chunks to keep memory bounded for large tables.
        cur = conn.execute("SELECT symbol, ts_open, o, h, l, c, v FROM bars_5m")
        batch: list[tuple] = []
        BATCH_SIZE = 5000
        inserted = 0
        for row in cur:
            symbol, ts_open_iso, o, h, l, c, v = row
            ts_ms = _iso_to_epoch_ms(ts_open_iso)
            batch.append((symbol, ts_ms, o, h, l, c, v))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO bars_5m_new VALUES (?,?,?,?,?,?,?)", batch,
                )
                inserted += len(batch)
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT INTO bars_5m_new VALUES (?,?,?,?,?,?,?)", batch,
            )
            inserted += len(batch)

        new_count = conn.execute("SELECT COUNT(*) FROM bars_5m_new").fetchone()[0]
        if new_count != old_count or inserted != old_count:
            conn.execute("DROP TABLE bars_5m_new")
            raise RuntimeError(
                f"row count mismatch: old={old_count} inserted={inserted} "
                f"new={new_count}; aborted (bars_5m untouched, backup at {backup})"
            )

        conn.execute("DROP TABLE bars_5m")
        conn.execute("ALTER TABLE bars_5m_new RENAME TO bars_5m")
        conn.commit()

    print(f"[migrate] done — {old_count} rows on the new schema.")


def main(argv: list[str]) -> int:
    db_path = Path(argv[1]) if len(argv) > 1 else Path("alerts.db")
    migrate(db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
