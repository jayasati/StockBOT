r"""Dry-run the Phase-7b + Phase-8 schema migrations against a COPY
of the live alerts.db. The original file is never touched.

Run from the project root:
    venv\Scripts\python.exe scripts\verify_migration.py

Reports:
  * Pre-migration tables, paper_trades columns, daily_levels presence
  * Migration log (each ALTER / CREATE / row count)
  * Post-migration shape
  * Row-count parity check (paper_trades + signal_indicators)
  * Sample SELECT against the migrated paper_trades to confirm the
    new columns are queryable and existing data is intact
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LIVE_DB = _PROJECT_ROOT / "alerts.db"
DRYRUN_DB = _PROJECT_ROOT / "alerts.db.dryrun"


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return -1  # table missing


def _check_constraint(conn: sqlite3.Connection, table: str) -> str:
    """Pull the CREATE TABLE SQL so we can show the user the CHECK
    constraint state before/after."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row[0] if row else "<absent>"


def main() -> None:
    if not LIVE_DB.exists():
        print(f"❌ Live DB not found at {LIVE_DB}")
        sys.exit(1)

    # ---------------- Copy ----------------------------------------------
    print(f"=== Phase-7b + Phase-8 migration dry-run ===\n")
    print(f"Source: {LIVE_DB}  ({LIVE_DB.stat().st_size:,} bytes)")
    print(f"Copy:   {DRYRUN_DB}")
    shutil.copy2(LIVE_DB, DRYRUN_DB)

    # ---------------- Pre-migration snapshot ----------------------------
    with sqlite3.connect(DRYRUN_DB) as conn:
        tables_before = _tables(conn)
        pt_cols_before = _columns(conn, "paper_trades")
        pt_rows_before = _row_count(conn, "paper_trades")
        si_rows_before = _row_count(conn, "signal_indicators")
        fa_cols_before = _columns(conn, "filter_audit")
        fa_rows_before = _row_count(conn, "filter_audit")
        dl_rows_before = _row_count(conn, "daily_levels")
        pt_status_count = conn.execute(
            "SELECT status, COUNT(*) FROM paper_trades GROUP BY status"
        ).fetchall()

    print("\n--- BEFORE ---")
    print(f"Tables: {tables_before}")
    print(f"paper_trades cols ({len(pt_cols_before)}): {pt_cols_before}")
    print(f"paper_trades rows: {pt_rows_before}")
    print(f"  by status: {pt_status_count}")
    print(f"signal_indicators rows: {si_rows_before}")
    print(f"filter_audit cols ({len(fa_cols_before)}): {fa_cols_before}")
    print(f"filter_audit rows: {fa_rows_before}")
    print(f"daily_levels rows: {dl_rows_before}  (-1 = table missing)")

    # ---------------- Run the migrations --------------------------------
    print("\n--- RUNNING MIGRATIONS ---")
    # Point every module's DB_PATH at the dry-run copy.
    import bot.config as bot_config
    import bot.db as bot_db
    import bot.storage as bot_storage
    import bot.suppression.rules as suppression_rules
    import paper.schema as paper_schema

    bot_config.DB_PATH = str(DRYRUN_DB)
    bot_db.DB_PATH = str(DRYRUN_DB)
    bot_storage.DB_PATH = str(DRYRUN_DB)
    suppression_rules.DB_PATH = str(DRYRUN_DB)

    try:
        bot_db.init_db()
        print("init_db(): OK")
    except Exception as e:
        print(f"❌ init_db() raised: {type(e).__name__}: {e}")
        raise

    # ---------------- Post-migration snapshot ---------------------------
    with sqlite3.connect(DRYRUN_DB) as conn:
        tables_after = _tables(conn)
        pt_cols_after = _columns(conn, "paper_trades")
        pt_rows_after = _row_count(conn, "paper_trades")
        si_rows_after = _row_count(conn, "signal_indicators")
        fa_cols_after = _columns(conn, "filter_audit")
        fa_rows_after = _row_count(conn, "filter_audit")
        dl_rows_after = _row_count(conn, "daily_levels")
        pt_check_after = _check_constraint(conn, "paper_trades")
        # Status distribution preserved
        pt_status_after = conn.execute(
            "SELECT status, COUNT(*) FROM paper_trades GROUP BY status"
        ).fetchall()
        # Sample row to confirm columns query cleanly
        sample = conn.execute(
            "SELECT id, symbol, status, tp1_filled, trailing_stop "
            "FROM paper_trades ORDER BY id DESC LIMIT 3"
        ).fetchall()

    print("\n--- AFTER ---")
    print(f"Tables: {tables_after}")
    print(f"paper_trades cols ({len(pt_cols_after)}): {pt_cols_after}")
    print(f"paper_trades rows: {pt_rows_after}")
    print(f"  by status: {pt_status_after}")
    print(f"signal_indicators rows: {si_rows_after}")
    print(f"filter_audit cols ({len(fa_cols_after)}): {fa_cols_after}")
    print(f"filter_audit rows: {fa_rows_after}")
    print(f"daily_levels rows: {dl_rows_after}")
    print(f"\npaper_trades CHECK constraint includes 'TRAIL': "
          f"{'TRAIL' in pt_check_after}")
    print(f"\nSample new-column read (id, symbol, status, tp1_filled, trailing_stop):")
    for row in sample:
        print(f"  {row}")

    # ---------------- Parity checks -------------------------------------
    print("\n--- PARITY CHECKS ---")
    issues = []
    if pt_rows_before != pt_rows_after:
        issues.append(f"paper_trades row count changed: "
                      f"{pt_rows_before} -> {pt_rows_after}")
    if si_rows_before != si_rows_after:
        issues.append(f"signal_indicators row count changed: "
                      f"{si_rows_before} -> {si_rows_after}")
    if fa_rows_before != fa_rows_after:
        issues.append(f"filter_audit row count changed: "
                      f"{fa_rows_before} -> {fa_rows_after}")
    # Phase-7b columns: we check the END STATE, not the delta — the
    # migration is idempotent, so if init_db ran previously these
    # columns are already present and "no new columns" is fine.
    expected_pt_cols = {
        "tp1_filled", "tp1_exit_ts", "tp1_exit_price", "tp1_qty",
        "tp1_pnl_gross", "tp1_pnl_net", "runner_qty",
        "trailing_stop", "running_high", "running_low",
    }
    missing_pt_cols = expected_pt_cols - set(pt_cols_after)
    if missing_pt_cols:
        issues.append(f"paper_trades missing expected columns: "
                      f"{missing_pt_cols}")
    if "daily_levels" not in tables_after:
        issues.append("daily_levels table not created")
    if "TRAIL" not in pt_check_after:
        issues.append("paper_trades CHECK constraint missing 'TRAIL'")

    # Did THIS dry-run change anything, or was the DB already migrated?
    already_migrated_pt = expected_pt_cols.issubset(set(pt_cols_before))
    already_had_daily_levels = "daily_levels" in tables_before
    new_pt_added = sorted(set(pt_cols_after) - set(pt_cols_before))

    if issues:
        print("[FAIL] Issues:")
        for x in issues:
            print(f"  - {x}")
        print(f"\nDry-run DB left at: {DRYRUN_DB}")
        sys.exit(2)
    else:
        print("[OK] All checks passed.")
        if new_pt_added:
            print(f"   - paper_trades: added {len(new_pt_added)} columns "
                  f"{new_pt_added}; {pt_rows_before} rows preserved")
        elif already_migrated_pt:
            print(f"   - paper_trades: already on Phase-7b schema; "
                  f"{pt_rows_before} rows preserved (idempotent)")
        if already_had_daily_levels:
            print(f"   - daily_levels: already present; "
                  f"{dl_rows_before} rows preserved")
        else:
            print(f"   - daily_levels: CREATED "
                  f"(empty, ready for 09:00 precompute)")
        print(f"   - signal_indicators: {si_rows_before} rows preserved")
        print(f"   - filter_audit: {fa_rows_before} rows preserved")
        print(f"   - status CHECK includes 'TRAIL'")
        print(f"\nDry-run DB left at: {DRYRUN_DB}")
        print(f"Safe to restart the bot - init_db() will apply the "
              f"same migration to the live alerts.db on startup.")


if __name__ == "__main__":
    main()
