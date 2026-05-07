"""``python -m bot.suppression refresh`` — refresh ASM/GSM and dump tables.

Replaces the old ``python suppression.py`` standalone path."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys

from .refresh import DB_PATH, refresh_asm_gsm


async def _refresh_and_dump() -> None:
    counts = await refresh_asm_gsm()
    print()
    print(f"ASM rows persisted: {counts['asm']}")
    print(f"GSM rows persisted: {counts['gsm']}")
    print()
    with sqlite3.connect(DB_PATH) as conn:
        for flag_type in ("asm", "gsm"):
            rows = conn.execute(
                "SELECT symbol, value FROM risk_flags WHERE flag_type = ? "
                "ORDER BY value, symbol",
                (flag_type,),
            ).fetchall()
            print(f"--- {flag_type.upper()} ({len(rows)} rows) ---")
            for sym, val in rows[:20]:
                print(f"  {sym:18s} {val}")
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more")
            print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if len(sys.argv) < 2:
        print("usage: python -m bot.suppression {refresh}")
        return 1
    cmd = sys.argv[1]
    if cmd == "refresh":
        asyncio.run(_refresh_and_dump())
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
