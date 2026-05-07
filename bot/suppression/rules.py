"""Suppression query path: cooldown + ASM-stage gate + GSM + pledge."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("alerts.db")


def _asm_suppresses(stage: str) -> bool:
    """Decide whether an ASM stage label warrants suppressing alerts.

    NSE has two ASM tracks:
      - LT-ASM (long-term): Stage I is informational (price band); Stage II
        adds 100% margin. Suppress only Stage II+.
      - ST-ASM (short-term): even Stage I imposes 100% margin and Stage II
        moves to trade-to-trade. Suppress any stage."""
    s = stage.upper()
    if "ST-ASM" in s:
        return True
    if "LT-ASM" in s:
        return any(
            tag in s
            for tag in ("STAGE II", "STAGE III", "STAGE IV", "STAGE 2", "STAGE 3", "STAGE 4")
        )
    # Unknown prefix — be conservative: any stage 2+ suppresses.
    return any(
        tag in s
        for tag in ("STAGE II", "STAGE III", "STAGE IV", "STAGE 2", "STAGE 3", "STAGE 4")
    )


def is_suppressed(symbol: str, cooldown_minutes: int = 60) -> tuple[bool, str]:
    """Return (suppressed, reason). Reason is empty when not suppressed."""
    cutoff = (datetime.now() - timedelta(minutes=cooldown_minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        # Cooldown — was alerted in the last N minutes
        row = conn.execute(
            "SELECT 1 FROM alerts_sent WHERE symbol = ? AND sent_at > ? LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        if row:
            return True, f"cooldown ({cooldown_minutes}m)"

        # ASM stage 2+
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'asm'",
            (symbol,),
        ).fetchone()
        if row and _asm_suppresses(row[0]):
            return True, f"ASM {row[0]}"

        # GSM — any stage suppresses
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'gsm'",
            (symbol,),
        ).fetchone()
        if row:
            return True, f"GSM {row[0]}"

        # Promoter pledge > 50%
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'pledge_pct'",
            (symbol,),
        ).fetchone()
        if row:
            try:
                if float(row[0]) > 50:
                    return True, f"pledge {row[0]}%"
            except ValueError:
                pass

    return False, ""
