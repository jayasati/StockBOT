"""Suppression query path: cooldown + ASM-stage gate + GSM + pledge."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("alerts.db")


_STAGE_2_PLUS = ("STAGE II", "STAGE III", "STAGE IV",
                 "STAGE 2", "STAGE 3", "STAGE 4")


def _asm_suppresses(stage: str) -> bool:
    """Decide whether an ASM stage label warrants HARD-suppressing alerts.

    Stage I (both ST-ASM and LT-ASM) is downgraded to a soft demote
    handled by ``filters.soft.asm_stage_one`` — ST-ASM Stage I imposes
    100% margin but trading is permitted, so killing the alert outright
    masked legitimate setups (e.g. GODREJIND on 2026-05-14). Stage II+
    moves the symbol toward trade-to-trade and is still suppressed."""
    s = stage.upper()
    return any(tag in s for tag in _STAGE_2_PLUS)


def get_asm_stage(symbol: str) -> str | None:
    """Return the raw ASM stage string for ``symbol`` (e.g. 'ST-ASM Stage
    I'), or None when no flag is set. Used by the soft filter so the
    Stage-I downgrade and the Stage-II+ hard kill read the same row."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value FROM risk_flags "
            "WHERE symbol = ? AND flag_type = 'asm'",
            (symbol,),
        ).fetchone()
    return row[0] if row else None


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

        # Paper-trade lock — if a paper trade for this symbol is still
        # OPEN, suppress further alerts until it closes (SL/TP/TIMEOUT).
        # Without this, the 60-min cooldown expires while a trade may
        # still be open for hours, causing Telegram re-alerts that have
        # no journal counterpart (the dup-guard in ``open_trade``
        # silently blocks the second insert).
        row = conn.execute(
            "SELECT 1 FROM paper_trades "
            "WHERE symbol = ? AND status = 'OPEN' LIMIT 1",
            (symbol,),
        ).fetchone()
        if row:
            return True, "paper trade open"

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
