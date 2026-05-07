"""ASM/GSM refresh orchestrator: fetch + parse + persist."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from .http import (
    NSE_ASM_API_CANDIDATES,
    NSE_ASM_PAGE,
    NSE_GSM_API_CANDIDATES,
    NSE_GSM_PAGE,
    _fetch_first_working,
    _nse_session,
)
from .parse import _parse_asm, _parse_gsm

log = logging.getLogger("alertbot.suppression")

DB_PATH = Path("alerts.db")


async def refresh_asm_gsm() -> dict[str, int]:
    """Pull current ASM/GSM lists from NSE and replace the rows in risk_flags."""
    from bot.db import init_db
    init_db()
    counts = {"asm": 0, "gsm": 0}
    client = await _nse_session()
    try:
        asm_data = await _fetch_first_working(
            client, NSE_ASM_API_CANDIDATES, NSE_ASM_PAGE
        )
        gsm_data = await _fetch_first_working(
            client, NSE_GSM_API_CANDIDATES, NSE_GSM_PAGE
        )
    finally:
        await client.aclose()

    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        # ASM/GSM are authoritative-state lists — wipe and replace each refresh.
        conn.execute("DELETE FROM risk_flags WHERE flag_type IN ('asm', 'gsm')")

        if asm_data is not None:
            for sym, stage in _parse_asm(asm_data):
                conn.execute(
                    "INSERT OR REPLACE INTO risk_flags VALUES (?, ?, ?, ?)",
                    (f"{sym}.NS", "asm", stage, now),
                )
                counts["asm"] += 1
        else:
            log.warning("ASM fetch returned no data — leaving table empty for this refresh")

        if gsm_data is not None:
            for sym, stage in _parse_gsm(gsm_data):
                conn.execute(
                    "INSERT OR REPLACE INTO risk_flags VALUES (?, ?, ?, ?)",
                    (f"{sym}.NS", "gsm", stage, now),
                )
                counts["gsm"] += 1
        else:
            log.warning("GSM fetch returned no data — leaving table empty for this refresh")

    log.info("ASM/GSM refresh: %d ASM flags, %d GSM flags", counts["asm"], counts["gsm"])
    return counts
