"""Watchlist loaded from the NIFTY 500 CSV at import time.

To change the universe, replace ind_nifty500list.csv (download from
https://archives.nseindia.com/content/indices/ind_nifty500list.csv).
Only EQ-series symbols are loaded; .NS suffix is appended for yfinance.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger("alertbot")

WATCHLIST_CSV = Path("ind_nifty500list.csv")


def _load_watchlist() -> list[str]:
    if not WATCHLIST_CSV.exists():
        log.error(
            "Watchlist CSV not found at %s — bot has nothing to scan",
            WATCHLIST_CSV,
        )
        return []
    symbols: list[str] = []
    with WATCHLIST_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sym = (row.get("Symbol") or "").strip()
            if sym:
                symbols.append(f"{sym}.NS")
    return symbols


WATCHLIST = _load_watchlist()
