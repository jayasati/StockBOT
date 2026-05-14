"""Watchlist loaded from the NIFTY 500 CSV at import time, plus an
optional ``custom_watchlist.csv`` for symbols outside the index that
the user wants tracked (e.g. ALKYLAMINE, mid/small-caps). The custom
list is merged AFTER the Nifty 500 so dedupe preserves the index
ordering and adds extras at the end.

To change the base universe, replace ind_nifty500list.csv (download from
https://archives.nseindia.com/content/indices/ind_nifty500list.csv).

To add stocks beyond the index, drop one symbol per line into
``custom_watchlist.csv`` (header row ``Symbol`` optional). Bare
symbols (``ALKYLAMINE``) and yfinance-style (``ALKYLAMINE.NS``) are
both accepted.

Only EQ-series symbols are supported; .NS suffix is appended for
yfinance/Fyers symbol mapping if missing.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger("alertbot.watchlist")

WATCHLIST_CSV = Path("ind_nifty500list.csv")
CUSTOM_WATCHLIST_CSV = Path("custom_watchlist.csv")


def _normalise(sym: str) -> str:
    sym = sym.strip().upper()
    if not sym:
        return ""
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return f"{sym}.NS"


def _read_csv_symbols(path: Path) -> list[str]:
    """Read symbols from a CSV. Tolerates: header row with ``Symbol``
    column (Nifty CSV format), or one bare symbol per line (custom
    list). Empty/comment lines (``#``) are skipped."""
    symbols: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        # Peek the first non-empty line to decide if there's a header.
        sample = f.read(2048)
        f.seek(0)
        has_header = "Symbol" in sample.split("\n", 1)[0]
        if has_header:
            for row in csv.DictReader(f):
                sym = _normalise(row.get("Symbol") or "")
                if sym:
                    symbols.append(sym)
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Take the first column if it's CSV-shaped.
                sym = _normalise(line.split(",", 1)[0])
                if sym:
                    symbols.append(sym)
    return symbols


def _load_watchlist() -> list[str]:
    if not WATCHLIST_CSV.exists():
        log.error(
            "Watchlist CSV not found at %s — bot has nothing to scan",
            WATCHLIST_CSV,
        )
        return []
    symbols = _read_csv_symbols(WATCHLIST_CSV)
    base_count = len(symbols)
    if CUSTOM_WATCHLIST_CSV.exists():
        extra = _read_csv_symbols(CUSTOM_WATCHLIST_CSV)
        log.info(
            "Custom watchlist: %d symbols from %s",
            len(extra), CUSTOM_WATCHLIST_CSV,
        )
        symbols.extend(extra)
    # Dedupe while preserving order — Nifty 500 entries keep their slots
    # and any custom additions land at the end. Duplicate base entries
    # would otherwise cause the same symbol to be scored twice per scan.
    deduped = list(dict.fromkeys(symbols))
    extras_added = len(deduped) - base_count
    if extras_added > 0:
        log.info(
            "Watchlist size: %d (Nifty500 base + %d custom additions)",
            len(deduped), extras_added,
        )
    if len(deduped) != len(symbols):
        log.warning(
            "Watchlist had %d duplicate symbols; using %d unique entries",
            len(symbols) - len(deduped), len(deduped),
        )
    return deduped


WATCHLIST = _load_watchlist()
