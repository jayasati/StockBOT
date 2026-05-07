"""BSE company name → NSE ticker mapping (built from NIFTY 500 CSV).

Keys are pre-normalized: uppercase, '&' → 'AND', punctuation stripped,
trailing LIMITED/LTD removed. The matcher applies the same normalization
to incoming filing titles, then does longest-prefix lookup.

To change the universe, replace ind_nifty500list.csv (download from:
https://archives.nseindia.com/content/indices/ind_nifty500list.csv).
"""
from __future__ import annotations

import csv
import html
import logging
import re
from pathlib import Path

log = logging.getLogger("alertbot.filings")

WATCHLIST_CSV = Path("ind_nifty500list.csv")


def _normalize(s: str) -> str:
    """Uppercase, '&' → 'AND', strip punctuation, drop LIMITED/LTD, collapse spaces."""
    s = html.unescape(s).upper().replace("&", " AND ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(LIMITED|LTD)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_name_to_ticker() -> dict[str, str]:
    """Build {normalized_company_name: ticker.NS} from the NIFTY 500 CSV."""
    if not WATCHLIST_CSV.exists():
        log.warning(
            "Watchlist CSV not found at %s — filings matcher will be empty",
            WATCHLIST_CSV,
        )
        return {}
    mapping: dict[str, str] = {}
    with WATCHLIST_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            company = (row.get("Company Name") or "").strip()
            symbol = (row.get("Symbol") or "").strip()
            if not (company and symbol):
                continue
            key = _normalize(company)
            if key:
                mapping[key] = f"{symbol}.NS"
    return mapping


NAME_TO_TICKER: dict[str, str] = _load_name_to_ticker()
_SORTED_NAMES = sorted(NAME_TO_TICKER.keys(), key=len, reverse=True)


def match_ticker(title: str) -> str | None:
    """Return the NSE ticker if the title starts with a known company name."""
    norm = _normalize(title)
    for name in _SORTED_NAMES:
        if norm == name or norm.startswith(name + " "):
            return NAME_TO_TICKER[name]
    return None
