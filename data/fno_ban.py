"""F&O ban list fetcher.

NSE publishes the list of securities banned from F&O trading daily
at ``archives.nseindia.com/content/fo/fo_secban.csv``. Banned
symbols typically have extreme OI and elevated volatility; opening
new positions on them is a discipline issue.

Phase-6 ``hard.fno_ban_list`` reads from ``ctx.fno_banned`` — a set
of bare NSE symbol names (no ``.NS`` suffix). The scanner is
responsible for calling :func:`refresh` periodically (typically
once per scan-batch) and populating ``ctx.fno_banned`` from
:func:`get_banned`.

Fail-open semantics: a fetch failure (NSE 5xx, network outage,
schema change) leaves the previous good set in place. If we've
never had a good fetch, the set is empty — and the filter
fails-open on empty.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import time
from typing import Iterable

import httpx

from .index_feed import IST  # reuse the IST zoneinfo

log = logging.getLogger("alertbot.fno_ban")

FNO_BAN_URL = "https://archives.nseindia.com/content/fo/fo_secban.csv"
"""NSE archive URL. CSV with one symbol per line (after a header
section). Endpoint occasionally returns HTML if NSE rate-limits us
— we sniff for the expected ``;Symbol`` pattern."""

REFRESH_TTL_SECONDS = 6 * 3600
"""Refresh at most every 6 hours. The list updates once a day."""

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_banned: set[str] = set()
_last_refresh_at: float = 0.0
_lock = asyncio.Lock()


def get_banned() -> set[str]:
    """Read the cached banned set. Returns a copy so callers can't
    mutate the cache. Empty when no successful fetch has occurred —
    the filter fails-open in that case."""
    return set(_banned)


def _parse(text: str) -> set[str]:
    """Parse the NSE secban CSV body. Format varies — historically
    a fixed header line of dashes followed by ``Sr.No;Symbol`` rows,
    sometimes with an explanatory preamble. We look for any line
    matching ``<digits>;<symbol>`` and take the symbol column."""
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue  # header / preamble lines
        symbol = parts[1].upper()
        if symbol and symbol.replace("-", "").replace("&", "").isalnum():
            out.add(symbol)
    return out


async def refresh() -> set[str]:
    """Fetch the CSV and update the in-memory cache. Returns the
    new set (or the previous one if fetch failed). Idempotent —
    respects the ``REFRESH_TTL_SECONDS`` throttle so calling
    every-scan is cheap."""
    global _banned, _last_refresh_at
    async with _lock:
        now = time.time()
        if (now - _last_refresh_at) < REFRESH_TTL_SECONDS and _banned:
            return set(_banned)
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=20, follow_redirects=True,
            ) as client:
                r = await client.get(FNO_BAN_URL)
            if r.status_code != 200:
                log.warning("F&O ban fetch HTTP %d", r.status_code)
                return set(_banned)
            new_set = _parse(r.text)
            if not new_set:
                # NSE sometimes returns an empty CSV on weekends or
                # when no symbols are banned. Either way: respect it.
                # But log so a parser regression is visible.
                log.info("F&O ban list parsed as empty (no bans today, "
                         "or parser regression — check response format)")
            _banned = new_set
            _last_refresh_at = now
            log.info("F&O ban list refreshed: %d symbols", len(_banned))
            return set(_banned)
        except httpx.RequestError as e:
            log.warning("F&O ban fetch network error: %s", e)
            return set(_banned)
        except Exception:
            log.exception("F&O ban refresh failed")
            return set(_banned)


def _reset_for_tests() -> None:
    """Test-only: clear the in-memory cache so a fresh refresh runs."""
    global _banned, _last_refresh_at
    _banned = set()
    _last_refresh_at = 0.0
