"""Filings poll orchestrator + recent high-priority lookup.

Reads/writes the ``filings_seen`` table (schema + master init in
:mod:`bot.db`). This module only does inserts + reads; it does not own
the DDL."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from .classify import classify
from .http import _fetch_announcements
from .matcher import match_ticker
from .parse import _filing_id, _item_dt_iso, _item_link, _item_title

log = logging.getLogger("alertbot.filings")

DB_PATH = Path("alerts.db")


def _existing_ids() -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT filing_id FROM filings_seen").fetchall()
    return {r[0] for r in rows}


async def poll_filings() -> list[tuple[str, str, str, str]]:
    """Fetch BSE announcements and return new (symbol, classification, title, link).

    On the first poll (empty DB), entries are seeded as 'seen' but NOT
    returned — we don't want to flood Telegram with stale filings as
    fundamental catalysts on bot startup.
    """
    items = await _fetch_announcements()
    if items is None:
        return []
    if not items:
        log.info("Filings poll: 0 announcements returned")
        return []

    seen_ids = _existing_ids()
    is_first_poll = len(seen_ids) == 0

    new: list[tuple[str, str, str, str]] = []
    matched = 0
    unmapped = 0

    with sqlite3.connect(DB_PATH) as conn:
        for item in items:
            fid = _filing_id(item)
            if not fid or fid in seen_ids:
                continue
            title = _item_title(item)
            if not title:
                continue
            symbol = match_ticker(title)
            if symbol is None:
                unmapped += 1
                continue
            matched += 1
            classification = classify(title)
            link = _item_link(item)
            conn.execute(
                "INSERT OR IGNORE INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (fid, symbol, title, classification, _item_dt_iso(item)),
            )
            if not is_first_poll:
                new.append((symbol, classification, title, link))

    if unmapped:
        log.warning(
            "Filings poll: %d entries with company names not in mapping (skipped)",
            unmapped,
        )
    if is_first_poll and matched:
        log.info(
            "Filings poll: seeded %d existing watchlist entries on first run "
            "(no alerts on this pass)",
            matched,
        )
    else:
        log.info(
            "Filings poll: %d entries, %d matched watchlist, %d new",
            len(items), matched, len(new),
        )
    return new


def recent_high_priority(minutes: int) -> dict[str, str]:
    """Return {symbol: title} of the most recent binary_high filings in the
    window. These are directionally clear positives (orders, dividends, PLI,
    buybacks, bonus issues) and earn the composite scorer's +30 bonus."""
    return _recent_by_class("binary_high", minutes)


def recent_unknown_events(minutes: int) -> dict[str, str]:
    """Return {symbol: title} of recent ``event_unknown`` filings — earnings,
    M&A, allotments, schemes. Direction can't be inferred from the title,
    so the scorer surfaces these as an informational tag without applying
    a score bonus (positive OR negative)."""
    return _recent_by_class("event_unknown", minutes)


def _recent_by_class(classification: str, minutes: int) -> dict[str, str]:
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT symbol, title FROM filings_seen "
            "WHERE classification = ? AND seen_at > ? "
            "ORDER BY seen_at DESC",
            (classification, cutoff),
        ).fetchall()
    result: dict[str, str] = {}
    for sym, title in rows:
        if sym not in result:
            result[sym] = title
    return result
