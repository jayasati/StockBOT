"""Per-tick scan: refresh caches, fetch intraday data, score, dispatch."""
from __future__ import annotations

import logging

import pandas as pd

import suppression
from data import filings

from . import market_data
from .config import settings
from .notifier import Telegram, format_alert
from .scoring import StockSignals, score_stock
from .storage import record_alert
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot")


async def scan_once(telegram: Telegram) -> None:
    market_data.refresh_daily_cache_if_stale()
    await market_data.refresh_asm_gsm_if_stale()

    new_filings = await filings.poll_filings()
    for symbol, classification, title, _link in new_filings:
        log.info("Filing [%s] %s: %s", classification, symbol, title)
    fundamentals = filings.recent_high_priority(60)

    log.info("Scanning %d symbols...", len(WATCHLIST))
    intraday_data = market_data.fetch_intraday(WATCHLIST)
    if not intraday_data:
        log.warning("No intraday data fetched (rate limit? network?). Skipping scan.")
        return

    candidates: list[StockSignals] = []
    for symbol in WATCHLIST:
        if symbol not in intraday_data:
            continue
        intraday = intraday_data[symbol]
        daily = market_data._daily_cache.get(symbol, pd.DataFrame())
        if daily.empty:
            continue
        signals = score_stock(symbol, intraday, daily)

        # Fundamental catalyst bonus: binary_high BSE filing in last 60 min.
        # +30 reflects that an earnings beat / order win / acquisition tends
        # to drive several ATRs of move, dwarfing microstructure signals.
        if symbol in fundamentals:
            signals.score = min(100, signals.score + 30)
            signals.filing_title = fundamentals[symbol]
            signals.reasons.append("📰 filing")

        log.debug("%s: score=%d %s", symbol, signals.score, signals.reasons)

        if signals.score >= settings.composite_threshold:
            blocked, reason = suppression.is_suppressed(
                symbol, settings.cooldown_minutes
            )
            if blocked:
                log.info("Suppressed %s: %s", symbol, reason)
                continue
            candidates.append(signals)

    candidates.sort(key=lambda s: -s.score)

    sent = 0
    for s in candidates[: settings.max_alerts_per_scan]:
        await telegram.send(format_alert(s))
        record_alert(s.symbol, s.score, ", ".join(s.reasons), s.price)
        log.info("Alerted: %s score=%d reasons=%s", s.symbol, s.score, s.reasons)
        sent += 1

    log.info(
        "Scan complete. Candidates: %d, Alerted: %d (capped at %d)",
        len(candidates), sent, settings.max_alerts_per_scan,
    )
