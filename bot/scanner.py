"""Per-tick scan: refresh caches, fetch intraday data, score, dispatch.

Two callers:
  * ``scan_once`` — periodic batch scan over the whole watchlist (used by the
    runner loop). Also refreshes filings / ASM / daily-cache.
  * ``scan_symbol`` — single-symbol scan, called from the bar-event consumer
    when a 5-min bar closes for that symbol. Sub-second latency from bar
    close to Telegram message.

Both paths share ``_evaluate_symbol``, which scores + applies suppression so
the alert criterion stays identical."""
from __future__ import annotations

import logging

import pandas as pd

from data import filings

from . import market_data, suppression
from .config import settings
from .notifier import Telegram, format_alert
from .scoring import StockSignals, score_stock
from .storage import record_alert
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot.scan")


def _evaluate_symbol(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    fundamentals: dict[str, str],
    unknown_events: dict[str, str] | None = None,
) -> StockSignals | None:
    """Score one symbol; apply filing bonus; return signals only if it
    crosses the configured threshold *and* is not suppressed. Returns None
    otherwise. Pure over already-fetched data.

    ``fundamentals`` is the binary_high filing map (directionally positive:
    order wins, dividends, PLI). These get the +30 score bonus.

    ``unknown_events`` is the event_unknown map (earnings, M&A, allotments).
    Surfaced as a ``📢 event`` tag with the filing title so the user sees
    the catalyst, but no score change — direction is ambiguous without the
    body text (PVRINOX 2026-05-11 fired the +30 on weak earnings, sank 4%)."""
    if daily.empty:
        return None
    signals = score_stock(symbol, intraday, daily)
    if symbol in fundamentals:
        # Directionally positive filing: an order win / dividend / PLI
        # tends to drive several ATRs of move, dwarfing microstructure
        # signals. The +30 is intentionally large because the direction is
        # known.
        signals.score = min(100, signals.score + 30)
        signals.filing_title = fundamentals[symbol]
        signals.reasons.append("📰 filing")
    elif unknown_events and symbol in unknown_events:
        # Direction-ambiguous event (e.g. earnings just released). Flag
        # for awareness but do not bias the score. If microstructure is
        # already strong enough to cross threshold on its own, we'll alert
        # and the user can read the PDF themselves.
        signals.filing_title = unknown_events[symbol]
        signals.reasons.append("📢 event")
    log.debug("%s: score=%d %s", symbol, signals.score, signals.reasons)
    if signals.score < settings.composite_threshold:
        return None
    blocked, reason = suppression.is_suppressed(symbol, settings.cooldown_minutes)
    if blocked:
        log.info("Suppressed %s: %s", symbol, reason)
        return None
    return signals


async def _dispatch(telegram: Telegram, signals: StockSignals) -> None:
    """Persist + Telegram. Record_alert lands first so a concurrent path
    (e.g. periodic scan and bar-event firing back-to-back) sees the
    cooldown row and skips the duplicate."""
    record_alert(signals.symbol, signals.score, ", ".join(signals.reasons), signals.price)
    await telegram.send(format_alert(signals))
    log.info("Alerted: %s score=%d reasons=%s",
             signals.symbol, signals.score, signals.reasons)


async def scan_symbol(
    telegram: Telegram,
    symbol: str,
    fundamentals: dict[str, str],
    unknown_events: dict[str, str] | None = None,
) -> None:
    """Event-driven scan for a single symbol. Called from the bar-event
    consumer when a 5-min bar closes. No max-alerts cap — alerts arrive
    one-at-a-time as bars complete, so spam control is handled by the
    cooldown row alone."""
    if symbol not in WATCHLIST:
        return
    intraday_data = market_data.fetch_intraday([symbol])
    if symbol not in intraday_data:
        return
    daily = market_data._daily_cache.get(symbol, pd.DataFrame())
    signals = _evaluate_symbol(
        symbol, intraday_data[symbol], daily, fundamentals, unknown_events,
    )
    if signals is None:
        return
    await _dispatch(telegram, signals)


async def scan_once(telegram: Telegram) -> None:
    """Periodic batch scan: refresh daily/ASM/filings, score every symbol,
    rank by score, dispatch up to ``max_alerts_per_scan``. Acts as a safety
    net for symbols whose bar-complete events were missed (e.g. during a
    websocket reconnect)."""
    market_data.refresh_daily_cache_if_stale()
    await market_data.refresh_asm_gsm_if_stale()

    new_filings = await filings.poll_filings()
    for symbol, classification, title, _link in new_filings:
        log.info("Filing [%s] %s: %s", classification, symbol, title)
    fundamentals = filings.recent_high_priority(60)
    unknown_events = filings.recent_unknown_events(60)

    log.info("Scanning %d symbols...", len(WATCHLIST))
    intraday_data = market_data.fetch_intraday(WATCHLIST)
    if not intraday_data:
        log.warning("No intraday data fetched (rate limit? network?). Skipping scan.")
        return

    candidates: list[StockSignals] = []
    for symbol in WATCHLIST:
        if symbol not in intraday_data:
            continue
        daily = market_data._daily_cache.get(symbol, pd.DataFrame())
        signals = _evaluate_symbol(
            symbol, intraday_data[symbol], daily, fundamentals, unknown_events,
        )
        if signals is not None:
            candidates.append(signals)

    candidates.sort(key=lambda s: -s.score)

    sent = 0
    for s in candidates[: settings.max_alerts_per_scan]:
        await _dispatch(telegram, s)
        sent += 1

    log.info(
        "Scan complete. Candidates: %d, Alerted: %d (capped at %d)",
        len(candidates), sent, settings.max_alerts_per_scan,
    )
