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
from datetime import date as _date
from typing import TYPE_CHECKING

import pandas as pd

from data import filings
from paper import tracker as paper_tracker

from . import market_data, suppression
from .config import settings
from .notifier import Telegram, format_alert
from .scoring import StockSignals, score_stock
from .storage import record_alert
from .watchlist import WATCHLIST

if TYPE_CHECKING:
    from indicators import IndicatorSnapshot

log = logging.getLogger("alertbot.scan")


def _build_snapshot(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    session_date: _date,
) -> "IndicatorSnapshot | None":
    """Build the Phase-4 indicator snapshot for the scorer.

    Restricted to the indicators the current ``score_stock`` actually
    reads (RSI on 5m). Other Phase-4 indicators are computed and tested
    in the registry but not yet consumed by scoring — Phase 7 will
    expand this ``indicators`` tuple as it pulls in more components
    (MACD, ADX, ATR, levels, regime gating, …).

    Returns None on empty input or any compute error so ``score_stock``
    transparently falls back to the legacy SMA-rolling RSI path."""
    if intraday.empty or daily.empty:
        return None
    bars_lower = intraday.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    try:
        # Local import keeps the indicators package out of the bot.scan
        # import path until it's actually needed (test discovery + cold
        # imports stay fast).
        from indicators import compute_all
        return compute_all(
            symbol=symbol,
            bars=bars_lower,
            daily_df=daily,
            session_date=session_date,
            target_timeframes=("5m",),
            # Phase-5: expanded from rsi-only so signal_indicators
            # captures enough to drive win_rate_by_indicator. RSI is
            # still the only one score_stock currently reads; the
            # rest are journaled but not scored until Phase 7.
            indicators=("rsi", "atr", "adx", "macd"),
        )
    except Exception:
        log.exception("compute_all failed for %s; falling back to legacy RSI",
                      symbol)
        return None


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
    # Late-session cutoff: skip the whole evaluation past 14:30 IST.
    # A 14:30 entry has <60 min before same-day TIMEOUT at 15:30 — not
    # enough for a typical MIS setup to develop. Cheapest path runs first.
    if paper_tracker._past_entry_cutoff():
        log.debug("Late-session cutoff: skipping %s", symbol)
        return None
    if daily.empty:
        return None
    session_date = (
        intraday.index[-1].date()
        if not intraday.empty and isinstance(intraday.index, pd.DatetimeIndex)
        else None
    )
    snapshot = (
        _build_snapshot(symbol, intraday, daily, session_date)
        if session_date is not None
        else None
    )
    signals = score_stock(symbol, intraday, daily, snapshot=snapshot)
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
    _attach_sl_tp(signals, snapshot)
    return signals


def _attach_sl_tp(
    signals: StockSignals,
    snapshot: "IndicatorSnapshot | None",
) -> None:
    """Compute ATR-based SL/TP/TP2 for the alert and stash them, plus
    the indicator snapshot, on the ``StockSignals``. ``format_alert``
    renders SL/TP into the Telegram message; ``_dispatch`` hands the
    same numbers to the paper-tracker. Single derivation point so the
    two never drift."""
    atr_5m = snapshot.values.get("atr_5m") if snapshot is not None else None
    sl, tp1, tp2 = paper_tracker.derive_sl_tp(signals.price, atr_5m)
    signals.sl = sl
    signals.tp1 = tp1
    signals.tp2 = tp2
    signals.snapshot = snapshot


async def _dispatch(telegram: Telegram, signals: StockSignals) -> None:
    """Persist + (maybe) Telegram + paper trade.

    Three-tier gating (see idea/promptchain_noise_reduction.txt):

      score < composite_threshold       — never reaches here (filtered upstream)
      composite_threshold <= score < telegram_threshold
                                        — silent paper trade, no Telegram
      score >= telegram_threshold       — paper trade + Telegram

    ``record_alert`` lands first regardless so the cooldown row blocks
    a concurrent path (e.g. periodic scan + bar-event back-to-back)
    from opening a second paper trade on the same symbol."""
    record_alert(signals.symbol, signals.score, ", ".join(signals.reasons), signals.price)
    if signals.score >= settings.telegram_threshold:
        await telegram.send(format_alert(signals))
        log.info("Alerted: %s score=%d reasons=%s",
                 signals.symbol, signals.score, signals.reasons)
    else:
        log.info("Silent paper trade: %s score=%d (below telegram_threshold=%d)",
                 signals.symbol, signals.score, settings.telegram_threshold)
    try:
        paper_tracker.from_signal(signals)
    except Exception:
        log.exception("Paper trade open failed for %s", signals.symbol)


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
