"""Paper-trading tracker — writes, monitor loop, and pure rule helpers.

Public surface:
  open_trade()         — record an alert as a simulated trade
  close_manual()       — operator-driven close (delisting, /skipped, etc.)
  monitor()            — async loop that polls realtime_feed and closes
                         trades on SL/TP/TIMEOUT

Internal helpers (module-private but exercised by unit tests):
  _evaluate_bar        — pure rule logic: side+SL+TP+bar → outcome | None
  _compute_pnl         — gross + net P&L via trading.costs
  _flatten_snapshot    — IndicatorSnapshot → rows for signal_indicators
  _is_timed_out        — predicate using the NSE trading calendar

This module is built up across Phase-5 steps; the helpers below come
first because the rule logic is what every other layer depends on."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import time as _time
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from data.trading_calendar import get_session_close
from trading.costs import round_trip_cost

from .schema import connect, ensure_paper_schema

log = logging.getLogger("alertbot.paper.tracker")

IST = ZoneInfo("Asia/Kolkata")

POLL_INTERVAL_SECONDS = 30
"""How often :func:`monitor` polls realtime_feed for each OPEN trade."""

DEFAULT_ACCOUNT_INR = 100_000.0
"""Default paper-account capital used by :func:`from_signal` for
position sizing. Override per-call when the bot grows multi-portfolio."""

SL_ATR_MULT = 1.5
"""Multiplier on 5m ATR for the stop. Conservative (~1.5×) so the
typical 5m chop doesn't stop us out before the move develops."""

TP1_ATR_MULT = 2.0
"""First-target multiplier on 5m ATR — 1R:1.33R against the stop."""

TP2_ATR_MULT = 3.5
"""Second-target multiplier on 5m ATR — 1R:2.33R against the stop.
Optional partial exit; trades without TP2 just exit fully on TP1."""

ATR_FALLBACK_PCT = 0.015
"""When the indicator snapshot doesn't have a usable atr_5m
(insufficient warmup, NaN, error), fall back to this fraction of
the entry price. 1.5% is roughly the median 5m ATR across the
NSE 500."""

LATE_SESSION_CUTOFF = _time(14, 30)
"""IST time after which no new paper trades are opened. An entry
after 14:30 has <60 min before the same-day TIMEOUT at 15:30 — not
enough room for an MIS setup to develop. Mirrors the no-late-entries
rule a real MIS trader follows (brokers begin auto-squareoff around
15:15-15:20)."""


def _past_entry_cutoff(now_ist: pd.Timestamp | None = None) -> bool:
    """True if it's too late in the session to open new paper trades.

    Used by both ``paper.tracker.from_signal`` and the scanner's
    ``_evaluate_symbol`` so the cutoff applies whether an alert came
    from event-driven or periodic paths. ``now_ist`` is exposed for
    test injection only."""
    if now_ist is None:
        now_ist = pd.Timestamp.now(tz=IST)
    return now_ist.time() >= LATE_SESSION_CUTOFF

_TF_PATTERN = re.compile(r"^\d+[mhd]$")
"""Match a timeframe token like ``5m``, ``15m``, ``60m``, ``1h``,
``1d``. Used by :func:`_flatten_snapshot` to peel the timeframe out
of a namespaced indicator key. Pure-digit suffixes (``orh_15``) do
not match and are treated as part of the indicator name."""


# ---------------------------------------------------------------------------
# Pure helpers — exercised by unit tests without DB or realtime feed
# ---------------------------------------------------------------------------

def _evaluate_bar(
    side: Literal["LONG", "SHORT"],
    sl: float,
    tp1: float,
    tp2: float | None,
    bar,
) -> tuple[str, float, str] | None:
    """Apply SL/TP rules to a single bar's OHLC. Returns:

      ``None``                          — no fill on this bar
      ``(status, exit_price, notes)``   — fill (status in
                                          ``SL`` / ``TP1`` / ``TP2``)

    Tie-break: if SL and TP both hit in the same bar, take TP. Real
    fill-order depends on intra-bar tape and is unknowable from OHLC,
    so the ``notes`` field carries the disclaimer ``coin-flip IRL``.

    ``bar`` is a ``data.realtime_feed.Bar`` (duck-typed on ``.high``,
    ``.low``); ``None`` returns ``None`` so callers can treat
    "no LTP available" uniformly with "no fill yet"."""
    if bar is None:
        return None
    if side == "LONG":
        hit_sl = bar.low <= sl
        hit_tp2 = tp2 is not None and bar.high >= tp2
        hit_tp1 = bar.high >= tp1
    else:
        hit_sl = bar.high >= sl
        hit_tp2 = tp2 is not None and bar.low <= tp2
        hit_tp1 = bar.low <= tp1

    if hit_sl and (hit_tp1 or hit_tp2):
        if hit_tp2:
            return ("TP2", tp2, "SL+TP2 same bar; took TP2 (coin-flip IRL)")
        return ("TP1", tp1, "SL+TP1 same bar; took TP1 (coin-flip IRL)")
    if hit_sl:
        return ("SL", sl, "")
    if hit_tp2:
        return ("TP2", tp2, "")
    if hit_tp1:
        return ("TP1", tp1, "")
    return None


def _compute_pnl(
    side: Literal["LONG", "SHORT"],
    entry: float,
    exit_price: float,
    qty: int,
) -> tuple[float, float]:
    """Return ``(pnl_gross, pnl_net)``. Net = gross minus round-trip
    statutory + brokerage charges at the trade's notional.

    Slippage is NOT applied here. Phase-5 logs signal prices only;
    ``trading.costs.net_r_multiple`` already models fill slippage as
    an R-multiple and would double-count if its delta got pulled into
    this function."""
    if side == "LONG":
        gross = (exit_price - entry) * qty
    else:
        gross = (entry - exit_price) * qty
    notional = entry * qty
    costs = round_trip_cost(notional)["total"]
    return gross, gross - costs


def _flatten_snapshot(snapshot) -> dict[str, tuple[float, str]]:
    """Map an ``IndicatorSnapshot`` (or its ``.values`` dict) to the
    shape that ``signal_indicators`` wants: ``{indicator: (value, tf)}``.

    Namespacing rules (matches ``indicators/compute.py:17-25``):

      ``rsi_5m``              → ``('rsi', '5m')``
      ``macd_5m_histogram``   → ``('macd_histogram', '5m')``
      ``adx_15m_di_plus``     → ``('adx_di_plus', '15m')``
      ``pdh``                 → ``('pdh', 'session')``
      ``orh_15``              → ``('orh_15', 'session')``
                                 (trailing ``15`` is minutes, not a TF
                                 token — no letter suffix)
      ``pivot_classic_r1``    → ``('pivot_classic_r1', 'session')``

    Drops keys whose value is None — those represent insufficient
    warmup or an indicator exception, neither of which deserves a
    row in the signal_indicators table."""
    if snapshot is None:
        return {}
    raw = snapshot if isinstance(snapshot, dict) else snapshot.values
    out: dict[str, tuple[float, str]] = {}
    for key, value in raw.items():
        if value is None:
            continue
        parts = key.split("_")
        timeframe = "session"
        indicator = key
        for i, part in enumerate(parts):
            if _TF_PATTERN.match(part):
                timeframe = part
                indicator = "_".join(parts[:i] + parts[i + 1:])
                break
        out[indicator] = (float(value), timeframe)
    return out


# ---------------------------------------------------------------------------
# SL/TP derivation + signal → trade glue
# ---------------------------------------------------------------------------

def derive_sl_tp(
    price: float,
    atr: float | None,
    *,
    sl_mult: float = SL_ATR_MULT,
    tp1_mult: float = TP1_ATR_MULT,
    tp2_mult: float = TP2_ATR_MULT,
    atr_fallback_pct: float = ATR_FALLBACK_PCT,
) -> tuple[float, float, float]:
    """Return ``(sl, tp1, tp2)`` for a LONG entry at ``price``.

    ``atr`` is the 5m ATR from :func:`indicators.compute_all`; ``None``
    or non-positive falls back to ``price * atr_fallback_pct`` so the
    function never returns NaN. Values rounded to paise (2dp)."""
    if atr is None or atr <= 0:
        atr = price * atr_fallback_pct
    sl = round(price - sl_mult * atr, 2)
    tp1 = round(price + tp1_mult * atr, 2)
    tp2 = round(price + tp2_mult * atr, 2)
    return sl, tp1, tp2


def from_signal(
    signals,
    *,
    account_inr: float = DEFAULT_ACCOUNT_INR,
    side: Literal["LONG", "SHORT"] = "LONG",
) -> int | None:
    """Open a paper trade from a ``StockSignals``-shaped object.

    ``signals`` is duck-typed on ``.symbol``, ``.price``, ``.sl``,
    ``.tp1``, ``.tp2``, ``.score``, ``.snapshot`` — typically a
    :class:`bot.scoring.StockSignals`, but any object with those
    attributes works (tests pass stubs).

    Returns the new ``paper_trades.id``, or ``None`` if the signal
    lacks SL/TP1 or position sizing returns zero. Long-only in
    Phase 5 — SHORT will be wired when scoring starts producing
    short signals."""
    # Local import to keep trading/* out of paper.tracker's module-
    # level path until this glue is actually called.
    from trading import risk as trading_risk

    if _past_entry_cutoff():
        log.info(
            "Late-session cutoff (%s): not opening paper trade for %s",
            LATE_SESSION_CUTOFF, signals.symbol,
        )
        return None
    if signals.sl is None or signals.tp1 is None:
        log.debug("No SL/TP on signal for %s; skipping paper trade",
                  signals.symbol)
        return None
    qty = trading_risk.size_position(account_inr, signals.price, signals.sl)
    if qty <= 0:
        log.info("Paper trade %s: sizing returned qty=0 (price=%.2f sl=%.2f)",
                 signals.symbol, signals.price, signals.sl)
        return None
    return open_trade(
        symbol=signals.symbol,
        side=side,
        entry=signals.price,
        sl=signals.sl,
        tp1=signals.tp1,
        tp2=signals.tp2,
        qty=qty,
        confidence=signals.score / 100.0,
        indicator_snapshot=signals.snapshot,
    )


# ---------------------------------------------------------------------------
# Writes — open_trade / close_manual / _close
# ---------------------------------------------------------------------------

def open_trade(
    symbol: str,
    side: Literal["LONG", "SHORT"],
    entry: float,
    sl: float,
    tp1: float,
    tp2: float | None,
    qty: int,
    confidence: float,
    indicator_snapshot,
    *,
    entry_ts: str | None = None,
) -> int:
    """Persist one simulated trade + its indicator snapshot.

    Returns the new ``paper_trades.id``. If an OPEN trade for
    ``symbol`` already exists, logs a warning and returns its id
    without inserting — this is the in-tracker dedupe guard that
    complements the cooldown row in ``alerts_sent``.

    ``indicator_snapshot`` may be an :class:`indicators.IndicatorSnapshot`,
    a raw ``{key: value}`` dict, or ``None``. ``None`` simply means
    no signal_indicators rows are written (the trade is still recorded).
    """
    ensure_paper_schema()
    if entry_ts is None:
        entry_ts = pd.Timestamp.now(tz=IST).isoformat()

    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM paper_trades "
            "WHERE symbol = ? AND status = 'OPEN' LIMIT 1",
            (symbol,),
        ).fetchone()
        if existing is not None:
            log.info(
                "Paper trade for %s already OPEN (id=%d); skipping duplicate",
                symbol, existing[0],
            )
            return existing[0]

        cur = conn.execute(
            "INSERT INTO paper_trades "
            "(symbol, side, entry_ts, entry_price, qty, stop_loss, "
            " target_1, target_2, confidence, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')",
            (symbol, side, entry_ts, entry, qty, sl, tp1, tp2, confidence),
        )
        trade_id = cur.lastrowid

        flat = _flatten_snapshot(indicator_snapshot)
        if flat:
            conn.executemany(
                "INSERT INTO signal_indicators "
                "(paper_trade_id, indicator, value, timeframe) "
                "VALUES (?, ?, ?, ?)",
                [
                    (trade_id, indicator, value, timeframe)
                    for indicator, (value, timeframe) in flat.items()
                ],
            )
        log.info("Opened paper trade %d: %s %s @ %.2f (SL %.2f, TP1 %.2f)",
                 trade_id, side, symbol, entry, sl, tp1)
        return trade_id


def close_manual(trade_id: int, reason: str) -> None:
    """Operator-driven close (delisted, /skipped reply, /mistake).

    Uses the last partial-bar close from ``realtime_feed`` for the
    exit price. If no LTP is available (e.g. ticks stopped, symbol
    delisted intraday), ``exit_price`` and ``pnl_*`` columns are left
    NULL — ``status='MANUAL'`` and ``notes=reason`` are still written
    so the trade is unambiguously closed. No-op if the trade isn't
    OPEN."""
    from data.realtime_feed import get_current_partial

    ensure_paper_schema()
    with connect() as conn:
        row = conn.execute(
            "SELECT symbol, status FROM paper_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"paper_trades id={trade_id} not found")
    symbol, status = row
    if status != "OPEN":
        log.warning("close_manual on non-OPEN trade %d (status=%s); noop",
                    trade_id, status)
        return

    partial = get_current_partial(symbol)
    exit_price = float(partial.close) if partial is not None else None
    _close(trade_id, "MANUAL", exit_price, notes=reason)


def _close(
    trade_id: int,
    status: Literal["TP1", "TP2", "SL", "TIMEOUT", "MANUAL"],
    exit_price: float | None,
    *,
    notes: str | None = None,
) -> None:
    """Mark a trade closed. Computes pnl_gross/pnl_net via
    :func:`_compute_pnl` when ``exit_price`` is available; leaves
    them NULL otherwise (MANUAL on a delisted symbol).

    Used by both :func:`monitor` (SL/TP/TIMEOUT fills) and
    :func:`close_manual`."""
    with connect() as conn:
        row = conn.execute(
            "SELECT side, entry_price, qty FROM paper_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"paper_trades id={trade_id} not found")
        side, entry_price, qty = row
        exit_ts = pd.Timestamp.now(tz=IST).isoformat()
        if exit_price is not None:
            gross, net = _compute_pnl(side, entry_price, exit_price, qty)
        else:
            gross, net = None, None
        conn.execute(
            "UPDATE paper_trades "
            "SET status=?, exit_ts=?, exit_price=?, "
            "    pnl_gross=?, pnl_net=?, notes=? "
            "WHERE id = ?",
            (status, exit_ts, exit_price, gross, net, notes, trade_id),
        )
    log.info("Closed paper trade %d as %s @ %s",
             trade_id, status,
             f"{exit_price:.2f}" if exit_price is not None else "no LTP")


# ---------------------------------------------------------------------------
# TIMEOUT predicate
# ---------------------------------------------------------------------------

def _is_timed_out(
    entry_ts_iso: str,
    now_ist: pd.Timestamp | None = None,
) -> bool:
    """True if a trade entered at ``entry_ts_iso`` is past 15:30 IST
    of the SAME trading day as entry. Strict intraday — every paper
    trade auto-squares-off at session close.

    Why: ``trading.costs.round_trip_cost`` quotes intraday MIS rates
    (see ``trading/costs.py:9``). Holding overnight in real trading
    would either trigger broker auto-squareoff at ~15:15-15:20 or
    convert to delivery, both of which have different cost profiles.
    Same-day TIMEOUT keeps the simulation honest with the cost
    model."""
    entry_ts = pd.Timestamp(entry_ts_iso)
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize(IST)
    else:
        entry_ts = entry_ts.tz_convert(IST)
    if now_ist is None:
        now_ist = pd.Timestamp.now(tz=IST)
    deadline = get_session_close(entry_ts.date())
    return now_ist >= deadline


# ---------------------------------------------------------------------------
# Monitor: async loop + sync per-tick body
# ---------------------------------------------------------------------------

def _monitor_once(
    get_partial: Callable[[str], object],
    market_open_check: Callable[[], bool],
) -> None:
    """Scan every OPEN trade once and close any that hit SL/TP/TIMEOUT.

    Pure synchronous — :func:`monitor` wraps this in an async sleep
    loop, but every action inside the tick (DB reads/writes, the
    realtime-feed fetch) is sync, so unit tests drive single ticks
    without needing pytest-asyncio.

    Rule precedence: SL/TP fills win over TIMEOUT. If a bar at 15:29
    would hit TP1, we record TP1 even though TIMEOUT (15:30 same-day)
    would fire on the next tick. The informative status (we know
    *why* it closed) is more useful for backtest analysis than the
    coincidence of running out of clock at the same tick."""
    # Local import to break the import cycle (journal imports tracker
    # indirectly via schema; this side keeps it lazy).
    from .journal import list_open

    market_open = market_open_check()
    for trade in list_open():
        bar = get_partial(trade.symbol)
        # Rule check first — only valid when market is open (no point
        # evaluating SL/TP against stale after-hours data).
        if market_open:
            result = _evaluate_bar(
                trade.side, trade.stop_loss, trade.target_1,
                trade.target_2, bar,
            )
            if result is not None:
                status, exit_price, notes = result
                _close(trade.id, status, exit_price, notes=notes or None)
                continue
        # No rule fill — check the wall-clock TIMEOUT (fires regardless
        # of market state).
        if _is_timed_out(trade.entry_ts):
            if bar is not None:
                _close(
                    trade.id, "TIMEOUT", float(bar.close),
                    notes="timeout at session close",
                )
            else:
                _close(
                    trade.id, "TIMEOUT", None,
                    notes="timeout; no LTP available",
                )


async def monitor(stop_event: asyncio.Event | None = None) -> None:
    """Long-running coroutine: poll OPEN trades every
    :data:`POLL_INTERVAL_SECONDS` and close them when rules fire.

    Mounted in ``bot.runner.main`` alongside the scanner / health /
    fast-move tasks. ``stop_event`` is honored for clean shutdown —
    pass one in from the runner; tests pass a fresh ``asyncio.Event``
    and set it after the iterations they want to exercise.

    Catches and logs exceptions from :func:`_monitor_once` so a
    single bad tick (e.g. transient DB error) doesn't take the whole
    loop down."""
    from bot.schedule import is_market_open
    from data.realtime_feed import get_current_partial

    log.info("Paper-tracker monitor starting (poll=%ds)", POLL_INTERVAL_SECONDS)
    if stop_event is None:
        stop_event = asyncio.Event()

    while not stop_event.is_set():
        try:
            _monitor_once(get_current_partial, is_market_open)
        except Exception:
            log.exception("paper.monitor: tick failed")
        try:
            # Sleep until either POLL_INTERVAL_SECONDS elapses (TimeoutError)
            # or stop_event is set (returns normally) — whichever first.
            await asyncio.wait_for(
                stop_event.wait(), timeout=POLL_INTERVAL_SECONDS,
            )
            break  # stop_event was set
        except (asyncio.TimeoutError, TimeoutError):
            continue
    log.info("Paper-tracker monitor stopped")
