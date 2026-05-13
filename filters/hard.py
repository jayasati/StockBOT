"""Hard filters: kill-or-pass. Each function returns a kill-reason
string to short-circuit the chain, or ``None`` to let downstream
filters run.

Order matters for cost — cheap pure-Python checks first (time,
liquidity), DB-bound checks (suppression, filings) next, remote-
fetch checks (F&O ban, circuit bands) last. The chain
short-circuits on the FIRST non-empty return, so audit rows carry
exactly one kill reason."""
from __future__ import annotations

import logging
import sqlite3
from datetime import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.scoring import StockSignals
    from .chain import FilterContext


log = logging.getLogger("alertbot.filters.hard")

MARKET_OPEN_START = _time(9, 30)
"""09:30 IST — first 15 min is opening-window noise (handled as a
soft penalty by :mod:`filters.time`). Hard filter kills before that."""

MARKET_OPEN_END = _time(15, 0)
"""15:00 IST — gives the same-day TIMEOUT (15:30) 30 min of runway.
Tighter than the broker's auto-squareoff (15:15-15:20) to leave
slack for paper-trade rule evaluation."""

LIQUIDITY_MIN_TURNOVER_CR = 5.0
"""20-day average daily RUPEE turnover floor, in crores. Module-level
default; the filter reads ``settings.liquidity_min_turnover_cr`` at
call time so the env var ``LIQUIDITY_MIN_TURNOVER_CR`` is the
operator-facing knob.

Share-count floors penalise high-priced names — MRF at ~7,000
shares/day is ₹86cr/day turnover, very liquid in money terms but
nuked by any reasonable share-count threshold. The rupee metric
matches how slippage actually scales (depth in rupees, not in
shares)."""


# ---------------------------------------------------------------------------
# Time / market state
# ---------------------------------------------------------------------------

def market_open(signals: "StockSignals", ctx: "FilterContext") -> str | None:
    """Kill outside MARKET_OPEN_START..MARKET_OPEN_END IST."""
    t = ctx.now.time()
    if t < MARKET_OPEN_START or t >= MARKET_OPEN_END:
        return f"market_closed ({t.strftime('%H:%M')})"
    return None


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------

def liquidity(signals: "StockSignals", ctx: "FilterContext") -> str | None:
    """Kill if 20-day average daily RUPEE turnover is below the floor.

    Turnover = ``mean(Volume × Close)`` over the last 20 daily bars,
    divided by 1e7 to express in crores. Compared against
    ``settings.liquidity_min_turnover_cr`` (default 5.0, env-tunable
    via ``LIQUIDITY_MIN_TURNOVER_CR``).

    Fail-open on missing data: empty/missing daily_df, missing
    Close/Volume columns, or NaN turnover all return None so a
    cache miss can't silence the whole universe."""
    from bot.config import settings

    if ctx.daily_df is None or ctx.daily_df.empty:
        return None
    tail = ctx.daily_df.tail(20)
    if "Volume" not in tail.columns or "Close" not in tail.columns:
        return None
    try:
        turnover = float((tail["Volume"] * tail["Close"]).mean())
    except (TypeError, ValueError):
        return None
    if turnover != turnover:  # NaN
        return None
    turnover_cr = turnover / 1e7
    threshold_cr = float(settings.liquidity_min_turnover_cr)
    if turnover_cr < threshold_cr:
        return (
            f"liquidity (20d avg turnover ₹{turnover_cr:,.2f}cr "
            f"< ₹{threshold_cr:,.2f}cr)"
        )
    return None


# ---------------------------------------------------------------------------
# Suppression layer delegate (ASM / GSM / pledge / paper-open / cooldown)
# ---------------------------------------------------------------------------

def ban_period(signals: "StockSignals", ctx: "FilterContext") -> str | None:
    """Delegate to :func:`bot.suppression.rules.is_suppressed`. This
    is the merge point — the inline ``is_suppressed`` call in
    ``bot/scanner.py`` is REMOVED in Phase 6 in favour of this
    filter, so the same suppression layer runs exactly once."""
    from bot.config import settings
    from bot.suppression import is_suppressed
    blocked, reason = is_suppressed(signals.symbol, settings.cooldown_minutes)
    if blocked:
        return reason
    return None


# ---------------------------------------------------------------------------
# Filings — corporate action today
# ---------------------------------------------------------------------------

def corporate_action_today(
    signals: "StockSignals", ctx: "FilterContext",
) -> str | None:
    """Kill if a directionally-clear binary_high filing
    (dividend/split/bonus/order) for this symbol landed today.

    Filings classifier already routes those to ``binary_high``
    (``data/filings/classify.py:86``); we just consult the seen log."""
    from bot.config import DB_PATH

    today = ctx.now.date().isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT 1 FROM filings_seen "
                "WHERE symbol = ? "
                "  AND classification = 'binary_high' "
                "  AND substr(seen_at, 1, 10) = ? "
                "LIMIT 1",
                (signals.symbol, today),
            ).fetchone()
    except sqlite3.OperationalError:
        # filings_seen missing: fail-open.
        return None
    if row:
        return "corporate_action_today"
    return None


# ---------------------------------------------------------------------------
# Filings — earnings within 3 sessions
# ---------------------------------------------------------------------------

def earnings_within_3d(
    signals: "StockSignals", ctx: "FilterContext",
) -> str | None:
    """Kill if an ``event_unknown`` filing (earnings, M&A, allotment)
    landed in the last 3 calendar days. Earnings drift continues to
    move the stock; we don't want microstructure signals adding
    noise on top of a fundamental re-rating event."""
    from bot.config import DB_PATH
    from datetime import timedelta

    cutoff = (ctx.now - timedelta(days=3)).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT 1 FROM filings_seen "
                "WHERE symbol = ? "
                "  AND classification = 'event_unknown' "
                "  AND seen_at >= ? "
                "LIMIT 1",
                (signals.symbol, cutoff),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row:
        return "earnings_within_3d"
    return None


# ---------------------------------------------------------------------------
# Stubs — wired in subsequent Phase-6 tasks. Pinned-pass behaviour so
# the audit table doesn't lose every signal the day they're enabled.
# ---------------------------------------------------------------------------

CIRCUIT_DEFAULT_BAND_PCT = 0.10
"""Default ±10% price band — matches the majority of NSE Series-EQ
stocks. The real NSE rules vary by series (T2T: 5%, low-price: 20%,
F&O: unlimited). To replace with authoritative per-symbol bands,
fetch from the NSE quote-equity API and store in ``risk_flags``;
this filter reads from there preferentially when present."""

CIRCUIT_PROXIMITY_PCT = 0.02
"""Within 2% of either band → kill. Spec value."""


def circuit_proximity(
    signals: "StockSignals", ctx: "FilterContext",
) -> str | None:
    """Kill if today's LTP is within 2% of the upper or lower
    circuit band.

    HEURISTIC version: assumes a ±10% band from prev day's close
    (the most common NSE Series-EQ rule). Misses T2T-segment 5%
    bands and low-price-tier 20% bands — an authoritative
    per-symbol-band fetcher would replace this; until then the
    heuristic catches the obvious cases without an NSE API
    dependency.

    Fail-open on missing daily_df (no prev close → no band)."""
    if ctx.daily_df is None or ctx.daily_df.empty:
        return None
    try:
        prev_close = float(ctx.daily_df["Close"].iloc[-1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    if prev_close <= 0:
        return None

    upper_band = prev_close * (1 + CIRCUIT_DEFAULT_BAND_PCT)
    lower_band = prev_close * (1 - CIRCUIT_DEFAULT_BAND_PCT)
    price = signals.price

    upper_proximity_floor = upper_band * (1 - CIRCUIT_PROXIMITY_PCT)
    lower_proximity_ceiling = lower_band * (1 + CIRCUIT_PROXIMITY_PCT)

    if price >= upper_proximity_floor:
        gap = (upper_band - price) / upper_band * 100.0
        return f"circuit_proximity (upper, {gap:+.2f}%)"
    if price <= lower_proximity_ceiling:
        gap = (price - lower_band) / lower_band * 100.0
        return f"circuit_proximity (lower, {gap:+.2f}%)"
    return None


def fno_ban_list(
    signals: "StockSignals", ctx: "FilterContext",
) -> str | None:
    """Kill if symbol is on today's F&O ban list. Reads
    ``ctx.fno_banned`` — the index_feed/scanner is responsible for
    refreshing the set daily.

    Fail-open when the set is empty (don't kill every symbol just
    because the fetcher failed)."""
    if not ctx.fno_banned:
        return None
    # Strip ``.NS`` suffix for matching — NSE ban list uses bare symbols.
    bare = signals.symbol.replace(".NS", "")
    if bare in ctx.fno_banned:
        return f"fno_ban ({bare})"
    return None


def nifty_crash(
    signals: "StockSignals", ctx: "FilterContext",
) -> str | None:
    """Kill if NIFTY 50 is down >1.5% on the day. ``ctx.nifty_pct``
    is filled by the scanner from the shared index feed.
    Fail-open when None (don't kill on missing data)."""
    if ctx.nifty_pct is None:
        return None
    if ctx.nifty_pct <= -1.5:
        return f"nifty_crash ({ctx.nifty_pct:.2f}%)"
    return None


HARD_FILTERS = (
    # Cheap / local first
    market_open,
    liquidity,
    # Then DB-bound
    ban_period,
    corporate_action_today,
    earnings_within_3d,
    # Then context-driven (depend on shared FilterContext)
    nifty_crash,
    fno_ban_list,
    # STUB last (always-pass for now)
    circuit_proximity,
)
