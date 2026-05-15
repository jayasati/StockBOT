"""Same-day 5-min bar backfill from Fyers history.

Why this exists: when the bot starts mid-session (process restart,
late launch, network blip), the live WebSocket only sees ticks from
``subscribe()`` onward. Bars before that point are missing from
``bars_5m``, so indicators that depend on a contiguous series
(RSI, EMA, ADX, VWAP) emit garbage on the first scan.

This module fills the gap. At startup, for each watchlist symbol:

  1. Count today's bars already in ``bars_5m``.
  2. If short of the expected count for ``[session_open, last_closed_slot]``,
     pull from :mod:`fyers_client.history` and seed via
     :func:`data.realtime_feed.seed_bars`.

Boundary with the live feed: we only fetch *completed* slots — the
current in-progress slot is owned by the live aggregator. Cutoff is
``floor_to_5m(now) - 5min``, which gives a one-bar safety margin.
``INSERT OR IGNORE`` in ``seed_bars`` makes the whole thing idempotent
under any restart / replay ordering.

Fyers' history endpoint is per-symbol REST. The documented limits are
10 req/s + 200 req/min — the per-minute cap is the binding one for a
500-symbol watchlist, so we pace requests with a min-interval rate
limiter (:class:`_AsyncRateLimiter`) at 2.5 req/s and retry 429s with
exponential backoff. Other per-symbol failures (-300 invalid symbol,
network errors) are logged and skipped — the cold-start path in
:func:`bot.market_data.fetch_intraday` remains a yfinance-backed
safety net.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from data.trading_calendar import get_session_open, is_trading_day

log = logging.getLogger("alertbot.backfill")

IST = ZoneInfo("Asia/Kolkata")

BAR_WIDTH_MIN = 5
FYERS_RESOLUTION = "5"
DEFAULT_CONCURRENCY = 3
"""Max in-flight Fyers history calls. With per-request throttling (see
:data:`DEFAULT_RATE_PER_SEC`) concurrency only needs to be high enough
to keep the pipeline full while one HTTP roundtrip is outstanding;
3 is plenty and avoids piling up retries during 429 storms."""

DEFAULT_RATE_PER_SEC = 2.5
"""Fyers documented limits are 10 req/s and 200 req/min; the per-minute
cap (~3.33 req/s sustained) is the binding one. 2.5 req/s leaves
headroom so a single bot run won't drift into 429 cascades, even when
WebSocket reconnects retry the backfill."""

MAX_RETRIES = 5
"""Total attempts per symbol on a 429. With backoff 2s, 4s, 8s, 16s the
worst-case wait is ~30s before giving up — long enough to ride out a
brief rate-limit blip without stalling startup forever."""

RETRY_BASE_DELAY_S = 2.0


class _AsyncRateLimiter:
    """Async-safe minimum-interval gate between Fyers history calls.

    Not a token bucket — bursts past the configured rate aren't allowed,
    which matches Fyers' strict per-second enforcement. Each :meth:`acquire`
    blocks until at least ``1 / rate_per_sec`` seconds have elapsed since
    the previous successful acquire."""

    def __init__(self, rate_per_sec: float):
        self._min_interval = 1.0 / rate_per_sec
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = asyncio.get_event_loop().time()
            self._next_allowed = now + self._min_interval


def _floor_to_5m(ts: datetime) -> datetime:
    """Floor to the start of the current 5-min slot (session-aligned)."""
    minute = ts.minute - (ts.minute % BAR_WIDTH_MIN)
    return ts.replace(minute=minute, second=0, microsecond=0)


def _expected_bar_count(session_open: datetime, cutoff: datetime) -> int:
    """How many 5-min bars exist in ``[session_open, cutoff]`` inclusive
    on the open end and inclusive on the close end (both slot opens).

    ``cutoff < session_open`` returns 0 (pre-market call). The math is
    ``(cutoff - session_open) / 5min + 1`` for a slot-aligned cutoff.
    """
    if cutoff < session_open:
        return 0
    return int((cutoff - session_open).total_seconds() // (BAR_WIDTH_MIN * 60)) + 1


def _count_today_bars(symbol: str, session_open: datetime, cutoff: datetime) -> int:
    """How many bars are already in ``bars_5m`` for today's window.

    ``session_open`` and ``cutoff`` are tz-aware IST datetimes coming
    from :func:`_compute_window`. The on-disk ``ts`` column is UTC
    epoch milliseconds; convert via ``pandas.Timestamp.value`` (ns)."""
    # Local import: avoids circulars during module load and keeps the
    # aggregator's lazy-singleton init under the caller's control.
    from data.realtime_feed import get_aggregator

    agg = get_aggregator()
    start_ms = int(pd.Timestamp(session_open).value // 1_000_000)
    end_ms = int(pd.Timestamp(cutoff).value // 1_000_000)
    with sqlite3.connect(agg.db_path, timeout=5) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {agg.table_name} "
            "WHERE symbol = ? AND ts BETWEEN ? AND ?",
            (symbol, start_ms, end_ms),
        ).fetchone()
    return int(row[0]) if row else 0


def _compute_window(now: datetime) -> tuple[datetime, datetime] | None:
    """Resolve ``(session_open_today, last_closed_slot)`` in IST.

    Returns ``None`` when there's nothing to backfill: pre-market,
    weekend, or holiday. Post-market (after 15:30) still returns a
    window — a restart at 16:00 is the same problem as a restart at
    14:00, and the catch-up is harmless.
    """
    today = now.date()
    if not is_trading_day(today):
        return None
    session_open_ts = get_session_open(today)
    session_open = session_open_ts.to_pydatetime()
    if now < session_open:
        return None
    # Last *fully closed* slot. floor_to_5m(now) is the current
    # in-progress slot's open — subtract one bar to skip it.
    last_closed = _floor_to_5m(now) - timedelta(minutes=BAR_WIDTH_MIN)
    if last_closed < session_open:
        # Less than one bar since session open — nothing closed yet.
        return None
    return session_open, last_closed


async def _backfill_one(
    fy_symbol: str,
    session_open: datetime,
    cutoff: datetime,
    expected: int,
    sem: asyncio.Semaphore,
    limiter: _AsyncRateLimiter,
    *,
    client=None,
) -> int:
    """Backfill a single symbol. Returns the number of bars inserted
    (0 means no gap or the fetch failed — both are non-fatal).
    """
    have = _count_today_bars(fy_symbol, session_open, cutoff)
    if have >= expected:
        return 0
    log.debug(
        "Backfill %s: have %d/%d bars for today; fetching",
        fy_symbol, have, expected,
    )

    # Local import to avoid pulling the Fyers SDK at module load.
    from fyers_client import FyersHistoryError, fetch_history
    from data.realtime_feed import get_aggregator

    df = None
    async with sem:
        for attempt in range(MAX_RETRIES):
            await limiter.acquire()
            try:
                df = await asyncio.to_thread(
                    fetch_history,
                    fy_symbol, FYERS_RESOLUTION,
                    session_open.date(), cutoff.date(),
                    client=client,
                )
                break
            except FyersHistoryError as e:
                # 429 = rate-limited; retry with backoff. Anything else
                # (e.g. -300 invalid symbol) is terminal — fail fast.
                if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                    backoff = RETRY_BASE_DELAY_S * (2 ** attempt)
                    log.info(
                        "Rate limited on %s; retrying in %.1fs "
                        "(attempt %d/%d)",
                        fy_symbol, backoff, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(backoff)
                    continue
                log.warning("Fyers history failed for %s: %s", fy_symbol, e)
                return 0
            except Exception:
                log.exception("Backfill fetch raised for %s", fy_symbol)
                return 0
    if df is None:
        return 0

    if df.empty:
        return 0
    # Trim to the closed window — Fyers may return the in-progress slot
    # too (especially close to a 5-min boundary), and we don't want to
    # race the live aggregator on that one. ``cutoff`` is already tz-aware
    # IST, so wrap directly; passing ``tz=`` to Timestamp() on a tz-aware
    # datetime raises.
    cutoff_ts = pd.Timestamp(cutoff)
    df = df[df.index <= cutoff_ts]
    if df.empty:
        return 0

    try:
        inserted = await asyncio.to_thread(
            get_aggregator().seed_bars, fy_symbol, df,
        )
    except Exception:
        log.exception("seed_bars failed for %s", fy_symbol)
        return 0
    return inserted


async def backfill_today(
    fy_symbols: list[str],
    *,
    now: datetime | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_per_sec: float = DEFAULT_RATE_PER_SEC,
    client=None,
) -> dict[str, int]:
    """Catch ``bars_5m`` up for today across ``fy_symbols``.

    Args:
      fy_symbols: Watchlist in Fyers form (``"NSE:RELIANCE-EQ"`` …).
        That's the exact key the live aggregator writes and reads, so
        no symbol-form translation happens here.
      now: Defaults to current IST time. Overridable for tests.
      concurrency: Max in-flight Fyers history calls. See
        :data:`DEFAULT_CONCURRENCY`.
      rate_per_sec: Sustained per-request throttle. See
        :data:`DEFAULT_RATE_PER_SEC` for why the binding limit isn't
        the documented 10 req/s.
      client: Optional pre-built ``FyersModel`` for tests; production
        callers pass ``None`` and the cached daily token is used.

    Returns:
      ``{symbol: bars_inserted}`` for every symbol that had a gap.
      Empty dict on a pre-market / non-trading-day call. Symbols that
      already had all expected bars are *not* in the result — absence
      = no work needed.
    """
    if not fy_symbols:
        return {}
    now = now or datetime.now(IST)
    window = _compute_window(now)
    if window is None:
        log.info("Backfill skipped: now=%s outside trading window", now)
        return {}
    session_open, cutoff = window
    expected = _expected_bar_count(session_open, cutoff)
    if expected <= 0:
        return {}

    log.info(
        "Backfill: filling up to %d bars per symbol for %d symbols "
        "(window %s → %s)",
        expected, len(fy_symbols),
        session_open.strftime("%H:%M"), cutoff.strftime("%H:%M"),
    )

    sem = asyncio.Semaphore(max(1, concurrency))
    limiter = _AsyncRateLimiter(rate_per_sec)
    tasks = [
        _backfill_one(
            sym, session_open, cutoff, expected, sem, limiter, client=client,
        )
        for sym in fy_symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    filled = {sym: n for sym, n in zip(fy_symbols, results) if n > 0}
    log.info(
        "Backfill done: filled %d/%d symbols (%d bars total)",
        len(filled), len(fy_symbols), sum(filled.values()),
    )
    return filled
