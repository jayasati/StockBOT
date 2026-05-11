"""
Public realtime feed API.

Consumers (bot.py, dashboard, paper tracker) call:
    subscribe(symbols)                     — once at startup, opens the LiveFeed
    get_5m_bars(symbol, n=100) -> DataFrame — last n completed 5-min OHLCV bars
    get_current_partial(symbol) -> Bar     — the in-progress bar
    seed_from_yfinance(symbol, df_5m)      — backfill the SQLite cache from yfinance

Bars are aggregated by ``BarAggregator`` from the raw Fyers tick stream.
Slots are aligned to the NSE cash-equity session (09:15–15:30 IST). The first
bar timestamp of the day is 09:15; the last is 15:25. Ticks outside that
window are silently dropped — features.py's session VWAP and opening range
both depend on that invariant.

Returned DataFrames have:
  * tz-aware ``DatetimeIndex`` in IST
  * lowercase columns: ``open``, ``high``, ``low``, ``close``, ``volume``

This matches features.py exactly so the same DataFrame is consumed by both.

Bars are persisted to ``bars_5m`` in alerts.db keyed on (symbol, ts) where
``ts`` is the bar's open time as UTC epoch milliseconds. On reconnect or
process restart, completed bars survive — only the in-progress bar is
lost. ``get_5m_bars`` falls back to SQLite when the in-memory cache is
cold.

Per-bar volume is computed from Fyers' ``vol_traded_today`` cumulative
day-volume field (delta between the first tick observed in the bar and
the first tick observed in the next bar). The naive alternative — summing
``ltq`` per snapshot — silently under-counts by ~5–10× because the
``SymbolUpdate`` channel publishes throttled snapshots, not every print,
and only the most recent trade's quantity survives in ``ltq``.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from bot.schedule import (  # re-exported for legacy callers
    SESSION_LAST_BAR_OPEN,
    SESSION_OPEN,
    in_session,
)

log = logging.getLogger("alertbot.realtime_feed")

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_DB_PATH = Path("alerts.db")
MAX_BARS_PER_SYMBOL = 200
# Template is interpolated with the table name at init time. SQLite doesn't
# bind table names as parameters; the name is a class-controlled constant
# (never user input), so f-string interpolation is safe here.
SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    symbol  TEXT    NOT NULL,
    ts      INTEGER NOT NULL,   -- UTC epoch milliseconds (bar open time)
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  REAL    NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""


def _ts_open_to_epoch_ms(ts_open: pd.Timestamp) -> int:
    """Bar open time → UTC epoch milliseconds (the canonical on-disk form)."""
    return int(ts_open.value // 1_000_000)


# ---------------------------------------------------------------------------
# Bar dataclass + helpers
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    symbol: str
    ts_open: pd.Timestamp   # tz-aware IST
    open: float
    high: float
    low: float
    close: float
    volume: float


def _floor_to_slot(ts_ist: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """Floor a tz-aware IST timestamp to the start of its ``minutes``-wide
    bar slot. Both 1 and 5 align cleanly to the 09:15 session open."""
    minute = ts_ist.minute - (ts_ist.minute % minutes)
    return ts_ist.replace(minute=minute, second=0, microsecond=0)


def _floor_to_5m(ts_ist: pd.Timestamp) -> pd.Timestamp:
    """Floor to a 5-min slot. Kept for backward compatibility with callers
    (and tests) that import this symbol directly."""
    return _floor_to_slot(ts_ist, 5)


def _is_in_session(slot_ts: pd.Timestamp) -> bool:
    return in_session(slot_ts, mode="bar_slot")


def _epoch_to_ist(epoch_seconds: float) -> pd.Timestamp:
    return pd.Timestamp(epoch_seconds, unit="s", tz=timezone.utc).tz_convert(IST)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class BarAggregator:
    """Tick → 5-min OHLCV aggregator with SQLite persistence.

    Thread-safe: the Fyers WebSocket runs on a daemon thread that calls
    ``on_tick`` while consumers (the async scan loop) call ``get_5m_bars``
    from the main thread. A single mutex around all mutable state is enough
    — the work under the lock is O(1) per tick and contention is negligible
    even at 500 symbols × multiple ticks/sec.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        max_bars: int = MAX_BARS_PER_SYMBOL,
        on_bar_complete: Callable[[Bar], None] | None = None,
        *,
        bar_width_minutes: int = 5,
        table_name: str = "bars_5m",
    ) -> None:
        if bar_width_minutes not in (1, 5):
            # The session-aligned floors only work for divisors of 5
            # starting at minute 15 (the session open). 1 and 5 are the
            # only widths we use today; widen this gate if you add 15m.
            raise ValueError(f"unsupported bar width: {bar_width_minutes}m")
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"unsafe table name: {table_name!r}")
        self.db_path = Path(db_path)
        self.max_bars = max_bars
        self.on_bar_complete = on_bar_complete
        self.bar_width_minutes = bar_width_minutes
        self.table_name = table_name
        self._lock = threading.Lock()
        self._completed: dict[str, deque[Bar]] = {}
        self._current: dict[str, Bar] = {}
        self._bar_start_vol: dict[str, float] = {}
        self._latest_tick_ts: dict[str, datetime] = {}
        self._tick_count = 0
        self._dropped_out_of_session = 0
        self.init_db()

    # -- DB --------------------------------------------------------------

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_TEMPLATE.format(table=self.table_name))

    def _persist(self, bar: Bar) -> None:
        # INSERT OR IGNORE makes the reconnect-replay path idempotent: if a
        # crash interleaves with persistence, the second attempt is a no-op.
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    f"INSERT OR IGNORE INTO {self.table_name} "
                    "(symbol, ts, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        bar.symbol,
                        _ts_open_to_epoch_ms(bar.ts_open),
                        bar.open, bar.high, bar.low, bar.close, bar.volume,
                    ),
                )
        except sqlite3.Error as e:
            log.error("Failed to persist bar %s @ %s: %s", bar.symbol, bar.ts_open, e)

    # -- Tick ingestion --------------------------------------------------

    def on_tick(self, msg: dict) -> None:
        """Process a single Fyers tick. Lifecycle frames are ignored.

        Per-bar volume is the delta of ``vol_traded_today`` between the first
        tick of this bar and the first tick of the next bar. ``ltq`` is not
        used: it captures only the most recent trade, while snapshots are
        throttled, so summing ltq under-counts by ~5–10×.
        """
        if msg.get("type") != "sf":
            return
        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        ts_epoch = msg.get("last_traded_time")
        if not symbol or ltp is None or ts_epoch is None:
            return

        ltp = float(ltp)
        vol_today = float(msg.get("vol_traded_today") or 0.0)
        ts_ist = _epoch_to_ist(float(ts_epoch))
        slot = _floor_to_slot(ts_ist, self.bar_width_minutes)

        if not _is_in_session(slot):
            with self._lock:
                self._dropped_out_of_session += 1
            return

        emitted: Bar | None = None
        with self._lock:
            self._tick_count += 1
            self._latest_tick_ts[symbol] = ts_ist.to_pydatetime()
            cur = self._current.get(symbol)
            if cur is None or cur.ts_open != slot:
                if cur is not None:
                    # Close the previous bar. Its volume is the delta from
                    # its start_vol to *this* tick's vol_traded_today (the
                    # first tick of the new slot) — that captures every
                    # trade printed in between.
                    prev_start = self._bar_start_vol.get(symbol, vol_today)
                    cur.volume = max(0.0, vol_today - prev_start)
                    self._completed.setdefault(
                        symbol, deque(maxlen=self.max_bars)
                    ).append(cur)
                    emitted = cur
                self._current[symbol] = Bar(
                    symbol=symbol, ts_open=slot,
                    open=ltp, high=ltp, low=ltp, close=ltp, volume=0.0,
                )
                self._bar_start_vol[symbol] = vol_today
            else:
                cur.high = max(cur.high, ltp)
                cur.low = min(cur.low, ltp)
                cur.close = ltp
                # Clamp guards against a day rollover (vol_today resets to 0)
                # or an out-of-order snapshot. Without the clamp a transient
                # negative would corrupt the in-progress bar.
                cur.volume = max(0.0, vol_today - self._bar_start_vol[symbol])

        if emitted is not None:
            self._persist(emitted)
            if self.on_bar_complete is not None:
                try:
                    self.on_bar_complete(emitted)
                except Exception:
                    log.exception("on_bar_complete callback failed")

    def flush(self, symbol: str | None = None) -> list[Bar]:
        """Close any in-progress bar(s) and persist. Used at session close
        and at the end of a replay test. Returns the bars that were flushed."""
        flushed: list[Bar] = []
        with self._lock:
            symbols = [symbol] if symbol else list(self._current.keys())
            for sym in symbols:
                cur = self._current.pop(sym, None)
                if cur is None:
                    continue
                self._completed.setdefault(
                    sym, deque(maxlen=self.max_bars)
                ).append(cur)
                flushed.append(cur)
        for bar in flushed:
            self._persist(bar)
            if self.on_bar_complete is not None:
                try:
                    self.on_bar_complete(bar)
                except Exception:
                    log.exception("on_bar_complete callback failed")
        return flushed

    # -- Reads -----------------------------------------------------------

    def _bars_to_df(self, bars: list[Bar]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz=IST, name="ts"),
            )
        idx = pd.DatetimeIndex([b.ts_open for b in bars], name="ts")
        if idx.tz is None:
            idx = idx.tz_localize(IST)
        else:
            idx = idx.tz_convert(IST)
        return pd.DataFrame(
            {
                "open":   [b.open   for b in bars],
                "high":   [b.high   for b in bars],
                "low":    [b.low    for b in bars],
                "close":  [b.close  for b in bars],
                "volume": [b.volume for b in bars],
            },
            index=idx,
        )

    def get_5m_bars(self, symbol: str, n: int = 100) -> pd.DataFrame:
        """Return last ``n`` completed 5-min bars for ``symbol``.

        In-memory cache first; falls back to SQLite when the cache hasn't
        warmed up yet (cold start, fresh process).
        """
        with self._lock:
            cached = list(self._completed.get(symbol, ()))
        if len(cached) >= n:
            return self._bars_to_df(cached[-n:])

        # Cold cache — pull from SQLite. The cache has the most recent bars
        # so we ask SQLite for everything older, then concatenate.
        oldest_in_cache = cached[0].ts_open if cached else None
        rows = self._read_db(symbol, n, oldest_in_cache)
        db_bars = [
            Bar(
                symbol=symbol,
                ts_open=pd.Timestamp(r[0], unit="ms", tz="UTC").tz_convert(IST),
                open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            )
            for r in rows
        ]
        merged = db_bars + cached
        if len(merged) > n:
            merged = merged[-n:]
        return self._bars_to_df(merged)

    def _read_db(
        self, symbol: str, n: int, before_ts: pd.Timestamp | None
    ) -> list[tuple]:
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            if before_ts is None:
                cur = conn.execute(
                    f"SELECT ts, open, high, low, close, volume FROM {self.table_name} "
                    "WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                    (symbol, n),
                )
            else:
                cur = conn.execute(
                    f"SELECT ts, open, high, low, close, volume FROM {self.table_name} "
                    "WHERE symbol = ? AND ts < ? "
                    "ORDER BY ts DESC LIMIT ?",
                    (symbol, _ts_open_to_epoch_ms(before_ts), n),
                )
            rows = list(cur.fetchall())
        rows.reverse()  # we want ascending
        return rows

    def get_current_partial(self, symbol: str) -> Bar | None:
        with self._lock:
            cur = self._current.get(symbol)
            if cur is None:
                return None
            return Bar(**cur.__dict__)

    def latest_tick_ts(self, symbol: str) -> datetime | None:
        with self._lock:
            return self._latest_tick_ts.get(symbol)

    def latest_overall_tick_ts(
        self, symbols: list[str] | None = None
    ) -> tuple[str | None, datetime | None]:
        """Return ``(symbol, ts)`` of the most recent tick across the given
        symbols (or all subscribed symbols if ``None``), or ``(None, None)``
        if nothing has been seen yet. Used by the ``ticks_fresh`` health
        check — one lock acquisition, O(n) over the watchlist."""
        with self._lock:
            if not self._latest_tick_ts:
                return None, None
            if symbols is None:
                sym, ts = max(self._latest_tick_ts.items(), key=lambda kv: kv[1])
                return sym, ts
            best_sym: str | None = None
            best_ts: datetime | None = None
            for s in symbols:
                ts = self._latest_tick_ts.get(s)
                if ts is None:
                    continue
                if best_ts is None or ts > best_ts:
                    best_sym, best_ts = s, ts
            return best_sym, best_ts

    def stats(self) -> dict:
        with self._lock:
            return {
                "ticks": self._tick_count,
                "out_of_session_dropped": self._dropped_out_of_session,
                "open_bars":      len(self._current),
                "completed_bars": sum(len(b) for b in self._completed.values()),
                "symbols_seen":   len(self._latest_tick_ts),
            }

    # -- Backfill --------------------------------------------------------

    def seed_from_yfinance(self, symbol: str, df_5m: pd.DataFrame) -> int:
        """Insert yfinance 5-min bars into bars_5m. Idempotent on (symbol, ts).

        yfinance gives tz-aware UTC timestamps with capitalised columns; we
        normalise to IST + lowercase before persisting, and store ``ts`` as
        UTC epoch milliseconds. Bars outside the 09:15–15:30 IST window are
        dropped.
        """
        if df_5m is None or df_5m.empty:
            return 0
        df = df_5m.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return 0
        if idx.tz is None:
            idx = idx.tz_localize(IST)
        else:
            idx = idx.tz_convert(IST)

        rows: list[tuple] = []
        for ts, row in zip(idx, df.itertuples(index=False)):
            if not _is_in_session(ts):
                continue
            rows.append((
                symbol, _ts_open_to_epoch_ms(ts),
                float(row.open), float(row.high), float(row.low),
                float(row.close), float(row.volume),
            ))
        if not rows:
            return 0
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            conn.executemany(
                f"INSERT OR IGNORE INTO {self.table_name} "
                "(symbol, ts, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_AGG: BarAggregator | None = None         # 5-min aggregator (primary)
_AGG_1M: BarAggregator | None = None      # 1-min aggregator (parallel)
_LIVE_FEED = None
_SUBSCRIBED: set[str] = set()
_TICK_OBSERVERS: list[Callable[[dict], None]] = []
_LOCK = threading.Lock()


def get_aggregator() -> BarAggregator:
    """Lazy singleton for the 5-min aggregator. Always touch the aggregator
    through this — never construct your own unless you're a test passing a
    temp DB path."""
    global _AGG
    if _AGG is None:
        with _LOCK:
            if _AGG is None:
                _AGG = BarAggregator(bar_width_minutes=5, table_name="bars_5m")
    return _AGG


def get_aggregator_1m() -> BarAggregator:
    """Lazy singleton for the 1-min aggregator. Runs alongside the 5-min
    one — every tick is processed by both. Provides ``bars_1m`` for
    future scoring paths that need sub-bar latency."""
    global _AGG_1M
    if _AGG_1M is None:
        with _LOCK:
            if _AGG_1M is None:
                _AGG_1M = BarAggregator(bar_width_minutes=1, table_name="bars_1m")
    return _AGG_1M


def set_aggregator(agg: BarAggregator) -> None:
    """Test seam — replace the singleton. Production code should not call this."""
    global _AGG
    with _LOCK:
        _AGG = agg


def add_tick_observer(fn: Callable[[dict], None]) -> None:
    """Register an extra synchronous tick consumer. The observer is called
    after the bar aggregators have processed the tick. Use this for
    side-channel detectors (fast-mover, custom metrics) that need raw
    ticks without owning the WebSocket lifecycle.

    Observers are invoked on the WebSocket thread; do not block. Bridge
    to asyncio via ``loop.call_soon_threadsafe`` if you need async work."""
    with _LOCK:
        _TICK_OBSERVERS.append(fn)


def _dispatch_tick(msg: dict) -> None:
    """Fan-out: 5-min aggregator, 1-min aggregator, then registered observers.

    Each consumer is independent — a failure in one must not prevent the
    others from seeing the tick."""
    try:
        get_aggregator().on_tick(msg)
    except Exception:
        log.exception("5m aggregator on_tick raised")
    try:
        get_aggregator_1m().on_tick(msg)
    except Exception:
        log.exception("1m aggregator on_tick raised")
    # Snapshot the observer list under the lock so a concurrent
    # add_tick_observer doesn't perturb iteration.
    with _LOCK:
        observers = list(_TICK_OBSERVERS)
    for fn in observers:
        try:
            fn(msg)
        except Exception:
            log.exception("tick observer %r raised", fn)


def subscribe(symbols: list[str]) -> None:
    """Open the live feed (if not already) and subscribe to ``symbols``.

    Idempotent: re-calling with the same symbols is a no-op. Calling with
    new symbols extends the subscription.
    """
    global _LIVE_FEED
    if not symbols:
        return
    new = [s for s in symbols if s not in _SUBSCRIBED]
    if not new:
        return
    # Eager-init both aggregators so the first tick has no construction lag.
    get_aggregator()
    get_aggregator_1m()
    # Local import to avoid pulling in fyers SDK at module load time
    # (tests instantiate BarAggregator without ever needing the SDK).
    from fyers_client import LiveFeed

    with _LOCK:
        if _LIVE_FEED is None:
            _LIVE_FEED = LiveFeed(symbols=new, on_tick=_dispatch_tick)
            _LIVE_FEED.start()
        else:
            _LIVE_FEED.add_symbols(new)
        _SUBSCRIBED.update(new)


def stop() -> None:
    """Close the live feed and flush any in-progress bars."""
    global _LIVE_FEED
    with _LOCK:
        if _LIVE_FEED is not None:
            try:
                _LIVE_FEED.stop()
            except Exception:
                log.exception("LiveFeed.stop() failed")
            _LIVE_FEED = None
    if _AGG is not None:
        _AGG.flush()
    if _AGG_1M is not None:
        _AGG_1M.flush()


def get_5m_bars(symbol: str, n: int = 100) -> pd.DataFrame:
    return get_aggregator().get_5m_bars(symbol, n)


def get_1m_bars(symbol: str, n: int = 100) -> pd.DataFrame:
    """Most recent ``n`` completed 1-min bars for ``symbol``. Same OHLCV
    contract as ``get_5m_bars``. Cold-cache fallback to SQLite — but no
    yfinance backfill: 1-min bars accumulate from process start only."""
    return get_aggregator_1m().get_5m_bars(symbol, n)


def get_current_partial(symbol: str) -> Bar | None:
    return get_aggregator().get_current_partial(symbol)


def seed_from_yfinance(symbol: str, df_5m: pd.DataFrame) -> int:
    return get_aggregator().seed_from_yfinance(symbol, df_5m)


def stats() -> dict:
    return get_aggregator().stats()


def is_live_feed_connected() -> bool:
    """True if a LiveFeed has been started and reports its websocket up.

    Returns False if no feed has been subscribed yet, the feed has been
    stopped, or the underlying websocket is between connection attempts."""
    if _LIVE_FEED is None:
        return False
    try:
        return _LIVE_FEED.is_connected()
    except Exception:
        return False
