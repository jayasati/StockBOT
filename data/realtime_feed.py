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

Bars are persisted to ``bars_5m`` in alerts.db keyed on (symbol, ts_open).
On reconnect or process restart, completed bars survive — only the
in-progress bar is lost. ``get_5m_bars`` falls back to SQLite when the
in-memory cache is cold.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger("alertbot.realtime_feed")

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_LAST_BAR_OPEN = time(15, 25)  # last valid 5-min slot starts at 15:25

DEFAULT_DB_PATH = Path("alerts.db")
MAX_BARS_PER_SYMBOL = 200
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_5m (
    symbol  TEXT NOT NULL,
    ts_open TEXT NOT NULL,   -- ISO-8601 IST, tz-aware
    o       REAL NOT NULL,
    h       REAL NOT NULL,
    l       REAL NOT NULL,
    c       REAL NOT NULL,
    v       REAL NOT NULL,
    PRIMARY KEY (symbol, ts_open)
);
"""


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


def _floor_to_5m(ts_ist: pd.Timestamp) -> pd.Timestamp:
    """Floor a tz-aware IST timestamp to its 5-min slot start.

    Standard ``minute - (minute % 5)`` aligns naturally to 09:15 since
    15 % 5 == 0; the test suite covers the boundary case explicitly.
    """
    minute = ts_ist.minute - (ts_ist.minute % 5)
    return ts_ist.replace(minute=minute, second=0, microsecond=0)


def _is_in_session(slot_ts: pd.Timestamp) -> bool:
    t = slot_ts.time()
    if t < SESSION_OPEN or t > SESSION_LAST_BAR_OPEN:
        return False
    return slot_ts.weekday() < 5


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
    ) -> None:
        self.db_path = Path(db_path)
        self.max_bars = max_bars
        self.on_bar_complete = on_bar_complete
        self._lock = threading.Lock()
        self._completed: dict[str, deque[Bar]] = {}
        self._current: dict[str, Bar] = {}
        self._latest_tick_ts: dict[str, datetime] = {}
        self._tick_count = 0
        self._dropped_out_of_session = 0
        self.init_db()

    # -- DB --------------------------------------------------------------

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def _persist(self, bar: Bar) -> None:
        # INSERT OR IGNORE makes the reconnect-replay path idempotent: if a
        # crash interleaves with persistence, the second attempt is a no-op.
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO bars_5m "
                    "(symbol, ts_open, o, h, l, c, v) VALUES (?,?,?,?,?,?,?)",
                    (
                        bar.symbol,
                        bar.ts_open.isoformat(),
                        bar.open, bar.high, bar.low, bar.close, bar.volume,
                    ),
                )
        except sqlite3.Error as e:
            log.error("Failed to persist bar %s @ %s: %s", bar.symbol, bar.ts_open, e)

    # -- Tick ingestion --------------------------------------------------

    def on_tick(self, msg: dict) -> None:
        """Process a single Fyers tick. Lifecycle frames are ignored."""
        if msg.get("type") != "sf":
            return
        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        ltq = msg.get("ltq") or msg.get("last_traded_qty") or 0
        ts_epoch = msg.get("last_traded_time")
        if not symbol or ltp is None or ts_epoch is None:
            return

        ltp = float(ltp)
        ltq = float(ltq)
        ts_ist = _epoch_to_ist(float(ts_epoch))
        slot = _floor_to_5m(ts_ist)

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
                    self._completed.setdefault(
                        symbol, deque(maxlen=self.max_bars)
                    ).append(cur)
                    emitted = cur
                self._current[symbol] = Bar(
                    symbol=symbol, ts_open=slot,
                    open=ltp, high=ltp, low=ltp, close=ltp, volume=ltq,
                )
            else:
                cur.high = max(cur.high, ltp)
                cur.low = min(cur.low, ltp)
                cur.close = ltp
                cur.volume += ltq

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
                ts_open=pd.Timestamp(r[0]).tz_convert(IST),
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
                    "SELECT ts_open, o, h, l, c, v FROM bars_5m "
                    "WHERE symbol = ? ORDER BY ts_open DESC LIMIT ?",
                    (symbol, n),
                )
            else:
                cur = conn.execute(
                    "SELECT ts_open, o, h, l, c, v FROM bars_5m "
                    "WHERE symbol = ? AND ts_open < ? "
                    "ORDER BY ts_open DESC LIMIT ?",
                    (symbol, before_ts.isoformat(), n),
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
        """Insert yfinance 5-min bars into bars_5m. Idempotent on (symbol, ts_open).

        yfinance gives tz-aware UTC timestamps with capitalised columns; we
        normalise to IST + lowercase before persisting. Bars outside the
        09:15–15:30 IST window are dropped.
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
                symbol, ts.isoformat(),
                float(row.open), float(row.high), float(row.low),
                float(row.close), float(row.volume),
            ))
        if not rows:
            return 0
        with sqlite3.connect(self.db_path, timeout=5) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO bars_5m "
                "(symbol, ts_open, o, h, l, c, v) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_AGG: BarAggregator | None = None
_LIVE_FEED = None
_SUBSCRIBED: set[str] = set()
_LOCK = threading.Lock()


def get_aggregator() -> BarAggregator:
    """Lazy singleton. Always touch the aggregator through this — never
    construct your own unless you're a test passing a temp DB path."""
    global _AGG
    if _AGG is None:
        with _LOCK:
            if _AGG is None:
                _AGG = BarAggregator()
    return _AGG


def set_aggregator(agg: BarAggregator) -> None:
    """Test seam — replace the singleton. Production code should not call this."""
    global _AGG
    with _LOCK:
        _AGG = agg


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
    agg = get_aggregator()
    # Local import to avoid pulling in fyers SDK at module load time
    # (tests instantiate BarAggregator without ever needing the SDK).
    from fyers_client import LiveFeed

    with _LOCK:
        if _LIVE_FEED is None:
            _LIVE_FEED = LiveFeed(symbols=new, on_tick=agg.on_tick)
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


def get_5m_bars(symbol: str, n: int = 100) -> pd.DataFrame:
    return get_aggregator().get_5m_bars(symbol, n)


def get_current_partial(symbol: str) -> Bar | None:
    return get_aggregator().get_current_partial(symbol)


def seed_from_yfinance(symbol: str, df_5m: pd.DataFrame) -> int:
    return get_aggregator().seed_from_yfinance(symbol, df_5m)


def stats() -> dict:
    return get_aggregator().stats()
