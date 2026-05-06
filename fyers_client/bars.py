"""Tick → 5-minute OHLCV bar aggregator (legacy in-memory store).

Used by the ``python -m fyers_client bars`` CLI verifier. The production
aggregator that bot/market_data.py reads from lives in data/realtime_feed.py."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

BAR_INTERVAL_SECONDS = 300  # 5 minutes
MAX_BARS_PER_SYMBOL = 100


@dataclass
class Bar:
    ts: datetime          # bar start time, IST naive (matching yfinance shape)
    open: float
    high: float
    low: float
    close: float
    start_vol: float
    end_vol: float

    @property
    def volume(self) -> float:
        return max(0.0, self.end_vol - self.start_vol)


def _floor_to_bar(ts: datetime) -> datetime:
    minute = ts.minute - (ts.minute % 5)
    return ts.replace(minute=minute, second=0, microsecond=0)


def _epoch_to_ist(epoch_seconds: int | float) -> datetime:
    """Fyers last_traded_time is epoch seconds (UTC)."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .astimezone(IST)
        .replace(tzinfo=None)
    )


class TickStore:
    """Thread-safe latest-price + rolling-bar store for the CLI verifier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_price: dict[str, float] = {}
        self._latest_ts: dict[str, datetime] = {}
        self._bars: dict[str, deque[Bar]] = {}
        self._current_bar: dict[str, Bar] = {}
        self._tick_count = 0

    def on_tick(self, msg: dict) -> None:
        if msg.get("type") != "sf":
            return
        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        ts_epoch = msg.get("last_traded_time")
        vol = float(msg.get("vol_traded_today") or 0)
        if not symbol or ltp is None or ts_epoch is None:
            return
        ltp = float(ltp)
        ts = _epoch_to_ist(ts_epoch)
        bar_ts = _floor_to_bar(ts)

        with self._lock:
            self._tick_count += 1
            self._latest_price[symbol] = ltp
            self._latest_ts[symbol] = ts
            cur = self._current_bar.get(symbol)
            if cur is None or cur.ts != bar_ts:
                if cur is not None:
                    cur.end_vol = vol
                    self._bars.setdefault(
                        symbol, deque(maxlen=MAX_BARS_PER_SYMBOL)
                    ).append(cur)
                self._current_bar[symbol] = Bar(
                    ts=bar_ts, open=ltp, high=ltp, low=ltp, close=ltp,
                    start_vol=vol, end_vol=vol,
                )
            else:
                cur.high = max(cur.high, ltp)
                cur.low = min(cur.low, ltp)
                cur.close = ltp
                cur.end_vol = vol

    def get_bars_df(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            archived = list(self._bars.get(symbol, ()))
            current = self._current_bar.get(symbol)
        all_bars: list[Bar] = archived + ([current] if current is not None else [])
        if not all_bars:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([b.ts for b in all_bars], name="Datetime")
        return pd.DataFrame(
            {
                "Open":   [b.open for b in all_bars],
                "High":   [b.high for b in all_bars],
                "Low":    [b.low for b in all_bars],
                "Close":  [b.close for b in all_bars],
                "Volume": [b.volume for b in all_bars],
            },
            index=idx,
        )

    def latest_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._latest_price.get(symbol)

    def stats(self) -> dict:
        with self._lock:
            return {
                "ticks": self._tick_count,
                "symbols_seen": len(self._latest_price),
                "open_bars":  len(self._current_bar),
                "archived_bars": sum(len(b) for b in self._bars.values()),
            }


TICK_STORE = TickStore()
