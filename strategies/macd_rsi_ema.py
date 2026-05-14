"""MACD + RSI + EMA — momentum-confirmation strategy.

Read: This is a momentum-triple-confirm setup designed for the first
~90 minutes after the cash session open, when trending stocks set up
their day-direction move. We require THREE independent signals to line
up before taking a trade — so it rarely fires, but when it does the
read is clean.

Long entry — all of:
  - MACD histogram flips from <= 0 to > 0   (momentum turning up)
  - RSI crosses above 50                    (trend-direction confirm)
  - Close > EMA(trend_period)               (price-above-trend confirm)
  - RSI not overextended (< rsi_high)       (no fade-risk at top)

Short entry — mirror image.

Time gate: entries only between session_open and ``minutes_after_open``
(default 90 → 09:15-10:45 IST). Outside that window, signal() returns
None regardless of indicator state. The backtest engine will still
flatten at 15:15 IST.

Exit (handled inside ``signal``, returned as a SignalKind.EXIT so the
engine sees it):
  - LONG: histogram crosses back from > 0 to <= 0 (MACD reversal).
  - SHORT: histogram crosses back from < 0 to >= 0.

The "RSI divergence" exit in the original spec is intentionally NOT
implemented — proper bullish/bearish divergence detection needs swing-
point identification and is genuinely noisy at 5-min granularity.
We use MACD reversal alone, which is the more reliable signal.

Reuses ``indicators.momentum.macd``, ``indicators.momentum.rsi``,
``indicators.trend.ema``."""
from __future__ import annotations

from datetime import time

import pandas as pd

from indicators import momentum, trend

from .base import Signal, SignalKind, Strategy

SESSION_OPEN = time(9, 15)


class MacdRsiEma(Strategy):
    name = "macd_rsi_ema"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_mid: float = 50.0,
        rsi_low: float = 25.0,
        rsi_high: float = 75.0,
        ema_period: int = 20,
        minutes_after_open: int = 90,
    ) -> None:
        if macd_slow <= macd_fast:
            raise ValueError(
                f"macd_slow ({macd_slow}) must be > macd_fast ({macd_fast})"
            )
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if not (0 < rsi_low < rsi_mid < rsi_high < 100):
            raise ValueError(
                f"need 0 < rsi_low ({rsi_low}) < rsi_mid ({rsi_mid}) "
                f"< rsi_high ({rsi_high}) < 100"
            )
        if ema_period < 2:
            raise ValueError(f"ema_period must be >= 2, got {ema_period}")
        if minutes_after_open < 5:
            raise ValueError(
                f"minutes_after_open must be >= 5, got {minutes_after_open}"
            )
        super().__init__(
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            rsi_period=rsi_period, rsi_mid=rsi_mid,
            rsi_low=rsi_low, rsi_high=rsi_high,
            ema_period=ema_period, minutes_after_open=minutes_after_open,
        )
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_mid = rsi_mid
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.ema_period = ema_period
        self.minutes_after_open = minutes_after_open

        # Per-DataFrame indicator caches.
        self._cache_key: tuple[int, int] | None = None
        self._hist: pd.Series | None = None
        self._rsi: pd.Series | None = None
        self._ema: pd.Series | None = None
        # Position tracking — strategy emits its own EXIT signals.
        self._position: str = "FLAT"

    def _ensure(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._hist is not None:
            return
        macd_df = momentum.macd(
            df, fast=self.macd_fast, slow=self.macd_slow,
            signal=self.macd_signal,
        )
        self._hist = macd_df["histogram"]
        self._rsi = momentum.rsi(df, period=self.rsi_period)
        self._ema = trend.ema(df, period=self.ema_period)
        self._cache_key = key
        self._position = "FLAT"

    def _in_window(self, ts: pd.Timestamp) -> bool:
        """True if ``ts`` is within the first ``minutes_after_open`` of the
        cash session (using the bar's local hh:mm). Falls back to True if
        the index isn't datetime-like."""
        if not isinstance(ts, pd.Timestamp):
            return True
        t = ts.time()
        if t < SESSION_OPEN:
            return False
        elapsed_min = (
            (t.hour - SESSION_OPEN.hour) * 60
            + (t.minute - SESSION_OPEN.minute)
        )
        return elapsed_min < self.minutes_after_open

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure(df)
        h_prev = self._hist.iat[i - 1]
        h_curr = self._hist.iat[i]
        r_prev = self._rsi.iat[i - 1]
        r_curr = self._rsi.iat[i]
        e_curr = self._ema.iat[i]
        c_curr = float(df["close"].iat[i])
        if (pd.isna(h_prev) or pd.isna(h_curr) or pd.isna(r_prev)
                or pd.isna(r_curr) or pd.isna(e_curr)):
            return None

        # ---- Exits first so we don't re-enter the same bar we exit ----
        if self._position == "LONG" and h_prev > 0 and h_curr <= 0:
            self._position = "FLAT"
            return Signal(SignalKind.EXIT, "MacdRevL")
        if self._position == "SHORT" and h_prev < 0 and h_curr >= 0:
            self._position = "FLAT"
            return Signal(SignalKind.EXIT, "MacdRevS")

        # ---- Entries — only inside the open-window time gate ----
        if self._position != "FLAT":
            return None
        if not self._in_window(df.index[i]):
            return None

        hist_up = h_prev <= 0 and h_curr > 0
        hist_dn = h_prev >= 0 and h_curr < 0
        rsi_up = r_prev <= self.rsi_mid and r_curr > self.rsi_mid
        rsi_dn = r_prev >= self.rsi_mid and r_curr < self.rsi_mid
        not_overbought = r_curr < self.rsi_high
        not_oversold = r_curr > self.rsi_low

        if hist_up and rsi_up and c_curr > e_curr and not_overbought:
            self._position = "LONG"
            return Signal(SignalKind.ENTER_LONG, "MacdRsiEmaLE")
        if hist_dn and rsi_dn and c_curr < e_curr and not_oversold:
            self._position = "SHORT"
            return Signal(SignalKind.ENTER_SHORT, "MacdRsiEmaSE")
        return None
