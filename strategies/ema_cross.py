"""EMA cross strategy — classic fast/slow exponential MA trend-follow.

Pine reference (v5)::

    fast = ta.ema(close, fast_len)
    slow = ta.ema(close, slow_len)
    if (ta.crossover(fast, slow))   strategy.entry("Long",  strategy.long)
    if (ta.crossunder(fast, slow))  strategy.entry("Short", strategy.short)

Trend-follow read: LONG when the fast EMA crosses ABOVE the slow EMA
(bullish trend onset); SHORT when it crosses BELOW (bearish trend onset).
Defaults are 9/21 (popular intraday combo). Reuses ``indicators.trend.ema``
which matches Pine's ``ta.ema`` (alpha = 2/(period+1), SMA-seeded)."""
from __future__ import annotations

import pandas as pd

from indicators import trend

from .base import Signal, SignalKind, Strategy


class EMACross(Strategy):
    name = "ema_cross"

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        if fast < 2:
            raise ValueError(f"fast must be >= 2, got {fast}")
        if slow <= fast:
            raise ValueError(
                f"slow ({slow}) must be > fast ({fast}); the strategy "
                f"is meaningless when fast >= slow"
            )
        super().__init__(fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow
        self._cache_key: tuple[int, int] | None = None
        self._fast_ema: pd.Series | None = None
        self._slow_ema: pd.Series | None = None

    def _ensure_emas(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._fast_ema is not None:
            return
        self._fast_ema = trend.ema(df, period=self.fast)
        self._slow_ema = trend.ema(df, period=self.slow)
        self._cache_key = key

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure_emas(df)
        f_prev = self._fast_ema.iat[i - 1]
        f_curr = self._fast_ema.iat[i]
        s_prev = self._slow_ema.iat[i - 1]
        s_curr = self._slow_ema.iat[i]
        if (pd.isna(f_prev) or pd.isna(f_curr)
                or pd.isna(s_prev) or pd.isna(s_curr)):
            return None
        # ta.crossover(fast, slow): prev fast <= prev slow AND curr fast > curr slow
        if f_prev <= s_prev and f_curr > s_curr:
            return Signal(SignalKind.ENTER_LONG, "EmaLE")
        if f_prev >= s_prev and f_curr < s_curr:
            return Signal(SignalKind.ENTER_SHORT, "EmaSE")
        return None
