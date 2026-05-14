"""MACD signal-line cross strategy.

Pine reference (v5)::

    [macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)
    if (ta.crossover(macdLine, signalLine))   strategy.entry("Long",  strategy.long)
    if (ta.crossunder(macdLine, signalLine))  strategy.entry("Short", strategy.short)

Momentum read: LONG when MACD line crosses ABOVE its signal line (bullish
momentum onset); SHORT when it crosses BELOW (bearish onset). The signal
line is itself a smoothing of the MACD line, so this is a smoother /
slower cousin of the raw EMA cross. Reuses ``indicators.momentum.macd``."""
from __future__ import annotations

import pandas as pd

from indicators import momentum

from .base import Signal, SignalKind, Strategy


class MACDCross(Strategy):
    name = "macd_cross"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast < 2:
            raise ValueError(f"fast must be >= 2, got {fast}")
        if slow <= fast:
            raise ValueError(f"slow ({slow}) must be > fast ({fast})")
        if signal < 1:
            raise ValueError(f"signal must be >= 1, got {signal}")
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self._cache_key: tuple[int, int] | None = None
        self._macd: pd.Series | None = None
        self._sig: pd.Series | None = None

    def _ensure(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._macd is not None:
            return
        out = momentum.macd(
            df, fast=self.fast, slow=self.slow, signal=self.signal_period,
        )
        self._macd = out["macd"]
        self._sig = out["signal"]
        self._cache_key = key

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure(df)
        m_prev = self._macd.iat[i - 1]
        m_curr = self._macd.iat[i]
        s_prev = self._sig.iat[i - 1]
        s_curr = self._sig.iat[i]
        if (pd.isna(m_prev) or pd.isna(m_curr)
                or pd.isna(s_prev) or pd.isna(s_curr)):
            return None
        if m_prev <= s_prev and m_curr > s_curr:
            return Signal(SignalKind.ENTER_LONG, "MacdLE")
        if m_prev >= s_prev and m_curr < s_curr:
            return Signal(SignalKind.ENTER_SHORT, "MacdSE")
        return None
