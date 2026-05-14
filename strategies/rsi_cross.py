"""RSI cross strategy — TradingView's built-in 'RSI Strategy'.

Pine reference (v5)::

    length = input.int(14)
    overSold = input.int(30)
    overBought = input.int(70)
    vrsi = ta.rsi(close, length)
    if (ta.crossover(vrsi, overSold))   strategy.entry("RsiLE", strategy.long)
    if (ta.crossunder(vrsi, overBought)) strategy.entry("RsiSE", strategy.short)

Mean-reversion read: LONG when RSI crosses back UP through the oversold
threshold (price was beaten down, now recovering); SHORT when RSI crosses
back DOWN through the overbought threshold (price was extended, now
retreating). Reuses ``indicators.momentum.rsi`` (Wilder's, matches
``ta.rsi``)."""
from __future__ import annotations

import pandas as pd

from indicators import momentum

from .base import Signal, SignalKind, Strategy


class RSICross(Strategy):
    name = "rsi_cross"

    def __init__(
        self,
        length: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if length < 2:
            raise ValueError(f"length must be >= 2, got {length}")
        if not (0 < oversold < overbought < 100):
            raise ValueError(
                f"thresholds must satisfy 0 < oversold ({oversold}) < "
                f"overbought ({overbought}) < 100"
            )
        super().__init__(length=length, oversold=oversold, overbought=overbought)
        self.length = length
        self.oversold = oversold
        self.overbought = overbought
        self._cache_key: tuple[int, int] | None = None
        self._rsi: pd.Series | None = None

    def _ensure_rsi(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._rsi is not None:
            return
        self._rsi = momentum.rsi(df, period=self.length)
        self._cache_key = key

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        # Need RSI valid at both i-1 and i; RSI's own warmup is handled by
        # the NaN check below, but a cheap pre-filter saves work.
        if i < 1:
            return None
        self._ensure_rsi(df)
        prev = self._rsi.iat[i - 1]
        curr = self._rsi.iat[i]
        if pd.isna(prev) or pd.isna(curr):
            return None
        # ta.crossover(rsi, oversold) → prev <= oversold AND curr > oversold
        if prev <= self.oversold and curr > self.oversold:
            return Signal(SignalKind.ENTER_LONG, "RsiLE")
        # ta.crossunder(rsi, overbought) → prev >= overbought AND curr < overbought
        if prev >= self.overbought and curr < self.overbought:
            return Signal(SignalKind.ENTER_SHORT, "RsiSE")
        return None
