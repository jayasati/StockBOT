"""Bollinger Bands Strategy Directed — TradingView built-in.

Pine source (v5):

    basis = ta.sma(source, length)
    dev   = mult * ta.stdev(source, length)
    upper = basis + dev
    lower = basis - dev
    if (ta.crossover(source, lower))   strategy.entry("BBandLE", strategy.long,  stop=lower, ...)
    if (ta.crossunder(source, upper))  strategy.entry("BBandSE", strategy.short, stop=upper, ...)

Mean-reversion read: LONG when close crosses back UP through the lower band
(was below, now above); SHORT when close crosses back DOWN through the upper
band (was above, now below). The ``direction`` parameter mirrors Pine's
``strategy.risk.allow_entry_in``: -1 = short-only, 0 = both, 1 = long-only.

Pine ``ta.stdev`` is the *population* stddev (divisor n), not pandas' default
sample stddev — so we pass ``ddof=0`` to ``rolling().std()``."""
from __future__ import annotations

import pandas as pd

from .base import Signal, SignalKind, Strategy


class BollingerBandsDirected(Strategy):
    name = "bollinger_bands"

    def __init__(
        self,
        length: int = 20,
        mult: float = 2.0,
        direction: int = 0,
    ) -> None:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if not (0.001 <= mult <= 50.0):
            raise ValueError(f"mult must be in [0.001, 50], got {mult}")
        if direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or 1, got {direction}")
        super().__init__(length=length, mult=mult, direction=direction)
        self.length = length
        self.mult = mult
        self.direction = direction
        # Bands cached per-DataFrame so we don't redo the rolling math on
        # every signal() call inside the engine's bar loop.
        self._cache_key: tuple[int, int] | None = None
        self._upper: pd.Series | None = None
        self._lower: pd.Series | None = None

    def _ensure_bands(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._upper is not None:
            return
        close = df["close"].astype(float)
        roll = close.rolling(self.length, min_periods=self.length)
        basis = roll.mean()
        dev = self.mult * roll.std(ddof=0)
        self._upper = basis + dev
        self._lower = basis - dev
        self._cache_key = key

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < self.length:
            return None
        self._ensure_bands(df)
        upper = self._upper
        lower = self._lower
        u_prev = upper.iat[i - 1]
        u_curr = upper.iat[i]
        l_prev = lower.iat[i - 1]
        l_curr = lower.iat[i]
        if pd.isna(u_prev) or pd.isna(u_curr) or pd.isna(l_prev) or pd.isna(l_curr):
            return None
        c_prev = float(df["close"].iat[i - 1])
        c_curr = float(df["close"].iat[i])
        crossover_lower = c_prev <= l_prev and c_curr > l_curr
        crossunder_upper = c_prev >= u_prev and c_curr < u_curr
        if crossover_lower and self.direction in (0, 1):
            return Signal(SignalKind.ENTER_LONG, "BBandLE")
        if crossunder_upper and self.direction in (0, -1):
            return Signal(SignalKind.ENTER_SHORT, "BBandSE")
        return None
