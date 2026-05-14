"""BarUpDn strategy — the TradingView built-in.

Pine source (v5):

    if (close > open and open > close[1])
        strategy.entry("BarUp", strategy.long)
    if (close < open and open < close[1])
        strategy.entry("BarDn", strategy.short)

Long when the current bar is green AND opens above the previous close.
Short when the current bar is red AND opens below the previous close.
The max-intraday-loss kill switch lives in the backtest engine, not here."""
from __future__ import annotations

import pandas as pd

from .base import Signal, SignalKind, Strategy


class BarUpDn(Strategy):
    name = "bar_up_dn"

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        prev_close = float(df["close"].iat[i - 1])
        curr_open = float(df["open"].iat[i])
        curr_close = float(df["close"].iat[i])
        if curr_close > curr_open and curr_open > prev_close:
            return Signal(SignalKind.ENTER_LONG, "BarUp")
        if curr_close < curr_open and curr_open < prev_close:
            return Signal(SignalKind.ENTER_SHORT, "BarDn")
        return None
