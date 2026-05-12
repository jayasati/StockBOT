"""Volatility indicators (ATR, Bollinger Bands).

Pure; lowercase OHLCV input; warmup → NaN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import _rma, _true_range
# Keltner builds on ATR but importing volatility.atr from itself creates
# no cycle since we only need it for run-time, not import-time, references.


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder. Pine: ``ta.atr(length)``.

    atr = rma(true_range, length)

    Warmup = ``period`` bars."""
    return _rma(_true_range(df), period).astype(np.float64)


def bollinger(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands. Pine: ``ta.bb(source, length, mult)``.

    Returns ``[upper, middle, lower, bandwidth, percent_b]``:
      middle    = ta.sma(close, length)
      stdev     = ta.stdev(close, length)
      upper     = middle + mult * stdev
      lower     = middle - mult * stdev
      bandwidth = (upper - lower) / middle
      percent_b = (close - lower) / (upper - lower)

    Pine's ``ta.stdev`` uses the population formula (ddof=0). Use
    ``rolling().std(ddof=0)`` to match exactly — pandas' default
    ``std()`` is sample (ddof=1) and would diverge.

    Warmup = ``period`` bars."""
    close = df["close"].astype(np.float64)
    middle = close.rolling(period, min_periods=period).mean()
    stdev = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + std * stdev
    lower = middle - std * stdev
    bandwidth = (upper - lower) / middle.replace(0.0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "upper": upper.astype(np.float64),
            "middle": middle.astype(np.float64),
            "lower": lower.astype(np.float64),
            "bandwidth": bandwidth.astype(np.float64),
            "percent_b": percent_b.astype(np.float64),
        },
        index=df.index,
    )


def keltner(
    df: pd.DataFrame, period: int = 20, mult: float = 2.0, atr_period: int = 10,
) -> pd.DataFrame:
    """Keltner Channels. Pine: ``ta.kc(source, length, mult, atrLength)``.

      middle = ema(close, length)
      upper  = middle + mult * atr(atr_length)
      lower  = middle - mult * atr(atr_length)

    Smoother than Bollinger because it uses ATR (range) instead of stdev
    (variance). The TTM squeeze setup is "Bollinger Bands inside Keltner".

    Warmup = max(``period``, ``atr_period``)."""
    close = df["close"].astype(np.float64)
    middle = close.ewm(span=period, adjust=False, min_periods=period).mean()
    atr_val = atr(df, period=atr_period)
    upper = middle + mult * atr_val
    lower = middle - mult * atr_val
    return pd.DataFrame(
        {"upper": upper.astype(np.float64),
         "middle": middle.astype(np.float64),
         "lower": lower.astype(np.float64)},
        index=df.index,
    )


def ttm_squeeze(
    df: pd.DataFrame, period: int = 20, bb_std: float = 2.0,
    kc_mult: float = 1.5, atr_period: int = 10,
) -> pd.DataFrame:
    """John Carter's TTM Squeeze.

      in_squeeze = 1 when (BB.upper < KC.upper) AND (BB.lower > KC.lower)
      momentum   = linreg slope of (close - midline) over `period` bars,
                   where midline = avg((hh+ll)/2, sma(close))

    ``in_squeeze`` going from 1 → 0 is the "squeeze release" trigger.
    Momentum sign indicates direction of breakout, magnitude its force.

    Warmup = ``period`` bars (linreg + BB + KC all need ``period``)."""
    bb = bollinger(df, period=period, std=bb_std)
    kc = keltner(df, period=period, mult=kc_mult, atr_period=atr_period)
    in_squeeze = (
        (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    ).astype(np.float64)

    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    sma_close = df["close"].rolling(period, min_periods=period).mean()
    midline = ((hh + ll) / 2.0 + sma_close) / 2.0
    delta = df["close"].astype(np.float64) - midline

    def _linreg_last(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        n = len(window)
        x = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, window, 1)
        return slope * (n - 1) + intercept   # value of the line at the last point

    momentum = delta.rolling(period, min_periods=period).apply(
        _linreg_last, raw=True,
    )
    return pd.DataFrame(
        {"in_squeeze": in_squeeze, "momentum": momentum.astype(np.float64)},
        index=df.index,
    )
