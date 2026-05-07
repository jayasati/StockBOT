"""Indicators used by score_stock: RSI, session VWAP, volume ratio,
breakout/pullback detection, and EMA-extension penalty input.

The wall-clock read in compute_volume_ratio is parameterised — backtest
replay passes ``as_of=<bar timestamp>`` so the session-elapsed fraction is
deterministic; live path leaves it None."""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

import features

from .config import IST


def _to_lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename yfinance-style uppercase OHLCV to features.py's lowercase."""
    return df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Standard 14-period RSI on closing prices."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if not pd.isna(last) else 50.0


def compute_session_vwap(df: pd.DataFrame) -> float:
    """Cumulative session VWAP at the latest bar (legacy scalar shape).

    Thin wrapper over ``features.session_vwap``. The legacy callers in this
    package pass yfinance-style uppercase OHLCV; ``features`` is the
    canonical lowercase implementation.
    """
    if df.empty:
        return 0.0
    bars = _to_lower_columns(df)
    session_date = bars.index[-1].date()
    vwap = features.session_vwap(bars, session_date)
    if vwap.empty:
        return 0.0
    return float(vwap.iloc[-1])


def compute_volume_ratio(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    as_of=None,
) -> float:
    """Volume ratio. Thin wrapper over ``features.volume_ratio``.

    Pass ``as_of=<Timestamp>`` from the backtest replay; live callers leave it
    None to use wall-clock now."""
    return features.volume_ratio(intraday, daily, as_of=as_of)


def detect_breakout(intraday: pd.DataFrame, daily: pd.DataFrame) -> tuple[bool, float]:
    """Did current price clear the 20-day high? (Diagnostic only; not scored.)"""
    if intraday.empty or len(daily) < 20:
        return False, 0.0
    recent_high = float(daily["High"].tail(20).max())
    current = float(intraday["Close"].iloc[-1])
    pct_from_high = (current - recent_high) / recent_high * 100
    return current > recent_high, pct_from_high


def detect_pullback(intraday: pd.DataFrame, daily: pd.DataFrame) -> tuple[bool, float]:
    """
    'Buy the retest' pattern. The 20-day high was set in the last 5 sessions
    AND current price has come back to within [-3%, 0%] of that high.

    This replaces the raw breakout reward, which the backtest showed was
    anti-predictive (-2.6pp lift) — the bot was buying the parabola top.
    """
    if intraday.empty or len(daily) < 20:
        return False, 0.0
    window = daily.tail(20)
    high_idx = int(np.argmax(window["High"].values))
    days_since_high = (len(window) - 1) - high_idx
    recent_high = float(window["High"].iloc[high_idx])
    current = float(intraday["Close"].iloc[-1])
    pct_from_high = (current - recent_high) / recent_high * 100
    is_pullback = days_since_high <= 5 and -3.0 <= pct_from_high <= 0.0
    return is_pullback, pct_from_high


def compute_extension(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    """
    Percent above the 20-day EMA of close. Positive = stretched above trend.

    The backtest showed alerts firing on parabolic moves (e.g. ATGL VR 26.7x
    at +14.7% breakout → −13% next day). This metric drives the extension
    penalty in score_stock so the bot stops chasing those tops.
    """
    if intraday.empty or len(daily) < 20:
        return 0.0
    ema20 = float(daily["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    if ema20 <= 0:
        return 0.0
    current = float(intraday["Close"].iloc[-1])
    return (current - ema20) / ema20 * 100
