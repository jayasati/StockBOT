"""Indicators used by score_stock: RSI, session VWAP, volume ratio,
breakout/pullback detection, and EMA-extension penalty input.

Kept side-effect-free (apart from the wall-clock read in compute_volume_ratio,
which the backtest monkey-patches at runtime — see backtest.py)."""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from .config import IST


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
    """VWAP for the current session only (today's bars)."""
    if df.empty:
        return 0.0
    today = df.index[-1].date()
    today_df = df[df.index.date == today]
    if today_df.empty:
        return 0.0
    typical = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
    vwap = (typical * today_df["Volume"]).cumsum() / today_df["Volume"].cumsum()
    return float(vwap.iloc[-1])


def compute_volume_ratio(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    """
    Today's volume so far divided by what we'd expect at this point in the
    session, based on the 10-day average daily volume.

    Above 2.0 = unusual activity. Above 3.0 = strong institutional footprint.
    """
    if intraday.empty or daily.empty or len(daily) < 10:
        return 1.0

    today = intraday.index[-1].date()
    today_data = intraday[intraday.index.date == today]
    if today_data.empty:
        return 1.0
    today_vol = float(today_data["Volume"].sum())

    avg_daily_vol = float(daily["Volume"].tail(10).mean())
    if avg_daily_vol == 0:
        return 1.0

    # Fraction of NSE session elapsed (9:15 to 15:30 = 375 minutes)
    now_t = datetime.now(IST).time()
    if now_t < time(9, 15):
        fraction = 0.01
    elif now_t > time(15, 30):
        fraction = 1.0
    else:
        elapsed = (now_t.hour - 9) * 60 + now_t.minute - 15
        fraction = max(0.05, elapsed / 375)

    expected = avg_daily_vol * fraction
    return float(today_vol / expected) if expected > 0 else 1.0


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
