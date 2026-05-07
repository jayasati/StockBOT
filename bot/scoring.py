"""Composite microstructure scoring.

The backtest replay path passes ``as_of=<bar timestamp>`` so the volume-ratio
fraction is anchored to the bar being scored, not wall-clock now."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import indicators


@dataclass
class StockSignals:
    symbol: str
    price: float
    rsi: float
    volume_ratio: float
    above_vwap: bool
    breakout: bool
    pct_from_high: float
    score: int
    reasons: list[str] = field(default_factory=list)
    filing_title: str | None = None


def score_stock(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> StockSignals:
    """
    Composite microstructure score (0-100). Weights tuned from a 60-day
    NSE backtest (see ``python -m backtest --sweep`` — 4796 alerts).

    Positive components:
      - Volume ratio vs 10-day expected (max 40 pts)  ← only +lift signal
      - Pullback to 20-day high made in last 5 sessions (25 pts)
      - RSI in momentum zone 60-70 (max 15 pts)       ← was anti-predictive at 25
      - Above session VWAP (15 pts)

    Penalties:
      - Price > 20-day EMA × 1.07 (-20 pts), > 1.10 (-40 pts)
      - RSI > 80 (-15 pts)

    The raw breakout reward was removed — the backtest showed -2.6pp lift,
    consistent with chasing the parabola top.
    """
    if intraday.empty or daily.empty:
        return StockSignals(symbol, 0.0, 50.0, 1.0, False, False, 0.0, 0, ["no data"])

    price = float(intraday["Close"].iloc[-1])

    rsi = indicators.compute_rsi(intraday["Close"])

    vol_ratio = indicators.compute_volume_ratio(intraday, daily, as_of=as_of)

    vwap = indicators.compute_session_vwap(intraday)

    above_vwap = price > vwap if vwap > 0 else False

    breakout, pct_from_high = indicators.detect_breakout(intraday, daily)

    is_pullback, _ = indicators.detect_pullback(intraday, daily)
    
    extension = indicators.compute_extension(intraday, daily)

    score = 0
    reasons: list[str] = []

    # Volume ratio — only component with positive lift in backtest
    if vol_ratio >= 3.0:
        score += 40
        reasons.append(f"VR {vol_ratio:.1f}x")
    elif vol_ratio >= 2.0:
        score += 30
        reasons.append(f"VR {vol_ratio:.1f}x")
    elif vol_ratio >= 1.5:
        score += 20
        reasons.append(f"VR {vol_ratio:.1f}x")

    # RSI in momentum zone — weight reduced (slightly anti-predictive)
    if 60 <= rsi <= 70:
        score += 15
        reasons.append(f"RSI {rsi:.0f}")
    elif 55 <= rsi < 60 or 70 < rsi <= 75:
        score += 8
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi > 80:
        score -= 15  # overbought penalty (deepened from -10)

    # Above session VWAP
    if above_vwap:
        score += 15
        reasons.append("> VWAP")

    # Pullback to recent high — replaces the old breakout reward
    if is_pullback:
        score += 25
        reasons.append(f"pullback {pct_from_high:+.1f}%")

    # Extension penalty — kills the "buying parabola top" pattern
    if extension > 10:
        score -= 40
        reasons.append(f"extended +{extension:.1f}%")
    elif extension > 7:
        score -= 20
        reasons.append(f"extended +{extension:.1f}%")

    score = max(0, min(100, score))

    return StockSignals(
        symbol=symbol,
        price=price,
        rsi=rsi,
        volume_ratio=vol_ratio,
        above_vwap=above_vwap,
        breakout=breakout,
        pct_from_high=pct_from_high,
        score=score,
        reasons=reasons,
    )
