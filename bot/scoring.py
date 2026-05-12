"""Composite microstructure scoring.

The backtest replay path passes ``as_of=<bar timestamp>`` so the volume-ratio
fraction is anchored to the bar being scored, not wall-clock now."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from . import indicators

if TYPE_CHECKING:
    from indicators import IndicatorSnapshot


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
    # Phase-5 paper-tracker plumbing: scanner attaches ATR-derived
    # SL/TP and the raw indicator snapshot so ``_dispatch`` can open
    # a paper trade with the same prices Telegram shows. Defaults
    # keep every pre-Phase-5 call site (backtest, swing, tests) byte-
    # identical to before.
    sl: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    snapshot: "IndicatorSnapshot | None" = None
    # Phase-6 filter chain plumbing. ``side`` is inferred by
    # :func:`score_stock` from RSI + VWAP position (bearish-momentum
    # combination → ``SHORT``; everything else → ``LONG``). Soft and
    # hard filters in ``filters/`` mutate the next three. ``confidence``
    # is the post-multiplier confidence in 0..~1.10 range — Phase 5b
    # threshold comparisons use ``confidence * 100``. ``market_context``
    # is a free-form bag the scanner populates with nifty %, vix,
    # bank-nifty direction, etc., so individual filters don't re-fetch.
    side: str = "LONG"
    kill_reasons: list[str] = field(default_factory=list)
    soft_adjustments: list[tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    market_context: dict = field(default_factory=dict)


def score_stock(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    snapshot: "IndicatorSnapshot | None" = None,
) -> StockSignals:
    """
    Composite microstructure score (0-100). Weights tuned from a 60-day
    NSE backtest (see ``python -m backtest --sweep`` — 4796 alerts).

    Positive components:
      - Volume ratio vs 10-day expected (max 50 pts)  ← only +lift signal
        Extreme tier (>=5x) bypasses other-signal confirmation: a single
        bar with 5x+ volume is a strong enough one-shot tell that we want
        the alert to fire even without RSI/VWAP/pullback agreement.
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

    # Phase 4 wiring: when a precomputed snapshot is provided, prefer
    # snapshot.values["rsi_5m"] (Wilder's RMA, TV-parity) over the legacy
    # SMA-rolling compute_rsi. Snapshot=None keeps the legacy path so all
    # existing call sites (backtest, swing, anything that hasn't been
    # plumbed through compute_all yet) stay byte-identical.
    # TODO(Phase 7): replace this entire function with a REGISTRY-driven
    # weighted aggregation that reads from `snapshot.values` for every
    # component (RSI, MACD, ADX, volume, levels, regime gating, ...).
    if snapshot is not None and snapshot.values.get("rsi_5m") is not None:
        rsi = float(snapshot.values["rsi_5m"])
    else:
        rsi = indicators.compute_rsi(intraday["Close"])

    vol_ratio = indicators.compute_volume_ratio(intraday, daily, as_of=as_of)

    vwap = indicators.compute_session_vwap(intraday)

    above_vwap = price > vwap if vwap > 0 else False

    breakout, pct_from_high = indicators.detect_breakout(intraday, daily)

    is_pullback, _ = indicators.detect_pullback(intraday, daily)
    
    extension = indicators.compute_extension(intraday, daily)

    score = 0
    reasons: list[str] = []

    # Volume ratio — only component with positive lift in backtest.
    # The 5x tier is a one-shot trigger (+50 puts a clean bar at score 65
    # by itself + VWAP, crossing threshold 60 without needing pullback or
    # RSI agreement). This catches PVR-style intra-bar spikes that the
    # composite would otherwise wait one more bar to confirm.
    if vol_ratio >= 5.0:
        score += 50
        reasons.append(f"VR {vol_ratio:.1f}x ⚡")
    elif vol_ratio >= 3.0:
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

    # Phase-6 side inference. Only flag SHORT when momentum is
    # genuinely bearish on both axes (RSI<35 AND below VWAP). The
    # current scoring layer is long-biased — SHORT signals would still
    # need their own scoring weights before they fire alerts; this
    # field exists so the ADX-counter-trend filter can read it.
    side = "SHORT" if (rsi < 35 and not above_vwap) else "LONG"

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
        side=side,
    )
