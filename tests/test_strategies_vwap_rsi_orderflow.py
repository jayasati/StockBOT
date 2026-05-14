"""Tests for the VWAP + RSI + Order Flow scalp strategy.

Indicator math (VWAP/RSI/ATR) is covered by tests/test_indicators.py.
These tests mock those primitives and probe the strategy's three-way
confluence: VWAP location, RSI zone+bounce, candlestick engulfing."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import REGISTRY, SignalKind, VwapRsiOrderflow, run_backtest
from strategies import vwap_rsi_orderflow as mod

IST = ZoneInfo("Asia/Kolkata")


def _bars(
    n: int, *,
    opens=None, highs=None, lows=None, closes=None,
) -> pd.DataFrame:
    """Build a tz-aware OHLC frame with arbitrary per-bar arrays."""
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
    if closes is None:
        closes = np.full(n, 100.0, dtype=float)
    if opens is None:
        opens = closes
    if highs is None:
        highs = np.asarray(closes, dtype=float) + 1.0
    if lows is None:
        lows = np.asarray(closes, dtype=float) - 1.0
    return pd.DataFrame({
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(highs, dtype=float),
        "low": np.asarray(lows, dtype=float),
        "close": np.asarray(closes, dtype=float),
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _patch(monkeypatch, *, vwap, rsi, atr):
    def _as_series(vals, df):
        s = pd.Series(vals, index=df.index, dtype=float)
        assert len(s) == len(df), "mock length mismatch"
        return s

    monkeypatch.setattr(mod.volume, "vwap",
                        lambda df: _as_series(vwap, df))
    monkeypatch.setattr(mod.momentum, "rsi",
                        lambda df, period=14: _as_series(rsi, df))
    monkeypatch.setattr(mod.volatility, "atr",
                        lambda df, period=14: _as_series(atr, df))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = VwapRsiOrderflow()
        assert s.rsi_long_low == 40.0 and s.rsi_long_high == 50.0
        assert s.rsi_short_low == 50.0 and s.rsi_short_high == 60.0
        assert s.atr_target_mult == 1.0

    def test_bad_rsi_long_zone(self):
        with pytest.raises(ValueError, match="rsi_long_low"):
            VwapRsiOrderflow(rsi_long_low=60.0, rsi_long_high=50.0)

    def test_bad_atr_target(self):
        with pytest.raises(ValueError, match="atr_target_mult"):
            VwapRsiOrderflow(atr_target_mult=0)


# ---------------------------------------------------------------------------
# Entry: all three gates aligned
# ---------------------------------------------------------------------------

class TestLongEntry:
    def test_long_when_all_three_align(self, monkeypatch):
        # Bar 0: red candle (open 101, close 99). Bar 1: green engulfing
        # (open 98 <= prev close 99; close 102 >= prev open 101).
        # Bar 1: low 97 <= VWAP 100 <= close 102 (support hold).
        # RSI: 45 → 48 (in zone, rising).
        df = _bars(
            2,
            opens=[101.0, 98.0],
            highs=[102.0, 103.0],
            lows=[98.0, 97.0],
            closes=[99.0, 102.0],
        )
        _patch(monkeypatch,
               vwap=[100.0, 100.0], rsi=[45.0, 48.0], atr=[1.0, 1.0])
        sig = VwapRsiOrderflow().signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "VwapRsiOFLE"

    def test_blocked_when_not_engulfing(self, monkeypatch):
        # Prev bar is green — no bullish engulf possible.
        df = _bars(
            2,
            opens=[99.0, 98.0], highs=[102.0, 103.0],
            lows=[98.0, 97.0], closes=[101.0, 102.0],
        )
        _patch(monkeypatch,
               vwap=[100.0, 100.0], rsi=[45.0, 48.0], atr=[1.0, 1.0])
        assert VwapRsiOrderflow().signal(df, 1) is None

    def test_blocked_when_rsi_out_of_zone(self, monkeypatch):
        df = _bars(
            2,
            opens=[101.0, 98.0], highs=[102.0, 103.0],
            lows=[98.0, 97.0], closes=[99.0, 102.0],
        )
        # RSI 35 → 38 (below the 40-50 long zone).
        _patch(monkeypatch,
               vwap=[100.0, 100.0], rsi=[35.0, 38.0], atr=[1.0, 1.0])
        assert VwapRsiOrderflow().signal(df, 1) is None

    def test_blocked_when_rsi_not_rising(self, monkeypatch):
        df = _bars(
            2,
            opens=[101.0, 98.0], highs=[102.0, 103.0],
            lows=[98.0, 97.0], closes=[99.0, 102.0],
        )
        # RSI 48 → 45 — in zone but FALLING.
        _patch(monkeypatch,
               vwap=[100.0, 100.0], rsi=[48.0, 45.0], atr=[1.0, 1.0])
        assert VwapRsiOrderflow().signal(df, 1) is None

    def test_blocked_when_no_vwap_touch(self, monkeypatch):
        # Bar 1 low 97 — but VWAP at 95: bar never touches VWAP.
        df = _bars(
            2,
            opens=[101.0, 98.0], highs=[102.0, 103.0],
            lows=[98.0, 97.0], closes=[99.0, 102.0],
        )
        _patch(monkeypatch,
               vwap=[95.0, 95.0], rsi=[45.0, 48.0], atr=[1.0, 1.0])
        assert VwapRsiOrderflow().signal(df, 1) is None


class TestShortEntry:
    def test_short_when_all_three_align(self, monkeypatch):
        # Bar 0: green (open 99, close 101). Bar 1: red engulfing.
        df = _bars(
            2,
            opens=[99.0, 102.0],
            highs=[103.0, 103.0],
            lows=[98.0, 97.0],
            closes=[101.0, 98.0],
        )
        # Bar 1: high 103 >= VWAP 100 >= close 98 (resistance reject).
        # RSI 55 → 52 (in 50-60 zone, falling).
        _patch(monkeypatch,
               vwap=[100.0, 100.0], rsi=[55.0, 52.0], atr=[1.0, 1.0])
        sig = VwapRsiOrderflow().signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "VwapRsiOFSE"


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------

class TestExit:
    def test_long_exit_on_atr_target(self, monkeypatch):
        # Enter at close=102, ATR=1, target=103. Bar 2 close=104 → target hit.
        df = _bars(
            3,
            opens=[101.0, 98.0, 103.0],
            highs=[102.0, 103.0, 105.0],
            lows=[98.0, 97.0, 102.0],
            closes=[99.0, 102.0, 104.0],
        )
        _patch(monkeypatch,
               vwap=[100.0, 100.0, 100.0],
               rsi=[45.0, 48.0, 48.0],
               atr=[1.0, 1.0, 1.0])
        s = VwapRsiOrderflow()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert "Target" in sig.reason

    def test_long_exit_on_vwap_loss(self, monkeypatch):
        df = _bars(
            3,
            opens=[101.0, 98.0, 102.0],
            highs=[102.0, 103.0, 102.5],
            lows=[98.0, 97.0, 98.5],
            closes=[99.0, 102.0, 99.0],   # bar 2 close 99 < VWAP 100
        )
        _patch(monkeypatch,
               vwap=[100.0, 100.0, 100.0],
               rsi=[45.0, 48.0, 48.0],
               atr=[1.0, 1.0, 1.0])
        s = VwapRsiOrderflow()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert "Lost" in sig.reason

    def test_short_exit_on_atr_target(self, monkeypatch):
        df = _bars(
            3,
            opens=[99.0, 102.0, 97.0],
            highs=[103.0, 103.0, 99.0],
            lows=[98.0, 97.0, 95.0],
            closes=[101.0, 98.0, 96.0],   # entry 98, target 97, bar 2 close 96 — hit
        )
        _patch(monkeypatch,
               vwap=[100.0, 100.0, 100.0],
               rsi=[55.0, 52.0, 52.0],
               atr=[1.0, 1.0, 1.0])
        s = VwapRsiOrderflow()
        assert s.signal(df, 1).kind == SignalKind.ENTER_SHORT
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert "Target" in sig.reason


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_round_trip(self, monkeypatch):
        n = 4
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [101.0, 98.0, 102.0, 99.0],
            "high":   [102.0, 103.0, 102.5, 100.0],
            "low":    [98.0, 97.0, 98.5, 98.5],
            "close":  [99.0, 102.0, 99.0, 99.5],
            "volume": [1000] * n,
        }, index=idx)
        _patch(monkeypatch,
               vwap=[100.0, 100.0, 100.0, 100.0],
               rsi=[45.0, 48.0, 48.0, 48.0],
               atr=[1.0, 1.0, 1.0, 1.0])
        r = run_backtest(VwapRsiOrderflow(), df,
                         max_intraday_loss_pct=50.0, apply_costs=False)
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.side == "LONG"
        # Bar-1 ENTER → fills bar-2 open=102; bar-2 EXIT (VWAP lost) →
        # fills bar-3 open=99.
        assert t.entry_price == pytest.approx(102.0)
        assert t.exit_price == pytest.approx(99.0)
        assert t.exit_reason == "EXIT_SIGNAL"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "vwap_rsi_orderflow" in REGISTRY
        assert REGISTRY["vwap_rsi_orderflow"] is VwapRsiOrderflow
