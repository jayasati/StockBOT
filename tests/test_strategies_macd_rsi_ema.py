"""Tests for the MACD + RSI + EMA momentum-confirmation strategy.

Indicator math (MACD/RSI/EMA) is covered by tests/test_indicators.py;
these tests mock those primitives and probe the strategy's combined-gate
logic + time window + exit on MACD reversal."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import MacdRsiEma, REGISTRY, SignalKind, run_backtest
from strategies import macd_rsi_ema as mod

IST = ZoneInfo("Asia/Kolkata")


def _bars(n: int, *, start="2026-05-04 09:15", closes=None) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz=IST)
    if closes is None:
        closes = np.full(n, 100.0, dtype=float)
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr + 1, "low": arr - 1, "close": arr,
         "volume": np.full(n, 1000.0)},
        index=idx,
    )


def _patch(monkeypatch, hist, rsi, ema):
    def _as_series(vals, df):
        s = pd.Series(vals, index=df.index, dtype=float)
        assert len(s) == len(df), "mock length mismatch"
        return s

    monkeypatch.setattr(
        mod.momentum, "macd",
        lambda df, fast=12, slow=26, signal=9: pd.DataFrame(
            {"macd": _as_series(hist, df),
             "signal": _as_series([0.0] * len(df), df),
             "histogram": _as_series(hist, df)},
            index=df.index,
        ),
    )
    monkeypatch.setattr(mod.momentum, "rsi",
                        lambda df, period=14: _as_series(rsi, df))
    monkeypatch.setattr(mod.trend, "ema",
                        lambda df, period=20: _as_series(ema, df))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = MacdRsiEma()
        assert s.macd_fast == 12 and s.macd_slow == 26
        assert s.rsi_mid == 50.0 and s.rsi_low == 25.0 and s.rsi_high == 75.0
        assert s.ema_period == 20 and s.minutes_after_open == 90

    def test_bad_macd_order(self):
        with pytest.raises(ValueError, match="macd_slow"):
            MacdRsiEma(macd_fast=26, macd_slow=12)

    def test_bad_rsi_thresholds(self):
        with pytest.raises(ValueError, match="rsi_low"):
            MacdRsiEma(rsi_low=60.0, rsi_high=50.0)

    def test_bad_window(self):
        with pytest.raises(ValueError, match="minutes_after_open"):
            MacdRsiEma(minutes_after_open=1)


# ---------------------------------------------------------------------------
# Entry signal
# ---------------------------------------------------------------------------

class TestEntry:
    def test_long_when_all_three_align(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1],   # turns positive
               rsi=[49.0, 55.0],   # crosses up through 50
               ema=[100.0, 100.0]) # close 101 > EMA 100
        sig = MacdRsiEma().signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "MacdRsiEmaLE"

    def test_short_when_all_three_align(self, monkeypatch):
        df = _bars(2, closes=[101.0, 99.0])
        _patch(monkeypatch,
               hist=[0.1, -0.1],   # turns negative
               rsi=[51.0, 45.0],   # crosses down through 50
               ema=[100.0, 100.0]) # close 99 < EMA 100
        sig = MacdRsiEma().signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "MacdRsiEmaSE"

    def test_long_blocked_when_rsi_overbought(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 80.0], ema=[100.0, 100.0])
        assert MacdRsiEma().signal(df, 1) is None

    def test_short_blocked_when_rsi_oversold(self, monkeypatch):
        df = _bars(2, closes=[101.0, 99.0])
        _patch(monkeypatch,
               hist=[0.1, -0.1], rsi=[51.0, 20.0], ema=[100.0, 100.0])
        assert MacdRsiEma().signal(df, 1) is None

    def test_long_blocked_when_price_below_ema(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 55.0], ema=[102.0, 102.0])
        assert MacdRsiEma().signal(df, 1) is None

    def test_long_blocked_when_only_two_align(self, monkeypatch):
        # MACD up + RSI up but no EMA confirmation.
        df = _bars(2, closes=[99.0, 99.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 55.0], ema=[100.0, 100.0])
        assert MacdRsiEma().signal(df, 1) is None


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_entry_outside_window_is_blocked(self, monkeypatch):
        # 11:00 bar — well past 09:15 + 90 min = 10:45.
        df = _bars(2, start="2026-05-04 10:55", closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 55.0], ema=[100.0, 100.0])
        assert MacdRsiEma().signal(df, 1) is None

    def test_entry_at_window_boundary(self, monkeypatch):
        # 10:40 bar = 85 min after open → still inside default 90-min window.
        df = _bars(2, start="2026-05-04 10:35", closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 55.0], ema=[100.0, 100.0])
        sig = MacdRsiEma().signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_custom_window(self, monkeypatch):
        # 30-min window: 09:50 is outside.
        df = _bars(2, start="2026-05-04 09:45", closes=[99.0, 101.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1], rsi=[49.0, 55.0], ema=[100.0, 100.0])
        assert MacdRsiEma(minutes_after_open=30).signal(df, 1) is None


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------

class TestExit:
    def test_long_exits_on_macd_reversal(self, monkeypatch):
        # Enter long at bar 1, exit at bar 2 when histogram flips back.
        df = _bars(3, closes=[99.0, 101.0, 100.0])
        _patch(monkeypatch,
               hist=[-0.1, 0.1, -0.1],
               rsi=[49.0, 55.0, 55.0],
               ema=[100.0, 100.0, 100.0])
        s = MacdRsiEma()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert sig.reason == "MacdRevL"

    def test_short_exits_on_macd_reversal(self, monkeypatch):
        df = _bars(3, closes=[101.0, 99.0, 100.0])
        _patch(monkeypatch,
               hist=[0.1, -0.1, 0.1],
               rsi=[51.0, 45.0, 45.0],
               ema=[100.0, 100.0, 100.0])
        s = MacdRsiEma()
        assert s.signal(df, 1).kind == SignalKind.ENTER_SHORT
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert sig.reason == "MacdRevS"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_round_trip(self, monkeypatch):
        n = 4
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [99.0, 99.0, 101.0, 100.0],
            "high":   [100.0, 102.0, 102.0, 100.5],
            "low":    [98.0, 98.5, 100.0, 99.5],
            "close":  [99.0, 101.0, 100.0, 100.0],
            "volume": [1000] * n,
        }, index=idx)
        _patch(monkeypatch,
               hist=[-0.1, 0.1, -0.1, -0.1],
               rsi=[49.0, 55.0, 55.0, 55.0],
               ema=[100.0, 100.0, 100.0, 100.0])
        r = run_backtest(MacdRsiEma(), df,
                         max_intraday_loss_pct=50.0, apply_costs=False)
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.side == "LONG"
        # Bar-1 ENTER → fills bar-2 open=101; bar-2 EXIT → fills bar-3 open=100.
        assert t.entry_price == pytest.approx(101.0)
        assert t.exit_price == pytest.approx(100.0)
        assert t.exit_reason == "EXIT_SIGNAL"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "macd_rsi_ema" in REGISTRY
        assert REGISTRY["macd_rsi_ema"] is MacdRsiEma
