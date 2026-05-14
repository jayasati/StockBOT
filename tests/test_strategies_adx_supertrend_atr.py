"""Tests for the ADX + Supertrend + ATR strategy.

Indicator math is covered by tests/test_indicators.py. These tests mock
supertrend / adx / atr and probe the strategy's gates + stops + exits."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import AdxSupertrendAtr, REGISTRY, SignalKind, run_backtest
from strategies import adx_supertrend_atr as mod

IST = ZoneInfo("Asia/Kolkata")


def _bars(n: int, *, closes=None) -> pd.DataFrame:
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
    if closes is None:
        closes = np.full(n, 100.0, dtype=float)
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr + 1, "low": arr - 1, "close": arr,
         "volume": np.full(n, 1000.0)},
        index=idx,
    )


def _patch(monkeypatch, *, direction, adx, dip, dim, atr):
    def _as_series(vals, df):
        s = pd.Series(vals, index=df.index, dtype=float)
        assert len(s) == len(df), "mock length mismatch"
        return s

    def fake_supertrend(df, period=10, multiplier=3.0):
        return pd.DataFrame(
            {"supertrend": _as_series([0.0] * len(df), df),
             "direction": _as_series(direction, df)},
            index=df.index,
        )

    def fake_adx(df, period=14):
        return pd.DataFrame(
            {"adx": _as_series(adx, df),
             "di_plus": _as_series(dip, df),
             "di_minus": _as_series(dim, df)},
            index=df.index,
        )

    monkeypatch.setattr(mod.trend, "supertrend", fake_supertrend)
    monkeypatch.setattr(mod.trend, "adx", fake_adx)
    monkeypatch.setattr(mod.volatility, "atr",
                        lambda df, period=14: _as_series(atr, df))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = AdxSupertrendAtr()
        assert s.min_adx == 25.0 and s.atr_stop_mult == 1.0

    def test_bad_min_adx(self):
        with pytest.raises(ValueError, match="min_adx"):
            AdxSupertrendAtr(min_adx=0.0)
        with pytest.raises(ValueError, match="min_adx"):
            AdxSupertrendAtr(min_adx=100.0)

    def test_bad_atr_stop_mult(self):
        with pytest.raises(ValueError, match="atr_stop_mult"):
            AdxSupertrendAtr(atr_stop_mult=0.0)


# ---------------------------------------------------------------------------
# Entry signal
# ---------------------------------------------------------------------------

class TestEntry:
    def test_long_when_all_gates_align(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               direction=[-1, 1],   # green flip
               adx=[26.0, 30.0],    # > 25 and rising
               dip=[20.0, 28.0], dim=[15.0, 12.0],  # +DI > -DI
               atr=[1.0, 1.0])
        s = AdxSupertrendAtr()
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "AdxSupertrendLE"
        # stop = entry - 1 * ATR = 101 - 1 = 100
        assert s._stop_price == pytest.approx(100.0)

    def test_short_when_all_gates_align(self, monkeypatch):
        df = _bars(2, closes=[101.0, 99.0])
        _patch(monkeypatch,
               direction=[1, -1],  # red flip
               adx=[26.0, 30.0],
               dip=[15.0, 12.0], dim=[20.0, 28.0],  # -DI > +DI
               atr=[1.0, 1.0])
        s = AdxSupertrendAtr()
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "AdxSupertrendSE"
        assert s._stop_price == pytest.approx(100.0)  # 99 + 1

    def test_blocked_when_adx_below_threshold(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               direction=[-1, 1], adx=[20.0, 22.0],   # < 25
               dip=[20.0, 28.0], dim=[15.0, 12.0], atr=[1.0, 1.0])
        assert AdxSupertrendAtr().signal(df, 1) is None

    def test_blocked_when_adx_not_rising(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               direction=[-1, 1], adx=[30.0, 28.0],   # falling
               dip=[20.0, 28.0], dim=[15.0, 12.0], atr=[1.0, 1.0])
        assert AdxSupertrendAtr().signal(df, 1) is None

    def test_blocked_when_di_disagrees_with_flip(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch(monkeypatch,
               direction=[-1, 1], adx=[26.0, 30.0],
               dip=[10.0, 12.0], dim=[20.0, 28.0],   # -DI > +DI on long flip
               atr=[1.0, 1.0])
        assert AdxSupertrendAtr().signal(df, 1) is None

    def test_blocked_when_supertrend_did_not_flip(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        # Stayed bullish — no flip from -1 → +1.
        _patch(monkeypatch,
               direction=[1, 1], adx=[26.0, 30.0],
               dip=[20.0, 28.0], dim=[15.0, 12.0], atr=[1.0, 1.0])
        assert AdxSupertrendAtr().signal(df, 1) is None


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

class TestExit:
    def test_long_exits_on_atr_stop(self, monkeypatch):
        # Enter long @ 101 with ATR=1 ⇒ stop=100. Bar 2 close=99 → stop hit.
        df = _bars(3, closes=[99.0, 101.0, 99.0])
        _patch(monkeypatch,
               direction=[-1, 1, 1],   # still green
               adx=[26.0, 30.0, 32.0],
               dip=[20.0, 28.0, 28.0], dim=[15.0, 12.0, 12.0],
               atr=[1.0, 1.0, 1.0])
        s = AdxSupertrendAtr()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert sig.reason == "AdxStopL"

    def test_long_exits_on_supertrend_flip(self, monkeypatch):
        df = _bars(3, closes=[99.0, 101.0, 100.5])
        _patch(monkeypatch,
               direction=[-1, 1, -1],   # flipped back
               adx=[26.0, 30.0, 32.0],
               dip=[20.0, 28.0, 28.0], dim=[15.0, 12.0, 12.0],
               atr=[1.0, 1.0, 1.0])
        s = AdxSupertrendAtr()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert sig.reason == "AdxSupertrendL_Flip"

    def test_short_exits_on_atr_stop(self, monkeypatch):
        # Enter short @ 99 with ATR=1 ⇒ stop=100. Bar 2 close=101 → stop hit.
        df = _bars(3, closes=[101.0, 99.0, 101.0])
        _patch(monkeypatch,
               direction=[1, -1, -1],
               adx=[26.0, 30.0, 32.0],
               dip=[15.0, 12.0, 12.0], dim=[20.0, 28.0, 28.0],
               atr=[1.0, 1.0, 1.0])
        s = AdxSupertrendAtr()
        assert s.signal(df, 1).kind == SignalKind.ENTER_SHORT
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert sig.reason == "AdxStopS"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_round_trip(self, monkeypatch):
        n = 4
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [99.0, 99.0, 101.0, 100.5],
            "high":   [100.0, 102.0, 102.0, 101.0],
            "low":    [98.0, 98.5, 100.0, 99.5],
            "close":  [99.0, 101.0, 100.5, 100.5],
            "volume": [1000] * n,
        }, index=idx)
        _patch(monkeypatch,
               direction=[-1, 1, -1, -1],
               adx=[26.0, 30.0, 32.0, 32.0],
               dip=[20.0, 28.0, 28.0, 28.0],
               dim=[15.0, 12.0, 12.0, 12.0],
               atr=[1.0, 1.0, 1.0, 1.0])
        r = run_backtest(AdxSupertrendAtr(), df,
                         max_intraday_loss_pct=50.0, apply_costs=False)
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.side == "LONG"
        # Bar-1 ENTER → fills bar-2 open=101; bar-2 EXIT → fills bar-3 open=100.5
        assert t.entry_price == pytest.approx(101.0)
        assert t.exit_price == pytest.approx(100.5)
        assert t.exit_reason == "EXIT_SIGNAL"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "adx_supertrend_atr" in REGISTRY
        assert REGISTRY["adx_supertrend_atr"] is AdxSupertrendAtr
