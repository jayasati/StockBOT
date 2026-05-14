"""Tests for the RSI cross strategy.

We mock ``indicators.momentum.rsi`` with a hand-built RSI series so the
tests probe the strategy's crossing logic, not the RSI math itself
(``tests/test_indicators.py`` covers that)."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import REGISTRY, RSICross, SignalKind, run_backtest
from strategies import rsi_cross as rsi_mod

IST = ZoneInfo("Asia/Kolkata")


def _flat_bars(n_bars: int, price: float = 100.0):
    """Build n flat OHLCV bars; the close values don't matter because RSI
    is mocked. Pin to early-morning so intraday auto-flatten won't fire."""
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz=IST)
    arr = np.full(n_bars, price, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr,
         "volume": np.full(n_bars, 1000.0)},
        index=idx,
    )


def _mock_rsi(values: list[float], monkeypatch):
    """Replace momentum.rsi with one that returns the given Series."""
    def fake_rsi(df, period=14):
        n = len(df)
        if len(values) != n:
            raise AssertionError(
                f"mock RSI expected {len(values)} values, frame has {n}"
            )
        return pd.Series(values, index=df.index, dtype=float)
    monkeypatch.setattr(rsi_mod.momentum, "rsi", fake_rsi)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = RSICross()
        assert s.length == 14
        assert s.oversold == 30.0
        assert s.overbought == 70.0

    def test_custom(self):
        s = RSICross(length=21, oversold=20, overbought=80)
        assert s.length == 21 and s.oversold == 20 and s.overbought == 80

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length"):
            RSICross(length=1)

    def test_inverted_thresholds_raise(self):
        with pytest.raises(ValueError, match="thresholds"):
            RSICross(oversold=70, overbought=30)

    def test_oversold_at_zero_raises(self):
        with pytest.raises(ValueError, match="thresholds"):
            RSICross(oversold=0, overbought=70)

    def test_overbought_at_100_raises(self):
        with pytest.raises(ValueError, match="thresholds"):
            RSICross(oversold=30, overbought=100)


# ---------------------------------------------------------------------------
# Signal layer
# ---------------------------------------------------------------------------

class TestSignal:
    def test_long_on_crossover_oversold(self, monkeypatch):
        # RSI: 25 → 35 across the threshold of 30 = ENTER_LONG
        s = RSICross()
        df = _flat_bars(2)
        _mock_rsi([25.0, 35.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "RsiLE"

    def test_short_on_crossunder_overbought(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        _mock_rsi([75.0, 65.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "RsiSE"

    def test_first_bar_returns_none(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        _mock_rsi([25.0, 35.0], monkeypatch)
        assert s.signal(df, 0) is None

    def test_no_signal_when_rsi_doesnt_cross(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        # Both bars in midrange, no crossing.
        _mock_rsi([45.0, 55.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_no_signal_when_rsi_touches_threshold_but_doesnt_cross(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        # prev=30 (at oversold), curr=29 (below) — that's not a crossover up.
        _mock_rsi([30.0, 29.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_pine_crossover_uses_le_for_prev(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        # prev=30 (== oversold), curr=31 (above): Pine ta.crossover treats
        # "prev <= threshold" as eligible, so this IS a crossover.
        _mock_rsi([30.0, 31.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_pine_crossunder_uses_ge_for_prev(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        _mock_rsi([70.0, 69.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT

    def test_nan_rsi_returns_none(self, monkeypatch):
        s = RSICross()
        df = _flat_bars(2)
        _mock_rsi([float("nan"), 35.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_custom_thresholds_no_cross_in_midrange(self, monkeypatch):
        s = RSICross(oversold=20, overbought=80)
        df = _flat_bars(2)
        # 35 -> 45 crosses neither the 20 nor 80 threshold.
        _mock_rsi([35.0, 45.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_custom_oversold_threshold_honored(self, monkeypatch):
        # Fresh instance so the per-DataFrame RSI cache doesn't carry over.
        s = RSICross(oversold=20, overbought=80)
        df = _flat_bars(2)
        # 18 -> 22 crosses the (custom) oversold of 20.
        _mock_rsi([18.0, 22.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_executes_long_at_next_bar_open(self, monkeypatch):
        s = RSICross()
        # 4 bars: bar 1 = LONG signal, bar 2 = fill at open, bar 3 = end.
        idx = pd.date_range("2026-05-04 09:15", periods=4, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100.0, 100.0, 105.0, 110.0],
            "high":   [101.0, 101.0, 106.0, 111.0],
            "low":    [ 99.0,  99.0, 104.0, 109.0],
            "close":  [100.0, 100.0, 105.0, 110.0],
            "volume": [1000, 1000, 1000, 1000],
        }, index=idx)
        # Mock RSI: stable 50 / 25 / 35 / 50 → crossover at bar 2 (25→35).
        _mock_rsi([50.0, 25.0, 35.0, 50.0], monkeypatch)
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 110.0  # bar 3's open

    def test_engine_handles_long_to_short_reversal(self, monkeypatch):
        s = RSICross()
        n = 8
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 100, 100, 100, 100, 100, 105, 105],
            "high":   [105] * n,
            "low":    [95] * n,
            "close":  [100, 100, 100, 100, 100, 100, 100, 100],
            "volume": [1000] * n,
        }, index=idx, dtype=float)
        # RSI 25->35 at bar 2 = LONG (fills bar 3); 75->65 at bar 5 = SHORT
        # (fills bar 6 as REVERSAL); bar 7 = END_OF_DATA closes the SHORT.
        _mock_rsi([50, 25, 35, 50, 75, 65, 50, 50], monkeypatch)
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        sides = [t.side for t in r.trades]
        reasons = [t.exit_reason for t in r.trades]
        assert sides == ["LONG", "SHORT"]
        assert reasons[0] == "REVERSAL"
        assert reasons[1] == "END_OF_DATA"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "rsi_cross" in REGISTRY
        assert REGISTRY["rsi_cross"] is RSICross
