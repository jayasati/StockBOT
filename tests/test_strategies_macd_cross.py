"""Tests for the MACD signal-line cross strategy.

MACD math is mocked (covered in tests/test_indicators.py); these tests
probe the strategy's crossing logic + engine integration."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import MACDCross, REGISTRY, SignalKind, run_backtest
from strategies import macd_cross as macd_mod

IST = ZoneInfo("Asia/Kolkata")


def _flat_bars(n_bars: int):
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz=IST)
    arr = np.full(n_bars, 100.0, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr,
         "volume": np.full(n_bars, 1000.0)},
        index=idx,
    )


def _mock_macd(macd_values, sig_values, monkeypatch):
    """Replace momentum.macd with a fake returning the given series."""
    def fake_macd(df, fast=12, slow=26, signal=9):
        n = len(df)
        if len(macd_values) != n or len(sig_values) != n:
            raise AssertionError(
                f"mock macd expected {n} values, got "
                f"{len(macd_values)} macd / {len(sig_values)} signal"
            )
        return pd.DataFrame({
            "macd":      pd.Series(macd_values, index=df.index, dtype=float),
            "signal":    pd.Series(sig_values,  index=df.index, dtype=float),
            "histogram": pd.Series(
                [m - s for m, s in zip(macd_values, sig_values)],
                index=df.index, dtype=float),
        }, index=df.index)
    monkeypatch.setattr(macd_mod.momentum, "macd", fake_macd)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = MACDCross()
        assert s.fast == 12 and s.slow == 26 and s.signal_period == 9

    def test_custom(self):
        s = MACDCross(fast=8, slow=21, signal=5)
        assert s.fast == 8 and s.slow == 21 and s.signal_period == 5

    def test_fast_too_small(self):
        with pytest.raises(ValueError, match="fast"):
            MACDCross(fast=1, slow=26)

    def test_slow_not_greater_than_fast(self):
        with pytest.raises(ValueError, match="slow"):
            MACDCross(fast=26, slow=12)

    def test_signal_zero(self):
        with pytest.raises(ValueError, match="signal"):
            MACDCross(signal=0)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_long_on_macd_crossover_signal(self, monkeypatch):
        s = MACDCross()
        df = _flat_bars(2)
        # macd: -1, 1; sig: 0, 0 → prev -1 <= 0 ✓; curr 1 > 0 ✓ → ENTER_LONG
        _mock_macd([-1.0, 1.0], [0.0, 0.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "MacdLE"

    def test_short_on_macd_crossunder_signal(self, monkeypatch):
        s = MACDCross()
        df = _flat_bars(2)
        # macd: 1, -1; sig: 0, 0 → prev 1 >= 0 ✓; curr -1 < 0 ✓ → ENTER_SHORT
        _mock_macd([1.0, -1.0], [0.0, 0.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "MacdSE"

    def test_first_bar_returns_none(self, monkeypatch):
        s = MACDCross()
        df = _flat_bars(2)
        _mock_macd([-1.0, 1.0], [0.0, 0.0], monkeypatch)
        assert s.signal(df, 0) is None

    def test_no_signal_when_macd_above_signal_throughout(self, monkeypatch):
        s = MACDCross()
        df = _flat_bars(2)
        _mock_macd([2.0, 3.0], [0.0, 0.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_pine_crossover_uses_le_for_prev(self, monkeypatch):
        # prev macd == prev signal (touching from below) → eligible
        s = MACDCross()
        df = _flat_bars(2)
        _mock_macd([0.0, 1.0], [0.0, 0.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_nan_returns_none(self, monkeypatch):
        s = MACDCross()
        df = _flat_bars(2)
        _mock_macd([float("nan"), 1.0], [0.0, 0.0], monkeypatch)
        assert s.signal(df, 1) is None


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_then_short_reversal(self, monkeypatch):
        s = MACDCross()
        n = 6
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 100, 100, 100, 110, 110],
            "high":   [105] * n, "low": [95] * n,
            "close":  [100] * n, "volume": [1000] * n,
        }, index=idx, dtype=float)
        # Bar 1: macd crosses ABOVE signal → LONG signal → fills bar 2.
        # Bar 4: macd crosses BELOW signal → SHORT → fills bar 5 (REVERSAL).
        _mock_macd([-1, 1, 1, 1, -1, -1], [0, 0, 0, 0, 0, 0], monkeypatch)
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        sides = [t.side for t in r.trades]
        reasons = [t.exit_reason for t in r.trades]
        assert sides == ["LONG", "SHORT"]
        assert reasons[0] == "REVERSAL"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "macd_cross" in REGISTRY
        assert REGISTRY["macd_cross"] is MACDCross
