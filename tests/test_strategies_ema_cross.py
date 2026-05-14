"""Tests for the EMA cross strategy.

EMA math is mocked (covered in tests/test_indicators.py); these tests
probe the strategy's crossing logic + engine integration."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import EMACross, REGISTRY, SignalKind, run_backtest
from strategies import ema_cross as ema_mod

IST = ZoneInfo("Asia/Kolkata")


def _flat_bars(n_bars: int):
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz=IST)
    arr = np.full(n_bars, 100.0, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr,
         "volume": np.full(n_bars, 1000.0)},
        index=idx,
    )


def _mock_emas(fast_values, slow_values, monkeypatch):
    """Replace trend.ema with a stateful fake that returns fast then slow."""
    calls = {"n": 0}

    def fake_ema(df, period=20):
        n = len(df)
        if calls["n"] % 2 == 0:
            vals = fast_values
        else:
            vals = slow_values
        calls["n"] += 1
        if len(vals) != n:
            raise AssertionError(
                f"mock ema expected {len(vals)} values, frame has {n}"
            )
        return pd.Series(vals, index=df.index, dtype=float)

    monkeypatch.setattr(ema_mod.trend, "ema", fake_ema)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = EMACross()
        assert s.fast == 9 and s.slow == 21

    def test_custom(self):
        s = EMACross(fast=5, slow=20)
        assert s.fast == 5 and s.slow == 20

    def test_fast_too_small(self):
        with pytest.raises(ValueError, match="fast"):
            EMACross(fast=1, slow=21)

    def test_slow_not_greater_than_fast_raises(self):
        with pytest.raises(ValueError, match="slow"):
            EMACross(fast=20, slow=20)
        with pytest.raises(ValueError, match="slow"):
            EMACross(fast=20, slow=10)


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignal:
    def test_long_on_fast_crossover_slow(self, monkeypatch):
        s = EMACross()
        df = _flat_bars(2)
        # fast: 99, 102; slow: 100, 101
        # prev: 99 <= 100 ✓; curr: 102 > 101 ✓ → ENTER_LONG
        _mock_emas([99.0, 102.0], [100.0, 101.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "EmaLE"

    def test_short_on_fast_crossunder_slow(self, monkeypatch):
        s = EMACross()
        df = _flat_bars(2)
        # fast: 102, 99; slow: 101, 100
        # prev: 102 >= 101 ✓; curr: 99 < 100 ✓ → ENTER_SHORT
        _mock_emas([102.0, 99.0], [101.0, 100.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "EmaSE"

    def test_first_bar_returns_none(self, monkeypatch):
        s = EMACross()
        df = _flat_bars(2)
        _mock_emas([99.0, 102.0], [100.0, 101.0], monkeypatch)
        assert s.signal(df, 0) is None

    def test_no_signal_when_emas_dont_cross(self, monkeypatch):
        s = EMACross()
        df = _flat_bars(2)
        # Both bars: fast > slow (no flip)
        _mock_emas([105.0, 106.0], [100.0, 101.0], monkeypatch)
        assert s.signal(df, 1) is None

    def test_pine_crossover_uses_le_for_prev(self, monkeypatch):
        # prev fast == prev slow (touching): Pine treats as crossover-eligible.
        s = EMACross()
        df = _flat_bars(2)
        _mock_emas([100.0, 102.0], [100.0, 101.0], monkeypatch)
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_nan_returns_none(self, monkeypatch):
        s = EMACross()
        df = _flat_bars(2)
        _mock_emas([float("nan"), 102.0], [100.0, 101.0], monkeypatch)
        assert s.signal(df, 1) is None


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_then_short_reversal(self, monkeypatch):
        s = EMACross()
        n = 6
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 100, 100, 100, 110, 110],
            "high":   [105] * n, "low": [95] * n,
            "close":  [100] * n,
            "volume": [1000] * n,
        }, index=idx, dtype=float)
        # Bar 1: fast crosses ABOVE slow → LONG signal → fills bar 2 open=100
        # Bar 4: fast crosses BELOW slow → SHORT signal → fills bar 5 open=110
        # Bar 5 = end-of-data → SHORT closes
        _mock_emas([99, 102, 102, 102, 99, 99],   # fast
                   [100, 101, 101, 101, 102, 102],  # slow
                   monkeypatch)
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
        assert "ema_cross" in REGISTRY
        assert REGISTRY["ema_cross"] is EMACross
