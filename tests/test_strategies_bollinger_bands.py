"""Tests for the Bollinger Bands Directed strategy."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import (
    REGISTRY, BollingerBandsDirected, SignalKind, run_backtest,
)

IST = ZoneInfo("Asia/Kolkata")


def _bars_from_closes(closes, start="2026-05-04 09:15", freq="5min", tz=IST):
    """Build an OHLCV frame where open=high=low=close (so OHLC math is
    irrelevant — only close drives BB signals)."""
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz=tz)
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr,
         "volume": np.full(len(arr), 1000.0)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = BollingerBandsDirected()
        assert s.length == 20
        assert s.mult == 2.0
        assert s.direction == 0

    def test_custom(self):
        s = BollingerBandsDirected(length=10, mult=1.5, direction=1)
        assert s.length == 10 and s.mult == 1.5 and s.direction == 1

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="length"):
            BollingerBandsDirected(length=0)

    def test_invalid_mult_low(self):
        with pytest.raises(ValueError, match="mult"):
            BollingerBandsDirected(mult=0.0)

    def test_invalid_mult_high(self):
        with pytest.raises(ValueError, match="mult"):
            BollingerBandsDirected(mult=51.0)

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            BollingerBandsDirected(direction=2)


# ---------------------------------------------------------------------------
# Signal layer
# ---------------------------------------------------------------------------

class TestBollingerSignal:
    def test_warmup_returns_none(self):
        s = BollingerBandsDirected(length=5, mult=1.0)
        df = _bars_from_closes([100.0] * 4)
        # Even with len(df)>=length-1, i < length must be None.
        for i in range(min(len(df), 5)):
            assert s.signal(df, i) is None

    def test_long_on_crossover_lower_band(self):
        # 5 flat bars → bar 5 dips below lower band → bar 6 closes back above.
        s = BollingerBandsDirected(length=5, mult=1.0)
        closes = [100, 100, 100, 100, 100, 98, 99]
        df = _bars_from_closes(closes)
        # At bar 5: close=98, basis=99.6, stdev≈0.8 → lower≈98.8 → close < lower
        # At bar 6: close=99, basis=99.4, stdev≈0.8 → lower≈98.6 → close > lower
        sig = s.signal(df, 6)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "BBandLE"

    def test_short_on_crossunder_upper_band(self):
        s = BollingerBandsDirected(length=5, mult=1.0)
        closes = [100, 100, 100, 100, 100, 102, 101]
        df = _bars_from_closes(closes)
        sig = s.signal(df, 6)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "BBandSE"

    def test_no_signal_when_close_inside_bands(self):
        s = BollingerBandsDirected(length=5, mult=2.0)
        # All closes well inside ±2σ band → no crossings.
        df = _bars_from_closes([100, 100.1, 99.9, 100.05, 99.95, 100.0, 100.1])
        for i in range(len(df)):
            assert s.signal(df, i) is None

    def test_direction_long_only_blocks_short(self):
        s = BollingerBandsDirected(length=5, mult=1.0, direction=1)
        df = _bars_from_closes([100, 100, 100, 100, 100, 102, 101])
        # Same setup that triggered SHORT above; now direction=1 must skip it.
        assert s.signal(df, 6) is None

    def test_direction_short_only_blocks_long(self):
        s = BollingerBandsDirected(length=5, mult=1.0, direction=-1)
        df = _bars_from_closes([100, 100, 100, 100, 100, 98, 99])
        assert s.signal(df, 6) is None

    def test_direction_long_only_allows_long(self):
        s = BollingerBandsDirected(length=5, mult=1.0, direction=1)
        df = _bars_from_closes([100, 100, 100, 100, 100, 98, 99])
        sig = s.signal(df, 6)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_direction_short_only_allows_short(self):
        s = BollingerBandsDirected(length=5, mult=1.0, direction=-1)
        df = _bars_from_closes([100, 100, 100, 100, 100, 102, 101])
        sig = s.signal(df, 6)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT

    def test_uses_population_stdev(self):
        # Pine's ta.stdev divides by n, not n-1. Hand-compute to verify.
        s = BollingerBandsDirected(length=5, mult=1.0)
        closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
        df = _bars_from_closes(closes)
        s._ensure_bands(df)
        # SMA of bars 0..4 = 100
        # Population variance = mean((x-100)^2) = (0+1+1+4+4)/5 = 2.0
        # Population stdev = sqrt(2.0) ≈ 1.41421
        expected_basis = 100.0
        expected_lower = 100.0 - np.sqrt(2.0)
        expected_upper = 100.0 + np.sqrt(2.0)
        assert s._lower.iat[4] == pytest.approx(expected_lower, abs=1e-9)
        assert s._upper.iat[4] == pytest.approx(expected_upper, abs=1e-9)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_entry_fills_at_next_bar_open(self):
        # Warmup must have non-zero stdev or the prev-close-on-the-band
        # situation triggers a spurious crossunder on the first move.
        s = BollingerBandsDirected(length=5, mult=1.0)
        idx = pd.date_range("2026-05-04 09:15", periods=10, freq="5min", tz=IST)
        closes = [99, 100, 101, 100, 99, 98, 99, 100, 101, 102]
        # Distinct open at bar 7 so we can verify next-bar-open fill price.
        opens = [99, 100, 101, 100, 99, 98, 99, 99.5, 101, 102]
        df = pd.DataFrame({
            "open": opens, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1000] * 10,
        }, index=idx)
        # Crossover at bar 6 (close 98→99 through lower band) → bar 7 open.
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 99.5
        assert t.exit_price == 102
        assert t.exit_reason == "END_OF_DATA"

    def test_short_entry_then_reversal(self):
        s = BollingerBandsDirected(length=5, mult=1.0)
        # Warmup oscillates so prev-close isn't sitting on a degenerate band:
        # bars 0-4 give upper≈100.95 / lower≈99.45.
        # bar 6: close 102→101 through upper → SHORT signal → fills bar 7 open.
        # bar 10: close 96→99 through lower → LONG signal → fills bar 11 open
        # which reverses the SHORT.
        closes = [101, 100, 99, 100, 101,
                  102, 101, 100, 99, 96, 99, 100]
        df = _bars_from_closes(closes)
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        sides = [t.side for t in r.trades]
        reasons = [t.exit_reason for t in r.trades]
        assert sides == ["SHORT", "LONG"]
        assert reasons == ["REVERSAL", "END_OF_DATA"]

    def test_no_trades_when_market_inside_bands(self):
        s = BollingerBandsDirected(length=5, mult=2.0)
        df = _bars_from_closes([100 + 0.1 * i % 0.3 for i in range(20)])
        r = run_backtest(s, df, apply_costs=False)
        assert r.num_trades == 0

    def test_direction_filter_propagates_to_engine(self):
        # Setup that would generate both LONG and SHORT signals over the
        # series. With direction=1, only the LONG fires.
        closes = [100, 100, 100, 100, 100, 102, 101,   # would be SHORT
                  100, 98,  99]                         # would be LONG
        df = _bars_from_closes(closes)
        s = BollingerBandsDirected(length=5, mult=1.0, direction=1)
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        assert all(t.side == "LONG" for t in r.trades)
        assert r.num_trades >= 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "bollinger_bands" in REGISTRY
        assert REGISTRY["bollinger_bands"] is BollingerBandsDirected
