"""Tests for the VWAP + EMA + Volume strategy.

Indicator math (VWAP/EMA/RVOL/ATR) is covered by tests/test_indicators.py;
these tests mock those primitives and probe the strategy's combined-gate
logic + exits + engine integration."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import REGISTRY, SignalKind, VwapEmaVolume, run_backtest
from strategies import vwap_ema_volume as mod

IST = ZoneInfo("Asia/Kolkata")


def _bars(n: int, *, opens=None, closes=None) -> pd.DataFrame:
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
    if closes is None:
        closes = np.full(n, 100.0, dtype=float)
    if opens is None:
        opens = closes
    arr_c = np.asarray(closes, dtype=float)
    arr_o = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {"open": arr_o, "high": arr_c + 1, "low": arr_c - 1,
         "close": arr_c, "volume": np.full(n, 1000.0)},
        index=idx,
    )


def _patch_inds(
    monkeypatch,
    vwap_vals: list[float],
    fast_vals: list[float],
    slow_vals: list[float],
    rvol_vals: list[float],
    atr_vals: list[float],
) -> None:
    """Replace the four indicator primitives with deterministic series."""
    def _as_series(vals, df):
        s = pd.Series(vals, index=df.index, dtype=float)
        assert len(s) == len(df), "mock series length mismatch"
        return s

    monkeypatch.setattr(mod.volume, "vwap",
                        lambda df: _as_series(vwap_vals, df))
    # ema is called twice (fast then slow) on the same df, distinguish by period.
    def fake_ema(df, period=20):
        return _as_series(fast_vals if period == 9 else slow_vals, df)
    monkeypatch.setattr(mod.trend, "ema", fake_ema)
    monkeypatch.setattr(mod.volume, "volume_surge_ratio",
                        lambda df, period=20: _as_series(rvol_vals, df))
    monkeypatch.setattr(mod.volatility, "atr",
                        lambda df, period=14: _as_series(atr_vals, df))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = VwapEmaVolume()
        assert s.fast == 9 and s.slow == 21
        assert s.volume_period == 20 and s.volume_spike == 1.5
        assert s.atr_period == 14 and s.atr_mult == 1.5

    def test_bad_fast(self):
        with pytest.raises(ValueError, match="fast"):
            VwapEmaVolume(fast=1)

    def test_bad_slow(self):
        with pytest.raises(ValueError, match="slow"):
            VwapEmaVolume(fast=10, slow=10)

    def test_bad_volume_spike(self):
        with pytest.raises(ValueError, match="volume_spike"):
            VwapEmaVolume(volume_spike=0)

    def test_bad_atr_mult(self):
        with pytest.raises(ValueError, match="atr_mult"):
            VwapEmaVolume(atr_mult=0)


# ---------------------------------------------------------------------------
# Entry signal
# ---------------------------------------------------------------------------

class TestEntry:
    def test_long_on_vwap_reclaim_with_all_gates(self, monkeypatch):
        # Bar 0: close=99, vwap=100 → below.  Bar 1: close=101, vwap=100 → reclaim.
        df = _bars(2, closes=[99.0, 101.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0],
                    fast_vals=[100.0, 102.0],   # fast > slow at bar 1
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 2.0],        # spike at bar 1
                    atr_vals=[1.0, 1.0])
        s = VwapEmaVolume()
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "VwapEmaVolLE"

    def test_short_on_vwap_rejection_with_all_gates(self, monkeypatch):
        # Bar 0: close=101, vwap=100 → above.  Bar 1: close=99, vwap=100 → reject.
        df = _bars(2, closes=[101.0, 99.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0],
                    fast_vals=[100.0, 99.0],    # fast < slow at bar 1
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 2.0],
                    atr_vals=[1.0, 1.0])
        s = VwapEmaVolume()
        sig = s.signal(df, 1)
        assert sig is not None and sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "VwapEmaVolSE"

    def test_long_blocked_by_low_volume(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0],
                    fast_vals=[100.0, 102.0],
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 1.0],        # no spike
                    atr_vals=[1.0, 1.0])
        assert VwapEmaVolume().signal(df, 1) is None

    def test_long_blocked_by_ema_alignment(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0],
                    fast_vals=[100.0, 99.0],    # fast < slow → blocks long
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 2.0],
                    atr_vals=[1.0, 1.0])
        assert VwapEmaVolume().signal(df, 1) is None

    def test_first_bar_returns_none(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0],
                    fast_vals=[100.0, 102.0],
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 2.0],
                    atr_vals=[1.0, 1.0])
        assert VwapEmaVolume().signal(df, 0) is None

    def test_nan_returns_none(self, monkeypatch):
        df = _bars(2, closes=[99.0, 101.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[float("nan"), 100.0],
                    fast_vals=[100.0, 102.0],
                    slow_vals=[100.0, 101.0],
                    rvol_vals=[1.0, 2.0],
                    atr_vals=[1.0, 1.0])
        assert VwapEmaVolume().signal(df, 1) is None


# ---------------------------------------------------------------------------
# Exit signal
# ---------------------------------------------------------------------------

class TestExit:
    def test_long_exit_on_vwap_cross_back(self, monkeypatch):
        # Bar 0: enter long.  Bar 1: close drops back under VWAP → EXIT.
        df = _bars(3, closes=[99.0, 101.0, 99.5])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0, 100.0],
                    fast_vals=[100.0, 102.0, 102.0],
                    slow_vals=[100.0, 101.0, 101.0],
                    rvol_vals=[1.0, 2.0, 1.0],
                    atr_vals=[1.0, 1.0, 1.0])
        s = VwapEmaVolume()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert "Cross" in sig.reason

    def test_long_exit_on_atr_target_hit(self, monkeypatch):
        # Entry at 101 with ATR=1 ⇒ target = 101 + 1.5*1 = 102.5. Bar 2 close
        # = 103 → above target.
        df = _bars(3, closes=[99.0, 101.0, 103.0])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0, 100.0],
                    fast_vals=[100.0, 102.0, 102.0],
                    slow_vals=[100.0, 101.0, 101.0],
                    rvol_vals=[1.0, 2.0, 1.0],
                    atr_vals=[1.0, 1.0, 1.0])
        s = VwapEmaVolume()
        assert s.signal(df, 1).kind == SignalKind.ENTER_LONG
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT
        assert "Target" in sig.reason

    def test_short_exit_on_vwap_cross_back(self, monkeypatch):
        df = _bars(3, closes=[101.0, 99.0, 100.5])
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0, 100.0],
                    fast_vals=[100.0, 99.0, 99.0],
                    slow_vals=[100.0, 101.0, 101.0],
                    rvol_vals=[1.0, 2.0, 1.0],
                    atr_vals=[1.0, 1.0, 1.0])
        s = VwapEmaVolume()
        assert s.signal(df, 1).kind == SignalKind.ENTER_SHORT
        sig = s.signal(df, 2)
        assert sig is not None and sig.kind == SignalKind.EXIT


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_round_trip(self, monkeypatch):
        # Bar 1: ENTER_LONG  (fills bar 2 open).  Bar 2: EXIT (fills bar 3 open).
        n = 4
        idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [99.0, 99.0, 101.0, 99.5],
            "high":   [100.0, 102.0, 102.0, 100.0],
            "low":    [98.0, 98.5, 100.0, 99.0],
            "close":  [99.0, 101.0, 99.5, 99.5],
            "volume": [1000] * n,
        }, index=idx)
        _patch_inds(monkeypatch,
                    vwap_vals=[100.0, 100.0, 100.0, 100.0],
                    fast_vals=[100.0, 102.0, 102.0, 102.0],
                    slow_vals=[100.0, 101.0, 101.0, 101.0],
                    rvol_vals=[1.0, 2.0, 1.0, 1.0],
                    atr_vals=[1.0, 1.0, 1.0, 1.0])
        r = run_backtest(VwapEmaVolume(), df,
                         max_intraday_loss_pct=50.0, apply_costs=False)
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.side == "LONG"
        # Bar-2 open = 101 entry, bar-3 open = 99.5 exit (EXIT_SIGNAL).
        assert t.entry_price == pytest.approx(101.0)
        assert t.exit_price == pytest.approx(99.5)
        assert t.exit_reason == "EXIT_SIGNAL"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "vwap_ema_volume" in REGISTRY
        assert REGISTRY["vwap_ema_volume"] is VwapEmaVolume
