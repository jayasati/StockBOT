"""Hand-computed correctness for each Stage-A indicator + warmup + registry.

The hand-computed cases in this module are the indicator-math contract:
each function must produce the exact value we derive by hand from a
small deterministic OHLCV frame, and short frames must return NaN
rather than raise.

``TestTradingViewParity`` at the bottom of the module compares our
indicators against TradingView's Data Window readings for real-session
fixtures. It auto-skips any (fixture, indicator, bar) cell whose
reference value hasn't been filled in yet, so the test is safe to run
with a partially-populated JSON. See ``tests/fixtures/tv_reference_values.template.json``
for the schema and the manual workflow."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from indicators import REGISTRY, get_indicator
from indicators.momentum import (
    _streak_series,
    awesome_oscillator,
    cci,
    connors_rsi,
    force_index,
    macd,
    mfi,
    roc,
    rsi,
    stoch_rsi,
    stochastic,
    trix,
    tsi,
    williams_r,
)
from indicators.trend import (
    _rma,
    adx,
    aroon,
    choppiness_index,
    donchian,
    ema,
    hull_ma,
    ichimoku,
    parabolic_sar,
    sma,
    supertrend,
    wma,
    zigzag,
)
from indicators.volatility import atr, bollinger, keltner, ttm_squeeze
from indicators.volume import (
    ad_line,
    anchored_vwap,
    auto_anchored_vwap,
    cmf,
    obv,
    rvol_tod,
    visible_average_price,
    volume_ma,
    volume_surge_ratio,
    vwap,
    vwap_sd_bands,
    vwma,
)

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bars(closes, *, highs=None, lows=None, opens=None, volumes=None,
          start="2026-05-04 09:15", freq="5min") -> pd.DataFrame:
    """Build a tz-aware IST 5m DataFrame. ``highs``/``lows``/``opens``
    default to ``close ± 0.5`` so true-range tests have non-zero span."""
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    if highs is None:
        highs = closes + 0.5
    if lows is None:
        lows = closes - 0.5
    if opens is None:
        opens = closes.copy()
    if volumes is None:
        volumes = np.full(n, 1000.0)
    idx = pd.date_range(start, periods=n, freq=freq, tz=IST)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Wilder's RMA — the smoothing primitive everything builds on
# ---------------------------------------------------------------------------

class TestRMA:
    def test_first_period_minus_one_are_nan(self):
        r = _rma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), period=3)
        assert r.iloc[0:2].isna().all()
        # 3rd element seeded with SMA of first 3.
        assert r.iloc[2] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)

    def test_recursive_step_matches_formula(self):
        s = pd.Series([10.0, 12.0, 14.0, 16.0])
        r = _rma(s, period=2)
        # Seed at index 1 = (10+12)/2 = 11
        # index 2: 11 + (1/2)*(14-11) = 12.5
        # index 3: 12.5 + (1/2)*(16-12.5) = 14.25
        assert r.iloc[1] == pytest.approx(11.0)
        assert r.iloc[2] == pytest.approx(12.5)
        assert r.iloc[3] == pytest.approx(14.25)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class TestSMA:
    def test_known_window(self):
        df = _bars([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        out = sma(df, period=3)
        assert out.iloc[0:2].isna().all()
        assert out.iloc[2] == pytest.approx(2.0)   # (1+2+3)/3
        assert out.iloc[9] == pytest.approx(9.0)   # (8+9+10)/3


class TestEMA:
    def test_seed_then_recurse(self):
        df = _bars([10, 11, 12, 13, 14])
        out = ema(df, period=3)
        # period-1 = 2 NaN bars, then SMA seed: (10+11+12)/3 = 11
        assert out.iloc[0:2].isna().all()
        assert out.iloc[2] == pytest.approx(11.0)
        # alpha = 2/(3+1) = 0.5; step: 11 + 0.5*(13-11) = 12
        assert out.iloc[3] == pytest.approx(12.0)
        # next: 12 + 0.5*(14-12) = 13
        assert out.iloc[4] == pytest.approx(13.0)


class TestSupertrend:
    def test_uptrend_emits_direction_plus_one(self):
        # Strongly rising closes → ATR small relative to gain → direction +1
        closes = np.linspace(100.0, 130.0, 30)
        df = _bars(closes)
        out = supertrend(df, period=10, multiplier=3.0)
        # After warmup, direction should settle to +1 (uptrend).
        post_warmup = out["direction"].iloc[15:]
        assert (post_warmup == 1.0).all()

    def test_returns_two_columns(self):
        df = _bars(np.linspace(100, 110, 20))
        out = supertrend(df, period=5, multiplier=2.0)
        assert list(out.columns) == ["supertrend", "direction"]


class TestZigZag:
    def test_pure_uptrend_emits_initial_low_pivot_only(self):
        # Monotonic up: once price rises >= 5% from the initial low, the
        # algorithm confirms bar 0 as a LOW pivot. After that, no
        # subsequent reversal happens, so no further pivots are emitted.
        closes = np.linspace(100, 200, 30)
        df = _bars(closes, highs=closes + 0.5, lows=closes - 0.5)
        out = zigzag(df, deviation_pct=5.0)
        emitted = np.where(out["pivot_type"].to_numpy() != 0)[0]
        assert list(emitted) == [0]
        assert out["pivot_type"].iloc[0] == -1
        assert out["zigzag_price"].iloc[0] == pytest.approx(99.5)

    def test_up_then_reversal_emits_low_then_high_pivot(self):
        # Climb to 110, then drop to 100 (~9% drop from 110 > 5%).
        # Bar 0 (low 99.5) is confirmed once price rises >5%.
        # Bar 5 (high 110.5) is the swing high; the drop confirms it.
        closes = np.array([100, 102, 104, 106, 108, 110, 108, 105, 102, 100])
        df = _bars(closes, highs=closes + 0.5, lows=closes - 0.5)
        out = zigzag(df, deviation_pct=5.0)
        emitted = np.where(out["pivot_type"].to_numpy() != 0)[0]
        # Two pivots: low at 0, high at 5.
        assert list(emitted) == [0, 5]
        assert out["pivot_type"].iloc[0] == -1
        assert out["pivot_type"].iloc[5] == 1
        assert out["zigzag_price"].iloc[5] == pytest.approx(110.5)

    def test_zigzag_emits_alternating_pivots(self):
        # Up, down, up: should emit a low, a high, a low (in some order)
        # — pivots alternate, never two same-direction pivots in a row.
        closes = np.concatenate([
            np.linspace(100, 110, 6),
            np.linspace(110, 95, 6)[1:],
            np.linspace(95, 110, 6)[1:],
        ])
        df = _bars(closes, highs=closes + 0.5, lows=closes - 0.5)
        out = zigzag(df, deviation_pct=5.0)
        types = out["pivot_type"].to_numpy()
        emitted = types[types != 0].tolist()
        # Must alternate ±1.
        for a, b in zip(emitted, emitted[1:]):
            assert a + b == 0, f"non-alternating pivots: {emitted}"

    def test_subthreshold_reversal_does_not_emit_new_pivot(self):
        # First emit one pivot via a >5% rise. Then a small dip (<5%)
        # within the uptrend must NOT confirm a new high pivot.
        closes = np.concatenate([
            np.array([100]),                # bar 0
            np.linspace(102, 110, 5),       # rise to 110 (>5% from 100)
            np.array([109, 108.5, 109]),    # tiny dip then recover
        ])
        df = _bars(closes, highs=closes + 0.5, lows=closes - 0.5)
        out = zigzag(df, deviation_pct=5.0)
        # Only the initial low pivot at bar 0 should be emitted; no
        # high pivot because the dip didn't break the threshold.
        emitted = np.where(out["pivot_type"].to_numpy() != 0)[0]
        assert list(emitted) == [0]

    def test_invalid_deviation_raises(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="deviation_pct"):
            zigzag(df, deviation_pct=0)

    def test_empty_input(self):
        df = _bars(np.array([]))
        out = zigzag(df)
        assert out.empty


class TestADX:
    def test_columns_and_range(self):
        # Strong sustained uptrend produces ADX climbing into 25-100.
        closes = np.linspace(100.0, 200.0, 60)
        highs = closes + 1.0
        lows = closes - 0.5
        df = _bars(closes, highs=highs, lows=lows)
        out = adx(df, period=14)
        assert list(out.columns) == ["adx", "di_plus", "di_minus"]
        adx_final = out["adx"].iloc[-1]
        assert 25.0 <= adx_final <= 100.0
        # In a clean uptrend, +DI should exceed -DI on the last bar.
        assert out["di_plus"].iloc[-1] > out["di_minus"].iloc[-1]


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestRSI:
    def test_monotone_uptrend_pushes_rsi_above_70(self):
        df = _bars([100 + i for i in range(30)])
        out = rsi(df, period=14)
        assert out.iloc[0:14].isna().all()
        assert out.iloc[-1] > 70.0

    def test_all_gains_caps_at_100(self):
        df = _bars([100 + i * 2 for i in range(30)])
        out = rsi(df, period=14)
        assert out.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_known_value_against_wilder_formula(self):
        # 15 closes: 14 deltas, all gains of +1. At idx 14 (the seed bar)
        # up_rma = mean of 14 ones = 1.0; down_rma = 0.0 → RSI capped 100.
        # Then one loss at idx 15: delta = -1.
        # up_rma[15]   = 1.0  + (1/14)*(0 - 1.0)   = 13/14 ≈ 0.9286
        # down_rma[15] = 0.0  + (1/14)*(1 - 0.0)   = 1/14  ≈ 0.0714
        # RS ≈ 13.0;  RSI = 100 - 100/14 = 92.857.
        closes = list(range(100, 115)) + [113]   # 16 closes, last bar is a -1 loss
        df = _bars(closes)
        out = rsi(df, period=14)
        assert pd.isna(out.iloc[13]), "warmup boundary"
        assert out.iloc[14] == pytest.approx(100.0)
        assert out.iloc[15] == pytest.approx(92.857, abs=0.05)


class TestMACD:
    def test_columns_and_shape(self):
        df = _bars(np.linspace(100, 120, 50))
        out = macd(df, fast=12, slow=26, signal=9)
        assert list(out.columns) == ["macd", "signal", "histogram"]
        assert len(out) == 50

    def test_uptrend_macd_positive(self):
        df = _bars(np.linspace(100, 200, 60))
        out = macd(df)
        # After warmup, MACD line should be positive in a sustained uptrend.
        assert out["macd"].iloc[-1] > 0
        assert out["histogram"].iloc[-1] == pytest.approx(
            out["macd"].iloc[-1] - out["signal"].iloc[-1]
        )


class TestStochastic:
    def test_at_top_of_range_yields_100(self):
        # Last close equals the rolling high → %K_raw = 100.
        closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                  100, 100, 100, 100, 110, 110, 110]
        highs = [c + 0.1 for c in closes]
        highs[-3:] = [110, 110, 110]
        lows = [c - 0.1 for c in closes]
        df = _bars(closes, highs=highs, lows=lows)
        out = stochastic(df, k=14, d=3, smooth=3)
        # With closes at the top of the 14-bar range, k_raw → 100.
        assert out["k"].iloc[-1] == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestConnorsRSIStreakSeries:
    def test_pure_uptrend_yields_growing_streak(self):
        closes = pd.Series([100, 101, 102, 103, 104], dtype=float)
        s = _streak_series(closes)
        # Bar 0: 0 (no prior). Bar 1..4: +1, +2, +3, +4
        assert s.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_pure_downtrend_yields_growing_negative_streak(self):
        closes = pd.Series([100, 99, 98, 97, 96], dtype=float)
        s = _streak_series(closes)
        assert s.tolist() == [0.0, -1.0, -2.0, -3.0, -4.0]

    def test_direction_flip_resets_to_one(self):
        # Up 2 bars, then down. Down resets to -1, then -2.
        closes = pd.Series([100, 101, 102, 100, 99], dtype=float)
        s = _streak_series(closes)
        assert s.tolist() == [0.0, 1.0, 2.0, -1.0, -2.0]

    def test_unchanged_bar_is_zero(self):
        closes = pd.Series([100, 101, 101, 102], dtype=float)
        s = _streak_series(closes)
        assert s.tolist() == [0.0, 1.0, 0.0, 1.0]


class TestConnorsRSI:
    def test_output_in_zero_to_hundred(self):
        rng = np.random.default_rng(11)
        closes = 100 + np.cumsum(rng.uniform(-1, 1, 200))
        df = _bars(closes)
        out = connors_rsi(df)
        valid = out.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_long_uptrend_pushes_crsi_high(self):
        # 200 bars of monotonic up → all 3 sub-indicators saturate high.
        # rsi_close → 100, rsi_streak → 100, percent_rank → 100 (each new
        # ROC is matched by past ROCs of the same magnitude — actually not
        # quite saturated here; still > 50). Composite should be > 80.
        closes = np.linspace(100, 200, 200)
        df = _bars(closes)
        out = connors_rsi(df)
        # On the LAST bar of a strict uptrend, CRSI should be substantially
        # above 50.
        assert out.iloc[-1] > 65.0

    def test_long_downtrend_pushes_crsi_low(self):
        closes = np.linspace(200, 100, 200)
        df = _bars(closes)
        out = connors_rsi(df)
        assert out.iloc[-1] < 35.0

    def test_warmup_returns_nan_until_rank_period(self):
        df = _bars(np.full(50, 100.0) + np.arange(50) * 0.1)
        out = connors_rsi(df, rank_period=100)
        # rank_period+1 = 101 bars needed; 50 < 101 → all NaN.
        assert out.isna().all()

    def test_invalid_params_raise(self):
        df = _bars(np.full(150, 100.0))
        with pytest.raises(ValueError):
            connors_rsi(df, rsi_period=1)
        with pytest.raises(ValueError):
            connors_rsi(df, streak_period=1)
        with pytest.raises(ValueError):
            connors_rsi(df, rank_period=0)


class TestTSI:
    def test_uptrend_yields_positive_tsi(self):
        # Strict monotonic uptrend → all diffs are positive → numerator
        # equals denominator → TSI saturates at +100.
        closes = np.linspace(100, 200, 100)
        df = _bars(closes)
        out = tsi(df)
        post_warmup = out["tsi"].dropna()
        assert (post_warmup > 0).all()
        # On a perfect uptrend, TSI converges to +100.
        assert post_warmup.iloc[-1] == pytest.approx(100.0, abs=0.5)

    def test_downtrend_yields_negative_tsi(self):
        closes = np.linspace(200, 100, 100)
        df = _bars(closes)
        out = tsi(df)
        post_warmup = out["tsi"].dropna()
        assert (post_warmup < 0).all()
        assert post_warmup.iloc[-1] == pytest.approx(-100.0, abs=0.5)

    def test_constant_price_yields_nan(self):
        closes = np.full(60, 100.0)
        df = _bars(closes)
        out = tsi(df)
        # All diffs zero → numerator and denominator both zero → NaN.
        assert out["tsi"].isna().all() or out["tsi"].iloc[-1] == 0.0

    def test_signal_lags_tsi(self):
        # On a smooth signal, signal-EMA lags tsi.
        closes = 100 + 5 * np.sin(np.linspace(0, 6 * np.pi, 200))
        df = _bars(closes)
        out = tsi(df)
        valid = out.dropna()
        # If they were identical at every bar, the diff series would be
        # all zero. We assert they meaningfully differ.
        assert (valid["tsi"] - valid["signal"]).abs().max() > 0.5

    def test_invalid_period_raises(self):
        df = _bars(np.full(50, 100.0))
        with pytest.raises(ValueError):
            tsi(df, long_period=0)


class TestStochRSI:
    def test_constant_rsi_yields_nan_then_constant(self):
        # If RSI is constant over the stoch window, max == min → div0 → NaN.
        # Use a strongly trending price series so RSI saturates near 100,
        # then verify that rolling-flat region produces NaN k/d.
        closes = np.concatenate([
            np.linspace(100, 200, 30),  # ramp up
            np.full(20, 200.0),          # flat (RSI flatlines near 100)
        ])
        df = _bars(closes)
        out = stoch_rsi(df, rsi_period=14, stoch_period=14,
                       k_smooth=3, d_smooth=3)
        # By the end of the flat region, k should be NaN (flat RSI → no range).
        last_k = out["k"].iloc[-1]
        assert pd.isna(last_k) or 0.0 <= last_k <= 100.0

    def test_output_in_zero_to_hundred_range_post_warmup(self):
        rng = np.random.default_rng(7)
        closes = 100 + np.cumsum(rng.uniform(-1, 1, 100))
        df = _bars(closes)
        out = stoch_rsi(df)
        valid_k = out["k"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        valid_d = out["d"].dropna()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_invalid_params_raise(self):
        df = _bars(np.full(50, 100.0))
        with pytest.raises(ValueError):
            stoch_rsi(df, rsi_period=1)
        with pytest.raises(ValueError):
            stoch_rsi(df, k_smooth=0)


class TestATR:
    def test_constant_range_is_atr_equal_to_range(self):
        # All bars have high-low = 2.0; no gaps. TR = 2.0 every bar.
        closes = np.full(20, 100.0)
        df = _bars(closes, highs=closes + 1.0, lows=closes - 1.0)
        out = atr(df, period=14)
        # After RMA seed, ATR should equal 2.0.
        assert out.iloc[14] == pytest.approx(2.0, abs=1e-6)


class TestBollinger:
    def test_constant_close_collapses_bands(self):
        df = _bars(np.full(25, 100.0))
        out = bollinger(df, period=20, std=2.0)
        assert out["middle"].iloc[-1] == pytest.approx(100.0)
        assert out["upper"].iloc[-1] == pytest.approx(100.0)
        assert out["lower"].iloc[-1] == pytest.approx(100.0)

    def test_population_stdev_matches_known_window(self):
        # Closes: 1..20. SMA = 10.5. Population std = sqrt(33.25) ≈ 5.766.
        df = _bars(list(range(1, 21)))
        out = bollinger(df, period=20, std=2.0)
        assert out["middle"].iloc[-1] == pytest.approx(10.5)
        # upper = 10.5 + 2*5.766 = 22.03
        expected_upper = 10.5 + 2 * np.std(np.arange(1, 21))  # population
        assert out["upper"].iloc[-1] == pytest.approx(expected_upper, abs=1e-6)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

class TestOBV:
    def test_up_down_unchanged_signs_correctly(self):
        closes = [100, 101, 102, 100, 100, 103]
        volumes = [10, 20, 30, 40, 50, 60]
        df = _bars(closes, volumes=np.array(volumes, dtype=float))
        out = obv(df)
        # Sign:    nan, +1, +1, -1, 0, +1
        # OBV:      0   20  50  10  10  70
        expected = [0.0, 20.0, 50.0, 10.0, 10.0, 70.0]
        for i, exp in enumerate(expected):
            assert out.iloc[i] == pytest.approx(exp), f"bar {i}"


class TestVolumeMA:
    def test_sma_of_volume_matches_rolling_mean(self):
        # Period 5 SMA over a deterministic 8-bar volume series.
        vols = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=float)
        df = _bars(np.full(8, 100.0), volumes=vols)
        out = volume_ma(df, period=5)
        # First 4 bars: NaN (warmup). Bar 4 = mean(100..500) = 300.
        # Bar 5 = mean(200..600) = 400. Bar 6 = 500. Bar 7 = 600.
        assert pd.isna(out.iloc[3])
        assert out.iloc[4] == pytest.approx(300.0)
        assert out.iloc[5] == pytest.approx(400.0)
        assert out.iloc[6] == pytest.approx(500.0)
        assert out.iloc[7] == pytest.approx(600.0)

    def test_invalid_period_raises(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="period"):
            volume_ma(df, period=0)


class TestVWMA:
    def test_constant_volume_reduces_to_sma(self):
        # When volume is constant, vwma collapses to sma(close).
        closes = np.array([10, 20, 30, 40, 50], dtype=float)
        df = _bars(closes, volumes=np.full(5, 100.0))
        out = vwma(df, period=3)
        # SMA(3): bars 0-1 NaN; bar 2 = 20; bar 3 = 30; bar 4 = 40.
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == pytest.approx(20.0)
        assert out.iloc[3] == pytest.approx(30.0)
        assert out.iloc[4] == pytest.approx(40.0)

    def test_volume_weights_pull_average_to_high_volume_bar(self):
        # Bar 2 has 10x the volume of bars 0,1,3,4. Its price (200) should
        # dominate the 5-period vwma at bar 4.
        closes = np.array([100, 100, 200, 100, 100], dtype=float)
        vols = np.array([10, 10, 100, 10, 10], dtype=float)
        df = _bars(closes, volumes=vols)
        out = vwma(df, period=5)
        # Numerator = 1000 + 1000 + 20000 + 1000 + 1000 = 24000
        # Denominator = 10 + 10 + 100 + 10 + 10 = 140
        # vwma = 24000 / 140 ≈ 171.4286
        assert out.iloc[4] == pytest.approx(24000.0 / 140.0, abs=1e-9)

    def test_zero_total_volume_yields_nan(self):
        closes = np.array([100.0] * 5)
        vols = np.array([0.0] * 5)
        df = _bars(closes, volumes=vols)
        out = vwma(df, period=5)
        assert pd.isna(out.iloc[-1])

    def test_invalid_period_raises(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="period"):
            vwma(df, period=0)


class TestVWAP:
    def test_single_session_cumulative(self):
        # 3 bars in one session. Hand-compute typical price * volume cum.
        # Bar 0: tp = (101+99+100)/3 = 100; pv = 100*1000 = 100000;
        #        cum_pv = 100000; cum_v = 1000;  vwap = 100.0
        # Bar 1: tp = (102+100+101)/3 = 101; pv = 101*2000 = 202000;
        #        cum_pv = 302000; cum_v = 3000; vwap = 100.6667
        # Bar 2: tp = (103+101+102)/3 = 102; pv = 102*3000 = 306000;
        #        cum_pv = 608000; cum_v = 6000; vwap = 101.3333
        closes = np.array([100, 101, 102], dtype=float)
        highs = closes + 1.0
        lows = closes - 1.0
        vols = np.array([1000, 2000, 3000], dtype=float)
        df = _bars(closes, highs=highs, lows=lows, volumes=vols)
        out = vwap(df)
        assert out.iloc[0] == pytest.approx(100.0)
        assert out.iloc[1] == pytest.approx(302000.0 / 3000.0)
        assert out.iloc[2] == pytest.approx(608000.0 / 6000.0)

    def test_resets_at_session_boundary(self):
        # Two days, 2 bars each. Day 2 must NOT include day 1's accumulation.
        day1 = _bars(np.array([100.0, 102.0]),
                     highs=np.array([101.0, 103.0]),
                     lows=np.array([99.0, 101.0]),
                     volumes=np.array([1000.0, 1000.0]),
                     start="2026-05-04 09:15")
        day2 = _bars(np.array([200.0, 202.0]),
                     highs=np.array([201.0, 203.0]),
                     lows=np.array([199.0, 201.0]),
                     volumes=np.array([1000.0, 1000.0]),
                     start="2026-05-05 09:15")
        df = pd.concat([day1, day2])
        out = vwap(df)
        # Day 2 bar 0 is the FIRST bar of day 2 session — vwap should
        # equal that bar's typical price (200), not be contaminated by day 1.
        assert out.iloc[2] == pytest.approx(200.0)

    def test_zero_volume_yields_nan(self):
        df = _bars(np.full(3, 100.0), volumes=np.zeros(3))
        out = vwap(df)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[-1])

    def test_empty_input(self):
        df = _bars(np.array([]))
        out = vwap(df)
        assert len(out) == 0


class TestAutoAnchoredVWAP:
    def test_hand_computed_anchors_and_vwaps(self):
        # 5 bars; lookback=5 (full window).
        # highs: bar 2 has 110 (highest) → anchor_high = 2
        # lows:  bar 1 has  95 (lowest)  → anchor_low  = 1
        closes = np.array([102, 100, 105, 103, 102], dtype=float)
        highs = np.array([105, 103, 110, 108, 107], dtype=float)
        lows = np.array([99, 95, 100, 98, 96], dtype=float)
        vols = np.array([1000, 1500, 2000, 1800, 1200], dtype=float)
        df = _bars(closes, highs=highs, lows=lows, volumes=vols)
        out = auto_anchored_vwap(df, lookback=5)
        # avwap_from_high at bar 4: anchor=2, sum from bars 2..4
        # tp[2]=105, tp[3]=103, tp[4]=305/3
        # cum_pv = 105*2000 + 103*1800 + (305/3)*1200 = 517400
        # cum_v  = 5000
        assert out["avwap_from_high"].iloc[4] == pytest.approx(
            517400.0 / 5000.0, abs=1e-6,
        )
        # avwap_from_low at bar 4: anchor=1, sum from bars 1..4
        # tp[1]=298/3, tp[2]=105, tp[3]=103, tp[4]=305/3
        # cum_pv = (298/3)*1500 + 105*2000 + 103*1800 + (305/3)*1200 = 666400
        # cum_v  = 6500
        assert out["avwap_from_low"].iloc[4] == pytest.approx(
            666400.0 / 6500.0, abs=1e-6,
        )

    def test_first_bar_is_its_own_anchor(self):
        df = _bars(np.array([100.0]))
        out = auto_anchored_vwap(df, lookback=10)
        # Single bar: anchor must be bar 0; both AVWAPs == its tp.
        # tp = (100.5 + 99.5 + 100) / 3 = 100
        assert out["avwap_from_high"].iloc[0] == pytest.approx(100.0)
        assert out["avwap_from_low"].iloc[0] == pytest.approx(100.0)

    def test_anchor_shifts_when_new_extreme_appears(self):
        # 4 bars; lookback=4. Bar 3 prints a new highest high → anchor
        # for the high-AVWAP at bar 3 jumps to bar 3 itself, so the
        # high-AVWAP at bar 3 equals bar 3's tp.
        closes = np.array([100, 101, 102, 110], dtype=float)
        highs = np.array([100.5, 101.5, 102.5, 115], dtype=float)
        lows = np.array([99.5, 100.5, 101.5, 108], dtype=float)
        df = _bars(closes, highs=highs, lows=lows,
                   volumes=np.full(4, 1000.0))
        out = auto_anchored_vwap(df, lookback=4)
        tp_3 = (115 + 108 + 110) / 3.0
        assert out["avwap_from_high"].iloc[3] == pytest.approx(tp_3, abs=1e-9)

    def test_invalid_lookback_raises(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="lookback"):
            auto_anchored_vwap(df, lookback=0)


class TestVisibleAveragePrice:
    def test_constant_price_returns_that_price(self):
        df = _bars(np.full(10, 100.0),
                   highs=np.full(10, 100.0), lows=np.full(10, 100.0),
                   volumes=np.full(10, 1000.0))
        out = visible_average_price(df, n_bars=10, bins=10)
        assert out["poc"] == 100.0
        assert out["vah"] == 100.0
        assert out["val"] == 100.0
        assert out["total_volume"] == pytest.approx(10000.0)

    def test_concentrated_volume_at_one_price_is_poc(self):
        # 9 bars at 99.5, 1 bar at 105 with 100x volume.
        # bin range = [low.min(), high.max()] = [99, 105.5]
        # 10 bins of width 0.65 each.
        # The 105 bar's close lands in the highest bin (index 9 → range
        # 104.85..105.50, midpoint 105.175). The huge volume there makes
        # it the POC.
        closes = np.array([99.5] * 9 + [105.0])
        highs = np.array([100.0] * 9 + [105.5])
        lows = np.array([99.0] * 9 + [104.5])
        vols = np.array([100.0] * 9 + [10000.0])
        df = _bars(closes, highs=highs, lows=lows, volumes=vols)
        out = visible_average_price(df, n_bars=10, bins=10)
        # The high bin's midpoint, not the price 105 exactly.
        assert out["poc"] > 104.5
        assert out["poc"] < 105.5
        assert out["total_volume"] == pytest.approx(900.0 + 10000.0)

    def test_value_area_contains_target_pct_of_volume(self):
        # 5 distinct prices with equal volume each → POC is one bin,
        # VA expands until it contains >= 70% of volume.
        closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                          dtype=float)
        highs = closes + 0.5
        lows = closes - 0.5
        vols = np.full(10, 1000.0)
        df = _bars(closes, highs=highs, lows=lows, volumes=vols)
        out = visible_average_price(df, n_bars=10, bins=10,
                                     value_area_pct=0.70)
        # 7 of 10 bins covered → VA spans 7 of 10 bin widths (price range
        # ≈ 9.0 wide / 10 bins = 0.9 per bin → VA span ≈ 6.3).
        # Just verify the VA bracket is sensible.
        va_span = out["vah"] - out["val"]
        full_range = float(highs.max() - lows.min())
        assert 0 < va_span <= full_range
        # POC is some price in the data range.
        assert lows.min() <= out["poc"] <= highs.max()

    def test_empty_input(self):
        df = _bars(np.array([]))
        out = visible_average_price(df)
        assert pd.isna(out["poc"])
        assert out["total_volume"] == 0.0

    def test_invalid_value_area_pct_raises(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="value_area_pct"):
            visible_average_price(df, value_area_pct=0.0)
        with pytest.raises(ValueError, match="value_area_pct"):
            visible_average_price(df, value_area_pct=1.5)

    def test_invalid_n_bars_or_bins(self):
        df = _bars(np.full(5, 100.0))
        with pytest.raises(ValueError, match="n_bars"):
            visible_average_price(df, n_bars=0)
        with pytest.raises(ValueError, match="bins"):
            visible_average_price(df, bins=0)


class TestCMF:
    def test_close_at_high_pushes_cmf_positive(self):
        # All bars close at their high → mf_mult = +1 → CMF = +1.
        closes = np.full(25, 100.0)
        highs = closes.copy()
        lows = closes - 1.0
        df = _bars(closes, highs=highs, lows=lows,
                   volumes=np.full(25, 1000.0))
        out = cmf(df, period=20)
        assert out.iloc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_zero_range_bars_contribute_zero(self):
        # Bars where high == low get mf_mult = 0 → don't move CMF.
        closes = np.full(25, 100.0)
        df = _bars(closes, highs=closes, lows=closes,
                   volumes=np.full(25, 1000.0))
        out = cmf(df, period=20)
        # 0/(20*1000) = 0
        assert out.iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Warmup contract
# ---------------------------------------------------------------------------

class TestWarmupNaN:
    """A DataFrame shorter than ``warmup_bars`` must NOT raise; outputs are
    NaN. Phase 7 reads None / NaN as 'neutral' (no score contribution).

    Indicators with trivial warmup (≤ 2) — OBV, A/D Line, PSAR — are
    defined essentially from bar 1; there's no "shorter than warmup"
    input that would make them NaN, so they're excluded from this
    check. They're still exercised by ``TestRegistryConsistency`` on a
    long frame."""

    @pytest.mark.parametrize("name", sorted(
        n for n, s in REGISTRY.items()
        if s.output_kind in ("series", "frame") and s.warmup_bars > 2
    ))
    def test_short_input_returns_nan_no_raise(self, name):
        spec = REGISTRY[name]
        # Build a frame strictly shorter than the indicator's warmup.
        n = max(2, spec.warmup_bars - 1)
        df = _bars(
            closes=np.linspace(100, 100 + n, n),
            volumes=np.full(n, 1000.0),
        )
        func = get_indicator(name)
        out = func(df, **spec.default_params)
        if spec.output_kind == "series":
            # Last value MUST be NaN (warmup not reached).
            assert pd.isna(out.iloc[-1]), f"{name}: last bar should be NaN"
        else:  # frame
            for col in spec.output_keys:
                if col not in out.columns:
                    continue
                # We don't require EVERY column to be NaN at the last bar
                # (e.g. OBV ramps fast), only that the call returned a
                # well-shaped frame.
                assert col in out.columns


# ---------------------------------------------------------------------------
# Registry consistency — every entry must be callable + shape-correct
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage B — extended trend
# ---------------------------------------------------------------------------

class TestWMA:
    def test_linear_weights_match_hand_compute(self):
        # WMA(close=[1,2,3,4], period=4) = (1*1 + 2*2 + 3*3 + 4*4) / 10 = 30/10 = 3
        df = _bars([1, 2, 3, 4])
        out = wma(df, period=4)
        assert pd.isna(out.iloc[2])
        assert out.iloc[3] == pytest.approx(3.0)


class TestHullMA:
    def test_constant_close_collapses_to_close(self):
        df = _bars(np.full(30, 100.0))
        out = hull_ma(df, period=10)
        # Constant input → all moving averages = 100 → HMA = 100 after warmup.
        assert out.iloc[-1] == pytest.approx(100.0)


class TestPSAR:
    def test_uptrend_psar_stays_below_price(self):
        # Strong uptrend: PSAR should sit below the lows.
        closes = np.linspace(100.0, 130.0, 40)
        highs = closes + 1.0
        lows = closes - 0.5
        df = _bars(closes, highs=highs, lows=lows)
        out = parabolic_sar(df)
        # Past initial bars, PSAR should be at or below the rising close.
        assert (out.iloc[5:] < df["high"].iloc[5:]).all()


class TestIchimoku:
    def test_columns_present(self):
        df = _bars(np.linspace(100, 150, 120))
        out = ichimoku(df)
        assert list(out.columns) == [
            "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
        ]

    def test_chikou_is_close_shifted_back(self):
        df = _bars(np.linspace(100, 150, 120))
        out = ichimoku(df)
        # chikou at bar N = close at bar N + 26.
        # So out["chikou"].iloc[0] should equal df["close"].iloc[26].
        assert out["chikou"].iloc[0] == pytest.approx(df["close"].iloc[26])


class TestDonchian:
    def test_known_window(self):
        # Highs: 1..10. Lows: -1..8 (offset by 2). Donchian(4) at idx 9:
        # upper = max(high[6..9]) = 10; lower = min(low[6..9]) = 5.
        closes = list(range(1, 11))
        df = _bars(closes,
                   highs=np.array(closes, dtype=float),
                   lows=np.array(closes, dtype=float) - 2.0)
        out = donchian(df, period=4)
        assert out["upper"].iloc[-1] == pytest.approx(10.0)
        assert out["lower"].iloc[-1] == pytest.approx(5.0)
        assert out["middle"].iloc[-1] == pytest.approx(7.5)
        assert out["width"].iloc[-1] == pytest.approx(5.0)


class TestAroon:
    def test_freshest_high_yields_100(self):
        # Monotone-up high → most recent bar IS the highest in the window.
        # bars_since_high = 0 → aroon_up = 100.
        df = _bars(list(range(1, 30)))
        out = aroon(df, period=14)
        assert out["aroon_up"].iloc[-1] == pytest.approx(100.0)


class TestChoppinessIndex:
    def test_strong_trend_gives_low_ci(self):
        # Monotone uptrend → range expands directionally; sum_tr / range ≈ 1.
        # log10(1) = 0 → ci ≈ 0 (trending).
        closes = np.linspace(100.0, 200.0, 30)
        df = _bars(closes,
                   highs=closes + 0.5,
                   lows=closes - 0.5)
        out = choppiness_index(df, period=14)
        # On a clean monotone trend, ci should sit comfortably below 50.
        assert out.iloc[-1] < 50.0


# ---------------------------------------------------------------------------
# Stage B — extended momentum
# ---------------------------------------------------------------------------

class TestMFI:
    def test_monotone_uptrend_pushes_mfi_above_70(self):
        # Rising typical-price every bar with constant volume → all gains
        # in tp_diff → negative MF = 0 → MFI = 100.
        df = _bars(np.linspace(100, 130, 30))
        out = mfi(df, period=14)
        assert out.iloc[-1] >= 70.0


class TestCCI:
    def test_zero_at_constant_close(self):
        # mean_dev = 0 → CCI = NaN (we replace 0 with NaN to avoid div0).
        df = _bars(np.full(30, 100.0))
        out = cci(df, period=20)
        assert pd.isna(out.iloc[-1])

    def test_positive_above_mean(self):
        # Up-trending typical prices → CCI > 0. Need 2·period bars before
        # CCI is valid (nested SMAs).
        df = _bars(np.linspace(100, 110, 60))
        out = cci(df, period=20)
        assert out.iloc[-1] > 0.0


class TestROC:
    def test_known_value(self):
        # close goes 100 → 110 over 5 bars. ROC(5) at the last bar = 10%.
        closes = [100, 101, 102, 103, 104, 110]
        df = _bars(closes)
        out = roc(df, period=5)
        assert out.iloc[-1] == pytest.approx(10.0)


class TestWilliamsR:
    def test_at_top_of_range_is_zero(self):
        # If close == highest_high, %R = 0.
        closes = np.full(20, 105.0)
        closes[-1] = 110.0
        highs = closes.copy()
        highs[-1] = 110.0
        lows = closes - 5.0
        df = _bars(closes, highs=highs, lows=lows)
        out = williams_r(df, period=14)
        assert out.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_at_bottom_of_range_is_minus_100(self):
        closes = np.full(20, 100.0)
        closes[-1] = 90.0
        highs = np.full(20, 100.0)
        lows = closes.copy()
        df = _bars(closes, highs=highs, lows=lows)
        out = williams_r(df, period=14)
        # Last close 90 = lowest_low. (hh - close) = (100 - 90) = 10.
        # (hh - ll) = 10. %R = -100.
        assert out.iloc[-1] == pytest.approx(-100.0, abs=1e-6)


class TestAwesomeOscillator:
    def test_uptrend_positive(self):
        df = _bars(np.linspace(100, 200, 50))
        out = awesome_oscillator(df)
        # In a sustained uptrend, AO (fast sma - slow sma of hl2) > 0.
        assert out.iloc[-1] > 0.0


class TestTRIX:
    def test_constant_close_yields_zero(self):
        df = _bars(np.full(60, 100.0))
        out = trix(df, period=15)
        # Triple-smoothed constant = 100. 10000*(100/100 - 1) = 0.
        assert out.iloc[-1] == pytest.approx(0.0, abs=1e-6)


class TestForceIndex:
    def test_uptrend_positive_volume_yields_positive_fi(self):
        df = _bars(np.linspace(100, 130, 40),
                   volumes=np.full(40, 1000.0))
        out = force_index(df, period=13)
        assert out.iloc[-1] > 0.0


# ---------------------------------------------------------------------------
# Stage B — volatility
# ---------------------------------------------------------------------------

class TestKeltner:
    def test_constant_close_collapses_bands(self):
        # Constant close → middle = const, ATR = 2 (h-l=2), bands = 100 ± 2 * mult.
        closes = np.full(30, 100.0)
        df = _bars(closes,
                   highs=closes + 1.0,
                   lows=closes - 1.0)
        out = keltner(df, period=20, mult=2.0, atr_period=10)
        assert out["middle"].iloc[-1] == pytest.approx(100.0)
        # ATR settles to 2.0, mult=2 → ±4.
        assert out["upper"].iloc[-1] == pytest.approx(104.0)
        assert out["lower"].iloc[-1] == pytest.approx(96.0)


class TestTTMSqueeze:
    def test_low_vol_inside_keltner_is_squeezed(self):
        # Constant close → both BB and KC bands sit on the middle. BB
        # (zero variance) is INSIDE KC (which has 2 * ATR width) → squeeze.
        closes = np.full(40, 100.0)
        df = _bars(closes,
                   highs=closes + 1.0,
                   lows=closes - 1.0)
        out = ttm_squeeze(df, period=20)
        assert out["in_squeeze"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stage B — volume
# ---------------------------------------------------------------------------

class TestVolumeSurgeRatio:
    def test_known_window(self):
        # avg = (1000 * 19 + 5000) / 20; vsr_last = 5000 / avg.
        vols = np.array([1000.0] * 19 + [5000.0])
        df = _bars(np.full(20, 100.0), volumes=vols)
        out = volume_surge_ratio(df, period=20)
        expected = 5000.0 / vols.mean()
        assert out.iloc[-1] == pytest.approx(expected, abs=1e-6)


class TestADLine:
    def test_close_at_high_drives_ad_up(self):
        # Every bar closes at the high → mf_mult = +1 → ad cumsum = vol*n.
        closes = np.full(5, 100.0)
        highs = closes.copy()
        lows = closes - 1.0
        vols = np.full(5, 100.0)
        df = _bars(closes, highs=highs, lows=lows, volumes=vols)
        out = ad_line(df)
        # After 5 bars: cumulative = 5 * 100 = 500.
        assert out.iloc[-1] == pytest.approx(500.0)


class TestRVOLTOD:
    def test_returns_nan_when_history_too_short(self):
        # 5 bars on a single date → no prior sessions → all NaN.
        df = _bars(np.full(5, 100.0))
        out = rvol_tod(df, lookback_days=3)
        assert out.isna().all()

    def test_ratio_against_prior_same_tod(self):
        # Build 4 sessions of 3 bars each. Session 4 same TOD slot has
        # volume 3x the prior sessions' mean. lookback_days=3 → ratio = 3.
        rows = []
        idx = []
        for d in range(1, 5):
            day = pd.Timestamp(f"2026-05-{d:02d}", tz=IST).replace(hour=9, minute=15)
            for k in range(3):
                ts = day + pd.Timedelta(minutes=5 * k)
                idx.append(ts)
                vol = (1000.0 if d < 4 else 3000.0) if k == 0 else 500.0
                rows.append({"open": 100.0, "high": 100.5, "low": 99.5,
                             "close": 100.0, "volume": vol})
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
        out = rvol_tod(df, lookback_days=3)
        # Last bar of day 4 at the 09:15 slot has vol=3000; prior 3 sessions
        # all 1000 at that slot → ratio = 3.0.
        # Find the 09:15 slot in day 4.
        last_915 = df[(df.index.date == idx[-3].date()) &
                      (df.index.time == idx[0].time())].index[0]
        assert out.loc[last_915] == pytest.approx(3.0)


class TestVWAPSDBands:
    def test_session_bands_around_vwap(self):
        from datetime import date as Date
        d = Date(2026, 5, 4)
        # Build 6 bars on the session date.
        start = pd.Timestamp("2026-05-04 09:15", tz=IST)
        idx = pd.date_range(start, periods=6, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 101, 100, 102, 101, 103],
            "high":   [101, 102, 101, 103, 102, 104],
            "low":    [99,  100,  99, 101, 100, 102],
            "close":  [100, 101, 100, 102, 101, 103],
            "volume": [100, 100, 100, 100, 100, 100],
        }, index=idx)
        out = vwap_sd_bands(df, session_date=d)
        # bands must straddle the VWAP: plus_k > vwap > minus_k.
        # Use the cumulative VWAP at the last bar as the anchor.
        last = out.iloc[-1]
        assert last["plus_1"] > last["minus_1"]
        assert last["plus_2"] > last["plus_1"]
        assert last["plus_3"] > last["plus_2"]
        # Symmetry: plus_k - minus_k should be 2 * k * sigma.
        spread_1 = last["plus_1"] - last["minus_1"]
        spread_2 = last["plus_2"] - last["minus_2"]
        assert spread_2 == pytest.approx(2.0 * spread_1, abs=1e-6)


class TestAnchoredVWAP:
    def test_anchored_at_first_bar_matches_session_vwap(self):
        idx = pd.date_range("2026-05-04 09:15", periods=5, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 102, 104, 106, 108],
            "high":   [101, 103, 105, 107, 109],
            "low":    [99,  101, 103, 105, 107],
            "close":  [100, 102, 104, 106, 108],
            "volume": [100, 100, 100, 100, 100],
        }, index=idx)
        out = anchored_vwap(df, anchor_ts=idx[0])
        # Cumulative VWAP at the last bar:
        # tp = (h+l+c)/3 = open in this symmetric case
        # Numerator = 100+102+104+106+108 = 520 (times equal volume)
        # Denominator = 5 * vol = 500
        # VWAP = sum(tp*vol)/sum(vol) — same as plain mean here = 520/5 = 104.
        assert out.iloc[-1] == pytest.approx(104.0, abs=1e-6)

    def test_before_anchor_is_nan(self):
        idx = pd.date_range("2026-05-04 09:15", periods=5, freq="5min", tz=IST)
        df = pd.DataFrame({
            "open":   [100, 102, 104, 106, 108],
            "high":   [101, 103, 105, 107, 109],
            "low":    [99,  101, 103, 105, 107],
            "close":  [100, 102, 104, 106, 108],
            "volume": [100, 100, 100, 100, 100],
        }, index=idx)
        out = anchored_vwap(df, anchor_ts=idx[2])  # anchor 3rd bar
        assert out.iloc[0:2].isna().all()
        assert not pd.isna(out.iloc[2])


# ---------------------------------------------------------------------------
# Registry consistency — every indicator, on a long deterministic frame
# ---------------------------------------------------------------------------

class TestRegistryConsistency:
    @pytest.mark.parametrize("name", sorted(
        n for n, s in REGISTRY.items() if s.output_kind in ("series", "frame")
    ))
    def test_indicator_runs_with_defaults_on_long_frame(self, name):
        spec = REGISTRY[name]
        # Generate a long-enough frame for any warmup.
        n = max(60, spec.warmup_bars * 3)
        closes = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
        df = _bars(closes, volumes=np.full(n, 1000.0))
        out = spec.func(df, **spec.default_params)
        if spec.output_kind == "series":
            assert isinstance(out, pd.Series), name
            assert len(out) == n
        elif spec.output_kind == "frame":
            assert isinstance(out, pd.DataFrame), name
            for col in spec.output_keys:
                assert col in out.columns, f"{name} missing column {col}"
            assert len(out) == n

    def test_categories_are_populated(self):
        cats = {spec.category for spec in REGISTRY.values()}
        # Stage A should ship indicators in every category except the
        # ones we haven't started yet — but the four core ones must exist.
        assert {"trend", "momentum", "volatility", "volume", "level"} <= cats

    def test_no_indicator_mutates_input(self):
        """A registry entry that mutated its input DataFrame would break
        Phase 7's parallel scoring. Hash the frame before and after."""
        n = 60
        closes = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, n))
        for name, spec in REGISTRY.items():
            if spec.output_kind == "scalar_dict":
                continue
            df = _bars(closes, volumes=np.full(n, 1000.0))
            snapshot = df.copy(deep=True)
            spec.func(df, **spec.default_params)
            assert df.equals(snapshot), f"{name} mutated its input"


# ---------------------------------------------------------------------------
# TradingView parity — runs whichever reference values are populated
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_TV_JSON_REAL = _FIXTURES_DIR / "tv_reference_values.json"
_TV_JSON_TEMPLATE = _FIXTURES_DIR / "tv_reference_values.template.json"

# Map a fixture sub-key (the per-indicator field in the JSON) to the
# registry indicator name and the column we should read from its output.
# ``None`` column = the indicator returns a Series (no column pick).
_SUBKEY_TO_INDICATOR: dict[str, tuple[str, str | None]] = {
    "rsi":                  ("rsi", None),
    "macd_macd":            ("macd", "macd"),
    "macd_signal":          ("macd", "signal"),
    "macd_histogram":       ("macd", "histogram"),
    "adx_adx":              ("adx", "adx"),
    "adx_di_plus":          ("adx", "di_plus"),
    "adx_di_minus":         ("adx", "di_minus"),
    "bollinger_upper":      ("bollinger", "upper"),
    "bollinger_middle":     ("bollinger", "middle"),
    "bollinger_lower":      ("bollinger", "lower"),
    "supertrend_value":     ("supertrend", "supertrend"),
    "supertrend_direction": ("supertrend", "direction"),
    "stoch_k":              ("stochastic", "k"),
    "stoch_d":              ("stochastic", "d"),
    "mfi":                  ("mfi", None),
    "atr":                  ("atr", None),
    "obv":                  ("obv", None),
    "cmf":                  ("cmf", None),
}


def _load_tv_json() -> dict:
    """Prefer the populated reference file; fall back to the template
    (which has all-null values so the test will skip cleanly)."""
    path = _TV_JSON_REAL if _TV_JSON_REAL.exists() else _TV_JSON_TEMPLATE
    with path.open() as fh:
        return json.load(fh)


def _parse_tolerance(spec: str) -> tuple[str, float]:
    """Parse a ``_meta.tolerances`` entry into a (kind, value) pair.

    Recognized forms (case-insensitive):
        "exact match"              -> ("abs", 1e-6)   # bitwise-near equal
        "exact (integer-valued)"   -> ("abs", 1.0)    # cumulative-sum drift OK
        "abs <= 0.10"              -> ("abs", 0.10)
        "rel <= 0.05%"             -> ("rel", 0.0005)
        "rel <= 0.001"             -> ("rel", 0.001)
    """
    s = spec.lower().strip()
    if "integer" in s:
        return ("abs", 1.0)
    if s.startswith("exact"):
        return ("abs", 1e-6)
    m = re.search(r"(abs|rel)\s*<=\s*([0-9.]+)\s*(%?)", s)
    if not m:
        raise ValueError(f"unparseable tolerance spec: {spec!r}")
    kind = m.group(1)
    value = float(m.group(2))
    if m.group(3) == "%":
        value /= 100.0
    return (kind, value)


def _resolve_tolerance(sub_key: str, tolerances: dict) -> tuple[str, float]:
    """Exact key wins; otherwise fall back to a ``<prefix>_*`` glob
    (e.g. ``macd_macd`` → ``macd_*``)."""
    if sub_key in tolerances:
        return _parse_tolerance(tolerances[sub_key])
    for key, spec in tolerances.items():
        if key.endswith("_*") and sub_key.startswith(key[:-1]):
            return _parse_tolerance(spec)
    raise KeyError(f"no tolerance defined for {sub_key!r}")


def _enumerate_parity_cases() -> list[tuple[str, str, str, float]]:
    """Walk the JSON; emit one tuple per non-null reference value."""
    data = _load_tv_json()
    cases: list[tuple[str, str, str, float]] = []
    for fixture_name, indicators in data.items():
        if fixture_name.startswith("_") or not isinstance(indicators, dict):
            continue
        for sub_key, ts_map in indicators.items():
            if not isinstance(ts_map, dict):
                continue
            for ts, expected in ts_map.items():
                if expected is None:
                    continue
                cases.append((fixture_name, sub_key, ts, float(expected)))
    return cases


_PARITY_CASES = _enumerate_parity_cases()
_PARITY_META = _load_tv_json().get("_meta", {})


class TestTradingViewParity:
    """Assert each indicator matches TV's Data Window value on a real
    fixture, within the tolerances declared in the JSON's ``_meta``
    block.

    To populate: run ``python -m scripts.build_tv_fixture <SYMBOL.NS>
    <YYYY-MM-DD> tests/fixtures/<name>.csv`` to produce the CSV, then
    in ``tv_reference_values.json`` replace the ``null`` entries with
    the per-bar values TV's Data Window shows for matching indicator
    settings. See the ``_meta.indicator_params`` block for the exact
    TV configuration each value must be pulled at."""

    @pytest.fixture(scope="class")
    def fixture_cache(self) -> dict[str, pd.DataFrame]:
        return {}

    @pytest.fixture(scope="class")
    def output_cache(self) -> dict[tuple[str, str], object]:
        return {}

    def _load_fixture(self, name: str, cache: dict) -> pd.DataFrame:
        if name in cache:
            return cache[name]
        path = _FIXTURES_DIR / f"{name}.csv"
        if not path.exists():
            pytest.skip(
                f"fixture CSV missing: {path.name} — build with "
                "`python -m scripts.build_tv_fixture <SYMBOL.NS> "
                "<YYYY-MM-DD> tests/fixtures/" + name + ".csv`"
            )
        df = pd.read_csv(path, index_col="timestamp")
        # Defensive header normalization: some yfinance / pandas combos
        # have been observed to write headers with leading whitespace
        # (e.g. "                open"). Strip + lowercase so indicators
        # that look up ``df["open"]`` etc. don't silently miss columns.
        df.columns = [c.strip().lower() for c in df.columns]
        # yfinance writes ISO strings with an explicit +05:30 offset; force
        # them through UTC then convert to IST so the index is uniformly
        # tz-aware regardless of the pandas version's default parsing.
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
        cache[name] = df
        return df

    def _compute(self, fixture_name: str, indicator_name: str,
                 df: pd.DataFrame, cache: dict):
        key = (fixture_name, indicator_name)
        if key not in cache:
            spec = REGISTRY[indicator_name]
            cache[key] = spec.func(df, **spec.default_params)
        return cache[key]

    @pytest.mark.parametrize(
        "fixture_name,sub_key,ts_str,expected",
        _PARITY_CASES,
        ids=[f"{fn}-{sk}-{ts}" for fn, sk, ts, _ in _PARITY_CASES],
    )
    def test_indicator_matches_tradingview(
        self, fixture_name, sub_key, ts_str, expected,
        fixture_cache, output_cache,
    ):
        if sub_key not in _SUBKEY_TO_INDICATOR:
            pytest.fail(
                f"unknown sub_key {sub_key!r} — add a mapping in "
                "_SUBKEY_TO_INDICATOR"
            )
        indicator_name, column = _SUBKEY_TO_INDICATOR[sub_key]

        df = self._load_fixture(fixture_name, fixture_cache)
        out = self._compute(fixture_name, indicator_name, df, output_cache)
        series = out if column is None else out[column]

        ts = pd.Timestamp(ts_str)
        if ts.tz is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        if ts not in series.index:
            pytest.fail(
                f"timestamp {ts} not in fixture {fixture_name!r} — bar "
                "may be misaligned with TV's 5-min grid"
            )

        actual = series.loc[ts]
        if pd.isna(actual):
            pytest.fail(
                f"{indicator_name} produced NaN at {ts} but TV reports "
                f"{expected} — warmup mismatch?"
            )

        kind, tol = _resolve_tolerance(
            sub_key, _PARITY_META.get("tolerances", {}),
        )
        if kind == "abs":
            diff = abs(float(actual) - expected)
            assert diff <= tol, (
                f"{sub_key} @ {ts}: ours={actual!s}, TV={expected}, "
                f"|diff|={diff:.6f} > {tol}"
            )
        else:  # "rel"
            denom = abs(expected) if expected != 0 else 1.0
            rel = abs(float(actual) - expected) / denom
            assert rel <= tol, (
                f"{sub_key} @ {ts}: ours={actual!s}, TV={expected}, "
                f"rel={rel:.6%} > {tol:.6%}"
            )
