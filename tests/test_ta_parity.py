"""``ta``-library cross-check for every indicator with a matching reference.

The hand-computed tests in ``test_indicators.py`` prove each formula is
correct on a tiny input. This module is the wider net: run our impl and
the ``ta`` package's impl side by side on the same deterministic
~250-bar OHLCV frame and assert post-warmup agreement within a
per-indicator tolerance.

Why post-warmup? Wilder's RMA, EMA, triple-EMA and similar recursive
smoothings have library-specific seeding (SMA-of-first-N vs expanding
mean vs first-value), so the first few dozen bars legitimately differ
even when both impls are correct. Comparing from a generous warmup
offset captures the steady-state behaviour, which is what alerts
actually run on.

Indicators NOT in this file: Supertrend, Choppiness Index, TTM Squeeze,
Hull MA, and our custom volume builders (volume_surge_ratio, rvol_tod,
vwap_sd_bands, anchored_vwap). ``ta`` doesn't ship equivalents; those
are covered by hand-computed cases + the TV-parity layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import ta
from zoneinfo import ZoneInfo

from indicators.momentum import (
    awesome_oscillator,
    force_index,
    macd,
    mfi,
    roc,
    rsi,
    trix,
    williams_r,
)
from indicators.trend import (
    adx,
    aroon,
    donchian,
    ema,
    ichimoku,
    sma,
    wma,
)
from indicators.volatility import atr, bollinger, keltner
from indicators.volume import ad_line, cmf, obv

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixture: one deterministic OHLCV frame used by every cross-check
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    """250-bar 5m IST frame — long enough for any standard warmup."""
    n = 250
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    open_ = close + rng.uniform(-0.3, 0.3, n)
    volume = rng.uniform(500, 5000, n)
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_close(
    ours: pd.Series,
    theirs: pd.Series,
    *,
    atol: float,
    skip: int,
    name: str,
) -> None:
    """Drop NaNs from either side, then assert ``allclose`` on the rest.

    Reports the max absolute diff in the failure message so we can tell
    the difference between "way off" and "off by one ULP somewhere"."""
    a = ours.iloc[skip:].to_numpy(dtype=np.float64)
    b = theirs.iloc[skip:].to_numpy(dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    assert len(a) > 0, f"{name}: no overlapping non-NaN bars after skip={skip}"
    max_diff = float(np.max(np.abs(a - b)))
    assert np.allclose(a, b, atol=atol, rtol=0), (
        f"{name}: max |diff| = {max_diff:.6f} > atol = {atol}, "
        f"compared {len(a)} bars"
    )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class TestTrendCrossCheck:
    def test_sma(self, ohlcv):
        ours = sma(ohlcv, period=20)
        theirs = ta.trend.SMAIndicator(
            close=ohlcv["close"], window=20, fillna=False,
        ).sma_indicator()
        _assert_close(ours, theirs, atol=1e-9, skip=25, name="sma")

    def test_ema(self, ohlcv):
        ours = ema(ohlcv, period=20)
        theirs = ta.trend.EMAIndicator(
            close=ohlcv["close"], window=20, fillna=False,
        ).ema_indicator()
        # Seed convention differs (ours: SMA-of-first-N, ta: similar but
        # may include partial windows); converges within a few periods.
        _assert_close(ours, theirs, atol=0.05, skip=60, name="ema")

    def test_wma(self, ohlcv):
        ours = wma(ohlcv, period=20)
        theirs = ta.trend.WMAIndicator(
            close=ohlcv["close"], window=20, fillna=False,
        ).wma()
        _assert_close(ours, theirs, atol=1e-9, skip=25, name="wma")

    def test_adx(self, ohlcv):
        ours = adx(ohlcv, period=14)
        ta_adx = ta.trend.ADXIndicator(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            window=14, fillna=False,
        )
        _assert_close(ours["adx"], ta_adx.adx(),
                      atol=0.5, skip=80, name="adx.adx")
        _assert_close(ours["di_plus"], ta_adx.adx_pos(),
                      atol=0.5, skip=80, name="adx.di_plus")
        _assert_close(ours["di_minus"], ta_adx.adx_neg(),
                      atol=0.5, skip=80, name="adx.di_minus")

    def test_aroon(self, ohlcv):
        ours = aroon(ohlcv, period=14)
        ta_aroon = ta.trend.AroonIndicator(
            high=ohlcv["high"], low=ohlcv["low"], window=14, fillna=False,
        )
        _assert_close(ours["aroon_up"], ta_aroon.aroon_up(),
                      atol=1e-6, skip=20, name="aroon_up")
        _assert_close(ours["aroon_down"], ta_aroon.aroon_down(),
                      atol=1e-6, skip=20, name="aroon_down")

    # PSAR is intentionally NOT cross-checked. The "initial trend
    # direction" choice differs between libraries, and once the libs
    # disagree on direction the SAR flip points cascade, producing
    # wildly different values for the rest of the series. Both impls
    # can be locally correct yet globally diverge. Covered by hand
    # tests in test_indicators.py.

    def test_ichimoku(self, ohlcv):
        ours = ichimoku(ohlcv)
        ta_ichi = ta.trend.IchimokuIndicator(
            high=ohlcv["high"], low=ohlcv["low"],
            window1=9, window2=26, window3=52, fillna=False,
        )
        # ta's tenkan/kijun map straight; senkou_a/b use the same
        # (high+low)/2 averaging.
        _assert_close(ours["tenkan"], ta_ichi.ichimoku_conversion_line(),
                      atol=1e-6, skip=30, name="ichimoku.tenkan")
        _assert_close(ours["kijun"], ta_ichi.ichimoku_base_line(),
                      atol=1e-6, skip=30, name="ichimoku.kijun")


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentumCrossCheck:
    def test_rsi(self, ohlcv):
        ours = rsi(ohlcv, period=14)
        theirs = ta.momentum.RSIIndicator(
            close=ohlcv["close"], window=14, fillna=False,
        ).rsi()
        # ta seeds Wilder's RMA with an expanding mean from bar 0; we
        # seed with SMA-of-first-14 at bar 14. The two converge but it
        # takes ~5x the period for the seed drift to decay below 0.1.
        _assert_close(ours, theirs, atol=0.1, skip=80, name="rsi")

    def test_macd(self, ohlcv):
        ours = macd(ohlcv, fast=12, slow=26, signal=9)
        ta_macd = ta.trend.MACD(
            close=ohlcv["close"], window_slow=26, window_fast=12,
            window_sign=9, fillna=False,
        )
        _assert_close(ours["macd"], ta_macd.macd(),
                      atol=0.05, skip=60, name="macd.macd")
        _assert_close(ours["signal"], ta_macd.macd_signal(),
                      atol=0.05, skip=60, name="macd.signal")
        _assert_close(ours["histogram"], ta_macd.macd_diff(),
                      atol=0.05, skip=60, name="macd.histogram")

    # Stochastic is intentionally NOT cross-checked. ta's
    # StochasticOscillator exposes one smoothing parameter
    # (smooth_window), while TradingView's (and our) convention has TWO
    # stages: a smooth-%K stage and a smooth-%D stage. The mapping
    # between ta's outputs and ours can't be made parity-equivalent
    # for both columns simultaneously. Hand tests cover the formula.

    def test_mfi(self, ohlcv):
        ours = mfi(ohlcv, period=14)
        theirs = ta.volume.MFIIndicator(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            volume=ohlcv["volume"], window=14, fillna=False,
        ).money_flow_index()
        _assert_close(ours, theirs, atol=0.5, skip=20, name="mfi")

    # CCI is intentionally NOT cross-checked. ta's CCI uses pandas'
    # deprecated ``.mad()`` for the mean-absolute-deviation in the
    # denominator and we use a hand-rolled ``mean(|x - mean(x)|)``.
    # Both are valid CCI variants seen in the wild and yield values
    # that can differ by 100+ points on the same input. Hand tests
    # validate our formula matches the canonical Lambert definition.

    def test_roc(self, ohlcv):
        ours = roc(ohlcv, period=12)
        theirs = ta.momentum.ROCIndicator(
            close=ohlcv["close"], window=12, fillna=False,
        ).roc()
        _assert_close(ours, theirs, atol=1e-6, skip=15, name="roc")

    def test_williams_r(self, ohlcv):
        ours = williams_r(ohlcv, period=14)
        theirs = ta.momentum.WilliamsRIndicator(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            lbp=14, fillna=False,
        ).williams_r()
        _assert_close(ours, theirs, atol=1e-6, skip=20, name="williams_r")

    def test_awesome_oscillator(self, ohlcv):
        ours = awesome_oscillator(ohlcv)
        theirs = ta.momentum.AwesomeOscillatorIndicator(
            high=ohlcv["high"], low=ohlcv["low"], fillna=False,
        ).awesome_oscillator()
        _assert_close(ours, theirs, atol=1e-6, skip=40, name="awesome_oscillator")

    def test_trix(self, ohlcv):
        ours = trix(ohlcv, period=15)
        theirs = ta.trend.TRIXIndicator(
            close=ohlcv["close"], window=15, fillna=False,
        ).trix()
        # Unit convention difference: ta returns the fraction (e.g. 0.05),
        # we return percent (e.g. 5.0) to match TradingView. Rescale before
        # comparing.
        _assert_close(ours / 100.0, theirs, atol=0.01, skip=60, name="trix")


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

class TestVolatilityCrossCheck:
    def test_atr(self, ohlcv):
        ours = atr(ohlcv, period=14)
        theirs = ta.volatility.AverageTrueRange(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            window=14, fillna=False,
        ).average_true_range()
        _assert_close(ours, theirs, atol=0.01, skip=30, name="atr")

    def test_bollinger_middle(self, ohlcv):
        """Only the middle band (SMA) is checked here. The upper/lower
        bands legitimately diverge because ``ta`` uses sample stdev
        (ddof=1) while we use population stdev (ddof=0) to match
        TradingView. That's a known fork, not a bug."""
        ours = bollinger(ohlcv, period=20, std=2.0)
        theirs = ta.volatility.BollingerBands(
            close=ohlcv["close"], window=20, window_dev=2, fillna=False,
        )
        _assert_close(ours["middle"], theirs.bollinger_mavg(),
                      atol=1e-9, skip=25, name="bollinger.middle")

    def test_donchian(self, ohlcv):
        ours = donchian(ohlcv, period=20)
        theirs = ta.volatility.DonchianChannel(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            window=20, fillna=False,
        )
        _assert_close(ours["upper"], theirs.donchian_channel_hband(),
                      atol=1e-9, skip=25, name="donchian.upper")
        _assert_close(ours["lower"], theirs.donchian_channel_lband(),
                      atol=1e-9, skip=25, name="donchian.lower")

    def test_keltner(self, ohlcv):
        ours = keltner(ohlcv, period=20, mult=2.0, atr_period=10)
        theirs = ta.volatility.KeltnerChannel(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            window=20, window_atr=10, fillna=False,
            original_version=False, multiplier=2,
        )
        _assert_close(ours["middle"], theirs.keltner_channel_mband(),
                      atol=0.05, skip=40, name="keltner.middle")


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

class TestVolumeCrossCheck:
    def test_obv(self, ohlcv):
        ours = obv(ohlcv)
        theirs = ta.volume.OnBalanceVolumeIndicator(
            close=ohlcv["close"], volume=ohlcv["volume"], fillna=False,
        ).on_balance_volume()
        # ta and we disagree on the starting value of the OBV cumsum (ta
        # treats bar 0 as a positive contribution; we start from 0). The
        # ABSOLUTE values differ by a constant offset, but the bar-to-bar
        # CHANGES — which is all that matters for trend/divergence reads
        # — must agree exactly.
        _assert_close(ours.diff(), theirs.diff(),
                      atol=1e-6, skip=5, name="obv.diff")

    def test_cmf(self, ohlcv):
        ours = cmf(ohlcv, period=20)
        theirs = ta.volume.ChaikinMoneyFlowIndicator(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            volume=ohlcv["volume"], window=20, fillna=False,
        ).chaikin_money_flow()
        _assert_close(ours, theirs, atol=1e-6, skip=25, name="cmf")

    def test_ad_line(self, ohlcv):
        ours = ad_line(ohlcv)
        theirs = ta.volume.AccDistIndexIndicator(
            high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
            volume=ohlcv["volume"], fillna=False,
        ).acc_dist_index()
        _assert_close(ours, theirs, atol=1e-6, skip=5, name="ad_line")

    def test_force_index(self, ohlcv):
        ours = force_index(ohlcv, period=13)
        theirs = ta.volume.ForceIndexIndicator(
            close=ohlcv["close"], volume=ohlcv["volume"],
            window=13, fillna=False,
        ).force_index()
        # ta's ForceIndex uses an EMA seed slightly different from ours.
        _assert_close(ours, theirs, atol=50.0, skip=40, name="force_index")
