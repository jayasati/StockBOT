"""Momentum / oscillator indicators.

All functions are pure; lowercase OHLCV input; warmup periods return NaN.

The TradingView Pine formula for each indicator is referenced in the
docstring so future maintainers can re-verify parity if pandas-ta
changes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import _rma


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's RSI (TradingView's ``ta.rsi``).

    Pine::

        up   = ta.rma(math.max(ta.change(source), 0), length)
        down = ta.rma(-math.min(ta.change(source), 0), length)
        rsi  = 100 - (100 / (1 + up / down))

    The legacy ``bot/indicators.compute_rsi`` used plain SMA-rolling
    instead of RMA — that's the simple/Cutler RSI, which lags Wilder's
    by ~5 points on trending sessions. Phase 4 standardises on Wilder
    because that's what TradingView traders see.

    Warmup = ``period + 1`` bars."""
    delta = df["close"].astype(np.float64).diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    up_rma = _rma(up, period)
    down_rma = _rma(down, period)
    rs = up_rma / down_rma.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When down is zero (all gains), RSI = 100 — keep the conventional cap.
    out = out.where(~down_rma.eq(0.0), 100.0)
    out = out.where(~up_rma.eq(0.0) | down_rma.eq(0.0), 0.0)
    return out.astype(np.float64)


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, histogram. Pine: ``ta.macd``.

    macd      = ema(close, fast) - ema(close, slow)
    signal    = ema(macd, signal)
    histogram = macd - signal

    Note: Pine's ``ta.ema`` uses adjust=False starting from the very
    first bar (no warmup mask), unlike our ``trend.ema`` which masks the
    first ``period - 1`` bars. To keep MACD output the same shape as
    Pine after warmup, we run ``ewm(span=, adjust=False)`` directly.

    Warmup = ``slow + signal``."""
    close = df["close"].astype(np.float64)
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line.astype(np.float64),
            "signal": signal_line.astype(np.float64),
            "histogram": histogram.astype(np.float64),
        },
        index=df.index,
    )


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI. Pine: ``ta.mfi``.

      tp        = (high + low + close) / 3
      raw_mf    = tp * volume
      pos_mf    = sum(raw_mf where tp > prev_tp, period)
      neg_mf    = sum(raw_mf where tp < prev_tp, period)
      mfi       = 100 - 100 / (1 + pos_mf / neg_mf)

    Warmup = ``period + 1`` (first tp diff is NaN)."""
    tp = (df["high"] + df["low"] + df["close"]).astype(np.float64) / 3.0
    rmf = tp * df["volume"].astype(np.float64)
    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0, 0.0)
    neg_mf = rmf.where(tp_diff < 0, 0.0)
    # First bar has no prior tp → propagate NaN so the rolling sum's
    # min_periods kicks in correctly. Pine's `na ? x : 0` resolves to na;
    # mirror that.
    nan_mask = tp_diff.isna()
    pos_mf = pos_mf.mask(nan_mask)
    neg_mf = neg_mf.mask(nan_mask)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    # All-gains window: neg_sum=0 → MFI=100. Mirror of the RSI cap.
    out = out.where(~neg_sum.eq(0.0), 100.0)
    # All-losses window: pos_sum=0 (and neg>0) → MFI=0.
    all_loss = pos_sum.eq(0.0) & neg_sum.gt(0.0)
    out = out.where(~all_loss, 0.0)
    return out.astype(np.float64)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index. Pine: ``ta.cci(source, length)``.

      tp        = (high + low + close) / 3
      sma_tp    = sma(tp, period)
      mean_dev  = sma(|tp - sma_tp|, period)       # mean ABSOLUTE deviation
      cci       = (tp - sma_tp) / (0.015 * mean_dev)

    Note: ``mean_dev`` uses mean-absolute-deviation, NOT std. Easy
    confusion — both Pine and Lambert's original definition use MAD.

    Typical ranges: ±100 normal, ±200 extreme. Warmup = ``period``."""
    tp = (df["high"] + df["low"] + df["close"]).astype(np.float64) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mean_dev = (tp - sma_tp).abs().rolling(period, min_periods=period).mean()
    return ((tp - sma_tp) / (0.015 * mean_dev.replace(0.0, np.nan))).astype(np.float64)


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Rate of Change. Pine: ``ta.roc(source, length)``.

      roc = 100 * (close - close[period]) / close[period]

    Warmup = ``period + 1``."""
    close = df["close"].astype(np.float64)
    return (100.0 * (close / close.shift(period) - 1.0)).astype(np.float64)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R. Pine: ``ta.wpr(length)``.

      %r = -100 * (highest_high(period) - close) / (highest_high - lowest_low)

    Values run [-100, 0]. -20..0 = overbought; -100..-80 = oversold.
    direction='bullish_low' in the registry (close to 0 is *bullish*).

    Warmup = ``period``."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return (-100.0 * (hh - close) / (hh - ll).replace(0.0, np.nan)).astype(np.float64)


def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """Bill Williams' Awesome Oscillator. Pine: ``ta.ao``.

      hl2 = (high + low) / 2
      ao  = sma(hl2, 5) - sma(hl2, 34)

    Zero-cross momentum signal. Warmup = 34 bars."""
    hl2 = (df["high"].astype(np.float64) + df["low"].astype(np.float64)) / 2.0
    fast = hl2.rolling(5, min_periods=5).mean()
    slow = hl2.rolling(34, min_periods=34).mean()
    return (fast - slow).astype(np.float64)


def trix(df: pd.DataFrame, period: int = 15) -> pd.Series:
    """Triple-smoothed EMA rate of change. Pine: ``ta.trix(source, length)``.

      e1 = ema(close, length)
      e2 = ema(e1,    length)
      e3 = ema(e2,    length)
      trix = 10000 * (e3 - e3[1]) / e3[1]

    The ×10000 scaling matches TV's display convention (TRIX of a smoothed
    series is typically a tiny percentage). Warmup ≈ 3*period."""
    close = df["close"].astype(np.float64)
    e1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
    e3 = e2.ewm(span=period, adjust=False, min_periods=period).mean()
    return (10000.0 * (e3 / e3.shift(1) - 1.0)).astype(np.float64)


def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Elder's Force Index. Pine: ``ta.efi(length)``.

      raw_fi = (close - close[1]) * volume
      force_index = ema(raw_fi, period)

    Combines price-change direction and volume into a single magnitude.
    Warmup = ``period + 1`` (raw_fi[0] is NaN)."""
    raw = df["close"].astype(np.float64).diff() * df["volume"].astype(np.float64)
    return raw.ewm(span=period, adjust=False, min_periods=period).mean().astype(np.float64)


def _streak_series(close: pd.Series) -> np.ndarray:
    """The Connors-RSI streak series. ``streak[i]`` = number of
    consecutive bars ending at ``i`` where close moved in the same
    direction (positive = up streak, negative = down streak, 0 = unchanged
    from prior). Resets when the direction flips."""
    arr = close.to_numpy(dtype=np.float64)
    n = arr.size
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = arr[i] - arr[i - 1]
        prev = streak[i - 1]
        if d > 0:
            streak[i] = prev + 1 if prev > 0 else 1
        elif d < 0:
            streak[i] = prev - 1 if prev < 0 else -1
        else:
            streak[i] = 0
    return streak


def connors_rsi(
    df: pd.DataFrame,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> pd.Series:
    """Connors RSI (Larry Connors, 2012). Composite of three sub-indicators::

        rsi_close    = ta.rsi(close, rsi_period)
        rsi_streak   = ta.rsi(streak_series, streak_period)
        pct_rank     = ta.percentrank(ta.roc(close, 1), rank_period)
        crsi         = (rsi_close + rsi_streak + pct_rank) / 3

    where ``percentrank`` is the Pine convention: % of the past
    ``rank_period`` values STRICTLY less than the current value (current
    bar excluded from the comparison set). Output is in [0, 100]; warmup
    ≈ ``rank_period + 1`` bars (the percent-rank lookback dominates)."""
    if rsi_period < 2 or streak_period < 2 or rank_period < 1:
        raise ValueError(
            "rsi_period >= 2, streak_period >= 2, rank_period >= 1 required"
        )
    close = df["close"].astype(np.float64)

    rsi_close = rsi(df, period=rsi_period)

    streak = _streak_series(close)
    streak_df = pd.DataFrame({"close": streak}, index=df.index)
    rsi_streak = rsi(streak_df, period=streak_period)

    roc_1 = close.pct_change() * 100.0  # 1-bar ROC, in percent

    def _pr(window: np.ndarray) -> float:
        past = window[:-1]
        cur = window[-1]
        if np.isnan(cur):
            return np.nan
        # Strict-less-than match Pine's percentrank semantics. NaN values
        # in the past window naturally don't satisfy `< cur`.
        return 100.0 * float(np.sum(past < cur)) / rank_period

    pct_rank = roc_1.rolling(
        rank_period + 1, min_periods=rank_period + 1,
    ).apply(_pr, raw=True)

    return ((rsi_close + rsi_streak + pct_rank) / 3.0).astype(np.float64)


def tsi(
    df: pd.DataFrame,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 13,
) -> pd.DataFrame:
    """True Strength Index (Blau, 1991).

    Pine reference (TradingView's built-in 'True Strength Index')::

        pc        = ta.change(close)
        smooth_pc = ta.ema(ta.ema(pc, long), short)
        smooth_abs= ta.ema(ta.ema(math.abs(pc), long), short)
        tsi       = 100 * smooth_pc / smooth_abs
        signal    = ta.ema(tsi, signal_length)

    Returns ``{tsi, signal}``, both in the [-100, 100] range. Uses
    recursive EMA seeding (``ewm adjust=False``) — matches the ``ta``
    Python library; differs slightly from Pine's SMA-seeded ``ta.ema``
    in the first ~3x(long+short) bars, then converges. Bars where the
    smoothed |Δprice| is exactly 0 produce NaN (constant-price stretch)."""
    if long_period < 1 or short_period < 1 or signal_period < 1:
        raise ValueError("all periods must be >= 1")
    pc = df["close"].astype(np.float64).diff()
    abs_pc = pc.abs()

    def _double_ema(s: pd.Series) -> pd.Series:
        return (
            s.ewm(span=long_period, adjust=False, min_periods=long_period).mean()
             .ewm(span=short_period, adjust=False, min_periods=short_period).mean()
        )

    num = _double_ema(pc)
    den = _double_ema(abs_pc)
    tsi_val = 100.0 * (num / den.replace(0.0, np.nan))
    signal = tsi_val.ewm(
        span=signal_period, adjust=False, min_periods=signal_period,
    ).mean()
    return pd.DataFrame(
        {"tsi": tsi_val.astype(np.float64),
         "signal": signal.astype(np.float64)},
        index=df.index,
    )


def stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI — Stochastic oscillator applied to RSI values.

    Pine reference (TradingView's built-in 'Stochastic RSI')::

        rsi1 = ta.rsi(src, lengthRSI)
        k    = ta.sma(ta.stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)
        d    = ta.sma(k, smoothD)

    Returns ``{k, d}`` in 0-100. Bars where the rolling RSI range is 0
    (RSI flat over the window) produce NaN to avoid div-by-zero. Warmup
    ≈ ``rsi_period + stoch_period + k_smooth + d_smooth - 3`` bars."""
    if rsi_period < 2 or stoch_period < 1 or k_smooth < 1 or d_smooth < 1:
        raise ValueError(
            "rsi_period >= 2, stoch_period/k_smooth/d_smooth >= 1 required"
        )
    rsi_series = rsi(df, period=rsi_period)
    rsi_min = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()
    rng = (rsi_max - rsi_min).replace(0.0, np.nan)
    raw_k = 100.0 * (rsi_series - rsi_min) / rng
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return pd.DataFrame(
        {"k": k.astype(np.float64), "d": d.astype(np.float64)},
        index=df.index,
    )


def stochastic(
    df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator (TradingView's "Stochastic", not "Slow Stoch").

    Pine::

        k_raw   = 100 * (close - ll) / (hh - ll)
        k       = ta.sma(k_raw, smoothK)
        d       = ta.sma(k, periodD)

    where ``hh = ta.highest(high, periodK)`` and ``ll = ta.lowest(low,
    periodK)``. The ``smooth`` parameter applies SMA to %K BEFORE %D is
    computed — common confusion point with Williams' definition.

    Warmup = ``k + smooth - 1`` for %K, plus ``d - 1`` more for %D."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    hh = high.rolling(k, min_periods=k).max()
    ll = low.rolling(k, min_periods=k).min()
    k_raw = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    k_smoothed = k_raw.rolling(smooth, min_periods=smooth).mean()
    d_line = k_smoothed.rolling(d, min_periods=d).mean()
    return pd.DataFrame(
        {"k": k_smoothed.astype(np.float64), "d": d_line.astype(np.float64)},
        index=df.index,
    )
