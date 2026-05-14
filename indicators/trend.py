"""Trend-direction and trend-strength indicators.

All functions are pure. Input is a tz-aware DatetimeIndex DataFrame with
columns ``[open, high, low, close, volume]`` (lowercase). No input
mutation. Warmup periods return NaN, never raise.

Formulas are referenced against TradingView Pine source. Where the
behaviour differs from textbook (e.g. Wilder's smoothing instead of
plain SMA) the Pine reference is in the docstring."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA — TradingView's ``ta.rma``.

    Equivalent to an EMA with alpha = 1/period, seeded with an SMA of the
    first ``period`` *non-NaN* values. This is what TV's RSI, ADX, ATR,
    and Supertrend use under the hood.

    Leading-NaN handling matters: callers like RSI feed in ``close.diff()``
    where the first element is NaN. The seed must advance to the first
    bar index where we've accumulated ``period`` real observations — not
    just to ``period - 1`` of the array. Otherwise the seed averages in
    a non-existent zeroth bar and downstream values are off by one bar."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    arr = series.to_numpy(dtype=np.float64)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pd.Series(out, index=series.index)
    valid_mask = ~np.isnan(arr)
    cumsum = np.cumsum(valid_mask)
    candidates = np.flatnonzero(cumsum >= period)
    if candidates.size == 0:
        return pd.Series(out, index=series.index)
    seed_idx = int(candidates[0])
    seed = arr[valid_mask][:period].mean()
    out[seed_idx] = seed
    alpha = 1.0 / period
    for i in range(seed_idx + 1, n):
        x = arr[i]
        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] + alpha * (x - out[i - 1])
    return pd.Series(out, index=series.index, name=series.name)


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of close. Warmup = ``period`` bars."""
    return df["close"].rolling(window=period, min_periods=period).mean().astype(np.float64)


def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Exponential moving average of close, TV-style.

    Pine: ``ta.ema(source, length)`` — uses alpha = 2/(length+1), SMA-
    seeded so the first ``length-1`` values are NaN and bar at index
    ``length-1`` is the SMA. ``ewm(span=period, adjust=False)`` matches
    this once we drop the warmup prefix."""
    arr = df["close"].to_numpy(dtype=np.float64)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return pd.Series(out, index=df.index)
    seed = np.nanmean(arr[:period])
    out[period - 1] = seed
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        prev = out[i - 1]
        out[i] = prev + alpha * (arr[i] - prev)
    return pd.Series(out, index=df.index)


def _true_range(df: pd.DataFrame) -> pd.Series:
    """``max(high-low, |high-prev_close|, |low-prev_close|)`` per bar.
    Used by ATR, ADX, Supertrend."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    prev_close = df["close"].shift(1).astype(np.float64)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Supertrend (Olivier Seban). Returns ``[supertrend, direction]``.

    Pine: ``ta.supertrend(factor, atrPeriod)``. ATR uses Wilder's RMA.
    Bands shift only in the direction of the trend (i.e. final upper is
    monotone non-increasing while in a downtrend; final lower is monotone
    non-decreasing while in an uptrend). Direction flips when price
    crosses the active band.

    direction = +1 if uptrend (close > supertrend), -1 if downtrend.

    Warmup = ``period`` bars (ATR seed)."""
    atr_series = _rma(_true_range(df), period)
    hl2 = (df["high"].astype(np.float64) + df["low"].astype(np.float64)) / 2.0
    upper_basic = hl2 + multiplier * atr_series
    lower_basic = hl2 - multiplier * atr_series

    close = df["close"].to_numpy(dtype=np.float64)
    ub = upper_basic.to_numpy(dtype=np.float64)
    lb = lower_basic.to_numpy(dtype=np.float64)
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]) or np.isnan(final_lower[i - 1]):
            final_upper[i] = ub[i]
            final_lower[i] = lb[i]
            # Seed direction from close vs midline (hl2).
            direction[i] = 1.0 if close[i] >= hl2.iloc[i] else -1.0
            st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            continue
        # Band locking
        final_upper[i] = ub[i] if (ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = lb[i] if (lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        # Direction flip
        if direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1.0
        elif direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1.0
        else:
            direction[i] = direction[i - 1]
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame(
        {"supertrend": st, "direction": direction},
        index=df.index,
    )


def _wma_series(series: pd.Series, period: int) -> pd.Series:
    """Linear-weight WMA of an arbitrary series — weights = 1..period.
    Used by both ``wma`` (on close) and ``hull_ma`` (recursively)."""
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    return series.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / denom, raw=True,
    ).astype(np.float64)


def wma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Linear-weighted moving average of close. Pine: ``ta.wma``.

    Weight on bar offset ``k`` (oldest=0) is ``k+1``; sum / triangular(N).
    Warmup = ``period``."""
    return _wma_series(df["close"].astype(np.float64), period)


def hull_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Hull Moving Average. Pine: ``ta.hma``.

    HMA = WMA(2*WMA(close, period/2) - WMA(close, period), sqrt(period))

    Designed by Alan Hull (2005) to reduce lag relative to SMA/WMA while
    staying smooth. Warmup ≈ ``period + sqrt(period)`` bars."""
    half = max(2, int(period / 2))
    sqrt_p = max(2, int(np.sqrt(period)))
    close = df["close"].astype(np.float64)
    wma_half = _wma_series(close, half)
    wma_full = _wma_series(close, period)
    diff = 2.0 * wma_half - wma_full
    return _wma_series(diff, sqrt_p)


def parabolic_sar(
    df: pd.DataFrame, af: float = 0.02, max_af: float = 0.2
) -> pd.Series:
    """Parabolic SAR (Wilder).

    Pine: ``ta.sar(start, increment, maximum)``. Acceleration factor
    starts at ``af`` (0.02), increments by ``af`` each time the trend
    posts a new extreme, capped at ``max_af`` (0.2). SAR is also capped
    by the prior two bars' opposite extremes.

    Direction is inferred from the first two closes. Expect ±1-bar lag
    relative to TV around trend flips (TV's implementation has subtle
    differences in flip-bar EP handling).

    Warmup = 2 bars (initial direction inference)."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)
    sar = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return pd.Series(sar, index=df.index)

    direction = 1 if close[1] >= close[0] else -1
    ep = high[0] if direction == 1 else low[0]
    cur_af = af
    sar[1] = low[0] if direction == 1 else high[0]

    for i in range(2, n):
        new_sar = sar[i - 1] + cur_af * (ep - sar[i - 1])
        if direction == 1:
            # SAR can't pierce either of the prior two lows.
            new_sar = min(new_sar, low[i - 1], low[i - 2])
            if low[i] < new_sar:
                # Flip down.
                direction = -1
                sar[i] = ep
                ep = low[i]
                cur_af = af
            else:
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    cur_af = min(cur_af + af, max_af)
        else:
            new_sar = max(new_sar, high[i - 1], high[i - 2])
            if high[i] > new_sar:
                direction = 1
                sar[i] = ep
                ep = high[i]
                cur_af = af
            else:
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    cur_af = min(cur_af + af, max_af)
    return pd.Series(sar, index=df.index)


def ichimoku(
    df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52
) -> pd.DataFrame:
    """Ichimoku Cloud — tenkan, kijun, senkou_a, senkou_b, chikou.

    Pine: ``ta.ichimoku``-style.
      tenkan   = (hh(tenkan) + ll(tenkan)) / 2
      kijun    = (hh(kijun)  + ll(kijun))  / 2
      senkou_a = (tenkan + kijun) / 2, shifted FORWARD by ``kijun``
      senkou_b = (hh(senkou) + ll(senkou)) / 2, shifted FORWARD by ``kijun``
      chikou   = close shifted BACKWARD by ``kijun`` (lagging plot)

    Senkou A and B span are *projected* — the values you see plotted at
    bar N were computed at bar N - kijun. The chikou span at bar N is
    plotted at N - kijun in TV; we represent it as ``close.shift(-kijun)``
    so a snapshot at bar N exposes the value that would be plotted there.

    Warmup = ``senkou + kijun`` for senkou_b (the longest line)."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)

    def _mid(period: int) -> pd.Series:
        h = high.rolling(period, min_periods=period).max()
        l = low.rolling(period, min_periods=period).min()
        return (h + l) / 2.0

    tenkan_line = _mid(tenkan)
    kijun_line = _mid(kijun)
    senkou_a = ((tenkan_line + kijun_line) / 2.0).shift(kijun)
    senkou_b = _mid(senkou).shift(kijun)
    chikou = close.shift(-kijun)

    return pd.DataFrame(
        {
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        },
        index=df.index,
    )


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel — highest high and lowest low over ``period``.

    Returns ``[upper, middle, lower, width]``. Trend-following channel
    used by classic turtles. Warmup = ``period``."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    width = upper - lower
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower, "width": width},
        index=df.index,
    )


def aroon(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Aroon Up / Down / Oscillator. Pine: ``ta.aroon(length)``.

      aroon_up   = 100 * (period - bars_since_highest_high) / period
      aroon_down = 100 * (period - bars_since_lowest_low)  / period
      oscillator = aroon_up - aroon_down

    Window is ``period + 1`` bars (current + period back) so bars_since
    ranges 0..period. Warmup = ``period + 1``."""
    win = period + 1

    def _bars_since_end_max(arr: np.ndarray) -> float:
        return float((len(arr) - 1) - int(np.argmax(arr)))

    def _bars_since_end_min(arr: np.ndarray) -> float:
        return float((len(arr) - 1) - int(np.argmin(arr)))

    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    bsh = high.rolling(win, min_periods=win).apply(_bars_since_end_max, raw=True)
    bsl = low.rolling(win, min_periods=win).apply(_bars_since_end_min, raw=True)
    aroon_up = 100.0 * (period - bsh) / period
    aroon_down = 100.0 * (period - bsl) / period
    oscillator = aroon_up - aroon_down
    return pd.DataFrame(
        {"aroon_up": aroon_up, "aroon_down": aroon_down,
         "oscillator": oscillator},
        index=df.index,
    )


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index (E.W. Dreiss). Pine: ``ta.cci``-adjacent custom.

      ci = 100 * log10(sum(TR, period) / (hh(period) - ll(period))) / log10(period)

    Range [0, 100]. >61.8 = sideways (choppy); <38.2 = trending. Useful
    as a gate on trend indicators (suppress trend signals when ci > 61.8).

    Warmup = ``period``."""
    tr = _true_range(df)
    sum_tr = tr.rolling(period, min_periods=period).sum()
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return (100.0 * np.log10(sum_tr / rng) / np.log10(period)).astype(np.float64)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index, Wilder's original. Returns
    ``[adx, di_plus, di_minus]``.

    Pine: ``ta.dmi(diLength, adxSmoothing)``. The smoothing is Wilder
    RMA throughout: +DM, -DM, TR are RMA-smoothed; DX = 100 * |+DI - -DI|
    / (+DI + -DI); ADX is RMA(DX, period).

    Warmup = ~2*period (one for DI smoothing, another for the ADX
    smoothing on top of DX)."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    tr = _true_range(df)

    atr_series = _rma(tr, period)
    plus_di = 100.0 * _rma(plus_dm, period) / atr_series.replace(0, np.nan)
    minus_di = 100.0 * _rma(minus_dm, period) / atr_series.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = _rma(dx, period)

    return pd.DataFrame(
        {
            "adx": adx_series.astype(np.float64),
            "di_plus": plus_di.astype(np.float64),
            "di_minus": minus_di.astype(np.float64),
        },
        index=df.index,
    )


def zigzag(df: pd.DataFrame, deviation_pct: float = 5.0) -> pd.DataFrame:
    """Percent-deviation ZigZag (TradingView's built-in 'Zig Zag').

    Marks confirmed swing highs and lows. A new pivot is only EMITTED
    once price reverses by at least ``deviation_pct`` against the current
    running extreme. The most recent (still-running) extreme is NOT
    marked until a future reversal confirms it — same convention as TV.

    Returns DataFrame:
        zigzag_price : pivot price at confirmed pivot bars, NaN elsewhere
        pivot_type   : +1 at high pivots, -1 at low pivots, 0 elsewhere

    Algorithm (per bar i):
      - if direction is undecided, track both running highest-high and
        lowest-low; whichever incurs the threshold reversal first
        becomes the first confirmed pivot
      - in an uptrend, update the running high; on a low that's >=
        deviation_pct below the running high, confirm that high as a
        pivot and switch to downtrend
      - symmetric for downtrend

    Warmup ≈ 2 bars (need at least one move). On a no-reversal series
    (e.g. monotonic uptrend), no pivots are emitted at all."""
    if deviation_pct <= 0:
        raise ValueError(f"deviation_pct must be > 0, got {deviation_pct}")
    n = len(df)
    pivot_price = np.full(n, np.nan, dtype=np.float64)
    pivot_type = np.zeros(n, dtype=np.int8)
    if n == 0:
        return pd.DataFrame(
            {"zigzag_price": pivot_price, "pivot_type": pivot_type},
            index=df.index,
        )
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    threshold = deviation_pct / 100.0

    direction = 0  # 0 = undecided, +1 = up, -1 = down
    high_extreme, high_idx = high[0], 0
    low_extreme, low_idx = low[0], 0

    for i in range(1, n):
        h, l = high[i], low[i]
        if direction == 0:
            if h > high_extreme:
                high_extreme, high_idx = h, i
            if l < low_extreme:
                low_extreme, low_idx = l, i
            drop = ((high_extreme - l) / high_extreme) if high_extreme > 0 else 0.0
            rise = ((h - low_extreme) / low_extreme) if low_extreme > 0 else 0.0
            confirm_high = drop >= threshold
            confirm_low = rise >= threshold
            if confirm_high and confirm_low:
                # Tie-break by chronology of the candidate pivots.
                if high_idx <= low_idx:
                    confirm_low = False
                else:
                    confirm_high = False
            if confirm_high:
                pivot_price[high_idx] = high_extreme
                pivot_type[high_idx] = 1
                direction = -1
                low_extreme, low_idx = l, i
            elif confirm_low:
                pivot_price[low_idx] = low_extreme
                pivot_type[low_idx] = -1
                direction = 1
                high_extreme, high_idx = h, i
        elif direction == 1:
            if h > high_extreme:
                high_extreme, high_idx = h, i
            if (high_extreme - l) / high_extreme >= threshold:
                pivot_price[high_idx] = high_extreme
                pivot_type[high_idx] = 1
                direction = -1
                low_extreme, low_idx = l, i
        else:  # direction == -1
            if l < low_extreme:
                low_extreme, low_idx = l, i
            if (h - low_extreme) / low_extreme >= threshold:
                pivot_price[low_idx] = low_extreme
                pivot_type[low_idx] = -1
                direction = 1
                high_extreme, high_idx = h, i

    return pd.DataFrame(
        {"zigzag_price": pivot_price, "pivot_type": pivot_type},
        index=df.index,
    )
