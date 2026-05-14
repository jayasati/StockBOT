"""Volume indicators.

OBV, CMF, A/D Line, Volume Surge Ratio (RVOL absolute), RVOL-TOD (time-
of-day relative volume), VWAP-SD bands, Anchored VWAP, Volume MA, VWMA,
session VWAP, Auto-Anchored VWAP, Visible Average Price.

Pure; lowercase OHLCV input; warmup → NaN."""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume. Pine: ``ta.obv``.

    Cumulative volume signed by close direction:
        sign = +1 if close > prev_close, -1 if close < prev_close, 0 if eq
        obv  = cumsum(sign * volume)

    First bar = 0 (no prior close to compare). Warmup = 1 bar."""
    close = df["close"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    diff = close.diff()
    sign = pd.Series(
        np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0)),
        index=df.index,
    )
    out = (sign * vol).fillna(0.0).cumsum()
    return out.astype(np.float64)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow. Pine: ``ta.cmf(length)``.

    mf_mult  = ((close - low) - (high - close)) / (high - low)
    mf_vol   = mf_mult * volume
    cmf      = sum(mf_vol, period) / sum(volume, period)

    Bars with high == low contribute 0 to the multiplier (avoid div0).
    Warmup = ``period`` bars."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    rng = (high - low).replace(0.0, np.nan)
    mf_mult = ((close - low) - (high - close)) / rng
    mf_mult = mf_mult.fillna(0.0)
    mf_vol = mf_mult * vol
    out = mf_vol.rolling(period, min_periods=period).sum() / vol.rolling(
        period, min_periods=period
    ).sum().replace(0.0, np.nan)
    return out.astype(np.float64)


def volume_surge_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple relative volume vs ``period``-bar SMA.

      vsr[t] = volume[t] / sma(volume, period)[t]

    >1 = above-average volume; values >2 are common at breakout candles.
    Warmup = ``period``."""
    vol = df["volume"].astype(np.float64)
    avg = vol.rolling(period, min_periods=period).mean()
    return (vol / avg.replace(0.0, np.nan)).astype(np.float64)


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line. Pine: ``ta.accdist``.

      mf_mult = ((close - low) - (high - close)) / (high - low)   [0 if h==l]
      mf_vol  = mf_mult * volume
      a/d     = cumsum(mf_vol)

    Conceptually CMF before averaging. Warmup = 1 bar."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    rng = (high - low).replace(0.0, np.nan)
    mf_mult = ((close - low) - (high - close)) / rng
    mf_mult = mf_mult.fillna(0.0)
    return (mf_mult * df["volume"].astype(np.float64)).cumsum().astype(np.float64)


def rvol_tod(df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    """Relative-volume normalised against the same time-of-day over the
    last ``lookback_days`` sessions.

    For each bar at time ``t`` on session ``D``, find the most recent
    ``lookback_days`` sessions strictly before ``D`` that contain a bar
    with the same time-of-day. ``rvol_tod = bar.volume / mean(volumes)``.

    Useful because raw vsr is contaminated by the intraday volume
    U-curve (open + close are always high-volume). This anchors against
    THE SAME slot.

    Returns NaN for any bar without ``lookback_days`` qualifying prior
    bars (early in the data; spec calls for ValueError but we prefer
    silent NaN — Phase 7 treats None as neutral, raising would force
    the scorer to special-case this one indicator)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("rvol_tod requires DatetimeIndex")
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pd.Series(out, index=df.index)
    dates = df.index.date
    times = df.index.time
    volumes = df["volume"].to_numpy(dtype=np.float64)
    # Group bar indices by time-of-day. For each bar, find prior same-TOD
    # bars on strictly earlier sessions and take the last `lookback_days`.
    tod_indices: dict[time, list[int]] = {}
    for i in range(n):
        key = times[i]
        tod_indices.setdefault(key, []).append(i)
    for slot, idx_list in tod_indices.items():
        seen_dates_volumes: list[tuple[date, float]] = []
        last_date_added: date | None = None
        for i in idx_list:
            cur_date = dates[i]
            # If we've moved to a new session, the prior session's bar at
            # this slot (if any) is fair game now.
            qualifying = [v for d, v in seen_dates_volumes if d < cur_date]
            if len(qualifying) >= lookback_days:
                avg = float(np.mean(qualifying[-lookback_days:]))
                if avg > 0:
                    out[i] = volumes[i] / avg
            # Record this bar's contribution for future bars on later sessions.
            seen_dates_volumes.append((cur_date, volumes[i]))
    return pd.Series(out, index=df.index)


def vwap_sd_bands(
    df: pd.DataFrame,
    session_date: date | None = None,
    n_sd_list: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Session-anchored VWAP ± k·σ bands.

    For a given trading session, computes the cumulative volume-weighted
    VWAP and the volume-weighted standard deviation of typical price
    around it. Returns ``plus_k, minus_k`` columns for each k in
    ``n_sd_list``.

    The output is reindexed to the input frame: rows outside the session
    are NaN. ``session_date=None`` defaults to the LAST bar's date.

    Used as horizontal-shaded zones on charts; in Phase 7, distance of
    current price from each band can score mean-reversion vs trend."""
    if df.empty:
        return pd.DataFrame(index=df.index)
    if session_date is None:
        session_date = df.index[-1].date()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("vwap_sd_bands requires DatetimeIndex")
    in_session_mask = pd.Series(df.index.date == session_date, index=df.index)
    session_df = df.loc[in_session_mask.to_numpy()]
    if session_df.empty:
        cols = [f"plus_{k}" for k in n_sd_list] + [f"minus_{k}" for k in n_sd_list]
        return pd.DataFrame(np.nan, index=df.index, columns=cols)

    tp = (session_df["high"] + session_df["low"] + session_df["close"]).astype(np.float64) / 3.0
    vol = session_df["volume"].astype(np.float64)
    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    vwap = (cum_pv / cum_v.replace(0.0, np.nan)).astype(np.float64)
    sq_dev = (tp - vwap) ** 2
    weighted_var = (sq_dev * vol).cumsum() / cum_v.replace(0.0, np.nan)
    sigma = np.sqrt(weighted_var)

    band_cols: dict[str, pd.Series] = {}
    for k in n_sd_list:
        band_cols[f"plus_{k}"] = vwap + k * sigma
        band_cols[f"minus_{k}"] = vwap - k * sigma

    session_bands = pd.DataFrame(band_cols, index=session_df.index)
    # Reindex back to full frame so the output shape matches the input.
    return session_bands.reindex(df.index)


def anchored_vwap(
    df: pd.DataFrame, anchor_ts: pd.Timestamp | None = None,
) -> pd.Series:
    """VWAP anchored to ``anchor_ts``; values before the anchor are NaN.

    If ``anchor_ts`` is None, defaults to the first bar of the last
    session present in the frame (so calling without args anchors at
    today's open).

    Used to score "is price above the anchored VWAP from the breakout
    bar / earnings bar / IPO listing day / etc.". Warmup = 1 bar
    after the anchor."""
    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("anchored_vwap requires DatetimeIndex")
    if anchor_ts is None:
        last_date = df.index[-1].date()
        same_session = df[df.index.date == last_date]
        if same_session.empty:
            return pd.Series(np.nan, index=df.index, dtype=np.float64)
        anchor_ts = same_session.index[0]
    anchor_ts = pd.Timestamp(anchor_ts)
    if df.index.tz is not None and anchor_ts.tz is None:
        anchor_ts = anchor_ts.tz_localize(df.index.tz)
    mask = df.index >= anchor_ts
    if not mask.any():
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    anchored = df.loc[mask]
    tp = (anchored["high"] + anchored["low"] + anchored["close"]).astype(np.float64) / 3.0
    vol = anchored["volume"].astype(np.float64)
    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    vwap = (cum_pv / cum_v.replace(0.0, np.nan)).astype(np.float64)
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    out.loc[mask] = vwap.values
    return out


def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """SMA of volume — the moving-average overlay on TradingView's
    built-in Volume indicator. Pine: ``ta.sma(volume, length)``.

    Used to gauge whether the current bar's volume is above or below its
    recent average. Warmup = ``period`` bars."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    vol = df["volume"].astype(np.float64)
    return vol.rolling(period, min_periods=period).mean()


def visible_average_price(
    df: pd.DataFrame,
    n_bars: int = 100,
    bins: int = 24,
    value_area_pct: float = 0.70,
) -> dict[str, float]:
    """Visible Range Volume Profile (TradingView 'Visible Average Price' /
    'Volume Profile').

    Buckets the last ``n_bars`` bars into ``bins`` equal-height price
    levels (between window low and window high), attributes each bar's
    full volume to the bin containing its close, then returns:

        poc          — midpoint of the highest-volume bin (Point of Control)
        vah, val     — upper / lower edges of the Value Area, the smallest
                       contiguous price band around POC containing
                       ``value_area_pct`` of total volume
        total_volume — sum of all volume in the window

    Returns NaN for poc/vah/val if the window is empty. ``value_area_pct``
    must be in (0, 1]."""
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if not (0 < value_area_pct <= 1.0):
        raise ValueError(
            f"value_area_pct must be in (0, 1]; got {value_area_pct}"
        )
    if df.empty:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "total_volume": 0.0}

    window = df.tail(n_bars)
    price_low = float(window["low"].min())
    price_high = float(window["high"].max())
    total_volume = float(window["volume"].sum())

    if price_high == price_low:
        return {
            "poc": price_low, "vah": price_high, "val": price_low,
            "total_volume": total_volume,
        }

    edges = np.linspace(price_low, price_high, bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    closes = window["close"].to_numpy(dtype=np.float64)
    vols = window["volume"].to_numpy(dtype=np.float64)
    # searchsorted finds insertion point on the right of each value, so
    # subtract 1 to get the bin index. Clip handles the right edge
    # (bars exactly at price_high otherwise land in bin == bins).
    bin_idx = np.clip(
        np.searchsorted(edges, closes, side="right") - 1, 0, bins - 1,
    )
    bin_volumes = np.zeros(bins, dtype=np.float64)
    np.add.at(bin_volumes, bin_idx, vols)

    poc_idx = int(np.argmax(bin_volumes))
    poc = float(midpoints[poc_idx])

    # Value-area expansion: walk outward from POC, taking whichever
    # neighbour bin has more volume, until cumulative >= target.
    target = total_volume * value_area_pct
    cum = float(bin_volumes[poc_idx])
    lo = hi = poc_idx
    while cum < target and (lo > 0 or hi < bins - 1):
        up_vol = bin_volumes[hi + 1] if hi < bins - 1 else -np.inf
        dn_vol = bin_volumes[lo - 1] if lo > 0 else -np.inf
        if up_vol >= dn_vol:
            hi += 1
            cum += float(bin_volumes[hi])
        else:
            lo -= 1
            cum += float(bin_volumes[lo])

    return {
        "poc": poc,
        "vah": float(edges[hi + 1]),
        "val": float(edges[lo]),
        "total_volume": total_volume,
    }


def auto_anchored_vwap(
    df: pd.DataFrame, lookback: int = 50,
) -> pd.DataFrame:
    """Auto-Anchored VWAP — VWAP from the rolling highest-high / lowest-low
    bar in the last ``lookback`` bars.

    At each bar T, two anchors are computed independently:

        anchor_high(T) = bar with the highest high in [T-lookback+1, T]
        anchor_low(T)  = bar with the lowest low  in [T-lookback+1, T]

    and the VWAP is summed from that anchor up to and including T using
    the typical-price ``hlc3`` convention. As new bars print, the anchor
    can SHIFT to a fresher swing extreme — that's the "auto" part.

    Returns a DataFrame with columns ``avwap_from_high`` and
    ``avwap_from_low``. Warmup = 1 bar (the first bar is its own anchor)."""
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    n = len(df)
    cols = ("avwap_from_high", "avwap_from_low")
    if n == 0:
        return pd.DataFrame({c: [] for c in cols}, index=df.index)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    tp = (df["high"] + df["low"] + df["close"]).astype(np.float64).to_numpy() / 3.0
    vol = df["volume"].astype(np.float64).to_numpy()
    pv = tp * vol
    pv_cum = np.concatenate([[0.0], np.cumsum(pv)])  # prefix sums for O(1) range
    v_cum = np.concatenate([[0.0], np.cumsum(vol)])

    avwap_high = np.full(n, np.nan)
    avwap_low = np.full(n, np.nan)
    for t in range(n):
        start = max(0, t - lookback + 1)
        # Argmax/argmin over the rolling window — small (lookback) so
        # the inner numpy call dominates, not Python-loop overhead.
        idx_h = start + int(np.argmax(high[start:t + 1]))
        idx_l = start + int(np.argmin(low[start:t + 1]))
        sum_pv_h = pv_cum[t + 1] - pv_cum[idx_h]
        sum_v_h = v_cum[t + 1] - v_cum[idx_h]
        sum_pv_l = pv_cum[t + 1] - pv_cum[idx_l]
        sum_v_l = v_cum[t + 1] - v_cum[idx_l]
        if sum_v_h > 0:
            avwap_high[t] = sum_pv_h / sum_v_h
        if sum_v_l > 0:
            avwap_low[t] = sum_pv_l / sum_v_l
    return pd.DataFrame(
        {"avwap_from_high": avwap_high, "avwap_from_low": avwap_low},
        index=df.index,
    )


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP — TradingView's built-in ``VWAP`` indicator.

    Pine: ``ta.vwap(hlc3)`` — resets at each new session::

        tp = (high + low + close) / 3
        vwap = cumsum(tp * volume) / cumsum(volume)   per session

    Sessions are grouped by the bar's local date (the index must be
    tz-aware). Bars where the session-cumulative volume is 0 produce
    NaN. Warmup = 1 bar per session."""
    if df.empty:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("vwap requires DatetimeIndex")
    tp = (df["high"] + df["low"] + df["close"]).astype(np.float64) / 3.0
    vol = df["volume"].astype(np.float64)
    pv = tp * vol
    session = pd.Series(df.index.date, index=df.index, name="_session")
    cum_pv = pv.groupby(session, sort=False).cumsum()
    cum_v = vol.groupby(session, sort=False).cumsum()
    return (cum_pv / cum_v.replace(0.0, np.nan)).astype(np.float64)


def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average. Pine: ``ta.vwma(source, length)``::

        vwma = sma(source * volume, length) / sma(volume, length)

    A weighted MA where each bar's contribution is scaled by its volume
    — high-volume bars pull the average more than low-volume bars. The
    rolling sums of (source*vol) and (vol) cancel the period factor, so
    we use rolling sums (equivalent to and a hair faster than rolling
    means). Warmup = ``period`` bars. Bars where the rolling volume sum
    is 0 produce NaN to avoid div-by-zero."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    src = df["close"].astype(np.float64)
    vol = df["volume"].astype(np.float64)
    pv = src * vol
    sum_pv = pv.rolling(period, min_periods=period).sum()
    sum_v = vol.rolling(period, min_periods=period).sum()
    return (sum_pv / sum_v.replace(0.0, np.nan)).astype(np.float64)
