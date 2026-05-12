"""Volume indicators.

OBV, CMF, A/D Line, Volume Surge Ratio (RVOL absolute), RVOL-TOD (time-
of-day relative volume), VWAP-SD bands, Anchored VWAP.

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
