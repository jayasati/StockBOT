"""
Per-symbol intraday & daily features (§3 row 4 of `idea/playbook_architecture.md`).

All public functions are pure: no I/O, no globals mutated, deterministic for a
given input. They are called from both the live scan tick and the backtest
replay, so any drift between the two would silently bias backtests.

Conventions
-----------
- 5-min bars: a DataFrame with columns ``[open, high, low, close, volume]``
  indexed by a tz-aware (or naive-IST) ``DatetimeIndex``. The timestamp marks
  the *open* of the 5-minute interval. NSE cash-equity session is 09:15–15:30
  IST; bars outside that window are ignored where it matters (e.g. session
  VWAP, opening range).
- Daily bars: same OHLCV columns, indexed by date (or DatetimeIndex truncated
  to date). One row per session.
- Times: `session_date` is a `datetime.date` (the IST trading date the caller
  wants computed). VWAP, opening range and RVOL all reset at the 09:15 IST
  session boundary.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from bot.schedule import SESSION_CLOSE, SESSION_OPEN  # re-exported for callers

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_ist_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``idx`` localised to IST. Naive timestamps are assumed IST."""
    if idx.tz is None:
        return idx.tz_localize(IST)
    return idx.tz_convert(IST)


def _bars_for_session(bars: pd.DataFrame, session_date: date) -> pd.DataFrame:
    if bars.empty:
        return bars
    ist_idx = _ensure_ist_index(bars.index)
    mask = ist_idx.date == session_date
    return bars.loc[mask]


def _daily_dates(bars_daily: pd.DataFrame) -> list[date]:
    idx = bars_daily.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            return list(idx.tz_convert(IST).date)
        return list(idx.date)
    return [d if isinstance(d, date) else pd.Timestamp(d).date() for d in idx]


# ---------------------------------------------------------------------------
# Public features
# ---------------------------------------------------------------------------

def session_vwap(bars_5m: pd.DataFrame, session_date: date) -> pd.Series:
    """Cumulative session VWAP for bars on ``session_date``.

    Resets at 09:15 IST: the cumulative price-volume sum starts from the first
    bar of the session, not from any earlier session present in ``bars_5m``.
    Returned Series is indexed by the same timestamps as the matching bars.
    """
    sess = _bars_for_session(bars_5m, session_date)
    if sess.empty:
        return pd.Series(dtype=float)
    typical = (sess["high"] + sess["low"] + sess["close"]) / 3.0
    cum_pv = (typical * sess["volume"]).cumsum()
    cum_vol = sess["volume"].cumsum()
    vwap = cum_pv / cum_vol
    vwap.name = "vwap"
    return vwap


def atr(bars_5m: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR on 5-min bars.

    Uses Wilder smoothing (EMA with α = 1/period), matching the convention in
    most charting platforms. The first bar's ATR is NaN because true range
    needs the prior close.
    """
    high = bars_5m["high"]
    low = bars_5m["low"]
    close = bars_5m["close"]
    prev_close = close.shift(1)
    # skipna=False: the first bar has no prev_close, so two of the three TR
    # components are NaN — the bar's TR must propagate as NaN, not silently
    # fall back to (high - low) which would understate the warmup ATR.
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1, skipna=False)
    out = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    out.name = f"atr_{period}"
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA (span=period, adjust=False)."""
    out = series.ewm(span=period, adjust=False).mean()
    out.name = f"ema_{period}"
    return out


def rvol_time_normalized(bars_5m: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Each bar's volume divided by the mean volume of the *same 5-min slot*
    over the prior ``lookback`` sessions.

    Compares 09:20 vs prior 09:20s, not vs a flat rolling mean — opening bars
    are intrinsically heavier than 11:00 bars and a flat mean would call every
    open "high RVOL." See §3.2 Setup B and §7 Q3.

    Returns NaN where there is no prior history for that slot.
    """
    if bars_5m.empty:
        return pd.Series(dtype=float)

    ist_idx = _ensure_ist_index(bars_5m.index)
    df = pd.DataFrame(
        {
            "volume": bars_5m["volume"].values,
            "slot": ist_idx.time,
            "session_date": ist_idx.date,
        },
        index=bars_5m.index,
    )

    out = pd.Series(index=bars_5m.index, dtype=float)
    for slot, slot_group in df.groupby("slot", sort=False):
        ordered = slot_group.sort_values("session_date")
        # mean of prior `lookback` sessions at this slot, not including current
        prior_mean = (
            ordered["volume"].shift(1).rolling(lookback, min_periods=1).mean()
        )
        ratio = ordered["volume"] / prior_mean
        out.loc[ordered.index] = ratio.values
    out.name = f"rvol_tn_{lookback}"
    return out


def opening_range(
    bars_5m: pd.DataFrame, session_date: date, minutes: int = 15
) -> tuple[float, float]:
    """High/low of the first ``minutes`` of ``session_date`` (default 09:15–09:30)."""
    sess = _bars_for_session(bars_5m, session_date)
    if sess.empty:
        return (float("nan"), float("nan"))
    start = pd.Timestamp.combine(session_date, SESSION_OPEN).tz_localize(IST)
    end = start + pd.Timedelta(minutes=minutes)
    ist_idx = _ensure_ist_index(sess.index)
    in_or = (ist_idx >= start) & (ist_idx < end)
    or_bars = sess.loc[in_or]
    if or_bars.empty:
        return (float("nan"), float("nan"))
    return (float(or_bars["high"].max()), float(or_bars["low"].min()))


def prior_day_high_low(
    bars_daily: pd.DataFrame, session_date: date
) -> tuple[float, float]:
    """PDH/PDL — high/low of the most recent session strictly before ``session_date``."""
    if bars_daily.empty:
        return (float("nan"), float("nan"))
    dates = _daily_dates(bars_daily)
    prior_positions = [i for i, d in enumerate(dates) if d < session_date]
    if not prior_positions:
        return (float("nan"), float("nan"))
    last_pos = max(prior_positions, key=lambda i: dates[i])
    row = bars_daily.iloc[last_pos]
    return (float(row["high"]), float(row["low"]))


def relative_strength(
    symbol_daily: pd.DataFrame, nifty_daily: pd.DataFrame, lookback: int = 20
) -> float:
    """Symbol return minus Nifty return over the trailing ``lookback`` sessions.

    Uses the last ``lookback + 1`` closes from each frame. Returns NaN if either
    frame is too short.
    """
    if len(symbol_daily) < lookback + 1 or len(nifty_daily) < lookback + 1:
        return float("nan")
    sym = symbol_daily["close"]
    nif = nifty_daily["close"]
    sym_ret = sym.iloc[-1] / sym.iloc[-lookback - 1] - 1.0
    nif_ret = nif.iloc[-1] / nif.iloc[-lookback - 1] - 1.0
    return float(sym_ret - nif_ret)


def volume_ratio(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    as_of: pd.Timestamp | datetime | None = None,
) -> float:
    """Today's session volume divided by what we'd expect at this elapsed
    fraction of the session, using a 10-day average daily volume.

    ``as_of=None`` → uses ``datetime.now(IST)`` (live path).
    ``as_of=<Timestamp>`` → uses that timestamp's date+time (backtest replay).

    Accepts either yfinance-style uppercase OHLCV columns or lowercase.
    Above 2.0 = unusual activity. Above 3.0 = strong institutional footprint.
    """
    if intraday.empty or daily.empty or len(daily) < 10:
        return 1.0

    intra_vol_col = "Volume" if "Volume" in intraday.columns else "volume"
    daily_vol_col = "Volume" if "Volume" in daily.columns else "volume"

    if as_of is None:
        as_of = datetime.now(IST)

    today_date = as_of.date()
    today_data = intraday[intraday.index.date == today_date]
    if today_data.empty:
        return 1.0
    today_vol = float(today_data[intra_vol_col].sum())

    avg_daily_vol = float(daily[daily_vol_col].tail(10).mean())
    if avg_daily_vol == 0:
        return 1.0

    # Fraction of NSE session elapsed (09:15 to 15:30 = 375 minutes)
    t = as_of.time()
    if t < SESSION_OPEN:
        fraction = 0.01
    elif t > SESSION_CLOSE:
        fraction = 1.0
    else:
        elapsed = (t.hour - 9) * 60 + t.minute - 15
        fraction = max(0.05, elapsed / 375)

    expected = avg_daily_vol * fraction
    return float(today_vol / expected) if expected > 0 else 1.0


def is_above_vwap(bar, vwap_value: float) -> bool:
    """True if the bar's close is strictly above ``vwap_value``.

    Accepts a Series/dict (uses ``close``) or a raw scalar price.
    """
    if isinstance(bar, pd.Series):
        price = bar["close"]
    elif isinstance(bar, dict):
        price = bar["close"]
    else:
        price = bar
    return bool(float(price) > float(vwap_value))
