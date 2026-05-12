"""Intraday-specific levels — PDH/PDL/PDC, Opening Range, Initial Balance,
Pivot Points (Classic, Fibonacci, Camarilla).

These are session-level constants, not per-bar series. They return
``dict[str, float]`` and Phase 7 scoring uses them as horizontal
reference lines (price-to-level distance, level breakouts).

Inputs use lowercase OHLCV. Dates are ``datetime.date`` in IST."""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

SESSION_OPEN_TIME = time(9, 15)   # NSE cash equity
INITIAL_BALANCE_MINUTES = 60      # Steidlmayer convention


def previous_day_hlc(
    daily_df: pd.DataFrame, session_date: date
) -> dict[str, float]:
    """Prior trading day's high/low/close.

    ``daily_df`` is a DataFrame indexed by date (or DatetimeIndex with
    midnight timestamps) with at least ``high``, ``low``, ``close``
    columns. Returns ``{pdh, pdl, pdc}`` for the most recent session
    strictly before ``session_date``. Missing data → NaN values
    (not an exception) so the scoring path can treat them neutrally."""
    if daily_df.empty:
        return {"pdh": np.nan, "pdl": np.nan, "pdc": np.nan}
    # Normalise index to dates.
    idx = daily_df.index
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.date
    else:
        dates = np.array([
            d.date() if hasattr(d, "date") else d for d in idx
        ], dtype=object)
    mask = dates < session_date
    if not mask.any():
        return {"pdh": np.nan, "pdl": np.nan, "pdc": np.nan}
    prev_idx = np.flatnonzero(mask)[-1]
    row = daily_df.iloc[prev_idx]
    cols = {c.lower(): c for c in daily_df.columns}
    return {
        "pdh": float(row[cols["high"]]),
        "pdl": float(row[cols["low"]]),
        "pdc": float(row[cols["close"]]),
    }


def _session_bars(
    bars_5m: pd.DataFrame, session_date: date
) -> pd.DataFrame:
    """Slice a 5m DataFrame down to a single IST session (09:15–15:30).

    The caller's index must be tz-aware. We compare each bar's local
    date in the index timezone."""
    if bars_5m.empty:
        return bars_5m
    idx = bars_5m.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("bars_5m must have a DatetimeIndex")
    if idx.tz is None:
        raise ValueError("bars_5m index must be tz-aware (IST)")
    mask = pd.Series(idx.date, index=idx) == session_date
    return bars_5m.loc[mask.to_numpy()]


def opening_range(
    bars_5m: pd.DataFrame, session_date: date, minutes: int = 15
) -> dict[str, float]:
    """High/low of the first ``minutes`` of the session.

    minutes ∈ {5, 15, 30, 60}. Returns ``{orh, orl, or_mid}``. NaN
    values if the session hasn't generated enough bars yet (early in the
    day) or if the session date isn't present."""
    if minutes not in (5, 15, 30, 60):
        raise ValueError(f"minutes must be one of 5,15,30,60; got {minutes}")
    if bars_5m.empty:
        return {"orh": np.nan, "orl": np.nan, "or_mid": np.nan}
    session = _session_bars(bars_5m, session_date)
    if session.empty:
        return {"orh": np.nan, "orl": np.nan, "or_mid": np.nan}
    n_bars = minutes // 5
    or_slice = session.iloc[:n_bars]
    if len(or_slice) < n_bars:
        # Not enough bars yet — fail soft.
        return {"orh": np.nan, "orl": np.nan, "or_mid": np.nan}
    orh = float(or_slice["high"].max())
    orl = float(or_slice["low"].min())
    return {"orh": orh, "orl": orl, "or_mid": (orh + orl) / 2.0}


def initial_balance(
    bars_5m: pd.DataFrame, session_date: date
) -> dict[str, float]:
    """First 60 minutes of the session — Steidlmayer's "initial balance".

    Equivalent to ``opening_range(minutes=60)`` but returns ``{ib_high,
    ib_low}`` and is registered separately because Phase 7 weighs IB
    differently from a 15m ORB."""
    orb = opening_range(bars_5m, session_date, minutes=INITIAL_BALANCE_MINUTES)
    return {"ib_high": orb["orh"], "ib_low": orb["orl"]}


def pivot_points(
    prev_day_hlc: dict[str, float], method: str = "classic"
) -> dict[str, float]:
    """Floor pivot points derived from prior-day H/L/C.

    methods:
      classic    — TradingView default; pivot = (H+L+C)/3; symmetric R/S.
      fibonacci  — pivot identical; R/S levels at 0.382, 0.618, 1.0 of
                   the prior-day range above/below pivot.
      camarilla  — pivot = C; R/S levels at C ± k * (H - L), where
                   k = 1.1/12, 1.1/6, 1.1/4, 1.1/2 for L1..L4. Returns
                   r1..r3 / s1..s3 (L4/-L4 omitted; rarely used by
                   intraday traders)."""
    pdh = prev_day_hlc.get("pdh")
    pdl = prev_day_hlc.get("pdl")
    pdc = prev_day_hlc.get("pdc")
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (pdh, pdl, pdc)):
        return {k: np.nan for k in ("pivot", "r1", "r2", "r3", "s1", "s2", "s3")}

    pdh = float(pdh); pdl = float(pdl); pdc = float(pdc)
    rng = pdh - pdl

    if method == "classic":
        p = (pdh + pdl + pdc) / 3.0
        r1 = 2.0 * p - pdl
        s1 = 2.0 * p - pdh
        r2 = p + rng
        s2 = p - rng
        r3 = pdh + 2.0 * (p - pdl)
        s3 = pdl - 2.0 * (pdh - p)
        return {"pivot": p, "r1": r1, "r2": r2, "r3": r3,
                "s1": s1, "s2": s2, "s3": s3}

    if method == "fibonacci":
        p = (pdh + pdl + pdc) / 3.0
        return {
            "pivot": p,
            "r1": p + 0.382 * rng,
            "r2": p + 0.618 * rng,
            "r3": p + 1.000 * rng,
            "s1": p - 0.382 * rng,
            "s2": p - 0.618 * rng,
            "s3": p - 1.000 * rng,
        }

    if method == "camarilla":
        p = pdc
        return {
            "pivot": p,
            "r1": pdc + rng * 1.1 / 12.0,
            "r2": pdc + rng * 1.1 / 6.0,
            "r3": pdc + rng * 1.1 / 4.0,
            "s1": pdc - rng * 1.1 / 12.0,
            "s2": pdc - rng * 1.1 / 6.0,
            "s3": pdc - rng * 1.1 / 4.0,
        }

    raise ValueError(f"unknown pivot method: {method!r}")
