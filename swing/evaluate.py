"""Swing-signal evaluator.

Single source of truth for the signal logic — both the backtest sweep
(``as_of_date=None``) and the live alert (``as_of_date=<today>``) call
this function."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    BREADTH_THRESHOLD_PCT,
    EMA_PERIOD,
    HORIZONS,
    NIFTY_UP_PCT,
    RANGE_POSITION_THRESHOLD,
    VOLUME_MULTIPLE,
)


def compute_breadth_series(
    daily_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """Return Series(date -> % of symbols closing above their own 20-day EMA)."""
    columns: dict[str, pd.Series] = {}
    for sym, df in daily_data.items():
        ema = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        columns[sym] = (df["Close"] > ema).astype(float)
    wide = pd.DataFrame(columns)
    return wide.mean(axis=1) * 100


@dataclass
class SwingAlert:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    volume_mult: float
    range_position: float
    ema_pct: float
    nifty_change: float
    breadth_pct: float
    regime_ok: bool
    exit_1d: Optional[float]
    exit_3d: Optional[float]
    exit_5d: Optional[float]
    ret_1d: Optional[float]
    ret_3d: Optional[float]
    ret_5d: Optional[float]


def evaluate_swing(
    daily_data: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    apply_regime: bool,
    max_extension_pct: Optional[float] = None,
    as_of_date: Optional[pd.Timestamp] = None,
) -> list[SwingAlert]:
    """Apply the swing signal across ``daily_data``.

    ``as_of_date=None`` (default): replay every qualifying day in the history
        (backtest mode). Each alert carries forward exits at +1/+3/+5 days.
    ``as_of_date=<Timestamp>``: evaluate only that day (alert mode). Forward
        exits and entry price are placeholders (the +1 open isn't known yet);
        callers should consume only the signal-bar fields."""
    breadth_series = compute_breadth_series(daily_data)
    nifty_change = nifty["Close"].pct_change() * 100

    alerts: list[SwingAlert] = []

    for sym, df in daily_data.items():
        if len(df) < EMA_PERIOD + 1:
            continue

        ema = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        avg_vol = df["Volume"].rolling(EMA_PERIOD).mean()
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        range_pos = (df["Close"] - df["Low"]) / rng

        if as_of_date is not None:
            if as_of_date not in df.index:
                continue
            target_i = df.index.get_loc(as_of_date)
            if not isinstance(target_i, (int, np.integer)):
                continue
            if target_i < EMA_PERIOD:
                continue
            i_range = [int(target_i)]
        else:
            if len(df) < EMA_PERIOD + max(HORIZONS) + 1:
                continue
            i_range = range(EMA_PERIOD, len(df) - max(HORIZONS))

        for i in i_range:
            day = df.index[i]
            close = float(df["Close"].iloc[i])
            vol = float(df["Volume"].iloc[i])
            av = avg_vol.iloc[i]
            if pd.isna(av) or av == 0:
                continue
            vol_mult = vol / av
            rp = range_pos.iloc[i]
            if pd.isna(rp):
                continue
            ema_now = float(ema.iloc[i])
            ema_pct = (close - ema_now) / ema_now * 100 if ema_now > 0 else 0.0

            if vol_mult < VOLUME_MULTIPLE:
                continue
            if rp < RANGE_POSITION_THRESHOLD:
                continue
            if close <= ema_now:
                continue
            if max_extension_pct is not None and ema_pct > max_extension_pct:
                continue

            n_change = nifty_change.get(day, np.nan)
            n_change = float(n_change) if not pd.isna(n_change) else 0.0
            breadth = float(breadth_series.get(day, 0.0))
            regime_ok = (
                n_change >= NIFTY_UP_PCT
                and breadth >= BREADTH_THRESHOLD_PCT
            )
            if apply_regime and not regime_ok:
                continue

            entry_idx = i + 1
            if entry_idx < len(df):
                entry_date = df.index[entry_idx]
                entry_open = float(df["Open"].iloc[entry_idx])
            else:
                # Alert-mode tail: tomorrow's open isn't in the data yet.
                entry_date = day
                entry_open = close

            def _exit(h: int) -> Optional[float]:
                idx = i + h
                if idx >= len(df):
                    return None
                return float(df["Close"].iloc[idx])

            def _ret(p: Optional[float]) -> Optional[float]:
                return (p - entry_open) / entry_open * 100 if p is not None else None

            x1, x3, x5 = _exit(1), _exit(3), _exit(5)
            alerts.append(SwingAlert(
                symbol=sym,
                signal_date=day,
                entry_date=entry_date,
                entry_price=entry_open,
                volume_mult=float(vol_mult),
                range_position=float(rp),
                ema_pct=float(ema_pct),
                nifty_change=n_change,
                breadth_pct=breadth,
                regime_ok=regime_ok,
                exit_1d=x1, exit_3d=x3, exit_5d=x5,
                ret_1d=_ret(x1), ret_3d=_ret(x3), ret_5d=_ret(x5),
            ))

    return alerts
