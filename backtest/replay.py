"""Replay engine: score every historical bar, build alert records."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd

import bot
import features

from .config import COOLDOWN_MINUTES, LOOKBACK_BARS, SCORE_THRESHOLD, WARMUP_BARS

log = logging.getLogger("backtest")


@dataclass
class AlertRecord:
    symbol: str
    timestamp: pd.Timestamp
    score: int
    reasons: str
    entry_price: float
    price_30m: Optional[float]
    price_1d: Optional[float]
    price_5d: Optional[float]
    ret_30m: Optional[float]
    ret_1d: Optional[float]
    ret_5d: Optional[float]


def _forward_intraday_price(
    intraday: pd.DataFrame, t: pd.Timestamp, offset: timedelta
) -> Optional[float]:
    target = t + offset
    future = intraday[intraday.index >= target]
    if len(future) == 0:
        return None
    # Reject if the next bar is more than 1 trading day away (gap over weekend
    # makes "+30min" meaningless).
    if (future.index[0] - t).total_seconds() > 6 * 3600:
        return None
    return float(future["Close"].iloc[0])


def _forward_daily_price(
    daily: pd.DataFrame, t: pd.Timestamp, n_days: int
) -> Optional[float]:
    t_date = t.date() if hasattr(t, "date") else t
    future = daily[daily.index.date > t_date]
    if len(future) >= n_days:
        return float(future["Close"].iloc[n_days - 1])
    return None


def _can_skip_bar(
    intraday: pd.DataFrame, daily_slice: pd.DataFrame, i: int, t: pd.Timestamp,
    threshold: int = SCORE_THRESHOLD,
) -> bool:
    """
    Pre-filter for the post-tuning scoring (see bot.score_stock):
      max without VR>=1.5 = pullback(25) + RSI(15) + VWAP(15) = 55
      max without VR>=1.5 and without pullback = 30

    For threshold >= 60: must have VR>=1.5.
    For threshold < 60 (sweep): must have VR>=1.5 OR pullback in [-3%, 0%]
    of the 20-day high.
    """
    if len(daily_slice) < 20:
        return True

    recent_high = float(daily_slice["High"].tail(20).max())
    current = float(intraday["Close"].iloc[i])
    pct_from_high = (current - recent_high) / recent_high * 100
    in_pullback_zone = -3.0 <= pct_from_high <= 0.0

    # Pre-filter uses the same volume-ratio math as score_stock.
    quick_vr = features.volume_ratio(
        intraday.iloc[max(0, i - 100): i + 1], daily_slice, as_of=t
    )
    high_vr = quick_vr >= 1.5

    if threshold >= 60:
        return not high_vr
    return not (high_vr or in_pullback_zone)


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def replay(
    symbols: list[str],
    intraday_data: dict[str, pd.DataFrame],
    daily_data: dict[str, pd.DataFrame],
    threshold: int = SCORE_THRESHOLD,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> list[AlertRecord]:
    import time as _time
    records: list[AlertRecord] = []
    cooldown_td = timedelta(minutes=cooldown_minutes)
    n_total = len(symbols)
    t_start = _time.monotonic()

    log.info("Replaying %d symbols (threshold=%d, cooldown=%dm)...",
             n_total, threshold, cooldown_minutes)

    for idx, sym in enumerate(symbols, 1):
        sym_start = _time.monotonic()
        intraday = intraday_data.get(sym)
        daily = daily_data.get(sym)
        if intraday is None or daily is None:
            log.info("[%3d/%d] %-14s SKIP (no data)", idx, n_total, sym)
            continue
        if len(intraday) < WARMUP_BARS or len(daily) < 25:
            log.info("[%3d/%d] %-14s SKIP (insufficient history: %d bars / %d days)",
                     idx, n_total, sym, len(intraday), len(daily))
            continue

        sym_alerts_before = len(records)
        last_alert: Optional[pd.Timestamp] = None
        intraday_dates = intraday.index.date  # cache once per symbol
        daily_dates = daily.index.date

        for i in range(WARMUP_BARS, len(intraday)):
            t = intraday.index[i]
            t_date = intraday_dates[i]
            daily_mask = daily_dates < t_date
            daily_slice = daily.loc[daily_mask]
            if len(daily_slice) < 20:
                continue

            if _can_skip_bar(intraday, daily_slice, i, t, threshold=threshold):
                continue

            slice_start = max(0, i - LOOKBACK_BARS)
            intraday_slice = intraday.iloc[slice_start: i + 1]

            signals = bot.score_stock(sym, intraday_slice, daily_slice, as_of=t)
            if signals.score < threshold:
                continue
            if last_alert is not None and (t - last_alert) < cooldown_td:
                continue
            last_alert = t

            entry = signals.price
            p30 = _forward_intraday_price(intraday, t, timedelta(minutes=30))
            p1d = _forward_daily_price(daily, t, 1)
            p5d = _forward_daily_price(daily, t, 5)

            def _ret(p: Optional[float]) -> Optional[float]:
                return (p - entry) / entry * 100 if p is not None else None

            records.append(AlertRecord(
                symbol=sym,
                timestamp=t,
                score=signals.score,
                reasons=", ".join(signals.reasons),
                entry_price=entry,
                price_30m=p30,
                price_1d=p1d,
                price_5d=p5d,
                ret_30m=_ret(p30),
                ret_1d=_ret(p1d),
                ret_5d=_ret(p5d),
            ))

        sym_dur = _time.monotonic() - sym_start
        sym_alerts = len(records) - sym_alerts_before
        elapsed = _time.monotonic() - t_start
        avg_per_sym = elapsed / idx
        eta = avg_per_sym * (n_total - idx)
        pct = idx / n_total * 100
        log.info(
            "[%3d/%d %5.1f%%] %-14s %3d alerts (%d bars in %4.1fs)  "
            "total=%d  elapsed=%s  eta=%s",
            idx, n_total, pct, sym, sym_alerts, len(intraday), sym_dur,
            len(records), _fmt_eta(elapsed), _fmt_eta(eta),
        )

    log.info("Replay complete: %d alerts in %s.",
             len(records), _fmt_eta(_time.monotonic() - t_start))
    return records
