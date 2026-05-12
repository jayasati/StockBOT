"""Multi-timeframe resampling for the indicator stack.

Single helper: ``resample_ohlcv(bars, target_tf)`` aggregates a fine-
grained intraday OHLCV frame into a coarser one using the canonical
rules:

    open   = first    high = max
    low    = min      close = last     volume = sum

Bar-label convention: ``label='left', closed='left'`` so the resampled
bar's timestamp is the OPEN time of its window — matching what we
already store in ``bars_5m`` and what TradingView shows on its intraday
charts. (The Phase-4 spec suggested ``label='right', closed='right'``,
but that puts a 09:15–09:20 bar at timestamp 09:20, which contradicts
the existing schema; the bar would also include the 09:20 print
instead of leaving it for the next bar. Diverging from the spec here
on purpose; commented for the next maintainer.)

Bucket alignment for 30m / 60m: pandas' default resample anchors to
the clock (00:30 / 01:00 / …). For NSE intraday, traders want buckets
anchored to the session open (09:15, 09:45, 10:15, … for 30m;
09:15, 10:15, 11:15, … for 60m). Passing ``origin='start'`` snaps the
resampler to the first bar's timestamp, which is the session open as
long as the caller's frame starts at it.

Daily resampling (``target_tf='1d'``) is intentionally **not** supported
here: NSE session boundaries (09:15-15:30 IST) don't align with
pandas' midnight-anchored ``1D`` rule. Callers that need daily bars
should pass them in directly via the existing ``daily_df`` channel.
"""
from __future__ import annotations

import pandas as pd

_RESAMPLE_RULES: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
}

_OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(bars: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample to a coarser intraday timeframe.

    Identity transform when ``target_tf`` matches the input's native
    cadence (e.g. 5m → 5m). All-NaN rows (empty resample buckets) are
    dropped so consumers don't have to filter."""
    if target_tf not in _RESAMPLE_RULES:
        raise ValueError(
            f"unsupported target_tf {target_tf!r}; "
            f"valid: {sorted(_RESAMPLE_RULES)}"
        )
    if bars.empty:
        return bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    rule = _RESAMPLE_RULES[target_tf]
    out = (
        bars.resample(rule, label="left", closed="left", origin="start")
        .agg(_OHLCV_AGG)
        .dropna(how="all")
    )
    return out
