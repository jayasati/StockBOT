"""NSE market-hours helpers and session-time constants.

Single source of truth for SESSION_OPEN / SESSION_CLOSE / SESSION_LAST_BAR_OPEN
and the ``in_session()`` predicate. Imported by costs.py, features.py,
data/realtime_feed.py, fyers_client/livefeed.py."""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd

from .config import IST

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
SESSION_LAST_BAR_OPEN = time(15, 25)  # last valid 5-min slot starts here


def in_session(ts=None, *, mode: str = "alert") -> bool:
    """Is ``ts`` (default: now in IST) inside the NSE cash-equity session?

    ``mode='alert'`` (default): allows up to ``SESSION_CLOSE`` (15:30). What
        the scanner uses to decide whether to scan / alert.
    ``mode='bar_slot'``: allows up to ``SESSION_LAST_BAR_OPEN`` (15:25). Used
        by the bar aggregator — ticks landing in slot 15:30 belong to the
        next session boundary, so they're rejected.
    """
    if ts is None:
        ts = datetime.now(IST)
    if isinstance(ts, pd.Timestamp):
        weekday = ts.weekday()
        t = ts.time()
    elif isinstance(ts, datetime):
        weekday = ts.weekday()
        t = ts.time()
    else:
        raise TypeError(
            f"ts must be datetime, pandas.Timestamp, or None; got {type(ts).__name__}"
        )

    if weekday >= 5:
        return False
    upper = SESSION_LAST_BAR_OPEN if mode == "bar_slot" else SESSION_CLOSE
    return SESSION_OPEN <= t <= upper


def is_market_open() -> bool:
    """NSE: Mon-Fri 09:15-15:30 IST. (Compatibility alias for ``in_session()``.)"""
    return in_session()


def seconds_until_market_open() -> int:
    """Approximate seconds until next market open (for sleep optimization)."""
    now = datetime.now(IST)
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() > SESSION_CLOSE or now.weekday() >= 5:
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        target = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return max(60, int((target - now).total_seconds()))
