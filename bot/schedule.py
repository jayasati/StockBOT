"""NSE market-hours helpers for the scheduler."""
from __future__ import annotations

from datetime import datetime, time, timedelta

from .config import IST


def is_market_open() -> bool:
    """NSE: Mon-Fri 09:15-15:30 IST."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


def seconds_until_market_open() -> int:
    """Approximate seconds until next market open (for sleep optimization)."""
    now = datetime.now(IST)
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() > time(15, 30) or now.weekday() >= 5:
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        target = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return max(60, int((target - now).total_seconds()))
