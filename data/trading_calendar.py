"""Lightweight NSE trading calendar.

Used by the multi-timeframe aggregation layer to anchor session-aligned
candle boundaries (09:15 IST open, 15:30 IST close) and to skip non-trading
days. Holidays are kept in a separate dict so future shortened sessions
(Muhurat, special early closes) can be handled without touching the
weekend rule.
"""

from __future__ import annotations

import datetime
from typing import Union

import pandas as pd

IST = "Asia/Kolkata"

SessionDate = Union[datetime.date, pd.Timestamp]

NSE_HOLIDAYS: dict[datetime.date, str] = {}


def _to_date(session_date: SessionDate) -> datetime.date:
    if isinstance(session_date, pd.Timestamp):
        return session_date.date()
    if isinstance(session_date, datetime.datetime):
        return session_date.date()
    if isinstance(session_date, datetime.date):
        return session_date
    raise TypeError(
        f"session_date must be datetime.date or pd.Timestamp, got {type(session_date)!r}"
    )


def get_session_open(session_date: SessionDate) -> pd.Timestamp:
    """Return 09:15:00 Asia/Kolkata for the given date as a tz-aware Timestamp."""
    d = _to_date(session_date)
    return pd.Timestamp(
        year=d.year, month=d.month, day=d.day,
        hour=9, minute=15, tz=IST,
    )


def get_session_close(session_date: SessionDate) -> pd.Timestamp:
    """Return 15:30:00 Asia/Kolkata for the given date as a tz-aware Timestamp."""
    d = _to_date(session_date)
    return pd.Timestamp(
        year=d.year, month=d.month, day=d.day,
        hour=15, minute=30, tz=IST,
    )


def is_trading_day(session_date: SessionDate) -> bool:
    """True if NSE is open on this date (Mon–Fri and not in NSE_HOLIDAYS)."""
    d = _to_date(session_date)
    if d.weekday() >= 5:
        return False
    if d in NSE_HOLIDAYS:
        return False
    return True
