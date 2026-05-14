"""Opening Range Breakout (ORB) — India-staple intraday strategy.

The opening range is the high/low of the first ``minutes`` of the cash
session (09:15-09:30 IST for the default 15-minute window). LONG when the
close breaks ABOVE the OR high; SHORT when it breaks BELOW the OR low.

Classic ORB rules: **one trade per direction per session**. Once a long
or short has been triggered today, no further breakouts are taken on
this symbol that day. Set ``single_shot=False`` to allow re-entries on
each subsequent breakout (which the engine then dedupes by current
position state).

Reuses ``indicators.levels.opening_range`` which validates the session
DataFrame (must have a tz-aware DatetimeIndex)."""
from __future__ import annotations

import pandas as pd

from indicators import levels

from .base import Signal, SignalKind, Strategy


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def __init__(self, minutes: int = 15, single_shot: bool = True) -> None:
        if minutes not in (5, 15, 30, 60):
            raise ValueError(
                f"minutes must be one of (5, 15, 30, 60); got {minutes}"
            )
        super().__init__(minutes=minutes, single_shot=single_shot)
        self.minutes = minutes
        self.single_shot = single_shot
        # Per-(df, session_date) caches: OR levels are immutable for a
        # given session; triggered flag tracks single-shot enforcement.
        self._or_cache: dict[tuple[int, object], dict[str, float]] = {}
        self._triggered: set[tuple[int, object]] = set()

    def _get_or(self, df: pd.DataFrame, session_date) -> dict[str, float]:
        key = (id(df), session_date)
        cached = self._or_cache.get(key)
        if cached is not None:
            return cached
        out = levels.opening_range(df, session_date, minutes=self.minutes)
        self._or_cache[key] = out
        return out

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        ts = df.index[i]
        if not isinstance(ts, pd.Timestamp):
            return None
        session_date = ts.date()
        triggered_key = (id(df), session_date)
        if self.single_shot and triggered_key in self._triggered:
            return None

        or_data = self._get_or(df, session_date)
        orh = or_data["orh"]
        orl = or_data["orl"]
        if pd.isna(orh) or pd.isna(orl):
            return None

        c_prev = float(df["close"].iat[i - 1])
        c_curr = float(df["close"].iat[i])

        if c_prev <= orh and c_curr > orh:
            if self.single_shot:
                self._triggered.add(triggered_key)
            return Signal(SignalKind.ENTER_LONG, "OrbLE")
        if c_prev >= orl and c_curr < orl:
            if self.single_shot:
                self._triggered.add(triggered_key)
            return Signal(SignalKind.ENTER_SHORT, "OrbSE")
        return None
