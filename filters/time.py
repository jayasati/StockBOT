"""Time-of-day filters: penalise signals fired in unfavourable
windows. Multipliers below 1.0 demote the signal's confidence.

Windows + multipliers (Phase-6 spec):
  09:15-09:30 IST  — opening_window_penalty    × 0.3
  12:30-13:30 IST  — lunch_window_penalty      × 0.7
  14:45-15:30 IST  — end_of_day_penalty        × 0.6

Outside these windows each filter returns ``None`` (no
multiplier). At most one filter fires per tick because the windows
don't overlap."""
from __future__ import annotations

from datetime import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.scoring import StockSignals
    from .chain import FilterContext


# Window boundaries — module-level so tests can monkeypatch.
OPENING_WINDOW_START = _time(9, 15)
OPENING_WINDOW_END = _time(9, 30)
LUNCH_WINDOW_START = _time(12, 30)
LUNCH_WINDOW_END = _time(13, 30)
END_OF_DAY_START = _time(14, 45)
END_OF_DAY_END = _time(15, 30)

OPENING_MULT = 0.3
LUNCH_MULT = 0.7
END_OF_DAY_MULT = 0.6


def opening_window_penalty(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """First 15 min of the session: spreads are wide, prices are
    discovering, and most setups are noise. Demote heavily."""
    t = ctx.now.timetz().replace(tzinfo=None)
    if OPENING_WINDOW_START <= t < OPENING_WINDOW_END:
        return ("opening_window", OPENING_MULT)
    return None


def lunch_window_penalty(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """12:30-13:30 IST: NIFTY volume drops ~40% on average; breakouts
    here are typically false (then resolve in the afternoon trend)."""
    t = ctx.now.timetz().replace(tzinfo=None)
    if LUNCH_WINDOW_START <= t < LUNCH_WINDOW_END:
        return ("lunch_window", LUNCH_MULT)
    return None


def end_of_day_penalty(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """14:45-15:30 IST: same-day TIMEOUT in :mod:`paper.tracker` is
    15:30, so a 14:45 entry has < 45 min to develop. Demote to
    reflect the squeezed runway."""
    t = ctx.now.timetz().replace(tzinfo=None)
    if END_OF_DAY_START <= t <= END_OF_DAY_END:
        return ("end_of_day", END_OF_DAY_MULT)
    return None


TIME_FILTERS = (
    opening_window_penalty,
    lunch_window_penalty,
    end_of_day_penalty,
)
