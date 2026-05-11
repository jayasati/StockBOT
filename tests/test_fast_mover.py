"""Tick-level fast-move detector — `data/fast_mover.py`.

Covers the alert-firing rules: window range threshold, cooldown,
session filter, watchlist filter, and direction inference. The detector
is synchronous and easy to drive — just feed it ``sf`` tick dicts in
order."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from data.fast_mover import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_PCT_THRESHOLD,
    DEFAULT_WINDOW_S,
    FastMove,
    FastMover,
    format_fast_move,
)

IST = ZoneInfo("Asia/Kolkata")


def _tick(symbol: str, ltp: float, ts_ist: datetime) -> dict:
    return {
        "type": "sf",
        "symbol": symbol,
        "ltp": ltp,
        "last_traded_time": int(ts_ist.timestamp()),
    }


def _collect():
    """Build a (callback, list) pair: callback appends each FastMove to the
    list so tests can introspect what fired."""
    out: list[FastMove] = []

    def cb(ev: FastMove) -> None:
        out.append(ev)

    return cb, out


# Use a clearly-in-session base time (11:00 IST on a Monday) for all tests.
T0 = datetime(2026, 5, 11, 11, 0, 0, tzinfo=IST)


def _at(seconds: int) -> datetime:
    return datetime.fromtimestamp(T0.timestamp() + seconds, tz=IST)


# ---------------------------------------------------------------------------
# Threshold / window
# ---------------------------------------------------------------------------

def test_fires_when_range_exceeds_threshold():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(0)))
    fm.on_tick(_tick("NSE:X-EQ", 1010.0, _at(30)))   # +1% — below threshold
    assert sink == []
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, _at(60)))   # +2.5% range → fires
    assert len(sink) == 1
    ev = sink[0]
    assert ev.symbol == "NSE:X-EQ"
    assert ev.pct == pytest.approx(2.5)
    assert ev.direction == "↑"
    assert ev.first_price == 1000.0
    assert ev.last_price == 1025.0


def test_below_threshold_does_not_fire():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    for i, p in enumerate([1000.0, 1005.0, 1015.0, 1019.0]):
        fm.on_tick(_tick("NSE:X-EQ", p, _at(i * 30)))
    # Range = (1019-1000)/1000 = 1.9% — under 2.0%.
    assert sink == []


def test_old_ticks_drop_out_of_window():
    """A tick from 4 minutes ago shouldn't anchor the window when the
    detector's window is 3 minutes."""
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(0)))
    # Window expires before the next tick — only one tick survives.
    fm.on_tick(_tick("NSE:X-EQ", 1030.0, _at(240)))  # +3%, but only one in window
    assert sink == []
    # Adding a third tick close to the second: 1030 → 1031, range 0.1%.
    fm.on_tick(_tick("NSE:X-EQ", 1031.0, _at(245)))
    assert sink == []


def test_direction_down_for_falling_price():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1100.0, _at(0)))
    fm.on_tick(_tick("NSE:X-EQ", 1050.0, _at(120)))   # -4.5% range, last < first
    assert len(sink) == 1
    assert sink[0].direction == "↓"
    assert sink[0].pct == pytest.approx((1100 - 1050) / 1050 * 100)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

def test_cooldown_suppresses_re_alert():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(0)))
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, _at(60)))    # alert #1 (cooldown→360)
    assert len(sink) == 1
    fm.on_tick(_tick("NSE:X-EQ", 1050.0, _at(120)))   # in cooldown — suppressed
    assert len(sink) == 1
    # Past cooldown but only one tick in the 180s window so far → no fire yet.
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(370)))
    assert len(sink) == 1
    # Two ticks in the window AND past cooldown → alert #2.
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, _at(400)))
    assert len(sink) == 2


def test_cooldown_is_per_symbol():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(0)))
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, _at(60)))    # X fires
    fm.on_tick(_tick("NSE:Y-EQ", 500.0, _at(60)))
    fm.on_tick(_tick("NSE:Y-EQ", 515.0, _at(120)))    # Y fires independently
    assert len(sink) == 2
    assert {sink[0].symbol, sink[1].symbol} == {"NSE:X-EQ", "NSE:Y-EQ"}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_watchlist_filter_drops_off_universe_symbols():
    cb, sink = _collect()
    fm = FastMover(
        window_s=180, pct_threshold=2.0, cooldown_s=300,
        on_alert=cb, watchlist={"NSE:X-EQ"},
    )
    fm.on_tick(_tick("NSE:Y-EQ", 100.0, _at(0)))
    fm.on_tick(_tick("NSE:Y-EQ", 110.0, _at(60)))     # would fire if tracked
    assert sink == []


def test_out_of_session_ticks_ignored():
    """A pre-open print at 09:00 IST must not feed the detector."""
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    pre_open = datetime(2026, 5, 11, 9, 0, 0, tzinfo=IST)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, pre_open))
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, datetime(
        2026, 5, 11, 9, 2, 0, tzinfo=IST,
    )))
    assert sink == []


def test_non_sf_messages_ignored():
    cb, sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick({"type": "if", "symbol": "NSE:X-EQ", "ltp": 1000.0})
    fm.on_tick({"type": "sf"})  # missing fields
    fm.on_tick({"type": "sf", "symbol": "NSE:X-EQ", "ltp": None, "last_traded_time": 1})
    assert sink == []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_stats_counters_advance_on_ticks_and_alerts():
    cb, _sink = _collect()
    fm = FastMover(window_s=180, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    fm.on_tick(_tick("NSE:X-EQ", 1000.0, _at(0)))
    fm.on_tick(_tick("NSE:X-EQ", 1025.0, _at(60)))
    s = fm.stats()
    assert s["ticks"] == 2
    assert s["alerts"] == 1
    assert s["tracked_symbols"] == 1


def test_format_fast_move_contains_key_fields():
    ev = FastMove(
        symbol="NSE:RELIANCE-EQ", direction="↑", pct=2.5,
        high=1025.0, low=1000.0,
        first_price=1000.0, last_price=1025.0, window_s=60,
    )
    out = format_fast_move(ev)
    assert "RELIANCE" in out
    assert "↑" in out
    assert "2.50" in out
    assert "60s" in out


def test_constructor_rejects_invalid_params():
    cb, _sink = _collect()
    with pytest.raises(ValueError):
        FastMover(window_s=0, pct_threshold=2.0, cooldown_s=300, on_alert=cb)
    with pytest.raises(ValueError):
        FastMover(window_s=180, pct_threshold=0, cooldown_s=300, on_alert=cb)


def test_defaults_are_sensible():
    """Sanity: defaults catch a PVR-style move (1100 → 1056 in 2 min)."""
    cb, sink = _collect()
    fm = FastMover(on_alert=cb)
    assert fm._window_s == DEFAULT_WINDOW_S
    assert fm._threshold == DEFAULT_PCT_THRESHOLD
    assert fm._cooldown_s == DEFAULT_COOLDOWN_S
    fm.on_tick(_tick("NSE:PVRINOX-EQ", 1100.0, _at(0)))
    fm.on_tick(_tick("NSE:PVRINOX-EQ", 1056.0, _at(120)))  # -4% in 2 min
    assert len(sink) == 1
    assert sink[0].direction == "↓"
