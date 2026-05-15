"""Tests for data.backfill.

The orchestrator has three behaviours worth pinning:

1. Window math — pre-market, weekend, and post-market return the
   correct catch-up range (or skip entirely).
2. Gap detection — symbols that already have today's bars are not
   re-fetched. Symbols with a partial gap are filled. INSERT OR IGNORE
   makes a re-run a no-op.
3. Failure isolation — one symbol's Fyers error must not abort the
   batch; the others still fill.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data import backfill, realtime_feed
from data.backfill import (
    _compute_window,
    _expected_bar_count,
    _floor_to_5m,
)
from data.realtime_feed import BarAggregator

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

def test_floor_to_5m_aligns_to_session_grid():
    """09:17 → 09:15, 09:20 → 09:20, 09:21 → 09:20."""
    assert _floor_to_5m(datetime(2026, 5, 4, 9, 17, 30, tzinfo=IST)) == \
        datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    assert _floor_to_5m(datetime(2026, 5, 4, 9, 20, 0, tzinfo=IST)) == \
        datetime(2026, 5, 4, 9, 20, tzinfo=IST)
    assert _floor_to_5m(datetime(2026, 5, 4, 9, 21, 0, tzinfo=IST)) == \
        datetime(2026, 5, 4, 9, 20, tzinfo=IST)


def test_expected_bar_count_inclusive_both_ends():
    """[09:15, 09:15] = 1 bar (just the opener). [09:15, 09:40] = 6
    bars (09:15, 09:20, 09:25, 09:30, 09:35, 09:40) — exactly the
    number missed in a 09:45 start."""
    open_ts = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    assert _expected_bar_count(open_ts, open_ts) == 1
    assert _expected_bar_count(
        open_ts, datetime(2026, 5, 4, 9, 40, tzinfo=IST)
    ) == 6


def test_expected_bar_count_pre_open_is_zero():
    """A cutoff before session open means nothing to fill."""
    open_ts = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    pre = datetime(2026, 5, 4, 9, 10, tzinfo=IST)
    assert _expected_bar_count(open_ts, pre) == 0


# ---------------------------------------------------------------------------
# _compute_window — when does backfill run?
# ---------------------------------------------------------------------------

def test_compute_window_skips_weekend():
    """Sunday 2026-05-03 — no trading day, no window."""
    sun = datetime(2026, 5, 3, 11, 0, tzinfo=IST)
    assert _compute_window(sun) is None


def test_compute_window_skips_pre_market():
    """Monday 08:00 — before session open."""
    pre = datetime(2026, 5, 4, 8, 0, tzinfo=IST)
    assert _compute_window(pre) is None


def test_compute_window_returns_gap_for_mid_session_start():
    """The original user scenario: bot starts at 09:45. The last
    fully-closed slot is 09:40 (09:45 is still in progress), and the
    expected bar count is 6 (09:15..09:40 inclusive)."""
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    window = _compute_window(now)
    assert window is not None
    session_open, cutoff = window
    assert session_open.time().isoformat() == "09:15:00"
    assert cutoff.time().isoformat() == "09:40:00"
    assert _expected_bar_count(session_open, cutoff) == 6


def test_compute_window_handles_post_market_restart():
    """Restart at 16:00 should still let us backfill the whole day
    (a long network outage during the session is exactly this case)."""
    now = datetime(2026, 5, 4, 16, 0, tzinfo=IST)
    window = _compute_window(now)
    assert window is not None
    _, cutoff = window
    assert cutoff.time().isoformat() == "15:55:00"


def test_compute_window_returns_none_before_first_close():
    """A start at 09:17 (before any bar has closed) — nothing to fill."""
    now = datetime(2026, 5, 4, 9, 17, tzinfo=IST)
    assert _compute_window(now) is None


# ---------------------------------------------------------------------------
# End-to-end: fake history client + real aggregator on temp DB
# ---------------------------------------------------------------------------

def _make_history_df(session_open: datetime, n_bars: int) -> pd.DataFrame:
    """Build a DataFrame matching what fetch_history would return for
    ``n_bars`` consecutive 5-min slots starting at ``session_open``."""
    idx = pd.DatetimeIndex(
        [session_open + timedelta(minutes=5 * i) for i in range(n_bars)],
        tz=IST, name="ts",
    )
    return pd.DataFrame(
        {
            "open":   [100.0 + i for i in range(n_bars)],
            "high":   [101.0 + i for i in range(n_bars)],
            "low":    [ 99.0 + i for i in range(n_bars)],
            "close":  [100.5 + i for i in range(n_bars)],
            "volume": [1000.0 * (i + 1) for i in range(n_bars)],
        },
        index=idx,
    )


@pytest.fixture
def temp_aggregator(tmp_path):
    """Swap the realtime_feed singleton for one backed by a temp DB so
    bars_5m writes don't leak into the real alerts.db."""
    agg = BarAggregator(db_path=tmp_path / "backfill_test.db")
    realtime_feed.set_aggregator(agg)
    yield agg
    realtime_feed.set_aggregator(None)  # reset so other tests get a fresh singleton


class _RecordingFakeFyers:
    """A fake ``FyersModel`` that returns canned data per symbol and
    records every call. ``per_symbol`` maps symbol → response (dict or
    Exception); missing symbols return an empty-ok response."""

    def __init__(self, per_symbol: dict):
        self.per_symbol = per_symbol
        self.calls: list[dict] = []

    def history(self, data):
        self.calls.append(dict(data))
        resp = self.per_symbol.get(data["symbol"], {"s": "ok", "candles": []})
        if isinstance(resp, Exception):
            raise resp
        return resp


def _candles_from_df(df: pd.DataFrame) -> list[list]:
    """Inverse of _candles_to_df — for building fake Fyers responses."""
    out = []
    for ts, row in zip(df.index, df.itertuples(index=False)):
        out.append([
            int(pd.Timestamp(ts).timestamp()),
            float(row.open), float(row.high), float(row.low),
            float(row.close), float(row.volume),
        ])
    return out


def test_backfill_fills_full_gap(temp_aggregator):
    """The original scenario: at 09:45 the DB is empty. backfill_today
    should insert all 6 missing bars for each symbol."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    df = _make_history_df(session_open, n_bars=6)
    fake = _RecordingFakeFyers({
        "NSE:RELIANCE-EQ": {"s": "ok", "candles": _candles_from_df(df)},
        "NSE:TCS-EQ":      {"s": "ok", "candles": _candles_from_df(df)},
    })
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ", "NSE:TCS-EQ"], now=now, client=fake,
    ))
    assert filled == {"NSE:RELIANCE-EQ": 6, "NSE:TCS-EQ": 6}
    out = temp_aggregator.get_5m_bars("NSE:RELIANCE-EQ")
    assert len(out) == 6


def test_backfill_skips_symbols_with_full_data(temp_aggregator):
    """If a symbol already has all 6 bars in bars_5m (e.g. a restart
    seconds after the first backfill), skip the fetch entirely — no
    Fyers call is made for it."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    full_df = _make_history_df(session_open, n_bars=6)
    temp_aggregator.seed_bars("NSE:RELIANCE-EQ", full_df)  # pre-fill

    fake = _RecordingFakeFyers({
        "NSE:TCS-EQ": {"s": "ok", "candles": _candles_from_df(full_df)},
    })
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ", "NSE:TCS-EQ"], now=now, client=fake,
    ))
    assert filled == {"NSE:TCS-EQ": 6}
    queried_symbols = {c["symbol"] for c in fake.calls}
    assert queried_symbols == {"NSE:TCS-EQ"}


def test_backfill_isolates_per_symbol_failures(temp_aggregator):
    """One symbol's Fyers error doesn't abort the batch — the rest
    still fill."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    df = _make_history_df(session_open, n_bars=6)
    fake = _RecordingFakeFyers({
        "NSE:BOGUS-EQ":    {"s": "error", "code": -50, "message": "Invalid"},
        "NSE:RELIANCE-EQ": {"s": "ok", "candles": _candles_from_df(df)},
    })
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:BOGUS-EQ", "NSE:RELIANCE-EQ"], now=now, client=fake,
    ))
    assert "NSE:RELIANCE-EQ" in filled
    assert filled["NSE:RELIANCE-EQ"] == 6
    assert "NSE:BOGUS-EQ" not in filled


def test_backfill_isolates_sdk_exceptions(temp_aggregator):
    """A raw SDK exception (network glitch, timeout) is also caught."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    df = _make_history_df(session_open, n_bars=6)
    fake = _RecordingFakeFyers({
        "NSE:RELIANCE-EQ": {"s": "ok", "candles": _candles_from_df(df)},
        "NSE:DOWN-EQ":     ConnectionError("network down"),
    })
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ", "NSE:DOWN-EQ"], now=now, client=fake,
    ))
    assert filled == {"NSE:RELIANCE-EQ": 6}


def test_backfill_pre_market_is_noop(temp_aggregator):
    """Bot started at 08:00 → window is None, no fetches happen."""
    fake = _RecordingFakeFyers({})
    now = datetime(2026, 5, 4, 8, 0, tzinfo=IST)
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ"], now=now, client=fake,
    ))
    assert filled == {}
    assert fake.calls == []


def test_backfill_is_idempotent(temp_aggregator):
    """Calling backfill_today twice in a row inserts no extra bars —
    the second call finds the same expected count and skips."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    df = _make_history_df(session_open, n_bars=6)
    fake = _RecordingFakeFyers({
        "NSE:RELIANCE-EQ": {"s": "ok", "candles": _candles_from_df(df)},
    })
    now = datetime(2026, 5, 4, 9, 45, 12, tzinfo=IST)
    first = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ"], now=now, client=fake,
    ))
    assert first == {"NSE:RELIANCE-EQ": 6}

    second = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ"], now=now, client=fake,
    ))
    assert second == {}
    assert len(temp_aggregator.get_5m_bars("NSE:RELIANCE-EQ")) == 6


def test_backfill_drops_in_progress_slot(temp_aggregator):
    """If Fyers returns the current in-progress slot too, we must
    trim it — the live aggregator owns that slot."""
    session_open = datetime(2026, 5, 4, 9, 15, tzinfo=IST)
    # 7 bars including the current 09:45 slot
    df = _make_history_df(session_open, n_bars=7)
    fake = _RecordingFakeFyers({
        "NSE:RELIANCE-EQ": {"s": "ok", "candles": _candles_from_df(df)},
    })
    now = datetime(2026, 5, 4, 9, 47, 30, tzinfo=IST)  # mid-09:45 slot
    filled = asyncio.run(backfill.backfill_today(
        ["NSE:RELIANCE-EQ"], now=now, client=fake,
    ))
    # Only the 6 fully-closed bars (09:15..09:40) should be inserted
    assert filled == {"NSE:RELIANCE-EQ": 6}
    bars = temp_aggregator.get_5m_bars("NSE:RELIANCE-EQ")
    assert len(bars) == 6
    assert bars.index[-1].time().isoformat() == "09:40:00"
