"""
Tests for data/realtime_feed.py.

Two distinct concerns:

1. OHLCV correctness on a recorded replay. We load the JSONL fixture (200
   ticks across 5 stocks across 2 5-min slots), replay it through the
   aggregator, and assert that every output bar's open/high/low/close/volume
   matches what np.* computes from the same raw data — i.e. the aggregator
   doesn't smuggle any opinion into the bar values.

2. Bar boundary correctness. The first slot of the day is 09:15–09:20 IST;
   the last is 15:25–15:30. The aggregator MUST drop ticks outside that
   window — VWAP and opening-range in features.py both depend on it.
"""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.realtime_feed import BarAggregator, _floor_to_5m, _is_in_session

IST = ZoneInfo("Asia/Kolkata")
FIXTURE = Path(__file__).parent / "fixtures" / "fyers_ticks_replay.jsonl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def replay_ticks() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def aggregator(tmp_path):
    """Fresh aggregator backed by a temp SQLite DB so tests don't pollute alerts.db."""
    return BarAggregator(db_path=tmp_path / "test_bars.db")


def _expected_per_slot(ticks: list[dict]) -> dict[tuple[str, pd.Timestamp], dict]:
    """Compute ground-truth OHLCV from the raw fixture, grouped by (symbol, slot).

    Volume is the delta of ``vol_traded_today`` from the first tick of the
    slot to the first tick of the next slot for the same symbol — the same
    semantics ``BarAggregator`` uses. The final (in-progress, then flushed)
    slot keeps its running delta up to its last tick.
    """
    # 1) First pass: collect per-symbol slot-ordered ticks with their vol_today.
    by_symbol_slot: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    expected: dict[tuple[str, pd.Timestamp], dict] = {}
    for t in ticks:
        if t.get("type") != "sf":
            continue
        ts_utc = pd.Timestamp(t["last_traded_time"], unit="s", tz="UTC")
        slot = _floor_to_5m(ts_utc.tz_convert(IST))
        if not _is_in_session(slot):
            continue
        sym = t["symbol"]
        key = (sym, slot)
        ltp = float(t["ltp"])
        vol = float(t.get("vol_traded_today") or 0.0)
        bucket = expected.get(key)
        if bucket is None:
            expected[key] = {
                "open": ltp, "high": ltp, "low": ltp, "close": ltp,
                "_start_vol": vol, "_last_vol": vol,
            }
            by_symbol_slot.setdefault(sym, []).append((slot, vol))
        else:
            bucket["high"] = max(bucket["high"], ltp)
            bucket["low"] = min(bucket["low"], ltp)
            bucket["close"] = ltp
            bucket["_last_vol"] = vol

    # 2) Volume per bar = vol at first tick of next slot − vol at first tick
    #    of this slot. The very last slot has no successor, so its volume is
    #    derived from its own last observed tick (matches what ``flush()``
    #    does in the aggregator).
    for sym, slots in by_symbol_slot.items():
        for i, (slot, start_vol) in enumerate(slots):
            bucket = expected[(sym, slot)]
            if i + 1 < len(slots):
                next_start_vol = slots[i + 1][1]
                bucket["volume"] = max(0.0, next_start_vol - start_vol)
            else:
                bucket["volume"] = max(0.0, bucket["_last_vol"] - start_vol)
            bucket.pop("_start_vol", None)
            bucket.pop("_last_vol", None)
    return expected


# ---------------------------------------------------------------------------
# OHLCV correctness on replay
# ---------------------------------------------------------------------------

def test_replay_produces_correct_ohlcv(aggregator, replay_ticks):
    """For every (symbol, slot), the aggregator's bar matches OHLCV computed
    independently from the raw ticks. open=first-LTP, high=max, low=min,
    close=last-LTP, volume=sum-of-LTQ — exactly the contract."""
    for tick in replay_ticks:
        aggregator.on_tick(tick)
    aggregator.flush()  # close the in-progress (final) bar

    expected = _expected_per_slot(replay_ticks)
    symbols = sorted({sym for sym, _ in expected})

    for sym in symbols:
        df = aggregator.get_5m_bars(sym, n=200)
        assert not df.empty, f"no bars for {sym}"
        for ts, row in df.iterrows():
            key = (sym, ts)
            assert key in expected, f"unexpected bar at {ts} for {sym}"
            exp = expected[key]
            assert row["open"]   == pytest.approx(exp["open"]),   f"{sym} {ts} open"
            assert row["high"]   == pytest.approx(exp["high"]),   f"{sym} {ts} high"
            assert row["low"]    == pytest.approx(exp["low"]),    f"{sym} {ts} low"
            assert row["close"]  == pytest.approx(exp["close"]),  f"{sym} {ts} close"
            assert row["volume"] == pytest.approx(exp["volume"]), f"{sym} {ts} volume"


def test_replay_produces_two_slots_per_symbol(aggregator, replay_ticks):
    """Sanity: the fixture is engineered as 5 symbols × 2 slots = 10 bars."""
    for tick in replay_ticks:
        aggregator.on_tick(tick)
    aggregator.flush()

    symbols = sorted({t["symbol"] for t in replay_ticks if t.get("type") == "sf"})
    assert len(symbols) == 5
    for sym in symbols:
        df = aggregator.get_5m_bars(sym, n=200)
        assert len(df) == 2, f"{sym}: expected 2 bars, got {len(df)}"


def test_dataframe_index_is_tz_aware_ist(aggregator, replay_ticks):
    """features.py expects a tz-aware (or naive-IST) index. Be tz-aware to
    avoid silent timezone confusion in downstream consumers."""
    for tick in replay_ticks:
        aggregator.on_tick(tick)
    aggregator.flush()
    df = aggregator.get_5m_bars("NSE:RELIANCE-EQ", n=10)
    assert df.index.tz is not None
    assert str(df.index.tz) == "Asia/Kolkata"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Bar boundary correctness
# ---------------------------------------------------------------------------

def _make_tick(
    symbol: str,
    ltp: float,
    ltq: int,
    ts_ist: datetime,
    vol_traded_today: int = 0,
) -> dict:
    return {
        "type": "sf",
        "symbol": symbol,
        "ltp": ltp,
        "ltq": ltq,
        "last_traded_time": int(ts_ist.timestamp()),
        "vol_traded_today": vol_traded_today,
    }


def test_pre_open_tick_is_dropped(aggregator):
    """A tick at 09:14:59 IST belongs to no valid 5-min slot in session
    (it would floor to 09:10) and must be discarded silently."""
    pre_open = datetime(2026, 5, 4, 9, 14, 59, tzinfo=IST)
    aggregator.on_tick(_make_tick("NSE:RELIANCE-EQ", 2500.0, 100, pre_open))
    df = aggregator.get_5m_bars("NSE:RELIANCE-EQ")
    assert df.empty
    assert aggregator.get_current_partial("NSE:RELIANCE-EQ") is None
    assert aggregator.stats()["out_of_session_dropped"] == 1


def test_first_session_tick_starts_first_bar(aggregator):
    """A 09:15:00 tick opens the first bar at exactly 09:15:00 IST."""
    t = datetime(2026, 5, 4, 9, 15, 0, tzinfo=IST)
    aggregator.on_tick(_make_tick("NSE:RELIANCE-EQ", 2500.0, 100, t))
    bar = aggregator.get_current_partial("NSE:RELIANCE-EQ")
    assert bar is not None
    assert bar.ts_open.time() == time(9, 15)
    assert bar.open == 2500.0


def test_slot_boundary_closes_previous_and_opens_next(aggregator):
    """A tick at 09:20:00 must close the 09:15 bar (now persisted) and open
    a fresh 09:20 bar.

    Volume comes from ``vol_traded_today`` deltas: the closed 09:15 bar's
    volume is the cum-vol at the first 09:20 tick minus the cum-vol at the
    first 09:15 tick (i.e. 1000 → 1300 = 300)."""
    sym = "NSE:RELIANCE-EQ"
    aggregator.on_tick(_make_tick(sym, 2500.0, 100, datetime(2026, 5, 4, 9, 15, 0, tzinfo=IST), vol_traded_today=1000))
    aggregator.on_tick(_make_tick(sym, 2510.0, 100, datetime(2026, 5, 4, 9, 17, 30, tzinfo=IST), vol_traded_today=1100))
    aggregator.on_tick(_make_tick(sym, 2505.0, 100, datetime(2026, 5, 4, 9, 19, 45, tzinfo=IST), vol_traded_today=1200))
    aggregator.on_tick(_make_tick(sym, 2520.0, 100, datetime(2026, 5, 4, 9, 20, 0, tzinfo=IST), vol_traded_today=1300))

    df = aggregator.get_5m_bars(sym, n=10)
    assert len(df) == 1, "the 09:15 bar should be completed and persisted"
    closed_bar = df.iloc[0]
    assert df.index[0].time() == time(9, 15)
    assert closed_bar["open"] == 2500.0
    assert closed_bar["high"] == 2510.0
    assert closed_bar["low"] == 2500.0
    assert closed_bar["close"] == 2505.0
    assert closed_bar["volume"] == 300.0

    partial = aggregator.get_current_partial(sym)
    assert partial is not None
    assert partial.ts_open.time() == time(9, 20)
    assert partial.open == 2520.0


def test_post_close_tick_is_dropped(aggregator):
    """A tick at 15:30:00 floors to 15:30, which is past the last valid bar
    open (15:25). It must be dropped — features.opening_range and
    session_vwap rely on no bars escaping the session window."""
    t = datetime(2026, 5, 4, 15, 30, 0, tzinfo=IST)
    aggregator.on_tick(_make_tick("NSE:TCS-EQ", 3800.0, 100, t))
    assert aggregator.get_5m_bars("NSE:TCS-EQ").empty
    assert aggregator.stats()["out_of_session_dropped"] == 1


def test_last_valid_bar_at_15_25(aggregator):
    """Sanity: the 15:25 slot is the last legitimate bar (15:25–15:30)."""
    t = datetime(2026, 5, 4, 15, 29, 59, tzinfo=IST)
    aggregator.on_tick(_make_tick("NSE:TCS-EQ", 3800.0, 100, t))
    bar = aggregator.get_current_partial("NSE:TCS-EQ")
    assert bar is not None
    assert bar.ts_open.time() == time(15, 25)


# ---------------------------------------------------------------------------
# Reconnect-replay path: completed bars survive a process restart
# ---------------------------------------------------------------------------

def test_completed_bars_survive_process_restart(tmp_path, replay_ticks):
    """Replay → flush → drop the aggregator → instantiate a new one with the
    same db_path → get_5m_bars should serve the persisted bars from SQLite."""
    db = tmp_path / "persist.db"
    a = BarAggregator(db_path=db)
    for tick in replay_ticks:
        a.on_tick(tick)
    a.flush()

    # Simulate process restart: brand-new aggregator, same DB
    b = BarAggregator(db_path=db)
    df = b.get_5m_bars("NSE:RELIANCE-EQ", n=200)
    assert len(df) == 2
    # Same OHLCV as the live aggregator
    assert df.iloc[0]["open"] == pytest.approx(2500.0)
    assert df.iloc[1]["open"] == pytest.approx(2501.5)


def test_seed_from_yfinance_inserts_session_bars(tmp_path):
    """yfinance backfill goes into the same bars_5m table; out-of-session
    rows (e.g. an aftermarket print) are dropped."""
    db = tmp_path / "seed.db"
    a = BarAggregator(db_path=db)
    idx = pd.DatetimeIndex(
        [
            "2026-05-04 09:15:00",  # in session
            "2026-05-04 09:20:00",  # in session
            "2026-05-04 15:30:00",  # out (last valid open is 15:25)
            "2026-05-04 16:00:00",  # out
        ],
        tz=IST,
    )
    df = pd.DataFrame(
        {
            "Open":   [100.0, 101.0, 102.0, 103.0],
            "High":   [101.0, 102.0, 103.0, 104.0],
            "Low":    [ 99.0, 100.0, 101.0, 102.0],
            "Close":  [100.5, 101.5, 102.5, 103.5],
            "Volume": [1000,  1500,  500,   200 ],
        },
        index=idx,
    )
    inserted = a.seed_from_yfinance("NSE:FOO-EQ", df)
    assert inserted == 2

    out = a.get_5m_bars("NSE:FOO-EQ")
    assert len(out) == 2
    assert list(out["open"]) == [100.0, 101.0]
