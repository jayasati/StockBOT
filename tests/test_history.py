"""Tests for fyers_client.history.

Two concerns:

1. The parser: given a Fyers-shaped response, do we produce the lowercase
   tz-aware IST DataFrame that ``seed_bars`` expects?
2. The error path: do non-ok responses, SDK exceptions, and missing
   token all surface as ``FyersHistoryError`` so callers can fall back?
"""
from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from fyers_client.history import (
    FyersHistoryError,
    _candles_to_df,
    fetch_history,
)

IST = ZoneInfo("Asia/Kolkata")


def _epoch_ist(year, month, day, hour, minute) -> int:
    """Helper: turn an IST wall-clock time into the Unix epoch seconds
    Fyers' history endpoint uses, so tests don't carry magic numbers."""
    return int(pd.Timestamp(
        year=year, month=month, day=day, hour=hour, minute=minute, tz=IST,
    ).timestamp())


# ---------------------------------------------------------------------------
# _candles_to_df: shape contract with seed_bars
# ---------------------------------------------------------------------------

def test_candles_to_df_converts_epoch_to_ist():
    """The on-the-wire ``[epoch_s, o, h, l, c, v]`` becomes a tz-aware
    IST index. Epochs are computed from explicit IST wall-clock times
    so the test reads as the intent, not as magic numbers."""
    candles = [
        [_epoch_ist(2026, 5, 4, 9, 15), 100.0, 101.0, 99.5, 100.5, 1234.0],
        [_epoch_ist(2026, 5, 4, 9, 20), 100.5, 102.0, 100.0, 101.5, 2345.0],
    ]
    df = _candles_to_df(candles)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert str(df.index.tz) == "Asia/Kolkata"
    assert df.index[0].time().isoformat() == "09:15:00"
    assert df.index[1].time().isoformat() == "09:20:00"
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[1]["volume"] == 2345.0


def test_candles_to_df_empty_input_returns_empty_frame():
    """Fyers can legitimately return ``"candles": []`` for off-hours
    requests; that's not an error, but downstream needs the right
    shape so it falls through cleanly."""
    df = _candles_to_df([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None


def test_candles_to_df_skips_malformed_rows():
    """Defensive: a partial row in the wire stream shouldn't poison
    the rest of the frame."""
    candles = [
        [_epoch_ist(2026, 5, 4, 9, 15), 100.0, 101.0, 99.5, 100.5, 1234.0],
        [_epoch_ist(2026, 5, 4, 9, 20)],  # truncated — must be skipped
        "not a row",                       # garbage — must be skipped
        [_epoch_ist(2026, 5, 4, 9, 25), 101.0, 102.0, 100.5, 101.5, 3456.0],
    ]
    df = _candles_to_df(candles)
    assert len(df) == 2


def test_candles_to_df_compatible_with_seed_bars(tmp_path):
    """End-to-end: feed _candles_to_df output into seed_bars and read
    back via the aggregator. Same shape as the live aggregator emits."""
    from data.realtime_feed import BarAggregator

    candles = [
        [_epoch_ist(2026, 5, 4, 9, 15), 100.0, 101.0, 99.5, 100.5, 1234.0],
        [_epoch_ist(2026, 5, 4, 9, 20), 100.5, 102.0, 100.0, 101.5, 2345.0],
    ]
    df = _candles_to_df(candles)
    agg = BarAggregator(db_path=tmp_path / "hist.db")
    inserted = agg.seed_bars("NSE:RELIANCE-EQ", df)
    assert inserted == 2

    out = agg.get_5m_bars("NSE:RELIANCE-EQ")
    assert len(out) == 2
    assert out.iloc[0]["open"] == 100.0


# ---------------------------------------------------------------------------
# fetch_history: client wiring + error surfaces
# ---------------------------------------------------------------------------

class _FakeFyers:
    """Pretends to be ``fyersModel.FyersModel`` for unit tests."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def history(self, data):
        self.calls.append(dict(data))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_fetch_history_builds_correct_payload():
    """Args should map to Fyers' documented fields:
    symbol/resolution/date_format=1/range_from/range_to/cont_flag=1."""
    fake = _FakeFyers({"s": "ok", "candles": []})
    fetch_history(
        "NSE:RELIANCE-EQ", "5", date(2026, 5, 4), date(2026, 5, 4),
        client=fake,
    )
    assert len(fake.calls) == 1
    payload = fake.calls[0]
    assert payload["symbol"] == "NSE:RELIANCE-EQ"
    assert payload["resolution"] == "5"
    assert payload["date_format"] == "1"
    assert payload["range_from"] == "2026-05-04"
    assert payload["range_to"] == "2026-05-04"
    assert payload["cont_flag"] == "1"


def test_fetch_history_parses_ok_response():
    """Happy path: ok response → parsed DataFrame."""
    fake = _FakeFyers({
        "s": "ok",
        "candles": [
            [_epoch_ist(2026, 5, 4, 9, 15), 100.0, 101.0, 99.5, 100.5, 1234.0],
        ],
    })
    df = fetch_history(
        "NSE:RELIANCE-EQ", "5", date(2026, 5, 4), date(2026, 5, 4),
        client=fake,
    )
    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.5


def test_fetch_history_non_ok_raises():
    """Fyers' rate-limit / auth-error / invalid-symbol all show up as
    ``s != "ok"``. The wrapper raises so the caller decides what to do."""
    fake = _FakeFyers({"s": "error", "code": -50, "message": "Invalid symbol"})
    with pytest.raises(FyersHistoryError):
        fetch_history(
            "NSE:BOGUS-EQ", "5", date(2026, 5, 4), date(2026, 5, 4),
            client=fake,
        )


def test_fetch_history_sdk_exception_raises():
    """A network / SDK exception is wrapped — never leaks raw."""
    fake = _FakeFyers(ConnectionError("boom"))
    with pytest.raises(FyersHistoryError):
        fetch_history(
            "NSE:RELIANCE-EQ", "5", date(2026, 5, 4), date(2026, 5, 4),
            client=fake,
        )


def test_fetch_history_no_client_raises(monkeypatch):
    """When no client is provided and no token is cached, ``_build_client``
    returns None — surface that as FyersHistoryError so the backfill
    orchestrator logs+skips this symbol."""
    import fyers_client.history as history_mod

    monkeypatch.setattr(history_mod, "_build_client", lambda: None)
    with pytest.raises(FyersHistoryError):
        fetch_history(
            "NSE:RELIANCE-EQ", "5", date(2026, 5, 4), date(2026, 5, 4),
        )
