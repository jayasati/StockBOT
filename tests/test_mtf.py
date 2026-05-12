"""Multi-timeframe resampling — OHLCV correctness + boundary alignment."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from indicators.mtf import resample_ohlcv

IST = ZoneInfo("Asia/Kolkata")


def _bars(closes, *, opens=None, highs=None, lows=None, volumes=None,
          start="2026-05-04 09:15", freq="5min") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    if opens is None:
        opens = closes.copy()
    if highs is None:
        highs = closes + 0.5
    if lows is None:
        lows = closes - 0.5
    if volumes is None:
        volumes = np.full(n, 1000.0)
    idx = pd.date_range(start, periods=n, freq=freq, tz=IST)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# OHLCV-correct aggregation
# ---------------------------------------------------------------------------

def test_5m_to_15m_aggregates_three_bars():
    # Closes 1..6 — first 3 → 15m at 09:15, next 3 → 15m at 09:30.
    df = _bars([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = resample_ohlcv(df, "15m")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 1.0           # first of bars 0..2
    assert first["high"] == 3.5           # max of (closes+0.5)
    assert first["low"] == 0.5            # min of (closes-0.5)
    assert first["close"] == 3.0          # last of bars 0..2
    assert first["volume"] == 3000.0
    second = out.iloc[1]
    assert second["open"] == 4.0
    assert second["close"] == 6.0


def test_5m_to_5m_is_identity():
    df = _bars(list(range(1, 13)))
    out = resample_ohlcv(df, "5m")
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_5m_to_60m_buckets_twelve_bars():
    df = _bars(list(range(1, 25)))  # 24 bars × 5m = 2 hours
    out = resample_ohlcv(df, "60m")
    assert len(out) == 2
    assert out.iloc[0]["open"] == 1.0
    assert out.iloc[0]["close"] == 12.0
    assert out.iloc[0]["volume"] == 12000.0


def test_invalid_timeframe_raises():
    df = _bars([1, 2, 3])
    with pytest.raises(ValueError):
        resample_ohlcv(df, "7m")


def test_empty_input_returns_empty():
    df = _bars([])
    out = resample_ohlcv(df, "15m")
    assert out.empty


# ---------------------------------------------------------------------------
# Boundary alignment — bar timestamp = OPEN time of its window
# ---------------------------------------------------------------------------

def test_15m_buckets_start_at_15_30_45_00():
    # 12 bars of 5m starting at 09:15 → 4 buckets of 15m at 09:15, 09:30,
    # 09:45, 10:00.
    df = _bars(list(range(1, 13)))
    out = resample_ohlcv(df, "15m")
    expected_starts = [
        pd.Timestamp("2026-05-04 09:15", tz=IST),
        pd.Timestamp("2026-05-04 09:30", tz=IST),
        pd.Timestamp("2026-05-04 09:45", tz=IST),
        pd.Timestamp("2026-05-04 10:00", tz=IST),
    ]
    assert list(out.index) == expected_starts
