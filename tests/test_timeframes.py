import pandas as pd
import pytest

from data.timeframes import get_bars
from tests.conftest import (
    AFTER_CLOSE,
    closes,
    highs,
    lows,
    opens,
    timestamps_utc_ms,
    volumes,
)


# ---------------------------------------------------------------------------
# 15m candles
# ---------------------------------------------------------------------------

def test_15m_bar_count_and_first_candle(full_session_db):
    df = get_bars("X", "15m", _now=AFTER_CLOSE)

    # Exactly 25 complete 15m candles in a full session
    assert len(df) == 25

    # First index is 09:15 IST — confirms 03:45 UTC offset anchor
    assert df.index[0] == pd.Timestamp("2026-05-04 09:15:00", tz="Asia/Kolkata")

    # OHLCV of first candle equals manual aggregation of bars 0, 1, 2
    assert df.iloc[0]["open"] == opens[0]
    assert df.iloc[0]["high"] == max(highs[0:3])
    assert df.iloc[0]["low"] == min(lows[0:3])
    assert df.iloc[0]["close"] == closes[2]
    assert df.iloc[0]["volume"] == pytest.approx(sum(volumes[0:3]))


# ---------------------------------------------------------------------------
# 1h candles
# ---------------------------------------------------------------------------

def test_1h_bar_count_and_boundaries(full_session_db):
    df = get_bars("X", "1h", _now=AFTER_CLOSE)

    # Exactly 6 hourly candles
    assert len(df) == 6

    # First candle: 09:15–10:15 (bars 0–11, 12 bars)
    assert df.index[0] == pd.Timestamp("2026-05-04 09:15:00", tz="Asia/Kolkata")
    assert df.iloc[0]["open"] == opens[0]
    assert df.iloc[0]["high"] == max(highs[0:12])
    assert df.iloc[0]["low"] == min(lows[0:12])
    assert df.iloc[0]["close"] == closes[11]
    assert df.iloc[0]["volume"] == pytest.approx(sum(volumes[0:12]))

    # Last candle: 14:15–15:30 (bars 60–74, 15 bars) — 75-minute bucket
    assert df.index[-1] == pd.Timestamp("2026-05-04 14:15:00", tz="Asia/Kolkata")
    assert df.iloc[-1]["volume"] == pytest.approx(sum(volumes[60:75]))


# ---------------------------------------------------------------------------
# 1d candle
# ---------------------------------------------------------------------------

def test_1d_single_candle(full_session_db):
    df = get_bars("X", "1d", _now=AFTER_CLOSE)

    # Exactly 1 daily candle
    assert len(df) == 1

    # Index is midnight IST of the session date
    assert df.index[0] == pd.Timestamp("2026-05-04 00:00:00", tz="Asia/Kolkata")

    # OHLCV spans all 75 bars
    assert df.iloc[0]["open"] == opens[0]
    assert df.iloc[0]["high"] == max(highs)
    assert df.iloc[0]["low"] == min(lows)
    assert df.iloc[0]["close"] == closes[-1]
    assert df.iloc[0]["volume"] == pytest.approx(sum(volumes))


# ---------------------------------------------------------------------------
# Incomplete bar removal
# ---------------------------------------------------------------------------

_now_partial = pd.Timestamp("2026-05-04 10:12:00", tz="Asia/Kolkata")


def test_incomplete_bar_excluded(partial_session_db):
    # --- 15m ---
    df_15m = get_bars("X", "15m", _now=_now_partial)
    # Only the 09:15, 09:30, 09:45 buckets are complete
    assert len(df_15m) == 3
    assert df_15m.index[-1] == pd.Timestamp("2026-05-04 09:45:00", tz="Asia/Kolkata")

    # The incomplete 10:00 bucket must not appear
    assert pd.Timestamp("2026-05-04 10:00:00", tz="Asia/Kolkata") not in df_15m.index

    # --- 1h ---
    df_1h = get_bars("X", "1h", _now=_now_partial)
    # The 09:15–10:15 bucket closes at 10:15 > 10:12 → not yet complete
    assert len(df_1h) == 0


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def test_cache_invalidated_on_new_bar(full_session_db):
    """Inserting the first bar of a new session date must bust the cache
    so the 1d result changes unambiguously.
    """
    from data.timeframes import _get_resampled_full  # noqa: F401  (cache hook)

    # Day 1 result — one daily candle for 2026-05-04
    _now_day1 = pd.Timestamp("2026-05-04 16:00:00", tz="Asia/Kolkata")
    df_before = get_bars("X", "1d", _now=_now_day1)
    assert len(df_before) == 1

    # Insert the first complete bar of session 2 (2026-05-05 09:15)
    bar2_ts_ist = pd.Timestamp("2026-05-05 09:15:00", tz="Asia/Kolkata")
    bar2_ts_utc_ms = int(bar2_ts_ist.value / 1e6)
    full_session_db.execute(
        "INSERT INTO bars_5m VALUES (?,?,?,?,?,?,?)",
        ("X", bar2_ts_utc_ms, 200.0, 205.0, 198.0, 202.0, 1000.0),
    )
    full_session_db.commit()

    # _now is after bar 2's close time (09:15 + 5m = 09:20)
    _now_day2 = pd.Timestamp("2026-05-05 09:21:00", tz="Asia/Kolkata")
    df_after = get_bars("X", "1d", _now=_now_day2)

    # Cache must have been invalidated; two daily candles now
    assert len(df_after) == 2
    assert df_after.index[-1] == pd.Timestamp("2026-05-05 00:00:00", tz="Asia/Kolkata")
