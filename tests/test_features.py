"""Golden-data tests for features.py.

Fixture: two consecutive trading sessions of 5-min bars (2026-04-30, 2026-05-01)
with hand-picked OHLCV so every expected value can be verified by paper
arithmetic. The session-boundary case (VWAP must reset at 09:15 IST) is the
single most important test here — getting it wrong silently corrupts every
downstream setup detector.
"""

from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import features as F


IST = ZoneInfo("Asia/Kolkata")
DAY1 = date(2026, 4, 30)
DAY2 = date(2026, 5, 1)


def _ts(d, t):
    return pd.Timestamp.combine(d, t).tz_localize(IST)


# Day-1 and day-2 5-min bars. Day 1 exists so VWAP reset and RVOL time-of-day
# normalisation actually have a prior session to be measured against.
_DAY1_BARS = [
    (time(9, 15), 100.0, 101.0, 99.0,   100.0,  1000),
    (time(9, 20), 100.0, 102.0, 99.5,   101.0,  1500),
    (time(9, 25), 101.0, 103.0, 100.5,  102.0,  2000),
    (time(9, 30), 102.0, 102.5, 101.0,  101.5,  800),
    (time(9, 35), 101.5, 102.0, 101.0,  101.8,  1200),
]
_DAY2_BARS = [
    (time(9, 15), 102.0, 103.0,  100.5, 102.0,  2000),
    (time(9, 20), 102.0, 104.0,  101.0, 103.0,  3000),
    (time(9, 25), 103.0, 104.5,  102.0, 104.0,  2500),
    (time(9, 30), 104.0, 104.5,  103.0, 103.5,  1000),
    (time(9, 35), 103.5, 104.0,  103.0, 103.8,  1400),
]


@pytest.fixture
def bars_5m():
    rows = []
    for d, day_bars in [(DAY1, _DAY1_BARS), (DAY2, _DAY2_BARS)]:
        for t, o, h, l, c, v in day_bars:
            rows.append(
                {"ts": _ts(d, t), "open": o, "high": h, "low": l, "close": c, "volume": v}
            )
    return pd.DataFrame(rows).set_index("ts")


@pytest.fixture
def bars_daily():
    # Aggregate the 5-min fixture into daily bars manually so the test stays
    # purely about features.py, not about resampling.
    return pd.DataFrame(
        {
            "open":   [100.0, 102.0],
            "high":   [103.0, 104.5],
            "low":    [99.0,  100.5],
            "close":  [101.8, 103.8],
            "volume": [6500,  9900],
        },
        index=pd.DatetimeIndex([pd.Timestamp(DAY1), pd.Timestamp(DAY2)]),
    )


# ---------------------------------------------------------------------------
# session_vwap — the boundary case (CRITICAL)
# ---------------------------------------------------------------------------

def test_session_vwap_resets_at_session_open(bars_5m):
    """The 09:15 bar of DAY2 must compute its VWAP from its own typical price
    only — no carry-over from DAY1's bars. If it doesn't reset, the value
    would be pulled toward DAY1's price/volume centre."""
    vwap = F.session_vwap(bars_5m, DAY2)
    first_bar_ts = _ts(DAY2, time(9, 15))
    expected = (103.0 + 100.5 + 102.0) / 3.0   # typical price of first bar
    assert vwap.loc[first_bar_ts] == pytest.approx(expected, rel=1e-12)


def test_session_vwap_cumulative_within_session(bars_5m):
    """Bar-2 VWAP = (sum of typical*vol) / (sum of vol) over bars 1..2 only."""
    vwap = F.session_vwap(bars_5m, DAY2)
    second_bar_ts = _ts(DAY2, time(9, 20))
    typ1 = (103.0 + 100.5 + 102.0) / 3.0
    typ2 = (104.0 + 101.0 + 103.0) / 3.0
    expected = (typ1 * 2000 + typ2 * 3000) / (2000 + 3000)
    assert vwap.loc[second_bar_ts] == pytest.approx(expected, rel=1e-12)


def test_session_vwap_only_returns_session_bars(bars_5m):
    vwap = F.session_vwap(bars_5m, DAY2)
    assert len(vwap) == len(_DAY2_BARS)
    assert all(ts.date() == DAY2 for ts in vwap.index)


# ---------------------------------------------------------------------------
# opening_range
# ---------------------------------------------------------------------------

def test_opening_range_15min_window(bars_5m):
    """First three 5-min bars (09:15, 09:20, 09:25) constitute the 15-min OR."""
    high, low = F.opening_range(bars_5m, DAY2, minutes=15)
    assert high == pytest.approx(104.5)   # max of (103, 104, 104.5)
    assert low == pytest.approx(100.5)    # min of (100.5, 101, 102)


def test_opening_range_excludes_post_window_bars(bars_5m):
    """A 5-min OR window must drop the 09:20 bar onward."""
    high, low = F.opening_range(bars_5m, DAY2, minutes=5)
    assert high == pytest.approx(103.0)
    assert low == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# prior_day_high_low
# ---------------------------------------------------------------------------

def test_prior_day_high_low(bars_daily):
    pdh, pdl = F.prior_day_high_low(bars_daily, DAY2)
    assert pdh == pytest.approx(103.0)
    assert pdl == pytest.approx(99.0)


def test_prior_day_high_low_no_history(bars_daily):
    pdh, pdl = F.prior_day_high_low(bars_daily, DAY1)
    assert pd.isna(pdh) and pd.isna(pdl)


# ---------------------------------------------------------------------------
# ema
# ---------------------------------------------------------------------------

def test_ema_period_2_known_values():
    """Span=2 → α = 2/3. Hand-computed recursion."""
    s = pd.Series([100.0, 101.0, 102.0, 103.0])
    out = F.ema(s, period=2)
    a = 2.0 / 3.0
    expected = [
        100.0,
        a * 101 + (1 - a) * 100,
        a * 102 + (1 - a) * (a * 101 + (1 - a) * 100),
        None,  # check the last one separately for clarity
    ]
    y3 = a * 103 + (1 - a) * (a * 102 + (1 - a) * (a * 101 + (1 - a) * 100))
    expected[3] = y3
    for i, v in enumerate(expected):
        assert out.iloc[i] == pytest.approx(v, rel=1e-12)


# ---------------------------------------------------------------------------
# atr (Wilder)
# ---------------------------------------------------------------------------

def test_atr_period_2_known_values(bars_5m):
    """ATR(2) on day-1 bars with α=0.5. TR[0] is NaN (no prev close), then
    TR series = [NaN, 2.5, 2.5, 1.5, 1.0]; ewm seeds at first valid."""
    day1 = bars_5m.loc[bars_5m.index.date == DAY1]
    out = F.atr(day1, period=2)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(2.5)
    assert out.iloc[2] == pytest.approx(2.5)         # 0.5*2.5 + 0.5*2.5
    assert out.iloc[3] == pytest.approx(2.0)         # 0.5*1.5 + 0.5*2.5
    assert out.iloc[4] == pytest.approx(1.5)         # 0.5*1.0 + 0.5*2.0


# ---------------------------------------------------------------------------
# rvol_time_normalized
# ---------------------------------------------------------------------------

def test_rvol_time_normalized_compares_same_slot(bars_5m):
    """Day-2's 09:15 bar volume (2000) divided by day-1's 09:15 bar volume
    (1000) → 2.0. A flat rolling mean would mix in 09:20/09:25 volumes and
    give the wrong answer."""
    rvol = F.rvol_time_normalized(bars_5m, lookback=10)
    assert rvol.loc[_ts(DAY2, time(9, 15))] == pytest.approx(2.0)
    assert rvol.loc[_ts(DAY2, time(9, 20))] == pytest.approx(3000 / 1500)
    assert rvol.loc[_ts(DAY2, time(9, 25))] == pytest.approx(2500 / 2000)


def test_rvol_time_normalized_first_session_is_nan(bars_5m):
    rvol = F.rvol_time_normalized(bars_5m, lookback=10)
    assert pd.isna(rvol.loc[_ts(DAY1, time(9, 15))])


# ---------------------------------------------------------------------------
# relative_strength
# ---------------------------------------------------------------------------

def test_relative_strength_20_session():
    sym = pd.DataFrame(
        {"close": [100.0 + i * 0.5 for i in range(21)]},   # 100 → 110
        index=pd.RangeIndex(21),
    )
    nif = pd.DataFrame(
        {"close": [100.0 + i * 0.25 for i in range(21)]},  # 100 → 105
        index=pd.RangeIndex(21),
    )
    rs = F.relative_strength(sym, nif, lookback=20)
    assert rs == pytest.approx(0.10 - 0.05, rel=1e-12)


def test_relative_strength_too_short_returns_nan():
    short = pd.DataFrame({"close": [100.0, 101.0]})
    nif = pd.DataFrame({"close": [100.0 + i * 0.25 for i in range(21)]})
    assert pd.isna(F.relative_strength(short, nif, lookback=20))


# ---------------------------------------------------------------------------
# is_above_vwap
# ---------------------------------------------------------------------------

def test_is_above_vwap_scalar():
    assert F.is_above_vwap(102.0, 100.0) is True
    assert F.is_above_vwap(99.0, 100.0) is False
    assert F.is_above_vwap(100.0, 100.0) is False  # strict greater


def test_is_above_vwap_series_row():
    bar = pd.Series({"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1})
    assert F.is_above_vwap(bar, 101.5) is True
    assert F.is_above_vwap(bar, 102.5) is False
