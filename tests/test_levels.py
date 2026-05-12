"""Session-level engine: PDH/PDL/PDC, Opening Range, Initial Balance, Pivots.

Levels are session-anchored constants, not per-bar series, so the test
shape is different from indicators.py: we feed a tiny daily/intraday
frame whose answers we can compute by hand and verify the dict matches."""
from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from indicators.levels import (
    initial_balance,
    opening_range,
    pivot_points,
    previous_day_hlc,
)

IST = ZoneInfo("Asia/Kolkata")


def _intraday(closes, *, highs=None, lows=None, session_date=date(2026, 5, 4)):
    """Build a 5m DataFrame anchored at the session open for ``session_date``."""
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    if highs is None:
        highs = closes + 0.5
    if lows is None:
        lows = closes - 0.5
    start = pd.Timestamp.combine(
        session_date, pd.Timestamp("09:15").time()
    ).tz_localize(IST)
    idx = pd.date_range(start, periods=n, freq="5min", tz=IST)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows,
         "close": closes, "volume": np.full(n, 1000.0)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# previous_day_hlc
# ---------------------------------------------------------------------------

class TestPreviousDayHLC:
    def test_returns_most_recent_prior_session(self):
        idx = pd.to_datetime([
            "2026-04-30", "2026-05-01", "2026-05-04",
        ])
        df = pd.DataFrame(
            {"high": [102, 105, 110],
             "low":  [ 98, 100, 105],
             "close":[100, 103, 108]},
            index=idx,
        )
        out = previous_day_hlc(df, session_date=date(2026, 5, 4))
        assert out == {"pdh": 105.0, "pdl": 100.0, "pdc": 103.0}

    def test_empty_history_returns_nans(self):
        out = previous_day_hlc(
            pd.DataFrame(columns=["high", "low", "close"]),
            session_date=date(2026, 5, 4),
        )
        assert all(np.isnan(v) for v in out.values())

    def test_no_session_strictly_before_returns_nans(self):
        idx = pd.to_datetime(["2026-05-04", "2026-05-05"])
        df = pd.DataFrame(
            {"high": [105, 110], "low": [100, 105], "close": [103, 108]},
            index=idx,
        )
        # Session date equals the FIRST row; nothing strictly before.
        out = previous_day_hlc(df, session_date=date(2026, 5, 4))
        assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# Opening Range
# ---------------------------------------------------------------------------

class TestOpeningRange:
    def test_15m_range_covers_first_three_5m_bars(self):
        # Bars 0,1,2 hit 105/103/107 highs and 99/100/101 lows.
        # Bars 3+ should NOT affect the 15m OR.
        closes = [100, 102, 104, 110, 95]
        highs = [105, 103, 107, 115, 100]
        lows = [99, 100, 101, 105, 95]
        df = _intraday(closes, highs=highs, lows=lows)
        out = opening_range(df, session_date=date(2026, 5, 4), minutes=15)
        assert out["orh"] == 107.0
        assert out["orl"] == 99.0
        assert out["or_mid"] == pytest.approx(103.0)

    def test_5m_range_first_bar_only(self):
        df = _intraday([100, 200, 300], highs=[101, 210, 320],
                       lows=[99, 195, 290])
        out = opening_range(df, session_date=date(2026, 5, 4), minutes=5)
        assert out["orh"] == 101.0
        assert out["orl"] == 99.0

    def test_insufficient_bars_returns_nan(self):
        df = _intraday([100, 101])  # only 10 minutes of bars
        out = opening_range(df, session_date=date(2026, 5, 4), minutes=15)
        assert all(np.isnan(v) for v in out.values())

    def test_invalid_minutes_raises(self):
        df = _intraday([100, 101, 102])
        with pytest.raises(ValueError):
            opening_range(df, session_date=date(2026, 5, 4), minutes=20)

    def test_session_date_with_no_bars_returns_nan(self):
        df = _intraday([100, 101, 102], session_date=date(2026, 5, 4))
        out = opening_range(df, session_date=date(2026, 5, 5), minutes=15)
        assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# Initial Balance — first 60 min = first 12 bars of 5m
# ---------------------------------------------------------------------------

class TestInitialBalance:
    def test_covers_first_twelve_bars(self):
        # Bar 11 has the highest high, bar 5 has the lowest low.
        highs = [100] * 5 + [99] + [101] * 5 + [120] + [80] * 5
        lows = [99] * 5 + [85] + [100] * 5 + [105] + [70] * 5
        closes = [100] * 17
        df = _intraday(closes, highs=highs, lows=lows)
        out = initial_balance(df, session_date=date(2026, 5, 4))
        assert out["ib_high"] == 120.0
        assert out["ib_low"] == 85.0

    def test_insufficient_bars_returns_nan(self):
        # Only 5 bars (25 minutes) — IB needs 12 (60 minutes).
        df = _intraday([100] * 5)
        out = initial_balance(df, session_date=date(2026, 5, 4))
        assert all(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# Pivot Points
# ---------------------------------------------------------------------------

class TestPivotPointsClassic:
    def test_known_values(self):
        # PDH=110, PDL=100, PDC=108
        # P = (110+100+108)/3 = 106
        # R1 = 2P - PDL = 112; S1 = 2P - PDH = 102
        # R2 = P + (PDH-PDL) = 116; S2 = P - 10 = 96
        # R3 = PDH + 2*(P - PDL) = 110 + 12 = 122
        # S3 = PDL - 2*(PDH - P) = 100 - 8 = 92
        out = pivot_points({"pdh": 110.0, "pdl": 100.0, "pdc": 108.0},
                           method="classic")
        assert out["pivot"] == pytest.approx(106.0)
        assert out["r1"] == pytest.approx(112.0)
        assert out["s1"] == pytest.approx(102.0)
        assert out["r2"] == pytest.approx(116.0)
        assert out["s2"] == pytest.approx(96.0)
        assert out["r3"] == pytest.approx(122.0)
        assert out["s3"] == pytest.approx(92.0)


class TestPivotPointsFibonacci:
    def test_known_values(self):
        # PDH=110, PDL=100, PDC=108; range=10; P=106
        out = pivot_points({"pdh": 110.0, "pdl": 100.0, "pdc": 108.0},
                           method="fibonacci")
        assert out["pivot"] == pytest.approx(106.0)
        assert out["r1"] == pytest.approx(106.0 + 0.382 * 10)
        assert out["r2"] == pytest.approx(106.0 + 0.618 * 10)
        assert out["r3"] == pytest.approx(106.0 + 1.000 * 10)
        assert out["s1"] == pytest.approx(106.0 - 0.382 * 10)


class TestPivotPointsCamarilla:
    def test_known_values(self):
        # PDH=110, PDL=100, PDC=108; range=10; pivot = PDC = 108
        out = pivot_points({"pdh": 110.0, "pdl": 100.0, "pdc": 108.0},
                           method="camarilla")
        assert out["pivot"] == pytest.approx(108.0)
        assert out["r1"] == pytest.approx(108.0 + 10 * 1.1 / 12.0)
        assert out["r2"] == pytest.approx(108.0 + 10 * 1.1 / 6.0)
        assert out["r3"] == pytest.approx(108.0 + 10 * 1.1 / 4.0)
        assert out["s3"] == pytest.approx(108.0 - 10 * 1.1 / 4.0)


class TestPivotPointsEdgeCases:
    def test_nan_in_prev_day_propagates(self):
        out = pivot_points({"pdh": np.nan, "pdl": 100.0, "pdc": 108.0})
        assert all(np.isnan(v) for v in out.values())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            pivot_points({"pdh": 110.0, "pdl": 100.0, "pdc": 108.0},
                         method="vapor")
