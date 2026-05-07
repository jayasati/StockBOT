"""Backtest unit tests (relocated from inline classes in ``backtest.py``)."""
from __future__ import annotations

import unittest
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.replay import _forward_daily_price, _forward_intraday_price
from backtest.report import compute_win_rate

IST = ZoneInfo("Asia/Kolkata")


class TestWinRate(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(compute_win_rate([]), 0.0)

    def test_all_wins(self):
        self.assertEqual(compute_win_rate([2.0, 3.0, 5.0]), 100.0)

    def test_all_losses(self):
        self.assertEqual(compute_win_rate([-1.0, -2.0, 0.5]), 0.0)

    def test_mixed_50_percent(self):
        self.assertAlmostEqual(compute_win_rate([2.0, -1.0, 3.0, 0.5]), 50.0)

    def test_threshold_boundary_inclusive(self):
        # exactly 1.5% counts as a win
        self.assertEqual(compute_win_rate([1.5, 1.49]), 50.0)

    def test_custom_threshold(self):
        self.assertEqual(compute_win_rate([2.0, 3.0], threshold_pct=2.5), 50.0)

    def test_drops_none(self):
        # None entries are valid trades that lacked a forward bar — they
        # should be excluded from the denominator.
        self.assertAlmostEqual(compute_win_rate([2.0, None, 1.0]), 50.0)

    def test_drops_nan(self):
        self.assertAlmostEqual(compute_win_rate([2.0, float("nan"), 1.0]), 50.0)

    def test_only_nones(self):
        self.assertEqual(compute_win_rate([None, None]), 0.0)


class TestForwardPrice(unittest.TestCase):
    def test_intraday_30m_offset(self):
        idx = pd.date_range("2025-01-15 09:15", periods=20, freq="5min", tz=IST)
        intraday = pd.DataFrame({"Close": np.arange(100.0, 120.0)}, index=idx)
        # bar at 09:40 (i=5); +30m = 10:10 = i=11 = 111
        result = _forward_intraday_price(intraday, idx[5], timedelta(minutes=30))
        self.assertEqual(result, 111.0)

    def test_intraday_no_future_bar(self):
        idx = pd.date_range("2025-01-15 09:15", periods=5, freq="5min", tz=IST)
        intraday = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)
        result = _forward_intraday_price(intraday, idx[-1], timedelta(minutes=30))
        self.assertIsNone(result)

    def test_intraday_rejects_overnight_gap(self):
        # 5m bar at end-of-day, next bar is next morning → +30m crosses
        # 17 hours of close. Should reject (return None).
        idx = pd.DatetimeIndex([
            pd.Timestamp("2025-01-15 15:25", tz=IST),
            pd.Timestamp("2025-01-16 09:15", tz=IST),
        ])
        intraday = pd.DataFrame({"Close": [100, 101]}, index=idx)
        result = _forward_intraday_price(intraday, idx[0], timedelta(minutes=30))
        self.assertIsNone(result)

    def test_daily_offset_1(self):
        days = pd.date_range("2025-01-10", periods=10, freq="D")
        daily = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=days)
        # t=01-12; +1d = 01-13 → 103.0
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-12"), 1)
        self.assertEqual(result, 103.0)

    def test_daily_offset_5(self):
        days = pd.date_range("2025-01-10", periods=10, freq="D")
        daily = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=days)
        # t=01-12; +5d = 01-17 → 107.0
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-12"), 5)
        self.assertEqual(result, 107.0)

    def test_daily_insufficient_history(self):
        days = pd.date_range("2025-01-10", periods=3, freq="D")
        daily = pd.DataFrame({"Close": [100, 101, 102]}, index=days)
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-10"), 5)
        self.assertIsNone(result)
