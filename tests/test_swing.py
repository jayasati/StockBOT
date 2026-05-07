"""Swing-signal unit tests (relocated from swing_backtest.py)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from swing.config import RANGE_POSITION_THRESHOLD
from swing.evaluate import compute_breadth_series, evaluate_swing


class TestRangePosition(unittest.TestCase):
    def test_close_at_high(self):
        rp = (110 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 1.0)

    def test_close_at_low(self):
        rp = (100 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 0.0)

    def test_close_in_top_10pct(self):
        # close=109, low=100, high=110 → rp = 0.9 → just qualifies
        rp = (109 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 0.9)
        self.assertGreaterEqual(rp, RANGE_POSITION_THRESHOLD)


class TestExtensionFilter(unittest.TestCase):
    """Build a synthetic stock that triggers the volume+range+EMA signal but
    is heavily extended above the 20-day EMA. With no cap, it should fire;
    with a tight cap, it should be filtered out."""

    def _make_data(self):
        n = 35
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        close = np.linspace(100.0, 130.0, n)  # rising — 30% over 35d
        # Final bar: huge volume spike, close at top of range, well above EMA
        close[-1] = close[-2] * 1.04
        high = close + 0.5
        low = np.minimum(close - 0.5, np.roll(close, 1) - 0.5)
        low[-1] = close[-1] - 0.5  # tiny range, close at top
        high[-1] = close[-1] + 0.05
        volume = np.full(n, 1000.0)
        volume[-1] = 5000.0  # 5x avg
        df = pd.DataFrame({
            "Open": close - 0.2, "High": high, "Low": low,
            "Close": close, "Volume": volume,
        }, index=idx)
        return df, idx

    def test_no_cap_fires(self):
        df, idx = self._make_data()
        # Need an entry+5d future, so add 6 dummy bars after the signal
        future = pd.DataFrame({
            "Open": [df["Close"].iloc[-1]] * 6,
            "High": [df["Close"].iloc[-1] + 1] * 6,
            "Low":  [df["Close"].iloc[-1] - 1] * 6,
            "Close": [df["Close"].iloc[-1]] * 6,
            "Volume": [1000.0] * 6,
        }, index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=6, freq="D"))
        full = pd.concat([df, future])
        nifty = pd.DataFrame({"Close": [100.0] * len(full)}, index=full.index)
        alerts = evaluate_swing({"X": full}, nifty, apply_regime=False,
                                max_extension_pct=None)
        self.assertGreaterEqual(len(alerts), 1)

    def test_tight_cap_filters(self):
        df, idx = self._make_data()
        future = pd.DataFrame({
            "Open": [df["Close"].iloc[-1]] * 6,
            "High": [df["Close"].iloc[-1] + 1] * 6,
            "Low":  [df["Close"].iloc[-1] - 1] * 6,
            "Close": [df["Close"].iloc[-1]] * 6,
            "Volume": [1000.0] * 6,
        }, index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=6, freq="D"))
        full = pd.concat([df, future])
        nifty = pd.DataFrame({"Close": [100.0] * len(full)}, index=full.index)
        # The signal bar is ~5–8% above EMA; cap of 2% must filter it.
        alerts = evaluate_swing({"X": full}, nifty, apply_regime=False,
                                max_extension_pct=2.0)
        self.assertEqual(len(alerts), 0)


class TestBreadthSeries(unittest.TestCase):
    def test_breadth_split(self):
        idx = pd.date_range("2025-01-01", periods=30, freq="D")
        rising = pd.DataFrame({
            "Close": np.linspace(100, 130, 30),
            "Open": np.linspace(100, 130, 30),
            "High": np.linspace(101, 131, 30),
            "Low":  np.linspace(99, 129, 30),
            "Volume": np.full(30, 1000),
        }, index=idx)
        falling = pd.DataFrame({
            "Close": np.linspace(100, 70, 30),
            "Open": np.linspace(100, 70, 30),
            "High": np.linspace(101, 71, 30),
            "Low":  np.linspace(99, 69, 30),
            "Volume": np.full(30, 1000),
        }, index=idx)
        breadth = compute_breadth_series({"R": rising, "F": falling})
        # Last day: rising is well above its EMA (100%), falling is well below (0%)
        self.assertAlmostEqual(breadth.iloc[-1], 50.0, places=1)
