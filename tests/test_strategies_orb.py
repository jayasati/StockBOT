"""Tests for the Opening Range Breakout strategy.

Uses real ``indicators.levels.opening_range`` (not mocked) because the
session-windowing logic IS what we need to verify the strategy interacts
with correctly. OR math itself is covered in tests/test_levels.py."""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import OpeningRangeBreakout, REGISTRY, SignalKind, run_backtest

IST = ZoneInfo("Asia/Kolkata")


def _bars(rows, date="2026-05-04"):
    """Build OHLCV bars from ``[(time_str, o, h, l, c), ...]`` for one IST session."""
    idx = []
    data = []
    for r in rows:
        t_str, o, h, l, c = r
        idx.append(pd.Timestamp(f"{date} {t_str}", tz=IST))
        data.append((o, h, l, c, 1000))
    return pd.DataFrame(
        data, columns=["open", "high", "low", "close", "volume"], index=idx,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        s = OpeningRangeBreakout()
        assert s.minutes == 15 and s.single_shot is True

    def test_invalid_minutes(self):
        with pytest.raises(ValueError, match="minutes"):
            OpeningRangeBreakout(minutes=10)

    def test_valid_minutes_options(self):
        for m in (5, 15, 30, 60):
            assert OpeningRangeBreakout(minutes=m).minutes == m


# ---------------------------------------------------------------------------
# Signal layer
# ---------------------------------------------------------------------------

class TestSignal:
    def test_no_signal_during_or_window(self):
        # First 15-min window = bars at 09:15, 09:20, 09:25 (3 bars).
        # Until those are complete, opening_range returns NaN.
        s = OpeningRangeBreakout(minutes=15)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
        ])
        for i in range(len(df)):
            assert s.signal(df, i) is None

    def test_long_on_close_above_orh(self):
        # 3-bar OR: orh = max(101, 102, 102) = 102, orl = min(99, 100, 101) = 99.
        # Bar 3 at 09:30 closes 105 → close > orh → ENTER_LONG.
        s = OpeningRangeBreakout(minutes=15)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),  # OR ends
            ("09:30:00", 102, 106, 102, 105),  # breakout up
        ])
        sig = s.signal(df, 3)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "OrbLE"

    def test_short_on_close_below_orl(self):
        s = OpeningRangeBreakout(minutes=15)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),  # OR: orh=102, orl=99
            ("09:30:00", 100,  100, 95,  96),  # breakout down (close 96 < orl 99)
        ])
        sig = s.signal(df, 3)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "OrbSE"

    def test_no_signal_when_close_inside_or(self):
        s = OpeningRangeBreakout(minutes=15)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),
            ("09:30:00", 101, 102, 100, 101),  # inside OR
        ])
        assert s.signal(df, 3) is None

    def test_single_shot_blocks_second_breakout_same_day(self):
        # Trigger LONG, then on a later bar another up-breakout should be
        # silenced by the single-shot flag.
        s = OpeningRangeBreakout(minutes=15, single_shot=True)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),
            ("09:30:00", 102, 106, 102, 105),  # LONG fires
            ("09:35:00", 105, 105, 100, 100),  # back inside OR
            ("09:40:00", 100, 107, 100, 106),  # would re-breakout up
        ])
        # Drive bar 3 first (consumes the single-shot flag for this date)
        assert s.signal(df, 3).kind == SignalKind.ENTER_LONG
        assert s.signal(df, 4) is None
        # Bar 5 has prev close 100 (<= 102 orh), curr 106 (> 102) — would
        # normally fire LONG, but single-shot suppresses it.
        assert s.signal(df, 5) is None

    def test_single_shot_resets_on_new_day(self):
        s = OpeningRangeBreakout(minutes=15, single_shot=True)
        day1 = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),
            ("09:30:00", 102, 106, 102, 105),
        ], date="2026-05-04")
        day2 = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),
            ("09:30:00", 102, 106, 102, 105),
        ], date="2026-05-05")
        df = pd.concat([day1, day2])
        # Day 1's breakout consumes the flag for 2026-05-04.
        assert s.signal(df, 3).kind == SignalKind.ENTER_LONG
        # Day 2's breakout (i=7) should fire fresh.
        sig = s.signal(df, 7)
        assert sig is not None and sig.kind == SignalKind.ENTER_LONG

    def test_single_shot_off_allows_re_signal(self):
        s = OpeningRangeBreakout(minutes=15, single_shot=False)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),
            ("09:30:00", 102, 106, 102, 105),  # LONG
            ("09:35:00", 105, 105, 100, 100),  # back inside
            ("09:40:00", 100, 107, 100, 106),  # re-breakout
        ])
        assert s.signal(df, 3).kind == SignalKind.ENTER_LONG
        assert s.signal(df, 4) is None
        assert s.signal(df, 5).kind == SignalKind.ENTER_LONG


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_long_breakout_executes_at_next_bar_open(self):
        s = OpeningRangeBreakout(minutes=15)
        df = _bars([
            ("09:15:00", 100, 101, 99, 100),
            ("09:20:00", 100, 102, 100, 101),
            ("09:25:00", 101, 102, 101, 102),  # OR locked
            ("09:30:00", 102, 106, 102, 105),  # breakout LONG signal
            ("09:35:00", 107, 108, 106, 108),  # fills LONG @ 107
            ("09:40:00", 108, 110, 107, 110),  # last bar
        ])
        r = run_backtest(s, df, max_intraday_loss_pct=20.0, apply_costs=False)
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 107
        # Force-close at 15:15 won't fire (all bars before then) → END_OF_DATA.
        assert t.exit_reason == "END_OF_DATA"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        assert "orb" in REGISTRY
        assert REGISTRY["orb"] is OpeningRangeBreakout
