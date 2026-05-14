"""Tests for the intraday MIS enforcement layer in strategies.backtest:
auto-flatten at session-close (default 15:15 IST) + late-entry cutoff
(default 14:30 IST)."""
from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from strategies import BarUpDn, run_backtest
from strategies.backtest import (
    DEFAULT_ENTRY_CUTOFF, DEFAULT_FORCE_CLOSE, SignalKind, Signal, Strategy,
)

IST = ZoneInfo("Asia/Kolkata")


def _bars_at(times_oc, date="2026-05-04"):
    """Build OHLCV at the specified IST clock times. Each row is
    ``(time_str, open, close)``; high/low set to max/min of o/c."""
    rows = []
    idx = []
    for t_str, o, c in times_oc:
        idx.append(pd.Timestamp(f"{date} {t_str}", tz=IST))
        rows.append((o, max(o, c) + 0.1, min(o, c) - 0.1, c, 1000))
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close", "volume"], index=idx,
    )


class _ForceLong(Strategy):
    """Test helper: signal LONG on every bar so we can probe gating."""
    name = "_force_long"

    def signal(self, df, i):
        return Signal(SignalKind.ENTER_LONG, "test")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_force_close_is_15_15(self):
        assert DEFAULT_FORCE_CLOSE == time(15, 15)

    def test_default_entry_cutoff_is_14_30(self):
        assert DEFAULT_ENTRY_CUTOFF == time(14, 30)


# ---------------------------------------------------------------------------
# Force close at session end
# ---------------------------------------------------------------------------

class TestForceClose:
    def test_open_position_auto_closes_at_15_15(self):
        # Force LONG at 12:00, then bar at 15:15 should auto-flatten.
        df = _bars_at([
            ("12:00:00", 100, 100),  # bar 0
            ("12:05:00", 100, 100),  # bar 1: pending LONG signal from bar 0 fills
            ("15:10:00", 101, 101),  # bar 2: still pre-cutoff, position holds
            ("15:15:00", 102, 102),  # bar 3: force-close at this bar's CLOSE
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=20.0,
                         apply_costs=False)
        assert r.session_close_triggers == 1
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.exit_reason == "SESSION_CLOSE"
        assert t.exit_price == 102

    def test_no_close_when_already_flat(self):
        df = _bars_at([
            ("15:15:00", 100, 100),
            ("15:20:00", 100, 100),
        ])
        # No prior position; the 15:15 bar shouldn't manufacture a close.
        # Use a strategy that never signals.
        from strategies.base import Signal as _S
        class _Never(Strategy):
            name = "_never"
            def signal(self, df, i):
                return None
        r = run_backtest(_Never(), df, apply_costs=False)
        assert r.session_close_triggers == 0
        assert r.num_trades == 0

    def test_force_close_disabled_when_none(self):
        # With force_close_time=None, position holds across days as before.
        df = pd.concat([
            _bars_at([("14:00:00", 100, 100), ("14:05:00", 100, 100),
                      ("15:20:00", 110, 110)], date="2026-05-04"),
            _bars_at([("09:15:00", 110, 110)], date="2026-05-05"),
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=50.0,
                         apply_costs=False, force_close_time=None,
                         entry_cutoff_time=None)
        assert r.session_close_triggers == 0
        # Single multi-day trade closed by END_OF_DATA.
        assert r.num_trades == 1
        assert r.trades[0].exit_reason == "END_OF_DATA"

    def test_force_close_resets_per_day(self):
        # Two-day series: force-close fires on each day independently.
        df = pd.concat([
            _bars_at([("12:00:00", 100, 100), ("12:05:00", 100, 100),
                      ("15:15:00", 105, 105)], date="2026-05-04"),
            _bars_at([("12:00:00", 105, 105), ("12:05:00", 105, 105),
                      ("15:15:00", 110, 110)], date="2026-05-05"),
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=50.0,
                         apply_costs=False)
        assert r.session_close_triggers == 2
        assert r.num_trades == 2
        for t in r.trades:
            assert t.exit_reason == "SESSION_CLOSE"


# ---------------------------------------------------------------------------
# Entry cutoff (no new entries after 14:30)
# ---------------------------------------------------------------------------

class TestEntryCutoff:
    def test_no_new_entry_after_cutoff(self):
        # ForceLong signals every bar. After 14:30, no new entries should
        # be generated, so no trade ever opens.
        df = _bars_at([
            ("14:30:00", 100, 100),  # at cutoff: blocked
            ("14:35:00", 100, 100),  # past cutoff: blocked
            ("14:40:00", 100, 100),
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=20.0,
                         apply_costs=False)
        assert r.num_trades == 0

    def test_signal_at_14_25_still_executes_at_14_30(self):
        # Signal generated at 14:25 (before cutoff) should still fill at
        # 14:30 open per Pine semantics.
        df = _bars_at([
            ("14:25:00", 100, 100),  # pre-cutoff: signal generated
            ("14:30:00", 101, 101),  # at-cutoff: pending signal still fills
            ("15:15:00", 102, 102),  # force close
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=20.0,
                         apply_costs=False)
        assert r.num_trades == 1
        assert r.trades[0].entry_price == 101
        assert r.trades[0].exit_reason == "SESSION_CLOSE"

    def test_entry_cutoff_disabled_when_none(self):
        # With cutoff disabled, late signals fill normally.
        df = _bars_at([
            ("14:35:00", 100, 100),
            ("14:40:00", 101, 101),
            ("15:00:00", 102, 102),
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=20.0,
                         apply_costs=False, entry_cutoff_time=None,
                         force_close_time=None)
        # Signal at 14:35 fills at 14:40. Held to END_OF_DATA at 15:00.
        assert r.num_trades == 1
        assert r.trades[0].entry_price == 101
        assert r.trades[0].exit_reason == "END_OF_DATA"


# ---------------------------------------------------------------------------
# Cross-day reset
# ---------------------------------------------------------------------------

class TestDayReset:
    def test_after_force_close_can_re_enter_next_day(self):
        df = pd.concat([
            _bars_at([("12:00:00", 100, 100), ("12:05:00", 100, 100),
                      ("15:15:00", 100, 100)], date="2026-05-04"),
            _bars_at([("09:15:00", 100, 100), ("09:20:00", 100, 100),
                      ("15:15:00", 100, 100)], date="2026-05-05"),
        ])
        r = run_backtest(_ForceLong(), df, max_intraday_loss_pct=50.0,
                         apply_costs=False)
        # Day 1: enter, force-close. Day 2: enter again, force-close.
        assert r.num_trades == 2
        assert r.session_close_triggers == 2


# ---------------------------------------------------------------------------
# Real-data integration
# ---------------------------------------------------------------------------

class TestRealStrategyIntegration:
    def test_bar_up_dn_with_intraday_default_holds_no_overnight(self):
        # Three days of bars. Bar 1 of each day fires a BarUp signal that
        # fills bar 2 of the same day. By 15:15 of each day, position must
        # be FLAT (no overnight holds).
        days = []
        for date in ("2026-05-04", "2026-05-05", "2026-05-06"):
            days.append(_bars_at([
                ("09:15:00", 100, 100),
                ("09:20:00", 101, 103),  # green & open > prev close → ENTER_LONG
                ("09:25:00", 103, 103),  # fills LONG at 103
                ("12:00:00", 103, 103),
                ("15:15:00", 105, 105),  # force-close at 15:15
            ], date=date))
        df = pd.concat(days)
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=50.0,
                         apply_costs=False)
        # Each day produces exactly one LONG that auto-flattens.
        assert r.num_trades == 3
        assert r.session_close_triggers == 3
        for t in r.trades:
            assert t.exit_reason == "SESSION_CLOSE"
            assert (pd.Timestamp(t.entry_ts).date()
                    == pd.Timestamp(t.exit_ts).date()), \
                "Trade spans multiple sessions"
