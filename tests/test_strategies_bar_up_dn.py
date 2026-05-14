"""Tests for the BarUpDn strategy + the generic backtest engine."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from strategies import (
    BacktestResult, BarUpDn, REGISTRY, Signal, SignalKind, run_backtest,
)
from strategies.backtest import _bar_date

IST = ZoneInfo("Asia/Kolkata")


def _bars(rows, start="2026-05-04 09:15", freq="5min", tz=IST):
    """Build an OHLCV DataFrame from list of (o,h,l,c[,v]) tuples."""
    idx = pd.date_range(start=start, periods=len(rows), freq=freq, tz=tz)
    cols = ["open", "high", "low", "close", "volume"]
    data = []
    for r in rows:
        if len(r) == 4:
            o, h, l, c = r
            v = 1000
        else:
            o, h, l, c, v = r
        data.append((o, h, l, c, v))
    return pd.DataFrame(data, columns=cols, index=idx)


# ---------------------------------------------------------------------------
# Signal layer
# ---------------------------------------------------------------------------

class TestBarUpDnSignal:
    def setup_method(self):
        self.s = BarUpDn()

    def test_long_when_green_and_open_above_prev_close(self):
        df = _bars([(100, 101, 99, 100), (101, 103, 101, 103)])
        sig = self.s.signal(df, 1)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_LONG
        assert sig.reason == "BarUp"

    def test_short_when_red_and_open_below_prev_close(self):
        df = _bars([(100, 101, 99, 100), (99, 100, 96, 97)])
        sig = self.s.signal(df, 1)
        assert sig is not None
        assert sig.kind == SignalKind.ENTER_SHORT
        assert sig.reason == "BarDn"

    def test_first_bar_returns_none(self):
        df = _bars([(100, 101, 99, 100), (101, 103, 101, 103)])
        assert self.s.signal(df, 0) is None

    def test_green_bar_open_equal_to_prev_close_is_no_signal(self):
        # open > close[1] is strict
        df = _bars([(100, 101, 99, 100), (100, 102, 99, 102)])
        assert self.s.signal(df, 1) is None

    def test_red_bar_open_equal_to_prev_close_is_no_signal(self):
        df = _bars([(100, 101, 99, 100), (100, 101, 97, 98)])
        assert self.s.signal(df, 1) is None

    def test_green_bar_open_below_prev_close_is_no_signal(self):
        # green but gapped down — no long entry
        df = _bars([(100, 101, 99, 100), (98, 101, 97, 100)])
        assert self.s.signal(df, 1) is None

    def test_red_bar_open_above_prev_close_is_no_signal(self):
        # red but gapped up — no short entry
        df = _bars([(100, 101, 99, 100), (102, 103, 99, 101)])
        assert self.s.signal(df, 1) is None

    def test_doji_open_equals_close_is_no_signal(self):
        df = _bars([(100, 101, 99, 100), (101, 102, 100, 101)])
        assert self.s.signal(df, 1) is None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_signal_at_close_executes_at_next_open(self):
        df = _bars([
            (100, 100, 100, 100),  # bar 0: no signal
            (101, 103, 101, 103),  # bar 1: green > prev close → ENTER_LONG signal
            (104, 105, 103, 104),  # bar 2: fills LONG @ open=104
            (104, 106, 103, 106),  # bar 3
        ])
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=10.0,
                         apply_costs=False)
        # No exit before end-of-data → one trade closed at last close
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 104.0
        assert t.exit_price == 106.0
        assert t.exit_reason == "END_OF_DATA"
        # qty: 10% of 100k = 10k → int(10000/104) = 96
        assert t.qty == 96

    def test_reversal_long_to_short(self):
        df = _bars([
            (100, 100, 100, 100),  # bar 0: nothing
            (101, 103, 101, 103),  # bar 1: → ENTER_LONG
            (104, 104, 102, 102),  # bar 2: opens LONG @ 104; close 102 (red but open > prev close → no signal)
            (101, 102, 99, 99),    # bar 3: red, open 101 < prev close 102 → ENTER_SHORT
            (98, 100, 96, 97),     # bar 4: reversal — close LONG @ 98, open SHORT @ 98
            (95, 96, 93, 94),      # bar 5
        ])
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=20.0,
                         apply_costs=False)
        # Expect: LONG opened bar 2 @ 104, closed bar 4 @ 98 (REVERSAL),
        #         SHORT opened bar 4 @ 98, closed bar 5 @ 94 (END_OF_DATA).
        assert r.num_trades == 2
        t1, t2 = r.trades
        assert t1.side == "LONG"
        assert t1.entry_price == 104.0
        assert t1.exit_price == 98.0
        assert t1.exit_reason == "REVERSAL"
        assert t2.side == "SHORT"
        assert t2.entry_price == 98.0
        assert t2.exit_price == 94.0
        assert t2.exit_reason == "END_OF_DATA"
        # SHORT P&L: (98 - 94) * qty
        assert t2.pnl_gross == pytest.approx(4.0 * t2.qty)

    def test_no_signal_no_trades(self):
        # Flat OHLC — every bar is a doji, no signals
        df = _bars([(100, 100, 100, 100)] * 10)
        r = run_backtest(BarUpDn(), df, apply_costs=False)
        assert r.num_trades == 0
        assert r.ending_equity == r.starting_equity

    def test_short_dataframe_returns_empty_result(self):
        df = _bars([(100, 100, 100, 100)])
        r = run_backtest(BarUpDn(), df)
        assert r.num_trades == 0
        assert r.ending_equity == r.starting_equity

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"open": [1.0], "close": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            run_backtest(BarUpDn(), df)

    def test_costs_reduce_pnl(self):
        df = _bars([
            (100, 100, 100, 100),
            (101, 103, 101, 103),  # → ENTER_LONG
            (104, 110, 104, 110),  # fills @ 104; closes at last (110)
        ])
        gross = run_backtest(BarUpDn(), df, max_intraday_loss_pct=50.0,
                             apply_costs=False)
        net = run_backtest(BarUpDn(), df, max_intraday_loss_pct=50.0,
                           apply_costs=True)
        assert gross.num_trades == 1 and net.num_trades == 1
        assert net.trades[0].pnl_net < gross.trades[0].pnl_net
        assert net.trades[0].pnl_gross == gross.trades[0].pnl_gross


# ---------------------------------------------------------------------------
# Max-intraday-loss kill switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kill_closes_position_and_blocks_new_entries_same_day(self):
        # Day 1: long entry then big drop crosses 1% kill threshold.
        # Subsequent setups same day should be ignored.
        day1 = "2026-05-04 09:15"
        df = _bars(
            [
                (100, 100, 100, 100),  # bar 0: nothing
                (101, 103, 101, 103),  # bar 1: → ENTER_LONG signal
                (104, 105, 80, 82),    # bar 2: opens LONG @104; close 82 → big MTM loss → kill armed
                (80, 82, 78, 80),      # bar 3: kill closes @80; (red, open 80 < prev close 82 → would-be SHORT, but kill_today)
                (81, 83, 81, 83),      # bar 4: green, open 81 < prev close 80? no, 81>80 → would-be LONG — still blocked
            ],
            start=day1,
        )
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=1.0,
                         apply_costs=False)
        assert r.kill_switch_triggers == 1
        # Exactly one trade — the killed LONG.
        assert r.num_trades == 1
        t = r.trades[0]
        assert t.side == "LONG"
        assert t.entry_price == 104.0
        assert t.exit_price == 80.0
        assert t.exit_reason == "MAX_INTRADAY_LOSS"

    def test_kill_resets_next_day(self):
        # Two-day series: kill on day 1, then a clean LONG signal on day 2 fires.
        day1 = "2026-05-04 09:15"
        day2 = "2026-05-05 09:15"
        df1 = _bars(
            [
                (100, 100, 100, 100),
                (101, 103, 101, 103),
                (104, 105, 80, 82),    # kill armed
                (80, 82, 78, 80),      # killed
            ],
            start=day1,
        )
        df2 = _bars(
            [
                (90, 91, 89, 90),      # day 2 bar 0
                (91, 94, 91, 94),      # day 2 bar 1: green, 91>90 → ENTER_LONG
                (95, 97, 94, 97),      # day 2 bar 2: opens LONG @ 95
                (97, 99, 96, 99),      # closes at end-of-data
            ],
            start=day2,
        )
        df = pd.concat([df1, df2])
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=1.0,
                         apply_costs=False)
        assert r.kill_switch_triggers == 1
        assert r.num_trades == 2
        kill_trade, day2_trade = r.trades
        assert kill_trade.exit_reason == "MAX_INTRADAY_LOSS"
        assert day2_trade.side == "LONG"
        assert day2_trade.entry_price == 95.0
        assert day2_trade.exit_reason == "END_OF_DATA"

    def test_no_kill_when_loss_under_threshold(self):
        df = _bars([
            (100, 100, 100, 100),
            (101, 103, 101, 103),  # → ENTER_LONG
            (104, 105, 103, 103),  # opens @ 104; mild drift
            (103, 104, 102, 103),
        ])
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=5.0,
                         apply_costs=False)
        assert r.kill_switch_triggers == 0


# ---------------------------------------------------------------------------
# Misc plumbing
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_bar_up_dn_registered(self):
        assert "bar_up_dn" in REGISTRY
        assert REGISTRY["bar_up_dn"] is BarUpDn

    def test_strategy_repr(self):
        s = BarUpDn(foo=1)
        assert "BarUpDn" in repr(s)


class TestBarDateHelper:
    def test_returns_date_for_timestamp(self):
        ts = pd.Timestamp("2026-05-04 09:15", tz=IST)
        assert _bar_date(ts) == ts.date()

    def test_returns_none_for_non_timestamp(self):
        assert _bar_date(42) is None


class TestResultShape:
    def test_trades_df_returns_empty_frame_when_no_trades(self):
        df = _bars([(100, 100, 100, 100)] * 5)
        r = run_backtest(BarUpDn(), df)
        assert isinstance(r, BacktestResult)
        out = r.trades_df()
        assert out.empty
        assert "entry_ts" in out.columns

    def test_trades_df_columns(self):
        df = _bars([
            (100, 100, 100, 100),
            (101, 103, 101, 103),
            (104, 105, 103, 105),
        ])
        r = run_backtest(BarUpDn(), df, max_intraday_loss_pct=10.0,
                         apply_costs=False)
        out = r.trades_df()
        assert len(out) == 1
        for col in ("entry_ts", "exit_ts", "side", "entry_price",
                    "exit_price", "qty", "pnl_gross", "pnl_net",
                    "return_pct", "entry_reason", "exit_reason"):
            assert col in out.columns

    def test_equity_curve_length_matches_input(self):
        df = _bars([(100, 100, 100, 100)] * 8)
        r = run_backtest(BarUpDn(), df)
        assert len(r.equity_curve) == len(df)
