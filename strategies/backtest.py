"""Generic strategy backtester with Pine semantics.

- A signal generated AT bar i's close is filled at bar (i+1)'s open.
- ENTER_LONG while SHORT (and vice-versa) reverses: close at next open,
  open the new side at the same open price.
- ``max_intraday_loss_pct`` mirrors Pine ``strategy.risk.max_intraday_loss``:
  when the day's P&L (realized + open mark-to-market) drops below the
  configured percentage of the day-start equity, all positions are closed
  at the next bar's open and no further entries fire until the next
  calendar day.

Intraday MIS enforcement (matches ``paper.tracker._is_timed_out`` +
``LATE_SESSION_CUTOFF``):

- ``entry_cutoff_time`` (default 14:30 IST) — no NEW signals are generated
  on or after this time. Pending signals from earlier still execute.
- ``force_close_time`` (default 15:15 IST) — any open position is force-
  closed at the bar's close with reason ``SESSION_CLOSE``. Brokers auto-
  squareoff MIS positions in the 15:15-15:20 window; this matches that.

Pass ``force_close_time=None`` and ``entry_cutoff_time=None`` to disable
intraday enforcement (e.g. for swing-style backtests, though the cost
model in ``trading.costs`` still assumes intraday MIS rates).

Costs route through ``trading.costs.round_trip_cost`` so live and backtest
P&L use the same charge model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from trading.costs import round_trip_cost

from .base import Signal, SignalKind, Strategy

DEFAULT_ENTRY_CUTOFF = time(14, 30)
DEFAULT_FORCE_CLOSE = time(15, 15)

log = logging.getLogger("strategies.backtest")


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str  # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    qty: int
    pnl_gross: float
    pnl_net: float
    return_pct: float  # net P&L as % of starting_equity
    entry_reason: str
    exit_reason: str  # REVERSAL / MAX_INTRADAY_LOSS / EXIT_SIGNAL / END_OF_DATA


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    num_trades: int
    num_wins: int
    win_rate_pct: float
    max_drawdown_pct: float
    kill_switch_triggers: int
    session_close_triggers: int = 0

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "entry_ts", "exit_ts", "side", "entry_price", "exit_price",
                "qty", "pnl_gross", "pnl_net", "return_pct",
                "entry_reason", "exit_reason",
            ])
        return pd.DataFrame([t.__dict__ for t in self.trades])


def _bar_date(ts) -> object:
    """Return the calendar date of a bar timestamp, or ``None`` if the
    index isn't datetime-like (in which case the engine treats the whole
    series as one trading session)."""
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return None


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    *,
    starting_equity: float = 100_000.0,
    qty_pct: float = 0.10,
    max_intraday_loss_pct: float = 1.0,
    apply_costs: bool = True,
    force_close_time: time | None = DEFAULT_FORCE_CLOSE,
    entry_cutoff_time: time | None = DEFAULT_ENTRY_CUTOFF,
) -> BacktestResult:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    n = len(df)
    if n < 2:
        empty = pd.Series(dtype=float, index=df.index, name="equity")
        return BacktestResult(
            trades=[], equity_curve=empty,
            starting_equity=starting_equity, ending_equity=starting_equity,
            total_return_pct=0.0, num_trades=0, num_wins=0, win_rate_pct=0.0,
            max_drawdown_pct=0.0, kill_switch_triggers=0,
        )

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    equity = starting_equity
    position = "FLAT"
    entry_price = 0.0
    entry_ts: Optional[pd.Timestamp] = None
    entry_reason = ""
    qty = 0

    realized_pnl_today = 0.0
    day_start_equity = starting_equity
    current_day = _bar_date(df.index[0])
    kill_today = False
    flattened_today = False
    pending_signal: Optional[Signal] = None
    pending_kill = False
    kill_triggers = 0
    session_close_triggers = 0

    trades: list[Trade] = []
    equity_history: list[float] = []

    def _close_position(at_price: float, at_ts, reason: str) -> None:
        nonlocal equity, realized_pnl_today, position
        nonlocal entry_price, entry_ts, entry_reason, qty
        if position == "FLAT":
            return
        if position == "LONG":
            gross = (at_price - entry_price) * qty
        else:
            gross = (entry_price - at_price) * qty
        notional = entry_price * qty
        costs = (
            round_trip_cost(notional)["total"]
            if apply_costs and notional > 0 else 0.0
        )
        net = gross - costs
        equity += net
        realized_pnl_today += net
        trades.append(Trade(
            entry_ts=entry_ts, exit_ts=at_ts, side=position,
            entry_price=entry_price, exit_price=at_price, qty=qty,
            pnl_gross=gross, pnl_net=net,
            return_pct=net / starting_equity * 100.0,
            entry_reason=entry_reason, exit_reason=reason,
        ))
        position = "FLAT"
        entry_price = 0.0
        entry_ts = None
        entry_reason = ""
        qty = 0

    def _open_position(side: str, at_price: float, at_ts, reason: str) -> None:
        nonlocal position, entry_price, entry_ts, entry_reason, qty
        if at_price <= 0:
            return
        notional = max(0.0, equity) * qty_pct
        new_qty = int(notional / at_price)
        if new_qty <= 0:
            return
        position = side
        entry_price = at_price
        entry_ts = at_ts
        entry_reason = reason
        qty = new_qty

    for i in range(n):
        ts = df.index[i]
        bar_day = _bar_date(ts)

        if bar_day != current_day:
            # Day rollover: any open position carries equity forward, but
            # day-start equity / kill / flatten flags reset for the new session.
            current_day = bar_day
            if position == "LONG":
                unreal = (opens[i] - entry_price) * qty
            elif position == "SHORT":
                unreal = (entry_price - opens[i]) * qty
            else:
                unreal = 0.0
            day_start_equity = equity + unreal
            realized_pnl_today = 0.0
            kill_today = False
            flattened_today = False

        bar_open = opens[i]
        bar_close = closes[i]

        if pending_kill:
            _close_position(bar_open, ts, "MAX_INTRADAY_LOSS")
            kill_today = True
            kill_triggers += 1
            pending_kill = False
            pending_signal = None

        if pending_signal is not None and not kill_today:
            kind = pending_signal.kind
            sig_reason = pending_signal.reason
            if kind == SignalKind.ENTER_LONG:
                if position == "SHORT":
                    _close_position(bar_open, ts, "REVERSAL")
                if position == "FLAT":
                    _open_position("LONG", bar_open, ts, sig_reason)
            elif kind == SignalKind.ENTER_SHORT:
                if position == "LONG":
                    _close_position(bar_open, ts, "REVERSAL")
                if position == "FLAT":
                    _open_position("SHORT", bar_open, ts, sig_reason)
            elif kind == SignalKind.EXIT:
                _close_position(bar_open, ts, "EXIT_SIGNAL")
            pending_signal = None

        if position == "LONG":
            unreal = (bar_close - entry_price) * qty
        elif position == "SHORT":
            unreal = (entry_price - bar_close) * qty
        else:
            unreal = 0.0
        equity_history.append(equity + unreal)

        bar_time = ts.time() if isinstance(ts, pd.Timestamp) else None
        if (force_close_time is not None and bar_time is not None
                and bar_time >= force_close_time
                and position != "FLAT" and not flattened_today):
            _close_position(bar_close, ts, "SESSION_CLOSE")
            flattened_today = True
            session_close_triggers += 1
            pending_signal = None

        day_pnl = realized_pnl_today + unreal
        threshold = -(max_intraday_loss_pct / 100.0) * day_start_equity
        if not kill_today and day_pnl < threshold:
            pending_kill = True
            pending_signal = None
            continue

        in_late_session = (
            entry_cutoff_time is not None and bar_time is not None
            and bar_time >= entry_cutoff_time
        )
        if not kill_today and not flattened_today and not in_late_session:
            new_sig = strategy.signal(df, i)
            if new_sig is not None:
                pending_signal = new_sig

    if position != "FLAT":
        _close_position(float(closes[-1]), df.index[-1], "END_OF_DATA")

    equity_series = pd.Series(equity_history, index=df.index, name="equity")
    if not equity_series.empty:
        running_max = equity_series.cummax()
        dd = (equity_series - running_max) / running_max * 100.0
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    wins = sum(1 for t in trades if t.pnl_net > 0)
    return BacktestResult(
        trades=trades,
        equity_curve=equity_series,
        starting_equity=starting_equity,
        ending_equity=equity,
        total_return_pct=(equity - starting_equity) / starting_equity * 100.0,
        num_trades=len(trades),
        num_wins=wins,
        win_rate_pct=(wins / len(trades) * 100.0) if trades else 0.0,
        max_drawdown_pct=max_dd,
        kill_switch_triggers=kill_triggers,
        session_close_triggers=session_close_triggers,
    )
