"""Trade-execution primitives: position sizing + transaction costs.

This package is the natural home for execution-side code (sizing, fills,
cost models, slippage). Future modules — order routing, position state,
fill simulation — also belong here.

Public surface:
  size_position(account, entry, stop, risk_pct=0.01)      shares to trade
  validate_trade(qty, entry, account_inr)                 (ok, reason)
  round_trip_cost(notional_inr, side='intraday_eq')       brokerage + STT + GST + ...
  slippage_bps(symbol_class, time_of_day=None)            per-leg slippage
  net_r_multiple(entry, stop, exit_price, qty, ...)       realised R after costs/slippage
"""
from .costs import net_r_multiple, round_trip_cost, slippage_bps
from .risk import (
    DEFAULT_RISK_PCT,
    MIN_NOTIONAL_INR,
    size_position,
    validate_trade,
)

__all__ = [
    "DEFAULT_RISK_PCT",
    "MIN_NOTIONAL_INR",
    "net_r_multiple",
    "round_trip_cost",
    "size_position",
    "slippage_bps",
    "validate_trade",
]
