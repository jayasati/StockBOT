"""Trading strategies — pluggable, one module per strategy.

Add a new strategy by:
1. Writing ``strategies/<name>.py`` with a ``Strategy`` subclass.
2. Registering it in ``REGISTRY`` below.
3. Adding tests in ``tests/test_strategies_<name>.py``.

The backtest engine in :mod:`strategies.backtest` is generic — strategies
only emit signals; the engine handles fills, reversals, costs, and the
max-intraday-loss kill switch."""
from __future__ import annotations

from .backtest import BacktestResult, Trade, run_backtest
from .bar_up_dn import BarUpDn
from .base import Signal, SignalKind, Strategy
from .bollinger_bands import BollingerBandsDirected
from .ema_cross import EMACross
from .macd_cross import MACDCross
from .orb import OpeningRangeBreakout
from .rsi_cross import RSICross

REGISTRY: dict[str, type[Strategy]] = {
    BarUpDn.name: BarUpDn,
    BollingerBandsDirected.name: BollingerBandsDirected,
    EMACross.name: EMACross,
    MACDCross.name: MACDCross,
    OpeningRangeBreakout.name: OpeningRangeBreakout,
    RSICross.name: RSICross,
}


def _import_sweep():
    # Lazy: sweep imports back from this module, so we expose it at the
    # package surface without creating a circular import at load time.
    from .sweep import rollup_by_strategy, run_sweep
    return run_sweep, rollup_by_strategy


run_sweep, rollup_by_strategy = _import_sweep()


__all__ = [
    "BacktestResult",
    "BarUpDn",
    "BollingerBandsDirected",
    "EMACross",
    "MACDCross",
    "OpeningRangeBreakout",
    "REGISTRY",
    "RSICross",
    "Signal",
    "SignalKind",
    "Strategy",
    "Trade",
    "rollup_by_strategy",
    "run_backtest",
    "run_sweep",
]
