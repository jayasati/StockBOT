"""Trading strategies — pluggable, one module per strategy.

Add a new strategy by:
1. Writing ``strategies/<name>.py`` with a ``Strategy`` subclass.
2. Registering it in ``REGISTRY`` below.
3. Adding tests in ``tests/test_strategies_<name>.py``.

The backtest engine in :mod:`strategies.backtest` is generic — strategies
only emit signals; the engine handles fills, reversals, costs, and the
max-intraday-loss kill switch."""
from __future__ import annotations

from .adx_supertrend_atr import AdxSupertrendAtr
from .backtest import BacktestResult, Trade, run_backtest
from .bar_up_dn import BarUpDn
from .base import Signal, SignalKind, Strategy
from .bollinger_bands import BollingerBandsDirected
from .ema_cross import EMACross
from .macd_cross import MACDCross
from .macd_rsi_ema import MacdRsiEma
from .orb import OpeningRangeBreakout
from .rsi_cross import RSICross
from .vwap_ema_volume import VwapEmaVolume
from .vwap_rsi_orderflow import VwapRsiOrderflow

REGISTRY: dict[str, type[Strategy]] = {
    AdxSupertrendAtr.name: AdxSupertrendAtr,
    BarUpDn.name: BarUpDn,
    BollingerBandsDirected.name: BollingerBandsDirected,
    EMACross.name: EMACross,
    MACDCross.name: MACDCross,
    MacdRsiEma.name: MacdRsiEma,
    OpeningRangeBreakout.name: OpeningRangeBreakout,
    RSICross.name: RSICross,
    VwapEmaVolume.name: VwapEmaVolume,
    VwapRsiOrderflow.name: VwapRsiOrderflow,
}


def _import_sweep():
    # Lazy: sweep imports back from this module, so we expose it at the
    # package surface without creating a circular import at load time.
    from .sweep import rollup_by_strategy, run_sweep
    return run_sweep, rollup_by_strategy


run_sweep, rollup_by_strategy = _import_sweep()


__all__ = [
    "AdxSupertrendAtr",
    "BacktestResult",
    "BarUpDn",
    "BollingerBandsDirected",
    "EMACross",
    "MACDCross",
    "MacdRsiEma",
    "OpeningRangeBreakout",
    "REGISTRY",
    "RSICross",
    "Signal",
    "SignalKind",
    "Strategy",
    "Trade",
    "VwapEmaVolume",
    "VwapRsiOrderflow",
    "rollup_by_strategy",
    "run_backtest",
    "run_sweep",
]
