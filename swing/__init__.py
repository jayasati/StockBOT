"""End-of-day swing strategy: replay (backtest) + alert (daily dispatch).

Single home for the swing-trade signal that used to be split across
``swing_backtest.py`` and ``swing_alert.py``. Both paths share one
implementation of the signal logic in :mod:`swing.evaluate`.

Strategy:
  Signal — vol >= 2x 20-day avg AND close in top 10% of today's range AND
           close > 20-day EMA
  Regime — NIFTY closed up >= 0.30% AND breadth >= 55% of watchlist
  Trade  — enter T+1 open, hold 3 days, exit T+3 close

Public surface:
  evaluate_swing(daily, nifty, ...)   the signal evaluator (alert + replay)
  SwingAlert                          per-hit dataclass
  compute_breadth_series(daily)       vectorised breadth %
  fetch_nifty(period_days)            NIFTY OHLC with parquet cache
  load_or_fetch_daily(symbols, ...)   watchlist OHLC with parquet cache
  summarize(alerts, label)            backtest report
  regime_lift(off, on)                regime-filter A/B
  save_csv(alerts, path)              CSV export

CLIs:
  ``python -m swing.alert``   daily dispatcher (replaces ``swing_alert.py``)
  ``python -m swing.replay``  backtest CLI    (replaces ``swing_backtest.py``)
"""
from .config import (
    BREADTH_THRESHOLD_PCT,
    EMA_PERIOD,
    HISTORY_DAYS,
    HORIZONS,
    IST,
    MAX_EXTENSION_PCT,
    NIFTY_CACHE,
    NIFTY_TICKER,
    NIFTY_UP_PCT,
    RANGE_POSITION_THRESHOLD,
    RESULTS_CSV,
    VOLUME_MULTIPLE,
)
from .data import fetch_alert_universe, fetch_nifty, load_or_fetch_daily
from .evaluate import SwingAlert, compute_breadth_series, evaluate_swing
from .report import regime_lift, save_csv, summarize

__all__ = [
    "BREADTH_THRESHOLD_PCT",
    "EMA_PERIOD",
    "HISTORY_DAYS",
    "HORIZONS",
    "IST",
    "MAX_EXTENSION_PCT",
    "NIFTY_CACHE",
    "NIFTY_TICKER",
    "NIFTY_UP_PCT",
    "RANGE_POSITION_THRESHOLD",
    "RESULTS_CSV",
    "SwingAlert",
    "VOLUME_MULTIPLE",
    "compute_breadth_series",
    "evaluate_swing",
    "fetch_alert_universe",
    "fetch_nifty",
    "load_or_fetch_daily",
    "regime_lift",
    "save_csv",
    "summarize",
]
