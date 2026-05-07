"""Backtest module for the stock alert bot.

Replays ``bot.score_stock()`` forward in time on historical 5-minute bars and
reports win rates at +30min, +1day, +5days. Records every alert (after
applying the bot's cooldown) to ``results/backtest.csv``.

Note on data range: yfinance hard-limits 5-minute data to the last ~60
calendar days. Daily data is fetched for the full 6 months. The replay
therefore covers roughly the most recent 60 days of intraday history.

CLI: ``python -m backtest`` (replaces the old ``python backtest.py``).

Public surface — re-exported here for the ``swing/`` package which depends
on ``CACHE_DIR``, ``DAILY_CACHE``, ``WIN_THRESHOLD_PCT``, ``compute_win_rate``."""
from .config import (
    CACHE_DIR,
    COOLDOWN_MINUTES,
    DAILY_CACHE,
    INTRADAY_CACHE,
    LOOKBACK_BARS,
    RESULTS_CSV,
    SCORE_THRESHOLD,
    WARMUP_BARS,
    WIN_THRESHOLD_PCT,
)
from .data import fetch_daily, fetch_intraday_5m
from .replay import AlertRecord, replay
from .report import (
    apply_cooldown,
    compute_win_rate,
    save_csv,
    summarize,
    sweep_summary,
)

__all__ = [
    "AlertRecord",
    "CACHE_DIR",
    "COOLDOWN_MINUTES",
    "DAILY_CACHE",
    "INTRADAY_CACHE",
    "LOOKBACK_BARS",
    "RESULTS_CSV",
    "SCORE_THRESHOLD",
    "WARMUP_BARS",
    "WIN_THRESHOLD_PCT",
    "apply_cooldown",
    "compute_win_rate",
    "fetch_daily",
    "fetch_intraday_5m",
    "replay",
    "save_csv",
    "summarize",
    "sweep_summary",
]
