"""Backtest constants — paths, thresholds, replay window."""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

CACHE_DIR = Path("backtest_cache")
INTRADAY_CACHE = CACHE_DIR / "intraday_5m.parquet"
DAILY_CACHE = CACHE_DIR / "daily.parquet"
RESULTS_CSV = Path("results/backtest.csv")

SCORE_THRESHOLD = 60
WIN_THRESHOLD_PCT = 1.5
COOLDOWN_MINUTES = 60
LOOKBACK_BARS = 100  # passed into score_stock for RSI + same-session VWAP
WARMUP_BARS = 20
