"""yfinance daily-bar fetchers used by the swing strategy.

Watchlist OHLC goes through ``data.yf_fetch.fetch_daily`` (with parquet
cache for backtest replays). NIFTY is fetched separately because it's a
single ticker and we cache it at a per-period filename so the backtest's
``--years`` flag doesn't fight the cache."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from backtest import CACHE_DIR, DAILY_CACHE
from data.yf_fetch import fetch_daily as _fetch_daily

from .config import HISTORY_DAYS, NIFTY_CACHE, NIFTY_TICKER

log = logging.getLogger("swing")


def _daily_cache_path(period_days: int) -> Path:
    """Separate cache file per period so changing --years doesn't return
    stale data. The 186-day default re-uses backtest.py's cache file."""
    if period_days == 186:
        return DAILY_CACHE
    return CACHE_DIR / f"daily_{period_days}d.parquet"


def _nifty_cache_path(period_days: int) -> Path:
    if period_days == 186:
        return NIFTY_CACHE
    return CACHE_DIR / f"nifty_{period_days}d.parquet"


def fetch_nifty(period_days: int = 186, use_cache: bool = True) -> pd.DataFrame:
    cache_path = _nifty_cache_path(period_days)
    if use_cache and cache_path.exists():
        log.info("Loading NIFTY data from cache: %s", cache_path)
        df = pd.read_parquet(cache_path, engine="pyarrow")
        return df.set_index("timestamp").sort_index()
    log.info("Fetching NIFTY daily for %d days from yfinance...", period_days)
    df = yf.download(
        tickers=NIFTY_TICKER, period=f"{period_days}d", interval="1d",
        progress=False, auto_adjust=False,
    )
    df = df.dropna(how="all")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index.name = "timestamp"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(cache_path, engine="pyarrow", index=False)
    log.info("Got %d days of NIFTY data", len(df))
    return df


def load_or_fetch_daily(
    symbols: list[str], period_days: int = 186, use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Watchlist daily OHLC. Backtest path — uses parquet cache."""
    return _fetch_daily(
        symbols,
        period_days=period_days,
        cache_path=_daily_cache_path(period_days),
        refresh=not use_cache,
    )


def fetch_alert_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Pull watchlist + NIFTY for the live alert path (no parquet cache)."""
    return _fetch_daily(symbols + [NIFTY_TICKER], period_days=HISTORY_DAYS)
