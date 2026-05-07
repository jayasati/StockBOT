"""yfinance fetchers for the backtest replay.

Daily delegates fully to ``data.yf_fetch.fetch_daily`` (with parquet cache).
The 5-min intraday path keeps its own implementation because yfinance's 5m
endpoint has different normalization needs (UTC → IST localization)."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from data.yf_fetch import (
    fetch_daily as _yf_fetch_daily,
    load_parquet,
    normalize_yf_batch,
    save_parquet,
)

from .config import DAILY_CACHE, INTRADAY_CACHE

log = logging.getLogger("backtest")


def fetch_intraday_5m(
    symbols: list[str], days: int = 60, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    if use_cache and INTRADAY_CACHE.exists():
        log.info("Loading 5m data from cache: %s", INTRADAY_CACHE)
        cached = load_parquet(INTRADAY_CACHE)
        if cached:
            log.info("  %d symbols loaded from cache", len(cached))
            return cached
    log.info("Fetching %d days of 5m data for %d symbols...", days, len(symbols))
    df = yf.download(
        tickers=symbols, period=f"{days}d", interval="5m",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )
    data = normalize_yf_batch(df, symbols, localize_ist=True)
    log.info("Got 5m data for %d/%d symbols", len(data), len(symbols))
    save_parquet(data, INTRADAY_CACHE)
    return data


def fetch_daily(
    symbols: list[str], months: int = 6, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    return _yf_fetch_daily(
        symbols,
        period_days=months * 31,
        cache_path=DAILY_CACHE,
        refresh=not use_cache,
    )
