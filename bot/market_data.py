"""Market-data layer: yfinance batch fetchers, the daily-history cache,
and the bridge to the Fyers live feed.

Holds module-level state (`_daily_cache`, `_LIVE_FEED_READY`,
`_asm_refresh_date`). Callers that need to read the cache should go through
``market_data._daily_cache`` rather than copying it at import time."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

import fyers_client
from data import realtime_feed

from . import suppression
from .config import IST
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot")

# Daily history changes once per day, so we cache it instead of re-fetching
# 130 times every scan. The cache is refreshed at bot start and once per day.
_daily_cache: dict[str, pd.DataFrame] = {}
_daily_cache_date: str = ""

# Chunk size for yfinance batch calls. At 500-symbol universes Yahoo
# starts throttling single batches; ~100 has been the empirical sweet
# spot between throughput and being blocked.
YF_CHUNK_SIZE = 100

_LIVE_FEED_READY = False
COLD_START_BAR_THRESHOLD = 30

_asm_refresh_date: str = ""


def _yf_download(symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    try:
        df = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as e:
        log.error("yfinance %s/%s fetch failed: %s", period, interval, e)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            sub = sub.dropna()
            if not sub.empty:
                result[sym] = sub
        except (KeyError, AttributeError):
            continue
    return result


def _yf_download_chunked(
    symbols: list[str], period: str, interval: str
) -> dict[str, pd.DataFrame]:
    if len(symbols) <= YF_CHUNK_SIZE:
        return _yf_download(symbols, period, interval)
    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), YF_CHUNK_SIZE):
        result.update(
            _yf_download(symbols[i : i + YF_CHUNK_SIZE], period, interval)
        )
    return result


def _to_upper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """realtime_feed returns lowercase columns (matches features.py); the
    legacy scoring functions expect yfinance's uppercase shape. Rename at
    the boundary so neither side has to know about the other."""
    return df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })


def ensure_live_feed() -> None:
    """Open the Fyers WebSocket once. Idempotent. If Fyers auth isn't set
    up (new install) we log and fall through to yfinance — the cold-start
    fallback below already handles that path."""
    global _LIVE_FEED_READY
    if _LIVE_FEED_READY or not WATCHLIST:
        return
    try:
        fy_symbols = [fyers_client.to_fyers(s) for s in WATCHLIST]
        realtime_feed.subscribe(fy_symbols)
        _LIVE_FEED_READY = True
        log.info("Live feed connected; subscribed to %d Fyers symbols", len(fy_symbols))
    except Exception as e:
        log.warning("Live feed unavailable (%s) — using yfinance only", e)


def fetch_intraday(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Return last 100 5-min bars per symbol from the live Fyers feed.

    Cold-start fallback: when a symbol has fewer than COLD_START_BAR_THRESHOLD
    bars cached, fetch from yfinance once, seed bars_5m, and serve from there.
    Subsequent scans hit the warm cache."""
    out: dict[str, pd.DataFrame] = {}
    cold: list[str] = []

    for sym in symbols:
        fy_sym = fyers_client.to_fyers(sym)
        df_lower = realtime_feed.get_5m_bars(fy_sym, n=100)
        if len(df_lower) < COLD_START_BAR_THRESHOLD:
            cold.append(sym)
        else:
            out[sym] = _to_upper_columns(df_lower)

    if cold:
        log.info(
            "Cold cache for %d/%d symbol(s); seeding from yfinance",
            len(cold), len(symbols),
        )
        yf_data = _yf_download_chunked(cold, period="5d", interval="5m")
        for sym, yf_df in yf_data.items():
            fy_sym = fyers_client.to_fyers(sym)
            try:
                realtime_feed.seed_from_yfinance(fy_sym, yf_df)
            except Exception as e:
                log.warning("seed_from_yfinance(%s) failed: %s", sym, e)
            # After seeding, the warm cache should serve the next call;
            # for THIS scan we use the yfinance frame directly so we don't
            # double-pay the SQLite round trip.
            out[sym] = yf_df

    return out


def fetch_daily_batch(symbols: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
    """Fetch daily history. Thin wrapper over ``data.yf_fetch.fetch_daily``;
    live path so caching is disabled."""
    from data.yf_fetch import fetch_daily as _fetch_daily
    return _fetch_daily(symbols, period_days=days, cache_path=None)


def refresh_daily_cache_if_stale() -> None:
    """Refresh the daily history cache once per calendar day."""
    global _daily_cache, _daily_cache_date
    today_str = datetime.now(IST).date().isoformat()
    if _daily_cache_date == today_str and _daily_cache:
        return
    log.info("Refreshing daily history cache for %d symbols...", len(WATCHLIST))
    _daily_cache = fetch_daily_batch(WATCHLIST, days=60)
    _daily_cache_date = today_str
    log.info("Daily cache loaded: %d/%d symbols", len(_daily_cache), len(WATCHLIST))
    missing = set(WATCHLIST) - set(_daily_cache.keys())
    if missing:
        log.warning("No daily data for: %s", ", ".join(sorted(missing)))


async def refresh_asm_gsm_if_stale() -> None:
    """Re-pull NSE ASM/GSM lists once per calendar day (IST)."""
    global _asm_refresh_date
    today_str = datetime.now(IST).date().isoformat()
    if _asm_refresh_date == today_str:
        return
    try:
        await suppression.refresh_asm_gsm()
        _asm_refresh_date = today_str
    except Exception as e:
        log.exception("ASM/GSM refresh failed: %s", e)
