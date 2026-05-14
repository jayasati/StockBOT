"""yfinance data loader + parquet cache for the strategies module.

One combined parquet per (interval, days) combo lives under
``backtest_cache/strategies/`` — e.g. ``5m_60d.parquet`` holds every symbol
ever fetched at 5-minute / 60-day. Subsequent fetches MERGE new symbols into
the existing file rather than overwriting.

Why a separate cache from ``backtest_cache/intraday_5m.parquet``: that one
is owned by the NIFTY-500 backtest runner and writes a fixed shape on every
run. Mixing the two would silently overwrite each other's data."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.yf_fetch import load_parquet, normalize_yf_batch, save_parquet

log = logging.getLogger("strategies.data")

CACHE_ROOT = Path("backtest_cache/strategies")

_OHLCV = ("open", "high", "low", "close", "volume")


def cache_path(interval: str, days: int) -> Path:
    return CACHE_ROOT / f"{interval}_{days}d.parquet"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, drop ``adj close``, keep only OHLCV in canonical order."""
    out = df.rename(columns=str.lower).copy()
    if "adj close" in out.columns:
        out = out.drop(columns=["adj close"])
    keep = [c for c in _OHLCV if c in out.columns]
    return out[keep].dropna(how="any")


def _yf_download(symbols: list[str], interval: str, days: int) -> dict[str, pd.DataFrame]:
    period = f"{days}d"
    log.info("yfinance: %d symbols, interval=%s, period=%s",
             len(symbols), interval, period)
    raw = yf.download(
        tickers=symbols, period=period, interval=interval,
        group_by="ticker", auto_adjust=False, progress=False, threads=True,
    )
    return normalize_yf_batch(raw, symbols, localize_ist=True)


def fetch(
    symbols: list[str],
    *,
    interval: str = "5m",
    days: int = 60,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch + cache 5m bars. Returns ``{symbol: lowercase-OHLCV DataFrame}``
    for the requested symbols only (not the whole cache).

    Cache merge rule: symbols already present in the cache are reused unless
    ``refresh=True``; missing symbols are downloaded and merged in. The
    merged dict is written back to the parquet so the next call is hot."""
    path = cache_path(interval, days)
    cached = {} if refresh else load_parquet(path)

    missing = [s for s in symbols if s not in cached]
    if missing:
        fetched = _yf_download(missing, interval, days)
        # Persist with capitalized columns (matches the existing parquet
        # convention used by backtest/swing) so this cache plays nicely
        # with anything else loading it via load_parquet.
        cached.update(fetched)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_parquet(cached, path)
        log.info("Cached %d new symbol(s) to %s (total: %d)",
                 len(fetched), path, len(cached))

    return {s: cached[s] for s in symbols if s in cached}


def load(
    symbol: str,
    *,
    interval: str = "5m",
    days: int = 60,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a single symbol as an engine-ready DataFrame (lowercase OHLCV).

    Raises ``KeyError`` if yfinance returned no data for the symbol."""
    data = fetch([symbol], interval=interval, days=days, refresh=refresh)
    if symbol not in data:
        raise KeyError(
            f"No yfinance data for {symbol!r} at interval={interval}, days={days}"
        )
    return _normalize_columns(data[symbol])


def cache_summary(interval: str = "5m", days: int = 60) -> pd.DataFrame:
    """Return a per-symbol summary (rows, first/last bar) of what's cached."""
    path = cache_path(interval, days)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "bars", "first_ts", "last_ts"])
    cached = load_parquet(path)
    rows = [
        (sym, len(df), df.index.min(), df.index.max())
        for sym, df in cached.items()
    ]
    return pd.DataFrame(rows, columns=["symbol", "bars", "first_ts", "last_ts"])
