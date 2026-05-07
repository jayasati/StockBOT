"""Unified yfinance daily-batch fetcher with optional parquet cache.

Replaces four near-identical implementations that used to live in
``bot/market_data.py``, ``backtest.py``, and the swing module. Live path
passes ``cache_path=None``; backtest replays pass a parquet path so
subsequent runs skip the network call.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger("alertbot.yf_fetch")

YF_CHUNK_SIZE = 100  # Yahoo throttles single batches above this


# ---------------------------------------------------------------------------
# Helpers (also reused by backtest's 5m-intraday parquet cache)
# ---------------------------------------------------------------------------

def normalize_yf_batch(
    df: pd.DataFrame,
    symbols: list[str],
    localize_ist: bool = False,
) -> dict[str, pd.DataFrame]:
    """Slice a yfinance multi-index batch result into per-symbol DataFrames.

    ``localize_ist=True`` — convert tz-aware UTC index to IST (5-min intraday).
    ``localize_ist=False`` — leave the index as-is (daily case).

    Drops rows where every column is NaN (typical for missing tickers in a
    multi-symbol batch); preserves rows with partial NaN.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            sub = df[sym].copy() if isinstance(df.columns, pd.MultiIndex) else df.copy()
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        if localize_ist:
            if sub.index.tz is None:
                sub.index = sub.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            else:
                sub.index = sub.index.tz_convert("Asia/Kolkata")
        sub.index.name = "timestamp"
        out[sym] = sub
    return out


def save_parquet(data: dict[str, pd.DataFrame], path: Path) -> None:
    if not data:
        return
    pieces = []
    for sym, df in data.items():
        d = df.reset_index()
        d["symbol"] = sym
        pieces.append(d)
    combined = pd.concat(pieces, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, engine="pyarrow", index=False)


def load_parquet(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    combined = pd.read_parquet(path, engine="pyarrow")
    out: dict[str, pd.DataFrame] = {}
    for sym, group in combined.groupby("symbol"):
        df = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        out[sym] = df
    return out


def _yf_download_chunked(
    symbols: list[str], period: str, interval: str
) -> pd.DataFrame:
    """yfinance batch download with chunking. Concatenates per-chunk results
    column-wise so the caller sees one multi-index DataFrame regardless of
    whether the universe needed chunking."""
    if len(symbols) <= YF_CHUNK_SIZE:
        return yf.download(
            tickers=symbols, period=period, interval=interval,
            group_by="ticker", progress=False, auto_adjust=False, threads=True,
        )
    pieces = []
    for i in range(0, len(symbols), YF_CHUNK_SIZE):
        chunk = symbols[i:i + YF_CHUNK_SIZE]
        df = yf.download(
            tickers=chunk, period=period, interval=interval,
            group_by="ticker", progress=False, auto_adjust=False, threads=True,
        )
        pieces.append(df)
    return pd.concat(pieces, axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_daily(
    symbols: list[str],
    period_days: int = 60,
    cache_path: Path | str | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV bars for ``symbols`` from yfinance.

    Uses ``YF_CHUNK_SIZE``-symbol batches under the hood — Yahoo throttles
    single calls above ~100 tickers.

    ``cache_path=None`` disables caching (live path).
    ``cache_path=<parquet>`` reads + writes to that file.
    ``refresh=True`` bypasses the read but still writes after fetching.
    """
    cache_path = Path(cache_path) if cache_path is not None else None

    if cache_path is not None and not refresh and cache_path.exists():
        cached = load_parquet(cache_path)
        out = {s: cached[s] for s in symbols if s in cached}
        if out:
            log.info(
                "Loaded daily cache (%dd) %s: %d/%d symbols",
                period_days, cache_path.name, len(out), len(symbols),
            )
            return out

    log.info("Fetching %d days of daily data for %d symbols...",
             period_days, len(symbols))
    raw = _yf_download_chunked(symbols, period=f"{period_days}d", interval="1d")
    data = normalize_yf_batch(raw, symbols, localize_ist=False)
    log.info("Got daily data for %d/%d symbols", len(data), len(symbols))

    if cache_path is not None:
        save_parquet(data, cache_path)
    return data
