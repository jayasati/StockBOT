"""Multi-timeframe aggregation layer.

Derives 15m, 1h, and 1d candles from canonical 5m bars stored in SQLite
(``bars_5m``). The single public entry point is :func:`get_bars`.

Design contract:
  * Boundaries match TradingView Indian-market candle convention:
    - 15m: session-anchored to 09:15 IST (offset="3h45min" from UTC midnight)
    - 1h:  six explicit buckets per session (last is 14:15–15:30, 75 min)
    - 1d:  one candle per session, indexed at midnight IST of that date
  * In-progress / partial candles are never returned. Completion is checked
    against ``_now`` (injected for testing, defaults to wall-clock IST).
  * LRU-cached on ``(symbol, tf, last_bar_ts)`` so cache invalidates the
    moment a new 5m bar lands. ``_now`` is intentionally NOT in the cache
    key — incomplete-bar removal happens after retrieval, so live calls
    still hit the warm cache.
  * All returned DataFrames carry a tz-aware DatetimeIndex in IST and the
    columns ``[open, high, low, close, volume]`` in that order.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from data.trading_calendar import (
    get_session_close,
    get_session_open,
    is_trading_day,
)

IST = "Asia/Kolkata"
DEFAULT_DB_PATH: Path = Path("alerts.db")

# 5m bars per derived candle. Used to size the SQL fetch window so that
# n derived bars are reachable without a full table scan.
_BARS_PER_TF: dict[str, int] = {"5m": 1, "15m": 3, "1h": 12, "1d": 75}

# fetch_limit is computed inside the cache, where the user's ``n`` is
# unavailable (n is intentionally not in the cache key). We size for the
# default n=200 — larger n requests fall back to whatever was cached.
_CACHE_DEFAULT_N: int = 200

_OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]

Timeframe = Literal["5m", "15m", "1h", "1d"]


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Open a fresh sqlite3 connection to the alerts DB.

    Tests monkeypatch this to inject an in-memory connection.
    """
    return sqlite3.connect(DEFAULT_DB_PATH)


def _fetch_last_bar_ts(symbol: str) -> Optional[int]:
    """Return the max ``ts`` (UTC epoch ms) for ``symbol``, or None if absent."""
    conn = _get_connection()
    cur = conn.execute(
        "SELECT MAX(ts) FROM bars_5m WHERE symbol = ?", (symbol,),
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _fetch_5m_raw(symbol: str, fetch_limit: int) -> pd.DataFrame:
    """Read the most recent ``fetch_limit`` 5m bars for ``symbol``.

    Returns a DataFrame in ascending time order with columns
    ``[ts, open, high, low, close, volume, ts_ist]`` where ``ts_ist`` is
    a tz-aware IST Timestamp. Returns an empty DataFrame if the symbol
    has no rows.
    """
    conn = _get_connection()
    cur = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars_5m "
        "WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
        (symbol, int(fetch_limit)),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume", "ts_ist"]
        )
    df = pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df = df.iloc[::-1].reset_index(drop=True)
    df["ts_ist"] = (
        pd.to_datetime(df["ts"], unit="ms", utc=True)
          .dt.tz_convert(IST)
    )
    return df


# ---------------------------------------------------------------------------
# Empty-DataFrame helpers
# ---------------------------------------------------------------------------

def _empty_resampled_df(with_completion: bool = False) -> pd.DataFrame:
    """Empty DataFrame with the public OHLCV schema and an IST DatetimeIndex.

    For 1d, the cached form additionally carries a ``_completion_ts`` column
    so :func:`_drop_incomplete` can decide which day's candle has finalised.
    """
    data: dict[str, pd.Series] = {col: pd.Series(dtype=float) for col in _OHLCV_COLS}
    if with_completion:
        data["_completion_ts"] = pd.Series(dtype=f"datetime64[ns, {IST}]")
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex([], tz=IST)
    return df


# ---------------------------------------------------------------------------
# 1h bucket assignment
# ---------------------------------------------------------------------------

def _assign_1h_bucket(ts_ist: pd.Timestamp) -> pd.Timestamp:
    """Map a 5m bar open time to the open timestamp of its 1h bucket.

    Returns the left boundary of the containing 1h window (tz-aware IST),
    or ``pd.NaT`` if ``ts_ist`` is outside the NSE session for that date.

    Buckets per session (mirrors TradingView Indian-market hourly candles):

        09:15–10:15   (60 min)
        10:15–11:15   (60 min)
        11:15–12:15   (60 min)
        12:15–13:15   (60 min)
        13:15–14:15   (60 min)
        14:15–15:30   (75 min — final bucket absorbs the session tail)
    """
    d = ts_ist.date()
    if not is_trading_day(d):
        return pd.NaT
    session_open = get_session_open(d)
    session_close = get_session_close(d)
    if ts_ist < session_open or ts_ist >= session_close:
        return pd.NaT
    offset_seconds = (ts_ist - session_open).total_seconds()
    hours_idx = int(offset_seconds // 3600)
    if hours_idx > 5:
        hours_idx = 5
    return session_open + pd.Timedelta(hours=hours_idx)


# ---------------------------------------------------------------------------
# Per-timeframe resampling
# ---------------------------------------------------------------------------

def _resample_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
    if df_5m.empty:
        return _empty_resampled_df()
    out = df_5m.set_index("ts_ist")[_OHLCV_COLS].copy()
    out.index.name = None
    return out


def _resample_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample to 15m candles anchored to 09:15 IST.

    The offset ``"3h45min"`` is the IST 09:15 anchor expressed relative to
    UTC midnight (09:15 IST = 03:45 UTC). With ``label="left"`` the index
    label is the bar's open time, matching TradingView convention.
    """
    if df_5m.empty:
        return _empty_resampled_df()
    df = df_5m.set_index("ts_ist")[_OHLCV_COLS]
    agg = (
        df.resample("15min", offset="3h45min", label="left", closed="left")
          .agg({
              "open": "first",
              "high": "max",
              "low": "min",
              "close": "last",
              "volume": "sum",
          })
          .dropna(subset=["open"])
    )
    agg.index.name = None
    return agg


def _resample_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample to TradingView-compatible 1h candles via explicit buckets."""
    if df_5m.empty:
        return _empty_resampled_df()
    df = df_5m.copy()
    df["bucket"] = df["ts_ist"].apply(_assign_1h_bucket)
    df = df[df["bucket"].notna()]
    if df.empty:
        return _empty_resampled_df()
    # apply() on a tz-aware Series may yield object dtype; pin to tz-aware
    # datetime so the resulting groupby index keeps the IST tz.
    df["bucket"] = pd.to_datetime(df["bucket"])
    grouped = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    if grouped.index.tz is None:
        grouped.index = grouped.index.tz_localize(IST)
    grouped.index.name = None
    return grouped


def _resample_1d(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the full session into one daily candle, indexed at midnight IST.

    The cached form carries ``_completion_ts`` (the last 5m bar's close time
    within that day). :func:`_drop_incomplete` uses it so the daily candle
    surfaces as soon as at least one 5m bar has closed for the day, even
    while the session is still in progress — matching TradingView's
    "live daily" candle semantics.
    """
    if df_5m.empty:
        return _empty_resampled_df(with_completion=True)
    df = df_5m.copy()
    df["session_date"] = df["ts_ist"].dt.normalize()
    grouped = df.groupby("session_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        _last_5m_open=("ts_ist", "max"),
    )
    grouped["_completion_ts"] = grouped["_last_5m_open"] + pd.Timedelta(minutes=5)
    grouped = grouped.drop(columns=["_last_5m_open"])
    if grouped.index.tz is None:
        grouped.index = grouped.index.tz_localize(IST)
    grouped.index.name = None
    return grouped


# ---------------------------------------------------------------------------
# Incomplete-bar removal
# ---------------------------------------------------------------------------

def _drop_incomplete(
    df: pd.DataFrame, tf: Timeframe, _now: pd.Timestamp,
) -> pd.DataFrame:
    """Remove the in-progress candle relative to ``_now``.

    A candle is complete when its window has fully elapsed:
      * 5m / 15m: ``index + duration <= _now``
      * 1h:       ``min(index + 1h, session_close(date)) <= _now`` to
                  honour the 14:15–15:30 last-bucket rule
      * 1d:       ``_completion_ts <= _now`` where ``_completion_ts`` is
                  the last 5m bar's close time within that session
    """
    if df.empty:
        return df
    if tf == "5m":
        complete_idx = df.index + pd.Timedelta(minutes=5)
        mask = complete_idx <= _now
    elif tf == "15m":
        complete_idx = df.index + pd.Timedelta(minutes=15)
        mask = complete_idx <= _now
    elif tf == "1h":
        completes = []
        for ts in df.index:
            cand = ts + pd.Timedelta(hours=1)
            session_close = get_session_close(ts.date())
            completes.append(min(cand, session_close))
        complete_idx = pd.DatetimeIndex(completes)
        if complete_idx.tz is None:
            complete_idx = complete_idx.tz_localize(IST)
        mask = complete_idx <= _now
    elif tf == "1d":
        mask = df["_completion_ts"] <= _now
    else:
        raise ValueError(f"Unknown timeframe: {tf!r}")
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Cached resample
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _get_resampled_full(
    symbol: str,
    tf: Timeframe,
    last_bar_ts: Optional[int],
) -> pd.DataFrame:
    """Cached resample of all available 5m bars for ``(symbol, tf)``.

    The cache key includes ``last_bar_ts`` (max ts in ``bars_5m`` for the
    symbol) so the moment a new 5m bar is persisted, the next call is a
    cache miss and a fresh resample runs. ``_now`` is intentionally not
    in the key — incomplete-bar removal is applied after retrieval so
    live callers stay on the warm cache entry.

    Returns the full resampled history (no .tail, no incomplete-bar drop).
    """
    if last_bar_ts is None:
        return _empty_resampled_df(with_completion=(tf == "1d"))

    fetch_limit = max(int(_CACHE_DEFAULT_N * _BARS_PER_TF[tf] * 1.5), 500)
    df_5m = _fetch_5m_raw(symbol, fetch_limit)
    if df_5m.empty:
        return _empty_resampled_df(with_completion=(tf == "1d"))

    if tf == "5m":
        return _resample_5m(df_5m)
    if tf == "15m":
        return _resample_15m(df_5m)
    if tf == "1h":
        return _resample_1h(df_5m)
    if tf == "1d":
        return _resample_1d(df_5m)
    raise ValueError(f"Unknown timeframe: {tf!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bars(
    symbol: str,
    tf: Timeframe,
    n: int = 200,
    _now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Return the last ``n`` complete bars for ``symbol`` at timeframe ``tf``.

    Parameters
    ----------
    symbol :
        Symbol whose 5m bars are stored in ``bars_5m``.
    tf :
        One of ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``.
    n :
        Maximum number of bars to return. If fewer complete bars are
        available, returns all of them.
    _now :
        Reference timestamp for incomplete-bar detection. Tests pass a
        fixed value for determinism; production calls leave it ``None``
        and use wall-clock IST.

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]``. Index is a tz-aware
        IST DatetimeIndex labelled at the bar's open time, sorted
        ascending. The current in-progress candle is never included.
    """
    if _now is None:
        _now = pd.Timestamp.now(tz=IST)
    last_bar_ts = _fetch_last_bar_ts(symbol)
    full_df = _get_resampled_full(symbol, tf, last_bar_ts)
    complete_df = _drop_incomplete(full_df, tf, _now)
    # .copy() is mandatory — lru_cache hands out the live DataFrame and an
    # in-place mutation by a caller would poison every future cache hit.
    result = complete_df.tail(n).copy()
    if "_completion_ts" in result.columns:
        result = result.drop(columns=["_completion_ts"])
    return result
