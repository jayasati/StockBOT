"""Fyers historical-candles fetcher.

Used by :mod:`data.backfill` to catch up ``bars_5m`` when the bot starts
late (e.g. 09:45 IST instead of 09:15) so indicators see a contiguous
series. Same data source as the WebSocket, so values are bit-exact with
what the live aggregator would have written.

Public surface:
  ``fetch_history(symbol, resolution, from_dt, to_dt) -> DataFrame``
      Sync wrapper over :meth:`FyersModel.history`. Returns the
      lowercase tz-aware IST OHLCV frame that
      :func:`data.realtime_feed.seed_bars` accepts.

The SDK's response shape is ``{"s": "ok", "candles":
[[epoch_seconds, open, high, low, close, volume], ...]}``; non-ok
responses raise :class:`FyersHistoryError` so callers can decide
whether to skip / fall back.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .creds import load_creds
from .logging import fyers_log_path
from .token_cache import _load_cached_token

log = logging.getLogger("alertbot.fyers.history")

IST = ZoneInfo("Asia/Kolkata")


class FyersHistoryError(RuntimeError):
    """Raised when Fyers returns a non-ok response or the SDK isn't usable.

    ``code`` carries the Fyers response code when available so callers can
    distinguish retryable failures (429 rate limit) from terminal ones
    (e.g. -300 invalid symbol)."""

    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message)
        self.code = code


def _build_client():
    """Lazy-import the SDK and use the cached token. Returns ``None``
    when the token isn't available — callers fall back gracefully."""
    token = _load_cached_token()
    if not token:
        log.info("Fyers history: no cached token; skipping fetch")
        return None
    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        log.warning("Fyers SDK not installed; history unavailable")
        return None
    creds = load_creds()
    return fyersModel.FyersModel(
        client_id=creds.app_id, token=token, log_path=fyers_log_path()
    )


def _to_date_str(d: date | datetime | pd.Timestamp) -> str:
    """Fyers ``date_format=1`` expects ``YYYY-MM-DD`` strings."""
    if isinstance(d, pd.Timestamp):
        d = d.date()
    elif isinstance(d, datetime):
        d = d.date()
    return d.isoformat()


def _candles_to_df(candles: list[list]) -> pd.DataFrame:
    """Fyers candles → lowercase tz-aware IST DataFrame.

    Each candle is ``[epoch_seconds, open, high, low, close, volume]``.
    Empty / malformed inputs yield an empty DataFrame with the correct
    columns and index shape so downstream code (``seed_bars``) handles
    the no-op path without special-casing.
    """
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz=IST, name="ts"),
        )
    rows = []
    for c in candles:
        if not isinstance(c, (list, tuple)) or len(c) < 6:
            continue
        rows.append(c[:6])
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz=IST, name="ts"),
        )
    df = pd.DataFrame(
        rows, columns=["ts_epoch", "open", "high", "low", "close", "volume"],
    )
    idx = pd.to_datetime(df["ts_epoch"], unit="s", utc=True).dt.tz_convert(IST)
    df = df.drop(columns=["ts_epoch"])
    df.index = pd.DatetimeIndex(idx, name="ts")
    return df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })


def fetch_history(
    symbol: str,
    resolution: str,
    from_dt: date | datetime | pd.Timestamp,
    to_dt: date | datetime | pd.Timestamp,
    *,
    client=None,
) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Fyers.

    Args:
      symbol: Fyers symbol form, e.g. ``"NSE:RELIANCE-EQ"``.
      resolution: Fyers resolution code — ``"1"`` (1-min), ``"5"``
        (5-min), ``"D"`` (daily). The bot uses ``"5"`` for backfill.
      from_dt, to_dt: Inclusive date range (Fyers' history endpoint
        operates at day granularity even for intraday resolutions —
        the SDK returns all candles within those days, and the caller
        is responsible for trimming).
      client: Optional pre-built ``FyersModel`` for tests. Production
        callers pass ``None`` and the cached daily token is used.

    Returns:
      Lowercase tz-aware IST DataFrame with columns ``open / high /
      low / close / volume`` and a ``DatetimeIndex`` named ``ts``.
      Empty (but well-shaped) when Fyers has no data for the window.

    Raises:
      FyersHistoryError: when no token is cached, the SDK isn't
        installed, or Fyers returns a non-ok response.
    """
    if client is None:
        client = _build_client()
    if client is None:
        raise FyersHistoryError("Fyers history client unavailable (no token / SDK)")

    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": _to_date_str(from_dt),
        "range_to": _to_date_str(to_dt),
        "cont_flag": "1",
    }
    try:
        response = client.history(data=payload)
    except Exception as e:
        raise FyersHistoryError(
            f"Fyers history call raised for {symbol}: {e}"
        ) from e

    if not isinstance(response, dict) or response.get("s") != "ok":
        code_raw = response.get("code") if isinstance(response, dict) else None
        try:
            code = int(code_raw) if code_raw is not None else None
        except (TypeError, ValueError):
            code = None
        raise FyersHistoryError(
            f"Fyers history non-ok for {symbol}: "
            f"{code_raw if isinstance(response, dict) else response} "
            f"{response.get('message') if isinstance(response, dict) else ''}",
            code=code,
        )

    return _candles_to_df(response.get("candles") or [])
