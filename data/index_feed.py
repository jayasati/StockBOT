"""Index + VIX feeder used by Phase-6 filters.

Three values consumed by the filter chain:
  ``get_intraday_pct(NIFTY)``      → today's open-to-LTP for NIFTY 50
  ``get_intraday_pct(BANK_NIFTY)`` → today's open-to-LTP for BANKNIFTY
  ``get_vix()``                    → India VIX (daily close, slow-moving)

yfinance is the data source (no Fyers subscription needed for indices;
all three symbols are public). Both APIs are cached — TTL 60s for
intraday (enough to coalesce per-scan reads), 6h for VIX (a daily
print). On fetch failure the functions return ``None`` so callers
fail-open rather than crash a scan."""
from __future__ import annotations

import logging
import time
from typing import Literal

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

log = logging.getLogger("alertbot.index_feed")

IST = ZoneInfo("Asia/Kolkata")

NIFTY = "^NSEI"
BANK_NIFTY = "^NSEBANK"
INDIA_VIX = "^INDIAVIX"

INTRADAY_TTL_SECONDS = 60
"""Per-symbol intraday cache TTL. One scan over 500 symbols takes a
few seconds; a single fetch covers an entire scan window without
hammering yfinance."""

VIX_TTL_SECONDS = 6 * 3600
"""VIX is a daily print — refresh sparingly."""

_intraday_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_vix_cache: tuple[float, float | None] | None = None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Single-ticker yfinance can return a MultiIndex column on some
    versions; collapse. Also tz-localize/convert to IST."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
    return df


def _fetch_intraday(yf_symbol: str) -> pd.DataFrame:
    """Cached 5m fetch for an index. Empty DataFrame on failure."""
    now = time.time()
    cached = _intraday_cache.get(yf_symbol)
    if cached and (now - cached[0]) < INTRADAY_TTL_SECONDS:
        return cached[1]
    try:
        df = yf.download(
            yf_symbol, period="1d", interval="5m",
            progress=False, auto_adjust=False,
        )
        df = _normalize_index(df) if df is not None else pd.DataFrame()
    except Exception:
        log.exception("yfinance intraday fetch failed for %s", yf_symbol)
        df = pd.DataFrame()
    _intraday_cache[yf_symbol] = (now, df)
    return df


def get_intraday_pct(symbol: str = NIFTY) -> float | None:
    """Today's open-to-last percent change for ``symbol``. Returns
    ``None`` when data is unavailable (weekend, pre-market, fetch
    error) so filters can fail-open."""
    df = _fetch_intraday(symbol)
    if df.empty:
        return None
    today = pd.Timestamp.now(tz=IST).date()
    today_df = df[df.index.date == today]
    if today_df.empty:
        return None
    open_price = float(today_df["Open"].iloc[0])
    last_price = float(today_df["Close"].iloc[-1])
    if open_price == 0:
        return None
    return (last_price - open_price) / open_price * 100.0


def get_direction(symbol: str = BANK_NIFTY) -> Literal["UP", "DOWN", "FLAT"]:
    """Classify intraday move. Threshold ±0.3% — anything inside
    that band is FLAT, not a directional signal."""
    pct = get_intraday_pct(symbol)
    if pct is None:
        return "FLAT"
    if pct > 0.3:
        return "UP"
    if pct < -0.3:
        return "DOWN"
    return "FLAT"


def get_vix() -> float | None:
    """Latest India VIX close. Cached for 6h."""
    global _vix_cache
    now = time.time()
    if _vix_cache is not None and (now - _vix_cache[0]) < VIX_TTL_SECONDS:
        return _vix_cache[1]
    try:
        df = yf.download(
            INDIA_VIX, period="5d", interval="1d",
            progress=False, auto_adjust=False,
        )
        df = _normalize_index(df) if df is not None else pd.DataFrame()
        value = float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception:
        log.exception("yfinance VIX fetch failed")
        value = None
    _vix_cache = (now, value)
    return value


def _reset_caches() -> None:
    """Test helper — clear in-memory caches between tests."""
    global _vix_cache
    _intraday_cache.clear()
    _vix_cache = None
