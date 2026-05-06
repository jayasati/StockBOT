"""Stock alert bot package.

Re-exports the public surface that older callers (backtest.py, swing_alert.py,
swing_backtest.py) reach for via ``bot.X``. The implementation is split across
config / watchlist / storage / indicators / scoring / notifier / schedule /
market_data / scanner / runner; this module is the back-compat seam."""
from .config import DB_PATH, IST, Settings, settings
from .db import init_db
from .indicators import (
    compute_extension,
    compute_rsi,
    compute_session_vwap,
    compute_volume_ratio,
    detect_breakout,
    detect_pullback,
)
from .market_data import (
    COLD_START_BAR_THRESHOLD,
    YF_CHUNK_SIZE,
    ensure_live_feed,
    fetch_daily_batch,
    fetch_intraday,
    refresh_asm_gsm_if_stale,
    refresh_daily_cache_if_stale,
)
from .notifier import Telegram, format_alert
from .scanner import scan_once
from .schedule import is_market_open, seconds_until_market_open
from .scoring import StockSignals, score_stock
from .storage import record_alert
from .watchlist import WATCHLIST, WATCHLIST_CSV

__all__ = [
    "COLD_START_BAR_THRESHOLD",
    "DB_PATH",
    "IST",
    "Settings",
    "StockSignals",
    "Telegram",
    "WATCHLIST",
    "WATCHLIST_CSV",
    "YF_CHUNK_SIZE",
    "compute_extension",
    "compute_rsi",
    "compute_session_vwap",
    "compute_volume_ratio",
    "detect_breakout",
    "detect_pullback",
    "ensure_live_feed",
    "fetch_daily_batch",
    "fetch_intraday",
    "format_alert",
    "init_db",
    "is_market_open",
    "record_alert",
    "refresh_asm_gsm_if_stale",
    "refresh_daily_cache_if_stale",
    "scan_once",
    "score_stock",
    "seconds_until_market_open",
    "settings",
]
