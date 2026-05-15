"""Fyers broker integration — manual auth + WebSocket data feed.

Auth is one-time per day: ``python -m fyers_client auth`` prints a Fyers
OAuth URL, you log in in a browser, paste the redirect URL back; the SDK
exchanges the auth_code for an access_token cached in .fyers_token.json
until 06:00 IST the next day. See :mod:`fyers_client.auth` for the full
walkthrough.

The bot reads the cached token at startup. As long as you do the manual
paste once each morning before 09:15 IST, the bot runs unattended for
the trading day.

This module re-exports the public surface that bot/market_data.py and
data/realtime_feed.py reach for via ``fyers_client.X``.
"""
from .auth import authenticate
from .bars import (
    BAR_INTERVAL_SECONDS,
    Bar,
    MAX_BARS_PER_SYMBOL,
    TICK_STORE,
    TickStore,
)
from .creds import FyersCreds, load_creds
from .history import FyersHistoryError, fetch_history
from .livefeed import LiveFeed
from .record import record_ticks
from .symbols import to_fyers, to_yf
from .token_cache import TOKEN_CACHE
from .websocket import TickHandler, start_data_socket

__all__ = [
    "BAR_INTERVAL_SECONDS",
    "Bar",
    "FyersCreds",
    "FyersHistoryError",
    "LiveFeed",
    "MAX_BARS_PER_SYMBOL",
    "TICK_STORE",
    "TOKEN_CACHE",
    "TickHandler",
    "TickStore",
    "authenticate",
    "fetch_history",
    "load_creds",
    "record_ticks",
    "start_data_socket",
    "to_fyers",
    "to_yf",
]
