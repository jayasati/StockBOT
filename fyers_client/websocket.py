"""Thin SDK-WebSocket wrapper used by the CLI verifiers.

Production code (bot/market_data.py via data/realtime_feed.py) uses the
``LiveFeed`` class instead — it adds reconnect, re-auth, and heartbeat.
"""
from __future__ import annotations

import logging
from typing import Callable

from .auth import authenticate
from .creds import load_creds

log = logging.getLogger("alertbot.fyers")

TickHandler = Callable[[dict], None]


def start_data_socket(
    symbols: list[str],
    on_tick: TickHandler,
    *,
    data_type: str = "SymbolUpdate",
    reconnect_in_sdk: bool = False,
):
    """Open a Fyers WebSocket, subscribe to symbols, call on_tick per message."""
    creds = load_creds()
    token = authenticate()
    from fyers_apiv3.FyersWebsocket import data_ws

    state: dict = {"ws": None}

    def _on_message(msg):
        try:
            on_tick(msg)
        except Exception as e:
            log.exception("on_tick handler failed: %s", e)

    def _on_error(err):
        log.error("WS error: %s", err)

    def _on_close(code):
        log.info("WS closed (code=%s)", code)

    def _on_connect():
        ws = state["ws"]
        log.info("WS connected; subscribing to %d symbol(s)", len(symbols))
        if ws is not None:
            ws.subscribe(symbols=symbols, data_type=data_type)

    state["ws"] = data_ws.FyersDataSocket(
        access_token=f"{creds.app_id}:{token}",
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=reconnect_in_sdk,
        on_connect=_on_connect,
        on_close=_on_close,
        on_error=_on_error,
        on_message=_on_message,
    )
    state["ws"].connect()
    return state["ws"]
