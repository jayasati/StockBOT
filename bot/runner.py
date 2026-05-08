"""Main loop: market-hours-aware scheduler around scan_once.

Two background tasks run concurrently with the scan loop:
  * ``health.MonitorLoop``  — periodic health checks + alerter
  * ``bot.commands.listen`` — Telegram getUpdates dispatcher (/status)"""
from __future__ import annotations

import asyncio
import logging

import fyers_client

from health import MonitorLoop, format_status

from . import commands, market_data
from .config import settings
from .db import init_db
from .instance_lock import single_instance
from .logging import setup_logging
from .notifier import Telegram
from .scanner import scan_once
from .schedule import is_market_open, seconds_until_market_open
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot.runner")


async def main() -> None:
    init_db()
    market_data.ensure_live_feed()
    await market_data.refresh_asm_gsm_if_stale()
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    await telegram.send(
        f"🤖 <b>Stock alert bot started</b>\n"
        f"Watching: {len(WATCHLIST)} symbols\n"
        f"Threshold: {settings.composite_threshold}/100\n"
        f"Scan interval: {settings.scan_interval_seconds}s\n"
        f"Cooldown: {settings.cooldown_minutes}m"
    )

    log.info("Bot started. Watchlist: %d symbols", len(WATCHLIST))

    # The health monitor needs Fyers-encoded symbols (NSE:X-EQ) to query
    # bars_5m, which is keyed on the Fyers form.
    fy_symbols = [fyers_client.to_fyers(s) for s in WATCHLIST]
    monitor = MonitorLoop(
        send_alert=telegram.send,
        watchlist=fy_symbols,
        telegram_bot_token=settings.telegram_bot_token,
    )

    async def _status_handler() -> str:
        return format_status(monitor.tracker)

    monitor_task = asyncio.create_task(monitor.run(), name="health-monitor")
    commands_task = asyncio.create_task(
        commands.listen(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            handlers={"/status": _status_handler},
        ),
        name="tg-commands",
    )

    try:
        while True:
            try:
                if is_market_open():
                    await scan_once(telegram)
                    await asyncio.sleep(settings.scan_interval_seconds)
                else:
                    wait = min(seconds_until_market_open(), 1800)  # cap at 30 min
                    log.info("Market closed. Sleeping %d seconds.", wait)
                    await asyncio.sleep(wait)
            except KeyboardInterrupt:
                log.info("Shutting down...")
                break
            except Exception as e:
                log.exception("Scan error: %s", e)
                await asyncio.sleep(60)
    finally:
        monitor.stop()
        commands.request_stop()
        # Give both tasks a moment to wind down cleanly.
        try:
            await asyncio.wait_for(
                asyncio.gather(monitor_task, commands_task, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            log.warning("Background tasks did not exit within 5s; cancelling")
            monitor_task.cancel()
            commands_task.cancel()


def run() -> None:
    setup_logging()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        log.error("Copy env.example to .env and fill in your credentials.")
        log.error("See README.md for setup instructions.")
        raise SystemExit(1)
    with single_instance():
        asyncio.run(main())
