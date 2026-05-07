"""Main loop: market-hours-aware scheduler around scan_once."""
from __future__ import annotations

import asyncio
import logging

from . import market_data
from .config import settings
from .db import init_db
from .instance_lock import single_instance
from .notifier import Telegram
from .scanner import scan_once
from .schedule import is_market_open, seconds_until_market_open
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot")


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


def run() -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        print("Copy .env.example to .env and fill in your credentials.")
        print("See README.md for setup instructions.")
        raise SystemExit(1)
    with single_instance():
        asyncio.run(main())
