"""Main loop: market-hours-aware scheduler.

Two paths feed alerts:

* **Event-driven** — ``BarAggregator.on_bar_complete`` is invoked from the
  Fyers WebSocket thread the moment a 5-min bar closes. We bounce the
  symbol onto an ``asyncio.Queue`` via ``call_soon_threadsafe`` and a
  consumer task scores it within ~seconds of the bar's close tick.
* **Periodic** — ``scanner.scan_once`` runs every ``SCAN_INTERVAL_SECONDS``.
  Refreshes filings / ASM / daily-cache, and serves as a safety net for
  symbols whose bar events were missed (e.g. during a websocket reconnect).

Background tasks running concurrently with the scan loop:
  * ``health.MonitorLoop``  — periodic health checks + alerter
  * ``bot.commands.listen`` — Telegram getUpdates dispatcher (/status)"""
from __future__ import annotations

import asyncio
import logging

import fyers_client
from data import filings, realtime_feed

from health import MonitorLoop, format_status

from . import commands, market_data
from .config import settings
from .db import init_db
from .instance_lock import single_instance
from .logging import setup_logging
from .notifier import Telegram
from .scanner import scan_once, scan_symbol
from .schedule import is_market_open, seconds_until_market_open
from .watchlist import WATCHLIST

log = logging.getLogger("alertbot.runner")


async def _bar_event_consumer(
    telegram: Telegram, queue: asyncio.Queue[str]
) -> None:
    """Drain bar-complete events; score the symbol; dispatch if it qualifies.

    Runs forever until cancelled. Failures on one symbol must not stop the
    consumer — log and continue."""
    while True:
        fy_symbol = await queue.get()
        try:
            yf_symbol = fyers_client.to_yf(fy_symbol)
            fundamentals = filings.recent_high_priority(60)
            await scan_symbol(telegram, yf_symbol, fundamentals)
        except Exception:
            log.exception("Event-driven scan failed for %s", fy_symbol)


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

    # Event-driven path: the Fyers WebSocket thread invokes the aggregator's
    # on_bar_complete callback from off-loop. We can't await Telegram from
    # there, so we bounce the symbol into an asyncio.Queue and drain it
    # on-loop. call_soon_threadsafe is the canonical thread→loop hop.
    bar_event_queue: asyncio.Queue[str] = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    def _on_bar_complete(bar) -> None:
        try:
            main_loop.call_soon_threadsafe(
                bar_event_queue.put_nowait, bar.symbol
            )
        except RuntimeError:
            # Loop already closed during shutdown — fine to drop.
            pass

    realtime_feed.get_aggregator().on_bar_complete = _on_bar_complete

    consumer_task = asyncio.create_task(
        _bar_event_consumer(telegram, bar_event_queue),
        name="bar-event-consumer",
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
        consumer_task.cancel()
        # Give all three a moment to wind down cleanly.
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    monitor_task, commands_task, consumer_task,
                    return_exceptions=True,
                ),
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
