"""Telegram command receiver: long-polls ``getUpdates`` and dispatches.

Currently dispatches ``/status`` (handled in the runner via a closure).
Future commands (``/pause``, ``/threshold``, etc.) plug into the same
``handlers`` dict — no changes needed here.

Authorisation: the bot only responds to messages from
``settings.telegram_chat_id``. Anything from another chat is silently
dropped (logged at WARNING)."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("alertbot.commands")

# Telegram long-poll timeout. Server holds the connection open up to this
# many seconds waiting for a new update; we time the HTTP client out a
# few seconds longer to absorb the response.
POLL_TIMEOUT_S = 25
HTTP_TIMEOUT_S = POLL_TIMEOUT_S + 10

CommandHandler = Callable[[], Awaitable[str]]

_STOP_EVENT: asyncio.Event | None = None


def request_stop() -> None:
    """Tell ``listen()`` to exit at its next loop turn."""
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()


async def listen(
    bot_token: str,
    chat_id: str,
    handlers: dict[str, CommandHandler],
) -> None:
    """Long-poll Telegram getUpdates and dispatch matching commands.

    handlers maps command text (e.g. ``"/status"``) to async handlers
    that return the reply text. Stop the loop by calling
    :func:`request_stop` from the main coroutine."""
    global _STOP_EVENT
    if not bot_token or not chat_id:
        log.warning("Telegram not configured — command listener idle")
        return

    _STOP_EVENT = asyncio.Event()
    api = f"https://api.telegram.org/bot{bot_token}"
    offset = 0
    log.info(
        "Telegram command listener started (commands: %s)",
        ", ".join(sorted(handlers)),
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
        while not _STOP_EVENT.is_set():
            try:
                r = await client.get(
                    f"{api}/getUpdates",
                    params={"offset": offset, "timeout": POLL_TIMEOUT_S},
                )
            except httpx.TimeoutException:
                # Long-poll timeout with no updates is normal.
                continue
            except httpx.RequestError as e:
                log.warning("getUpdates network error: %s; backing off 5s", e)
                await asyncio.sleep(5)
                continue

            if r.status_code != 200:
                log.warning(
                    "getUpdates HTTP %d (%s); backing off 5s",
                    r.status_code, r.text[:120],
                )
                await asyncio.sleep(5)
                continue

            try:
                data = r.json()
            except ValueError:
                log.warning("getUpdates returned non-JSON; backing off 5s")
                await asyncio.sleep(5)
                continue

            if not data.get("ok"):
                log.warning("getUpdates ok=false: %s", data.get("description"))
                await asyncio.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue

                sender_chat = str(msg.get("chat", {}).get("id", ""))
                if sender_chat != chat_id:
                    log.warning(
                        "Ignoring message from unauthorized chat: %s",
                        sender_chat,
                    )
                    continue

                text = (msg.get("text") or "").strip()
                handler = handlers.get(text)
                if handler is None:
                    continue

                log.info("Dispatching command %s", text)
                try:
                    reply = await handler()
                except Exception:
                    log.exception("Handler for %s raised", text)
                    reply = f"⚠️ {text} handler failed; check logs"
                await _send(client, api, chat_id, reply)

    log.info("Telegram command listener stopped")


async def _send(
    client: httpx.AsyncClient, api: str, chat_id: str, text: str,
) -> None:
    try:
        r = await client.post(
            f"{api}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )
        if r.status_code != 200:
            log.warning(
                "Reply send failed: HTTP %d %s", r.status_code, r.text[:120],
            )
    except httpx.RequestError as e:
        log.error("Reply send network error: %s", e)
