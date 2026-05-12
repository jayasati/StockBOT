"""Telegram notifier + alert message formatting."""
from __future__ import annotations

import logging

import httpx

from .scoring import StockSignals

log = logging.getLogger("alertbot.notifier")


class Telegram:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, text: str) -> None:
        if not self.bot_token or not self.chat_id:
            log.warning("Telegram not configured, would send: %s", text)
            return
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.post(
                    f"{self.api}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                if r.status_code != 200:
                    # Telegram returns useful diagnostics in the response body
                    try:
                        err = r.json()
                        desc = err.get("description", r.text)
                    except Exception:
                        desc = r.text
                    log.error(
                        "Telegram API error %d: %s", r.status_code, desc
                    )
                    if r.status_code == 400 and "chat not found" in desc.lower():
                        log.error(
                            "FIX: Open Telegram, search for your bot by its "
                            "username, and tap 'Start' to initiate the chat. "
                            "Bots cannot message users who haven't messaged "
                            "them first."
                        )
                    elif r.status_code == 401:
                        log.error(
                            "FIX: Your TELEGRAM_BOT_TOKEN is invalid. "
                            "Generate a new one from @BotFather with /token."
                        )
                    return
            except httpx.RequestError as e:
                log.error("Telegram network error: %s", e)
            except Exception as e:
                log.error("Telegram send failed: %s", e)


def format_alert(s: StockSignals) -> str:
    sym = s.symbol.replace(".NS", "")
    badge = "🚀" if s.score >= 80 else "📈" if s.score >= 70 else "👀"
    lines = [
        f"{badge} <b>{sym}</b>  ₹{s.price:,.2f}",
        f"<b>Score: {s.score}/100</b>",
        f"  • {' · '.join(s.reasons) if s.reasons else 'no signals'}",
    ]
    if s.filing_title:
        lines.append(f"  📰 {s.filing_title}")
    # Phase-5: show the SL/TP levels the paper-tracker is using so the
    # alert matches the trade row. Only render if scanner attached
    # them (back-compat for any pre-Phase-5 caller).
    if s.sl is not None and s.tp1 is not None:
        tp_line = f"🎯 SL ₹{s.sl:,.2f} · TP ₹{s.tp1:,.2f}"
        if s.tp2 is not None:
            tp_line += f" · TP2 ₹{s.tp2:,.2f}"
        lines.append(tp_line)
    lines.append(
        f"<a href='https://www.tradingview.com/symbols/NSE-{sym}/'>Chart</a>"
        f" · <a href='https://groww.in/stocks/{sym.lower()}'>Groww</a>"
    )
    return "\n".join(lines)
