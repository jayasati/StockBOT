"""Configuration: Settings dataclass + env loading + shared constants."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Suppress httpx INFO-level logs — they leak the bot token in request URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path("alerts.db")


@dataclass
class Settings:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Scoring threshold (0-100). Lower = more alerts.
    composite_threshold: int = 60

    # Cooldown to prevent same-stock spam
    cooldown_minutes: int = 60

    # Loop interval (seconds). 300 = every 5 minutes.
    scan_interval_seconds: int = 300

    # Cap alerts per scan to prevent flood
    max_alerts_per_scan: int = 15


settings = Settings(
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    composite_threshold=int(os.getenv("COMPOSITE_THRESHOLD", "60")),
    cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "60")),
    scan_interval_seconds=int(os.getenv("SCAN_INTERVAL_SECONDS", "300")),
)
