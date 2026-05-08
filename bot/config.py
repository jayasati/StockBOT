"""Configuration: Settings dataclass + env loading + shared constants.

Logging configuration lives in :mod:`bot.logging` (rotating file handler,
per-module names). Importing this module no longer touches logging — call
``bot.logging.setup_logging()`` from your entry point."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

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
