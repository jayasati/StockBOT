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

    # Scoring threshold (0-100). Score >= this opens a paper trade
    # (and writes alerts_sent for cooldown). Lower = more paper trades.
    composite_threshold: int = 60

    # Telegram threshold (0-100). Score >= this also fires a Telegram
    # message. Between composite_threshold and telegram_threshold a
    # paper trade is recorded silently — Phase-5b uses this band as
    # the "measurement" zone so we can study which mid-conviction
    # signals actually pay off before deciding whether to promote
    # them to Telegram. See idea/promptchain_noise_reduction.txt.
    telegram_threshold: int = 80

    # Cooldown to prevent same-stock spam
    cooldown_minutes: int = 60

    # Loop interval (seconds). 300 = every 5 minutes.
    scan_interval_seconds: int = 300

    # Cap alerts per scan to prevent flood
    max_alerts_per_scan: int = 15

    # Liquidity floor for the Phase-6 hard filter, expressed as 20-day
    # average daily RUPEE turnover (₹crore). Replaces the original
    # share-count floor, which unfairly killed high-priced names — MRF
    # at ~7,000 shares/day is ₹86cr/day turnover, well inside any
    # reasonable liquidity definition but well below a 500k-share floor.
    liquidity_min_turnover_cr: float = 5.0


settings = Settings(
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    composite_threshold=int(os.getenv("COMPOSITE_THRESHOLD", "60")),
    telegram_threshold=int(os.getenv("TELEGRAM_THRESHOLD", "80")),
    cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "60")),
    scan_interval_seconds=int(os.getenv("SCAN_INTERVAL_SECONDS", "300")),
    liquidity_min_turnover_cr=float(os.getenv("LIQUIDITY_MIN_TURNOVER_CR", "5.0")),
)
