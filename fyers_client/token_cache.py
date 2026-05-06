"""On-disk cache for the Fyers access token (rolls over at 06:00 IST daily)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("alertbot.fyers")

IST = ZoneInfo("Asia/Kolkata")
TOKEN_CACHE = Path(".fyers_token.json")


def _next_token_expiry() -> datetime:
    """Fyers tokens roll over near the start of the trading day. Next 06:00 IST minus a 5-min safety margin."""
    now = datetime.now(IST)
    expiry = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= expiry:
        expiry = expiry + timedelta(days=1)
    return expiry - timedelta(minutes=5)


def _load_cached_token() -> str | None:
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        expiry = datetime.fromisoformat(data["expiry"])
        if datetime.now(IST) < expiry:
            return data["access_token"]
        log.info("Cached Fyers token expired at %s", expiry.isoformat())
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        log.warning("Token cache unreadable (%s); re-authenticating", e)
    return None


def _save_token(access_token: str, expiry: datetime) -> None:
    TOKEN_CACHE.write_text(
        json.dumps(
            {"access_token": access_token, "expiry": expiry.isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        os.chmod(TOKEN_CACHE, 0o600)
    except OSError:
        pass  # Windows — ACLs already inherit from the user dir
