"""The seven health checks.

Each function returns a :class:`CheckResult`. Sync where possible; async
where the check must hit a network endpoint. All checks measure their
own latency in milliseconds via ``time.monotonic()``.

Some checks need runtime state (websocket instance, watchlist symbols).
Where the state is a singleton we read it directly (``realtime_feed``);
otherwise (telegram token, watchlist) the caller passes it in."""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

IST = ZoneInfo("Asia/Kolkata")

NSE_HOME = "https://www.nseindia.com/"
NSE_MARKET_STATUS_URL = "https://www.nseindia.com/api/marketStatus"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str
    latency_ms: int


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ---------------------------------------------------------------------------
# Filesystem & DB
# ---------------------------------------------------------------------------

def disk_space(path: Path | str = ".", min_gb: float = 1.0) -> CheckResult:
    """At least ``min_gb`` GB free on the drive holding ``path``."""
    start = time.monotonic()
    try:
        usage = shutil.disk_usage(str(path))
    except OSError as e:
        return CheckResult(False, f"disk_usage failed: {e}", _ms(start))
    free_gb = usage.free / (1024 ** 3)
    return CheckResult(
        ok=free_gb >= min_gb,
        detail=f"{free_gb:.1f} GB free (need >= {min_gb:.1f})",
        latency_ms=_ms(start),
    )


def db_writable(db_path: Path) -> CheckResult:
    """INSERT then DELETE a sentinel row in ``health_log``. Roundtrip ok."""
    start = time.monotonic()
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            conn.execute(
                "INSERT INTO health_log (ts, check_name, ok, latency_ms, detail) "
                "VALUES (?, '_canary', 1, 0, 'sentinel')",
                (datetime.now().isoformat(),),
            )
            conn.execute("DELETE FROM health_log WHERE check_name = '_canary'")
    except sqlite3.Error as e:
        return CheckResult(False, f"sqlite error: {e}", _ms(start))
    return CheckResult(True, "INSERT+DELETE roundtrip ok", _ms(start))


# ---------------------------------------------------------------------------
# Fyers
# ---------------------------------------------------------------------------

def fyers_token_valid() -> CheckResult:
    """Cached Fyers token exists and its expiry is in the future."""
    start = time.monotonic()
    from fyers_client.token_cache import TOKEN_CACHE
    if not TOKEN_CACHE.exists():
        return CheckResult(False, "no token cache file", _ms(start))
    try:
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        expiry = datetime.fromisoformat(data["expiry"])
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return CheckResult(False, f"token cache unreadable: {e}", _ms(start))
    now = datetime.now(IST)
    if now >= expiry:
        return CheckResult(False, f"expired at {expiry.isoformat()}", _ms(start))
    remaining_h = (expiry - now).total_seconds() / 3600
    return CheckResult(True, f"valid for {remaining_h:.1f}h", _ms(start))


def fyers_websocket_connected() -> CheckResult:
    """The live tick feed websocket is currently up."""
    start = time.monotonic()
    from data.realtime_feed import is_live_feed_connected
    connected = is_live_feed_connected()
    return CheckResult(
        ok=connected,
        detail="websocket connected" if connected else "websocket down",
        latency_ms=_ms(start),
    )


def bars_fresh(
    db_path: Path, symbols: list[str], max_age_s: int = 360
) -> CheckResult:
    """At least one tracked symbol has a bar in ``bars_5m`` newer than
    ``max_age_s`` seconds ago. (Spec: 6 minutes during market hours.)"""
    start = time.monotonic()
    if not symbols:
        return CheckResult(False, "no symbols passed", _ms(start))
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            placeholders = ",".join("?" * len(symbols))
            row = conn.execute(
                f"SELECT MAX(ts_open) FROM bars_5m WHERE symbol IN ({placeholders})",
                symbols,
            ).fetchone()
    except sqlite3.Error as e:
        return CheckResult(False, f"sqlite error: {e}", _ms(start))
    max_ts_str = row[0] if row else None
    if max_ts_str is None:
        return CheckResult(False, "no bars in DB for any tracked symbol", _ms(start))
    try:
        max_ts = datetime.fromisoformat(max_ts_str)
    except ValueError as e:
        return CheckResult(False, f"unparseable ts {max_ts_str!r}: {e}", _ms(start))
    if max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=IST)
    age_s = (datetime.now(IST) - max_ts).total_seconds()
    if age_s <= max_age_s:
        return CheckResult(
            True, f"latest bar {age_s:.0f}s old", _ms(start),
        )
    return CheckResult(
        False, f"latest bar {age_s:.0f}s old (> {max_age_s}s)", _ms(start),
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

async def nse_api_responsive() -> CheckResult:
    """GET nseindia.com/api/marketStatus with a 5s timeout. Expect HTTP 200.

    NSE rejects requests without a session cookie, so we GET the homepage
    first as a one-shot warmup. Both calls share the same 5s budget."""
    start = time.monotonic()
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": NSE_HOME,
    }
    try:
        async with httpx.AsyncClient(timeout=5, headers=headers) as client:
            try:
                await client.get(NSE_HOME)
            except httpx.RequestError:
                pass  # warmup is best-effort
            r = await client.get(NSE_MARKET_STATUS_URL)
    except httpx.TimeoutException:
        return CheckResult(False, "timeout (>5s)", _ms(start))
    except httpx.RequestError as e:
        return CheckResult(False, f"network error: {e}", _ms(start))
    if r.status_code == 200:
        return CheckResult(True, "HTTP 200", _ms(start))
    return CheckResult(False, f"HTTP {r.status_code}", _ms(start))


async def telegram_reachable(bot_token: str) -> CheckResult:
    """GET /getMe with a 5s timeout. Expect HTTP 200 with ok=true."""
    start = time.monotonic()
    if not bot_token:
        return CheckResult(False, "no telegram bot token configured", _ms(start))
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
    except httpx.TimeoutException:
        return CheckResult(False, "timeout (>5s)", _ms(start))
    except httpx.RequestError as e:
        return CheckResult(False, f"network error: {e}", _ms(start))
    if r.status_code != 200:
        return CheckResult(False, f"HTTP {r.status_code}", _ms(start))
    try:
        data = r.json()
    except ValueError:
        return CheckResult(False, "non-JSON response", _ms(start))
    if data.get("ok"):
        username = data.get("result", {}).get("username", "?")
        return CheckResult(True, f"@{username}", _ms(start))
    return CheckResult(
        False, f"ok=false: {data.get('description', '?')}", _ms(start),
    )
