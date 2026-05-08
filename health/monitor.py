"""Health-monitor loop + ad-hoc CLI.

The loop runs as a background async task inside ``bot.runner.main``.
The CLI (``python -m health --once``) runs the lightweight subset once
and exits 0/1 — useful for cron / one-shot diagnostics."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import checks
from .alerter import FailureTracker
from .checks import CheckResult
from .storage import DB_PATH, log_check_result

log = logging.getLogger("alertbot.health.monitor")

IST = ZoneInfo("Asia/Kolkata")

INTERVAL_MARKET_S = 60
INTERVAL_OFFHOURS_S = 300

# Spec: monitor full check set during 09:00-15:45 IST (slightly wider than
# the trading session 09:15-15:30 so we catch issues at session open + close).
MONITOR_OPEN = (9, 0)
MONITOR_CLOSE = (15, 45)


def _in_monitor_window(ts: datetime | None = None) -> bool:
    if ts is None:
        ts = datetime.now(IST)
    if ts.weekday() >= 5:
        return False
    hm = (ts.hour, ts.minute)
    return MONITOR_OPEN <= hm <= MONITOR_CLOSE


async def _run_full_checks(
    *, watchlist: list[str], telegram_bot_token: str, db_path: Path,
) -> dict[str, CheckResult]:
    results: dict[str, CheckResult] = {}
    # Sync checks first.
    results["disk_space"] = checks.disk_space(".")
    results["db_writable"] = checks.db_writable(db_path)
    results["fyers_token_valid"] = checks.fyers_token_valid()
    results["fyers_websocket_connected"] = checks.fyers_websocket_connected()
    results["bars_fresh"] = checks.bars_fresh(db_path, watchlist)
    # Async checks in parallel.
    nse_task = asyncio.create_task(checks.nse_api_responsive())
    tg_task = asyncio.create_task(checks.telegram_reachable(telegram_bot_token))
    results["nse_api_responsive"] = await nse_task
    results["telegram_reachable"] = await tg_task
    return results


async def _run_lightweight_checks(
    *, telegram_bot_token: str, db_path: Path,
) -> dict[str, CheckResult]:
    results: dict[str, CheckResult] = {}
    results["disk_space"] = checks.disk_space(".")
    results["db_writable"] = checks.db_writable(db_path)
    results["fyers_token_valid"] = checks.fyers_token_valid()
    results["telegram_reachable"] = await checks.telegram_reachable(telegram_bot_token)
    return results


async def run_once(
    *,
    watchlist: list[str] | None = None,
    telegram_bot_token: str = "",
    db_path: Path = DB_PATH,
    lightweight: bool | None = None,
) -> dict[str, CheckResult]:
    """Execute one tick of checks.

    ``lightweight=None`` auto-detects from the current time:
        in the 09:00-15:45 IST monitor window → full set
        outside it → lightweight subset.
    """
    if lightweight is None:
        lightweight = not _in_monitor_window()
    if lightweight:
        return await _run_lightweight_checks(
            telegram_bot_token=telegram_bot_token, db_path=db_path,
        )
    return await _run_full_checks(
        watchlist=watchlist or [],
        telegram_bot_token=telegram_bot_token,
        db_path=db_path,
    )


class MonitorLoop:
    """Background async task: runs checks on a cadence, persists results,
    drives the FailureTracker."""

    def __init__(
        self,
        send_alert,
        watchlist: list[str],
        telegram_bot_token: str,
        db_path: Path = DB_PATH,
    ):
        self._send_alert = send_alert
        self._watchlist = watchlist
        self._token = telegram_bot_token
        self._db_path = db_path
        self.tracker = FailureTracker(send_alert)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info(
            "Health monitor started (market interval %ds, off-hours %ds)",
            INTERVAL_MARKET_S, INTERVAL_OFFHOURS_S,
        )
        while not self._stop.is_set():
            in_market = _in_monitor_window()
            try:
                results = await run_once(
                    watchlist=self._watchlist,
                    telegram_bot_token=self._token,
                    db_path=self._db_path,
                    lightweight=not in_market,
                )
            except Exception:
                log.exception("run_once raised; skipping this tick")
                await self._sleep(INTERVAL_MARKET_S if in_market else INTERVAL_OFFHOURS_S)
                continue

            for name, r in results.items():
                try:
                    log_check_result(name, r, self._db_path)
                except Exception:
                    log.exception("Failed to persist %s result", name)

            try:
                await self.tracker.record(results)
            except Exception:
                log.exception("FailureTracker.record raised")

            await self._sleep(INTERVAL_MARKET_S if in_market else INTERVAL_OFFHOURS_S)

        log.info("Health monitor stopped")

    async def _sleep(self, seconds: int) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


def main() -> int:
    """``python -m health [--once]`` CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bot health monitor (--once for cron-friendly probe)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run all checks once and exit (0 if all pass, 1 otherwise)",
    )
    args = parser.parse_args()

    # Avoid importing bot.config (drags yfinance + fyers SDK at import time).
    # Read what we need directly from the env.
    from dotenv import load_dotenv
    load_dotenv()

    from bot.logging import setup_logging
    setup_logging()

    if not args.once:
        print("usage: python -m health --once")
        return 1

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    # CLI runs without a live websocket / scanner context. Force the
    # lightweight subset so we don't always-fail websocket / bars_fresh.
    results = asyncio.run(
        run_once(telegram_bot_token=token, lightweight=True)
    )

    all_ok = all(r.ok for r in results.values())
    for name, r in results.items():
        emoji = "OK " if r.ok else "FAIL"
        print(f"  [{emoji}] {name:<28} {r.detail} ({r.latency_ms}ms)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
