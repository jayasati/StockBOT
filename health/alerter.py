"""Failure tracker + Telegram alert dispatch.

Three rules (per the spec):
  1. Three consecutive failures of a single check → alert.
  2. Multiple checks crossing the 3-strike line in the SAME tick → ONE
     combined alert (don't spam).
  3. Recovery from a previously-alerted state → recovery alert."""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .checks import CheckResult

log = logging.getLogger("alertbot.health")

DEFAULT_THRESHOLD = 3

SendAlert = Callable[[str], Awaitable[None]]


class FailureTracker:
    """Per-check consecutive-failure counter + alert dispatcher.

    Pure state machine — does not touch the DB. Inject the Telegram
    sender as ``send_alert`` so tests can swap it for a mock collector."""

    def __init__(self, send_alert: SendAlert, threshold: int = DEFAULT_THRESHOLD):
        self._send = send_alert
        self._threshold = threshold
        self._counts: dict[str, int] = {}
        self._alerted: set[str] = set()

    async def record(self, results: dict[str, CheckResult]) -> None:
        """Update counters from one tick's results; emit Telegram if needed."""
        newly_failed: list[tuple[str, CheckResult]] = []
        newly_recovered: list[str] = []

        for name, r in results.items():
            if r.ok:
                self._counts[name] = 0
                if name in self._alerted:
                    self._alerted.discard(name)
                    newly_recovered.append(name)
            else:
                self._counts[name] = self._counts.get(name, 0) + 1
                if (
                    self._counts[name] >= self._threshold
                    and name not in self._alerted
                ):
                    self._alerted.add(name)
                    newly_failed.append((name, r))

        if newly_failed:
            try:
                await self._send(_format_failure(newly_failed))
            except Exception:
                log.exception("Failed to send failure alert")
        if newly_recovered:
            try:
                await self._send(_format_recovery(newly_recovered))
            except Exception:
                log.exception("Failed to send recovery alert")

    def is_alerted(self, name: str) -> bool:
        return name in self._alerted

    def consecutive_failures(self, name: str) -> int:
        return self._counts.get(name, 0)


def _format_failure(failures: list[tuple[str, CheckResult]]) -> str:
    if len(failures) == 1:
        name, r = failures[0]
        return f"🚨 <b>Health alert: {name}</b>\n{r.detail}"
    lines = [f"🚨 <b>Health alerts ({len(failures)})</b>"]
    for name, r in failures:
        lines.append(f"• <b>{name}</b>: {r.detail}")
    return "\n".join(lines)


def _format_recovery(names: list[str]) -> str:
    if len(names) == 1:
        return f"✅ <b>Recovered:</b> {names[0]}"
    return f"✅ <b>Recovered ({len(names)}):</b> {', '.join(names)}"
