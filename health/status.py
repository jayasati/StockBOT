"""Format the /status Telegram reply.

Reads the latest result per check from ``health_log`` and annotates with
the live ``FailureTracker`` state (consecutive-failure count) when it
is provided."""
from __future__ import annotations

from .alerter import FailureTracker
from .storage import last_failure_ts, latest_per_check


def format_status(tracker: FailureTracker | None = None) -> str:
    """Build the /status reply body."""
    rows = latest_per_check()
    if not rows:
        return "📊 <b>Health</b>\n(no checks yet)"

    lines = ["📊 <b>Health status</b>"]
    for name in sorted(rows):
        ok, detail, _ts = rows[name]
        emoji = "✅" if ok else "❌"
        line = f"{emoji} <b>{name}</b>"
        if not ok:
            line += f" — {detail}"
            last_fail = last_failure_ts(name)
            if last_fail is not None:
                line += f" (since {last_fail.strftime('%H:%M:%S')})"
            if tracker is not None:
                count = tracker.consecutive_failures(name)
                if count > 0:
                    line += f" [{count}× in a row]"
        lines.append(line)
    return "\n".join(lines)
