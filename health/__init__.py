"""Health monitoring + alerting.

Runs as a background async task inside the main bot event loop. Every
60s during market hours (09:00-15:45 IST), executes a battery of checks
(token validity, websocket health, bar freshness, network reachability,
db writability, disk space) and:

- logs each result to the ``health_log`` table in alerts.db
- emits a Telegram alert when a check fails 3 times in a row
- combines simultaneous failures into a single message
- emits a recovery alert when a previously-failed check returns to ok

Off-hours: every 5 minutes, lightweight subset only (no bar freshness,
no websocket).

CLIs:
  ``python -m health --once``  run lightweight checks once, exit 0/1
"""
from .alerter import DEFAULT_THRESHOLD, FailureTracker
from .checks import CheckResult
from .monitor import MonitorLoop, run_once
from .status import format_status
from .storage import last_failure_ts, latest_per_check, log_check_result

__all__ = [
    "CheckResult",
    "DEFAULT_THRESHOLD",
    "FailureTracker",
    "MonitorLoop",
    "format_status",
    "last_failure_ts",
    "latest_per_check",
    "log_check_result",
    "run_once",
]
