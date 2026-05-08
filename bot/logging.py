"""Logging setup: rotating file handler + per-module logger names.

Single entry point for the bot's logging configuration. Called once from
:func:`bot.runner.run` and once from :func:`health.monitor.main`.
Idempotent — calling it more than once is a no-op."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "alertbot.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Install stream + rotating file handlers on the root logger.

    Wipes any pre-existing handlers (e.g. a stray ``basicConfig`` from a
    library import) so we don't double-log."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    root.addHandler(stream)
    root.addHandler(file_handler)

    # Damp third-party noise. httpx leaks bot tokens at INFO; yfinance is
    # chatty about ratelimits; both should never speak below WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    _CONFIGURED = True
