"""Shared log-path helper for the Fyers SDK.

The SDK accepts a ``log_path`` argument and writes ``fyersApi.log``,
``fyersDataSocket.log``, ``fyersRequests.log`` into it. Pointing every
SDK construction at this directory keeps those files out of the repo
root."""
from __future__ import annotations

from pathlib import Path

LOG_DIR = Path("logs")


def fyers_log_path() -> str:
    """Return the directory the SDK should write its log files into.

    Ensures the directory exists; returns its string path so the SDK
    can pass it straight to ``open()``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return str(LOG_DIR)
