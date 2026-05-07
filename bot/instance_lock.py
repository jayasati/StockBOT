"""Single-instance PID-file lock for the bot.

Prevents two ``python -m bot`` processes from scanning + alerting against
the same alerts.db at the same time. The bot writes its PID to ``.bot.pid``
on start and removes it on clean shutdown. On crash, the PID file is left
behind — the next start detects whether the recorded PID is still alive
(via ctypes on Windows, ``os.kill(pid, 0)`` on POSIX) and either reclaims
the stale lock or refuses to start.

Cross-platform; no extra dependencies. If detection fails for any reason
(e.g. permission denied), we err on the side of "lock is held"."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("alertbot.lock")

PID_FILE = Path(".bot.pid")


def _is_pid_alive(pid: int) -> bool:
    """Best-effort liveness check for ``pid``. True if process is running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True  # conservative on API failure
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    # POSIX: signal 0 probes without delivering anything.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def acquire() -> None:
    """Write the current PID to ``.bot.pid``. Raises ``SystemExit`` if another
    bot process is already holding the lock."""
    existing = _read_pid()
    if existing is not None and _is_pid_alive(existing):
        raise SystemExit(
            f"Another bot instance is already running (PID {existing}). "
            f"Stop it first, or delete {PID_FILE} if the lock is stale."
        )
    if existing is not None:
        log.info("Reclaiming stale lock from PID %d", existing)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def release() -> None:
    """Remove the PID file if it points to us. No-op otherwise."""
    pid = _read_pid()
    if pid == os.getpid():
        try:
            PID_FILE.unlink()
        except OSError as e:
            log.warning("Failed to remove %s: %s", PID_FILE, e)


@contextmanager
def single_instance():
    """Context manager wrapping ``acquire()`` + ``release()``."""
    acquire()
    try:
        yield
    finally:
        release()
