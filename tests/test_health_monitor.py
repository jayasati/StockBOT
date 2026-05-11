"""MonitorLoop behaviour: startup warmup masking.

Other monitor semantics (interval choice, check-set selection) are exercised
in test_health_checks.py + test_health_alerter.py; this file targets the
piece those don't reach: time-based checks must not alert during the
process-startup grace window."""
from __future__ import annotations

import time as _time

import pytest

from health.checks import CheckResult
from health.monitor import MonitorLoop


async def _noop_send(_text: str) -> None:
    return None


def _make_loop(warmup_s: int = 360) -> MonitorLoop:
    return MonitorLoop(
        send_alert=_noop_send,
        watchlist=["NSE:X-EQ"],
        telegram_bot_token="",
        warmup_s=warmup_s,
    )


def test_warmup_masks_bars_fresh_failure_during_grace_window():
    """Inside the warmup window, a ``bars_fresh`` failure is rewritten to a
    benign pass so the FailureTracker never accrues strikes from it."""
    loop = _make_loop(warmup_s=360)
    # Simulate ``run()`` having just started.
    loop._warmup_until = _time.monotonic() + 360

    results = {
        "disk_space": CheckResult(True, "ok", 0),
        "bars_fresh": CheckResult(False, "no bars in DB for any tracked symbol", 0),
        "ticks_fresh": CheckResult(False, "no ticks received yet", 0),
    }
    loop._apply_warmup(results)

    assert results["bars_fresh"].ok
    assert results["bars_fresh"].detail == "warming up"
    assert results["ticks_fresh"].ok
    assert results["ticks_fresh"].detail == "warming up"
    # Unrelated checks are not touched.
    assert results["disk_space"].ok
    assert results["disk_space"].detail == "ok"


def test_warmup_does_not_mask_unrelated_failures():
    """Only the configured time-based checks are masked; everything else
    (disk_space, db_writable, fyers_token_valid, …) fails normally during
    warmup so a real boot-time problem still alerts."""
    loop = _make_loop(warmup_s=360)
    loop._warmup_until = _time.monotonic() + 360

    results = {
        "disk_space": CheckResult(False, "0.1 GB free (need >= 1.0)", 0),
        "fyers_token_valid": CheckResult(False, "no token cache file", 0),
        "bars_fresh": CheckResult(False, "no bars in DB for any tracked symbol", 0),
    }
    loop._apply_warmup(results)

    assert not results["disk_space"].ok
    assert not results["fyers_token_valid"].ok
    assert results["bars_fresh"].ok  # masked


def test_warmup_does_not_mask_after_window_elapses():
    """Once ``_warmup_until`` is in the past, time-based failures pass
    through unmodified."""
    loop = _make_loop(warmup_s=360)
    loop._warmup_until = _time.monotonic() - 1  # already expired

    results = {
        "bars_fresh": CheckResult(False, "latest bar 700s old", 0),
        "ticks_fresh": CheckResult(False, "latest tick 120s old", 0),
    }
    loop._apply_warmup(results)

    assert not results["bars_fresh"].ok
    assert not results["ticks_fresh"].ok


def test_warmup_does_not_mask_passing_results():
    """Passing checks are passed through (we mask failures only, not OKs)."""
    loop = _make_loop(warmup_s=360)
    loop._warmup_until = _time.monotonic() + 360

    results = {
        "bars_fresh": CheckResult(True, "latest bar 30s old (NSE:X-EQ)", 0),
    }
    loop._apply_warmup(results)
    assert results["bars_fresh"].ok
    assert "NSE:X-EQ" in results["bars_fresh"].detail  # not rewritten


def test_warmup_inactive_when_loop_has_not_run():
    """If ``run()`` was never called (warmup_until is None), no masking
    happens — failures pass through. Defensive: prevents accidental mask
    in unit-level callers."""
    loop = _make_loop(warmup_s=360)
    assert loop._warmup_until is None

    results = {
        "bars_fresh": CheckResult(False, "stale", 0),
    }
    loop._apply_warmup(results)
    assert not results["bars_fresh"].ok
