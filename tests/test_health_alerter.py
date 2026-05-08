"""FailureTracker semantics: 3-strike, combined, recovery.

Uses ``asyncio.run`` to drive the async ``record()`` so we don't need
the pytest-asyncio plugin."""
from __future__ import annotations

import asyncio

from health.alerter import FailureTracker
from health.checks import CheckResult


def _ok(detail: str = "ok") -> CheckResult:
    return CheckResult(True, detail, 0)


def _fail(detail: str = "boom") -> CheckResult:
    return CheckResult(False, detail, 0)


class _Collector:
    """Stand-in Telegram sender that records messages instead of sending."""
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __call__(self, text: str) -> None:
        self.sent.append(text)


def _drive(*ticks: dict[str, CheckResult]) -> tuple[FailureTracker, _Collector]:
    """Run a fresh tracker through a sequence of result ticks."""
    sink = _Collector()
    tracker = FailureTracker(sink, threshold=3)

    async def run() -> None:
        for tick in ticks:
            await tracker.record(tick)

    asyncio.run(run())
    return tracker, sink


def test_fires_after_three_consecutive_failures():
    _, sink = _drive(
        {"db_writable": _fail("io")},
        {"db_writable": _fail("io")},
        {"db_writable": _fail("io")},
    )
    assert len(sink.sent) == 1
    assert "db_writable" in sink.sent[0]
    assert "🚨" in sink.sent[0]


def test_no_alert_below_threshold():
    _, sink = _drive(
        {"db_writable": _fail()},
        {"db_writable": _fail()},
    )
    assert sink.sent == []


def test_does_not_re_alert_while_still_failing():
    _, sink = _drive(*[{"db_writable": _fail()} for _ in range(5)])
    assert len(sink.sent) == 1, "still-failing should fire only once"


def test_recovery_emits_recovery_alert():
    sink = _Collector()
    tracker = FailureTracker(sink, threshold=3)

    async def run() -> None:
        for _ in range(3):
            await tracker.record({"db_writable": _fail()})
        sink.sent.clear()
        await tracker.record({"db_writable": _ok()})

    asyncio.run(run())
    assert len(sink.sent) == 1
    assert "✅" in sink.sent[0]
    assert "db_writable" in sink.sent[0]


def test_recovery_only_for_previously_alerted_check():
    # 2 fails — never alerted — then recover. No recovery message.
    _, sink = _drive(
        {"db_writable": _fail()},
        {"db_writable": _fail()},
        {"db_writable": _ok()},
    )
    assert sink.sent == []


def test_combined_alert_for_simultaneous_failures():
    """Three checks all hit the 3-strike line in the SAME tick — one message."""
    names = ["a", "b", "c"]
    sink = _Collector()
    tracker = FailureTracker(sink, threshold=3)

    async def run() -> None:
        for _ in range(2):
            await tracker.record({n: _fail() for n in names})
        # Before the third tick: nothing sent yet.
        assert sink.sent == []
        await tracker.record({n: _fail(f"bad_{n}") for n in names})

    asyncio.run(run())
    assert len(sink.sent) == 1, "all three should combine into one alert"
    msg = sink.sent[0]
    for n in names:
        assert n in msg


def test_consecutive_failures_counter():
    sink = _Collector()
    tracker = FailureTracker(sink, threshold=3)

    async def run() -> None:
        await tracker.record({"x": _fail()})
        assert tracker.consecutive_failures("x") == 1
        await tracker.record({"x": _fail()})
        assert tracker.consecutive_failures("x") == 2
        await tracker.record({"x": _ok()})
        assert tracker.consecutive_failures("x") == 0

    asyncio.run(run())
    assert tracker.consecutive_failures("never_seen") == 0


def test_independent_check_state():
    """Two checks tracked independently — only the one hitting threshold fires."""
    sink = _Collector()
    tracker = FailureTracker(sink, threshold=3)

    async def run() -> None:
        # 'alpha' fails 3 times → fires
        # 'beta' is healthy throughout → never fires
        await tracker.record({"alpha": _fail(), "beta": _ok()})
        await tracker.record({"alpha": _fail(), "beta": _ok()})
        await tracker.record({"alpha": _fail(), "beta": _ok()})

    asyncio.run(run())
    assert len(sink.sent) == 1
    assert "alpha" in sink.sent[0]
    assert "beta" not in sink.sent[0]
    assert tracker.consecutive_failures("beta") == 0
