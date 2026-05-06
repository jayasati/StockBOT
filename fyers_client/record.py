"""Record raw tick stream to JSONL for offline replay (test fixture capture)."""
from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path

from .livefeed import LiveFeed

log = logging.getLogger("alertbot.fyers")


def record_ticks(
    symbols: list[str],
    output_path: str | Path,
    duration_sec: int,
) -> int:
    """Subscribe to ``symbols`` and dump every incoming tick to JSONL for ``duration_sec`` seconds."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = output_path.open("w", encoding="utf-8")
    counter = {"n": 0}

    def _capture(msg: dict) -> None:
        if msg.get("type") != "sf":
            return
        f.write(json.dumps(msg) + "\n")
        counter["n"] += 1

    feed = LiveFeed(symbols=symbols, on_tick=_capture)
    try:
        feed.start()
        deadline = _time.time() + duration_sec
        log.info("Recording ticks to %s for %ds...", output_path, duration_sec)
        while _time.time() < deadline:
            _time.sleep(1)
    finally:
        feed.stop()
        f.close()
    log.info("Captured %d ticks to %s", counter["n"], output_path)
    return counter["n"]
