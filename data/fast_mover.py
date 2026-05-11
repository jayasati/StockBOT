"""Tick-level fast-move detector — independent of the 5-min bar pipeline.

Watches the raw Fyers tick stream and fires a "fast move" alert whenever a
symbol's price range over a short rolling window exceeds a threshold. Built
to catch intra-bar spikes (e.g. PVRINOX +1.5% in 13 min then -4% in 2 min)
that the composite scorer can only see after the containing 5-min bar
closes — typically 3–5 minutes too late to act on.

Hooked into the WebSocket via :func:`data.realtime_feed.add_tick_observer`
so it runs alongside the bar aggregators, not in place of them.

The detector is purely synchronous: ``on_tick`` is invoked from the
WebSocket thread. To dispatch a Telegram message we need an asyncio path,
so callers pass a sync ``on_alert`` callback which typically does
``loop.call_soon_threadsafe(queue.put_nowait, payload)`` to bridge to the
event loop. The fast-mover itself stays unaware of asyncio.

Cooldown is in-memory (no SQLite); a process restart resets it. That's
acceptable because fast moves are an *opportunistic* signal — re-alerting
the same symbol shortly after a restart is fine, the user just sees an
extra ping."""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

from bot.schedule import in_session
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger("alertbot.fast_mover")

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_WINDOW_S = 180         # 3-min rolling window
DEFAULT_PCT_THRESHOLD = 2.0    # 2% range in the window triggers
DEFAULT_COOLDOWN_S = 300       # 5-min per-symbol cooldown after an alert


@dataclass(frozen=True)
class FastMove:
    """One fast-move event ready to be turned into a Telegram message."""
    symbol: str
    direction: str        # "↑" if last > first within window, else "↓"
    pct: float            # range / low * 100
    high: float
    low: float
    first_price: float    # oldest price still inside the window
    last_price: float     # most recent tick price
    window_s: int         # actual span (≤ configured window)


AlertCallback = Callable[[FastMove], None]


class FastMover:
    """Per-symbol rolling-window range detector.

    Thread-safety: ``on_tick`` is called from the Fyers WebSocket thread.
    Internal state is guarded by a single mutex; the work under the lock
    is O(window-size) per tick (deque trim + min/max) which is negligible
    at typical tick rates."""

    def __init__(
        self,
        *,
        window_s: int = DEFAULT_WINDOW_S,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
        cooldown_s: int = DEFAULT_COOLDOWN_S,
        on_alert: AlertCallback | None = None,
        watchlist: set[str] | None = None,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if pct_threshold <= 0:
            raise ValueError("pct_threshold must be positive")
        self._window_s = window_s
        self._threshold = pct_threshold
        self._cooldown_s = cooldown_s
        self._on_alert = on_alert
        self._watchlist = watchlist  # if set, only these symbols can fire
        self._lock = threading.Lock()
        self._history: dict[str, deque[tuple[float, float]]] = {}
        self._last_alert_ts: dict[str, float] = {}
        self._tick_count = 0
        self._alert_count = 0

    # -- ingestion -------------------------------------------------------

    def on_tick(self, msg: dict) -> None:
        """Process one Fyers ``sf`` (symbol-full) snapshot. Other frame
        types are ignored. Out-of-session ticks are dropped — we don't
        want post-market crossing prints to fire phantom alerts."""
        if msg.get("type") != "sf":
            return
        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        ts_epoch = msg.get("last_traded_time")
        if not symbol or ltp is None or ts_epoch is None:
            return
        if self._watchlist is not None and symbol not in self._watchlist:
            return
        try:
            ltp_f = float(ltp)
            ts_f = float(ts_epoch)
        except (TypeError, ValueError):
            return

        # Drop ticks outside the trading session. The aggregator does the
        # same; we duplicate the check because the fast-mover is wired as
        # an independent observer.
        ts_ist = datetime.fromtimestamp(ts_f, tz=IST)
        if not in_session(ts_ist, mode="alert"):
            return

        event: FastMove | None = None
        with self._lock:
            self._tick_count += 1
            dq = self._history.setdefault(symbol, deque())
            dq.append((ts_f, ltp_f))

            # Trim old entries.
            cutoff = ts_f - self._window_s
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            if len(dq) < 2:
                return

            high = max(p for _, p in dq)
            low = min(p for _, p in dq)
            if low <= 0:
                return
            pct = (high - low) / low * 100.0
            if pct < self._threshold:
                return

            # Cooldown per-symbol.
            last_alert = self._last_alert_ts.get(symbol, 0.0)
            if ts_f - last_alert < self._cooldown_s:
                return

            first_price = dq[0][1]
            event = FastMove(
                symbol=symbol,
                direction="↑" if ltp_f >= first_price else "↓",
                pct=pct,
                high=high,
                low=low,
                first_price=first_price,
                last_price=ltp_f,
                window_s=int(ts_f - dq[0][0]),
            )
            self._last_alert_ts[symbol] = ts_f
            self._alert_count += 1

        if event is not None and self._on_alert is not None:
            try:
                self._on_alert(event)
            except Exception:
                log.exception("on_alert callback raised for %s", event.symbol)

    # -- introspection ---------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "ticks": self._tick_count,
                "alerts": self._alert_count,
                "tracked_symbols": len(self._history),
                "on_cooldown": len(self._last_alert_ts),
            }


def format_fast_move(event: FastMove) -> str:
    """Telegram-ready HTML for one fast-move alert."""
    return (
        f"⚡ <b>FAST MOVE {event.direction} {event.symbol}</b>\n"
        f"{event.pct:+.2f}% in {event.window_s}s\n"
        f"{event.first_price:.2f} → {event.last_price:.2f} "
        f"(H {event.high:.2f} / L {event.low:.2f})"
    )
