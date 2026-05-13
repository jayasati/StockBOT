"""Production WebSocket wrapper.

Wraps FyersDataSocket with:
  * exponential-backoff reconnect (1, 2, 4, 8, 16, 32, 60, 60... seconds)
  * auto re-auth after N consecutive failures (token rolls over at 06:00 IST)
  * per-symbol heartbeat — logs WARNING if a subscribed symbol has been
    silent for >30s during market hours
"""
from __future__ import annotations

import logging
import threading
import time as _time

from core.session import in_session

from .auth import authenticate
from .creds import FyersCreds, load_creds
from .logging import fyers_log_path
from .token_cache import TOKEN_CACHE
from .websocket import TickHandler

log = logging.getLogger("alertbot.fyers")

_RECONNECT_BACKOFF = (1, 2, 4, 8, 16, 32, 60)
_REAUTH_AFTER_N_FAILURES = 3
_HEARTBEAT_INTERVAL_SEC = 5
_HEARTBEAT_SILENCE_WARN_SEC = 30


class LiveFeed:
    """Live tick stream with reconnect, re-auth, and heartbeat."""

    def __init__(
        self,
        symbols: list[str],
        on_tick: TickHandler,
        access_token: str | None = None,
        *,
        data_type: str = "SymbolUpdate",
    ) -> None:
        self._symbols: list[str] = list(symbols)
        self._on_tick_user = on_tick
        self._data_type = data_type
        self._access_token = access_token
        self._creds: FyersCreds | None = None

        self._lock = threading.Lock()
        self._ws = None
        self._running = False
        self._consecutive_failures = 0
        self._reconnect_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_tick_at: dict[str, float] = {}
        self._silence_logged: set[str] = set()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._running:
            log.warning("LiveFeed already running")
            return
        self._running = True
        self._stop_event.clear()
        self._connect()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="livefeed-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close_connection()
            except Exception:
                pass

    def is_connected(self) -> bool:
        """True if a websocket is currently up and the loop is running.

        Used by the health monitor's ``fyers_websocket_connected`` check.
        Best-effort: a brief reconnect window between ``_on_close`` and
        ``_reconnect_loop`` may report False even though service resumes
        within seconds — that's acceptable noise for a health probe."""
        with self._lock:
            return self._running and self._ws is not None

    def add_symbols(self, symbols: list[str]) -> None:
        new = [s for s in symbols if s not in self._symbols]
        if not new:
            return
        self._symbols.extend(new)
        with self._lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.subscribe(symbols=new, data_type=self._data_type)
                log.info("Extended subscription with %d symbol(s)", len(new))
            except Exception as e:
                log.error("subscribe(%s) failed: %s", new, e)

    # -- connection ------------------------------------------------------

    def _connect(self) -> None:
        from fyers_apiv3.FyersWebsocket import data_ws

        if self._creds is None:
            self._creds = load_creds()
        if self._access_token is None:
            self._access_token = authenticate()

        ws = data_ws.FyersDataSocket(
            access_token=f"{self._creds.app_id}:{self._access_token}",
            log_path=fyers_log_path(),
            litemode=False,
            write_to_file=False,
            reconnect=False,    # we own the policy
            on_connect=self._on_connect,
            on_close=self._on_close,
            on_error=self._on_error,
            on_message=self._on_message,
        )
        with self._lock:
            self._ws = ws
        ws.connect()

    def _schedule_reconnect(self) -> None:
        if not self._running:
            return
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, name="livefeed-reconnect", daemon=True
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        while self._running:
            self._consecutive_failures += 1
            idx = min(self._consecutive_failures - 1, len(_RECONNECT_BACKOFF) - 1)
            wait = _RECONNECT_BACKOFF[idx]
            log.warning(
                "Reconnecting in %ds (attempt %d, last close was failure-class)",
                wait, self._consecutive_failures,
            )
            if self._stop_event.wait(wait):
                return
            if self._consecutive_failures >= _REAUTH_AFTER_N_FAILURES:
                log.warning("Forcing token refresh before reconnect")
                try:
                    self._access_token = authenticate(force=True)
                except Exception as e:
                    log.error("Re-auth failed: %s", e)
                    continue
            try:
                self._connect()
                return
            except Exception as e:
                log.error("Reconnect failed: %s", e)

    # -- callbacks (run on the SDK's WS thread) --------------------------

    def _on_connect(self) -> None:
        with self._lock:
            ws = self._ws
        log.info("LiveFeed connected; subscribing to %d symbol(s)", len(self._symbols))
        self._consecutive_failures = 0
        self._silence_logged.clear()
        if ws is not None and self._symbols:
            try:
                ws.subscribe(symbols=self._symbols, data_type=self._data_type)
            except Exception as e:
                log.error("Initial subscribe failed: %s", e)

    def _on_close(self, code) -> None:
        log.info("LiveFeed closed (code=%s)", code)
        with self._lock:
            self._ws = None
        if self._running:
            self._schedule_reconnect()

    def _on_error(self, err) -> None:
        log.error("LiveFeed error: %s", err)
        msg = str(err).lower() if err is not None else ""
        if "401" in msg or "unauthor" in msg or "invalid token" in msg:
            log.warning("Auth failure detected — clearing cached token")
            try:
                if TOKEN_CACHE.exists():
                    TOKEN_CACHE.unlink()
            except OSError:
                pass
            self._access_token = None

    def _on_message(self, msg: dict) -> None:
        sym = msg.get("symbol") if isinstance(msg, dict) else None
        if sym:
            self._last_tick_at[sym] = _time.time()
            self._silence_logged.discard(sym)
        try:
            self._on_tick_user(msg)
        except Exception:
            log.exception("user on_tick handler failed")

    # -- heartbeat -------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(_HEARTBEAT_INTERVAL_SEC):
            if not self._running:
                return
            if not in_session():
                continue
            now = _time.time()
            for sym in self._symbols:
                last = self._last_tick_at.get(sym)
                if last is None:
                    continue
                gap = now - last
                if gap > _HEARTBEAT_SILENCE_WARN_SEC and sym not in self._silence_logged:
                    log.warning(
                        "No tick for %s in %.0fs (heartbeat warning)", sym, gap
                    )
                    self._silence_logged.add(sym)
