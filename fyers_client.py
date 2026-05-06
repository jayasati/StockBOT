"""
Fyers broker integration — manual auth + WebSocket data feed.

Auth flow (one-time per day, ~15 sec):
  1. ``python fyers_client.py auth`` prints a Fyers OAuth URL
  2. Open URL in your browser, log in (PIN + TOTP from your authenticator)
  3. Browser redirects to your registered redirect URI with ``?auth_code=...``
  4. Paste that full URL back into the terminal
  5. SDK exchanges auth_code → access_token; we cache it in .fyers_token.json
     until the next 06:00 IST (Fyers' daily token rollover)

The bot reads the cached token at startup. As long as you do step 1–4 once
each morning before 09:15 IST, the bot runs unattended for the trading day.

We previously tried a fully automated TOTP-based login through Fyers'
internal vagator endpoints; Fyers added a JS-minted captcha in 2026 that
made that flow impossible from raw Python. Dropped — manual is the only
supported path now.

CLI:
  python fyers_client.py auth          — open the manual login flow, cache token
  python fyers_client.py auth --force  — bypass cache and re-login
  python fyers_client.py validate      — use cached token to call get_profile()
  python fyers_client.py ticks         — verify WebSocket: print live ticks
  python fyers_client.py bars          — verify aggregator: print 5-min bars on Ctrl-C
  python fyers_client.py record <out.jsonl> <secs> [SYM ...]  — capture a tick fixture
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time as _time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger("alertbot.fyers")

IST = ZoneInfo("Asia/Kolkata")
TOKEN_CACHE = Path(".fyers_token.json")


# ============================================================================
# Credentials
# ============================================================================

@dataclass
class FyersCreds:
    app_id: str          # e.g. "XYZ123ABC-100"
    secret_id: str
    redirect_uri: str    # must match what's registered in myapi.fyers.in


def load_creds() -> FyersCreds:
    load_dotenv()
    required = ("FYERS_APP_ID", "FYERS_SECRET_ID", "FYERS_REDIRECT_URI")
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing Fyers env vars: {', '.join(missing)}. "
            f"Copy env.example to .env and fill them in."
        )
    return FyersCreds(
        app_id=os.environ["FYERS_APP_ID"].strip(),
        secret_id=os.environ["FYERS_SECRET_ID"].strip(),
        redirect_uri=os.environ["FYERS_REDIRECT_URI"].strip(),
    )


# ============================================================================
# Token cache
# ============================================================================

def _next_token_expiry() -> datetime:
    """Fyers tokens roll over near the start of the trading day. Next 06:00 IST minus a 5-min safety margin."""
    now = datetime.now(IST)
    expiry = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= expiry:
        expiry = expiry + timedelta(days=1)
    return expiry - timedelta(minutes=5)


def _load_cached_token() -> str | None:
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        expiry = datetime.fromisoformat(data["expiry"])
        if datetime.now(IST) < expiry:
            return data["access_token"]
        log.info("Cached Fyers token expired at %s", expiry.isoformat())
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        log.warning("Token cache unreadable (%s); re-authenticating", e)
    return None


def _save_token(access_token: str, expiry: datetime) -> None:
    TOKEN_CACHE.write_text(
        json.dumps(
            {"access_token": access_token, "expiry": expiry.isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        os.chmod(TOKEN_CACHE, 0o600)
    except OSError:
        pass  # Windows — ACLs already inherit from the user dir


# ============================================================================
# Manual login (browser OAuth flow)
# ============================================================================

def _exchange_auth_code(creds: FyersCreds, auth_code: str) -> str:
    """Use the SDK's SessionModel to do the standard auth_code → access_token swap."""
    from fyers_apiv3 import fyersModel
    session = fyersModel.SessionModel(
        client_id=creds.app_id,
        secret_key=creds.secret_id,
        redirect_uri=creds.redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    response = session.generate_token()
    token = response.get("access_token")
    if not token:
        raise RuntimeError(f"SDK token exchange failed: {response}")
    return token


def _authenticate_manual() -> str:
    """Print a login URL, wait for the user to paste back the redirect URL."""
    creds = load_creds()
    from fyers_apiv3 import fyersModel
    session = fyersModel.SessionModel(
        client_id=creds.app_id,
        secret_key=creds.secret_id,
        redirect_uri=creds.redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    login_url = session.generate_authcode()
    print()
    print("=" * 70)
    print("Fyers manual login")
    print("=" * 70)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {login_url}")
    print()
    print("2. Log in with PIN + TOTP from your authenticator app.")
    print()
    print("3. Your browser will redirect to your registered redirect URI")
    print(f"   ({creds.redirect_uri}). The page may show 'site can't be reached'")
    print("   — that's fine; we only need the URL from your address bar.")
    print()
    print("4. Copy the FULL URL from the browser address bar and paste it below.")
    print("   It will look like: <redirect_uri>?auth_code=XYZ&state=...")
    print()
    pasted = input("Paste redirect URL: ").strip()
    qs = parse_qs(urlparse(pasted).query)
    auth_code = (qs.get("auth_code") or qs.get("code") or [None])[0]
    if not auth_code:
        # Maybe they pasted just the auth_code, not the full URL
        if pasted and "?" not in pasted and " " not in pasted:
            auth_code = pasted
        else:
            raise RuntimeError(
                "Couldn't find auth_code in the pasted text. "
                "Make sure you copied the FULL URL from the browser address bar."
            )
    log.info("Got auth_code from pasted URL")
    access_token = _exchange_auth_code(creds, auth_code)
    log.info("access_token issued")
    return access_token


def authenticate(force: bool = False) -> str:
    """Return a valid Fyers access token. Uses the cached one if not expired,
    otherwise prompts for the manual paste flow."""
    if not force:
        cached = _load_cached_token()
        if cached:
            log.info("Using cached Fyers token")
            return cached

    access_token = _authenticate_manual()
    expiry = _next_token_expiry()
    _save_token(access_token, expiry)
    log.info("Fyers token cached until %s IST", expiry.isoformat())
    return access_token


# ============================================================================
# Tick → 5-minute OHLCV bar aggregator (legacy in-memory store)
# ============================================================================
# Used by the `python fyers_client.py bars` CLI verifier. The production
# aggregator that bot.py reads from lives in data/realtime_feed.py.
# ============================================================================

BAR_INTERVAL_SECONDS = 300  # 5 minutes
MAX_BARS_PER_SYMBOL = 100


@dataclass
class Bar:
    ts: datetime          # bar start time, IST naive (matching yfinance shape)
    open: float
    high: float
    low: float
    close: float
    start_vol: float
    end_vol: float

    @property
    def volume(self) -> float:
        return max(0.0, self.end_vol - self.start_vol)


def _floor_to_bar(ts: datetime) -> datetime:
    minute = ts.minute - (ts.minute % 5)
    return ts.replace(minute=minute, second=0, microsecond=0)


def _epoch_to_ist(epoch_seconds: int | float) -> datetime:
    """Fyers last_traded_time is epoch seconds (UTC)."""
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .astimezone(IST)
        .replace(tzinfo=None)
    )


class TickStore:
    """Thread-safe latest-price + rolling-bar store for the CLI verifier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_price: dict[str, float] = {}
        self._latest_ts: dict[str, datetime] = {}
        self._bars: dict[str, deque[Bar]] = {}
        self._current_bar: dict[str, Bar] = {}
        self._tick_count = 0

    def on_tick(self, msg: dict) -> None:
        if msg.get("type") != "sf":
            return
        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        ts_epoch = msg.get("last_traded_time")
        vol = float(msg.get("vol_traded_today") or 0)
        if not symbol or ltp is None or ts_epoch is None:
            return
        ltp = float(ltp)
        ts = _epoch_to_ist(ts_epoch)
        bar_ts = _floor_to_bar(ts)

        with self._lock:
            self._tick_count += 1
            self._latest_price[symbol] = ltp
            self._latest_ts[symbol] = ts
            cur = self._current_bar.get(symbol)
            if cur is None or cur.ts != bar_ts:
                if cur is not None:
                    cur.end_vol = vol
                    self._bars.setdefault(
                        symbol, deque(maxlen=MAX_BARS_PER_SYMBOL)
                    ).append(cur)
                self._current_bar[symbol] = Bar(
                    ts=bar_ts, open=ltp, high=ltp, low=ltp, close=ltp,
                    start_vol=vol, end_vol=vol,
                )
            else:
                cur.high = max(cur.high, ltp)
                cur.low = min(cur.low, ltp)
                cur.close = ltp
                cur.end_vol = vol

    def get_bars_df(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            archived = list(self._bars.get(symbol, ()))
            current = self._current_bar.get(symbol)
        all_bars: list[Bar] = archived + ([current] if current is not None else [])
        if not all_bars:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([b.ts for b in all_bars], name="Datetime")
        return pd.DataFrame(
            {
                "Open":   [b.open for b in all_bars],
                "High":   [b.high for b in all_bars],
                "Low":    [b.low for b in all_bars],
                "Close":  [b.close for b in all_bars],
                "Volume": [b.volume for b in all_bars],
            },
            index=idx,
        )

    def latest_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._latest_price.get(symbol)

    def stats(self) -> dict:
        with self._lock:
            return {
                "ticks": self._tick_count,
                "symbols_seen": len(self._latest_price),
                "open_bars":  len(self._current_bar),
                "archived_bars": sum(len(b) for b in self._bars.values()),
            }


TICK_STORE = TickStore()


# ============================================================================
# WebSocket — live tick subscription
# ============================================================================

TickHandler = Callable[[dict], None]


def start_data_socket(
    symbols: list[str],
    on_tick: TickHandler,
    *,
    data_type: str = "SymbolUpdate",
    reconnect_in_sdk: bool = False,
):
    """Open a Fyers WebSocket, subscribe to symbols, call on_tick per message.

    Thin helper retained for the CLI ticks/bars verifiers. Production code
    (bot.py) uses ``LiveFeed`` below — it adds reconnect, re-auth on close,
    and per-symbol heartbeat.
    """
    creds = load_creds()
    token = authenticate()
    from fyers_apiv3.FyersWebsocket import data_ws

    state: dict = {"ws": None}

    def _on_message(msg):
        try:
            on_tick(msg)
        except Exception as e:
            log.exception("on_tick handler failed: %s", e)

    def _on_error(err):
        log.error("WS error: %s", err)

    def _on_close(code):
        log.info("WS closed (code=%s)", code)

    def _on_connect():
        ws = state["ws"]
        log.info("WS connected; subscribing to %d symbol(s)", len(symbols))
        if ws is not None:
            ws.subscribe(symbols=symbols, data_type=data_type)

    state["ws"] = data_ws.FyersDataSocket(
        access_token=f"{creds.app_id}:{token}",
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=reconnect_in_sdk,
        on_connect=_on_connect,
        on_close=_on_close,
        on_error=_on_error,
        on_message=_on_message,
    )
    state["ws"].connect()
    return state["ws"]


# ============================================================================
# LiveFeed — production WebSocket wrapper
# ============================================================================
# Wraps FyersDataSocket with:
#   * exponential-backoff reconnect (1, 2, 4, 8, 16, 32, 60, 60... seconds)
#   * auto re-auth after N consecutive failures (token rolls over at 06:00 IST)
#   * per-symbol heartbeat — logs WARNING if a subscribed symbol has been
#     silent for >30s during market hours
# ============================================================================

_RECONNECT_BACKOFF = (1, 2, 4, 8, 16, 32, 60)
_REAUTH_AFTER_N_FAILURES = 3
_HEARTBEAT_INTERVAL_SEC = 5
_HEARTBEAT_SILENCE_WARN_SEC = 30
_MARKET_OPEN = datetime.min.time().replace(hour=9, minute=15)
_MARKET_CLOSE = datetime.min.time().replace(hour=15, minute=30)


def _is_market_hours_now() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return _MARKET_OPEN <= now.time() <= _MARKET_CLOSE


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
            log_path="",
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
            if not _is_market_hours_now():
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


# ============================================================================
# record_ticks — debug helper to capture a JSONL fixture
# ============================================================================

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


# ============================================================================
# Symbol conversion: yfinance "<SYM>.NS" <-> Fyers "NSE:<SYM>-EQ"
# ============================================================================

def to_fyers(yf_symbol: str) -> str:
    """'RELIANCE.NS' → 'NSE:RELIANCE-EQ'. Special chars (M&M, BAJAJ-AUTO) pass through."""
    sym = yf_symbol[:-3] if yf_symbol.endswith(".NS") else yf_symbol
    sym = sym[:-3] if sym.endswith(".BO") else sym
    return f"NSE:{sym}-EQ"


def to_yf(fy_symbol: str) -> str:
    """'NSE:RELIANCE-EQ' → 'RELIANCE.NS'."""
    s = fy_symbol.split(":", 1)[-1]
    if s.endswith("-EQ"):
        s = s[:-3]
    return f"{s}.NS"


# ============================================================================
# CLI
# ============================================================================

def _cli_auth(force: bool) -> int:
    try:
        token = authenticate(force=force)
    except Exception as e:
        log.exception("Auth failed: %s", e)
        return 2
    print()
    print(f"  Got Fyers access token (length={len(token)})")
    print(f"  Cached at: {TOKEN_CACHE.absolute()}")
    if TOKEN_CACHE.exists():
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        print(f"  Expires:   {data['expiry']} IST")
    return 0


def _cli_validate() -> int:
    try:
        token = authenticate()
        creds = load_creds()
    except Exception as e:
        log.exception("Auth failed: %s", e)
        return 2
    from fyers_apiv3 import fyersModel
    fyers = fyersModel.FyersModel(
        client_id=creds.app_id, token=token, log_path=""
    )
    profile = fyers.get_profile()
    print(json.dumps(profile, indent=2))
    if profile.get("s") != "ok":
        return 3
    return 0


def _cli_ticks() -> int:
    """Subscribe to 3 symbols and print incoming ticks."""
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("WebSocket test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Ticks flow only during NSE market hours (Mon-Fri 09:15-15:30 IST).")
    print("Press Ctrl+C to stop.")
    print()

    counter = {"n": 0}

    def on_tick(msg):
        counter["n"] += 1
        print(f"  [{counter['n']:5d}] {msg}")

    try:
        start_data_socket(symbols, on_tick)
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopped. {counter['n']} ticks received total.")
        return 0
    except Exception as e:
        log.exception("ticks command failed: %s", e)
        return 2


def _cli_bars() -> int:
    """Subscribe to 3 symbols and dump the aggregated bars on stop."""
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("Bar-aggregator test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Run during market hours (09:15-15:30 IST) for at least one full")
    print("5-minute boundary crossing. Ctrl+C to stop and dump.")
    print()
    try:
        start_data_socket(symbols, TICK_STORE.on_tick)
        last_log = 0.0
        while True:
            _time.sleep(2)
            now = _time.time()
            if now - last_log >= 10:
                stats = TICK_STORE.stats()
                print(
                    f"  [stats] ticks={stats['ticks']} symbols={stats['symbols_seen']} "
                    f"open_bars={stats['open_bars']} archived={stats['archived_bars']}"
                )
                last_log = now
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print(f"Final stats: {TICK_STORE.stats()}")
        print("=" * 70)
        for sym in symbols:
            df = TICK_STORE.get_bars_df(sym)
            print()
            print(f"--- {sym} ({len(df)} bars) ---")
            if df.empty:
                print("   (no bars built — no qualifying ticks received)")
                continue
            with pd.option_context("display.max_rows", 20, "display.width", 120):
                print(df)
        return 0
    except Exception as e:
        log.exception("bars command failed: %s", e)
        return 2


def _cli_record(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: python fyers_client.py record <output.jsonl> <duration_sec> [SYM1 SYM2 ...]")
        return 1
    out = argv[2]
    try:
        dur = int(argv[3])
    except ValueError:
        print(f"duration must be an integer; got {argv[3]!r}")
        return 1
    syms = argv[4:] or ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ", "NSE:ICICIBANK-EQ"]
    n = record_ticks(syms, out, dur)
    print(f"Captured {n} ticks to {out}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if len(sys.argv) < 2:
        print("usage: python fyers_client.py {auth|validate|ticks|bars|record} [--force]")
        return 1
    cmd = sys.argv[1]
    if cmd == "auth":
        return _cli_auth(force="--force" in sys.argv)
    if cmd == "validate":
        return _cli_validate()
    if cmd == "ticks":
        return _cli_ticks()
    if cmd == "bars":
        return _cli_bars()
    if cmd == "record":
        return _cli_record(sys.argv)
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
