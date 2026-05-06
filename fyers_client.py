"""
Fyers broker integration.

Phase A — auth + token cache. Phases B-F (WebSocket, bar builder,
reconnect, full subscription) come next.

Auto-login flow:
  1. POST send_login_otp_v2  → request_key
  2. POST verify_otp (TOTP)  → request_key
  3. POST verify_pin_v2      → bearer access token (login-domain only)
  4. POST /api/v3/token      → auth_code in redirect URL
  5. SDK exchanges auth_code → final access_token (data-domain)

Cached on disk in .fyers_token.json. Token is treated as expired at the
next 06:00 IST (Fyers tokens roll over around the start of the trading
day). On 401 from a downstream call, force a re-auth.

CLI:
  python fyers_client.py auth         — log in, cache token, print summary
  python fyers_client.py auth --force — bypass cache, full re-login
  python fyers_client.py validate     — use cached token to call get_profile()
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pyotp
from dotenv import load_dotenv

log = logging.getLogger("alertbot.fyers")

IST = ZoneInfo("Asia/Kolkata")
TOKEN_CACHE = Path(".fyers_token.json")

FYERS_VAGATOR = "https://api-t2.fyers.in/vagator/v2"
FYERS_API_T1 = "https://api-t1.fyers.in/api/v3"


# ============================================================================
# Credentials
# ============================================================================

@dataclass
class FyersCreds:
    app_id: str          # e.g. "XYZ123ABC-100"
    secret_id: str
    redirect_uri: str
    user_id: str         # Fyers login id, e.g. "AB12345"
    pin: str             # 4-digit Fyers PIN
    totp_secret: str     # base32 TOTP secret


def load_creds() -> FyersCreds:
    load_dotenv()
    required = (
        "FYERS_APP_ID", "FYERS_SECRET_ID", "FYERS_REDIRECT_URI",
        "FYERS_USER_ID", "FYERS_PIN", "FYERS_TOTP_SECRET",
    )
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing Fyers env vars: {', '.join(missing)}. "
            f"Copy env.example to .env and fill them in."
        )
    # Normalize TOTP secret — pyotp accepts only clean base32, but copy-paste
    # often introduces spaces or lowercase letters. Strip and uppercase.
    totp_secret = os.environ["FYERS_TOTP_SECRET"].replace(" ", "").upper()
    return FyersCreds(
        app_id=os.environ["FYERS_APP_ID"].strip(),
        secret_id=os.environ["FYERS_SECRET_ID"].strip(),
        redirect_uri=os.environ["FYERS_REDIRECT_URI"].strip(),
        user_id=os.environ["FYERS_USER_ID"].strip(),
        pin=os.environ["FYERS_PIN"].strip(),
        totp_secret=totp_secret,
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
# Auto-login (5 hops)
# ============================================================================

def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _post_json(client: httpx.Client, url: str, body: dict, headers: dict | None = None) -> dict:
    """POST + parse JSON, surfacing the response body on any error."""
    r = client.post(url, json=body, headers=headers or {})
    if r.status_code >= 400:
        # Fyers wraps the actual reason in the JSON body — surface it.
        snippet = r.text[:400]
        raise RuntimeError(f"{url} returned HTTP {r.status_code}: {snippet}")
    try:
        return r.json()
    except ValueError:
        raise RuntimeError(f"{url} returned non-JSON: {r.text[:200]}")


def _login_send_otp(client: httpx.Client, user_id: str) -> str:
    data = _post_json(
        client,
        f"{FYERS_VAGATOR}/send_login_otp_v2",
        {"fy_id": _b64(user_id), "app_id": "2"},
    )
    rk = data.get("request_key")
    if not rk:
        raise RuntimeError(f"send_login_otp returned no request_key: {data}")
    return rk


def _login_verify_otp(client: httpx.Client, request_key: str, otp: str) -> str:
    data = _post_json(
        client,
        f"{FYERS_VAGATOR}/verify_otp",
        {"request_key": request_key, "otp": otp},
    )
    rk = data.get("request_key")
    if not rk:
        raise RuntimeError(f"verify_otp returned no request_key: {data}")
    return rk


def _login_verify_pin(client: httpx.Client, request_key: str, pin: str) -> str:
    data = _post_json(
        client,
        f"{FYERS_VAGATOR}/verify_pin_v2",
        {
            "request_key": request_key,
            "identifier": pin,
            "identity_type": "pin",
        },
    )
    inner = data.get("data") or {}
    token = inner.get("access_token")
    if not token:
        raise RuntimeError(f"verify_pin failed: {data}")
    return token


def _login_get_auth_code(
    client: httpx.Client, bearer: str, creds: FyersCreds
) -> str:
    data = _post_json(
        client,
        f"{FYERS_API_T1}/token",
        {
            "fyers_id": creds.user_id,
            "app_id": creds.app_id.split("-")[0],  # strip "-100" suffix
            "redirect_uri": creds.redirect_uri,
            "appType": "100",
            "code_challenge": "",
            "state": "None",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        },
        headers={"Authorization": f"Bearer {bearer}"},
    )
    redirect = data.get("Url") or data.get("url") or ""
    qs = parse_qs(urlparse(redirect).query)
    auth_code = (qs.get("auth_code") or qs.get("code") or [None])[0]
    if not auth_code:
        raise RuntimeError(f"no auth_code in /token response: {data}")
    return auth_code


def _exchange_auth_code(creds: FyersCreds, auth_code: str) -> str:
    """Use the SDK's SessionModel to do the standard auth_code → access_token swap (handles appIdHash)."""
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


def _authenticate_auto() -> str:
    """Fully automated login via TOTP (5 hops through Fyers vagator endpoints)."""
    creds = load_creds()
    log.info("Starting Fyers auto-login for user %s", creds.user_id)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=20, headers=headers) as client:
        rk1 = _login_send_otp(client, creds.user_id)
        log.info("Step 1/5 OK — OTP request key acquired")
        otp = pyotp.TOTP(creds.totp_secret).now()
        rk2 = _login_verify_otp(client, rk1, otp)
        log.info("Step 2/5 OK — TOTP verified")
        bearer = _login_verify_pin(client, rk2, creds.pin)
        log.info("Step 3/5 OK — PIN verified")
        auth_code = _login_get_auth_code(client, bearer, creds)
        log.info("Step 4/5 OK — auth_code obtained")
    access_token = _exchange_auth_code(creds, auth_code)
    log.info("Step 5/5 OK — access_token issued")
    return access_token


def _authenticate_manual() -> str:
    """Manual flow: print a login URL, wait for the user to paste the redirect."""
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
    print("Manual Fyers login")
    print("=" * 70)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {login_url}")
    print()
    print("2. Log in with your Fyers credentials (PIN + TOTP from your")
    print("   authenticator app).")
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


def authenticate(force: bool = False, manual: bool = False) -> str:
    """Return a valid Fyers access token, doing a full login if cache is stale or `force`."""
    if not force:
        cached = _load_cached_token()
        if cached:
            log.info("Using cached Fyers token")
            return cached

    access_token = _authenticate_manual() if manual else _authenticate_auto()
    expiry = _next_token_expiry()
    _save_token(access_token, expiry)
    log.info("Fyers token cached until %s IST", expiry.isoformat())
    return access_token


# ============================================================================
# Tick → 5-minute OHLCV bar aggregator
# ============================================================================
# Ticks arrive on a daemon thread; the bot reads bars from the async loop.
# A single threading.Lock around mutations is plenty for this volume — even
# at 500 symbols × multiple ticks/sec, contention is negligible.
#
# Volume math: Fyers ticks carry vol_traded_today (cumulative since 09:15).
# Each bar's volume = (vol_traded_today at bar close) − (vol at bar open).
# When a tick crosses the bar boundary, we close the previous bar's end_vol
# at the new tick's vol, so no volume is lost in the seam.
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
    """Thread-safe latest-price + rolling-bar store. Singleton via module-level instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_price: dict[str, float] = {}
        self._latest_ts: dict[str, datetime] = {}
        self._bars: dict[str, deque[Bar]] = {}
        self._current_bar: dict[str, Bar] = {}
        self._tick_count = 0

    def on_tick(self, msg: dict) -> None:
        # Lifecycle frames (cn / ful / sub) carry no market data.
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
                    # Capture all volume up to the boundary, then archive.
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
        """Return bars in the same shape yfinance returns: index=Datetime, cols=Open/High/Low/Close/Volume."""
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


# Singleton — bot.py will read from this same instance.
TICK_STORE = TickStore()


# ============================================================================
# WebSocket — live tick subscription
# ============================================================================
# fyers_apiv3.FyersWebsocket.data_ws runs the connection on a background
# thread once .connect() is called. The access_token field expects the
# combined "<app_id>:<access_token>" format — passing just the token
# silently fails with no error.
# ============================================================================

from typing import Callable

TickHandler = Callable[[dict], None]


def start_data_socket(
    symbols: list[str],
    on_tick: TickHandler,
    *,
    data_type: str = "SymbolUpdate",
    reconnect_in_sdk: bool = False,
):
    """Open a Fyers WebSocket, subscribe to symbols, call on_tick per message.

    Returns the FyersDataSocket instance. Caller is responsible for
    keeping the main thread alive (the socket runs in a daemon thread).
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

def _cli_auth(force: bool, manual: bool) -> int:
    try:
        token = authenticate(force=force, manual=manual)
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


def _cli_totp() -> int:
    """Print the current TOTP and check system-clock drift vs an internet time source."""
    try:
        creds = load_creds()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 2
    secret = creds.totp_secret
    print(f"TOTP secret length: {len(secret)} chars")
    print(f"TOTP secret first/last 2 chars: {secret[:2]}...{secret[-2:]} (sanity check, not full secret)")
    try:
        code = pyotp.TOTP(secret).now()
    except Exception as e:
        print(f"FAIL: pyotp couldn't parse the secret ({e}).")
        print("Likely cause: secret has non-base32 characters. Valid chars: A-Z, 2-7, optionally =.")
        return 2
    now = datetime.now(IST)
    print(f"Current TOTP code (now): {code}")
    print(f"System time (IST):       {now.strftime('%Y-%m-%d %H:%M:%S')}")
    # Drift check: hit a public time API and compare
    try:
        r = httpx.get("https://worldtimeapi.org/api/timezone/Asia/Kolkata", timeout=8)
        net_dt = datetime.fromisoformat(r.json()["datetime"]).replace(tzinfo=None)
        local_dt = now.replace(tzinfo=None)
        drift = (local_dt - net_dt).total_seconds()
        print(f"Internet time (IST):     {net_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Drift: {drift:+.1f}s {'(OK)' if abs(drift) < 5 else '(TOO HIGH — fix with: w32tm /resync)'}")
    except Exception as e:
        print(f"(skipped drift check: {e})")
    print()
    print("Compare the TOTP code above with the one your authenticator app shows for Fyers.")
    print("They MUST match. If they don't, the FYERS_TOTP_SECRET in .env is wrong.")
    return 0


def _cli_ticks() -> int:
    """Phase B verify: subscribe to 3 symbols and print incoming ticks."""
    import time
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("Phase B WebSocket test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Ticks flow only during NSE market hours (Mon-Fri 09:15-15:30 IST).")
    print("Pre-market: you should see a 'WS connected' line and no ticks (that's OK).")
    print("During market: ticks should arrive within ~1s.")
    print("Press Ctrl+C to stop.")
    print()

    counter = {"n": 0}

    def on_tick(msg):
        counter["n"] += 1
        print(f"  [{counter['n']:5d}] {msg}")

    try:
        start_data_socket(symbols, on_tick)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopped. {counter['n']} ticks received total.")
        return 0
    except Exception as e:
        log.exception("ticks command failed: %s", e)
        return 2


def _cli_bars() -> int:
    """Phase C verify: subscribe to 3 symbols and dump the aggregated bars on stop."""
    import time
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("Phase C bar-aggregator test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Run during market hours (09:15-15:30 IST) for at least one full")
    print("5-minute boundary crossing (ideally 10+ minutes) so you see closed")
    print("bars plus the in-progress one. Ctrl+C to stop and dump.")
    print()
    try:
        start_data_socket(symbols, TICK_STORE.on_tick)
        last_log = 0.0
        while True:
            time.sleep(2)
            now = time.time()
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


def _cli_validate() -> int:
    """Use the token to make a real API call (get_profile)."""
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if len(sys.argv) < 2:
        print("usage: python fyers_client.py {auth|validate|totp|ticks|bars} [--force] [--manual]")
        return 1
    cmd = sys.argv[1]
    if cmd == "auth":
        return _cli_auth(
            force="--force" in sys.argv,
            manual="--manual" in sys.argv,
        )
    if cmd == "validate":
        return _cli_validate()
    if cmd == "totp":
        return _cli_totp()
    if cmd == "ticks":
        return _cli_ticks()
    if cmd == "bars":
        return _cli_bars()
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
