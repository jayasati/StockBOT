"""Fyers OAuth flow — manual paste-the-redirect-URL.

Auth flow (one-time per day, ~15 sec):
  1. ``python -m fyers_client auth`` prints a Fyers OAuth URL
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
"""
from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

from .creds import FyersCreds, load_creds
from .token_cache import _load_cached_token, _next_token_expiry, _save_token

log = logging.getLogger("alertbot.fyers")


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
