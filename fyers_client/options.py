"""Fyers options-chain fetcher — drop-in fallback for NSE's blocked
``/api/option-chain-indices`` endpoint.

NSE silently returns ``{}`` for the option-chain API to non-browser
clients (Akamai bot-detection). Fyers' authenticated endpoint
returns the same per-strike OI / IV data and works reliably as long
as the daily access token is present.

Public surface:
  ``fetch_option_chain_payload(fyers_symbol, strikecount=20)``
      Async wrapper that calls the sync ``FyersModel.optionchain()``
      on a worker thread (so the scanner's event loop doesn't stall
      while Fyers HTTP-roundtrips).

  ``normalise_to_nse_shape(fyers_payload)``
      Converts Fyers' option-chain shape to the same shape NSE
      returns, so the existing ``data.nse.parse_option_chain`` /
      ``parse_pcr`` keep working unchanged.

Fyers index symbols:
  ``NSE:NIFTY50-INDEX``   for NIFTY
  ``NSE:NIFTYBANK-INDEX`` for BANKNIFTY
``data.nse`` callers pass ``'NIFTY'``/``'BANKNIFTY'``; the local
mapping table here translates."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from zoneinfo import ZoneInfo

log = logging.getLogger("alertbot.fyers.options")

IST = ZoneInfo("Asia/Kolkata")


SYMBOL_MAP = {
    "NIFTY":      "NSE:NIFTY50-INDEX",
    "BANKNIFTY":  "NSE:NIFTYBANK-INDEX",
    "FINNIFTY":   "NSE:FINNIFTY-INDEX",
}


def _to_fyers_index_symbol(symbol: str) -> str:
    """Translate the NSE-style symbol the rest of the code uses
    (``'NIFTY'``) into the Fyers form (``'NSE:NIFTY50-INDEX'``).
    Passes through any value already in Fyers form."""
    upper = symbol.strip().upper()
    if upper in SYMBOL_MAP:
        return SYMBOL_MAP[upper]
    if ":" in upper:  # already Fyers-formatted
        return upper
    raise ValueError(
        f"Unknown index symbol for Fyers option chain: {symbol!r}. "
        f"Known: {list(SYMBOL_MAP)}",
    )


def _build_client():
    """Lazy-import the SDK + use the cached access token. Returns
    None when the token isn't available (haven't auth'd today) so
    the caller can fall through gracefully."""
    from .creds import load_creds
    from .token_cache import _load_cached_token

    token = _load_cached_token()
    if not token:
        log.info("Fyers options: no cached token; skipping fetch")
        return None
    creds = load_creds()
    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        log.warning("Fyers SDK not installed; option chain unavailable")
        return None
    return fyersModel.FyersModel(
        client_id=creds.app_id, token=token, is_async=False,
    )


def _sync_fetch(
    fyers_symbol: str, strikecount: int = 20,
) -> dict[str, Any] | None:
    """Blocking Fyers call. Wrapped by ``fetch_option_chain_payload``
    in ``asyncio.to_thread`` so the scanner's loop stays responsive."""
    client = _build_client()
    if client is None:
        return None
    try:
        # ``timestamp`` is the expiry filter — empty string means
        # "all expiries", which is what we need to compute total-OI PCR.
        response = client.optionchain(data={
            "symbol": fyers_symbol,
            "strikecount": strikecount,
            "timestamp": "",
        })
    except Exception:
        log.exception("Fyers optionchain call failed for %s", fyers_symbol)
        return None
    if not isinstance(response, dict):
        return None
    if response.get("s") != "ok":
        log.warning(
            "Fyers optionchain non-ok: %s %s",
            response.get("code"), response.get("message"),
        )
        return None
    return response.get("data")


async def fetch_option_chain_payload(
    symbol: str = "NIFTY", strikecount: int = 20,
) -> dict[str, Any] | None:
    """Async-friendly fetch. ``symbol`` is the NSE-style name
    (``'NIFTY'``); the Fyers symbol mapping is internal."""
    fyers_symbol = _to_fyers_index_symbol(symbol)
    return await asyncio.to_thread(_sync_fetch, fyers_symbol, strikecount)


# ---------------------------------------------------------------------------
# Shape adaptor — Fyers → NSE
# ---------------------------------------------------------------------------

def _expiry_label(expiry_ts: Any) -> str | None:
    """Fyers exposes expiries as unix epochs (UTC seconds). NSE
    serves them as ``'16-May-2026'`` strings. Convert so the
    NSE-shaped parsers (``parse_pcr``) match by expiry string."""
    if expiry_ts is None:
        return None
    try:
        ts = float(expiry_ts)
    except (TypeError, ValueError):
        if isinstance(expiry_ts, str):
            return expiry_ts
        return None
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(IST)
    except (OSError, OverflowError, ValueError):
        return None
    return dt.strftime("%d-%b-%Y")


def normalise_to_nse_shape(fyers_data: dict[str, Any] | None) -> dict[str, Any]:
    """Re-pack Fyers' optionsChain payload into the NSE shape that
    :func:`data.nse.parse_option_chain` already understands.

    Fyers shape (relevant fields):
      ``optionsChain``: [{strike_price, option_type, oi, ltp,
                          fyToken, ex_symbol, expiry, ...}, ...]
        — note: the FIRST entry is the underlying spot, not a strike;
          we drop entries where ``option_type`` isn't 'CE'/'PE'.
      ``expiryData``: [{expiry: epoch, date: 'DD-MM-YYYY'}, ...]

    NSE-shape output:
      ``records.expiryDates``: ['16-May-2026', '23-May-2026', ...]
      ``records.data``: [{strikePrice, expiryDate,
                         CE: {openInterest, impliedVolatility},
                         PE: {...}}, ...]"""
    if not isinstance(fyers_data, dict):
        return {"records": {"data": [], "expiryDates": []}}

    chain = fyers_data.get("optionsChain") or []
    # Group by (strike, expiry) → {CE: row, PE: row}.
    grouped: dict[tuple[float, str | None], dict[str, dict]] = {}
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        opt_type = (entry.get("option_type") or "").upper()
        if opt_type not in ("CE", "PE"):
            continue  # skip the underlying spot row
        strike = entry.get("strike_price")
        if strike is None:
            continue
        try:
            strike_f = float(strike)
        except (TypeError, ValueError):
            continue
        expiry_lbl = _expiry_label(entry.get("expiry"))
        key = (strike_f, expiry_lbl)
        bucket = grouped.setdefault(key, {})
        bucket[opt_type] = {
            "openInterest": entry.get("oi"),
            "changeinOpenInterest": entry.get("oich"),
            "impliedVolatility": entry.get("iv") or entry.get("implied_volatility"),
        }

    data_rows: list[dict[str, Any]] = []
    for (strike, expiry_lbl), legs in grouped.items():
        row: dict[str, Any] = {"strikePrice": strike, "expiryDate": expiry_lbl}
        if "CE" in legs:
            row["CE"] = legs["CE"]
        if "PE" in legs:
            row["PE"] = legs["PE"]
        data_rows.append(row)

    expiry_dates: list[str] = []
    for ed in fyers_data.get("expiryData") or []:
        lbl = _expiry_label(ed.get("expiry"))
        if lbl is None and isinstance(ed.get("date"), str):
            # Fyers' ``date`` is DD-MM-YYYY; convert to DD-MMM-YYYY
            try:
                lbl = datetime.strptime(ed["date"], "%d-%m-%Y").strftime("%d-%b-%Y")
            except ValueError:
                lbl = ed["date"]
        if lbl and lbl not in expiry_dates:
            expiry_dates.append(lbl)

    # Sort data by (expiry, strike) so the parse_pcr nearest-expiry
    # heuristic (first expiry in records.expiryDates) lines up.
    data_rows.sort(
        key=lambda r: (
            r.get("expiryDate") or "9999-99-99",
            r.get("strikePrice") or 0,
        ),
    )

    return {"records": {"data": data_rows, "expiryDates": expiry_dates}}


async def fetch_option_chain_nse_shape(
    symbol: str = "NIFTY", strikecount: int = 20,
) -> dict[str, Any] | None:
    """One-shot: fetch from Fyers + reshape. Returns None when the
    fetch failed entirely (no token, SDK error). Returns an empty
    NSE-shaped envelope when Fyers returned but with no chain rows."""
    raw = await fetch_option_chain_payload(symbol, strikecount=strikecount)
    if raw is None:
        return None
    return normalise_to_nse_shape(raw)
