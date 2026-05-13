"""Phase-9 NSE free-API client — VIX, FII/DII, PCR, options OI,
NIFTY/BANKNIFTY quotes, market-status, F&O ban.

NSE blocks every request that doesn't first hit a "real" page to seed
cookies (``nseappid``, ``bm_sv``, ``ak_bmsc``). The session manager
below replicates a browser: GET ``/`` then a landing page tied to the
endpoint, THEN the API call with a Referer header. Without the
warmup NSE returns 401, or a 200 with an HTML body instead of JSON.

Per-endpoint cache: the live bot scans every 5 min, but background
tasks (health checks, /status) may also call into these fetchers.
Every successful response is cached in-memory for 60 s and persisted
to the ``nse_snapshots`` table for replay analysis. The cache check
is keyed on ``(endpoint, params)``.

Failure handling: every public fetch is best-effort. Network errors,
non-JSON responses, schema mismatches and rate-limiting all return
``None`` (or empty collections); the caller's structure logic should
treat missing context as neutral, not error out the whole scan.

Parsers are exposed as pure ``parse_*`` functions so tests can drive
them with recorded JSON fixtures without touching the network. The
``async def fetch_*`` wrappers just glue ``parse_*`` to the HTTP
client.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from core.config import DB_PATH, IST

log = logging.getLogger("alertbot.nse")


# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------

NSE_HOME = "https://www.nseindia.com/"
NSE_API = "https://www.nseindia.com/api"

ENDPOINTS = {
    "market_status":  f"{NSE_API}/marketStatus",
    "all_indices":    f"{NSE_API}/allIndices",
    "stock_indices":  f"{NSE_API}/equity-stockIndices",
    "option_chain":   f"{NSE_API}/option-chain-indices",
    "fii_dii":        f"{NSE_API}/fiidiiTradeReact",
    "fno_ban":        f"{NSE_API}/snapshot-derivatives-equity",
}

# Per-endpoint Referer headers — NSE checks these. Pulled from each
# endpoint's actual page on nseindia.com.
REFERERS = {
    "market_status":  "https://www.nseindia.com/market-data",
    "all_indices":    "https://www.nseindia.com/market-data/live-market-indices",
    "stock_indices":  "https://www.nseindia.com/market-data/live-equity-market",
    "option_chain":   "https://www.nseindia.com/option-chain",
    "fii_dii":        "https://www.nseindia.com/reports/fii-dii",
    "fno_ban":        "https://www.nseindia.com/products-services/equity-derivatives-list-underlyings-information",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class NseDataError(Exception):
    """Schema mismatch or expected key missing in an NSE response."""


# ---------------------------------------------------------------------------
# Session manager — singleton, cookie-warmed
# ---------------------------------------------------------------------------

_session: httpx.AsyncClient | None = None
_session_lock = asyncio.Lock()
_session_warmed_at: float = 0.0
_landing_pages_warmed: set[str] = set()
"""Tracks which per-endpoint landing pages this session has already
visited. NSE's stricter endpoints (option-chain) refuse to serve
data unless the corresponding landing page has been GET-ted at least
once to set page-specific cookies (akamai's ``bm_sv``, etc.)."""

SESSION_WARMUP_TTL = 30 * 60   # re-seed cookies every 30 min


async def _get_session() -> httpx.AsyncClient:
    """Return the shared httpx.AsyncClient. Cookie warmup runs on
    first use AND every ``SESSION_WARMUP_TTL`` seconds — NSE cookies
    drift, so a long-running bot needs to re-seed periodically."""
    global _session, _session_warmed_at, _landing_pages_warmed
    async with _session_lock:
        now = time.time()
        needs_warmup = (
            _session is None
            or (now - _session_warmed_at) > SESSION_WARMUP_TTL
        )
        if _session is None:
            _session = httpx.AsyncClient(
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-IN,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    # Sec-Fetch headers — NSE's Akamai rules treat
                    # requests lacking these as bot traffic and
                    # silently return ``{}``. Values match what a
                    # modern Chrome XHR from an open NSE page sends.
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin",
                    "sec-ch-ua": (
                        '"Not_A Brand";v="8", "Chromium";v="120", '
                        '"Google Chrome";v="120"'
                    ),
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "Priority": "u=1, i",
                },
                timeout=httpx.Timeout(15.0, connect=10.0),
                follow_redirects=True,
            )
        if needs_warmup:
            try:
                await _session.get(NSE_HOME)
                _session_warmed_at = now
                # Cookies were just reset — forget which landing
                # pages we've visited so they re-warm on demand.
                _landing_pages_warmed.clear()
            except httpx.RequestError as e:
                log.warning("NSE cookie warmup failed: %s", e)
    return _session


async def _ensure_landing_warm(referer: str) -> None:
    """Hit ``referer`` once per session if we haven't already.

    NSE's option-chain endpoint specifically returns an empty
    ``{}`` body (HTTP 200, content-type application/json) when the
    request arrives without the landing-page cookies set. The fix
    is to GET the actual page first — that lets Akamai issue the
    page-specific ``bm_sv`` cookie, which the API endpoint then
    validates. Idempotent inside one session.

    Other endpoints work fine with homepage-only warmup, but
    pre-warming them here costs one extra GET per session and makes
    every endpoint equally reliable, so we do it unconditionally."""
    global _landing_pages_warmed
    if referer in _landing_pages_warmed:
        return
    session = _session
    if session is None:
        return
    try:
        async with _session_lock:
            if referer in _landing_pages_warmed:
                return
            await session.get(referer)
            _landing_pages_warmed.add(referer)
            log.debug("NSE landing-page warm: %s", referer)
    except httpx.RequestError as e:
        log.warning("NSE landing-page warm failed for %s: %s", referer, e)


async def close_session() -> None:
    """Best-effort cleanup. Used at bot shutdown; tests can call it
    in fixture teardown."""
    global _session, _session_warmed_at, _landing_pages_warmed
    async with _session_lock:
        if _session is not None:
            try:
                await _session.aclose()
            except Exception:
                log.exception("NSE session close failed")
            _session = None
            _session_warmed_at = 0.0
            _landing_pages_warmed.clear()


# ---------------------------------------------------------------------------
# Per-endpoint cache (in-memory) + snapshot persistence
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 60
"""Per-endpoint in-memory cache lifetime. NSE updates these on the
order of seconds-to-minutes; 60 s coalesces multi-task reads inside
one scan tick without serving stale numbers."""

_cache: dict[tuple[str, str], tuple[float, Any]] = {}
"""``{(endpoint_key, param_repr): (epoch_ts, payload)}``"""


def _cache_get(endpoint_key: str, params: dict | None) -> Any | None:
    key = (endpoint_key, _params_repr(params))
    cached = _cache.get(key)
    if cached is None:
        return None
    ts, payload = cached
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        return None
    return payload


def _cache_set(
    endpoint_key: str, params: dict | None, payload: Any,
    *, persist: bool = True, db_path: str | None = None,
) -> None:
    key = (endpoint_key, _params_repr(params))
    _cache[key] = (time.time(), payload)
    if persist:
        _persist_snapshot(endpoint_key, params, payload, db_path=db_path)


def _params_repr(params: dict | None) -> str:
    """Deterministic string-rep of a params dict for cache keying."""
    if not params:
        return ""
    return "&".join(f"{k}={params[k]}" for k in sorted(params))


def _persist_snapshot(
    endpoint_key: str, params: dict | None, payload: Any,
    db_path: str | None = None,
) -> None:
    """Append the raw response to ``nse_snapshots`` for replay /
    debugging. Best-effort: a DB write failure does NOT propagate to
    the scan loop."""
    path = db_path if db_path is not None else DB_PATH
    ts = datetime.now(IST).isoformat()
    kind = endpoint_key if not params else f"{endpoint_key}:{_params_repr(params)}"
    try:
        body = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return  # un-serialisable; skip
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO nse_snapshots (ts, kind, payload_json) "
                "VALUES (?, ?, ?)",
                (ts, kind, body),
            )
    except sqlite3.OperationalError:
        # nse_snapshots not yet created (test path / very early startup).
        return
    except Exception:
        log.exception("nse_snapshots write failed for %s", kind)


def clear_caches() -> None:
    """Test helper — drop the in-memory cache so the next fetch
    re-runs against the wire / fixture."""
    _cache.clear()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _is_empty_payload(payload: Any) -> bool:
    """NSE's silent-throttle pattern: 200 OK with an empty ``{}`` /
    ``[]`` body when the landing-page cookies weren't set. Distinguish
    that from a legitimate empty response by checking BOTH that the
    payload is a container AND that it has no useful keys."""
    if payload is None:
        return True
    if isinstance(payload, dict):
        return len(payload) == 0
    if isinstance(payload, list):
        return len(payload) == 0
    return False


async def _get_json(
    endpoint_key: str, params: dict | None = None,
) -> Any | None:
    """Cookie-warmed GET with Referer; returns parsed JSON or None.

    Two-tier warmup:
      1. ``_get_session`` does homepage warmup on first use / TTL.
      2. ``_ensure_landing_warm`` GETs the endpoint's landing page
         once per session so NSE's per-page cookies are set before
         the API call (option-chain in particular returns ``{}``
         without this).

    On an empty-payload response we DROP the page from the warmed
    set and retry once — this catches the case where the cookie
    aged out mid-session."""
    cached = _cache_get(endpoint_key, params)
    if cached is not None:
        return cached

    url = ENDPOINTS[endpoint_key]
    referer = REFERERS[endpoint_key]

    payload = await _fetch_once(endpoint_key, url, params, referer)
    if _is_empty_payload(payload):
        # NSE's Akamai layer often returns ``{}`` on the FIRST API
        # call after page-warm, and starts returning real data on
        # the second. The retry path here does NOT re-warm the
        # landing page — that just reissues the same useless cookie
        # set — it just hits the API again after a brief pause to
        # let Akamai propagate the per-request token from the first
        # response.
        log.info(
            "NSE %s returned empty payload; retrying API after settle",
            endpoint_key,
        )
        await asyncio.sleep(0.75)
        payload = await _fetch_once(endpoint_key, url, params, referer)
        if _is_empty_payload(payload):
            # Last resort — drop the landing-warm flag and try ONCE
            # more with a fresh page visit. Some NSE endpoints
            # (notably option-chain) rotate cookies aggressively.
            _landing_pages_warmed.discard(referer)
            await asyncio.sleep(0.5)
            payload = await _fetch_once(endpoint_key, url, params, referer)
        if _is_empty_payload(payload):
            log.warning(
                "NSE %s still empty after retries; giving up", endpoint_key,
            )
            return None
    _cache_set(endpoint_key, params, payload)
    return payload


async def _fetch_once(
    endpoint_key: str, url: str, params: dict | None, referer: str,
) -> Any | None:
    """Single HTTP attempt with landing-page warmup. Separate from
    ``_get_json`` so the retry-on-empty path stays cheap (no cache
    look-up twice)."""
    await _get_session()  # ensures _session is ready
    await _ensure_landing_warm(referer)
    session = _session
    if session is None:
        return None
    try:
        r = await session.get(url, params=params, headers={"Referer": referer})
    except httpx.RequestError as e:
        log.warning("NSE %s error: %s", endpoint_key, e)
        return None
    if r.status_code == 401:
        # Cookies stale — drop session so next call re-warms.
        log.warning("NSE %s 401; dropping session for re-warm", endpoint_key)
        await close_session()
        return None
    if r.status_code != 200:
        log.warning("NSE %s HTTP %d", endpoint_key, r.status_code)
        return None
    ct = r.headers.get("content-type", "").lower()
    if "json" not in ct and not r.text.lstrip().startswith(("{", "[")):
        log.warning("NSE %s non-JSON response", endpoint_key)
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("NSE %s JSON parse failed", endpoint_key)
        return None


# ---------------------------------------------------------------------------
# Pure parsers — exercised by tests with recorded fixtures
# ---------------------------------------------------------------------------

def parse_vix(payload: Any) -> float | None:
    """Pull the INDIA VIX ``last`` value out of an ``/allIndices``
    response. Returns None when the entry is missing — NSE has been
    known to omit VIX during pre-open windows."""
    if not isinstance(payload, dict):
        return None
    for idx in payload.get("data") or []:
        if (idx.get("index") or "").strip().upper() == "INDIA VIX":
            val = idx.get("last")
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def parse_index_quote(payload: Any, index_name: str) -> dict | None:
    """Pull ``{last, change_pct, prev_close}`` for a named index out
    of ``/allIndices``. Names are NSE's canonical labels —
    ``'NIFTY 50'``, ``'NIFTY BANK'``, ``'INDIA VIX'``."""
    if not isinstance(payload, dict):
        return None
    target = index_name.strip().upper()
    for idx in payload.get("data") or []:
        if (idx.get("index") or "").strip().upper() == target:
            try:
                last = float(idx.get("last"))
                pct = float(idx.get("percentChange", idx.get("pChange", 0.0)))
                prev = float(idx.get("previousClose", idx.get("prev", last)))
            except (TypeError, ValueError):
                return None
            return {"last": last, "change_pct": pct, "prev_close": prev}
    return None


def parse_market_status(payload: Any) -> dict[str, str]:
    """Return ``{market: status_string}`` for every ``marketState``
    entry. Caller decides which market to check (Capital Market is
    the cash-equity entry; F&O has its own row)."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for ms in payload.get("marketState") or []:
        market = (ms.get("market") or "").strip()
        status = (ms.get("marketStatus") or "").strip().upper()
        if market and status:
            out[market] = status
    return out


def parse_fii_dii(payload: Any) -> dict | None:
    """Latest FII/DII row from ``/fiidiiTradeReact``.

    Response is a list; each entry has ``category``, ``buyValue``,
    ``sellValue``, ``netValue``, ``date``. We return:
      ``{fii_net_cr, dii_net_cr, date}`` where values are in crores
      (the NSE response is already denominated in crores). Returns
      None when the response is malformed or missing both rows."""
    if not isinstance(payload, list):
        return None
    fii_net = dii_net = None
    date_str = None
    for row in payload:
        if not isinstance(row, dict):
            continue
        cat = (row.get("category") or "").upper()
        try:
            net = float(row.get("netValue"))
        except (TypeError, ValueError):
            continue
        if "FII" in cat or "FPI" in cat:
            fii_net = net
            date_str = row.get("date") or date_str
        elif "DII" in cat:
            dii_net = net
            date_str = row.get("date") or date_str
    if fii_net is None and dii_net is None:
        return None
    return {
        "fii_net_cr": fii_net,
        "dii_net_cr": dii_net,
        "date": date_str,
    }


def parse_option_chain(payload: Any) -> pd.DataFrame:
    """Flatten ``/option-chain-indices`` into a tidy DataFrame.

    Each ``records.data`` row carries the strike + (optionally) CE and
    PE objects with ``openInterest`` and ``impliedVolatility``. The
    response duplicates rows across expiries — we keep every row and
    let the caller filter by ``expiryDate``."""
    if not isinstance(payload, dict):
        return _empty_option_chain()
    records = payload.get("records") or {}
    rows = records.get("data") or []
    out: list[dict] = []
    for row in rows:
        strike = row.get("strikePrice")
        expiry = row.get("expiryDate")
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        out.append({
            "strike":   _safe_float(strike),
            "expiry":   expiry,
            "ce_oi":    _safe_float(ce.get("openInterest")),
            "pe_oi":    _safe_float(pe.get("openInterest")),
            "ce_iv":    _safe_float(ce.get("impliedVolatility")),
            "pe_iv":    _safe_float(pe.get("impliedVolatility")),
            "ce_chng_oi": _safe_float(ce.get("changeinOpenInterest")),
            "pe_chng_oi": _safe_float(pe.get("changeinOpenInterest")),
        })
    if not out:
        return _empty_option_chain()
    return pd.DataFrame(out)


def parse_pcr(payload: Any, expiry: str | None = None) -> float | None:
    """Put-Call Ratio = sum(PE OI) / sum(CE OI).

    When ``expiry`` is None, use the nearest expiry (first one in the
    ``records.expiryDates`` list). Returns None on missing/zero CE OI
    (PCR is undefined there)."""
    df = parse_option_chain(payload)
    if df.empty:
        return None
    if expiry is None and isinstance(payload, dict):
        expiries = (payload.get("records") or {}).get("expiryDates") or []
        if expiries:
            expiry = expiries[0]
    if expiry:
        df = df[df["expiry"] == expiry]
        if df.empty:
            return None
    ce_total = float(df["ce_oi"].fillna(0).sum())
    pe_total = float(df["pe_oi"].fillna(0).sum())
    if ce_total <= 0:
        return None
    return pe_total / ce_total


def parse_fno_ban(payload: Any) -> set[str]:
    """Extract banned-symbol set from ``/snapshot-derivatives-equity``.

    NSE's response shape has shifted historically; we look in a few
    plausible locations (``data.fnoBan``, ``fnoBan``, top-level
    ``data``) and accept either a list of strings or a list of
    ``{symbol: ...}`` objects."""
    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key in ("data", "fnoBan"):
            inner = payload.get(key)
            if inner is None:
                continue
            if isinstance(inner, dict) and "fnoBan" in inner:
                candidates.append(inner["fnoBan"])
            elif isinstance(inner, list):
                candidates.append(inner)
    out: set[str] = set()
    for cand in candidates:
        if not isinstance(cand, list):
            continue
        for entry in cand:
            if isinstance(entry, str):
                out.add(entry.strip().upper())
            elif isinstance(entry, dict):
                sym = entry.get("symbol") or entry.get("Symbol")
                if isinstance(sym, str):
                    out.add(sym.strip().upper())
    return out


def _empty_option_chain() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["strike", "expiry", "ce_oi", "pe_oi", "ce_iv", "pe_iv",
                 "ce_chng_oi", "pe_chng_oi"],
    )


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public async fetchers
# ---------------------------------------------------------------------------

async def fetch_market_status() -> dict[str, str]:
    """``{market_name: status}`` map. Empty dict on failure."""
    payload = await _get_json("market_status")
    return parse_market_status(payload)


async def fetch_vix() -> float | None:
    payload = await _get_json("all_indices")
    return parse_vix(payload)


async def fetch_nifty_quote() -> dict | None:
    payload = await _get_json("all_indices")
    return parse_index_quote(payload, "NIFTY 50")


async def fetch_banknifty_quote() -> dict | None:
    payload = await _get_json("all_indices")
    return parse_index_quote(payload, "NIFTY BANK")


async def fetch_fii_dii() -> dict | None:
    payload = await _get_json("fii_dii")
    return parse_fii_dii(payload)


async def _option_chain_payload(symbol: str) -> Any | None:
    """NSE-first, Fyers-fallback option-chain fetch.

    NSE blocks ``/api/option-chain-indices`` for non-browser clients
    (silent ``{}`` throttle even with full Akamai cookie warmup).
    When the NSE attempt returns empty/None we fall through to the
    Fyers SDK, which works reliably as long as the daily access
    token is cached. Both attempts share the same in-memory cache
    so the second-best path doesn't re-fetch on every call inside a
    scan tick."""
    cached = _cache_get("option_chain", {"symbol": symbol})
    if cached is not None:
        return cached

    payload = await _get_json("option_chain", params={"symbol": symbol})
    if not _is_empty_payload(payload) and isinstance(payload, dict):
        rows = (payload.get("records") or {}).get("data") or []
        if rows:
            return payload

    # NSE failed or empty — try Fyers.
    try:
        from fyers_client.options import fetch_option_chain_nse_shape
    except ImportError:
        log.debug("Fyers options module unavailable; no fallback")
        return payload
    try:
        fy_payload = await fetch_option_chain_nse_shape(symbol)
    except Exception:
        log.exception("Fyers option-chain fallback failed for %s", symbol)
        return payload
    if fy_payload is None:
        return payload
    rows = (fy_payload.get("records") or {}).get("data") or []
    if not rows:
        log.info("Fyers option-chain returned no rows for %s", symbol)
        return fy_payload
    # Cache the Fyers result under the SAME key as NSE so PCR /
    # options consumers see one source of truth for this tick.
    _cache_set("option_chain", {"symbol": symbol}, fy_payload)
    log.info("Option chain served by Fyers fallback for %s (%d rows)",
             symbol, len(rows))
    return fy_payload


async def fetch_options_chain(symbol: str = "NIFTY") -> pd.DataFrame:
    payload = await _option_chain_payload(symbol)
    return parse_option_chain(payload)


async def fetch_pcr(symbol: str = "NIFTY", expiry: str | None = None) -> float | None:
    payload = await _option_chain_payload(symbol)
    return parse_pcr(payload, expiry=expiry)


async def fetch_fno_ban_set() -> set[str]:
    """NSE-API path to the ban list. NOTE: ``data.fno_ban`` reads
    the archive CSV via a different route and is the production
    source — this function exists for parity with the Phase 9 spec
    and for tests that want to verify the snapshot-derivatives-equity
    endpoint contract."""
    payload = await _get_json("fno_ban", params={"index": "fnoBan"})
    return parse_fno_ban(payload)
