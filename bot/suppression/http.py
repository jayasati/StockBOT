"""NSE HTTP client with cookie warmup.

NSE blocks any request that doesn't first hit the homepage to set
session cookies (nseappid, bm_sv, etc.). The pattern is:
  1. GET /                  (homepage)
  2. GET /reports/asm       (the page the API call would come from)
  3. GET /api/...           with Referer pointing back to the page in step 2

Without this dance NSE returns 401/403 (and sometimes a 200 with HTML
instead of JSON). We retry across multiple known endpoint names since
NSE has renamed them in the past (reportSurveillance → reportASM/reportGSM)."""
from __future__ import annotations

import logging

import httpx

log = logging.getLogger("alertbot.suppression")

NSE_HOME = "https://www.nseindia.com/"
NSE_ASM_PAGE = "https://www.nseindia.com/reports/asm"
NSE_GSM_PAGE = "https://www.nseindia.com/reports/gsm"

NSE_ASM_API_CANDIDATES = ["https://www.nseindia.com/api/reportASM"]
NSE_GSM_API_CANDIDATES = ["https://www.nseindia.com/api/reportGSM"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


async def _nse_session() -> httpx.AsyncClient:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        # Skip brotli — would need a separate package; gzip is fine.
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    client = httpx.AsyncClient(
        headers=headers, timeout=20, follow_redirects=True
    )
    # Cookie warmup — NSE refuses API calls without these.
    try:
        await client.get(NSE_HOME)
        await client.get(NSE_ASM_PAGE)
    except httpx.RequestError as e:
        log.warning("NSE warmup failed: %s", e)
    return client


async def _fetch_json(
    client: httpx.AsyncClient, url: str, referer: str
) -> dict | list | None:
    try:
        r = await client.get(url, headers={"Referer": referer})
    except httpx.RequestError as e:
        log.warning("NSE %s fetch error: %s", url, e)
        return None
    if r.status_code != 200:
        log.warning("NSE %s returned HTTP %d", url, r.status_code)
        return None
    ct = r.headers.get("content-type", "")
    if "json" not in ct.lower() and not r.text.lstrip().startswith(("{", "[")):
        log.warning("NSE %s returned non-JSON (content-type=%s)", url, ct)
        return None
    try:
        return r.json()
    except ValueError as e:
        log.warning("NSE %s JSON parse failed: %s", url, e)
        return None


async def _fetch_first_working(
    client: httpx.AsyncClient, urls: list[str], referer: str
) -> dict | list | None:
    for url in urls:
        data = await _fetch_json(client, url, referer)
        if data is not None:
            log.info("NSE endpoint OK: %s", url)
            return data
    return None
