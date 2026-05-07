"""BSE corporate-announcements API client."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger("alertbot.filings")

IST = ZoneInfo("Asia/Kolkata")

BSE_ANN_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
BSE_HOME_URL = "https://www.bseindia.com/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


async def _fetch_announcements() -> list[dict] | None:
    """Fetch today's announcements from the BSE JSON API.

    The endpoint requires a Referer/Origin from bseindia.com; we also do
    a homepage GET first to pick up any session cookies the WAF expects.
    """
    today = datetime.now(IST).strftime("%Y%m%d")
    params = {
        "pageno": 1,
        "strCat": -1,
        "strPrevDate": today,
        "strScrip": "",
        "strSearch": "P",
        "strToDate": today,
        "strType": "C",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
        "Origin": "https://www.bseindia.com",
    }
    try:
        async with httpx.AsyncClient(
            timeout=20, headers=headers, follow_redirects=True
        ) as client:
            # Warm up cookies from the public site before hitting the API.
            try:
                await client.get(BSE_HOME_URL)
            except httpx.RequestError:
                pass

            r = await client.get(BSE_ANN_API_URL, params=params)
            if r.status_code != 200:
                log.warning("BSE API returned HTTP %d", r.status_code)
                return None
            try:
                data = r.json()
            except ValueError as e:
                log.warning("BSE API returned non-JSON: %s", e)
                return None
            return data.get("Table") or []
    except httpx.RequestError as e:
        log.warning("BSE API fetch failed: %s", e)
        return None
