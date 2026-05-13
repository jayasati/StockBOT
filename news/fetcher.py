"""RSS + NewsAPI fetchers, normalised to ``NewsItem`` for the scorer.

Sources (no API key required):
  * Economic Times — Markets
  * Moneycontrol — Market reports + Business
  * Livemint — Markets

NewsAPI is optional (set ``NEWSAPI_KEY`` in .env to enable). Without
it, we still get plenty of headlines from the RSS sources.

Why not reuse ``data.filings.poll``: that module already classifies
NSE/BSE filings with a regex taxonomy + writes ``filings_seen``.
News-pipeline items are different in shape (general-purpose
headlines about the market and its constituents, not corporate
filings) and need to flow through FinBERT for sentiment. They live
in their own ``news_items`` table so the analytics surface stays
clean.

Failure handling: per-source errors are absorbed. One dead RSS feed
does not stop the others from contributing. ``fetch_all`` always
returns a list, possibly empty."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("alertbot.news.fetcher")

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

RSS_SOURCES: dict[str, str] = {
    "economic_times":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol_markets":  "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "livemint_markets":      "https://www.livemint.com/rss/markets",
}

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
NEWSAPI_QUERY = "NSE OR Sensex OR Nifty OR \"Indian stock market\""
"""Free-tier-friendly query — broad enough to catch most Indian
equity headlines, narrow enough to keep the 100/day quota usable."""


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# NewsItem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsItem:
    """One unit of news, normalised across RSS / NewsAPI / filings.

    ``id`` is a content hash (sha1 of source+url+title) — deterministic
    so duplicate fetches across runs are idempotent at the DB layer.
    ``body`` may be empty when the source only exposes a headline."""
    id: str
    source: str
    title: str
    body: str
    url: str
    published_at: str  # ISO 8601 (UTC) when known, '' otherwise

    @classmethod
    def make(
        cls,
        *,
        source: str,
        title: str,
        body: str,
        url: str,
        published_at: str,
    ) -> "NewsItem":
        digest = hashlib.sha1(
            f"{source}|{url}|{title}".encode("utf-8", errors="ignore"),
        ).hexdigest()
        return cls(
            id=digest, source=source, title=(title or "").strip(),
            body=(body or "").strip(), url=(url or "").strip(),
            published_at=published_at or "",
        )


# ---------------------------------------------------------------------------
# RSS fetcher
# ---------------------------------------------------------------------------

async def _fetch_rss_one(client: httpx.AsyncClient, source: str, url: str) -> list[NewsItem]:
    """Fetch one feed and parse it. Returns ``[]`` on any failure
    so the orchestrator can keep going."""
    try:
        r = await client.get(url, timeout=15.0)
    except httpx.RequestError as e:
        log.warning("RSS %s fetch error: %s", source, e)
        return []
    if r.status_code != 200:
        log.warning("RSS %s HTTP %d", source, r.status_code)
        return []
    text = r.text

    # feedparser is sync — push the parse to a worker so we don't
    # block the loop on big feeds.
    try:
        feed = await asyncio.to_thread(_parse_feed, text)
    except Exception:
        log.exception("RSS %s parse failed", source)
        return []
    return [_entry_to_item(source, e) for e in feed.entries]


def _parse_feed(text: str) -> Any:
    """Sync feedparser entry point. Wrapped in ``to_thread`` by the
    async caller; exposed as a module-level function so tests can
    monkey-patch it without going through the network."""
    import feedparser
    return feedparser.parse(text)


def _entry_to_item(source: str, entry: Any) -> NewsItem:
    """One feedparser entry → NewsItem. Tolerant of every shape RSS
    feeds offer — title may be missing, published may be a struct or
    a string, summary may live under ``description``."""
    title = (
        getattr(entry, "title", None)
        or entry.get("title", "") if hasattr(entry, "get") else ""
    )
    summary = (
        getattr(entry, "summary", None)
        or (entry.get("summary", "") if hasattr(entry, "get") else "")
        or getattr(entry, "description", "")
    )
    url = (
        getattr(entry, "link", None)
        or (entry.get("link", "") if hasattr(entry, "get") else "")
    )
    published_at = _entry_publish_iso(entry)
    return NewsItem.make(
        source=source, title=title or "", body=summary or "",
        url=url or "", published_at=published_at,
    )


def _entry_publish_iso(entry: Any) -> str:
    """Pull a published-at timestamp out of a feedparser entry.
    Returns ISO-8601 UTC string or '' if the feed didn't expose one."""
    # feedparser parses RFC822 / ISO into entry.published_parsed (a
    # time.struct_time in UTC).
    pub_struct = getattr(entry, "published_parsed", None)
    if pub_struct is None and hasattr(entry, "get"):
        pub_struct = entry.get("published_parsed")
    if pub_struct is not None:
        try:
            import time as _time
            dt = datetime.fromtimestamp(_time.mktime(pub_struct), tz=timezone.utc)
            return dt.isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    raw = getattr(entry, "published", None) or (
        entry.get("published", "") if hasattr(entry, "get") else ""
    )
    return raw or ""


# ---------------------------------------------------------------------------
# NewsAPI fetcher (optional)
# ---------------------------------------------------------------------------

async def _fetch_newsapi(client: httpx.AsyncClient) -> list[NewsItem]:
    """Pull headlines from newsapi.org when ``NEWSAPI_KEY`` is set.

    Free tier: 100 requests/day. The scorer runs every 10 min during
    market hours = ~38 calls/day, well within the budget."""
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        return []
    params = {
        "q": NEWSAPI_QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": "50",
        "apiKey": api_key,
    }
    try:
        r = await client.get(NEWSAPI_BASE, params=params, timeout=15.0)
    except httpx.RequestError as e:
        log.warning("NewsAPI error: %s", e)
        return []
    if r.status_code != 200:
        log.warning("NewsAPI HTTP %d", r.status_code)
        return []
    try:
        payload = r.json()
    except ValueError:
        return []
    if payload.get("status") != "ok":
        return []
    out: list[NewsItem] = []
    for art in payload.get("articles") or []:
        out.append(NewsItem.make(
            source="newsapi",
            title=art.get("title") or "",
            body=art.get("description") or "",
            url=art.get("url") or "",
            published_at=art.get("publishedAt") or "",
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def fetch_all() -> list[NewsItem]:
    """Fan out to every configured RSS source + NewsAPI in parallel.

    Deduplication: items sharing an ``id`` (sha1 of source+url+title)
    are kept once. Cross-source duplicates (the same headline syndicated
    by ET and Moneycontrol) survive both because the source differs —
    that's fine, FinBERT will score them the same and the aggregator
    just gets a more confident sample."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True,
    ) as client:
        rss_tasks = [
            _fetch_rss_one(client, name, url)
            for name, url in RSS_SOURCES.items()
        ]
        newsapi_task = _fetch_newsapi(client)
        chunks = await asyncio.gather(
            *rss_tasks, newsapi_task, return_exceptions=True,
        )

    seen: dict[str, NewsItem] = {}
    for chunk in chunks:
        if isinstance(chunk, BaseException):
            log.warning("News source raised: %s", chunk)
            continue
        for item in chunk:
            if not item.title:
                continue
            if item.id not in seen:
                seen[item.id] = item
    log.info("News fetch: %d unique items across %d sources",
             len(seen), len(RSS_SOURCES) + 1)
    return list(seen.values())
