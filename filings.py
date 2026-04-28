"""
BSE corporate filings poller.

Polls the BSE corporate-announcements JSON API
(api.bseindia.com/BseIndiaAPI/api/AnnGetData/w) each scan, classifies
items into binary_high / binary_med / fluff, and persists what we've
seen so we don't re-process. Exposes:

  poll_filings()          fetch + classify + persist; return new entries
                          for symbols in our watchlist mapping
  recent_high_priority()  {symbol: title} for binary_high filings in the
                          last N minutes (used by the scorer to add a
                          fundamental catalyst bonus)
"""

import csv
import html
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger("alertbot.filings")

BSE_ANN_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
BSE_HOME_URL = "https://www.bseindia.com/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DB_PATH = Path("alerts.db")
IST = ZoneInfo("Asia/Kolkata")


SCHEMA = """
CREATE TABLE IF NOT EXISTS filings_seen (
    filing_id      TEXT PRIMARY KEY,
    symbol         TEXT NOT NULL,
    title          TEXT NOT NULL,
    classification TEXT NOT NULL,
    seen_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filings_sym_class_time
    ON filings_seen(symbol, classification, seen_at);
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


# ============================================================================
# BSE company name → NSE ticker mapping (built from NIFTY 500 CSV)
# ============================================================================
# Keys are pre-normalized: uppercase, '&' → 'AND', punctuation stripped,
# trailing LIMITED/LTD removed. The matcher applies the same normalization
# to incoming filing titles, then does longest-prefix lookup.
#
# To change the universe, replace ind_nifty500list.csv (download from:
# https://archives.nseindia.com/content/indices/ind_nifty500list.csv).
# ============================================================================

WATCHLIST_CSV = Path("ind_nifty500list.csv")


def _normalize(s: str) -> str:
    """Uppercase, '&' → 'AND', strip punctuation, drop LIMITED/LTD, collapse spaces."""
    s = html.unescape(s).upper().replace("&", " AND ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(LIMITED|LTD)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_name_to_ticker() -> dict[str, str]:
    """Build {normalized_company_name: ticker.NS} from the NIFTY 500 CSV."""
    if not WATCHLIST_CSV.exists():
        log.warning(
            "Watchlist CSV not found at %s — filings matcher will be empty",
            WATCHLIST_CSV,
        )
        return {}
    mapping: dict[str, str] = {}
    with WATCHLIST_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            company = (row.get("Company Name") or "").strip()
            symbol = (row.get("Symbol") or "").strip()
            if not (company and symbol):
                continue
            key = _normalize(company)
            if key:
                mapping[key] = f"{symbol}.NS"
    return mapping


NAME_TO_TICKER: dict[str, str] = _load_name_to_ticker()


_SORTED_NAMES = sorted(NAME_TO_TICKER.keys(), key=len, reverse=True)


def match_ticker(title: str) -> str | None:
    """Return the NSE ticker if the title starts with a known company name."""
    norm = _normalize(title)
    for name in _SORTED_NAMES:
        if norm == name or norm.startswith(name + " "):
            return NAME_TO_TICKER[name]
    return None


# ============================================================================
# Classification
# ============================================================================

_BINARY_HIGH = [
    re.compile(r"\b(quarterly|annual|financial|unaudited|audited)\s+results?\b", re.I),
    re.compile(r"\bq[1-4]\s*(fy)?[-\s]?\d{2,4}\b", re.I),
    re.compile(r"\bearnings?\b", re.I),
    re.compile(r"\bPLI\b"),
    re.compile(r"\bproduction[-\s]linked\s+incentive\b", re.I),
    re.compile(
        r"\b(order|contract|tender|loi|letter\s+of\s+intent|work\s+order)s?\s+"
        r"(win|won|received|secured|booked|wins?|bagged)\b",
        re.I,
    ),
    re.compile(
        r"\b(receipt|received|secured|won|bagged|bags|wins?|secures?)\s+"
        r"(of\s+)?(an?\s+|the\s+)?(order|contract|loi|work\s+order|tender)s?\b",
        re.I,
    ),
    re.compile(r"\bacquisition\b", re.I),
    re.compile(r"\bacquir(ed|ing|es)\b", re.I),
    re.compile(r"\bmerger\b", re.I),
    re.compile(r"\bdemerger\b", re.I),
    re.compile(r"\bscheme\s+of\s+arrangement\b", re.I),
    re.compile(r"\ballotment\b", re.I),
    re.compile(r"\bdividend\b", re.I),
]

_BINARY_MED = [
    re.compile(r"\bboard\s+meeting\b", re.I),
    re.compile(r"\bregulatory\s+approval\b", re.I),
    re.compile(
        r"\b(usfda|us\s*fda|cdsco|rbi|sebi|cci|ministry)\s+"
        r"(approval|nod|clearance)\b",
        re.I,
    ),
    re.compile(
        r"\bapproval\s+(from|by)\s+(usfda|us\s*fda|cdsco|rbi|sebi|cci)\b",
        re.I,
    ),
]


def classify(title: str) -> str:
    for pat in _BINARY_HIGH:
        if pat.search(title):
            return "binary_high"
    for pat in _BINARY_MED:
        if pat.search(title):
            return "binary_med"
    return "fluff"


# ============================================================================
# Polling
# ============================================================================

def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _filing_id(item: dict) -> str | None:
    """Stable id for a BSE announcement.

    NEWSID is the canonical identifier when present. Fall back to a
    composite key so we can still de-dupe even on schema drift.
    """
    nid = (item.get("NEWSID") or "").strip()
    if nid:
        return nid
    parts = [
        str(item.get("SCRIP_CD") or ""),
        str(item.get("DT_TM") or item.get("NEWS_DT") or ""),
        (item.get("NEWSSUB") or item.get("HEADLINE") or "").strip()[:80],
    ]
    composite = "|".join(parts).strip("|")
    return composite or None


def _item_dt_iso(item: dict) -> str:
    """Parse the announcement timestamp; fall back to now() if unparseable."""
    raw = (item.get("DT_TM") or item.get("NEWS_DT") or "").strip()
    if raw:
        # Trim trailing fractional seconds and timezone markers feedparser-style
        # since fromisoformat in older Pythons rejects them.
        candidate = raw.rstrip("Z")
        try:
            return datetime.fromisoformat(candidate).isoformat()
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d %b %Y %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt).isoformat()
            except ValueError:
                continue
    return datetime.now().isoformat()


def _item_link(item: dict) -> str:
    """Build the PDF/attachment link for an announcement."""
    attach = (item.get("ATTACHMENTNAME") or "").strip()
    if attach:
        return f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attach}"
    return (item.get("NSURL") or "").strip()


def _item_title(item: dict) -> str:
    """Combine the company name with the filing subject for matching + display.

    BSE's NEWSSUB usually arrives as "<Company> - <ScripCode> - Announcement
    under Regulation 30 (LODR)-<topic>". We strip those boilerplate parts
    so the title reads as the actual filing topic.
    """
    company = _strip_html(item.get("SLONGNAME") or "").strip()
    subject = _strip_html(item.get("NEWSSUB") or item.get("HEADLINE") or "").strip()
    scrip = str(item.get("SCRIP_CD") or "").strip()

    if subject and company and subject.lower().startswith(company.lower()):
        subject = subject[len(company):].lstrip(" -")
    if scrip and subject.startswith(scrip):
        subject = subject[len(scrip):].lstrip(" -")
    subject = re.sub(
        r"^Announcement under Regulation \d+\s*\(LODR\)\s*-?\s*",
        "", subject, flags=re.I,
    ).strip()

    if not subject:
        return company
    if not company:
        return subject
    return f"{company} - {subject}"


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


def _existing_ids() -> set[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT filing_id FROM filings_seen").fetchall()
    return {r[0] for r in rows}


async def poll_filings() -> list[tuple[str, str, str, str]]:
    """Fetch BSE announcements and return new (symbol, classification, title, link).

    On the first poll (empty DB), entries are seeded as 'seen' but NOT
    returned — we don't want to flood Telegram with stale filings as
    fundamental catalysts on bot startup.
    """
    items = await _fetch_announcements()
    if items is None:
        return []
    if not items:
        log.info("Filings poll: 0 announcements returned")
        return []

    seen_ids = _existing_ids()
    is_first_poll = len(seen_ids) == 0

    new: list[tuple[str, str, str, str]] = []
    matched = 0
    unmapped = 0

    with sqlite3.connect(DB_PATH) as conn:
        for item in items:
            fid = _filing_id(item)
            if not fid or fid in seen_ids:
                continue
            title = _item_title(item)
            if not title:
                continue
            symbol = match_ticker(title)
            if symbol is None:
                unmapped += 1
                continue
            matched += 1
            classification = classify(title)
            link = _item_link(item)
            conn.execute(
                "INSERT OR IGNORE INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (fid, symbol, title, classification, _item_dt_iso(item)),
            )
            if not is_first_poll:
                new.append((symbol, classification, title, link))

    if unmapped:
        log.warning(
            "Filings poll: %d entries with company names not in mapping (skipped)",
            unmapped,
        )
    if is_first_poll and matched:
        log.info(
            "Filings poll: seeded %d existing watchlist entries on first run "
            "(no alerts on this pass)",
            matched,
        )
    else:
        log.info(
            "Filings poll: %d entries, %d matched watchlist, %d new",
            len(items), matched, len(new),
        )
    return new


def recent_high_priority(minutes: int) -> dict[str, str]:
    """Return {symbol: title} of the most recent binary_high filings in the window."""
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT symbol, title FROM filings_seen "
            "WHERE classification = 'binary_high' AND seen_at > ? "
            "ORDER BY seen_at DESC",
            (cutoff,),
        ).fetchall()
    result: dict[str, str] = {}
    for sym, title in rows:
        if sym not in result:
            result[sym] = title
    return result




