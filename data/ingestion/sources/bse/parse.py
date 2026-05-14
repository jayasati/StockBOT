"""BSE-announcement JSON → clean fields (id, timestamp, link, title)."""
from __future__ import annotations

import html
import re
from datetime import datetime


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
