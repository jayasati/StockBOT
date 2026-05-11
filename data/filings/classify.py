"""Filing-title classifier: binary_high / event_unknown / binary_med / fluff.

binary_high     — directionally CLEAR positive events. Order wins, dividends,
                  PLI awards, buybacks, bonus issues. The composite scorer adds
                  a +30 bonus for these because we know the price reaction
                  ought to be positive.

event_unknown   — events whose direction depends on the body text we haven't
                  read yet. Earnings results (could be beat or miss), M&A
                  (acquirer often falls, target often rises), allotments
                  (bonus-issue vs ESOP), schemes of arrangement. The scorer
                  surfaces these as an informational tag but does NOT bias
                  the score — without reading the PDF we'd just be guessing.

binary_med      — board meetings, regulatory approvals (smaller expected
                  impact even when the direction is clear).

fluff           — everything else (no scorer bonus, no alert tag).

Why event_unknown exists: PVRINOX 2026-05-11 published "Audited Financial
Results" → old classifier flagged it binary_high → +30 bonus pushed the score
to 90 right as the stock was peaking. Results turned out weak; stock dumped
4%. Granting positive bias on the *event* without knowing the *outcome* is a
known fail mode. Until we read the PDF, treat these as informational."""
from __future__ import annotations

import re

# Directionally clear positives — keep these for the +30 score bonus.
_BINARY_HIGH = [
    # PLI / production-linked incentive
    re.compile(r"\bPLI\b"),
    re.compile(r"\bproduction[-\s]linked\s+incentive\b", re.I),
    # Order / contract / tender wins
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
    # Capital-return events
    re.compile(r"\bdividend\b", re.I),
    re.compile(r"\bbuy[-\s]?back\b", re.I),
    re.compile(r"\bbonus\s+(issue|share)s?\b", re.I),
]

# Direction unknown without reading the body. NO score bonus — surfaced
# only as an informational tag so the trader knows an event landed.
_EVENT_UNKNOWN = [
    # Earnings — could be beat or miss
    re.compile(r"\b(quarterly|annual|financial|unaudited|audited)\s+results?\b", re.I),
    re.compile(r"\bq[1-4]\s*(fy)?[-\s]?\d{2,4}\b", re.I),
    re.compile(r"\bearnings?\b", re.I),
    # M&A — acquirer vs target react oppositely; price depends on terms
    re.compile(r"\bacquisition\b", re.I),
    re.compile(r"\bacquir(ed|ing|es)\b", re.I),
    re.compile(r"\bmerger\b", re.I),
    re.compile(r"\bdemerger\b", re.I),
    re.compile(r"\bscheme\s+of\s+arrangement\b", re.I),
    # Allotment — could be bonus shares (positive) or ESOP (neutral)
    re.compile(r"\ballotment\b", re.I),
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
    """Order matters: directionally positive wins over event-unknown, which
    wins over binary_med. A title with both ``dividend`` and ``results`` in
    it (e.g. "Q4 results — dividend declared") gets ``binary_high`` because
    the dividend is the actionable positive signal."""
    for pat in _BINARY_HIGH:
        if pat.search(title):
            return "binary_high"
    for pat in _EVENT_UNKNOWN:
        if pat.search(title):
            return "event_unknown"
    for pat in _BINARY_MED:
        if pat.search(title):
            return "binary_med"
    return "fluff"
