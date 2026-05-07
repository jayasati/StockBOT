"""Filing-title classifier: binary_high / binary_med / fluff.

binary_high — earnings, big orders, M&A, dividend, allotment (drives several ATRs of move)
binary_med  — board meetings, regulatory approvals (smaller expected impact)
fluff       — everything else (no scorer bonus)
"""
from __future__ import annotations

import re

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
