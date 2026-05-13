"""Map a news headline / body to one or more Nifty-500 tickers.

We reuse ``data.filings.matcher.NAME_TO_TICKER`` (built from
``ind_nifty500list.csv``) as the canonical company-name → ticker
table, but extend it three ways:

  1. **Substring search** anywhere in the text — the filings matcher
     does longest-prefix from the start of the title, which is fine
     for "Reliance Industries: Q4 results"-style filings but misses
     "...as Reliance posts record profit" buried mid-sentence.

  2. **Curated alias dict** — short forms / common nicknames that the
     CSV doesn't carry ("RIL" → RELIANCE, "Maruti" → MARUTI, etc.).
     Expand this dict on a hit-by-hit basis; misses just mean the
     headline gets dropped from that symbol's news, which is the
     safe failure mode.

  3. **Relevance score** — a float in (0, 1] reflecting confidence
     that the headline is genuinely about this symbol:
       1.0  full company name matched
       0.7  ticker or alias matched
       0.4  ambiguous short-form ("Tata" with multiple candidates)
     The scorer downweights low-relevance items in the aggregate.

This module is *intentionally* heuristic — a perfect entity resolver
would use a finance-domain NER model, which is out of scope. The
file format (regex + dict) means a regression is one-line fixable."""
from __future__ import annotations

import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Curated alias table — expand as misses surface in real headlines
# ---------------------------------------------------------------------------

# Map alias (case-insensitive, used as whole-word in headline) → ticker
# WITHOUT the ``.NS`` suffix. The matcher adds ``.NS`` before returning
# so the format matches paper_trades.symbol everywhere downstream.
CURATED_ALIASES: dict[str, str] = {
    "RIL":         "RELIANCE",
    "RELIANCE":    "RELIANCE",
    "TCS":         "TCS",
    "INFY":        "INFY",
    "INFOSYS":     "INFY",
    "HDFC BANK":   "HDFCBANK",
    "HDFCBANK":    "HDFCBANK",
    "ICICI BANK":  "ICICIBANK",
    "ICICIBANK":   "ICICIBANK",
    "SBI":         "SBIN",
    "SBIN":        "SBIN",
    "AXIS BANK":   "AXISBANK",
    "KOTAK BANK":  "KOTAKBANK",
    "BAJAJ FINANCE": "BAJFINANCE",
    "BAJ FIN":     "BAJFINANCE",
    "MARUTI":      "MARUTI",
    "M&M":         "M&M",
    "MAHINDRA":    "M&M",
    "LARSEN":      "LT",
    "L&T":         "LT",
    "WIPRO":       "WIPRO",
    "HCL":         "HCLTECH",
    "HCLTECH":     "HCLTECH",
    "TECH MAHINDRA": "TECHM",
    "TATA STEEL":  "TATASTEEL",
    "TATA MOTORS": "TATAMOTORS",
    "TATA POWER":  "TATAPOWER",
    "ITC":         "ITC",
    "HUL":         "HINDUNILVR",
    "HINDUSTAN UNILEVER": "HINDUNILVR",
    "BHARTI AIRTEL": "BHARTIARTL",
    "AIRTEL":      "BHARTIARTL",
    "ONGC":        "ONGC",
    "NTPC":        "NTPC",
    "COAL INDIA":  "COALINDIA",
    "POWERGRID":   "POWERGRID",
    "ADANI":       "ADANIENT",   # ambiguous; relevance penalised below
    "ADANI ENTERPRISES": "ADANIENT",
    "ADANI PORTS": "ADANIPORTS",
    "ADANI GREEN": "ADANIGREEN",
    "ADANI POWER": "ADANIPOWER",
    "ASIAN PAINTS": "ASIANPAINT",
    "SUN PHARMA":  "SUNPHARMA",
    "DR REDDY":    "DRREDDY",
    "DRREDDY":     "DRREDDY",
    "BAJAJ AUTO":  "BAJAJ-AUTO",
    "HERO MOTOCORP": "HEROMOTOCO",
    "MOTHERSON":   "MOTHERSON",
    "PVR":         "PVRINOX",
    "INOX":        "PVRINOX",
    "DIXON":       "DIXON",
}

# Aliases that are GENERIC enough to need penalisation — they're
# common-noun-like or multi-company prefixes. Hit → relevance 0.4
# rather than the default 0.7.
AMBIGUOUS_ALIASES: frozenset[str] = frozenset({
    # Bare prefixes that could refer to multiple Tata/Adani/Bajaj
    # group companies. Full names like "Reliance Industries" or
    # "Adani Ports" are specific enough to NOT live here.
    "TATA", "ADANI", "BAJAJ",
})


def _word_boundary_search(needle: str, haystack: str) -> bool:
    """Whole-word case-insensitive substring search. Required to
    avoid ``'ITC'`` matching ``'switch'`` or ``'TCS'`` matching
    ``'practice'``."""
    pattern = r"\b" + re.escape(needle) + r"\b"
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


def match_symbols(
    text: str,
    watchlist: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Return ``[(ticker, relevance), ...]`` for every symbol in
    ``watchlist`` that the headline plausibly mentions.

    ``watchlist`` accepts ticker-with-``.NS`` form (the production
    convention); pass ``None`` to fall back to the full Nifty 500
    map from ``data.filings.matcher.NAME_TO_TICKER``.

    Two layers of matching:

      Layer A: full company name from the CSV (e.g. "Reliance Industries
      Limited" → normalised "RELIANCE INDUSTRIES"). Relevance 1.0.

      Layer B: curated alias / ticker token (e.g. "RIL", "TCS").
      Relevance 0.7, dropped to 0.4 for items in
      ``AMBIGUOUS_ALIASES``.

      Layer C: bare ticker token (RELIANCE, INFY, …) for symbols
      whose tickers are also plausible English nouns. Same scoring
      as B."""
    if not text:
        return []
    # Build the universe of candidate tickers to consider.
    allowed: set[str] | None
    if watchlist is None:
        allowed = None
    else:
        allowed = {s.replace(".NS", "").upper() for s in watchlist}

    matches: dict[str, float] = {}

    # ---- Layer A: company-name longest-match over the full text ----
    from data.filings.matcher import NAME_TO_TICKER, _normalize
    norm_text = _normalize(text)
    # Iterate name → ticker; require the FULL name as a whole-word match
    # in the normalised text.
    for name, ticker_ns in NAME_TO_TICKER.items():
        if not _word_boundary_search(name, norm_text):
            continue
        ticker = ticker_ns.replace(".NS", "").upper()
        if allowed is not None and ticker not in allowed:
            continue
        rel = 0.4 if name in AMBIGUOUS_ALIASES else 1.0
        matches[ticker] = max(matches.get(ticker, 0.0), rel)

    # ---- Layer B+C: alias dict, with ambiguous penalty ----
    for alias, ticker in CURATED_ALIASES.items():
        if allowed is not None and ticker.upper() not in allowed:
            continue
        if not _word_boundary_search(alias, text):
            continue
        rel = 0.4 if alias.upper() in AMBIGUOUS_ALIASES else 0.7
        # Don't downgrade a Layer-A 1.0 match.
        if matches.get(ticker.upper(), 0.0) < rel:
            matches[ticker.upper()] = rel

    return [(t + ".NS", r) for t, r in sorted(matches.items())]
