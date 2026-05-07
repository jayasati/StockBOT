"""BSE corporate filings poller.

Polls the BSE corporate-announcements JSON API
(api.bseindia.com/BseIndiaAPI/api/AnnGetData/w) each scan, classifies
items into binary_high / binary_med / fluff, and persists what we've
seen so we don't re-process.

Public surface:

  poll_filings()          fetch + classify + persist; return new entries
                          for symbols in our watchlist mapping
  recent_high_priority()  {symbol: title} for binary_high filings in the
                          last N minutes (used by the scorer to add a
                          fundamental catalyst bonus)
  classify(title)         str → 'binary_high' | 'binary_med' | 'fluff'
  match_ticker(title)     BSE company-name title → NSE ticker (or None)

Diagnostic CLI: ``python -m data.filings check`` prints the current BSE
returns and which company names match the watchlist.
"""
from .classify import classify
from .http import (
    BSE_ANN_API_URL,
    BSE_HOME_URL,
    USER_AGENT,
    _fetch_announcements,
)
from .matcher import NAME_TO_TICKER, WATCHLIST_CSV, match_ticker
from .parse import (
    _filing_id,
    _item_dt_iso,
    _item_link,
    _item_title,
    _strip_html,
)
from .poll import DB_PATH, _existing_ids, poll_filings, recent_high_priority

__all__ = [
    "BSE_ANN_API_URL",
    "BSE_HOME_URL",
    "DB_PATH",
    "NAME_TO_TICKER",
    "USER_AGENT",
    "WATCHLIST_CSV",
    "classify",
    "match_ticker",
    "poll_filings",
    "recent_high_priority",
]
