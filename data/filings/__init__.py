"""Backwards-compatible shim — canonical home is now ``data.ingestion.sources.bse``.

Existing callers in ``bot/``, ``swing/``, ``news/`` and the ``tests/``
suite import from ``data.filings`` (and submodules ``data.filings.matcher``,
``data.filings.classify``, etc.). Those import paths keep working via this
shim and the per-submodule re-export files alongside it.

New code should import from ``data.ingestion.sources.bse`` directly.

Diagnostic CLI: ``python -m data.filings check`` is preserved.
"""
from data.ingestion.sources.bse import (
    BSE_ANN_API_URL,
    BSE_HOME_URL,
    DB_PATH,
    NAME_TO_TICKER,
    USER_AGENT,
    WATCHLIST_CSV,
    _existing_ids,
    _fetch_announcements,
    _filing_id,
    _item_dt_iso,
    _item_link,
    _item_title,
    _strip_html,
    classify,
    match_ticker,
    poll_filings,
    recent_high_priority,
    recent_unknown_events,
)

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
    "recent_unknown_events",
]
