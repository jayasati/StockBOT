"""Shim — moved to ``data.ingestion.sources.bse.matcher``."""
from data.ingestion.sources.bse.matcher import (  # noqa: F401
    NAME_TO_TICKER,
    WATCHLIST_CSV,
    _SORTED_NAMES,
    _load_name_to_ticker,
    _normalize,
    log,
    match_ticker,
)
