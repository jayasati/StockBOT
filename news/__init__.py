"""Phase-10 news + sentiment pipeline.

Public surface:
    fetcher.fetch_all()         — async, returns list[NewsItem]
    symbol_match.match_symbols()— headline → list[(ticker, relevance)]
    finbert.FinBert             — singleton; ``score_batch`` returns
                                  -1..+1 (stub returns 0.0 when
                                  transformers/torch are absent)
    scorer.update_news_scores() — async, writes DB
    scorer.get_symbol_news_score(symbol, hours=24) -> float | None
"""
from .symbol_match import match_symbols
from .finbert import FinBert, get_finbert

__all__ = [
    "FinBert",
    "get_finbert",
    "match_symbols",
]
