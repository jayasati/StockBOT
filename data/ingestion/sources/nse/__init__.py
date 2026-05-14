"""NSE filings + market microstructure (bulk/block deals, corp actions).

Implementation lands in sub-phase 1.5. The Phase-9 NSE free-API client
(VIX, FII/DII, OI, PCR, F&O ban, market status) intentionally stays in
``data/nse.py`` — those are market data, not filings, and the existing
session-warmup logic is reused here via ``data/nse.py``'s helpers.
"""
