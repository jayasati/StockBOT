"""Symbol conversion: yfinance ``<SYM>.NS`` ↔ Fyers ``NSE:<SYM>-EQ``."""
from __future__ import annotations


def to_fyers(yf_symbol: str) -> str:
    """'RELIANCE.NS' → 'NSE:RELIANCE-EQ'. Special chars (M&M, BAJAJ-AUTO) pass through."""
    sym = yf_symbol[:-3] if yf_symbol.endswith(".NS") else yf_symbol
    sym = sym[:-3] if sym.endswith(".BO") else sym
    return f"NSE:{sym}-EQ"


def to_yf(fy_symbol: str) -> str:
    """'NSE:RELIANCE-EQ' → 'RELIANCE.NS'."""
    s = fy_symbol.split(":", 1)[-1]
    if s.endswith("-EQ"):
        s = s[:-3]
    return f"{s}.NS"
