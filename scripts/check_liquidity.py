r"""Verify the liquidity numbers logged by filters/hard.py against the
bot's daily cache AND against a fresh yfinance fetch.

Run from the project root:
    venv\Scripts\python.exe scripts\check_liquidity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running this file as a script
# from the scripts/ subdirectory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Symbols pulled straight from the user's scan log.
SYMBOLS = [
    ("3MINDIA.NS",      4_109),    # claimed
    ("MUTHOOTFIN.NS",   497_298),
    ("COLPAL.NS",       463_024),
    ("MRF.NS",          6_774),
    ("EMAMILTD.NS",     488_269),
    ("MANKIND.NS",      488_689),
    ("MARUTI.NS",       460_311),
    ("APOLLOHOSP.NS",   365_166),
]


def from_bot_cache() -> None:
    """What the bot itself thinks the 20d avg is. This is the EXACT
    number the filter compares against — so a mismatch with reality
    means stale cache."""
    print("\n=== From bot._daily_cache (what the filter sees) ===")
    from bot import market_data
    market_data.refresh_daily_cache_if_stale()
    for sym, claimed in SYMBOLS:
        df = market_data._daily_cache.get(sym)
        if df is None or df.empty:
            print(f"  {sym:18s}  NO CACHE  (claimed {claimed:,})")
            continue
        avg = float(df["Volume"].tail(20).mean())
        rows = len(df.tail(20))
        ok = "OK" if abs(avg - claimed) < 1 else "MISMATCH"
        print(f"  {sym:18s}  cache_avg={avg:>12,.0f}  "
              f"rows={rows:2d}  claimed={claimed:>10,}  {ok}")


def from_yfinance_fresh() -> None:
    """Authoritative ground truth — fetch fresh from Yahoo, no cache."""
    print("\n=== From fresh yfinance (ground truth) ===")
    import yfinance as yf
    for sym, claimed in SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="30d")
        except Exception as e:
            print(f"  {sym:18s}  FETCH ERROR: {e}")
            continue
        if df.empty:
            print(f"  {sym:18s}  EMPTY")
            continue
        avg = float(df["Volume"].tail(20).mean())
        close = float(df["Close"].iloc[-1])
        rupee_turnover_cr = avg * close / 1e7  # crore = 10^7
        print(f"  {sym:18s}  yahoo_avg={avg:>12,.0f}  "
              f"close=Rs.{close:>9,.1f}  turnover=Rs.{rupee_turnover_cr:>7,.1f}cr")


if __name__ == "__main__":
    if "--yahoo-only" in sys.argv:
        from_yfinance_fresh()
    elif "--cache-only" in sys.argv:
        from_bot_cache()
    else:
        from_bot_cache()
        from_yfinance_fresh()
