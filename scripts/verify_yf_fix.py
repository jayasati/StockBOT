"""Verify the yfinance MultiIndex flatten fix by fetching ALKYLAMINE."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot import market_data  # noqa: E402

print("Refreshing daily cache...")
market_data.refresh_daily_cache_if_stale()

print("Fetching intraday for ALKYLAMINE.NS...")
data = market_data.fetch_intraday(["ALKYLAMINE.NS"])

if "ALKYLAMINE.NS" not in data:
    print("FAIL: ALKYLAMINE.NS not in result")
    sys.exit(1)

df = data["ALKYLAMINE.NS"]
print(f"\nALKYLAMINE.NS DataFrame:")
print(f"  rows: {len(df)}")
print(f"  columns: {list(df.columns)}")
print(f"  column type: {type(df.columns).__name__}")
print(f"  is MultiIndex: {hasattr(df.columns, 'levels')}")

print(f"\nLast 3 bars:")
print(df.tail(3))

# Test the operation that was crashing
try:
    last_close = float(df["Close"].iloc[-1])
    print(f"\nfloat(df['Close'].iloc[-1]) = {last_close}  ← FIX WORKS")
except Exception as e:
    print(f"\nfloat(df['Close'].iloc[-1]) FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
