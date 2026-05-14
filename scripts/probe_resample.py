"""Investigate the 15m/60m resample anchoring."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot import market_data  # noqa: E402
from indicators.mtf import resample_ohlcv  # noqa: E402

SYM = "BHARTIARTL.NS"

market_data.refresh_daily_cache_if_stale()
intraday = market_data.fetch_intraday([SYM])[SYM]

print(f"Total 5m bars: {len(intraday)}")
print(f"First bar: {intraday.index[0]}")
print(f"Last bar:  {intraday.index[-1]}")
print()
print(f"First 5 bars:\n{intraday.head(5)}")
print()
print(f"Last 5 bars:\n{intraday.tail(5)}")

# Filter to today only
today = intraday.index[-1].date()
today_bars = intraday[intraday.index.date == today]
print(f"\nToday's 5m bars ({today}): {len(today_bars)}")
print(f"Today's first bar: {today_bars.index[0]}")
print(f"Today's last bar:  {today_bars.index[-1]}")

# Resample using the EXISTING function
bars_lower = intraday.rename(columns={
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume",
})
print("\n--- Existing resample_ohlcv with target_tf='15m' ---")
r15 = resample_ohlcv(bars_lower, "15m")
print(f"Total 15m bars: {len(r15)}")
print(f"First 5:\n{r15.head(5)}")
print(f"Last 5:\n{r15.tail(5)}")

# Filter to today and check anchoring
today_15m = r15[r15.index.date == today]
print(f"\nToday's 15m bars: {len(today_15m)}")
if len(today_15m) > 0:
    print(f"Today's first 15m bar: {today_15m.index[0]} "
          f"({today_15m.index[0].strftime('%H:%M')})")
    print("Expected if anchored to 09:15: 09:15")
    print(f"All today's 15m bar times: "
          f"{[t.strftime('%H:%M') for t in today_15m.index]}")

# Try ALTERNATIVE anchoring with explicit offset
import pandas as pd
print("\n--- Alternative: origin set to TODAY 09:15 IST ---")
ist = intraday.index.tz
anchor = pd.Timestamp(year=today.year, month=today.month, day=today.day,
                      hour=9, minute=15, tz=ist)
alt = (
    bars_lower.resample("15min", label="left", closed="left", origin=anchor)
    .agg({"open": "first", "high": "max", "low": "min",
          "close": "last", "volume": "sum"})
    .dropna(how="all")
)
alt_today = alt[alt.index.date == today]
print(f"Today's 15m bars (alt anchor): {len(alt_today)}")
if len(alt_today) > 0:
    print(f"Times: {[t.strftime('%H:%M') for t in alt_today.index]}")
    print(f"\nFirst 5 alt bars:\n{alt_today.head(5)}")
