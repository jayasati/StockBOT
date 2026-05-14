"""Live probe: run compute_all on intraday data for the user's symbols and
print the snapshot's actual values dict — find out exactly which keys are
populated vs None vs missing entirely."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'bot', 'indicators', etc. import.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from bot import market_data  # noqa: E402
from indicators import compute_all  # noqa: E402

SYMBOLS = ["BHARTIARTL.NS", "MCX.NS", "CIPLA.NS", "BSE.NS"]

print("Refreshing daily cache (one-time)...")
market_data.refresh_daily_cache_if_stale()

print("Fetching intraday for", SYMBOLS)
intraday = market_data.fetch_intraday(SYMBOLS)
print(f"  Got intraday for {len(intraday)} symbols")

for sym in SYMBOLS:
    print(f"\n========== {sym} ==========")
    if sym not in intraday:
        print("  NO INTRADAY DATA")
        continue
    bars = intraday[sym]
    daily = market_data._daily_cache.get(sym, pd.DataFrame())
    print(f"  intraday: {len(bars)} bars   daily: {len(daily)} bars")
    print(f"  intraday cols: {list(bars.columns)}")
    print(f"  intraday tail:\n{bars.tail(2)}")

    bars_lower = bars.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    session_date = bars.index[-1].date()
    snap = compute_all(
        symbol=sym,
        bars=bars_lower,
        daily_df=daily,
        session_date=session_date,
        target_timeframes=("5m", "15m", "60m"),
        indicators=(
            "rsi", "atr", "adx", "macd",
            "stochastic", "mfi", "cci",
            "supertrend", "bollinger", "cmf",
            "volume_surge_ratio", "ttm_squeeze",
            "pivot_points", "previous_day_hlc", "opening_range",
        ),
    )
    print(f"\n  WARNINGS ({len(snap.warnings)}):")
    for w in snap.warnings[:10]:
        print(f"    {w}")
    print(f"  INSUFFICIENT ({len(snap.insufficient)}): "
          f"{snap.insufficient[:20]}")
    print(f"\n  ALL VALUES (key: value):")
    for k in sorted(snap.values.keys()):
        v = snap.values[k]
        marker = " ← None" if v is None else ""
        print(f"    {k}: {v}{marker}")
