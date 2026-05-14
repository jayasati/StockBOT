"""Inspect what's actually in the indicator snapshot for user-named symbols."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB = Path("alerts.db")
SYMBOLS = [
    "BHARTIARTL.NS", "BSE.NS", "CIPLA.NS",
    "MCX.NS", "ADANIPOWER.NS", "HFCL.NS",
]

con = sqlite3.connect(DB)
cur = con.cursor()

print("=== signal_indicators schema ===")
for row in cur.execute("PRAGMA table_info(signal_indicators)"):
    print(" ", row)

print("\n=== signal_indicators row count today ===")
cur.execute(
    "SELECT COUNT(*) FROM signal_indicators "
    "WHERE date(ts) = date('now', 'localtime')"
)
print(" ", cur.fetchone())

print("\n=== signal_indicators latest row per user symbol ===")
cur.execute(
    "SELECT * FROM signal_indicators ORDER BY ts DESC LIMIT 1"
)
sample = cur.fetchone()
print("  sample row:", sample)

# Try to find rows for user symbols
for sym in SYMBOLS:
    print(f"\n  --- {sym} ---")
    cur.execute(
        "SELECT * FROM signal_indicators WHERE symbol = ? "
        "ORDER BY ts DESC LIMIT 1",
        (sym,),
    )
    row = cur.fetchone()
    if not row:
        print("    (no rows)")
        continue
    cols = [d[0] for d in cur.description]
    for col_name, val in zip(cols, row):
        if col_name in ("indicators_json", "snapshot_json", "values_json"):
            try:
                parsed = json.loads(val) if val else {}
                if isinstance(parsed, dict) and "values" in parsed:
                    parsed = parsed["values"]
                # Print only momentum / volatility / trend keys
                interesting = {
                    k: v for k, v in (parsed.items() if isinstance(parsed, dict) else [])
                    if any(s in k for s in ("rsi", "macd", "atr", "adx",
                                             "stoch", "cci", "mfi",
                                             "bollinger", "ttm_squeeze",
                                             "supertrend", "cmf",
                                             "volume_surge", "pdh", "pdl",
                                             "pivot", "orh", "orl"))
                }
                print(f"    {col_name} (interesting keys):")
                for k, v in sorted(interesting.items()):
                    print(f"        {k}: {v}")
            except Exception as exc:
                print(f"    {col_name}: parse failed ({exc}); raw[:300]={str(val)[:300]}")
        else:
            print(f"    {col_name}: {val}")
