"""Diagnostic: show what the paper-tracker monitor WOULD see for the
ASIANPAINT trade right now, and decide if a restart will close it.

Run from the project root:
    venv\\Scripts\\python.exe scripts\\debug_asianpaint.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

SYMBOL = "ASIANPAINT.NS"


def main() -> None:
    # 1. Pull the trade from paper_trades.
    from paper.schema import connect
    with connect() as conn:
        row = conn.execute(
            "SELECT id, entry_ts, entry_price, stop_loss, target_1, target_2, status "
            "FROM paper_trades WHERE symbol = ? AND status = 'OPEN' "
            "ORDER BY id DESC LIMIT 1",
            (SYMBOL,),
        ).fetchone()
    if row is None:
        print(f"No OPEN paper trade for {SYMBOL}")
        return
    trade_id, entry_ts, entry, sl, tp1, tp2, status = row
    print(f"\n=== Open trade {trade_id} ({SYMBOL}, {status}) ===")
    print(f"  entry: {entry}  SL: {sl}  TP1: {tp1}  TP2: {tp2}")
    print(f"  entry_ts: {entry_ts}")

    # 2. Read the bars_5m table directly for ASIANPAINT.
    #    bars_5m is keyed by the FYERS symbol (NSE:ASIANPAINT-EQ), not
    #    the yfinance form — this mismatch was the root-cause bug.
    from data.realtime_feed import DEFAULT_DB_PATH
    from fyers_client.symbols import to_fyers
    feed_symbol = to_fyers(SYMBOL)
    with sqlite3.connect(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars_5m "
            "WHERE symbol = ? ORDER BY ts ASC",
            (feed_symbol,),
        ).fetchall()
    print(f"\n  (querying bars_5m for feed symbol: {feed_symbol})")
    print(f"\n=== bars_5m table for {SYMBOL} ({len(rows)} rows) ===")
    if not rows:
        print("  EMPTY — Fyers WebSocket has not produced any bars for this")
        print("  symbol. Restart will NOT fix this trade. The fix only works")
        print("  when bars_5m has data covering the trade window. Check that")
        print("  Fyers subscriptions are alive and the symbol is in the")
        print("  watchlist.")
        return

    # 3. Filter to bars >= entry_ts and show their highs.
    entry_dt = pd.Timestamp(entry_ts)
    if entry_dt.tz is None:
        entry_dt = entry_dt.tz_localize(IST)
    print(f"\n  {'ts':<20s}  {'open':>10}  {'high':>10}  {'low':>10}  {'close':>10}")
    print("  " + "-" * 70)
    max_high = -float("inf")
    crossed_tp1 = crossed_tp2 = crossed_sl = False
    for ts_ms, o, h, l, c, v in rows:
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert(IST)
        if ts < entry_dt - pd.Timedelta(minutes=10):
            continue  # skip way-old bars
        marker = ""
        if h >= tp2:
            crossed_tp2 = True
            marker += " ← TP2 HIT"
        elif h >= tp1:
            crossed_tp1 = True
            marker += " ← TP1 HIT"
        if l <= sl:
            crossed_sl = True
            marker += " ← SL HIT"
        if h > max_high:
            max_high = h
        print(f"  {str(ts)[:19]:<20s}  "
              f"{o:>10.2f}  {h:>10.2f}  {l:>10.2f}  {c:>10.2f}{marker}")

    # 4. Verdict.
    print(f"\n  Max high since entry: {max_high:.2f}")
    print(f"  TP2 threshold:        {tp2:.2f}")
    print(f"  TP1 threshold:        {tp1:.2f}")
    print(f"  SL threshold:         {sl:.2f}")
    print()
    if crossed_tp2:
        print("  >> TP2 IS in the bars_5m data.")
        print("     If your trade is still OPEN, the running monitor is")
        print("     using OLD code (pre-fix). RESTART THE BOT and the")
        print("     completed-bar replay will close this trade as TP2.")
    elif crossed_tp1:
        print("  >> TP1 is in the data but not TP2.")
        print("     Restart will close as TP1 (you'd see TP1 in the journal).")
    elif crossed_sl:
        print("  >> SL is in the data.")
        print("     Restart will close as SL.")
    else:
        print("  >> No completed bar has crossed any threshold yet.")
        print("     Either: (a) the price genuinely hasn't hit TP2 in a 5m")
        print("     bar's HIGH (your screen may be showing a tick that")
        print("     hasn't aggregated to a completed bar yet), or (b) ticks")
        print("     for this symbol are not reaching the BarAggregator.")


if __name__ == "__main__":
    main()
