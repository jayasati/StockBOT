r"""Hit the live NSE API and print the parsed MarketContext.

Run from the project root:
    venv\Scripts\python.exe scripts\verify_nse.py

What it does:
  1. Ensures alerts.db has the Phase-9 nse_snapshots table.
  2. Calls each underlying fetcher individually with per-call timing,
     so a partial failure shows you WHICH endpoint NSE is blocking.
  3. Calls get_market_context() to verify the aggregate path.
  4. Confirms nse_snapshots actually grew (= persistence is working).

What it does NOT do:
  - Touch the live bot's state. This is a standalone session.
  - Run during market-closed hours (NSE still responds but values
    are stale; the script reports the timestamps so you can spot it).

Outcomes you might see:
  [OK]   field populated, value rendered
  [None] endpoint returned, parser found no match (NSE schema shift?)
  [FAIL] network / 401 / non-JSON — usually the cookie warmup,
         which happens on first call. Re-run once.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _fmt(label: str, value, suffix: str = "") -> str:
    if value is None:
        tag = "[None]"
        body = "endpoint returned, no data parsed"
    else:
        tag = "[OK]  "
        body = f"{value}{suffix}"
    return f"  {tag} {label:<22s} {body}"


async def _timed(label: str, coro):
    """Run a coroutine, print result + latency, return value."""
    t0 = time.monotonic()
    try:
        v = await coro
    except Exception as e:
        dt = (time.monotonic() - t0) * 1000
        print(f"  [FAIL] {label:<22s} {type(e).__name__}: {e} "
              f"({dt:.0f}ms)")
        return None
    dt = (time.monotonic() - t0) * 1000
    print(f"  [OK?]  {label:<22s} {dt:.0f}ms")
    return v


async def main() -> None:
    print("=== Phase-9 NSE live verification ===\n")

    # Step 1: schema
    from bot import db as bot_db
    bot_db.init_db()
    from bot.config import DB_PATH
    print(f"DB: {DB_PATH}\n")

    snapshots_before = _count_snapshots()

    # Step 2: per-endpoint smoke test (with timings)
    print("--- Per-endpoint fetch ---")
    from data import nse

    nifty_q     = await _timed("nifty_quote", nse.fetch_nifty_quote())
    bn_q        = await _timed("banknifty_quote", nse.fetch_banknifty_quote())
    vix         = await _timed("vix", nse.fetch_vix())
    market_stat = await _timed("market_status", nse.fetch_market_status())
    fii_dii     = await _timed("fii_dii", nse.fetch_fii_dii())
    chain       = await _timed("options_chain NIFTY", nse.fetch_options_chain("NIFTY"))
    pcr         = await _timed("pcr NIFTY", nse.fetch_pcr("NIFTY"))

    print("\n--- Parsed values ---")
    print(_fmt("nifty_quote.change_pct",
               (nifty_q or {}).get("change_pct"), "%"))
    print(_fmt("nifty_quote.last",
               (nifty_q or {}).get("last")))
    print(_fmt("banknifty.change_pct",
               (bn_q or {}).get("change_pct"), "%"))
    print(_fmt("vix", vix))
    print(_fmt("market_status[CapMkt]",
               (market_stat or {}).get("Capital Market")))
    print(_fmt("fii_net_cr",
               (fii_dii or {}).get("fii_net_cr"), " cr"))
    print(_fmt("dii_net_cr",
               (fii_dii or {}).get("dii_net_cr"), " cr"))
    print(_fmt("options_chain rows",
               len(chain) if chain is not None else None))
    print(_fmt("pcr_nifty", pcr))

    # Step 3: aggregate
    print("\n--- get_market_context() aggregate ---")
    # Clear in-memory cache so this run actually exercises the
    # aggregate path rather than serving stale per-endpoint cache.
    nse.clear_caches()
    from data.market_context import format_for_status, get_market_context

    t0 = time.monotonic()
    try:
        ctx = await get_market_context()
        dt = (time.monotonic() - t0) * 1000
        print(f"  Built in {dt:.0f}ms")
        print()
        print(format_for_status(ctx))
    except Exception as e:
        print(f"  [FAIL] get_market_context: {type(e).__name__}: {e}")

    # Step 4: snapshot persistence
    snapshots_after = _count_snapshots()
    delta = snapshots_after - snapshots_before
    print(f"\n--- nse_snapshots ---")
    print(f"  rows before: {snapshots_before}")
    print(f"  rows after:  {snapshots_after}  (+{delta})")
    if delta == 0:
        print("  [WARN] No new rows persisted. Either every fetch hit"
              " the in-memory cache, or DB writes are silently"
              " failing. Re-run after clear_caches() if the former.")
    else:
        print("  [OK]   Persistence is working.")

    # Clean up the session so the script exits without lingering tasks.
    await nse.close_session()
    print("\nDone.")


def _count_snapshots() -> int:
    from bot.config import DB_PATH
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute("SELECT COUNT(*) FROM nse_snapshots").fetchone()[0]
    except sqlite3.OperationalError:
        return -1


if __name__ == "__main__":
    asyncio.run(main())
