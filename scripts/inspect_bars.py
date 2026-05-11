"""Eyeball the multi-timeframe aggregation for a single symbol.

Prints the last few 5m / 15m / 1h / 1d bars derived from ``alerts.db`` so
you can compare boundaries against a TradingView chart.

Usage
-----
    python -m scripts.inspect_bars                # auto-picks the symbol
                                                  # with the most rows
    python -m scripts.inspect_bars NSE:RELIANCE-EQ
    python -m scripts.inspect_bars NSE:TCS-EQ --explain
                                                  # also dumps the 5m bars
                                                  # that feed the latest 1h
                                                  # bucket

Honours an optional ``--now ISO`` argument so you can replay how the layer
would have looked at a past timestamp:

    python -m scripts.inspect_bars NSE:TCS-EQ --now "2026-05-08 11:07"
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from data.timeframes import get_bars

DB_PATH = Path("alerts.db")


def _pick_default_symbol() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT symbol, COUNT(*) AS n FROM bars_5m "
            "GROUP BY symbol ORDER BY n DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise SystemExit("alerts.db has no rows in bars_5m")
    return row[0]


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "    (no complete bars)"
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].map(lambda v: f"{v:>10.2f}")
    out["volume"] = out["volume"].map(lambda v: f"{int(v):>10d}")
    return out.to_string()


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(_fmt(df))


def _explain_last_1h(symbol: str, now: pd.Timestamp | None) -> None:
    """For the most recent complete 1h bar, dump the 5m bars that fed it."""
    df_1h = get_bars(symbol, "1h", n=1, _now=now)
    if df_1h.empty:
        print("\n(no complete 1h bar — nothing to explain)")
        return
    bucket_open = df_1h.index[-1]
    # last bucket runs to session_close (15:30), others to bucket_open + 1h
    cand_close = bucket_open + pd.Timedelta(hours=1)
    last_hour_open = bucket_open.normalize() + pd.Timedelta(hours=14, minutes=15)
    bucket_close = (
        bucket_open.normalize() + pd.Timedelta(hours=15, minutes=30)
        if bucket_open == last_hour_open
        else cand_close
    )
    df_5m = get_bars(symbol, "5m", n=20, _now=now)
    feeders = df_5m[(df_5m.index >= bucket_open) & (df_5m.index < bucket_close)]

    print(f"\n=== 5m bars feeding 1h bucket {bucket_open.strftime('%H:%M')}"
          f"-{bucket_close.strftime('%H:%M')} ===")
    print(_fmt(feeders))
    print(f"  -> aggregated open  = {feeders['open'].iloc[0]:.2f}")
    print(f"  -> aggregated high  = {feeders['high'].max():.2f}")
    print(f"  -> aggregated low   = {feeders['low'].min():.2f}")
    print(f"  -> aggregated close = {feeders['close'].iloc[-1]:.2f}")
    print(f"  -> aggregated vol   = {int(feeders['volume'].sum())}")
    print("  1h row from get_bars:")
    print(_fmt(df_1h))


def main() -> int:
    p = argparse.ArgumentParser(prog="inspect_bars")
    p.add_argument("symbol", nargs="?", help="e.g. NSE:RELIANCE-EQ")
    p.add_argument("--now", help='Reference time (IST), e.g. "2026-05-08 11:07"')
    p.add_argument("--explain", action="store_true",
                   help="Also dump the 5m bars feeding the last 1h bucket")
    p.add_argument("--n5", type=int, default=6)
    p.add_argument("--n15", type=int, default=4)
    p.add_argument("--n1h", type=int, default=4)
    p.add_argument("--n1d", type=int, default=3)
    args = p.parse_args()

    symbol = args.symbol or _pick_default_symbol()
    now = (
        pd.Timestamp(args.now, tz="Asia/Kolkata") if args.now else None
    )

    print(f"symbol : {symbol}")
    print(f"db     : {DB_PATH.resolve()}")
    print(f"_now   : {now if now is not None else 'wall-clock IST'}")

    _print_section(f"5m  (last {args.n5})",
                   get_bars(symbol, "5m", n=args.n5, _now=now))
    _print_section(f"15m (last {args.n15})",
                   get_bars(symbol, "15m", n=args.n15, _now=now))
    _print_section(f"1h  (last {args.n1h})",
                   get_bars(symbol, "1h", n=args.n1h, _now=now))
    _print_section(f"1d  (last {args.n1d})",
                   get_bars(symbol, "1d", n=args.n1d, _now=now))

    if args.explain:
        _explain_last_1h(symbol, now)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
