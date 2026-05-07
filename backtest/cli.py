"""``python -m backtest`` CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys

import bot

from .config import COOLDOWN_MINUTES, RESULTS_CSV, SCORE_THRESHOLD
from .data import fetch_daily, fetch_intraday_5m
from .replay import replay
from .report import apply_cooldown, save_csv, summarize, sweep_summary

# Line-buffer stdout so progress lines appear immediately in PowerShell.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("backtest")
logging.getLogger("yfinance").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest bot.score_stock()")
    parser.add_argument("--refresh", action="store_true",
                        help="bypass parquet cache, re-fetch from yfinance")
    parser.add_argument("--symbols", type=int, default=None,
                        help="limit to first N symbols of WATCHLIST")
    parser.add_argument("--threshold", type=int, default=SCORE_THRESHOLD,
                        help="composite score threshold (default 60)")
    parser.add_argument("--sweep", action="store_true",
                        help="run threshold sweep (50/60/70/80) and print "
                             "comparison table — checks if higher score "
                             "actually means better trades")
    args = parser.parse_args()

    symbols = bot.WATCHLIST[: args.symbols] if args.symbols else bot.WATCHLIST

    intraday = fetch_intraday_5m(symbols, days=60, use_cache=not args.refresh)
    daily = fetch_daily(symbols, months=6, use_cache=not args.refresh)

    if args.sweep:
        # Record every event >= 50 with no cooldown; apply cooldown post-hoc
        # per threshold. One replay covers all sweep points.
        all_records = replay(symbols, intraday, daily,
                             threshold=50, cooldown_minutes=0)
        # Save the threshold-60 view to CSV for further analysis.
        save_csv(apply_cooldown(all_records, SCORE_THRESHOLD, COOLDOWN_MINUTES),
                 RESULTS_CSV)
        print(sweep_summary(all_records, [50, 60, 70, 80]))
        return

    records = replay(symbols, intraday, daily, threshold=args.threshold)
    save_csv(records, RESULTS_CSV)
    print(summarize(records))


if __name__ == "__main__":
    main()
