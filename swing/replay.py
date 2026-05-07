"""Swing-strategy backtest CLI.

Replays the swing signal across historical daily bars and reports win rates
at +1d / +3d / +5d horizons, with and without the NIFTY+breadth regime gate.

Signal (computed at end of trading day T, on each watchlist symbol):
    1. Today's volume >= 2x 20-day average
    2. Close in top 10% of today's range:  (close - low) / (high - low) >= 0.90
    3. Close > 20-day EMA of close

Regime filter (when enabled, computed at end of T from the watchlist universe):
    4. NIFTY 50 closed up >= 0.30%
    5. Breadth: >= 55% of WATCHLIST closed above their own 20-day EMA today

Trade convention:
    Entry  = open of T+1 (next day) — realistic, you can't fill at T's close
             after seeing T's bar.
    Exits  = close of T+1, T+3, T+5  (1-day, 3-day, 5-day horizons)

Reuses the daily parquet cache produced by ``backtest.py``. Only NIFTY data
is fetched fresh (one extra HTTP call, ~0.5s).

CLI:
    python -m swing.replay                # regime ON and OFF, side-by-side
    python -m swing.replay --refresh      # bypass parquet cache
    python -m swing.replay --symbols 30   # quick run on 30 symbols

The unit tests that used to live here have moved to ``tests/test_swing.py``
and run under pytest."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import bot

from .config import MAX_EXTENSION_PCT, RESULTS_CSV
from .data import fetch_nifty, load_or_fetch_daily
from .evaluate import evaluate_swing
from .report import regime_lift, save_csv, summarize

# Line-buffered stdout so progress shows up immediately in PowerShell.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("swing")
logging.getLogger("yfinance").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing-trade backtest")
    parser.add_argument("--refresh", action="store_true",
                        help="bypass parquet cache, re-fetch from yfinance")
    parser.add_argument("--symbols", type=int, default=None,
                        help="limit to first N symbols of WATCHLIST")
    parser.add_argument("--max-extension", type=float, default=MAX_EXTENSION_PCT,
                        help=(f"skip alerts where price is more than this %% "
                              f"above 20-day EMA (default {MAX_EXTENSION_PCT}; "
                              f"set to a large number like 999 to disable)"))
    parser.add_argument("--years", type=float, default=0.5,
                        help="years of daily history to backtest on. Default "
                             "0.5 (~6 months). Try 1, 2, 3, 5, 10. yfinance "
                             "daily has no 60-day cap. Recently listed stocks "
                             "(e.g. LICI, PAYTM, ETERNAL) won't have full "
                             "history and contribute fewer alerts.")
    args = parser.parse_args()

    period_days = max(int(args.years * 365), 30)
    log.info("Backtest window: %d calendar days (~%.1f years)",
             period_days, period_days / 365)

    symbols = bot.WATCHLIST[: args.symbols] if args.symbols else bot.WATCHLIST

    daily_data = load_or_fetch_daily(
        symbols, period_days=period_days, use_cache=not args.refresh,
    )
    nifty = fetch_nifty(period_days=period_days, use_cache=not args.refresh)

    # Coverage report — recently listed stocks won't have full history.
    if daily_data:
        lengths = pd.Series([len(df) for df in daily_data.values()])
        full = (lengths >= period_days * 0.85).sum()
        log.info("History coverage: median=%d days, min=%d, full window=%d/%d",
                 int(lengths.median()), int(lengths.min()),
                 full, len(daily_data))

    ext_cap = args.max_extension
    ext_label = (
        f"max emaΔ {ext_cap:.1f}%" if ext_cap < 100 else "no extension cap"
    )

    log.info("Evaluating swing signals (regime OFF, %s)...", ext_label)
    alerts_off = evaluate_swing(
        daily_data, nifty, apply_regime=False, max_extension_pct=ext_cap,
    )
    log.info("  %d alerts pre-regime", len(alerts_off))

    log.info("Evaluating swing signals (regime ON, %s)...", ext_label)
    alerts_on = evaluate_swing(
        daily_data, nifty, apply_regime=True, max_extension_pct=ext_cap,
    )
    log.info("  %d alerts post-regime", len(alerts_on))

    save_csv(alerts_on, RESULTS_CSV)

    print(summarize(
        alerts_off, f"regime OFF, {ext_label}"
    ))
    print(summarize(
        alerts_on,
        f"regime ON (NIFTY +0.3% AND breadth >= 55%), {ext_label}",
    ))
    print(regime_lift(alerts_off, alerts_on))


if __name__ == "__main__":
    main()
