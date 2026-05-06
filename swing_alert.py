"""
swing_alert.py — daily end-of-day swing alert dispatcher.

Runs once per day (schedule via Windows Task Scheduler at ~4 PM IST,
after NSE close at 3:30 PM). Fetches today's daily bars, applies the
validated swing signal + regime filter from swing_backtest.py, and
sends qualifying tickers to Telegram.

Strategy (validated by swing_backtest.py — n=33-46, WR_3d ~37%,
median 3-day return ~+0.5%):
    Signal: vol >= 2x avg, close in top 10% of range, close > 20-EMA
    Regime: NIFTY +0.3% AND breadth >= 55%
    Trade:  enter T+1 open, hold 3 days, exit T+3 close

Usage:
    python swing_alert.py                # send to Telegram (production)
    python swing_alert.py --dry-run      # print message, no Telegram
    python swing_alert.py --no-regime    # bypass regime gate (debug)
    python swing_alert.py --symbols 30   # quick run on 30 symbols (debug)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

import bot
from data.yf_fetch import fetch_daily as _fetch_daily
from swing_backtest import (
    BREADTH_THRESHOLD_PCT,
    EMA_PERIOD,
    MAX_EXTENSION_PCT,
    NIFTY_TICKER,
    NIFTY_UP_PCT,
    RANGE_POSITION_THRESHOLD,
    VOLUME_MULTIPLE,
    compute_breadth_series,
    evaluate_swing,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("swing_alert")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

IST = ZoneInfo("Asia/Kolkata")
HISTORY_DAYS = 60   # 60 calendar days ≈ 40 trading days; enough for 20-EMA


# ============================================================================
# Data fetch
# ============================================================================

def fetch_daily(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Pull daily bars for the watchlist + NIFTY (no parquet cache — live path)."""
    return _fetch_daily(symbols + [NIFTY_TICKER], period_days=HISTORY_DAYS)


# ============================================================================
# Signal evaluation (latest day only)
# ============================================================================

def evaluate_today(
    stock_data: dict[str, pd.DataFrame],
    max_extension_pct: float,
) -> tuple[list[dict], dict]:
    """Apply the swing signal + regime to the most recent trading day.

    Thin wrapper over ``swing_backtest.evaluate_swing(as_of_date=...)`` —
    the alert path used to re-implement the same filter and breadth math
    (loop fork). Now both paths share one implementation."""
    if NIFTY_TICKER not in stock_data:
        log.error("NIFTY data missing — cannot run.")
        return [], {}

    nifty = stock_data[NIFTY_TICKER]
    if len(nifty) < 2:
        log.error("Not enough NIFTY history to compute change.")
        return [], {}

    last_day = nifty.index[-1]
    watchlist = {s: df for s, df in stock_data.items() if s != NIFTY_TICKER}

    # Regime info, independent of any signal hits.
    nifty_change = float(
        (nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2])
        / nifty["Close"].iloc[-2] * 100
    )
    breadth_series = compute_breadth_series(watchlist)
    breadth = float(breadth_series.get(last_day, 0.0))
    regime = {
        "date": last_day,
        "nifty_change": nifty_change,
        "breadth": breadth,
        "regime_ok": (
            nifty_change >= NIFTY_UP_PCT and breadth >= BREADTH_THRESHOLD_PCT
        ),
    }

    swing_hits = evaluate_swing(
        watchlist, nifty,
        apply_regime=False,            # the regime gate is applied by the caller
        max_extension_pct=max_extension_pct,
        as_of_date=last_day,
    )

    alerts: list[dict] = []
    for sa in swing_hits:
        sym_df = watchlist.get(sa.symbol)
        if sym_df is None or sa.signal_date not in sym_df.index:
            continue
        alerts.append({
            "symbol": sa.symbol,
            "close": float(sym_df["Close"].loc[sa.signal_date]),
            "vol_mult": sa.volume_mult,
            "range_pos": sa.range_position,
            "ema_pct": sa.ema_pct,
        })
    alerts.sort(key=lambda a: -a["vol_mult"])  # highest conviction first
    return alerts, regime


# ============================================================================
# Telegram message
# ============================================================================

def format_message(
    alerts: list[dict], regime: dict, regime_required: bool
) -> str:
    if not regime:
        return "⚠️ Swing scan failed: missing regime data."

    date_str = regime["date"].strftime("%a %d %b %Y")
    lines = [f"📊 <b>Swing Scan — {date_str}</b>", ""]
    lines.append(
        f"Regime: NIFTY {regime['nifty_change']:+.2f}%  ·  "
        f"breadth {regime['breadth']:.0f}%"
    )

    if regime_required and not regime["regime_ok"]:
        lines.append("")
        lines.append("🛑 <b>Regime gate failed — no alerts today.</b>")
        lines.append(
            f"  needs: NIFTY ≥ +{NIFTY_UP_PCT}% "
            f"AND breadth ≥ {BREADTH_THRESHOLD_PCT:.0f}%"
        )
        return "\n".join(lines)

    if not alerts:
        lines.append("")
        lines.append("📭 No qualifying setups today.")
        return "\n".join(lines)

    lines.append("")
    plural = "s" if len(alerts) != 1 else ""
    lines.append(f"<b>{len(alerts)} setup{plural} for tomorrow's open:</b>")
    lines.append("")
    for a in alerts:
        sym = a["symbol"].replace(".NS", "")
        lines.append(
            f"• <b>{sym}</b>  ₹{a['close']:,.2f}\n"
            f"   vol {a['vol_mult']:.1f}x · close@{a['range_pos']:.0%} of range "
            f"· emaΔ {a['ema_pct']:+.1f}%\n"
            f"   <a href='https://www.tradingview.com/symbols/NSE-{sym}/'>chart</a> "
            f"· <a href='https://groww.in/stocks/{sym.lower()}'>groww</a>"
        )
    lines.append("")
    lines.append("📌 <i>Entry: tomorrow's open · Hold: 3 days · Exit: T+3 close</i>")
    return "\n".join(lines)


async def send_telegram(text: str, dry_run: bool) -> None:
    if dry_run:
        print(text)
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env. "
                  "Printing instead.")
        print(text)
        return
    await bot.Telegram(token, chat_id).send(text)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily swing alert dispatcher")
    parser.add_argument("--dry-run", action="store_true",
                        help="print message instead of sending to Telegram")
    parser.add_argument("--no-regime", action="store_true",
                        help="bypass regime gate (still report regime info)")
    parser.add_argument("--symbols", type=int, default=None,
                        help="limit to first N symbols (for debugging)")
    parser.add_argument("--max-extension", type=float, default=MAX_EXTENSION_PCT,
                        help=f"skip alerts where price > X%% above 20-EMA "
                             f"(default {MAX_EXTENSION_PCT}; pass 999 to disable)")
    args = parser.parse_args()

    symbols = bot.WATCHLIST[: args.symbols] if args.symbols else bot.WATCHLIST
    stock_data = fetch_daily(symbols)

    n_stocks = len([s for s in stock_data if s != NIFTY_TICKER])
    has_nifty = NIFTY_TICKER in stock_data
    log.info("Got data for %d/%d watchlist symbols + NIFTY=%s",
             n_stocks, len(symbols), "OK" if has_nifty else "MISSING")

    alerts, regime = evaluate_today(stock_data, args.max_extension)

    if regime:
        # Sanity: warn if data is stale (script run on weekend / holiday)
        last_date = regime["date"].date()
        today = datetime.now(IST).date()
        days_old = (today - last_date).days
        if days_old > 1:
            log.warning(
                "Latest data is %s (%d days old). Market may be closed today.",
                last_date, days_old,
            )
        log.info(
            "Regime: NIFTY %+.2f%%, breadth %.0f%%, gate=%s · %d signal hit%s",
            regime["nifty_change"], regime["breadth"],
            "PASS" if regime["regime_ok"] else "FAIL",
            len(alerts), "" if len(alerts) == 1 else "s",
        )

    text = format_message(alerts, regime, regime_required=not args.no_regime)
    asyncio.run(send_telegram(text, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
