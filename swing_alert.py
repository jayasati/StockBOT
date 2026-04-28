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
import yfinance as yf
from dotenv import load_dotenv

import bot
from swing_backtest import (
    BREADTH_THRESHOLD_PCT,
    EMA_PERIOD,
    MAX_EXTENSION_PCT,
    NIFTY_TICKER,
    NIFTY_UP_PCT,
    RANGE_POSITION_THRESHOLD,
    VOLUME_MULTIPLE,
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
    """Pull daily bars for the watchlist + NIFTY in one yfinance call."""
    tickers = symbols + [NIFTY_TICKER]
    log.info("Fetching %d days of daily data for %d tickers...",
             HISTORY_DAYS, len(tickers))
    df = yf.download(
        tickers=tickers, period=f"{HISTORY_DAYS}d", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = df[t].copy() if isinstance(df.columns, pd.MultiIndex) else df.copy()
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if not sub.empty:
            out[t] = sub
    return out


# ============================================================================
# Signal evaluation (latest day only)
# ============================================================================

def _compute_breadth(
    watchlist_data: dict[str, pd.DataFrame], day: pd.Timestamp
) -> float:
    above = total = 0
    for df in watchlist_data.values():
        if day not in df.index or len(df) < EMA_PERIOD:
            continue
        ema = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean().loc[day]
        if df["Close"].loc[day] > ema:
            above += 1
        total += 1
    return (above / total * 100) if total > 0 else 0.0


def evaluate_today(
    stock_data: dict[str, pd.DataFrame],
    max_extension_pct: float,
) -> tuple[list[dict], dict]:
    """Apply signal + regime to most recent trading day. Returns (alerts, regime)."""
    if NIFTY_TICKER not in stock_data:
        log.error("NIFTY data missing — cannot run.")
        return [], {}

    nifty = stock_data[NIFTY_TICKER]
    if len(nifty) < 2:
        log.error("Not enough NIFTY history to compute change.")
        return [], {}

    last_day = nifty.index[-1]
    nifty_change = (
        (nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2])
        / nifty["Close"].iloc[-2] * 100
    )

    watchlist = {s: df for s, df in stock_data.items() if s != NIFTY_TICKER}
    breadth = _compute_breadth(watchlist, last_day)

    regime = {
        "date": last_day,
        "nifty_change": float(nifty_change),
        "breadth": float(breadth),
        "regime_ok": (
            nifty_change >= NIFTY_UP_PCT and breadth >= BREADTH_THRESHOLD_PCT
        ),
    }

    alerts: list[dict] = []
    for sym, df in watchlist.items():
        if last_day not in df.index or len(df) < EMA_PERIOD + 1:
            continue

        close = float(df["Close"].loc[last_day])
        high = float(df["High"].loc[last_day])
        low = float(df["Low"].loc[last_day])
        vol = float(df["Volume"].loc[last_day])
        if high == low or vol == 0:
            continue

        ema_now = float(
            df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean().loc[last_day]
        )
        avg_vol = float(df["Volume"].rolling(EMA_PERIOD).mean().loc[last_day])
        if avg_vol == 0 or pd.isna(avg_vol):
            continue

        vol_mult = vol / avg_vol
        rp = (close - low) / (high - low)
        ema_pct = (close - ema_now) / ema_now * 100 if ema_now > 0 else 0.0

        if vol_mult < VOLUME_MULTIPLE:
            continue
        if rp < RANGE_POSITION_THRESHOLD:
            continue
        if close <= ema_now:
            continue
        if ema_pct > max_extension_pct:
            continue

        alerts.append({
            "symbol": sym,
            "close": close,
            "vol_mult": vol_mult,
            "range_pos": rp,
            "ema_pct": ema_pct,
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
