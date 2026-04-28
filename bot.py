"""
Personal stock momentum alert bot for NSE stocks.

Polls yfinance every 5 minutes during market hours, computes a microstructure
score based on volume ratio, RSI, VWAP position, and breakout detection, and
sends alerts to Telegram when the composite score crosses the threshold.

Run: python bot.py
"""

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Suppress httpx INFO-level logs — they leak the bot token in request URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("alertbot")

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path("alerts.db")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Settings:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Scoring threshold (0-100). Lower = more alerts.
    composite_threshold: int = 60

    # Cooldown to prevent same-stock spam
    cooldown_minutes: int = 60

    # Loop interval (seconds). 300 = every 5 minutes.
    scan_interval_seconds: int = 300

    # Cap alerts per scan to prevent flood
    max_alerts_per_scan: int = 15


settings = Settings(
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    composite_threshold=int(os.getenv("COMPOSITE_THRESHOLD", "60")),
    cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "60")),
    scan_interval_seconds=int(os.getenv("SCAN_INTERVAL_SECONDS", "300")),
)


# ============================================================================
# Watchlist — Indian large caps + mid caps (~130 stocks)
# Edit freely. Use NSE symbols with .NS suffix.
# If a symbol returns no data, the bot logs it on first scan — remove or fix.
# ============================================================================

WATCHLIST = [
    # ---- Banking & Financial Services ----
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "BANKBARODA.NS", "PNB.NS", "AUBANK.NS",
    "BANDHANBNK.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
    "HDFCAMC.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS",
    "LICI.NS", "POLICYBZR.NS", "PAYTM.NS", "PFC.NS", "RECLTD.NS",
    "IRFC.NS", "SHRIRAMFIN.NS", "M&MFIN.NS",

    # ---- IT Services ----
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS",
    "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "OFSS.NS", "KPITTECH.NS",
    "TANLA.NS",

    # ---- Pharma & Healthcare ----
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "BIOCON.NS", "ALKEM.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "GLENMARK.NS", "LAURUSLABS.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS",
    "FORTIS.NS",

    # ---- Auto & Auto Ancillaries ----
    "MARUTI.NS", "M&M.NS", "TMPV.NS", "TMCV.NS", "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS",
    "BOSCHLTD.NS", "MOTHERSON.NS", "MRF.NS", "BALKRISIND.NS", "EXIDEIND.NS",
    "ESCORTS.NS",

    # ---- Energy, Oil & Gas ----
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "GAIL.NS",
    "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
    "ADANIPOWER.NS", "ADANIGREEN.NS", "ATGL.NS", "JSWENERGY.NS",
    "SUZLON.NS", "IEX.NS", "IGL.NS", "GUJGASLTD.NS", "PETRONET.NS", "OIL.NS",

    # ---- Metals & Mining ----
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "JINDALSTEL.NS", "VEDL.NS",
    "NMDC.NS", "SAIL.NS", "HINDZINC.NS", "HINDCOPPER.NS", "APLAPOLLO.NS",

    # ---- FMCG & Consumer ----
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "GODREJCP.NS", "COLPAL.NS", "MARICO.NS", "TATACONSUM.NS", "VBL.NS",
    "UNITDSPR.NS", "TITAN.NS", "TRENT.NS", "DMART.NS", "NYKAA.NS",
    "ETERNAL.NS",

    # ---- Cement ----
    "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS",
    "DALBHARAT.NS",

    # ---- Capital Goods & Engineering ----
    "LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "CUMMINSIND.NS",
    "BEL.NS", "HAL.NS", "BHEL.NS", "VOLTAS.NS", "DIXON.NS", "POLYCAB.NS",
    "SOLARINDS.NS", "AIAENG.NS",

    # ---- Chemicals & Fertilisers ----
    "TATACHEM.NS", "SRF.NS", "PIDILITIND.NS", "UPL.NS", "DEEPAKNTR.NS",
    "AARTIIND.NS", "PIIND.NS", "NAVINFLUOR.NS", "FINEORG.NS",

    # ---- Construction & Realty ----
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS",
    "PHOENIXLTD.NS",

    # ---- Telecom ----
    "BHARTIARTL.NS",

    # ---- Paint ----
    "ASIANPAINT.NS", "BERGEPAINT.NS",

    # ---- Logistics & Ports ----
    "ADANIPORTS.NS", "ADANIENT.NS", "CONCOR.NS", "IRCTC.NS",
]


# ============================================================================
# Storage (SQLite — tracks sent alerts for cooldown)
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts_sent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    score INTEGER NOT NULL,
    reasons TEXT,
    price REAL,
    sent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_sym ON alerts_sent(symbol, sent_at);
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


def was_alerted_recently(symbol: str, minutes: int) -> bool:
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM alerts_sent WHERE symbol = ? AND sent_at > ? LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
    return row is not None


def record_alert(symbol: str, score: int, reasons: str, price: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO alerts_sent (symbol, score, reasons, price, sent_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol, score, reasons, price, datetime.now().isoformat()),
        )


# ============================================================================
# Market data fetchers (yfinance, with daily history cache)
# ============================================================================

# Daily history changes once per day, so we cache it instead of re-fetching
# 130 times every scan. The cache is refreshed at bot start and once per day.
_daily_cache: dict[str, pd.DataFrame] = {}
_daily_cache_date: str = ""


def fetch_intraday(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch 5-day, 5-minute OHLCV for all symbols in one batch call."""
    try:
        df = yf.download(
            tickers=symbols,
            period="5d",
            interval="5m",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as e:
        log.error("yfinance intraday fetch failed: %s", e)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            sub = sub.dropna()
            if not sub.empty:
                result[sym] = sub
        except (KeyError, AttributeError):
            continue
    return result


def fetch_daily_batch(symbols: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
    """Batch-fetch daily history for all symbols in one yfinance call."""
    try:
        df = yf.download(
            tickers=symbols,
            period=f"{days}d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as e:
        log.error("yfinance daily batch fetch failed: %s", e)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            sub = df[sym] if len(symbols) > 1 else df
            sub = sub.dropna()
            if not sub.empty:
                result[sym] = sub
        except (KeyError, AttributeError):
            continue
    return result


def refresh_daily_cache_if_stale() -> None:
    """Refresh the daily history cache once per calendar day."""
    global _daily_cache, _daily_cache_date
    today_str = datetime.now(IST).date().isoformat()
    if _daily_cache_date == today_str and _daily_cache:
        return
    log.info("Refreshing daily history cache for %d symbols...", len(WATCHLIST))
    _daily_cache = fetch_daily_batch(WATCHLIST, days=60)
    _daily_cache_date = today_str
    log.info("Daily cache loaded: %d/%d symbols", len(_daily_cache), len(WATCHLIST))
    missing = set(WATCHLIST) - set(_daily_cache.keys())
    if missing:
        log.warning("No daily data for: %s", ", ".join(sorted(missing)))


# ============================================================================
# Indicators
# ============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Standard 14-period RSI on closing prices."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if not pd.isna(last) else 50.0


def compute_session_vwap(df: pd.DataFrame) -> float:
    """VWAP for the current session only (today's bars)."""
    if df.empty:
        return 0.0
    today = df.index[-1].date()
    today_df = df[df.index.date == today]
    if today_df.empty:
        return 0.0
    typical = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
    vwap = (typical * today_df["Volume"]).cumsum() / today_df["Volume"].cumsum()
    return float(vwap.iloc[-1])


def compute_volume_ratio(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    """
    Today's volume so far divided by what we'd expect at this point in the
    session, based on the 10-day average daily volume.

    Above 2.0 = unusual activity. Above 3.0 = strong institutional footprint.
    """
    if intraday.empty or daily.empty or len(daily) < 10:
        return 1.0

    today = intraday.index[-1].date()
    today_data = intraday[intraday.index.date == today]
    if today_data.empty:
        return 1.0
    today_vol = float(today_data["Volume"].sum())

    avg_daily_vol = float(daily["Volume"].tail(10).mean())
    if avg_daily_vol == 0:
        return 1.0

    # Fraction of NSE session elapsed (9:15 to 15:30 = 375 minutes)
    now_t = datetime.now(IST).time()
    if now_t < time(9, 15):
        fraction = 0.01
    elif now_t > time(15, 30):
        fraction = 1.0
    else:
        elapsed = (now_t.hour - 9) * 60 + now_t.minute - 15
        fraction = max(0.05, elapsed / 375)

    expected = avg_daily_vol * fraction
    return float(today_vol / expected) if expected > 0 else 1.0


def detect_breakout(intraday: pd.DataFrame, daily: pd.DataFrame) -> tuple[bool, float]:
    """Did current price clear the 20-day high?"""
    if intraday.empty or len(daily) < 20:
        return False, 0.0
    recent_high = float(daily["High"].tail(20).max())
    current = float(intraday["Close"].iloc[-1])
    pct_from_high = (current - recent_high) / recent_high * 100
    return current > recent_high, pct_from_high


# ============================================================================
# Scoring
# ============================================================================

@dataclass
class StockSignals:
    symbol: str
    price: float
    rsi: float
    volume_ratio: float
    above_vwap: bool
    breakout: bool
    pct_from_high: float
    score: int
    reasons: list[str] = field(default_factory=list)


def score_stock(
    symbol: str, intraday: pd.DataFrame, daily: pd.DataFrame
) -> StockSignals:
    """
    Composite microstructure score (0-100). Components:
      - Volume ratio above 10-day expected (max 30 pts)
      - RSI in momentum zone 55-75 (max 25 pts)
      - Trading above session VWAP (15 pts)
      - Breakout above 20-day high (max 30 pts)
    """
    if intraday.empty or daily.empty:
        return StockSignals(symbol, 0.0, 50.0, 1.0, False, False, 0.0, 0, ["no data"])

    price = float(intraday["Close"].iloc[-1])
    rsi = compute_rsi(intraday["Close"])
    vol_ratio = compute_volume_ratio(intraday, daily)
    vwap = compute_session_vwap(intraday)
    above_vwap = price > vwap if vwap > 0 else False
    breakout, pct_from_high = detect_breakout(intraday, daily)

    score = 0
    reasons: list[str] = []

    # Volume ratio
    if vol_ratio >= 3.0:
        score += 30
        reasons.append(f"VR {vol_ratio:.1f}x")
    elif vol_ratio >= 2.0:
        score += 20
        reasons.append(f"VR {vol_ratio:.1f}x")
    elif vol_ratio >= 1.5:
        score += 10

    # RSI in momentum zone
    if 60 <= rsi <= 70:
        score += 25
        reasons.append(f"RSI {rsi:.0f}")
    elif 55 <= rsi < 60 or 70 < rsi <= 75:
        score += 15
        reasons.append(f"RSI {rsi:.0f}")
    elif rsi > 80:
        score -= 10  # overbought penalty

    # Above session VWAP
    if above_vwap:
        score += 15
        reasons.append("> VWAP")

    # Breakout above 20-day high
    if breakout:
        score += 30
        reasons.append(f"BO +{pct_from_high:.1f}%")
    elif -2 <= pct_from_high < 0:
        score += 10
        reasons.append("near high")

    score = max(0, min(100, score))

    return StockSignals(
        symbol=symbol,
        price=price,
        rsi=rsi,
        volume_ratio=vol_ratio,
        above_vwap=above_vwap,
        breakout=breakout,
        pct_from_high=pct_from_high,
        score=score,
        reasons=reasons,
    )


# ============================================================================
# Telegram notifier
# ============================================================================

class Telegram:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, text: str) -> None:
        if not self.bot_token or not self.chat_id:
            log.warning("Telegram not configured, would send: %s", text)
            return
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                r = await client.post(
                    f"{self.api}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                if r.status_code != 200:
                    # Telegram returns useful diagnostics in the response body
                    try:
                        err = r.json()
                        desc = err.get("description", r.text)
                    except Exception:
                        desc = r.text
                    log.error(
                        "Telegram API error %d: %s", r.status_code, desc
                    )
                    if r.status_code == 400 and "chat not found" in desc.lower():
                        log.error(
                            "FIX: Open Telegram, search for your bot by its "
                            "username, and tap 'Start' to initiate the chat. "
                            "Bots cannot message users who haven't messaged "
                            "them first."
                        )
                    elif r.status_code == 401:
                        log.error(
                            "FIX: Your TELEGRAM_BOT_TOKEN is invalid. "
                            "Generate a new one from @BotFather with /token."
                        )
                    return
            except httpx.RequestError as e:
                log.error("Telegram network error: %s", e)
            except Exception as e:
                log.error("Telegram send failed: %s", e)


def format_alert(s: StockSignals) -> str:
    sym = s.symbol.replace(".NS", "")
    badge = "🚀" if s.score >= 80 else "📈" if s.score >= 70 else "👀"
    lines = [
        f"{badge} <b>{sym}</b>  ₹{s.price:,.2f}",
        f"<b>Score: {s.score}/100</b>",
        f"  • {' · '.join(s.reasons) if s.reasons else 'no signals'}",
        (
            f"<a href='https://www.tradingview.com/symbols/NSE-{sym}/'>Chart</a>"
            f" · <a href='https://groww.in/stocks/{sym.lower()}'>Groww</a>"
        ),
    ]
    return "\n".join(lines)


# ============================================================================
# Market hours
# ============================================================================

def is_market_open() -> bool:
    """NSE: Mon-Fri 09:15-15:30 IST."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


def seconds_until_market_open() -> int:
    """Approximate seconds until next market open (for sleep optimization)."""
    now = datetime.now(IST)
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() > time(15, 30) or now.weekday() >= 5:
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        target = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return max(60, int((target - now).total_seconds()))


# ============================================================================
# Main scan
# ============================================================================

async def scan_once(telegram: Telegram) -> None:
    refresh_daily_cache_if_stale()

    log.info("Scanning %d symbols...", len(WATCHLIST))
    intraday_data = fetch_intraday(WATCHLIST)
    if not intraday_data:
        log.warning("No intraday data fetched (rate limit? network?). Skipping scan.")
        return

    candidates: list[StockSignals] = []
    for symbol in WATCHLIST:
        if symbol not in intraday_data:
            continue
        intraday = intraday_data[symbol]
        daily = _daily_cache.get(symbol, pd.DataFrame())
        if daily.empty:
            continue
        signals = score_stock(symbol, intraday, daily)
        log.debug("%s: score=%d %s", symbol, signals.score, signals.reasons)

        if signals.score >= settings.composite_threshold:
            if was_alerted_recently(symbol, settings.cooldown_minutes):
                continue
            candidates.append(signals)

    candidates.sort(key=lambda s: -s.score)

    sent = 0
    for s in candidates[: settings.max_alerts_per_scan]:
        await telegram.send(format_alert(s))
        record_alert(s.symbol, s.score, ", ".join(s.reasons), s.price)
        log.info("Alerted: %s score=%d reasons=%s", s.symbol, s.score, s.reasons)
        sent += 1

    log.info(
        "Scan complete. Candidates: %d, Alerted: %d (capped at %d)",
        len(candidates), sent, settings.max_alerts_per_scan,
    )


# ============================================================================
# Main loop
# ============================================================================

async def main() -> None:
    init_db()
    telegram = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)

    await telegram.send(
        f"🤖 <b>Stock alert bot started</b>\n"
        f"Watching: {len(WATCHLIST)} symbols\n"
        f"Threshold: {settings.composite_threshold}/100\n"
        f"Scan interval: {settings.scan_interval_seconds}s\n"
        f"Cooldown: {settings.cooldown_minutes}m"
    )

    log.info("Bot started. Watchlist: %d symbols", len(WATCHLIST))

    while True:
        try:
            if is_market_open():
                await scan_once(telegram)
                await asyncio.sleep(settings.scan_interval_seconds)
            else:
                wait = min(seconds_until_market_open(), 1800)  # cap at 30 min
                log.info("Market closed. Sleeping %d seconds.", wait)
                await asyncio.sleep(wait)
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.exception("Scan error: %s", e)
            await asyncio.sleep(60)


if __name__ == "__main__":
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        print("Copy .env.example to .env and fill in your credentials.")
        print("See README.md for setup instructions.")
        exit(1)
    asyncio.run(main())