# Stock alert bot

Personal momentum-alert bot for NSE stocks. Polls market data every 5 minutes during market hours, scores stocks on volume ratio, RSI, VWAP position, and breakout strength, and pings you on Telegram when something looks interesting.

Built for one user (you). No accounts, no multi-tenancy, no deployment overhead. Runs in a single Python process.

## What it does

- Watches a configurable list of NSE stocks (default: 30+ large/mid caps)
- Fetches OHLCV data from Yahoo Finance via `yfinance` (free, no API key)
- Computes a 0-100 composite score every scan based on:
  - **Volume ratio** vs 10-day expected volume at this point in the session
  - **RSI** in the 55-75 momentum zone
  - **Position relative to session VWAP**
  - **Breakout** above the 20-day high
- Sends Telegram alerts when score crosses a threshold
- Tracks sent alerts in SQLite to enforce a cooldown (no spam)

## What it does not do (yet)

- Real-time tick data (yfinance is delayed 15-20 min — fine for swing alerts, not for HFT)
- F&O signals (OI buildup, IV expansion) — needs a broker API like Fyers
- News classification — easy to add via a corporate filings RSS feed
- Suppression rules for ASM/GSM/pledge — needs additional data sources

These are deliberate omissions for v1. Add them as you need them.

## Setup (5 minutes)

### 1. Create a Telegram bot

Open Telegram, search for **@BotFather**, and send `/newbot`. Follow the prompts:
- Pick a name (e.g. "My Stock Alerts")
- Pick a username ending in "bot" (e.g. "mystockalerts_bot")

BotFather will reply with a token like `1234567890:ABCdefGhIJKlmNOpqrsTUVwxyz`. Save it.

### 2. Get your chat ID

Search for **@userinfobot** in Telegram and send it `/start`. It will reply with your numeric user ID — that's your chat ID.

(Alternative: send a message to your new bot, then visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser. Your chat ID is in the response.)

### 3. Install Python dependencies

Requires Python 3.10 or newer.

```bash
cd stock_alert_bot
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` and paste your bot token and chat ID.

### 5. Run

```bash
python bot.py
```

You should immediately get a "Bot started" message on Telegram. From here it scans every 5 minutes during market hours (Mon-Fri, 09:15-15:30 IST).

## Customization

### Watchlist

Edit the `WATCHLIST` list in `bot.py`. Use NSE symbols with `.NS` suffix:

```python
WATCHLIST = [
    "RELIANCE.NS",
    "TATACHEM.NS",
    "YOURFAVORITE.NS",
]
```

For BSE stocks use `.BO` suffix instead. To find the right symbol, search the company on Yahoo Finance India.

### Threshold

`COMPOSITE_THRESHOLD=60` is reasonable to start. Tune it after watching the bot for a week:
- Too many alerts (more than 10/day)? Raise to 65 or 70.
- Too few? Drop to 55 or 50.
- Tier-1 conviction trades only? Set to 80.

### Scoring weights

The weights live in `score_stock()` inside `bot.py`. Adjust freely — that function is where the alpha is. The current weights are tuned to the Indian-market PDF you provided. You'll likely want to retune them after seeing what kinds of moves they catch and miss.

## Running 24/7

For market-hours-only scanning, the bot itself handles sleeping outside of trading hours, so you just need a way to keep the Python process alive.

### Option 1: Linux server (cheap cloud VM)
Run as a systemd service. Create `/etc/systemd/system/stockbot.service`:

```ini
[Unit]
Description=Stock alert bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/stock_alert_bot
ExecStart=/home/youruser/stock_alert_bot/venv/bin/python bot.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Then `sudo systemctl enable --now stockbot`.

### Option 2: Raspberry Pi
Same systemd setup as above. A Pi 4 handles this comfortably and costs nothing to run.

### Option 3: Free cloud tiers
- **Oracle Cloud Always Free** has an ARM VM with 4 vCPU and 24GB RAM, free forever. Overkill for this but you cannot beat the price.
- **Google Cloud e2-micro** in us-west1 is also free.
- **fly.io** has a generous free tier suitable for this.

### Option 4: Just your laptop
Run it in a `tmux` or `screen` session while you work. Fine for evaluation; closes when your laptop sleeps.

## Inspecting results

The SQLite DB at `alerts.db` stores every alert sent. Inspect with:

```bash
sqlite3 alerts.db "SELECT symbol, score, price, sent_at FROM alerts_sent ORDER BY sent_at DESC LIMIT 20;"
```

Use this to backtest manually: did the alerted stocks actually move in the next 30 minutes / 1 day / 5 days? After 100+ alerts you'll have enough data to retune the weights.

## Troubleshooting

**"No intraday data fetched"** — yfinance occasionally rate-limits. The bot logs and retries on the next scan; if it persists, check your IP is not blocked by Yahoo, or reduce the watchlist size.

**No alerts at all for hours** — the threshold may be too high, or the market is genuinely quiet. Lower `COMPOSITE_THRESHOLD` to 50 temporarily and see what scores you're actually getting (look at the DEBUG logs).

**Telegram messages not arriving** — verify the bot token by visiting `https://api.telegram.org/bot<TOKEN>/getMe` in a browser. Verify the chat ID is your numeric ID, not your @username.

**Symbols missing data** — some smaller NSE stocks have intermittent yfinance coverage. Use Trendlyne or the broker's own data feed (Fyers, Kite) when you graduate beyond yfinance.

## What to build next

In rough order of value:

1. **Backtest the scoring on historical data.** Before tuning anything, replay the last 6 months and label each alert as profitable/not. This tells you whether the signals work for your style.
2. **Add a corporate filings news source** (BSE XML feed or NSE corp announcements RSS) and a basic regex classifier for binary news (earnings, PLI, M&A).
3. **Replace yfinance with Fyers WebSocket** for real-time data. The yfinance delay is fine for swing alerts but lethal for intraday momentum.
4. **Add F&O signals** (OI buildup, IV expansion) for F&O-eligible stocks. This is the highest-leverage addition.
5. **Add suppression rules** (ASM/GSM list check, pledge spike detector).
6. **Write the feedback loop** — label every alert post-hoc with its 30-minute and 1-day return, and use that to retune weights weekly.

Don't build all six. Build the next one when the absence hurts.
