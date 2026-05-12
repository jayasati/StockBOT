"""Build a TradingView-parity fixture CSV from yfinance 5m bars.

Pulls one NSE trading session of 5-min OHLCV for the supplied symbol,
filters to 09:15-15:30 IST, and writes a CSV in the schema the
indicator parity tests will consume:

    timestamp,open,high,low,close,volume

Where ``timestamp`` is a tz-aware IST ISO string. The first column
becomes the DataFrame index when the test loader reads it back.

Usage
-----
    python -m scripts.build_tv_fixture RELIANCE.NS 2026-05-09 \\
        tests/fixtures/reliance_trending_5m.csv

yfinance gives the last ~60 days of 5-min bars for NSE-listed names
(``.NS`` suffix). Pick a session inside that window or the fetch will
return empty. The script does NOT handle date ranges or batch
downloads — one symbol, one session, one CSV by design."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)


def fetch_session(symbol: str, session_date_str: str, out_path: Path) -> int:
    try:
        session_date = datetime.strptime(session_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: bad date {session_date_str!r} — use YYYY-MM-DD",
              file=sys.stderr)
        return 0

    start = session_date
    end = session_date + timedelta(days=1)
    df = yf.download(
        tickers=symbol, start=start, end=end, interval="5m",
        progress=False, auto_adjust=False, prepost=False,
    )
    if df is None or df.empty:
        print(f"ERROR: yfinance returned no rows for {symbol} on "
              f"{session_date} — outside the 60-day window?",
              file=sys.stderr)
        return 0

    # yfinance returns MultiIndex columns when given a single ticker on
    # some versions; collapse to single level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    in_session = pd.Series(
        (df.index.date == session_date)
        & (pd.Series(df.index.time).between(
            SESSION_OPEN, SESSION_CLOSE, inclusive="left",
        ).to_numpy()),
        index=df.index,
    )
    session = df.loc[in_session.to_numpy()].copy()
    if session.empty:
        print(f"ERROR: no bars in 09:15-15:30 IST for {symbol} on "
              f"{session_date} — was this a market holiday?",
              file=sys.stderr)
        return 0

    # Map whatever casing/whitespace yfinance returned to clean lowercase
    # column names. Building a fresh frame from arrays guarantees the CSV
    # header is exactly "open,high,low,close,volume" — earlier versions of
    # this script ended up with padded headers like "  open" through some
    # yfinance/pandas interaction.
    normalized = {c.strip().lower(): c for c in session.columns}
    out = pd.DataFrame(
        {
            "open":   session[normalized["open"]].astype("float64").to_numpy(),
            "high":   session[normalized["high"]].astype("float64").to_numpy(),
            "low":    session[normalized["low"]].astype("float64").to_numpy(),
            "close":  session[normalized["close"]].astype("float64").to_numpy(),
            "volume": session[normalized["volume"]].astype("float64").to_numpy(),
        },
        index=session.index,
    )
    out.index.name = "timestamp"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(f"OK  wrote {len(out)} bars -> {out_path}")
    print(f"    first: {out.index[0]}")
    print(f"    last:  {out.index[-1]}")
    print(f"    last 3 closes: "
          f"{', '.join(f'{c:.2f}' for c in out['close'].tail(3))}")
    return len(out)


def main() -> int:
    p = argparse.ArgumentParser(prog="build_tv_fixture")
    p.add_argument("symbol", help="yfinance symbol, e.g. RELIANCE.NS")
    p.add_argument("date", help="Session date YYYY-MM-DD (IST)")
    p.add_argument("out", help="Output CSV path")
    args = p.parse_args()
    n = fetch_session(args.symbol, args.date, Path(args.out))
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
