"""
EOD swing backtest — daily-bar signals with optional NIFTY/breadth regime filter.

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

Reuses the daily parquet cache produced by backtest.py. Only NIFTY data
is fetched fresh (one extra HTTP call, ~0.5s).

Usage:
    python swing_backtest.py                # regime ON and OFF, side-by-side
    python swing_backtest.py --refresh      # bypass parquet cache
    python swing_backtest.py --symbols 30   # quick run on 30 symbols
    python swing_backtest.py --tests        # unit tests only
"""

import argparse
import logging
import sys
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import bot
from backtest import (
    CACHE_DIR,
    DAILY_CACHE,
    WIN_THRESHOLD_PCT,
    _load_parquet,
    _normalize_yf_batch,
    _save_parquet,
    compute_win_rate,
)

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

NIFTY_TICKER = "^NSEI"
NIFTY_CACHE = CACHE_DIR / "nifty_daily.parquet"
RESULTS_CSV = Path("swing_results.csv")

# Signal parameters
VOLUME_MULTIPLE = 2.0
RANGE_POSITION_THRESHOLD = 0.90
EMA_PERIOD = 20
# Skip alerts where price is >X% above 20-day EMA (parabolic / exhaustion).
# Backtest showed every disaster trade had emaΔ > 10%. Default 8% is the
# conservative cut; pass --max-extension to override.
MAX_EXTENSION_PCT = 8.0

# Regime parameters
NIFTY_UP_PCT = 0.30
BREADTH_THRESHOLD_PCT = 55.0

HORIZONS = [1, 3, 5]


# ============================================================================
# Data fetching
# ============================================================================

def _daily_cache_path(period_days: int) -> Path:
    """Separate cache file per period so changing --years doesn't return
    stale data. The 186-day default re-uses backtest.py's cache file."""
    if period_days == 186:
        return DAILY_CACHE
    return CACHE_DIR / f"daily_{period_days}d.parquet"


def _nifty_cache_path(period_days: int) -> Path:
    if period_days == 186:
        return NIFTY_CACHE
    return CACHE_DIR / f"nifty_{period_days}d.parquet"


def fetch_nifty(period_days: int = 186, use_cache: bool = True) -> pd.DataFrame:
    cache_path = _nifty_cache_path(period_days)
    if use_cache and cache_path.exists():
        log.info("Loading NIFTY data from cache: %s", cache_path)
        df = pd.read_parquet(cache_path, engine="pyarrow")
        return df.set_index("timestamp").sort_index()
    log.info("Fetching NIFTY daily for %d days from yfinance...", period_days)
    df = yf.download(
        tickers=NIFTY_TICKER, period=f"{period_days}d", interval="1d",
        progress=False, auto_adjust=False,
    )
    df = df.dropna(how="all")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index.name = "timestamp"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(cache_path, engine="pyarrow", index=False)
    log.info("Got %d days of NIFTY data", len(df))
    return df


def load_or_fetch_daily(
    symbols: list[str], period_days: int = 186, use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    cache_path = _daily_cache_path(period_days)
    if use_cache and cache_path.exists():
        cached = _load_parquet(cache_path)
        out = {s: cached[s] for s in symbols if s in cached}
        if out:
            log.info("Loaded daily cache (%d days): %d/%d symbols",
                     period_days, len(out), len(symbols))
            return out
    log.info("Fetching %d days of daily data for %d symbols...",
             period_days, len(symbols))
    df = yf.download(
        tickers=symbols, period=f"{period_days}d", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )
    data = _normalize_yf_batch(df, symbols, localize_ist=False)
    _save_parquet(data, cache_path)
    return data


# ============================================================================
# Breadth pre-computation (one pass, vectorised)
# ============================================================================

def compute_breadth_series(
    daily_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """Return Series(date -> % of symbols closing above their own 20-day EMA)."""
    columns: dict[str, pd.Series] = {}
    for sym, df in daily_data.items():
        ema = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        columns[sym] = (df["Close"] > ema).astype(float)
    wide = pd.DataFrame(columns)
    return wide.mean(axis=1) * 100


# ============================================================================
# Signal evaluation
# ============================================================================

@dataclass
class SwingAlert:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    volume_mult: float
    range_position: float
    ema_pct: float
    nifty_change: float
    breadth_pct: float
    regime_ok: bool
    exit_1d: Optional[float]
    exit_3d: Optional[float]
    exit_5d: Optional[float]
    ret_1d: Optional[float]
    ret_3d: Optional[float]
    ret_5d: Optional[float]


def evaluate_swing(
    daily_data: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    apply_regime: bool,
    max_extension_pct: Optional[float] = None,
) -> list[SwingAlert]:
    breadth_series = compute_breadth_series(daily_data)
    nifty_change = nifty["Close"].pct_change() * 100

    alerts: list[SwingAlert] = []

    for sym, df in daily_data.items():
        if len(df) < EMA_PERIOD + max(HORIZONS) + 1:
            continue

        ema = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        avg_vol = df["Volume"].rolling(EMA_PERIOD).mean()
        rng = (df["High"] - df["Low"]).replace(0, np.nan)
        range_pos = (df["Close"] - df["Low"]) / rng

        for i in range(EMA_PERIOD, len(df) - max(HORIZONS)):
            day = df.index[i]
            close = float(df["Close"].iloc[i])
            vol = float(df["Volume"].iloc[i])
            av = avg_vol.iloc[i]
            if pd.isna(av) or av == 0:
                continue
            vol_mult = vol / av
            rp = range_pos.iloc[i]
            if pd.isna(rp):
                continue
            ema_now = float(ema.iloc[i])
            ema_pct = (close - ema_now) / ema_now * 100 if ema_now > 0 else 0.0

            if vol_mult < VOLUME_MULTIPLE:
                continue
            if rp < RANGE_POSITION_THRESHOLD:
                continue
            if close <= ema_now:
                continue
            if max_extension_pct is not None and ema_pct > max_extension_pct:
                continue

            n_change = nifty_change.get(day, np.nan)
            n_change = float(n_change) if not pd.isna(n_change) else 0.0
            breadth = float(breadth_series.get(day, 0.0))
            regime_ok = (
                n_change >= NIFTY_UP_PCT
                and breadth >= BREADTH_THRESHOLD_PCT
            )
            if apply_regime and not regime_ok:
                continue

            entry_idx = i + 1
            entry_date = df.index[entry_idx]
            entry_open = float(df["Open"].iloc[entry_idx])

            def _exit(h: int) -> Optional[float]:
                idx = i + h
                if idx >= len(df):
                    return None
                return float(df["Close"].iloc[idx])

            def _ret(p: Optional[float]) -> Optional[float]:
                return (p - entry_open) / entry_open * 100 if p is not None else None

            x1, x3, x5 = _exit(1), _exit(3), _exit(5)
            alerts.append(SwingAlert(
                symbol=sym,
                signal_date=day,
                entry_date=entry_date,
                entry_price=entry_open,
                volume_mult=float(vol_mult),
                range_position=float(rp),
                ema_pct=float(ema_pct),
                nifty_change=n_change,
                breadth_pct=breadth,
                regime_ok=regime_ok,
                exit_1d=x1, exit_3d=x3, exit_5d=x5,
                ret_1d=_ret(x1), ret_3d=_ret(x3), ret_5d=_ret(x5),
            ))

    return alerts


# ============================================================================
# Reporting
# ============================================================================

def summarize(alerts: list[SwingAlert], label: str) -> str:
    if not alerts:
        return f"\n{label}\n  (no alerts)"

    df = pd.DataFrame([asdict(a) for a in alerts])
    df["signal_date"] = pd.to_datetime(df["signal_date"])

    out: list[str] = []
    bar = "=" * 76
    out.append("")
    out.append(bar)
    out.append(f"SWING BACKTEST  ({label})")
    out.append(f"Alerts: {len(alerts)}   "
               f"Period: {df['signal_date'].min().date()} → "
               f"{df['signal_date'].max().date()}")
    out.append(f"Win threshold: >= {WIN_THRESHOLD_PCT}% move")
    out.append(bar)
    out.append("")
    out.append(f"{'Hold':<8} {'Count':>7} {'WinRate':>10} {'AvgRet':>10} "
               f"{'MedRet':>10} {'StdDev':>10}")
    out.append("-" * 60)
    for h_label, col in [("1-day", "ret_1d"), ("3-day", "ret_3d"), ("5-day", "ret_5d")]:
        rets = df[col].dropna().tolist()
        if not rets:
            continue
        wr = compute_win_rate(rets)
        avg = float(np.mean(rets))
        med = float(np.median(rets))
        std = float(np.std(rets))
        out.append(f"{h_label:<8} {len(rets):>7} {wr:>9.1f}% "
                   f"{avg:>9.2f}% {med:>9.2f}% {std:>9.2f}%")

    have_5 = df[df["ret_5d"].notna()]
    if not have_5.empty:
        out.append("")
        out.append("BEST 5 (5-day return):")
        for _, r in have_5.nlargest(5, "ret_5d").iterrows():
            out.append(f"  {r['symbol']:<14} {r['signal_date'].date()}  "
                       f"vol={r['volume_mult']:>4.1f}x  "
                       f"rp={r['range_position']:.2f}  "
                       f"emaΔ={r['ema_pct']:+.1f}%  "
                       f"ret_5d={r['ret_5d']:+.2f}%")
        out.append("")
        out.append("WORST 5 (5-day return):")
        for _, r in have_5.nsmallest(5, "ret_5d").iterrows():
            out.append(f"  {r['symbol']:<14} {r['signal_date'].date()}  "
                       f"vol={r['volume_mult']:>4.1f}x  "
                       f"rp={r['range_position']:.2f}  "
                       f"emaΔ={r['ema_pct']:+.1f}%  "
                       f"ret_5d={r['ret_5d']:+.2f}%")

    # Conviction analysis: split by volume multiple bucket
    out.append("")
    out.append("BY VOLUME-MULTIPLE BUCKET (5-day return):")
    out.append(f"  {'Bucket':<14} {'n':>5} {'WinRate':>9} {'AvgRet':>9}")
    buckets = [
        ("vol 2.0–3.0x", lambda v: 2.0 <= v < 3.0),
        ("vol 3.0–5.0x", lambda v: 3.0 <= v < 5.0),
        ("vol >= 5.0x",  lambda v: v >= 5.0),
    ]
    for name, pred in buckets:
        sub = have_5[have_5["volume_mult"].apply(pred)]
        if sub.empty:
            out.append(f"  {name:<14} {0:>5}    (none)")
            continue
        rets = sub["ret_5d"].tolist()
        out.append(f"  {name:<14} {len(sub):>5} "
                   f"{compute_win_rate(rets):>8.1f}% "
                   f"{float(np.mean(rets)):>+8.2f}%")

    out.append(bar)
    return "\n".join(out)


def regime_lift(
    alerts_off: list[SwingAlert], alerts_on: list[SwingAlert]
) -> str:
    """Compare 5-day stats with and without regime filter."""
    def stats(alerts):
        if not alerts:
            return (0, 0.0, 0.0)
        rets = [a.ret_5d for a in alerts if a.ret_5d is not None]
        if not rets:
            return (len(alerts), 0.0, 0.0)
        return (len(alerts), compute_win_rate(rets), float(np.mean(rets)))

    n_off, wr_off, av_off = stats(alerts_off)
    n_on, wr_on, av_on = stats(alerts_on)

    out: list[str] = []
    out.append("")
    out.append("=" * 76)
    out.append("REGIME FILTER LIFT  (5-day hold)")
    out.append("=" * 76)
    out.append(f"  {'Mode':<14} {'Alerts':>8} {'WinRate':>10} {'AvgRet':>10}")
    out.append(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    out.append(f"  {'regime OFF':<14} {n_off:>8} {wr_off:>9.1f}% {av_off:>+9.2f}%")
    out.append(f"  {'regime ON':<14} {n_on:>8} {wr_on:>9.1f}% {av_on:>+9.2f}%")
    out.append(f"  {'lift':<14} {n_on - n_off:>+8} {wr_on - wr_off:>+9.1f}pp "
               f"{av_on - av_off:>+9.2f}pp")
    out.append("")
    out.append("Read: positive lift on win rate AND avg return = regime filter")
    out.append("is selecting the up-tape days where momentum signals work.")
    out.append("=" * 76)
    return "\n".join(out)


def save_csv(alerts: list[SwingAlert], path: Path) -> None:
    if not alerts:
        log.warning("No alerts to save.")
        return
    pd.DataFrame([asdict(a) for a in alerts]).to_csv(path, index=False)
    log.info("Saved %d alerts to %s", len(alerts), path)


# ============================================================================
# Tests
# ============================================================================

class TestRangePosition(unittest.TestCase):
    def test_close_at_high(self):
        rp = (110 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 1.0)

    def test_close_at_low(self):
        rp = (100 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 0.0)

    def test_close_in_top_10pct(self):
        # close=109, low=100, high=110 → rp = 0.9 → just qualifies
        rp = (109 - 100) / (110 - 100)
        self.assertAlmostEqual(rp, 0.9)
        self.assertGreaterEqual(rp, RANGE_POSITION_THRESHOLD)


class TestExtensionFilter(unittest.TestCase):
    """Build a synthetic stock that triggers the volume+range+EMA signal but
    is heavily extended above the 20-day EMA. With no cap, it should fire;
    with a tight cap, it should be filtered out."""

    def _make_data(self):
        n = 35
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        close = np.linspace(100.0, 130.0, n)  # rising — 30% over 35d
        # Final bar: huge volume spike, close at top of range, well above EMA
        close[-1] = close[-2] * 1.04
        high = close + 0.5
        low = np.minimum(close - 0.5, np.roll(close, 1) - 0.5)
        low[-1] = close[-1] - 0.5  # tiny range, close at top
        high[-1] = close[-1] + 0.05
        volume = np.full(n, 1000.0)
        volume[-1] = 5000.0  # 5x avg
        df = pd.DataFrame({
            "Open": close - 0.2, "High": high, "Low": low,
            "Close": close, "Volume": volume,
        }, index=idx)
        return df, idx

    def test_no_cap_fires(self):
        df, idx = self._make_data()
        # Need an entry+5d future, so add 6 dummy bars after the signal
        future = pd.DataFrame({
            "Open": [df["Close"].iloc[-1]] * 6,
            "High": [df["Close"].iloc[-1] + 1] * 6,
            "Low":  [df["Close"].iloc[-1] - 1] * 6,
            "Close": [df["Close"].iloc[-1]] * 6,
            "Volume": [1000.0] * 6,
        }, index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=6, freq="D"))
        full = pd.concat([df, future])
        nifty = pd.DataFrame({"Close": [100.0] * len(full)}, index=full.index)
        alerts = evaluate_swing({"X": full}, nifty, apply_regime=False,
                                max_extension_pct=None)
        self.assertGreaterEqual(len(alerts), 1)

    def test_tight_cap_filters(self):
        df, idx = self._make_data()
        future = pd.DataFrame({
            "Open": [df["Close"].iloc[-1]] * 6,
            "High": [df["Close"].iloc[-1] + 1] * 6,
            "Low":  [df["Close"].iloc[-1] - 1] * 6,
            "Close": [df["Close"].iloc[-1]] * 6,
            "Volume": [1000.0] * 6,
        }, index=pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=6, freq="D"))
        full = pd.concat([df, future])
        nifty = pd.DataFrame({"Close": [100.0] * len(full)}, index=full.index)
        # The signal bar is ~5–8% above EMA; cap of 2% must filter it.
        alerts = evaluate_swing({"X": full}, nifty, apply_regime=False,
                                max_extension_pct=2.0)
        self.assertEqual(len(alerts), 0)


class TestBreadthSeries(unittest.TestCase):
    def test_breadth_split(self):
        idx = pd.date_range("2025-01-01", periods=30, freq="D")
        rising = pd.DataFrame({
            "Close": np.linspace(100, 130, 30),
            "Open": np.linspace(100, 130, 30),
            "High": np.linspace(101, 131, 30),
            "Low":  np.linspace(99, 129, 30),
            "Volume": np.full(30, 1000),
        }, index=idx)
        falling = pd.DataFrame({
            "Close": np.linspace(100, 70, 30),
            "Open": np.linspace(100, 70, 30),
            "High": np.linspace(101, 71, 30),
            "Low":  np.linspace(99, 69, 30),
            "Volume": np.full(30, 1000),
        }, index=idx)
        breadth = compute_breadth_series({"R": rising, "F": falling})
        # Last day: rising is well above its EMA (100%), falling is well below (0%)
        self.assertAlmostEqual(breadth.iloc[-1], 50.0, places=1)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Swing-trade backtest")
    parser.add_argument("--tests", action="store_true",
                        help="run unit tests only and exit")
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

    if args.tests:
        unittest.main(argv=[sys.argv[0], "-v"], exit=True)

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
