"""
Backtest module for the stock alert bot.

Replays bot.score_stock() forward in time on historical 5-minute bars and
reports win rates at +30min, +1day, +5days. Records every alert (after
applying the bot's cooldown) to backtest_results.csv.

Usage:
    python backtest.py                  # full run, use cached parquet if present
    python backtest.py --refresh        # force re-fetch from yfinance
    python backtest.py --symbols 30     # limit to first N symbols (faster)
    python backtest.py --tests          # run unit tests only

Note on data range:
    yfinance hard-limits 5-minute data to the last ~60 calendar days. Daily
    data is fetched for the full 6 months. The replay therefore covers
    roughly the most recent 60 days of intraday history.
"""

import argparse
import logging
import sys
import unittest
from dataclasses import asdict, dataclass
from datetime import time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import bot

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

IST = ZoneInfo("Asia/Kolkata")
CACHE_DIR = Path("backtest_cache")
INTRADAY_CACHE = CACHE_DIR / "intraday_5m.parquet"
DAILY_CACHE = CACHE_DIR / "daily.parquet"
RESULTS_CSV = Path("backtest_results.csv")

SCORE_THRESHOLD = 60
WIN_THRESHOLD_PCT = 1.5
COOLDOWN_MINUTES = 60
LOOKBACK_BARS = 100  # passed into score_stock for RSI + same-session VWAP
WARMUP_BARS = 20


# ============================================================================
# Backtest-aware override of compute_volume_ratio
#
# bot.compute_volume_ratio uses datetime.now(IST).time() to compute the
# session-elapsed fraction. That's wrong for replay — we need the timestamp
# of the bar being scored, not wall-clock time. We monkey-patch the bot
# module so score_stock picks up this version.
# ============================================================================

def _bt_volume_ratio(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    if intraday.empty or daily.empty or len(daily) < 10:
        return 1.0
    last_ts = intraday.index[-1]
    today = last_ts.date()
    today_data = intraday[intraday.index.date == today]
    if today_data.empty:
        return 1.0
    today_vol = float(today_data["Volume"].sum())
    avg_daily_vol = float(daily["Volume"].tail(10).mean())
    if avg_daily_vol == 0:
        return 1.0
    last_t = last_ts.time()
    if last_t < time(9, 15):
        fraction = 0.01
    elif last_t > time(15, 30):
        fraction = 1.0
    else:
        elapsed = (last_t.hour - 9) * 60 + last_t.minute - 15
        fraction = max(0.05, elapsed / 375)
    expected = avg_daily_vol * fraction
    return float(today_vol / expected) if expected > 0 else 1.0


# ============================================================================
# Data loading: yfinance + parquet cache
# ============================================================================

def _normalize_yf_batch(
    df: pd.DataFrame, symbols: list[str], localize_ist: bool
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            sub = df[sym].copy() if isinstance(df.columns, pd.MultiIndex) else df.copy()
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        if localize_ist:
            if sub.index.tz is None:
                sub.index = sub.index.tz_localize("UTC").tz_convert(IST)
            else:
                sub.index = sub.index.tz_convert(IST)
        sub.index.name = "timestamp"
        out[sym] = sub
    return out


def _save_parquet(data: dict[str, pd.DataFrame], path: Path) -> None:
    if not data:
        return
    pieces = []
    for sym, df in data.items():
        d = df.reset_index()
        d["symbol"] = sym
        pieces.append(d)
    combined = pd.concat(pieces, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, engine="pyarrow", index=False)


def _load_parquet(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    combined = pd.read_parquet(path, engine="pyarrow")
    out: dict[str, pd.DataFrame] = {}
    for sym, group in combined.groupby("symbol"):
        df = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        out[sym] = df
    return out


def fetch_intraday_5m(
    symbols: list[str], days: int = 60, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    if use_cache and INTRADAY_CACHE.exists():
        log.info("Loading 5m data from cache: %s", INTRADAY_CACHE)
        cached = _load_parquet(INTRADAY_CACHE)
        if cached:
            log.info("  %d symbols loaded from cache", len(cached))
            return cached
    log.info("Fetching %d days of 5m data for %d symbols...", days, len(symbols))
    df = yf.download(
        tickers=symbols, period=f"{days}d", interval="5m",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )
    data = _normalize_yf_batch(df, symbols, localize_ist=True)
    log.info("Got 5m data for %d/%d symbols", len(data), len(symbols))
    _save_parquet(data, INTRADAY_CACHE)
    return data


def fetch_daily(
    symbols: list[str], months: int = 6, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    if use_cache and DAILY_CACHE.exists():
        log.info("Loading daily data from cache: %s", DAILY_CACHE)
        cached = _load_parquet(DAILY_CACHE)
        if cached:
            log.info("  %d symbols loaded from cache", len(cached))
            return cached
    days = months * 31
    log.info("Fetching %d days of daily data for %d symbols...", days, len(symbols))
    df = yf.download(
        tickers=symbols, period=f"{days}d", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False, threads=True,
    )
    data = _normalize_yf_batch(df, symbols, localize_ist=False)
    log.info("Got daily data for %d/%d symbols", len(data), len(symbols))
    _save_parquet(data, DAILY_CACHE)
    return data


# ============================================================================
# Replay
# ============================================================================

@dataclass
class AlertRecord:
    symbol: str
    timestamp: pd.Timestamp
    score: int
    reasons: str
    entry_price: float
    price_30m: Optional[float]
    price_1d: Optional[float]
    price_5d: Optional[float]
    ret_30m: Optional[float]
    ret_1d: Optional[float]
    ret_5d: Optional[float]


def _forward_intraday_price(
    intraday: pd.DataFrame, t: pd.Timestamp, offset: timedelta
) -> Optional[float]:
    target = t + offset
    future = intraday[intraday.index >= target]
    if len(future) == 0:
        return None
    # Reject if the next bar is more than 1 trading day away (gap over weekend
    # makes "+30min" meaningless).
    if (future.index[0] - t).total_seconds() > 6 * 3600:
        return None
    return float(future["Close"].iloc[0])


def _forward_daily_price(
    daily: pd.DataFrame, t: pd.Timestamp, n_days: int
) -> Optional[float]:
    t_date = t.date() if hasattr(t, "date") else t
    future = daily[daily.index.date > t_date]
    if len(future) >= n_days:
        return float(future["Close"].iloc[n_days - 1])
    return None


def _can_skip_bar(
    intraday: pd.DataFrame, daily_slice: pd.DataFrame, i: int, t: pd.Timestamp,
    threshold: int = SCORE_THRESHOLD,
) -> bool:
    """
    Pre-filter for the post-tuning scoring (see bot.score_stock):
      max without VR>=1.5 = pullback(25) + RSI(15) + VWAP(15) = 55
      max without VR>=1.5 and without pullback = 30

    For threshold >= 60: must have VR>=1.5.
    For threshold < 60 (sweep): must have VR>=1.5 OR pullback in [-3%, 0%]
    of the 20-day high.
    """
    if len(daily_slice) < 20:
        return True

    recent_high = float(daily_slice["High"].tail(20).max())
    current = float(intraday["Close"].iloc[i])
    pct_from_high = (current - recent_high) / recent_high * 100
    in_pullback_zone = -3.0 <= pct_from_high <= 0.0

    t_date = t.date()
    today_bars = intraday.iloc[max(0, i - 100): i + 1]
    today_bars = today_bars[today_bars.index.date == t_date]
    if today_bars.empty:
        return True
    today_vol = float(today_bars["Volume"].sum())
    avg_vol = float(daily_slice["Volume"].tail(10).mean())
    if avg_vol == 0:
        return True
    elapsed = (t.hour - 9) * 60 + t.minute - 15
    fraction = max(0.05, elapsed / 375) if elapsed >= 0 else 0.05
    quick_vr = today_vol / (avg_vol * fraction)
    high_vr = quick_vr >= 1.5

    if threshold >= 60:
        return not high_vr
    return not (high_vr or in_pullback_zone)


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def replay(
    symbols: list[str],
    intraday_data: dict[str, pd.DataFrame],
    daily_data: dict[str, pd.DataFrame],
    threshold: int = SCORE_THRESHOLD,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> list[AlertRecord]:
    import time as _time
    records: list[AlertRecord] = []
    cooldown_td = timedelta(minutes=cooldown_minutes)
    n_total = len(symbols)
    t_start = _time.monotonic()

    log.info("Replaying %d symbols (threshold=%d, cooldown=%dm)...",
             n_total, threshold, cooldown_minutes)

    for idx, sym in enumerate(symbols, 1):
        sym_start = _time.monotonic()
        intraday = intraday_data.get(sym)
        daily = daily_data.get(sym)
        if intraday is None or daily is None:
            log.info("[%3d/%d] %-14s SKIP (no data)", idx, n_total, sym)
            continue
        if len(intraday) < WARMUP_BARS or len(daily) < 25:
            log.info("[%3d/%d] %-14s SKIP (insufficient history: %d bars / %d days)",
                     idx, n_total, sym, len(intraday), len(daily))
            continue

        sym_alerts_before = len(records)
        last_alert: Optional[pd.Timestamp] = None
        intraday_dates = intraday.index.date  # cache once per symbol
        daily_dates = daily.index.date

        for i in range(WARMUP_BARS, len(intraday)):
            t = intraday.index[i]
            t_date = intraday_dates[i]
            daily_mask = daily_dates < t_date
            daily_slice = daily.loc[daily_mask]
            if len(daily_slice) < 20:
                continue

            if _can_skip_bar(intraday, daily_slice, i, t, threshold=threshold):
                continue

            slice_start = max(0, i - LOOKBACK_BARS)
            intraday_slice = intraday.iloc[slice_start: i + 1]

            signals = bot.score_stock(sym, intraday_slice, daily_slice)
            if signals.score < threshold:
                continue
            if last_alert is not None and (t - last_alert) < cooldown_td:
                continue
            last_alert = t

            entry = signals.price
            p30 = _forward_intraday_price(intraday, t, timedelta(minutes=30))
            p1d = _forward_daily_price(daily, t, 1)
            p5d = _forward_daily_price(daily, t, 5)

            def _ret(p: Optional[float]) -> Optional[float]:
                return (p - entry) / entry * 100 if p is not None else None

            records.append(AlertRecord(
                symbol=sym,
                timestamp=t,
                score=signals.score,
                reasons=", ".join(signals.reasons),
                entry_price=entry,
                price_30m=p30,
                price_1d=p1d,
                price_5d=p5d,
                ret_30m=_ret(p30),
                ret_1d=_ret(p1d),
                ret_5d=_ret(p5d),
            ))

        sym_dur = _time.monotonic() - sym_start
        sym_alerts = len(records) - sym_alerts_before
        elapsed = _time.monotonic() - t_start
        avg_per_sym = elapsed / idx
        eta = avg_per_sym * (n_total - idx)
        pct = idx / n_total * 100
        log.info(
            "[%3d/%d %5.1f%%] %-14s %3d alerts (%d bars in %4.1fs)  "
            "total=%d  elapsed=%s  eta=%s",
            idx, n_total, pct, sym, sym_alerts, len(intraday), sym_dur,
            len(records), _fmt_eta(elapsed), _fmt_eta(eta),
        )

    log.info("Replay complete: %d alerts in %s.",
             len(records), _fmt_eta(_time.monotonic() - t_start))
    return records


# ============================================================================
# Reporting
# ============================================================================

def compute_win_rate(
    returns: list[Optional[float]], threshold_pct: float = WIN_THRESHOLD_PCT
) -> float:
    valid = [r for r in returns if r is not None and not (isinstance(r, float) and np.isnan(r))]
    if not valid:
        return 0.0
    wins = sum(1 for r in valid if r >= threshold_pct)
    return wins / len(valid) * 100


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:+.2f}%" if x is not None and not np.isnan(x) else "  n/a"


def summarize(records: list[AlertRecord]) -> str:
    if not records:
        return "No alerts generated. Try lowering --threshold or check data coverage."

    df = pd.DataFrame([asdict(r) for r in records])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    out: list[str] = []
    bar = "=" * 76
    out.append("")
    out.append(bar)
    out.append(f"BACKTEST REPORT — {len(records)} alerts "
               f"(score >= {SCORE_THRESHOLD}, cooldown {COOLDOWN_MINUTES}m)")
    out.append(f"Period: {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    out.append(f"Win threshold: >= {WIN_THRESHOLD_PCT}% move")
    out.append(bar)

    out.append("")
    out.append(f"{'Horizon':<10} {'Count':>7} {'WinRate':>10} {'AvgRet':>10} "
               f"{'MedRet':>10} {'StdDev':>10}")
    out.append("-" * 60)
    for label, col in [("+30min", "ret_30m"), ("+1 day", "ret_1d"), ("+5 days", "ret_5d")]:
        rets = df[col].dropna().tolist()
        if not rets:
            out.append(f"{label:<10} {0:>7} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        wr = compute_win_rate(rets)
        avg = float(np.mean(rets))
        med = float(np.median(rets))
        std = float(np.std(rets))
        out.append(f"{label:<10} {len(rets):>7} {wr:>9.1f}% "
                   f"{avg:>9.2f}% {med:>9.2f}% {std:>9.2f}%")

    out.append("")
    out.append("BEST 5 TRADES (by +1d return):")
    have_1d = df[df["ret_1d"].notna()]
    for _, r in have_1d.nlargest(5, "ret_1d").iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        out.append(f"  {r['symbol']:<14} {ts}  score={r['score']:>3}  "
                   f"+1d={_fmt_pct(r['ret_1d'])}  [{r['reasons']}]")

    out.append("")
    out.append("WORST 5 TRADES (by +1d return):")
    for _, r in have_1d.nsmallest(5, "ret_1d").iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        out.append(f"  {r['symbol']:<14} {ts}  score={r['score']:>3}  "
                   f"+1d={_fmt_pct(r['ret_1d'])}  [{r['reasons']}]")

    # Per-component analysis: each row in the reasons string contains tokens
    # like "VR 3.2x", "RSI 64", "> VWAP", "BO +1.5%", "near high". We bucket
    # alerts by which components fired, then compare +1d win rate of "with"
    # vs "without" each component.
    out.append("")
    out.append("PER-COMPONENT CONTRIBUTION  (+1d return, win >= "
               f"{WIN_THRESHOLD_PCT}%):")
    out.append(f"  {'Component':<16} {'With_n':>7} {'WR_with':>9} "
               f"{'Without_n':>10} {'WR_without':>11} {'Lift':>8}")
    out.append("  " + "-" * 70)

    components = {
        "Volume ratio": lambda s: "VR " in s,
        "RSI in zone":  lambda s: "RSI " in s,
        "Above VWAP":   lambda s: "VWAP" in s,
        "Pullback":     lambda s: "pullback" in s,
        "Extended":     lambda s: "extended" in s,
    }
    df_clean = df[df["ret_1d"].notna()].copy()
    for name, predicate in components.items():
        mask = df_clean["reasons"].apply(predicate)
        with_g = df_clean.loc[mask, "ret_1d"].tolist()
        without_g = df_clean.loc[~mask, "ret_1d"].tolist()
        wr_w = compute_win_rate(with_g) if with_g else 0.0
        wr_o = compute_win_rate(without_g) if without_g else 0.0
        lift = wr_w - wr_o
        out.append(f"  {name:<16} {len(with_g):>7} {wr_w:>8.1f}% "
                   f"{len(without_g):>10} {wr_o:>10.1f}% {lift:>+7.1f}pp")

    out.append("")
    out.append(f"All alerts saved to: {RESULTS_CSV}")
    out.append(bar)
    return "\n".join(out)


def save_csv(records: list[AlertRecord], path: Path) -> None:
    if not records:
        log.warning("No records to save.")
        return
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(path, index=False)
    log.info("Saved %d alerts to %s", len(df), path)


def apply_cooldown(
    records: list[AlertRecord], threshold: int, cooldown_minutes: int
) -> list[AlertRecord]:
    """Filter records to score >= threshold, then apply per-symbol cooldown."""
    cooldown_td = timedelta(minutes=cooldown_minutes)
    by_symbol: dict[str, list[AlertRecord]] = {}
    for r in records:
        if r.score < threshold:
            continue
        by_symbol.setdefault(r.symbol, []).append(r)
    out: list[AlertRecord] = []
    for rs in by_symbol.values():
        rs.sort(key=lambda r: r.timestamp)
        last: Optional[pd.Timestamp] = None
        for r in rs:
            if last is not None and (r.timestamp - last) < cooldown_td:
                continue
            last = r.timestamp
            out.append(r)
    out.sort(key=lambda r: r.timestamp)
    return out


def sweep_summary(
    all_records: list[AlertRecord], thresholds: list[int]
) -> str:
    """Print a comparison table across score thresholds.

    Sanity check: a well-calibrated score should produce HIGHER win rates as
    the threshold rises, even if absolute win rates are weak.
    """
    out: list[str] = []
    bar = "=" * 78
    out.append("")
    out.append(bar)
    out.append(f"THRESHOLD SWEEP  ({len(all_records)} raw events scored, "
               f"cooldown {COOLDOWN_MINUTES}m applied per threshold)")
    out.append(bar)
    out.append(f"{'Threshold':>10} {'Alerts':>8} {'WR_30m':>9} {'WR_1d':>9} "
               f"{'WR_5d':>9} {'AvgRet_1d':>11} {'AvgRet_5d':>11}")
    out.append("-" * 78)
    for thr in thresholds:
        recs = apply_cooldown(all_records, thr, COOLDOWN_MINUTES)
        if not recs:
            out.append(f"{thr:>10} {0:>8}   (no alerts at this threshold)")
            continue
        ret30 = [r.ret_30m for r in recs if r.ret_30m is not None]
        ret1 = [r.ret_1d for r in recs if r.ret_1d is not None]
        ret5 = [r.ret_5d for r in recs if r.ret_5d is not None]
        wr30 = compute_win_rate(ret30) if ret30 else 0.0
        wr1 = compute_win_rate(ret1) if ret1 else 0.0
        wr5 = compute_win_rate(ret5) if ret5 else 0.0
        avg1 = float(np.mean(ret1)) if ret1 else 0.0
        avg5 = float(np.mean(ret5)) if ret5 else 0.0
        out.append(f"{thr:>10} {len(recs):>8} {wr30:>8.1f}% {wr1:>8.1f}% "
                   f"{wr5:>8.1f}% {avg1:>10.2f}% {avg5:>10.2f}%")
    out.append("")
    out.append("Read: a calibrated score gives rising win rates as threshold")
    out.append("rises. Flat or falling = the score is not ranking trades.")
    out.append(bar)
    return "\n".join(out)


# ============================================================================
# Unit tests
# ============================================================================

class TestWinRate(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(compute_win_rate([]), 0.0)

    def test_all_wins(self):
        self.assertEqual(compute_win_rate([2.0, 3.0, 5.0]), 100.0)

    def test_all_losses(self):
        self.assertEqual(compute_win_rate([-1.0, -2.0, 0.5]), 0.0)

    def test_mixed_50_percent(self):
        self.assertAlmostEqual(compute_win_rate([2.0, -1.0, 3.0, 0.5]), 50.0)

    def test_threshold_boundary_inclusive(self):
        # exactly 1.5% counts as a win
        self.assertEqual(compute_win_rate([1.5, 1.49]), 50.0)

    def test_custom_threshold(self):
        self.assertEqual(compute_win_rate([2.0, 3.0], threshold_pct=2.5), 50.0)

    def test_drops_none(self):
        # None entries are valid trades that lacked a forward bar — they
        # should be excluded from the denominator.
        self.assertAlmostEqual(compute_win_rate([2.0, None, 1.0]), 50.0)

    def test_drops_nan(self):
        self.assertAlmostEqual(compute_win_rate([2.0, float("nan"), 1.0]), 50.0)

    def test_only_nones(self):
        self.assertEqual(compute_win_rate([None, None]), 0.0)


class TestForwardPrice(unittest.TestCase):
    def test_intraday_30m_offset(self):
        idx = pd.date_range("2025-01-15 09:15", periods=20, freq="5min", tz=IST)
        intraday = pd.DataFrame({"Close": np.arange(100.0, 120.0)}, index=idx)
        # bar at 09:40 (i=5); +30m = 10:10 = i=11 = 111
        result = _forward_intraday_price(intraday, idx[5], timedelta(minutes=30))
        self.assertEqual(result, 111.0)

    def test_intraday_no_future_bar(self):
        idx = pd.date_range("2025-01-15 09:15", periods=5, freq="5min", tz=IST)
        intraday = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)
        result = _forward_intraday_price(intraday, idx[-1], timedelta(minutes=30))
        self.assertIsNone(result)

    def test_intraday_rejects_overnight_gap(self):
        # 5m bar at end-of-day, next bar is next morning → +30m crosses
        # 17 hours of close. Should reject (return None).
        idx = pd.DatetimeIndex([
            pd.Timestamp("2025-01-15 15:25", tz=IST),
            pd.Timestamp("2025-01-16 09:15", tz=IST),
        ])
        intraday = pd.DataFrame({"Close": [100, 101]}, index=idx)
        result = _forward_intraday_price(intraday, idx[0], timedelta(minutes=30))
        self.assertIsNone(result)

    def test_daily_offset_1(self):
        days = pd.date_range("2025-01-10", periods=10, freq="D")
        daily = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=days)
        # t=01-12; +1d = 01-13 → 103.0
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-12"), 1)
        self.assertEqual(result, 103.0)

    def test_daily_offset_5(self):
        days = pd.date_range("2025-01-10", periods=10, freq="D")
        daily = pd.DataFrame({"Close": np.arange(100.0, 110.0)}, index=days)
        # t=01-12; +5d = 01-17 → 107.0
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-12"), 5)
        self.assertEqual(result, 107.0)

    def test_daily_insufficient_history(self):
        days = pd.date_range("2025-01-10", periods=3, freq="D")
        daily = pd.DataFrame({"Close": [100, 101, 102]}, index=days)
        result = _forward_daily_price(daily, pd.Timestamp("2025-01-10"), 5)
        self.assertIsNone(result)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest bot.score_stock()")
    parser.add_argument("--tests", action="store_true",
                        help="run unit tests only and exit")
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

    if args.tests:
        unittest.main(argv=[sys.argv[0], "-v"], exit=True)

    # Make replay deterministic — score_stock will call our patched version
    # instead of the wall-clock-dependent one.
    bot.compute_volume_ratio = _bt_volume_ratio

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
