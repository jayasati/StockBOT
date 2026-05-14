"""Side-by-side comparison: TradingView vs our bot.

Pulls indicator values from TradingView (via tradingview-ta — same data
TV's web UI shows) and from our bot's snapshot pipeline, then prints a
table with the delta. Highlights divergences > tolerance so you can
quickly see what's wrong.

Usage:
    python scripts/compare_indicators.py BHARTIARTL
    python scripts/compare_indicators.py BHARTIARTL --tf 5m,15m
    python scripts/compare_indicators.py --csv expected_BHARTIARTL.csv

Without --csv, prints a default indicator panel for the symbol.
With --csv, also writes bot_value/delta back into the file."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradingview_ta import TA_Handler, Interval  # noqa: E402

from bot import market_data  # noqa: E402
from indicators import compute_all  # noqa: E402


TF_TO_TV = {
    "1m":  Interval.INTERVAL_1_MINUTE,
    "5m":  Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "30m": Interval.INTERVAL_30_MINUTES,
    "60m": Interval.INTERVAL_1_HOUR,
    "1h":  Interval.INTERVAL_1_HOUR,
    "4h":  Interval.INTERVAL_4_HOURS,
    "1d":  Interval.INTERVAL_1_DAY,
}


@dataclass
class Row:
    """One indicator comparison line."""
    timeframe: str
    indicator: str
    tv_value: float | None
    bot_value: float | None
    tv_key: str = ""           # the TV-side key we read (for debugging)
    bot_key: str = ""          # the snapshot key we read (for debugging)

    @property
    def delta(self) -> float | None:
        if self.tv_value is None or self.bot_value is None:
            return None
        return self.tv_value - self.bot_value

    @property
    def pct(self) -> float | None:
        d = self.delta
        if d is None or not self.tv_value:
            return None
        return d / abs(self.tv_value) * 100.0


# ---------------------------------------------------------------------------
# TradingView side
# ---------------------------------------------------------------------------

def fetch_tv(symbol: str, tf: str) -> dict[str, float]:
    """Pull TV's analysis for symbol+timeframe. Returns flat dict of
    indicator key → value, plus convenience keys MACD_hist (= macd −
    signal) and BB_middle (= (upper+lower)/2) that TV doesn't surface
    directly. Symbol is the bare NSE symbol (no .NS)."""
    interval = TF_TO_TV[tf]
    handler = TA_Handler(
        symbol=symbol, screener="india",
        exchange="NSE", interval=interval,
    )
    a = handler.get_analysis()
    ind = dict(a.indicators)
    # Derived keys
    if "MACD.macd" in ind and "MACD.signal" in ind:
        ind["MACD_hist"] = ind["MACD.macd"] - ind["MACD.signal"]
    if "BB.upper" in ind and "BB.lower" in ind:
        ind["BB_middle"] = (ind["BB.upper"] + ind["BB.lower"]) / 2.0
        if "close" in ind:
            width = ind["BB.upper"] - ind["BB.lower"]
            ind["BB_percent_B"] = (
                (ind["close"] - ind["BB.lower"]) / width if width else None
            )
    return ind


# Map our indicator name → (TV key, our snapshot key formatter)
# TV key stays constant across timeframes (the handler is set per TF).
# Snapshot key is parameterised on tf to pick adx_5m_adx vs adx_15m_adx etc.
INDICATOR_MAP: dict[str, tuple[str, str]] = {
    # name           tv key                   snapshot-key template ({tf})
    "price":         ("close",                "__price__"),
    "bar_open":      ("open",                 ""),
    "bar_high":      ("high",                 ""),
    "bar_low":       ("low",                  ""),
    "bar_close":     ("close",                ""),
    "bar_volume":    ("volume",               ""),
    "RSI":           ("RSI",                  "rsi_{tf}"),
    "MACD":          ("MACD.macd",            "macd_{tf}_macd"),
    "MACD_signal":   ("MACD.signal",          "macd_{tf}_signal"),
    "MACD_hist":     ("MACD_hist",            "macd_{tf}_histogram"),
    "ADX":           ("ADX",                  "adx_{tf}_adx"),
    "DI_plus":       ("ADX+DI",               "adx_{tf}_di_plus"),
    "DI_minus":      ("ADX-DI",               "adx_{tf}_di_minus"),
    "Stoch_K":       ("Stoch.K",              "stochastic_{tf}_k"),
    "Stoch_D":       ("Stoch.D",              "stochastic_{tf}_d"),
    "CCI":           ("CCI20",                "cci_{tf}"),
    "BB_upper":      ("BB.upper",             "bollinger_{tf}_upper"),
    "BB_middle":     ("BB_middle",            "bollinger_{tf}_middle"),
    "BB_lower":      ("BB.lower",             "bollinger_{tf}_lower"),
    "BB_percent_B":  ("BB_percent_B",         "bollinger_{tf}_percent_b"),
    # TV doesn't directly expose ATR / MFI / CMF / Supertrend / VWAP /
    # volume_ratio in the screener payload — those rows will show "—"
    # on the TV side. Bot still computes them so they're useful to see
    # in absolute terms.
    "ATR":           ("",                     "atr_{tf}"),
    "MFI":           ("",                     "mfi_{tf}"),
    "CMF":           ("",                     "cmf_{tf}"),
    "Supertrend":    ("",                     "supertrend_{tf}_supertrend"),
    "Supertrend_dir":("",                     "supertrend_{tf}_direction"),
    "VWAP":          ("",                     "__vwap__"),
    "volume_ratio":  ("",                     "__volume_ratio__"),
    "volume_surge_ratio": ("",                "volume_surge_ratio_{tf}"),
    # Daily levels (from precompute / bot snapshot, classic pivots)
    "PDH":           ("",                     "pdh"),
    "PDL":           ("",                     "pdl"),
    "PDC":           ("",                     "pdc"),
    "pivot":         ("Pivot.M.Classic.Middle", "pivot_classic"),
    "R1":            ("Pivot.M.Classic.R1",   "pivot_classic_r1"),
    "R2":            ("Pivot.M.Classic.R2",   "pivot_classic_r2"),
    "S1":            ("Pivot.M.Classic.S1",   "pivot_classic_s1"),
    "S2":            ("Pivot.M.Classic.S2",   "pivot_classic_s2"),
}


# ---------------------------------------------------------------------------
# Bot snapshot side
# ---------------------------------------------------------------------------

def fetch_bot_snapshot(symbol_with_ns: str):
    """Build a fresh indicator snapshot for symbol (e.g. 'BHARTIARTL.NS')
    using the same code path the scanner runs. Returns (snapshot, last_price,
    intraday_df) so we can resolve VWAP / volume_ratio too."""
    intraday_data = market_data.fetch_intraday([symbol_with_ns])
    if symbol_with_ns not in intraday_data:
        return None, None, None
    bars = intraday_data[symbol_with_ns]
    daily = market_data._daily_cache.get(symbol_with_ns, pd.DataFrame())
    if bars.empty or daily.empty:
        return None, None, bars
    bars_lower = bars.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    snap = compute_all(
        symbol=symbol_with_ns,
        bars=bars_lower,
        daily_df=daily,
        session_date=bars.index[-1].date(),
        target_timeframes=("5m", "15m", "60m"),
        indicators=(
            "rsi", "atr", "adx", "macd",
            "stochastic", "mfi", "cci",
            "supertrend", "bollinger", "cmf",
            "volume_surge_ratio", "ttm_squeeze",
            "pivot_points", "previous_day_hlc", "opening_range",
        ),
    )
    last_price = float(bars["Close"].iloc[-1])
    return snap, last_price, bars


def resolve_bot_value(snapshot, intraday, last_price, daily,
                      indicator: str, tf: str) -> tuple[float | None, str]:
    """Map our indicator+timeframe to a value from snapshot/derived."""
    if indicator not in INDICATOR_MAP:
        return None, ""
    _, snap_template = INDICATOR_MAP[indicator]
    if not snap_template:
        return None, ""

    # Special-case derived values
    if snap_template == "__price__":
        return last_price, "intraday[Close].iloc[-1]"
    if snap_template == "__vwap__":
        from bot import indicators as bot_inds
        try:
            v = float(bot_inds.compute_session_vwap(intraday))
            return v, "compute_session_vwap"
        except Exception:
            return None, "compute_session_vwap (failed)"
    if snap_template == "__volume_ratio__":
        from bot import indicators as bot_inds
        try:
            v = float(bot_inds.compute_volume_ratio(intraday, daily))
            return v, "compute_volume_ratio"
        except Exception:
            return None, "compute_volume_ratio (failed)"

    key = snap_template.format(tf=tf) if "{tf}" in snap_template else snap_template
    if snapshot is None:
        return None, key
    v = snapshot.values.get(key)
    return (None if v is None else float(v)), key


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

ABS_TOLERANCE = {  # absolute delta below this = ✓
    "price": 0.5, "bar_open": 0.5, "bar_high": 0.5, "bar_low": 0.5,
    "bar_close": 0.5, "VWAP": 1.0,
    "RSI": 1.0, "ADX": 1.0, "DI_plus": 1.0, "DI_minus": 1.0,
    "MACD": 0.05, "MACD_signal": 0.05, "MACD_hist": 0.05,
    "Stoch_K": 1.0, "Stoch_D": 1.0,
    "CCI": 5.0, "MFI": 1.0,
    "BB_upper": 0.5, "BB_middle": 0.5, "BB_lower": 0.5, "BB_percent_B": 0.02,
}
DEFAULT_TOL = 0.01  # for ratios/etc. that aren't listed


def _fmt(v):
    if v is None:
        return "       —"
    if abs(v) >= 1000:
        return f"{v:>10,.2f}"
    return f"{v:>10.4f}"


def print_table(rows: list[Row]):
    print(f"{'TF':>4}  {'Indicator':<20} {'TV':>10}  {'Bot':>10}  "
          f"{'Δ (TV−Bot)':>12}  {'Δ%':>7}  Status")
    print("-" * 88)
    last_tf = None
    for r in rows:
        if r.tv_value is None and r.bot_value is None:
            continue
        if last_tf is not None and r.timeframe != last_tf:
            print()
        last_tf = r.timeframe
        delta = r.delta
        tol = ABS_TOLERANCE.get(r.indicator, DEFAULT_TOL)
        if delta is None:
            status = "(missing one side)"
        elif abs(delta) <= tol:
            status = "OK"
        else:
            status = "DIVERGE"
        delta_s = _fmt(delta)
        pct = r.pct
        pct_s = "       —" if pct is None else f"{pct:>6.2f}%"
        print(f"{r.timeframe:>4}  {r.indicator:<20} {_fmt(r.tv_value)}  "
              f"{_fmt(r.bot_value)}  {delta_s}  {pct_s}  {status}")


# ---------------------------------------------------------------------------
# CSV mode
# ---------------------------------------------------------------------------

def update_csv(path: Path, rows: list[Row]):
    """Replace bot_value & delta columns in the CSV. Preserves comment
    lines (#-prefixed) and leaves the user's expected_value alone — but
    cross-checks: if expected_value is filled in, we ALSO compute a
    second delta against TV's value for completeness."""
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    # Find header line
    header_idx = next(
        (i for i, ln in enumerate(raw_lines)
         if ln and not ln.startswith("#") and "indicator" in ln.lower()),
        None,
    )
    if header_idx is None:
        print(f"  WARN: no header row found in {path}, skipping CSV update")
        return
    header = [c.strip() for c in raw_lines[header_idx].split(",")]
    # Build a (tf, indicator) → Row index for fast lookup
    by_key = {(r.timeframe, r.indicator): r for r in rows}

    out_lines = list(raw_lines[: header_idx + 1])
    for ln in raw_lines[header_idx + 1:]:
        if not ln or ln.startswith("#"):
            out_lines.append(ln)
            continue
        cells = [c.strip() for c in ln.split(",")]
        # Pad to header width
        while len(cells) < len(header):
            cells.append("")
        cell = dict(zip(header, cells))
        tf = cell.get("timeframe", "").strip()
        ind = cell.get("indicator", "").strip()
        match = by_key.get((tf, ind))
        if match is not None:
            if "bot_value" in cell:
                cell["bot_value"] = (
                    "" if match.bot_value is None else f"{match.bot_value:.4f}"
                )
            if "delta" in cell and cell.get("expected_value", "").strip():
                try:
                    expected = float(cell["expected_value"])
                    if match.bot_value is not None:
                        cell["delta"] = f"{expected - match.bot_value:.4f}"
                except ValueError:
                    pass
        out_lines.append(",".join(cell.get(h, "") for h in header))

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"  Wrote bot_value into {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_INDICATORS = [
    "price", "VWAP",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "ADX", "DI_plus", "DI_minus",
    "Stoch_K", "Stoch_D", "CCI", "MFI",
    "BB_upper", "BB_middle", "BB_lower", "BB_percent_B",
    "ATR", "CMF", "Supertrend", "Supertrend_dir",
    "volume_ratio", "volume_surge_ratio",
    "pivot", "R1", "R2", "S1", "S2", "PDH", "PDL", "PDC",
    "bar_open", "bar_high", "bar_low", "bar_close", "bar_volume",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="BHARTIARTL",
                   help="Bare NSE symbol (no .NS), default BHARTIARTL")
    p.add_argument("--tf", default="5m,15m,60m",
                   help="Comma-separated timeframes (default 5m,15m,60m)")
    p.add_argument("--csv", help="Optional CSV to update with bot_value")
    args = p.parse_args()

    symbol_bare = args.symbol.upper().replace(".NS", "")
    symbol_ns = f"{symbol_bare}.NS"
    tfs = [tf.strip() for tf in args.tf.split(",") if tf.strip()]

    print(f"Refreshing daily cache for {symbol_ns} …")
    market_data.refresh_daily_cache_if_stale()

    print(f"Pulling TradingView data for {symbol_bare} …")
    tv_per_tf: dict[str, dict] = {}
    for tf in tfs:
        if tf not in TF_TO_TV:
            print(f"  skip unknown TF {tf}")
            continue
        try:
            tv_per_tf[tf] = fetch_tv(symbol_bare, tf)
            print(f"  TV {tf}: {len(tv_per_tf[tf])} indicators "
                  f"close={tv_per_tf[tf].get('close')}")
        except Exception as e:
            print(f"  TV {tf}: FAILED {type(e).__name__}: {e}")
            tv_per_tf[tf] = {}

    print(f"Building bot snapshot for {symbol_ns} …")
    snapshot, last_price, intraday = fetch_bot_snapshot(symbol_ns)
    daily = market_data._daily_cache.get(symbol_ns, pd.DataFrame())
    if snapshot is None:
        print("  bot snapshot FAILED (no intraday or daily data)")

    rows: list[Row] = []
    for tf in tfs:
        for ind in DEFAULT_INDICATORS:
            tv_key, _ = INDICATOR_MAP.get(ind, ("", ""))
            tv_v = tv_per_tf.get(tf, {}).get(tv_key) if tv_key else None
            bot_v, bot_key = resolve_bot_value(
                snapshot, intraday, last_price, daily, ind, tf,
            )
            rows.append(Row(
                timeframe=tf, indicator=ind,
                tv_value=tv_v, bot_value=bot_v,
                tv_key=tv_key, bot_key=bot_key,
            ))

    print(f"\n=== {symbol_bare}  comparison @ {pd.Timestamp.now()} ===\n")
    print_table(rows)

    diverge = [
        r for r in rows
        if r.delta is not None
        and abs(r.delta) > ABS_TOLERANCE.get(r.indicator, DEFAULT_TOL)
    ]
    if diverge:
        print(f"\n{len(diverge)} indicators DIVERGE beyond tolerance:")
        for r in diverge:
            print(f"  {r.timeframe} {r.indicator:20s}  "
                  f"TV={_fmt(r.tv_value)}  Bot={_fmt(r.bot_value)}  "
                  f"Δ={_fmt(r.delta)}  ({r.pct:+.2f}%)")

    if args.csv:
        update_csv(Path(args.csv), rows)


if __name__ == "__main__":
    main()
