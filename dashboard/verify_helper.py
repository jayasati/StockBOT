"""TV-vs-bot indicator comparison — shared between the Verify page and
the Live drill-down. Reuses the mapping from scripts/compare_indicators.py
but returns a tidy DataFrame the Streamlit pages can render directly."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingview_ta import TA_Handler, Interval  # noqa: E402

from bot import market_data  # noqa: E402
from indicators import compute_all  # noqa: E402


TF_TO_TV = {
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "60m": Interval.INTERVAL_1_HOUR,
    "1d": Interval.INTERVAL_1_DAY,
}

# Indicator name → (TV key, snapshot key template). Mirror of
# scripts/compare_indicators.py:INDICATOR_MAP.
INDICATOR_MAP: dict[str, tuple[str, str]] = {
    "price":         ("close",                "__price__"),
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
    "ATR":           ("",                     "atr_{tf}"),
    "MFI":           ("",                     "mfi_{tf}"),
    "CMF":           ("",                     "cmf_{tf}"),
    "Supertrend":    ("",                     "supertrend_{tf}_supertrend"),
    "Supertrend_dir":("",                     "supertrend_{tf}_direction"),
    "VWAP":          ("",                     "__vwap__"),
    "volume_ratio":  ("",                     "__volume_ratio__"),
}

ABS_TOL = {
    "price": 0.5, "VWAP": 1.0,
    "RSI": 1.0, "ADX": 1.0, "DI_plus": 1.0, "DI_minus": 1.0,
    "MACD": 0.05, "MACD_signal": 0.05, "MACD_hist": 0.05,
    "Stoch_K": 1.0, "Stoch_D": 1.0,
    "CCI": 5.0, "MFI": 1.0,
    "BB_upper": 0.5, "BB_middle": 0.5, "BB_lower": 0.5, "BB_percent_B": 0.02,
}


@dataclass
class VerifyResult:
    rows: pd.DataFrame
    diverge_count: int


def _fetch_tv(symbol: str, tf: str) -> dict[str, float]:
    handler = TA_Handler(
        symbol=symbol, screener="india",
        exchange="NSE", interval=TF_TO_TV[tf],
    )
    a = handler.get_analysis()
    ind = dict(a.indicators)
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


def _bot_snapshot(symbol_yf: str):
    intraday_data = market_data.fetch_intraday([symbol_yf])
    if symbol_yf not in intraday_data:
        return None, None, None
    bars = intraday_data[symbol_yf]
    daily = market_data._daily_cache.get(symbol_yf, pd.DataFrame())
    if bars.empty or daily.empty:
        return None, None, bars
    bars_lower = bars.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    snap = compute_all(
        symbol=symbol_yf,
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


def _resolve_bot(snapshot, intraday, last_price, daily, ind: str, tf: str):
    _, snap_template = INDICATOR_MAP.get(ind, ("", ""))
    if not snap_template:
        return None
    if snap_template == "__price__":
        return last_price
    if snap_template == "__vwap__":
        from bot import indicators as bi
        try:
            return float(bi.compute_session_vwap(intraday))
        except Exception:
            return None
    if snap_template == "__volume_ratio__":
        from bot import indicators as bi
        try:
            return float(bi.compute_volume_ratio(intraday, daily))
        except Exception:
            return None
    if snapshot is None:
        return None
    key = snap_template.format(tf=tf) if "{tf}" in snap_template else snap_template
    v = snapshot.values.get(key)
    return None if v is None else float(v)


def compare_one_symbol(
    symbol_bare: str, timeframes: list[str],
    indicators: list[str] | None = None,
) -> VerifyResult:
    """Run a TV-vs-bot comparison for a single bare NSE symbol across
    the given timeframes. Returns a tidy DataFrame with columns
    ``timeframe, indicator, tv, bot, delta, pct, status``."""
    if indicators is None:
        indicators = list(INDICATOR_MAP.keys())
    symbol_yf = f"{symbol_bare}.NS"

    # Refresh daily cache once (no-op if already today)
    market_data.refresh_daily_cache_if_stale()

    tv_per_tf: dict[str, dict] = {}
    for tf in timeframes:
        if tf not in TF_TO_TV:
            continue
        try:
            tv_per_tf[tf] = _fetch_tv(symbol_bare, tf)
        except Exception as e:
            tv_per_tf[tf] = {"__error__": f"{type(e).__name__}: {e}"}

    snapshot, last_price, intraday = _bot_snapshot(symbol_yf)
    daily = market_data._daily_cache.get(symbol_yf, pd.DataFrame())

    rows = []
    for tf in timeframes:
        for ind in indicators:
            tv_key, _ = INDICATOR_MAP.get(ind, ("", ""))
            tv_v = tv_per_tf.get(tf, {}).get(tv_key) if tv_key else None
            bot_v = _resolve_bot(
                snapshot, intraday, last_price, daily, ind, tf,
            )
            if tv_v is None and bot_v is None:
                continue
            delta = (tv_v - bot_v) if (tv_v is not None and bot_v is not None) else None
            pct = (delta / abs(tv_v) * 100.0) if (delta is not None and tv_v) else None
            tol = ABS_TOL.get(ind, 0.01)
            status = (
                "—" if delta is None
                else ("OK" if abs(delta) <= tol else "DIVERGE")
            )
            rows.append({
                "timeframe": tf,
                "indicator": ind,
                "tv": tv_v,
                "bot": bot_v,
                "delta": delta,
                "pct": pct,
                "status": status,
            })
    df = pd.DataFrame(rows)
    diverge = int((df["status"] == "DIVERGE").sum()) if not df.empty else 0
    return VerifyResult(rows=df, diverge_count=diverge)


def compare_many_symbols(
    symbols_bare: list[str], timeframe: str,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side comparison across many symbols on ONE timeframe.
    Returns a wide DataFrame: row per (symbol, indicator), columns for
    tv/bot/delta/pct/status."""
    out = []
    for sym in symbols_bare:
        try:
            r = compare_one_symbol(sym, [timeframe], indicators)
            df = r.rows.copy()
            df.insert(0, "symbol", sym)
            out.append(df)
        except Exception as e:
            out.append(pd.DataFrame([{
                "symbol": sym, "timeframe": timeframe,
                "indicator": "(error)", "tv": None, "bot": None,
                "delta": None, "pct": None,
                "status": f"ERR: {type(e).__name__}",
            }]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
