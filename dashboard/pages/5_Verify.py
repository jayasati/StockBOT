"""Verify page — batch TV-vs-bot indicator comparison across many symbols."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import db
from dashboard.verify_helper import (
    INDICATOR_MAP, compare_many_symbols,
)

st.set_page_config(page_title="Verify · Stock Alert Bot",
                   page_icon="✅", layout="wide")
st.title("✅ TradingView vs Bot — Indicator Verify")

st.caption(
    "Pulls indicator values from TradingView (via `tradingview-ta`) and "
    "compares to the bot's `compute_all` snapshot. Use this to confirm "
    "a divergence is real before chasing a phantom math bug. "
    "First call per symbol takes ~5s (TV API + bot snapshot)."
)

# -- Symbol picker ------------------------------------------------------------
audit = db.latest_audit_per_symbol()
default_picks = ["BHARTIARTL", "MCX", "BSE"]
all_symbols_bare = (
    sorted(audit["symbol"].str.replace(".NS", "", regex=False).tolist())
    if not audit.empty else default_picks
)

cc1, cc2, cc3 = st.columns([2, 1, 1])
chosen_symbols = cc1.multiselect(
    "Symbols (max 8 recommended — TV calls are sequential)",
    all_symbols_bare,
    default=[s for s in default_picks if s in all_symbols_bare][:3],
    max_selections=15,
)
timeframe = cc2.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=0)
indicator_picks = cc3.multiselect(
    "Indicators",
    list(INDICATOR_MAP.keys()),
    default=["price", "RSI", "MACD_hist", "ADX", "BB_percent_B", "CCI"],
)

run = st.button("Run verify", type="primary", disabled=not chosen_symbols)

if not run:
    st.info(
        "Pick symbols + timeframe + indicators above, then hit Run verify. "
        "OK = within tolerance, DIVERGE = check the data source / math."
    )
    st.stop()

# -- Run ---------------------------------------------------------------------
with st.spinner(
    f"Comparing {len(chosen_symbols)} symbol(s) on {timeframe} "
    f"({len(indicator_picks)} indicator(s) each)…"
):
    df = compare_many_symbols(
        chosen_symbols, timeframe, indicator_picks or None,
    )

if df.empty:
    st.warning("No comparable rows returned.")
    st.stop()

# -- Summary -----------------------------------------------------------------
total = len(df)
diverge = int((df["status"] == "DIVERGE").sum())
ok = int((df["status"] == "OK").sum())
errs = int(df["status"].astype(str).str.startswith("ERR").sum())
unknown = total - diverge - ok - errs

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total comparisons", total)
c2.metric("OK", ok)
c3.metric("DIVERGE", diverge,
          delta=f"-{diverge}" if diverge else None,
          delta_color="inverse")
c4.metric("Errors / —", errs + unknown)

st.divider()

# -- Pivot view (symbol vs indicator) ----------------------------------------
st.subheader(f"Side-by-side ({timeframe})")

def _row_style(r):
    s = r.get("status", "")
    if s == "DIVERGE":
        return ["background-color: #5d4037"] * len(r)
    if s == "OK":
        return ["background-color: #1b5e20"] * len(r)
    if str(s).startswith("ERR"):
        return ["background-color: #b71c1c"] * len(r)
    return [""] * len(r)


styled = df.style.apply(_row_style, axis=1).format({
    "tv": "{:.4f}", "bot": "{:.4f}",
    "delta": "{:+.4f}", "pct": "{:+.2f}%",
}, na_rep="—")
st.dataframe(styled, height=600, width="stretch")

# -- Diverging rows highlighted ----------------------------------------------
if diverge:
    st.subheader("Diverging only")
    st.dataframe(
        df[df["status"] == "DIVERGE"]
        .sort_values("pct", key=lambda s: s.abs(), ascending=False),
        height=400, width="stretch",
    )

st.caption(
    f"Page rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST. "
    f"Indicators with no TV-side value (ATR, CMF, MFI, Supertrend, VWAP, "
    f"volume_ratio) show '—' on the TV column — TV's screener API does "
    f"not expose them."
)
