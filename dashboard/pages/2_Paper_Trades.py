"""Paper Trades page — table + cumulative P&L + win-rate analytics."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import db

st.set_page_config(page_title="Paper Trades · Stock Alert Bot",
                   page_icon="📒", layout="wide")
st.title("📒 Paper Trades")

# -- Filters ------------------------------------------------------------------
fc1, fc2, fc3, fc4 = st.columns([1.2, 1, 1, 1.5])
default_start = date.today() - timedelta(days=14)
date_range = fc1.date_input(
    "Date range",
    value=(default_start, date.today()),
    max_value=date.today(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d = end_d = date_range

status = fc2.selectbox(
    "Status", ["All", "OPEN", "TP1", "TP2", "SL", "TRAIL", "TIMEOUT"],
)
side = fc3.selectbox("Side", ["All", "LONG", "SHORT"])

symbol_options = ["All"] + db.distinct_symbols_in_paper_trades()
symbol_filter = fc4.selectbox("Symbol", symbol_options)

trades = db.paper_trades(
    start_date=start_d.isoformat(), end_date=end_d.isoformat(),
    status=status, side=side, symbol=symbol_filter,
)

if trades.empty:
    st.warning("No paper trades match these filters.")
    st.stop()

# -- KPI strip ----------------------------------------------------------------
closed = trades[trades["status"] != "OPEN"].copy()
closed["pnl_total"] = (
    closed["pnl_net"].fillna(0) + closed["tp1_pnl_net"].fillna(0)
)
total_pnl = closed["pnl_total"].sum()
n_total = len(closed)
n_wins = int((closed["pnl_total"] > 0).sum())
n_losses = int((closed["pnl_total"] < 0).sum())
win_rate = (n_wins / n_total * 100.0) if n_total else 0.0
gross_win = closed.loc[closed["pnl_total"] > 0, "pnl_total"].sum()
gross_loss = -closed.loc[closed["pnl_total"] < 0, "pnl_total"].sum()
profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total P&L (₹)", f"{total_pnl:+,.0f}")
k2.metric("Trades", f"{n_total} ({len(trades) - n_total} open)")
k3.metric("Win rate", f"{win_rate:.1f}%")
k4.metric("Wins / Losses", f"{n_wins} / {n_losses}")
k5.metric(
    "Profit factor",
    f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
)

st.divider()

# -- Trades table -------------------------------------------------------------
st.subheader("Trades")
display_cols = [
    "id", "symbol", "side", "status", "entry_ts", "entry_price",
    "qty", "stop_loss", "target_1", "target_2",
    "exit_ts", "exit_price", "pnl_net", "tp1_pnl_net",
    "hold_minutes", "confidence",
]
present = [c for c in display_cols if c in trades.columns]
st.dataframe(trades[present], height=380, width="stretch")

st.divider()

# -- Cumulative P&L curve -----------------------------------------------------
st.subheader("Cumulative P&L")
if not closed.empty:
    closed_sorted = closed.sort_values("exit_dt").copy()
    closed_sorted["cum_pnl"] = closed_sorted["pnl_total"].cumsum()
    fig_cum = px.line(
        closed_sorted, x="exit_dt", y="cum_pnl",
        title="Cumulative P&L (₹)",
        template="plotly_dark",
    )
    fig_cum.update_traces(line=dict(color="#26a69a", width=2))
    fig_cum.update_layout(
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Exit time", yaxis_title="Cumulative ₹",
    )
    st.plotly_chart(fig_cum, width="stretch")
else:
    st.info("No closed trades in range.")

# -- Win-rate by hour-of-day --------------------------------------------------
st.subheader("Win-rate by entry hour (IST)")
if not closed.empty:
    closed["entry_hour"] = closed["entry_dt"].dt.tz_convert("Asia/Kolkata").dt.hour
    by_hour = (
        closed.groupby("entry_hour")
        .agg(
            trades=("id", "count"),
            wins=("pnl_total", lambda s: int((s > 0).sum())),
            avg_pnl=("pnl_total", "mean"),
        )
        .reset_index()
    )
    by_hour["win_rate_pct"] = (by_hour["wins"] / by_hour["trades"] * 100).round(1)
    fig_hr = px.bar(
        by_hour, x="entry_hour", y="win_rate_pct",
        text="trades", template="plotly_dark",
        labels={
            "entry_hour": "Entry hour (IST)",
            "win_rate_pct": "Win rate %",
            "trades": "n",
        },
    )
    fig_hr.update_traces(marker_color="#42a5f5")
    fig_hr.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_hr, width="stretch")

# -- Win-rate by indicator combination ----------------------------------------
st.subheader("Win-rate by indicator combo (top 10 by sample count)")
# `notes` field on paper_trades historically carries the reasons string.
# We bucket by the comma-separated reasons fingerprint.
if "notes" in closed.columns and not closed.empty:
    closed["combo"] = (
        closed["notes"].fillna("").str.split(";").str[0].str.strip()
    )
    combo = (
        closed[closed["combo"] != ""]
        .groupby("combo")
        .agg(
            trades=("id", "count"),
            wins=("pnl_total", lambda s: int((s > 0).sum())),
            avg_pnl=("pnl_total", "mean"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
        .head(10)
    )
    if not combo.empty:
        combo["win_rate_pct"] = (combo["wins"] / combo["trades"] * 100).round(1)
        st.dataframe(combo, height=320, width="stretch")
    else:
        st.info("No combo notes available for the selected trades.")
else:
    st.info("No notes column to mine combos from.")
