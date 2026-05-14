"""Performance page — equity curve, drawdown, Sharpe, leaderboard,
calendar heatmap of daily P&L."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import db

st.set_page_config(page_title="Performance · Stock Alert Bot",
                   page_icon="📈", layout="wide")
st.title("📈 Performance")

# Pull a generous window — page is for analysis, not real-time
default_start = date.today() - timedelta(days=180)
date_range = st.date_input(
    "Range", value=(default_start, date.today()), max_value=date.today(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d = end_d = date_range

trades = db.paper_trades(
    start_date=start_d.isoformat(), end_date=end_d.isoformat(),
    status=None, side=None, symbol=None,
)
closed = trades[trades["status"] != "OPEN"].copy()
if closed.empty:
    st.warning("No closed trades in range.")
    st.stop()
closed["pnl_total"] = (
    closed["pnl_net"].fillna(0) + closed["tp1_pnl_net"].fillna(0)
)
closed = closed.sort_values("exit_dt").reset_index(drop=True)
closed["cum_pnl"] = closed["pnl_total"].cumsum()
closed["exit_date"] = closed["exit_dt"].dt.tz_convert("Asia/Kolkata").dt.date

# -- Headline KPIs ------------------------------------------------------------
total_pnl = float(closed["pnl_total"].sum())
n = len(closed)
wins = int((closed["pnl_total"] > 0).sum())
losses = int((closed["pnl_total"] < 0).sum())
win_rate = wins / n * 100.0 if n else 0.0
gross_win = closed.loc[closed["pnl_total"] > 0, "pnl_total"].sum()
gross_loss = -closed.loc[closed["pnl_total"] < 0, "pnl_total"].sum()
profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
avg_win = closed.loc[closed["pnl_total"] > 0, "pnl_total"].mean() if wins else 0
avg_loss = closed.loc[closed["pnl_total"] < 0, "pnl_total"].mean() if losses else 0
expectancy = (wins / n * avg_win + losses / n * avg_loss) if n else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total P&L (₹)", f"{total_pnl:+,.0f}")
k2.metric("Trades", f"{n}")
k3.metric("Win rate", f"{win_rate:.1f}%")
k4.metric(
    "Profit factor",
    f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
)
k5.metric("Expectancy ₹/trade", f"{expectancy:+,.0f}")

st.divider()

# -- Equity curve + drawdown --------------------------------------------------
st.subheader("Equity curve & drawdown")
running_max = closed["cum_pnl"].cummax()
closed["drawdown"] = closed["cum_pnl"] - running_max
max_dd = float(closed["drawdown"].min())

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=closed["exit_dt"], y=closed["cum_pnl"], name="Equity",
    line=dict(color="#26a69a", width=2),
))
fig.add_trace(go.Scatter(
    x=closed["exit_dt"], y=closed["drawdown"], name="Drawdown",
    line=dict(color="#ef5350", width=1), yaxis="y2",
    fill="tozeroy",
))
fig.update_layout(
    template="plotly_dark", height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="Cumulative ₹"),
    yaxis2=dict(title="Drawdown ₹", overlaying="y", side="right",
                showgrid=False),
    legend=dict(orientation="h", yanchor="top", y=1.05,
                xanchor="left", x=0),
)
st.plotly_chart(fig, width="stretch")
st.caption(f"Max drawdown over range: ₹{max_dd:,.0f}")

# -- Rolling Sharpe -----------------------------------------------------------
st.subheader("Rolling 30-day Sharpe")
daily = (
    closed.groupby("exit_date")["pnl_total"]
    .sum().rename("daily_pnl").to_frame()
)
daily.index = pd.to_datetime(daily.index)
# Reindex to fill missing trading days (no trades = 0 P&L)
full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
daily = daily.reindex(full_idx).fillna(0)
rolling = daily["daily_pnl"].rolling(30)
sharpe = (rolling.mean() / rolling.std() * np.sqrt(252)).fillna(0)

fig_s = go.Figure(go.Scatter(
    x=sharpe.index, y=sharpe, mode="lines",
    line=dict(color="#ab47bc", width=1.5), name="Sharpe (30d)",
))
fig_s.add_hline(y=1.0, line=dict(color="#ffffff", dash="dot", width=1))
fig_s.update_layout(
    template="plotly_dark", height=300,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Sharpe (annualised)",
)
st.plotly_chart(fig_s, width="stretch")

# -- Per-stock leaderboard ----------------------------------------------------
st.subheader("Per-stock leaderboard")
by_sym = (
    closed.groupby("symbol")
    .agg(
        trades=("id", "count"),
        wins=("pnl_total", lambda s: int((s > 0).sum())),
        total_pnl=("pnl_total", "sum"),
        avg_pnl=("pnl_total", "mean"),
    )
    .reset_index()
)
by_sym["win_rate_pct"] = (by_sym["wins"] / by_sym["trades"] * 100).round(1)
by_sym = by_sym.sort_values("total_pnl", ascending=False)

cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Top 10 winners**")
    st.dataframe(by_sym.head(10), height=340, width="stretch")
with cc2:
    st.markdown("**Top 10 losers**")
    st.dataframe(
        by_sym.tail(10).iloc[::-1], height=340, width="stretch",
    )

# -- Daily P&L heatmap --------------------------------------------------------
st.subheader("Daily P&L calendar")
cal = daily.copy().reset_index().rename(columns={"index": "date"})
cal["date"] = pd.to_datetime(cal["date"])
cal["weekday"] = cal["date"].dt.day_name()
cal["week"] = cal["date"].dt.isocalendar().week
cal["year_week"] = cal["date"].dt.strftime("%G-W%V")

# Wide format for heatmap
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
pivot = cal.pivot_table(
    index="weekday", columns="year_week",
    values="daily_pnl", aggfunc="sum",
).reindex(weekday_order)

fig_h = px.imshow(
    pivot,
    color_continuous_scale=[
        (0.0, "#b71c1c"), (0.5, "#1a1d24"), (1.0, "#1b5e20"),
    ],
    aspect="auto", template="plotly_dark",
    labels=dict(color="Daily ₹"),
)
fig_h.update_layout(
    height=240, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="ISO week", yaxis_title="",
)
st.plotly_chart(fig_h, width="stretch")
