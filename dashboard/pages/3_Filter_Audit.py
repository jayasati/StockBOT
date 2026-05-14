"""Filter Audit page — today's killed signals grouped by reason.

Helps you spot a filter that's too aggressive (killing 90% of valid
signals → tune in `scoring.yaml` or `filters/soft.py`)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import db

st.set_page_config(page_title="Filter Audit · Stock Alert Bot",
                   page_icon="🛡️", layout="wide")
st.title("🛡️ Filter Audit")

session_d = st.date_input(
    "Session", value=date.today(), max_value=date.today(),
).isoformat()

killed = db.killed_signals_today(session_d)
if killed.empty:
    st.success(
        f"No hard-killed signals on {session_d}. Either the bot didn't "
        "scan, or every signal passed the filter chain."
    )
    st.stop()

# -- Headline metrics ---------------------------------------------------------
total_killed = len(killed)
unique_symbols = killed["symbol"].nunique()
unique_reasons = killed["kill_reasons"].nunique()
c1, c2, c3 = st.columns(3)
c1.metric("Total kills", total_killed)
c2.metric("Distinct symbols killed", unique_symbols)
c3.metric("Distinct kill reasons", unique_reasons)

st.divider()

# -- Kills by reason ----------------------------------------------------------
st.subheader("Kills grouped by reason")

# Normalise liquidity reasons (which embed turnover values) to a common
# bucket so the chart is meaningful — otherwise every symbol looks like
# its own reason.
def _normalise_reason(r: str) -> str:
    if r.startswith("liquidity"):
        return "liquidity"
    if r.startswith("circuit_proximity"):
        return "circuit_proximity"
    if r.startswith("market_closed"):
        return "market_closed"
    if r.startswith("nifty_crash"):
        return "nifty_crash"
    if r.startswith("fno_ban"):
        return "fno_ban"
    return r.strip()


killed["reason_bucket"] = killed["kill_reasons"].apply(_normalise_reason)
by_reason = (
    killed.groupby("reason_bucket")
    .agg(kills=("symbol", "count"),
         distinct_symbols=("symbol", "nunique"))
    .reset_index()
    .sort_values("kills", ascending=False)
)
by_reason["pct_of_kills"] = (by_reason["kills"] / total_killed * 100).round(1)

cc1, cc2 = st.columns([1, 1])
with cc1:
    st.dataframe(by_reason, height=420, width="stretch")
with cc2:
    fig = px.pie(
        by_reason, values="kills", names="reason_bucket",
        template="plotly_dark", title="Share of kills by reason",
        hole=0.45,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# -- Drill-in by reason -------------------------------------------------------
st.subheader("Drill in by reason")
chosen = st.selectbox(
    "Reason bucket", by_reason["reason_bucket"].tolist(),
)
sub = killed[killed["reason_bucket"] == chosen].copy()
sub_display = sub[["symbol", "ts", "side", "score", "final_score",
                   "kill_reasons"]].copy()
sub_display["symbol"] = sub_display["symbol"].str.replace(".NS", "", regex=False)
st.dataframe(sub_display, height=380, width="stretch")

st.caption(
    f"Hint: a reason killing >40% of all kills (e.g. `liquidity`) is "
    f"most likely fine — those symbols ARE illiquid. But a kill like "
    f"`already_extended` or `vix_high` over many alerts means the soft "
    f"thresholds in `filters/soft.py` may need a tune."
)
