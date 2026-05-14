"""Live page — ranked watchlist + drill-down per stock."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import db
from dashboard.charts import plotly_ohlc

st.set_page_config(page_title="Live · Stock Alert Bot",
                   page_icon="📊", layout="wide")
st.title("📊 Live Watchlist")

# -- Top status strip ---------------------------------------------------------
ctx = db.latest_market_context()
open_trades = db.open_paper_trade_count()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nifty",
          f"{ctx['nifty_pct']:+.2f}%" if ctx["nifty_pct"] is not None else "—")
c2.metric("Bank Nifty",
          f"{ctx['banknifty_pct']:+.2f}%" if ctx["banknifty_pct"] is not None else "—")
c3.metric("VIX",
          f"{ctx['vix']:.2f}" if ctx["vix"] is not None else "—")
c4.metric("FII Net (₹cr)",
          f"{ctx['fii_net_cr']:+,.0f}" if ctx["fii_net_cr"] is not None else "—")
c5.metric("Open paper trades", open_trades)

st.divider()

# -- Controls -----------------------------------------------------------------
controls = st.container()
with controls:
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    session_date = cc1.date_input(
        "Session date", value=datetime.now().date(),
    ).isoformat()
    refresh = cc2.button("🔄 Refresh now")
    if refresh:
        st.cache_data.clear()
    cc3.caption(
        "Rows are the most-recent filter_audit per symbol on the chosen "
        "session. Confidence is the post-multiplier final score (0–100)."
    )

# -- Ranked table -------------------------------------------------------------
df = db.latest_audit_per_symbol(session_date)
if df.empty:
    st.warning(
        f"No filter_audit rows for {session_date}. The bot likely hasn't "
        "scanned today yet, or the date is non-trading."
    )
    st.stop()

# Display columns
def _num(col: str) -> pd.Series:
    """Coerce a possibly-None object column to numeric so .round() works."""
    return pd.to_numeric(df[col], errors="coerce")


display = pd.DataFrame({
    "symbol": df["symbol"].str.replace(".NS", "", regex=False),
    "side": df["side"],
    "raw_score": _num("score"),
    "confidence": _num("final_score").round(1),
    "trend": _num("trend").round(0),
    "momentum": _num("momentum").round(0),
    "volume": _num("volume").round(0),
    "structure": _num("structure").round(0),
    "kill": df["kill_reasons"].fillna(""),
    "alerted": df["alerted"].astype(bool),
})


def _bg_color(v):
    if pd.isna(v):
        return ""
    if v >= 75:
        return "background-color: #1b5e20; color: #e8f5e9"  # dark green
    if v >= 60:
        return "background-color: #2e7d32; color: #e8f5e9"  # green
    if v >= 50:
        return "background-color: #f57f17; color: #fffde7"  # amber
    return "background-color: #b71c1c; color: #ffebee"      # red


styled = (
    display.style
    .map(_bg_color, subset=["confidence"])
    .format({
        "confidence": "{:.1f}",
        "trend": "{:.0f}", "momentum": "{:.0f}",
        "volume": "{:.0f}", "structure": "{:.0f}",
    }, na_rep="—")
)
st.dataframe(styled, height=520, width="stretch")

# -- Drill-down ---------------------------------------------------------------
st.subheader("Drill down")
symbol_options = display["symbol"].tolist()
selected_bare = st.selectbox(
    "Pick a symbol", symbol_options, index=0,
    help="Lists every symbol audited on the chosen session, "
         "highest confidence first."
)
if not selected_bare:
    st.stop()
selected_yf = f"{selected_bare}.NS"

# Detail row from the table
row = df[df["symbol"] == selected_yf].iloc[0]

# Tabs
tab_chart, tab_indicators, tab_audit, tab_news, tab_verify = st.tabs(
    ["📈 Chart", "🔢 Indicators", "🛡️ Filter audit", "📰 News", "✅ TV verify"],
)

with tab_chart:
    bars = db.recent_5m_bars(selected_yf, n=200)
    if bars.empty:
        st.info(
            f"No bars_5m data for {selected_yf}. The bot's bars_5m table "
            "is populated by the live Fyers feed; if the bot hasn't been "
            "running, it may be empty for this symbol."
        )
    else:
        overlays_picked = st.multiselect(
            "Overlays",
            ["vwap", "ema9", "ema21", "ema50", "supertrend"],
            default=["vwap", "ema9", "ema21"],
            key="live_overlays",
        )
        fig = plotly_ohlc(
            bars, overlays=overlays_picked,
            title=f"{selected_bare}  ·  5m  ·  last {len(bars)} bars",
            height=620,
        )
        st.plotly_chart(fig, width="stretch")

with tab_indicators:
    st.markdown(
        f"**Latest filter_audit @ "
        f"{pd.to_datetime(row['ts']).strftime('%Y-%m-%d %H:%M:%S')}**"
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Score**")
        st.markdown(f"- Side: `{row['side']}`")
        st.markdown(f"- Raw score: `{row['score']}`")
        st.markdown(
            f"- Final score: `{row['final_score']:.2f}`"
            if pd.notna(row['final_score']) else "- Final score: —"
        )
        st.markdown(
            f"- Confidence: `{row['final_confidence']:.3f}`"
            if pd.notna(row['final_confidence']) else "- Confidence: —"
        )
        st.markdown(f"- Alerted: `{bool(row['alerted'])}`")
        st.markdown(f"- Kill reasons: `{row['kill_reasons'] or '(none)'}`")
        try:
            adj = json.loads(row["soft_adjustments_json"] or "[]")
        except Exception:
            adj = []
        if adj:
            st.markdown("**Soft adjustments:**")
            for name, mult in adj:
                st.markdown(f"  - `{name}` × `{mult}`")
    with cc2:
        st.markdown("**Components (0–100)**")
        for cat in ("trend", "momentum", "volume", "volatility",
                    "structure", "market", "news"):
            v = row.get(cat)
            v_str = f"`{v:.1f}`" if pd.notna(v) else "—"
            st.markdown(f"- {cat}: {v_str}")

    levels = db.daily_levels(selected_yf, session_date)
    if levels:
        st.markdown("**Daily levels (precomputed at 09:00 IST)**")
        st.json(levels, expanded=False)

with tab_audit:
    hist = db.audit_history_for_symbol(selected_yf, session_date)
    if hist.empty:
        st.info("No audit history for this symbol on this session.")
    else:
        st.dataframe(hist, height=380, width="stretch")

with tab_news:
    news = db.news_for_symbol(selected_yf, limit=10)
    if news.empty:
        st.info("No scored news for this symbol.")
    else:
        for _, n in news.iterrows():
            label = n.get("finbert_label") or "?"
            score = n.get("finbert_score")
            score_s = f"{score:+.2f}" if pd.notna(score) else "—"
            colour = (
                "🟢" if label == "positive"
                else "🔴" if label == "negative"
                else "⚪"
            )
            url = n.get("url") or "#"
            st.markdown(
                f"{colour} **[{n['title']}]({url})**  ·  "
                f"`{label}`  `{score_s}`  ·  {n.get('source')}  ·  "
                f"{n.get('published_at')}"
            )

with tab_verify:
    st.caption(
        "Live TradingView vs bot indicator comparison. Calls the TV API "
        "(via `tradingview-ta`) and the bot's `compute_all` snapshot. "
        "Slow first call (~5s), faster on re-runs (cached 60s)."
    )
    tfs = st.multiselect(
        "Timeframes", ["5m", "15m", "60m", "1d"],
        default=["5m", "15m"], key=f"verify_tfs_{selected_bare}",
    )
    run_verify = st.button(
        "Run verify", key=f"run_verify_{selected_bare}",
    )
    if run_verify and tfs:
        from dashboard.verify_helper import compare_one_symbol
        with st.spinner(f"Comparing {selected_bare} on {', '.join(tfs)}…"):
            try:
                result = compare_one_symbol(selected_bare, tfs)
            except Exception as e:
                st.error(f"Verify failed: {type(e).__name__}: {e}")
            else:
                st.success(
                    f"{result.diverge_count} indicators diverge beyond "
                    f"tolerance."
                )
                # Highlight diverging rows
                def _row_style(r):
                    if r["status"] == "DIVERGE":
                        return ["background-color: #5d4037"] * len(r)
                    if r["status"] == "OK":
                        return ["background-color: #1b5e20"] * len(r)
                    return [""] * len(r)

                styled = result.rows.style.apply(_row_style, axis=1).format({
                    "tv": "{:.4f}", "bot": "{:.4f}",
                    "delta": "{:+.4f}", "pct": "{:+.2f}%",
                }, na_rep="—")
                st.dataframe(styled, height=540, width="stretch")

st.caption(
    f"Page rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST"
)
