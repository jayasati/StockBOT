"""Stock Alert Bot — Streamlit dashboard entry point.

Run from the project root:
    streamlit run dashboard/app.py

Pages live in dashboard/pages/ and are auto-discovered by Streamlit.
Bind is localhost-only via .streamlit/config.toml. Do NOT expose to
the network without auth in front."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from dashboard import db

st.set_page_config(
    page_title="Stock Alert Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Stock Alert Bot — Dashboard")
st.caption(
    "Localhost-only. Reads alerts.db live alongside the running bot. "
    "Click pages in the sidebar to navigate."
)

# Quick top-of-page status so users have a sanity check that the data
# layer is alive before they click into a page.
ctx = db.latest_market_context()
open_trades = db.open_paper_trade_count()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Nifty",
    f"{ctx['nifty_pct']:+.2f}%" if ctx["nifty_pct"] is not None else "—",
)
c2.metric(
    "Bank Nifty",
    f"{ctx['banknifty_pct']:+.2f}%" if ctx["banknifty_pct"] is not None else "—",
)
c3.metric(
    "VIX",
    f"{ctx['vix']:.2f}" if ctx["vix"] is not None else "—",
)
c4.metric(
    "FII Net (₹cr)",
    f"{ctx['fii_net_cr']:+,.0f}" if ctx["fii_net_cr"] is not None else "—",
)
c5.metric("Open paper trades", open_trades)

st.divider()

st.subheader("Pages")
st.markdown(
    """
- **Live** — Ranked watchlist by current confidence; click any row for the
  drill-down (chart + indicator panel + filter audit + news + TV verify).
- **Paper Trades** — Filter by date / status / side / symbol. Cumulative
  P&L, win-rate by hour, win-rate by indicator combination.
- **Filter Audit** — Today's killed signals grouped by reason. Spot
  over-aggressive filters that need tuning in `scoring.yaml`.
- **Performance** — Equity curve, drawdown, rolling Sharpe, profit
  factor, per-stock leaderboard, daily P&L heatmap.
- **Verify** — Side-by-side TV vs bot indicator comparison across many
  symbols at once. Pinpoints data-source vs math divergences.
- **VWAP + EMA + Volume** — Per-symbol backtest of the `vwap_ema_volume`
  strategy (VWAP reclaim/rejection + EMA 9>21 + volume spike, 1.5×ATR
  target). Per-stock scorecard, equity curve, and trades drilldown.
- **MACD + RSI + EMA** — Per-symbol backtest of the `macd_rsi_ema`
  strategy (MACD histogram flip + RSI 50-cross + EMA 20 trend filter,
  first-90-min entry window). Hour-of-day + day-of-week WR breakdowns.
- **ADX + Supertrend + ATR** — Per-symbol backtest of the
  `adx_supertrend_atr` strategy (Supertrend flip + ADX > 25 & rising +
  DMI bias, 1×ATR stop loss). Trades only when trend strength is real.
"""
)

st.divider()

with st.expander("Refresh & cache"):
    st.markdown(
        "SQL queries are cached for 30–60 seconds. Click below to clear "
        "the cache and force a fresh read on the next page render."
    )
    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Reload any page to refetch.")

st.caption(
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST  ·  "
    f"DB: alerts.db"
)
