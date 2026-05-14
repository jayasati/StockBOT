"""Indicator visualizer — pull a cached symbol, render every implemented
indicator on the chart, and show the latest-bar values next to the
chart so you can verify against TradingView's Data Window directly.

Reads symbol bars via ``strategies.data.load`` (the 60d 5m parquet cache
populated via ``python -m strategies fetch``). Indicators come from
``indicators.REGISTRY`` so any addition there shows up here without
touching this file."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from indicators import REGISTRY
from strategies import data as strat_data

st.set_page_config(
    page_title="Indicators · Stock Alert Bot",
    page_icon="📐",
    layout="wide",
)
st.title("📐 Indicator Chart — verify against TradingView")
st.caption(
    "Loads cached 5m bars, renders selected indicators on the chart, and "
    "shows the latest-bar values for cross-check against TV's Data Window. "
    "First populate the cache with `python -m strategies fetch --symbols "
    "RELIANCE.NS,TCS.NS,...`."
)

# ---------------------------------------------------------------------------
# Indicator categorization for layout
# ---------------------------------------------------------------------------

# Drawn ON the candle pane (price overlays).
_PRICE_OVERLAYS = {
    "sma", "ema", "wma", "hull_ma", "vwap", "vwma", "anchored_vwap",
    "auto_anchored_vwap", "bollinger", "keltner", "donchian", "supertrend",
    "parabolic_sar", "ichimoku", "vwap_sd_bands",
}
# Drawn on the volume pane (overlay on volume bars).
_VOLUME_OVERLAYS = {"volume_ma"}
# Discrete pivot markers on the candle pane.
_PIVOT_MARKERS = {"zigzag"}
# Horizontal-level scalars (POC / pivots / etc.) — drawn as hlines.
_HORIZONTAL_LEVELS = {
    "visible_average_price", "previous_day_hlc", "opening_range",
    "initial_balance", "pivot_points",
}
# Everything else lands in its OWN subplot below the volume pane.
# (RSI, MACD, ADX, Stoch, MFI, CCI, ROC, TSI, StochRSI, Connors RSI,
#  Choppiness, Awesome Osc, TRIX, OBV, CMF, AD Line, Force Index,
#  Williams %R, ATR, TTM Squeeze, RVOL TOD, Volume Surge, Aroon)
def _is_oscillator(name: str) -> bool:
    return (
        name not in _PRICE_OVERLAYS
        and name not in _VOLUME_OVERLAYS
        and name not in _PIVOT_MARKERS
        and name not in _HORIZONTAL_LEVELS
    )


# ---------------------------------------------------------------------------
# Sidebar: symbol + window + indicator selection
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _cached_symbols() -> list[str]:
    summary = strat_data.cache_summary("5m", 60)
    if summary.empty:
        return []
    return sorted(summary["symbol"].tolist())


with st.sidebar:
    st.header("Data")
    symbols = _cached_symbols()
    if not symbols:
        st.error(
            "No cached data. Run e.g.:\n\n"
            "`python -m strategies fetch --symbols "
            "RELIANCE.NS,TCS.NS,INFY.NS --days 60`"
        )
        st.stop()
    symbol = st.selectbox("Symbol", symbols)
    interval = st.selectbox("Interval", ["5m"], index=0,
                            help="Only 5m is currently cached.")
    days = st.number_input("Lookback (days)", 5, 60, value=60, step=5)
    bars_to_show = st.number_input(
        "Bars to display (right-most)", 50, 5000, value=400, step=50,
        help="Truncate to the last N bars so the chart is readable.",
    )

    st.header("Indicators")
    # Group selection helpers so the user doesn't have to drag-pick 30 names.
    available = sorted(REGISTRY.keys())
    selected_default = ["bollinger", "vwap", "supertrend",
                        "rsi", "macd", "adx", "stoch_rsi"]
    selected = st.multiselect(
        "Pick indicators to render",
        available,
        default=[s for s in selected_default if s in available],
    )

    st.caption(
        f"{len(REGISTRY)} indicators registered. Price overlays render on "
        "the candle pane; oscillators get their own subplot below."
    )


# ---------------------------------------------------------------------------
# Load bars
# ---------------------------------------------------------------------------

try:
    df_full = strat_data.load(symbol, interval=interval, days=int(days))
except KeyError as e:
    st.error(str(e))
    st.stop()

df = df_full.tail(int(bars_to_show))
if df.empty:
    st.warning("No bars in window.")
    st.stop()


# ---------------------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------------------

# A handful of registered indicators take a `session_date` positional arg
# that REGISTRY.default_params doesn't include (it's intentionally caller-
# injected so the same impl can be reused across sessions). Pre-fill it
# here with the LAST session present in the bars.
# A handful of registered indicators take a session-anchored argument
# the registry's default_params doesn't include (intentionally, so the
# same impl can be reused across sessions). Inject what each one needs:
_SESSION_DATE_INDICATORS = {
    "previous_day_hlc", "opening_range", "initial_balance",
    "vwap_sd_bands",
}


@st.cache_data(ttl=120, show_spinner=False)
def _compute_indicator(symbol: str, interval: str, days: int,
                       n_bars: int, name: str) -> object:
    """Cache the indicator output keyed by data identity. Symbol/interval/
    days/n_bars uniquely identify the bars; ``name`` picks the indicator."""
    bars = strat_data.load(symbol, interval=interval, days=days).tail(n_bars)
    spec = REGISTRY[name]
    kwargs = dict(spec.default_params)
    last_ts = bars.index[-1]
    if name in _SESSION_DATE_INDICATORS and hasattr(last_ts, "date"):
        kwargs.setdefault("session_date", last_ts.date())
    if name == "pivot_points":
        # pivot_points is a derived indicator: takes a prev-day HLC dict,
        # not a DataFrame. Build that dict from the cached bars first.
        from indicators import levels
        prev = levels.previous_day_hlc(bars, last_ts.date())
        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in prev.values()):
            raise ValueError(
                "pivot_points needs a prior session in the loaded window"
            )
        return spec.func(prev, **kwargs)
    return spec.func(bars, **kwargs)


computed: dict[str, object] = {}
errors: dict[str, str] = {}
for name in selected:
    try:
        computed[name] = _compute_indicator(
            symbol, interval, int(days), int(bars_to_show), name,
        )
    except Exception as e:
        errors[name] = repr(e)

if errors:
    with st.expander(f"⚠ {len(errors)} indicator(s) failed to compute"):
        for name, msg in errors.items():
            st.code(f"{name}: {msg}")


# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------

oscillator_indicators = [n for n in computed if _is_oscillator(n)]
n_subplots = 2 + len(oscillator_indicators)  # candles, volume, then each osc.
row_heights = [0.55, 0.15] + [0.30 / max(1, len(oscillator_indicators))] \
    * len(oscillator_indicators)

subplot_titles = [f"{symbol}  ({interval}, {len(df)} bars)", "Volume"]
subplot_titles.extend(oscillator_indicators)

fig = make_subplots(
    rows=n_subplots, cols=1, shared_xaxes=True,
    vertical_spacing=0.025,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
)

# --- Candles (row 1) -------------------------------------------------------
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name="OHLC", showlegend=False,
    ),
    row=1, col=1,
)

# --- Price overlays --------------------------------------------------------
_OVERLAY_COLORS = {
    "sma":               "#42a5f5",
    "ema":               "#42a5f5",
    "wma":               "#7e57c2",
    "hull_ma":           "#7e57c2",
    "vwap":              "#ffd54f",
    "vwma":              "#ffb74d",
    "anchored_vwap":     "#fff176",
    "supertrend":        "#80deea",
    "parabolic_sar":     "#ce93d8",
}


def _add_series(values: pd.Series, name: str, color: str,
                row: int = 1, dash: str | None = None) -> None:
    fig.add_trace(
        go.Scatter(
            x=values.index, y=values, mode="lines",
            line=dict(color=color, width=1.3,
                      **({"dash": dash} if dash else {})),
            name=name,
        ),
        row=row, col=1,
    )


for name in selected:
    if name not in computed or name not in _PRICE_OVERLAYS:
        continue
    out = computed[name]
    color = _OVERLAY_COLORS.get(name, "#90a4ae")
    if isinstance(out, pd.Series):
        _add_series(out, name, color)
    elif isinstance(out, pd.DataFrame):
        # Multi-line overlays: Bollinger upper/lower, Auto-AVWAP, Donchian, etc.
        sub_palette = ["#90caf9", "#ef9a9a", "#a5d6a7", "#ffcc80",
                       "#ce93d8", "#80cbc4"]
        for i, col in enumerate(out.columns):
            # Skip non-line columns like ttm_squeeze.in_squeeze (binary).
            ser = out[col]
            if not np.issubdtype(ser.dtype, np.number):
                continue
            if name == "supertrend" and col == "direction":
                continue  # direction is integer, plotted as supertrend line color
            _add_series(
                ser, f"{name}.{col}",
                sub_palette[i % len(sub_palette)],
                dash="dot" if "lower" in col or "minus" in col else None,
            )

# --- ZigZag pivot markers --------------------------------------------------
if "zigzag" in computed:
    zz = computed["zigzag"]
    high_pivots = zz[zz["pivot_type"] == 1]
    low_pivots = zz[zz["pivot_type"] == -1]
    fig.add_trace(
        go.Scatter(
            x=high_pivots.index, y=high_pivots["zigzag_price"],
            mode="markers+text", marker=dict(symbol="triangle-down",
                                              size=10, color="#ef5350"),
            text=["H"] * len(high_pivots), textposition="top center",
            name="ZigZag H",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=low_pivots.index, y=low_pivots["zigzag_price"],
            mode="markers+text", marker=dict(symbol="triangle-up",
                                              size=10, color="#26a69a"),
            text=["L"] * len(low_pivots), textposition="bottom center",
            name="ZigZag L",
        ), row=1, col=1,
    )
    # Optional: connect pivots with a line so it looks like TV's ZigZag.
    pivots_only = zz.dropna(subset=["zigzag_price"])
    if not pivots_only.empty:
        fig.add_trace(
            go.Scatter(
                x=pivots_only.index, y=pivots_only["zigzag_price"],
                mode="lines",
                line=dict(color="#bdbdbd", width=1, dash="dot"),
                name="ZigZag", showlegend=False,
            ), row=1, col=1,
        )

# --- Horizontal-level overlays (VAP / pivots / OR) ------------------------
def _add_hline(y: float, label: str, color: str) -> None:
    fig.add_hline(y=y, line_dash="dot", line_color=color,
                  annotation_text=label, annotation_position="right",
                  row=1, col=1)


for name in selected:
    if name not in computed or name not in _HORIZONTAL_LEVELS:
        continue
    out = computed[name]
    if isinstance(out, dict):
        palette = ["#ffd54f", "#80cbc4", "#ce93d8", "#ef9a9a",
                   "#a5d6a7", "#fff176", "#b39ddb"]
        for i, (k, v) in enumerate(out.items()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            _add_hline(float(v), f"{name}.{k}", palette[i % len(palette)])

# --- Volume bars (row 2) ---------------------------------------------------
vol_colors = [
    "#26a69a" if c >= o else "#ef5350"
    for o, c in zip(df["open"], df["close"])
]
fig.add_trace(
    go.Bar(
        x=df.index, y=df["volume"], marker_color=vol_colors,
        name="Volume", showlegend=False,
    ),
    row=2, col=1,
)

# --- Volume overlays (Volume MA on row 2) ---------------------------------
for name in selected:
    if name not in computed or name not in _VOLUME_OVERLAYS:
        continue
    out = computed[name]
    if isinstance(out, pd.Series):
        _add_series(out, name, "#ffeb3b", row=2)

# --- Oscillator subplots (rows 3..N) --------------------------------------
osc_palette = ["#42a5f5", "#ef5350", "#26a69a", "#ffb74d", "#ab47bc",
               "#80deea", "#ff8a65"]
for row_idx, name in enumerate(oscillator_indicators, start=3):
    out = computed[name]
    if isinstance(out, pd.Series):
        fig.add_trace(
            go.Scatter(
                x=out.index, y=out, mode="lines",
                line=dict(color="#42a5f5", width=1.3),
                name=name, showlegend=False,
            ), row=row_idx, col=1,
        )
        # Add reference lines for known-bounded oscillators.
        if name in {"rsi", "mfi", "stochastic", "stoch_rsi", "connors_rsi"}:
            for level, color in [(70, "#ef5350"), (30, "#26a69a")]:
                fig.add_hline(y=level, line_dash="dot", line_color=color,
                              opacity=0.4, row=row_idx, col=1)
        elif name == "tsi":
            fig.add_hline(y=0, line_dash="dot", line_color="#bdbdbd",
                          opacity=0.4, row=row_idx, col=1)
    elif isinstance(out, pd.DataFrame):
        for i, col in enumerate(out.columns):
            ser = out[col]
            if not np.issubdtype(ser.dtype, np.number):
                continue
            fig.add_trace(
                go.Scatter(
                    x=ser.index, y=ser, mode="lines",
                    line=dict(color=osc_palette[i % len(osc_palette)],
                              width=1.2),
                    name=f"{name}.{col}", showlegend=False,
                ), row=row_idx, col=1,
            )
        if name == "macd":
            fig.add_hline(y=0, line_dash="dot", line_color="#bdbdbd",
                          opacity=0.4, row=row_idx, col=1)
        elif name == "stoch_rsi":
            for level, color in [(80, "#ef5350"), (20, "#26a69a")]:
                fig.add_hline(y=level, line_dash="dot", line_color=color,
                              opacity=0.4, row=row_idx, col=1)
        elif name == "adx":
            fig.add_hline(y=25, line_dash="dot", line_color="#ffd54f",
                          opacity=0.4, row=row_idx, col=1)


# --- Layout ---------------------------------------------------------------
fig_height = 600 + 150 * len(oscillator_indicators)
fig.update_layout(
    template="plotly_dark",
    height=fig_height,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=1.02,
                xanchor="left", x=0),
    hovermode="x unified",
)
fig.update_xaxes(rangebreaks=[
    dict(bounds=["sat", "mon"]),
    dict(bounds=[15.5, 9.25], pattern="hour"),
])

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Latest-bar values table — for direct TV Data Window comparison
# ---------------------------------------------------------------------------

def _last_value_rows(computed: dict[str, object]) -> list[dict]:
    rows: list[dict] = []
    for name, out in computed.items():
        if isinstance(out, pd.Series):
            v = out.iloc[-1] if len(out) else np.nan
            rows.append({
                "indicator": name, "key": name,
                "latest_value": float(v) if not pd.isna(v) else None,
                "params": str(REGISTRY[name].default_params),
            })
        elif isinstance(out, pd.DataFrame):
            if out.empty:
                continue
            for col in out.columns:
                ser = out[col]
                if not np.issubdtype(ser.dtype, np.number):
                    continue
                v = ser.iloc[-1]
                rows.append({
                    "indicator": name, "key": col,
                    "latest_value": float(v) if not pd.isna(v) else None,
                    "params": str(REGISTRY[name].default_params),
                })
        elif isinstance(out, dict):
            for k, v in out.items():
                rows.append({
                    "indicator": name, "key": k,
                    "latest_value": float(v) if v is not None
                        and not (isinstance(v, float) and np.isnan(v))
                        else None,
                    "params": str(REGISTRY[name].default_params),
                })
    return rows


st.subheader("Latest-bar values  (compare to TradingView Data Window)")
st.caption(
    f"Latest bar: **{df.index[-1]}** · close = **{df['close'].iloc[-1]:.2f}** · "
    f"volume = {int(df['volume'].iloc[-1]):,}"
)
values_df = pd.DataFrame(_last_value_rows(computed))
if not values_df.empty:
    values_df["latest_value"] = values_df["latest_value"].astype(float)
    st.dataframe(
        values_df.style.format({"latest_value": "{:.4f}"}, na_rep="—"),
        height=400, width="stretch",
    )

st.caption(
    f"Page rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST  ·  "
    f"data: cached parquet (5m, 60d window)"
)
