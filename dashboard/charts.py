"""Plotly chart helpers — TradingView-feel candles + indicator overlays.

Single public entry: ``plotly_ohlc(df, overlays)`` returns a Figure with
candles, volume subplot, and any overlays you pass in. Dark theme, IST
x-axis, no rangeslider (uses too much vertical space)."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP per session (resets at IST midnight). DataFrame
    is assumed to have a tz-aware index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    by_session = df.index.date
    out = pd.Series(index=df.index, dtype=float)
    for session in pd.unique(by_session):
        mask = by_session == session
        cum_pv = pv[mask].cumsum()
        cum_vol = df["volume"][mask].cumsum()
        out.loc[mask] = (cum_pv / cum_vol).fillna(method="ffill")
    return out


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Lightweight supertrend for chart overlay only. The bot's scoring
    uses a different (more rigorous) implementation; this one is good
    enough for a visual line."""
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2.0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(index=df.index, dtype=float)
    direction = 1
    prev = lower.iloc[0]
    for i in range(len(df)):
        if close.iloc[i] > prev:
            direction = 1
            prev = max(lower.iloc[i], prev)
        elif close.iloc[i] < prev:
            direction = -1
            prev = min(upper.iloc[i], prev)
        st.iloc[i] = prev
    return st


def plotly_ohlc(
    df: pd.DataFrame,
    overlays: Iterable[str] = ("vwap", "ema9", "ema21"),
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """Build a candle + volume figure. ``df`` must have lowercase
    columns ``open/high/low/close/volume`` and a tz-aware DatetimeIndex.

    Supported overlays: ``vwap``, ``ema9``, ``ema21``, ``ema50``,
    ``supertrend``. Unknown names are silently skipped so callers can
    pass user input."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume"),
    )
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

    overlays = set(overlays or ())
    if "vwap" in overlays:
        try:
            vwap = _session_vwap(df)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=vwap, mode="lines",
                    line=dict(color="#ffd54f", width=1.5),
                    name="VWAP",
                ), row=1, col=1,
            )
        except Exception:
            pass
    if "ema9" in overlays:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=_ema(df["close"], 9), mode="lines",
                line=dict(color="#42a5f5", width=1.0), name="EMA 9",
            ), row=1, col=1,
        )
    if "ema21" in overlays:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=_ema(df["close"], 21), mode="lines",
                line=dict(color="#ab47bc", width=1.0), name="EMA 21",
            ), row=1, col=1,
        )
    if "ema50" in overlays:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=_ema(df["close"], 50), mode="lines",
                line=dict(color="#ff7043", width=1.0), name="EMA 50",
            ), row=1, col=1,
        )
    if "supertrend" in overlays:
        try:
            st = _supertrend(df)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=st, mode="lines",
                    line=dict(color="#80deea", width=1.0, dash="dot"),
                    name="Supertrend",
                ), row=1, col=1,
            )
        except Exception:
            pass

    # Volume bars colour-coded by candle direction
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

    fig.update_layout(
        template="plotly_dark",
        title=title or None,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=1.05,
            xanchor="left", x=0,
        ),
        # Hide weekends + overnight gaps so the chart is dense intraday
        xaxis2=dict(rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ]),
    )
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[15.5, 9.25], pattern="hour"),
    ], row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig
