"""Cached SQL helpers for the dashboard.

All public functions are wrapped in ``@st.cache_data(ttl=N)`` so a page
that calls the same query several times (e.g. once for a chart, once for
a leaderboard) pays the SQLite round-trip only once per refresh window.
TTL defaults to 60s — long enough to render a page without hammering
the file, short enough that the next manual refresh sees fresh data."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "alerts.db"


def _conn() -> sqlite3.Connection:
    """Open a read-only connection. Multiple Streamlit reruns sharing
    the file with the live bot is fine — SQLite WAL mode handles it."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def latest_market_context() -> dict:
    """Pull the most recent FII/DII + indices snapshot. Returns the dict
    a page can render directly, with None for fields not present."""
    out: dict = {
        "ts": None,
        "nifty_pct": None,
        "banknifty_pct": None,
        "vix": None,
        "fii_net_cr": None,
        "dii_net_cr": None,
    }
    with _conn() as c:
        # FII/DII
        row = c.execute(
            "SELECT ts, payload_json FROM nse_snapshots "
            "WHERE kind='fii_dii' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row:
            out["ts"] = row[0]
            try:
                payload = json.loads(row[1])
                for entry in payload:
                    cat = entry.get("category", "")
                    buy = float(entry.get("buyValue", 0) or 0)
                    sell = float(entry.get("sellValue", 0) or 0)
                    net = buy - sell
                    if cat.upper() == "FII" or cat.upper() == "FPI":
                        out["fii_net_cr"] = net
                    elif cat.upper() == "DII":
                        out["dii_net_cr"] = net
            except Exception:
                pass
        # All-indices snapshot — find Nifty 50, Bank Nifty, VIX in the list
        row = c.execute(
            "SELECT payload_json FROM nse_snapshots "
            "WHERE kind='all_indices' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                payload = json.loads(row[0])
                items = payload.get("data", payload) if isinstance(payload, dict) else payload
                for it in items if isinstance(items, list) else []:
                    name = (it.get("indexSymbol") or it.get("index") or "").upper()
                    pct = it.get("percentChange") or it.get("perChange")
                    if pct is None:
                        continue
                    try:
                        pct_v = float(pct)
                    except (TypeError, ValueError):
                        continue
                    if "NIFTY 50" in name and "NEXT" not in name:
                        out["nifty_pct"] = pct_v
                    elif "BANK" in name and "NIFTY" in name:
                        out["banknifty_pct"] = pct_v
                    elif "VIX" in name:
                        last = it.get("last") or it.get("lastPrice")
                        if last is not None:
                            try:
                                out["vix"] = float(last)
                            except (TypeError, ValueError):
                                pass
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Filter audit (latest score per symbol — drives the Live page table)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def latest_audit_per_symbol(session_date: str | None = None) -> pd.DataFrame:
    """Most recent filter_audit row per symbol for the given session
    (defaults to today). One row per symbol so the Live page can render
    a 500-row sortable table without a self-join scroll."""
    if session_date is None:
        session_date = datetime.now().date().isoformat()
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT fa.symbol, fa.ts, fa.side, fa.score, fa.final_score,
                   fa.final_confidence, fa.kill_reasons,
                   fa.soft_adjustments_json, fa.components_json,
                   fa.alerted
            FROM filter_audit fa
            JOIN (
                SELECT symbol, MAX(ts) AS max_ts
                FROM filter_audit
                WHERE date(ts) = ?
                GROUP BY symbol
            ) latest ON fa.symbol = latest.symbol AND fa.ts = latest.max_ts
            ORDER BY fa.final_score DESC
            """,
            c, params=(session_date,),
        )
    if df.empty:
        return df
    # Parse components JSON into separate columns
    def _parse_components(s):
        if not s:
            return {}
        try:
            return json.loads(s).get("components", {})
        except Exception:
            return {}
    components = df["components_json"].apply(_parse_components)
    for cat in ("trend", "momentum", "volume", "volatility",
                "structure", "market", "news"):
        df[cat] = components.apply(lambda c: c.get(cat))
    return df


@st.cache_data(ttl=60)
def audit_history_for_symbol(symbol: str, session_date: str | None = None) -> pd.DataFrame:
    """Every filter_audit row for one symbol on the session — chronological."""
    if session_date is None:
        session_date = datetime.now().date().isoformat()
    with _conn() as c:
        return pd.read_sql_query(
            """
            SELECT ts, side, score, final_score, final_confidence,
                   kill_reasons, soft_adjustments_json, alerted
            FROM filter_audit
            WHERE symbol = ? AND date(ts) = ?
            ORDER BY ts ASC
            """,
            c, params=(symbol, session_date),
        )


@st.cache_data(ttl=60)
def killed_signals_today(session_date: str | None = None) -> pd.DataFrame:
    """Filter_audit rows where the chain hard-killed the signal —
    grouped output sits on Filter Audit page."""
    if session_date is None:
        session_date = datetime.now().date().isoformat()
    with _conn() as c:
        return pd.read_sql_query(
            """
            SELECT symbol, ts, side, kill_reasons, score, final_score
            FROM filter_audit
            WHERE date(ts) = ?
              AND kill_reasons IS NOT NULL AND kill_reasons != ''
            ORDER BY ts DESC
            """,
            c, params=(session_date,),
        )


# ---------------------------------------------------------------------------
# Paper trades
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def paper_trades(
    start_date: str | None = None,
    end_date: str | None = None,
    status: str | None = None,
    side: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Filtered paper_trades. Adds ``hold_minutes`` derived column."""
    where = ["1=1"]
    params: list = []
    if start_date:
        where.append("date(entry_ts) >= ?")
        params.append(start_date)
    if end_date:
        where.append("date(entry_ts) <= ?")
        params.append(end_date)
    if status and status != "All":
        where.append("status = ?")
        params.append(status)
    if side and side != "All":
        where.append("side = ?")
        params.append(side)
    if symbol and symbol != "All":
        where.append("symbol = ?")
        params.append(symbol)
    sql = f"""
        SELECT * FROM paper_trades
        WHERE {' AND '.join(where)}
        ORDER BY entry_ts DESC
    """
    with _conn() as c:
        df = pd.read_sql_query(sql, c, params=params)
    if df.empty:
        return df
    # Hold duration
    df["entry_dt"] = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True)
    df["exit_dt"] = pd.to_datetime(df["exit_ts"], errors="coerce", utc=True)
    df["hold_minutes"] = (
        (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 60.0
    ).round(1)
    return df


@st.cache_data(ttl=30)
def open_paper_trade_count() -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status = 'OPEN'"
        ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Bars (for charts)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def recent_5m_bars(symbol_yf: str, n: int = 100) -> pd.DataFrame:
    """Recent 5m OHLCV bars for charting. ``symbol_yf`` is the
    yfinance form (e.g. 'BHARTIARTL.NS'); we convert to Fyers form
    for the bars_5m lookup."""
    # bars_5m stores Fyers symbol form
    bare = symbol_yf.replace(".NS", "").replace(".BO", "")
    fy_symbol = f"NSE:{bare}-EQ"
    with _conn() as c:
        df = pd.read_sql_query(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            c, params=(fy_symbol, n),
        )
    if df.empty:
        return df
    df = df.sort_values("ts").reset_index(drop=True)
    # ts is epoch ms
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    df = df.set_index("dt").drop(columns=["ts"])
    return df


# ---------------------------------------------------------------------------
# Alerts sent
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def alerts_sent(start_date: str | None = None) -> pd.DataFrame:
    if start_date is None:
        start_date = (date.today() - timedelta(days=7)).isoformat()
    with _conn() as c:
        return pd.read_sql_query(
            """
            SELECT id, symbol, score, reasons, price, sent_at
            FROM alerts_sent
            WHERE date(sent_at) >= ?
            ORDER BY sent_at DESC
            """,
            c, params=(start_date,),
        )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def news_for_symbol(symbol: str, limit: int = 10) -> pd.DataFrame:
    """News items scored for a symbol, most recent first."""
    with _conn() as c:
        return pd.read_sql_query(
            """
            SELECT ni.title, ni.url, ni.source, ni.published_at,
                   ni.finbert_score, ni.finbert_label, ns.relevance
            FROM news_scores ns
            JOIN news_items ni ON ns.news_id = ni.id
            WHERE ns.symbol = ?
            ORDER BY ni.published_at DESC
            LIMIT ?
            """,
            c, params=(symbol, limit),
        )


# ---------------------------------------------------------------------------
# Daily levels
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def daily_levels(symbol: str, session_date: str | None = None) -> dict:
    if session_date is None:
        session_date = datetime.now().date().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT levels_json FROM daily_levels "
            "WHERE symbol = ? AND session_date = ?",
            (symbol, session_date),
        ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Distinct lookups (for filter dropdowns)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def distinct_symbols_in_paper_trades() -> list[str]:
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT symbol FROM paper_trades ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]
