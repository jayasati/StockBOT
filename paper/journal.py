"""Paper-trading journal — read-side queries + CLI report.

Public surface:
  Trade                   — frozen dataclass mirror of one
                            ``paper_trades`` row
  list_open()             — every trade with status='OPEN'
  list_closed(since=...)  — closed trades (status != 'OPEN'),
                            optionally filtered by exit_ts >= date
  daily_summary(date)     — aggregate stats for trades CLOSED on date
  win_rate_by_indicator() — DataFrame pivot of signal_indicators
                            joined against closed trades

CLI (built up in Step 5):
  python -m paper.journal report --since YYYY-MM-DD

Date comparisons use ``substr(exit_ts, 1, 10)`` rather than the
SQLite ``date()`` function so the +05:30 IST offset baked into every
ISO timestamp doesn't trip the parser."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date as _Date

import pandas as pd

from .schema import connect, ensure_paper_schema

_TRADE_COLS = (
    "id, symbol, side, entry_ts, entry_price, qty, stop_loss, "
    "target_1, target_2, confidence, status, exit_ts, exit_price, "
    "pnl_gross, pnl_net, notes"
)


@dataclass(frozen=True)
class Trade:
    """One row of ``paper_trades`` as a Python value type. Field order
    mirrors :data:`_TRADE_COLS` so ``Trade.from_row`` can splat the
    tuple in."""
    id: int
    symbol: str
    side: str
    entry_ts: str
    entry_price: float
    qty: int
    stop_loss: float
    target_1: float
    target_2: float | None
    confidence: float
    status: str
    exit_ts: str | None
    exit_price: float | None
    pnl_gross: float | None
    pnl_net: float | None
    notes: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> "Trade":
        return cls(*row)


def _date_str(d: _Date | str) -> str:
    """Normalize a date-like input to ``YYYY-MM-DD``."""
    return d.isoformat() if isinstance(d, _Date) else str(d)[:10]


def list_open() -> list[Trade]:
    """Every trade currently in OPEN status, oldest entry first."""
    ensure_paper_schema()
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {_TRADE_COLS} FROM paper_trades "
            "WHERE status = 'OPEN' "
            "ORDER BY entry_ts ASC"
        ).fetchall()
    return [Trade.from_row(r) for r in rows]


def list_closed(since: _Date | str | None = None) -> list[Trade]:
    """Closed trades (status != 'OPEN'), oldest exit first.

    ``since`` filters by ``exit_ts >= date``; accepts a ``datetime.date``
    or an ISO date string. ``None`` returns every closed trade."""
    ensure_paper_schema()
    sql = f"SELECT {_TRADE_COLS} FROM paper_trades WHERE status != 'OPEN'"
    params: tuple = ()
    if since is not None:
        sql += " AND substr(exit_ts, 1, 10) >= ?"
        params = (_date_str(since),)
    sql += " ORDER BY exit_ts ASC"
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Trade.from_row(r) for r in rows]


def daily_summary(d: _Date | str) -> dict:
    """Aggregate stats for trades CLOSED on ``d`` (by ``exit_ts``
    date in IST).

    Returns a dict with:
      n_trades        — closed count for the day
      win_rate        — fraction with pnl_net > 0  (0.0 if n_trades=0)
      gross_pnl       — sum(pnl_gross)
      net_pnl         — sum(pnl_net)
      avg_winner      — mean pnl_net of winners (0.0 if none)
      avg_loser       — mean pnl_net of losers  (0.0 if none)
      profit_factor   — sum(winners) / |sum(losers)|; inf if no losers
                        (and at least one winner), 0.0 if no winners

    Trades with NULL pnl_net (MANUAL close on a delisted symbol) are
    excluded — they don't have enough data to count for-or-against."""
    ensure_paper_schema()
    d_str = _date_str(d)
    with connect() as conn:
        rows = conn.execute(
            "SELECT pnl_gross, pnl_net FROM paper_trades "
            "WHERE status != 'OPEN' "
            "  AND substr(exit_ts, 1, 10) = ? "
            "  AND pnl_net IS NOT NULL",
            (d_str,),
        ).fetchall()
    n = len(rows)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0,
            "gross_pnl": 0.0, "net_pnl": 0.0,
            "avg_winner": 0.0, "avg_loser": 0.0, "profit_factor": 0.0,
        }
    gross_total = sum(g for g, _ in rows)
    net_total = sum(n for _, n in rows)
    winners = [n for _, n in rows if n > 0]
    losers = [n for _, n in rows if n < 0]
    wins_sum = sum(winners)
    losses_sum = sum(losers)
    if not losers and winners:
        profit_factor = float("inf")
    elif not losers:
        profit_factor = 0.0
    else:
        profit_factor = wins_sum / abs(losses_sum)
    return {
        "n_trades": n,
        "win_rate": len(winners) / n,
        "gross_pnl": gross_total,
        "net_pnl": net_total,
        "avg_winner": (wins_sum / len(winners)) if winners else 0.0,
        "avg_loser": (losses_sum / len(losers)) if losers else 0.0,
        "profit_factor": profit_factor,
    }


def win_rate_by_indicator() -> pd.DataFrame:
    """Pivot of signal_indicators against closed-trade outcomes.

    Columns: ``indicator``, ``timeframe``, ``n_trades``, ``win_rate``,
    ``avg_pnl_net``. Sorted by win_rate DESC then n_trades DESC so the
    "looks great until you check sample size" rows surface last.

    Excludes trades with NULL pnl_net (MANUAL closes without LTP).
    Returns an empty DataFrame when no closed trades exist yet."""
    ensure_paper_schema()
    sql = (
        "SELECT "
        "  si.indicator AS indicator, "
        "  si.timeframe AS timeframe, "
        "  COUNT(*) AS n_trades, "
        "  AVG(CASE WHEN pt.pnl_net > 0 THEN 1.0 ELSE 0.0 END) AS win_rate, "
        "  AVG(pt.pnl_net) AS avg_pnl_net "
        "FROM signal_indicators si "
        "JOIN paper_trades pt ON pt.id = si.paper_trade_id "
        "WHERE pt.status != 'OPEN' AND pt.pnl_net IS NOT NULL "
        "GROUP BY si.indicator, si.timeframe "
        "ORDER BY win_rate DESC, n_trades DESC"
    )
    with connect() as conn:
        return pd.read_sql_query(sql, conn)


# ---------------------------------------------------------------------------
# End-of-session digest (Phase-5b noise-reduction)
# ---------------------------------------------------------------------------

def build_eod_digest(d: _Date | str | None = None) -> str | None:
    """HTML-formatted Telegram message summarising the day's paper
    trades. Returns ``None`` when nothing closed today — caller skips
    the send so quiet days don't add noise.

    Layout:
      📊 EOD Digest YYYY-MM-DD
      Closed: N  |  Win rate: X%
      Net P&L: ₹+1,234.56  |  PF: 2.10

      🏆 Top winners:
        • SYM1 TP1  ₹+550.00
        • SYM2 TP1  ₹+320.00
        • SYM3 TP2  ₹+180.00

      ⚠️ N open (will TIMEOUT next session)   — only if any still OPEN"""
    from .tracker import IST
    if d is None:
        d = pd.Timestamp.now(tz=IST).date()
    d_str = _date_str(d)

    summary = daily_summary(d)
    if summary["n_trades"] == 0:
        return None

    today_closed = [
        t for t in list_closed(since=d_str)
        if t.exit_ts and t.exit_ts.startswith(d_str)
    ]
    winners = sorted(
        [t for t in today_closed if t.pnl_net is not None and t.pnl_net > 0],
        key=lambda t: -(t.pnl_net or 0),
    )[:3]
    open_count = len(list_open())

    pf = summary["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"

    lines = [
        f"📊 <b>EOD Digest {d_str}</b>",
        f"Closed: {summary['n_trades']}  |  Win rate: {summary['win_rate']:.0%}",
        f"Net P&amp;L: ₹{summary['net_pnl']:+,.2f}  |  PF: {pf_str}",
    ]
    if winners:
        lines.append("")
        lines.append("🏆 <b>Top winners:</b>")
        for t in winners:
            sym = t.symbol.replace(".NS", "")
            lines.append(
                f"  • {sym} {t.status}  ₹{t.pnl_net:+,.2f}"
            )
    if open_count:
        # With strict-intraday TIMEOUT we expect 0 OPEN trades at 15:35.
        # Surfacing it as a warning makes monitor lag immediately visible.
        lines.append("")
        lines.append(
            f"⚠️ {open_count} still OPEN — monitor may be lagging."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: python -m paper.journal report --since YYYY-MM-DD
# ---------------------------------------------------------------------------

def _fmt_trades_table(trades: list[Trade]) -> str:
    """Fixed-width text table for a list of trades. SL/TP1/TP2 columns
    are present for both OPEN and CLOSED rows so the report doubles as
    a "what risk is live right now?" snapshot. TP2 prints ``—`` when
    the trade was opened with a single target only.

    Returns ``(no trades)`` when the list is empty."""
    if not trades:
        return "  (no trades)"
    header = (
        f"  {'ID':>4}  {'SYMBOL':<12}  {'SIDE':<5}  {'STATUS':<7}  "
        f"{'ENTRY':>9}  {'SL':>9}  {'TP1':>9}  {'TP2':>9}  "
        f"{'EXIT':>9}  {'PNL_NET':>10}  {'ENTRY_TS':<19}"
    )
    sep = "  " + "-" * (len(header) - 2)
    out = [header, sep]
    for t in trades:
        sl = f"{t.stop_loss:9.2f}" if t.stop_loss is not None else "        —"
        tp1 = f"{t.target_1:9.2f}" if t.target_1 is not None else "        —"
        tp2 = f"{t.target_2:9.2f}" if t.target_2 is not None else "        —"
        exit_price = f"{t.exit_price:9.2f}" if t.exit_price is not None else "        —"
        pnl_net = f"{t.pnl_net:+10.2f}" if t.pnl_net is not None else "         —"
        out.append(
            f"  {t.id:>4}  {t.symbol:<12}  {t.side:<5}  {t.status:<7}  "
            f"{t.entry_price:>9.2f}  {sl}  {tp1}  {tp2}  "
            f"{exit_price}  {pnl_net}  "
            f"{t.entry_ts[:19]:<19}"
        )
    return "\n".join(out)


def _fmt_summary(summary: dict) -> str:
    """Multi-line text rendering of a :func:`daily_summary` dict."""
    if summary["n_trades"] == 0:
        return "  (no closed trades on this date)"
    pf = summary["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
    return "\n".join([
        f"  trades closed   : {summary['n_trades']}",
        f"  win rate        : {summary['win_rate']:.1%}",
        f"  gross pnl       : ₹{summary['gross_pnl']:+,.2f}",
        f"  net pnl         : ₹{summary['net_pnl']:+,.2f}",
        f"  avg winner      : ₹{summary['avg_winner']:+,.2f}",
        f"  avg loser       : ₹{summary['avg_loser']:+,.2f}",
        f"  profit factor   : {pf_str}",
    ])


def _cli(argv: list[str] | None = None) -> int:
    """``python -m paper.journal <subcommand>`` entry point.

    Supported subcommands:

      report [--since YYYY-MM-DD|today]
          Print the OPEN trades, CLOSED trades since the given date
          (default: today in IST), and a daily summary for today."""
    parser = argparse.ArgumentParser(prog="paper.journal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    report = sub.add_parser("report", help="Print the trade journal")
    report.add_argument(
        "--since",
        default="today",
        help="ISO date YYYY-MM-DD or 'today' (default: today in IST)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "report":
        # Local import keeps the ``paper.journal`` module light at top
        # level (no IST/ZoneInfo at import time for plain reads).
        from .tracker import IST
        if args.since == "today":
            since_date = pd.Timestamp.now(tz=IST).date()
        else:
            since_date = _Date.fromisoformat(args.since)

        opens = list_open()
        closeds = list_closed(since=since_date)

        print(f"=== Paper Journal Report (closed since {since_date}) ===\n")
        print(f"OPEN TRADES ({len(opens)})")
        print(_fmt_trades_table(opens))
        print()
        print(f"CLOSED TRADES ({len(closeds)})")
        print(_fmt_trades_table(closeds))
        print()
        today_ist = pd.Timestamp.now(tz=IST).date()
        summary = daily_summary(today_ist)
        print(f"=== Today ({today_ist}) ===")
        print(_fmt_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
