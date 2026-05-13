"""Phase-8 morning precompute — populate ``daily_levels`` once at
09:00 IST.

Pivot, CPR, Fibonacci, and PDH/PDL/PDC are session-constant: they
depend only on the prior trading day's H/L/C (and a swing window for
fib). Computing them per-tick is wasted work. This module:

  1. ``run_morning_compute``  — async entry-point scheduled at 09:00.
     For every watchlist symbol, derives prior-day OHLC from the
     in-memory ``bot.market_data._daily_cache`` (yfinance-fed), bundles
     levels via ``indicators.structure.bundle_daily_levels``, and
     UPSERTs into ``daily_levels``. Sends a Telegram summary
     (``N/M symbols ready, K missing: ...``).

  2. ``get_daily_levels``    — synchronous reader used by the scoring
     engine. Returns ``dict | None``; None means the precompute
     hasn't run for ``(symbol, session_date)`` yet, and the caller
     should treat structure as neutral.

Failure handling: per-symbol exceptions are caught and counted; one
bad symbol doesn't break the whole precompute. The Telegram digest
lists up to 20 missing symbols so the operator can see whether the
miss-set is pathological (yfinance outage) or routine (handful of
delistings)."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date as _Date, datetime, timedelta
from typing import Iterable

import pandas as pd

from core.config import DB_PATH, IST
from indicators.structure import bundle_daily_levels

log = logging.getLogger("alertbot.precompute")

SWING_LOOKBACK_DAYS = 20
"""Daily bars used to derive ``swing_high`` / ``swing_low`` for the
Fibonacci block. 20 sessions ≈ one trading month — long enough that
the swing isn't dominated by yesterday's bar, short enough to remain
intraday-relevant."""

MAX_MISSING_REPORTED = 20
"""Cap on the missing-symbols list inside the Telegram digest. Above
this we summarise with a count to keep the message readable."""


# ---------------------------------------------------------------------------
# Pure helpers — exercised directly by tests
# ---------------------------------------------------------------------------

def _prev_session_row(
    daily_df: pd.DataFrame, session_date: _Date,
) -> pd.Series | None:
    """Return the most recent daily row strictly BEFORE ``session_date``.

    Lowercases column lookup so the caller can pass yfinance-style
    ``High``/``Low``/``Close`` or pre-normalised lowercase frames."""
    if daily_df is None or daily_df.empty:
        return None
    idx = daily_df.index
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.date
    else:
        dates = [
            d.date() if hasattr(d, "date") else d for d in idx
        ]
    mask = pd.Series([d < session_date for d in dates], index=idx)
    if not mask.any():
        return None
    return daily_df.loc[mask].iloc[-1]


def _swing_high_low(
    daily_df: pd.DataFrame, session_date: _Date,
    lookback: int = SWING_LOOKBACK_DAYS,
) -> tuple[float, float] | None:
    """Last ``lookback`` daily bars strictly before ``session_date`` →
    ``(max(high), min(low))`` for the Fibonacci swing. Returns None
    when there isn't enough history."""
    if daily_df is None or daily_df.empty:
        return None
    idx = daily_df.index
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.date
    else:
        dates = [
            d.date() if hasattr(d, "date") else d for d in idx
        ]
    mask = pd.Series([d < session_date for d in dates], index=idx)
    if not mask.any():
        return None
    window = daily_df.loc[mask].tail(lookback)
    if len(window) < 2:
        return None
    cols = {c.lower(): c for c in window.columns}
    if "high" not in cols or "low" not in cols:
        return None
    try:
        return float(window[cols["high"]].max()), float(window[cols["low"]].min())
    except (ValueError, TypeError):
        return None


def compute_for_symbol(
    symbol: str, daily_df: pd.DataFrame, session_date: _Date,
) -> dict[str, object] | None:
    """Build the bundle for one symbol. Returns None when prior-day
    data is missing — that's the signal to the caller to bucket the
    symbol into the "missing" list rather than write an empty row."""
    prev = _prev_session_row(daily_df, session_date)
    if prev is None:
        return None
    cols = {c.lower(): c for c in daily_df.columns}
    required = ("high", "low", "close")
    for col in required:
        if col not in cols:
            log.warning("precompute %s: missing column %s in daily_df",
                        symbol, col)
            return None
    try:
        pdh = float(prev[cols["high"]])
        pdl = float(prev[cols["low"]])
        pdc = float(prev[cols["close"]])
    except (TypeError, ValueError):
        return None
    if not all(p == p for p in (pdh, pdl, pdc)):  # NaN guard
        return None

    swing = _swing_high_low(daily_df, session_date)
    if swing is not None:
        swing_high, swing_low = swing
    else:
        swing_high = swing_low = None
    return bundle_daily_levels(
        pdh, pdl, pdc,
        swing_high=swing_high, swing_low=swing_low,
    )


# ---------------------------------------------------------------------------
# DB read / write
# ---------------------------------------------------------------------------

def _upsert_levels(
    symbol: str, session_date: _Date, bundle: dict[str, object],
    db_path: str | None = None,
) -> None:
    path = db_path if db_path is not None else DB_PATH
    payload = json.dumps(bundle, default=float)
    ts = datetime.now(IST).isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO daily_levels (symbol, session_date, levels_json, computed_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(symbol, session_date) DO UPDATE SET "
            "  levels_json = excluded.levels_json, "
            "  computed_at = excluded.computed_at",
            (symbol, session_date.isoformat(), payload, ts),
        )


def get_daily_levels(
    symbol: str, session_date: _Date,
    db_path: str | None = None,
) -> dict[str, object] | None:
    """Read the precomputed bundle for ``(symbol, session_date)`` from
    SQLite. Returns ``None`` when no row exists yet — callers should
    fall back to treating structure as neutral, not error out."""
    path = db_path if db_path is not None else DB_PATH
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT levels_json FROM daily_levels "
                "WHERE symbol = ? AND session_date = ?",
                (symbol, session_date.isoformat()),
            ).fetchone()
    except sqlite3.OperationalError:
        # Schema not yet initialised (test path, very early startup).
        return None
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        log.exception("daily_levels: malformed JSON for %s %s",
                      symbol, session_date)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_morning_compute(
    telegram=None,
    *,
    watchlist: Iterable[str] | None = None,
    daily_cache: dict[str, pd.DataFrame] | None = None,
    session_date: _Date | None = None,
    db_path: str | None = None,
) -> dict[str, object]:
    """Compute daily levels for every watchlist symbol and store in
    ``daily_levels``.

    Parameters defaulted from the live bot:
      * ``watchlist`` — ``bot.watchlist.WATCHLIST`` when None
      * ``daily_cache`` — ``bot.market_data._daily_cache`` (refreshed
        on entry when None)
      * ``session_date`` — today in IST when None

    Returns a summary dict ``{ok, missing, errors, total}`` (the same
    payload used to format the Telegram digest)."""
    # Local imports so the precompute module stays importable in tests
    # that don't load the full bot package.
    from bot import market_data
    from bot.watchlist import WATCHLIST as _WL

    if watchlist is None:
        watchlist = _WL
    if session_date is None:
        session_date = datetime.now(IST).date()
    if daily_cache is None:
        market_data.refresh_daily_cache_if_stale()
        daily_cache = market_data._daily_cache

    ok: list[str] = []
    missing: list[str] = []
    errors: list[tuple[str, str]] = []

    for symbol in watchlist:
        try:
            df = daily_cache.get(symbol)
            if df is None or df.empty:
                missing.append(symbol)
                continue
            bundle = compute_for_symbol(symbol, df, session_date)
            if bundle is None:
                missing.append(symbol)
                continue
            _upsert_levels(symbol, session_date, bundle, db_path=db_path)
            ok.append(symbol)
        except Exception as e:  # noqa: BLE001 — one bad symbol shouldn't kill the run
            log.exception("precompute %s: %s", symbol, e)
            errors.append((symbol, str(e)))

    summary = {
        "ok": len(ok),
        "missing": missing,
        "errors": errors,
        "total": len(watchlist) if hasattr(watchlist, "__len__") else len(ok) + len(missing) + len(errors),
        "session_date": session_date.isoformat(),
    }
    log.info(
        "Morning precompute %s: %d ok / %d missing / %d errors",
        summary["session_date"], summary["ok"], len(missing), len(errors),
    )

    if telegram is not None:
        try:
            await telegram.send(_format_summary(summary))
        except Exception:
            log.exception("precompute Telegram send failed")
    return summary


def _format_summary(summary: dict[str, object]) -> str:
    """Build the HTML-formatted Telegram digest. Truncates the missing
    list at :data:`MAX_MISSING_REPORTED` so the message stays readable
    even when half the watchlist fails (yfinance outage)."""
    miss = summary["missing"]
    err = summary["errors"]
    n_miss = len(miss)
    n_err = len(err)
    ok = summary["ok"]
    total = summary["total"]
    lines = [
        f"📐 <b>Pre-compute done {summary['session_date']}</b>",
        f"{ok}/{total} symbols ready",
    ]
    if n_miss:
        head = ", ".join(sorted(miss)[:MAX_MISSING_REPORTED])
        if n_miss > MAX_MISSING_REPORTED:
            head += f"  (+{n_miss - MAX_MISSING_REPORTED} more)"
        lines.append(f"⚠️ {n_miss} missing: {head}")
    if n_err:
        head = ", ".join(s for s, _ in err[:5])
        lines.append(f"❌ {n_err} errors (first 5): {head}")
    return "\n".join(lines)
