"""Filter chain orchestrator + audit writer + ``FilterContext``.

The chain runs in two phases:

1. **Hard pass** — iterate :data:`HARD_FILTERS`. First filter to
   return a non-empty kill reason short-circuits: kill_reason
   appended to ``signals.kill_reasons``, audit row written with
   ``alerted=False``, and ``None`` returned.

2. **Multiplier pass** — iterate
   :data:`SOFT_FILTERS` + :data:`TIME_FILTERS` + :data:`EVENT_FILTERS`.
   Each filter returns ``(name, mult) | None``. Multipliers compose
   onto a product; ``confidence = (score/100) × product``.

The chain MUTATES ``signals`` in place. Callers should treat the
returned object as the same instance they passed in.

Audit row policy: the chain writes ``alerted=False`` ONLY for
hard-killed signals. Signals that pass the chain (returned, not
None) have their audit row written by the scanner after the
threshold / suppression decision, because the chain itself doesn't
know whether the alert ultimately fires."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from bot.scoring import StockSignals

log = logging.getLogger("alertbot.filters")


@dataclass
class FilterContext:
    """Per-tick shared state. Built once in ``scan_once`` /
    ``scan_symbol`` so individual filters don't refetch indices /
    F&O ban list / etc.

    Every field is optional except ``now`` because individual
    filters fail-open on missing context (they return ``None`` and
    let the next filter try). This keeps the chain robust to
    partial-data scans (e.g. when the index fetcher is down)."""
    now: datetime
    daily_df: "pd.DataFrame | None" = None
    intraday_df: "pd.DataFrame | None" = None
    nifty_pct: float | None = None
    bank_nifty_pct: float | None = None
    vix: float | None = None
    fno_banned: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Filter registries — populated by the four sibling modules
# ---------------------------------------------------------------------------

# Hard filter: (signals, ctx) -> str | None (kill reason or pass)
HardFilter = Callable[["StockSignals", FilterContext], "str | None"]

# Multiplier filter: (signals, ctx) -> tuple[name, mult] | None
SoftFilter = Callable[
    ["StockSignals", FilterContext], "tuple[str, float] | None",
]

HARD_FILTERS: tuple[HardFilter, ...] = ()
SOFT_FILTERS: tuple[SoftFilter, ...] = ()
TIME_FILTERS: tuple[SoftFilter, ...] = ()
EVENT_FILTERS: tuple[SoftFilter, ...] = ()


def register_filters() -> None:
    """Populate the registries from sibling modules. Called once
    from :mod:`filters.__init__` after every submodule has finished
    loading — that way module-level ``from .hard import ...`` would
    have caused a circular dependency, but the runtime call from
    ``__init__`` is safe.

    Tests that want to substitute filter sets do so by
    ``monkeypatch.setattr(chain, "HARD_FILTERS", (my_filter,))``;
    they should NOT call this function (it would clobber the patch)."""
    global HARD_FILTERS, SOFT_FILTERS, TIME_FILTERS, EVENT_FILTERS
    from . import event, hard, soft
    from . import time as time_filters
    HARD_FILTERS = hard.HARD_FILTERS
    SOFT_FILTERS = soft.SOFT_FILTERS
    TIME_FILTERS = time_filters.TIME_FILTERS
    EVENT_FILTERS = event.EVENT_FILTERS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_filters(
    signals: "StockSignals",
    ctx: FilterContext,
) -> "StockSignals | None":
    """Run the full filter chain against ``signals``. Mutates and
    returns ``signals`` on pass; writes an audit row and returns
    ``None`` on hard-kill.

    Multiplier composition: ``confidence = (score/100) × Π
    multipliers``. With a score of 70 and three multipliers
    ``(0.85, 1.10, 0.4)`` the resulting ``confidence`` is
    ``0.70 × 0.85 × 1.10 × 0.4 ≈ 0.2618`` → compared as
    ``26.18`` against Phase-5b thresholds."""
    # ---- Hard pass --------------------------------------------------
    for f in HARD_FILTERS:
        reason = f(signals, ctx)
        if reason:
            signals.kill_reasons.append(reason)
            log.info("Filter kill %s: %s", signals.symbol, reason)
            write_audit(signals, alerted=False)
            return None

    # ---- Multiplier pass --------------------------------------------
    product = 1.0
    for f in SOFT_FILTERS + TIME_FILTERS + EVENT_FILTERS:
        adj = f(signals, ctx)
        if adj is None:
            continue
        name, mult = adj
        signals.soft_adjustments.append((name, mult))
        product *= mult

    signals.confidence = (signals.score / 100.0) * product
    return signals


def write_audit(signals: "StockSignals", *, alerted: bool) -> None:
    """Append a row to ``filter_audit``. Caller decides ``alerted``
    based on whether the signal ultimately reached _dispatch.

    Phase-7 columns ``components_json`` + ``final_score`` are pulled
    from ``signals.score_breakdown`` when present (set by the
    scanner after :func:`scoring.score_signal`). Hard-killed signals
    short-circuit before scoring, so those rows leave both NULL.

    Best-effort: a DB-write failure here must NOT break the scan
    loop. Errors are logged and swallowed."""
    from bot.config import DB_PATH

    soft_json = json.dumps(
        [[name, mult] for name, mult in signals.soft_adjustments]
    )
    kill_str = ", ".join(signals.kill_reasons) if signals.kill_reasons else None

    breakdown = signals.score_breakdown
    if breakdown is None:
        components_json = None
        final_score: float | None = None
    else:
        components_json = json.dumps(breakdown.get("components") or {})
        raw_final = breakdown.get("final")
        final_score = float(raw_final) if raw_final is not None else None

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO filter_audit "
                "(ts, symbol, side, score, kill_reasons, "
                " soft_adjustments_json, final_confidence, alerted, "
                " components_json, final_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    signals.symbol,
                    signals.side,
                    signals.score,
                    kill_str,
                    soft_json,
                    signals.confidence,
                    1 if alerted else 0,
                    components_json,
                    final_score,
                ),
            )
    except Exception:
        log.exception("filter_audit write failed for %s", signals.symbol)
