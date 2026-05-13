"""Orchestrator: fetch → dedupe → symbol-match → FinBERT batch → DB.

Public surface:
  ``update_news_scores(watchlist=None)``  — async, called by the
        runner every 10 minutes during market hours.
  ``get_symbol_news_score(symbol, hours=24, db_path=None)``  —
        synchronous read used by ``master.score_signal``.

Aggregation rule (LONG-side semantics):
  ``score = Σ (finbert_score · relevance · recency_weight) /
            Σ (relevance · recency_weight)``
  where ``recency_weight = exp(-age_hours / 12)`` — half-life ≈ 8 h.

  Cap output at [-1, +1]. Empty / no-news → ``None`` (the scorer
  treats that as neutral 50 in score_news, not -1)."""
from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from core.config import DB_PATH

from . import fetcher, finbert, symbol_match

log = logging.getLogger("alertbot.news.scorer")

RECENCY_HALF_LIFE_HOURS = 12.0
"""``recency_weight = exp(-age/12)``. 12 hours roughly tracks the
intraday news cycle in Indian markets — by next-session-open,
yesterday-evening news still has ~30% of its initial weight; one
week old → ~0.3%, effectively zero."""

DEFAULT_LOOKBACK_HOURS = 24


# ---------------------------------------------------------------------------
# Pure aggregation (test-friendly)
# ---------------------------------------------------------------------------

def _recency_weight(age_hours: float) -> float:
    if age_hours <= 0:
        return 1.0
    return math.exp(-age_hours / RECENCY_HALF_LIFE_HOURS)


def aggregate_rows(
    rows: list[tuple[float, float, float]],
) -> float | None:
    """``rows`` is a list of ``(finbert_score, relevance, age_hours)``.
    Returns the weighted-average sentiment in [-1, +1], or ``None``
    if no rows / total weight is zero."""
    if not rows:
        return None
    num = 0.0
    den = 0.0
    for score, rel, age in rows:
        if rel <= 0:
            continue
        w = rel * _recency_weight(age)
        num += score * w
        den += w
    if den <= 0:
        return None
    return max(-1.0, min(1.0, num / den))


# ---------------------------------------------------------------------------
# DB read
# ---------------------------------------------------------------------------

def get_symbol_news_score(
    symbol: str,
    hours: int = DEFAULT_LOOKBACK_HOURS,
    db_path: str | None = None,
    now_utc: datetime | None = None,
) -> float | None:
    """Aggregate weighted sentiment for ``symbol`` over the last
    ``hours``. Returns ``None`` when no rows exist for the window —
    that's the "no news" case the scorer treats as neutral."""
    path = db_path if db_path is not None else DB_PATH
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    try:
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                "SELECT ni.finbert_score, ns.relevance, ni.published_at, ni.fetched_at "
                "FROM news_scores ns "
                "JOIN news_items ni ON ni.id = ns.news_id "
                "WHERE ns.symbol = ? "
                "  AND ni.finbert_score IS NOT NULL",
                (symbol,),
            )
            raw = cur.fetchall()
    except sqlite3.OperationalError:
        return None

    rows: list[tuple[float, float, float]] = []
    cutoff_secs = hours * 3600
    for fb_score, rel, published, fetched in raw:
        ts_str = published or fetched
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (now_utc - ts).total_seconds()
        if age < 0 or age > cutoff_secs:
            continue
        rows.append((float(fb_score), float(rel), age / 3600.0))

    return aggregate_rows(rows)


# ---------------------------------------------------------------------------
# DB write — orchestrator
# ---------------------------------------------------------------------------

async def update_news_scores(
    watchlist: Iterable[str] | None = None,
    *,
    db_path: str | None = None,
) -> dict[str, int]:
    """Pull fresh items, run them through FinBERT in one batch, write
    ``news_items`` + ``news_scores``.

    Returns a summary dict ``{fetched, new, scored, symbols_with_news}``.

    Idempotent on ``news_items.id`` (sha1 of source+url+title) —
    re-fetching the same headline is a no-op at the row level."""
    if watchlist is None:
        from bot.watchlist import WATCHLIST as _WL
        watchlist = list(_WL)
    path = db_path if db_path is not None else DB_PATH

    items = await fetcher.fetch_all()
    if not items:
        log.info("update_news_scores: 0 items fetched")
        return {"fetched": 0, "new": 0, "scored": 0, "symbols_with_news": 0}

    # Filter out items we've already scored — saves both the FinBERT
    # batch cost and a redundant write.
    fresh = _filter_unseen(items, path)
    log.info("update_news_scores: %d fetched, %d unseen", len(items), len(fresh))

    if fresh:
        fb = finbert.get_finbert()
        texts = [_text_for_inference(it) for it in fresh]
        scored = fb.score_batch_with_labels(texts)
    else:
        scored = []

    symbols_with_news: set[str] = set()
    fetched_iso = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(path) as conn:
        for item, (score, label) in zip(fresh, scored):
            conn.execute(
                "INSERT OR IGNORE INTO news_items "
                "(id, source, title, body, url, published_at, fetched_at, "
                " finbert_score, finbert_label) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id, item.source, item.title, item.body, item.url,
                    item.published_at or None, fetched_iso, score, label,
                ),
            )
            matches = symbol_match.match_symbols(
                f"{item.title} {item.body}", watchlist=watchlist,
            )
            for ticker, rel in matches:
                conn.execute(
                    "INSERT OR REPLACE INTO news_scores "
                    "(symbol, news_id, relevance) VALUES (?, ?, ?)",
                    (ticker, item.id, float(rel)),
                )
                symbols_with_news.add(ticker)

    log.info(
        "update_news_scores: scored %d items, %d symbols affected",
        len(scored), len(symbols_with_news),
    )
    return {
        "fetched": len(items),
        "new": len(fresh),
        "scored": len(scored),
        "symbols_with_news": len(symbols_with_news),
    }


def _filter_unseen(items: list, db_path: str) -> list:
    """Return only the items whose ``id`` is NOT already in
    ``news_items``. Reduces FinBERT batch size on the steady-state
    path where most items repeat across fetches."""
    if not items:
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            ids = ",".join(["?"] * len(items))
            cur = conn.execute(
                f"SELECT id FROM news_items WHERE id IN ({ids})",
                [it.id for it in items],
            )
            existing = {row[0] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        existing = set()
    return [it for it in items if it.id not in existing]


def _text_for_inference(item) -> str:
    """Concatenate title + body, capped at 512 chars — FinBERT's
    tokenizer truncates to 512 tokens anyway; this avoids paying for
    encoding text the model is going to drop."""
    raw = f"{item.title}. {item.body}".strip(". ")
    return raw[:512]
