"""Filter-chain smoke / backfill tool.

Runs the Phase-6 chain across the entire watchlist using the
currently-available data (daily cache, today's intraday from
yfinance). Writes a ``filter_audit`` row for every symbol that
reaches scoring, exactly as the live scanner would — but without
sending Telegram messages or opening paper trades.

Use AFTER market close to inspect the day's kill-reason distribution
and verify the chain is behaving sensibly. Suitable as the
"backfill yesterday's day" check in the Phase-6 acceptance tests.

Caveat: historical days pre-Phase-6 have no audit data — this
script can only snapshot the CURRENT state, not reconstruct past
sessions retroactively. For per-bar replay across an entire day,
extend this with a loop over ``bars_5m`` timestamps; the chain
is deterministic in ``ctx.now`` so frozen-time replay is safe.

Usage:
    python -m scripts.backfill_filter_audit
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

import pandas as pd

from bot import market_data
from bot.config import IST, settings
from bot.db import init_db
from bot.scoring import score_stock
from bot.watchlist import WATCHLIST
from data import fno_ban, index_feed
from filters import FilterContext, apply_filters
from filters.chain import write_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("backfill")


def _build_shared_ctx_fields() -> dict:
    """Fetch the cross-symbol context once — same fields the scanner
    builds per-symbol in ``_evaluate_symbol``, but reused across the
    full watchlist for efficiency."""
    return {
        "now": pd.Timestamp.now(tz=IST).to_pydatetime(),
        "nifty_pct": index_feed.get_intraday_pct(index_feed.NIFTY),
        "bank_nifty_pct": index_feed.get_intraday_pct(index_feed.BANK_NIFTY),
        "vix": index_feed.get_vix(),
        "fno_banned": fno_ban.get_banned(),
    }


def run() -> dict:
    init_db()
    market_data.refresh_daily_cache_if_stale()
    asyncio.run(fno_ban.refresh())

    shared = _build_shared_ctx_fields()
    log.info("Shared context: nifty=%.2f%% banknifty=%.2f%% vix=%.2f banned=%d",
             shared["nifty_pct"] or 0.0,
             shared["bank_nifty_pct"] or 0.0,
             shared["vix"] or 0.0,
             len(shared["fno_banned"]))

    log.info("Fetching intraday data for %d watchlist symbols...",
             len(WATCHLIST))
    intraday_data = market_data.fetch_intraday(WATCHLIST)

    counts = Counter()
    kill_reasons = Counter()
    confidence_buckets = Counter()

    for symbol in WATCHLIST:
        intraday = intraday_data.get(symbol)
        daily = market_data._daily_cache.get(symbol, pd.DataFrame())
        if intraday is None or intraday.empty or daily.empty:
            counts["no_data"] += 1
            continue
        try:
            signals = score_stock(symbol, intraday, daily)
        except Exception:
            log.exception("score_stock failed for %s", symbol)
            counts["score_error"] += 1
            continue

        ctx = FilterContext(
            daily_df=daily, intraday_df=intraday, **shared,
        )

        result = apply_filters(signals, ctx)
        if result is None:
            counts["killed"] += 1
            for reason in signals.kill_reasons:
                # Bucket "cooldown (60m)" → "cooldown" etc. for tidier counts.
                reason_root = reason.split("(")[0].split(",")[0].strip()
                kill_reasons[reason_root] += 1
            continue

        confidence_pct = signals.confidence * 100
        if confidence_pct < settings.composite_threshold:
            counts["below_threshold"] += 1
            write_audit(signals, alerted=False)
        elif confidence_pct < settings.telegram_threshold:
            counts["silent_paper"] += 1
            write_audit(signals, alerted=False)
        else:
            counts["telegram_alert"] += 1
            write_audit(signals, alerted=True)

        # Confidence histogram (10pp buckets)
        bucket = int(confidence_pct // 10) * 10
        confidence_buckets[f"{bucket}-{bucket + 9}"] += 1

    return {
        "counts": dict(counts),
        "kill_reasons": dict(kill_reasons),
        "confidence_buckets": dict(confidence_buckets),
    }


def main() -> int:
    result = run()
    print("\n=== Filter-chain snapshot ===\n")
    print("Outcome distribution:")
    for k, v in sorted(result["counts"].items(), key=lambda x: -x[1]):
        print(f"  {v:>5}  {k}")
    print("\nTop kill reasons:")
    for reason, count in sorted(
        result["kill_reasons"].items(), key=lambda x: -x[1]
    )[:10]:
        print(f"  {count:>5}  {reason}")
    print("\nConfidence histogram (passed signals only):")
    for bucket, count in sorted(result["confidence_buckets"].items()):
        bar = "█" * min(count, 40)
        print(f"  {bucket:>8}  {count:>4}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
