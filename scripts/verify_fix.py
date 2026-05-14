"""Verify the snapshot-attach + ADX-key fixes by re-scoring the user's
named stocks against today's intraday data, end-to-end through the
patched scoring pipeline (mirroring scanner._evaluate_symbol)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncio  # noqa: E402

import pandas as pd  # noqa: E402

from bot import market_data  # noqa: E402
from bot.config import IST  # noqa: E402
from bot.scoring import score_stock  # noqa: E402
from data import fno_ban, precompute as daily_precompute  # noqa: E402
from data.market_context import get_market_context  # noqa: E402
from filters import FilterContext, apply_filters  # noqa: E402
from indicators import compute_all  # noqa: E402
import scoring as scoring_engine  # noqa: E402

SYMBOLS = [
    "BHARTIARTL.NS", "BSE.NS", "CIPLA.NS",
    "MCX.NS", "ADANIPOWER.NS", "HFCL.NS",
    "GODREJIND.NS", "ALKYLAMINE.NS",
]


def build_snapshot(symbol, intraday, daily):
    if intraday.empty or daily.empty:
        return None
    bars = intraday.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return compute_all(
        symbol=symbol,
        bars=bars,
        daily_df=daily,
        session_date=intraday.index[-1].date(),
        target_timeframes=("5m", "15m", "60m"),
        indicators=(
            "rsi", "atr", "adx", "macd",
            "stochastic", "mfi", "cci",
            "supertrend", "bollinger", "cmf",
            "volume_surge_ratio", "ttm_squeeze",
            "pivot_points", "previous_day_hlc", "opening_range",
        ),
    )


async def main():
    print("Refreshing daily cache...")
    market_data.refresh_daily_cache_if_stale()
    await fno_ban.refresh()

    print("Fetching intraday for", SYMBOLS)
    intraday_data = market_data.fetch_intraday(SYMBOLS)

    market_context = await get_market_context()
    print(f"\nMarket context: nifty={market_context.nifty_change_pct}% "
          f"bn={market_context.banknifty_change_pct}% "
          f"vix={market_context.vix} fii={market_context.fii_net_cr}")

    threshold = scoring_engine.get_alert_threshold(default=60)
    print(f"Alert threshold: {threshold}\n")
    print("=" * 110)

    for sym in SYMBOLS:
        if sym not in intraday_data:
            print(f"\n{sym}: NO INTRADAY DATA")
            continue
        intraday = intraday_data[sym]
        daily = market_data._daily_cache.get(sym, pd.DataFrame())

        snapshot = build_snapshot(sym, intraday, daily)
        signals = score_stock(sym, intraday, daily, snapshot=snapshot)
        signals.snapshot = snapshot  # the patch
        signals.market_context = market_context.to_dict()

        ctx = FilterContext(
            now=pd.Timestamp.now(tz=IST).to_pydatetime(),
            daily_df=daily,
            intraday_df=intraday,
            nifty_pct=market_context.nifty_change_pct,
            bank_nifty_pct=market_context.banknifty_change_pct,
            vix=market_context.vix,
            fno_banned=fno_ban.get_banned(),
        )
        kept = apply_filters(signals, ctx)
        if kept is None:
            print(f"\n{sym}: HARD-KILLED — {signals.kill_reasons}")
            continue

        daily_levels = daily_precompute.get_daily_levels(
            sym, intraday.index[-1].date(),
        )
        breakdown = scoring_engine.score_signal(
            signals, daily_levels=daily_levels, news_score=None,
        )

        print(f"\n{sym}  price={signals.price:.2f}  side={signals.side}")
        print(f"  components:")
        for k, v in breakdown.components.items():
            print(f"    {k:12s} {v:6.2f}")
        print(f"  base score (pre-multipliers): {breakdown.base:.2f}")
        print(f"  soft adjustments: {signals.soft_adjustments}")
        print(f"  multiplier product: {breakdown.multiplier_product:.4f}")
        print(f"  FINAL SCORE: {breakdown.final:.2f} "
              f"(threshold {threshold:.0f}) "
              f"{'>>> WOULD ALERT <<<' if breakdown.final >= threshold else '(below gate)'}")


if __name__ == "__main__":
    asyncio.run(main())
