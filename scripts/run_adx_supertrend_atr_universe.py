"""Run the ADX+Supertrend+ATR strategy across the full NIFTY 500 universe.

Mirrors ``scripts/run_vwap_universe.py``. Writes:

  - results/sweep_adx_supertrend_atr_universe.csv      (per-symbol summary)
  - results/adx_supertrend_atr_universe_trades.parquet (trades, for the
                                                        dashboard's time
                                                        bucket analytics)

Run::

    python scripts/run_adx_supertrend_atr_universe.py             # fetch + sweep
    python scripts/run_adx_supertrend_atr_universe.py --no-fetch  # cache-only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from bot.watchlist import WATCHLIST  # noqa: E402
from strategies import REGISTRY, run_backtest  # noqa: E402
from strategies import data as strat_data  # noqa: E402

log = logging.getLogger("adx_supertrend_atr_universe")

STRATEGY_NAME = "adx_supertrend_atr"
OUT_SUMMARY = Path("results/sweep_adx_supertrend_atr_universe.csv")
OUT_TRADES = Path("results/adx_supertrend_atr_universe_trades.parquet")


def _chunk(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _fetch_chunked(symbols: list[str], chunk_size: int, sleep_s: float) -> None:
    chunks = _chunk(symbols, chunk_size)
    for n, batch in enumerate(chunks, 1):
        log.info("Fetch batch %d/%d (%d symbols)...",
                 n, len(chunks), len(batch))
        try:
            strat_data.fetch(batch, interval="5m", days=60, refresh=False)
        except Exception as e:
            log.warning("Batch %d failed: %s", n, e)
        if sleep_s > 0 and n < len(chunks):
            time.sleep(sleep_s)


def _sweep(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat_cls = REGISTRY[STRATEGY_NAME]
    summary_rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []

    for i, sym in enumerate(symbols, 1):
        try:
            df = strat_data.load(sym, interval="5m", days=60, refresh=False)
        except KeyError:
            log.warning("Skip %s: not in cache", sym)
            continue
        if i % 25 == 0:
            log.info("Sweep progress: %d / %d", i, len(symbols))

        strat = strat_cls()  # defaults
        r = run_backtest(
            strat, df,
            starting_equity=100_000.0, qty_pct=0.10,
            max_intraday_loss_pct=1.0, apply_costs=True,
        )
        summary_rows.append({
            "symbol": sym, "bars": len(df),
            "trades": r.num_trades, "wins": r.num_wins,
            "win_rate_pct": r.win_rate_pct,
            "total_return_pct": r.total_return_pct,
            "max_dd_pct": r.max_drawdown_pct,
            "kill_fires": r.kill_switch_triggers,
            "ending_equity": r.ending_equity,
        })
        tdf = r.trades_df()
        if not tdf.empty:
            tdf = tdf.copy()
            tdf["symbol"] = sym
            all_trades.append(tdf)

    summary = pd.DataFrame(summary_rows)
    trades = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades else pd.DataFrame()
    )
    return summary, trades


def main() -> int:
    p = argparse.ArgumentParser(
        prog="scripts/run_adx_supertrend_atr_universe.py",
    )
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip yfinance fetch; sweep only what's already cached.")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    universe = WATCHLIST[: args.limit] if args.limit else list(WATCHLIST)
    log.info("Universe size: %d symbols", len(universe))

    if not args.no_fetch:
        _fetch_chunked(universe, chunk_size=args.chunk, sleep_s=args.sleep)
    else:
        log.info("--no-fetch: skipping yfinance, sweeping cache only")

    summary, trades = _sweep(universe)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    log.info("Wrote %d summary rows -> %s", len(summary), OUT_SUMMARY)

    if not trades.empty:
        trades.to_parquet(OUT_TRADES, index=False)
        log.info("Wrote %d trades -> %s", len(trades), OUT_TRADES)
    else:
        log.warning("No trades generated — nothing written to %s", OUT_TRADES)

    total_trades = int(summary["trades"].sum()) if not summary.empty else 0
    total_wins = int(summary["wins"].sum()) if not summary.empty else 0
    weighted_wr = (total_wins / total_trades * 100.0) if total_trades else 0.0
    profitable = (
        int((summary["total_return_pct"] > 0).sum())
        if not summary.empty else 0
    )

    bar = "=" * 72
    print()
    print(bar)
    print(f"ADX+SUPERTREND+ATR — UNIVERSE SWEEP   ({len(summary)} symbols)")
    print(bar)
    print(f"Total trades             : {total_trades:>12d}")
    print(f"Trade-weighted win rate  : {weighted_wr:>11.2f}%")
    print(f"Profitable symbols       : {profitable:>5d} / {len(summary)}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
