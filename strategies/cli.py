"""CLI for the strategies package.

Run a strategy on a single symbol (auto-caches the data):

    python -m strategies bollinger_bands --symbol RELIANCE.NS

Bulk-prefetch 5m data for a basket so subsequent runs are offline-fast:

    python -m strategies fetch --symbols RELIANCE.NS,TCS.NS,INFY.NS --days 60

Inspect what's already cached:

    python -m strategies cache
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import REGISTRY, rollup_by_strategy, run_backtest, run_sweep
from . import data as strat_data

log = logging.getLogger("strategies.cli")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_params(items: list[str]) -> dict[str, object]:
    """Parse ``--param key=value`` pairs, coercing to int / float / str."""
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param expected KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        try:
            value: object = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        out[key] = value
    return out


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# `fetch` subcommand — bulk prefetch
# ---------------------------------------------------------------------------

def _fetch_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m strategies fetch")
    p.add_argument("--symbols", required=True,
                   help="Comma-separated tickers, e.g. RELIANCE.NS,TCS.NS")
    p.add_argument("--interval", default="5m",
                   help="yfinance interval. Default 5m.")
    p.add_argument("--days", type=int, default=60,
                   help="Lookback days. yfinance caps 5m at ~60d. Default 60.")
    p.add_argument("--refresh", action="store_true",
                   help="Re-download even if already cached.")
    args = p.parse_args(argv)

    _setup_logging()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        raise SystemExit("--symbols cannot be empty")

    data = strat_data.fetch(syms, interval=args.interval, days=args.days,
                            refresh=args.refresh)
    path = strat_data.cache_path(args.interval, args.days)

    print()
    print(f"Cache file : {path}")
    print(f"Got data   : {len(data)}/{len(syms)} symbols")
    print()
    for sym in syms:
        df = data.get(sym)
        if df is None:
            print(f"  {sym:<20}  (no data)")
            continue
        print(f"  {sym:<20}  {len(df):>5} bars  "
              f"{df.index[0]} -> {df.index[-1]}")
    return 0


# ---------------------------------------------------------------------------
# `cache` subcommand — show what's stored
# ---------------------------------------------------------------------------

def _cache_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m strategies cache")
    p.add_argument("--interval", default="5m")
    p.add_argument("--days", type=int, default=60)
    args = p.parse_args(argv)

    summary = strat_data.cache_summary(args.interval, args.days)
    path = strat_data.cache_path(args.interval, args.days)
    print()
    print(f"Cache file: {path}")
    if summary.empty:
        print("(empty — nothing cached for this interval/days combo)")
        return 0
    print(f"Symbols   : {len(summary)}")
    print()
    print(summary.to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# `run` flow — execute a strategy on one symbol
# ---------------------------------------------------------------------------

def _run_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m strategies")
    parser.add_argument("strategy", choices=sorted(REGISTRY.keys()))
    parser.add_argument("--symbol", required=True,
                        help="yfinance ticker, e.g. RELIANCE.NS")
    parser.add_argument("--interval", default="5m",
                        help="yfinance interval (5m / 15m / 1h / 1d). Default 5m.")
    parser.add_argument("--days", type=int, default=60,
                        help="Lookback days. yfinance caps 5m at ~60d. Default 60.")
    parser.add_argument("--refresh", action="store_true",
                        help="Bypass the parquet cache and re-fetch.")
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="Starting equity (INR). Default 100,000.")
    parser.add_argument("--qty-pct", type=float, default=0.10,
                        help="Equity fraction per trade. Default 0.10 (10%%).")
    parser.add_argument("--max-loss", type=float, default=1.0,
                        help="Max intraday loss %% (kill switch). Default 1.0.")
    parser.add_argument("--no-costs", action="store_true",
                        help="Disable round-trip costs (gross-only P&L).")
    parser.add_argument("--csv", default=None,
                        help="Path to write trades CSV.")
    parser.add_argument("--param", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Strategy param (repeatable). e.g. "
                             "--param length=20 --param direction=1")
    args = parser.parse_args(argv)

    _setup_logging()

    log.info("Loading %s %s for %dd (refresh=%s)...",
             args.symbol, args.interval, args.days, args.refresh)
    try:
        df = strat_data.load(
            args.symbol, interval=args.interval, days=args.days,
            refresh=args.refresh,
        )
    except KeyError as e:
        raise SystemExit(str(e))
    log.info("Got %d bars (%s -> %s)", len(df), df.index[0], df.index[-1])

    strat_cls = REGISTRY[args.strategy]
    strat_kwargs = _parse_params(args.param)
    strat = strat_cls(**strat_kwargs)
    log.info("Running %s on %d bars...", strat, len(df))
    result = run_backtest(
        strat, df,
        starting_equity=args.equity,
        qty_pct=args.qty_pct,
        max_intraday_loss_pct=args.max_loss,
        apply_costs=not args.no_costs,
    )

    print()
    bar = "=" * 72
    print(bar)
    print(f"{strat.name.upper()}  {args.symbol}  ({args.interval}, {args.days}d)")
    print(bar)
    print(f"Starting equity     : {result.starting_equity:>14,.2f}")
    print(f"Ending equity       : {result.ending_equity:>14,.2f}")
    print(f"Total return        : {result.total_return_pct:>14.2f}%")
    print(f"Trades              : {result.num_trades:>14d}")
    print(f"Wins                : {result.num_wins:>14d}")
    print(f"Win rate            : {result.win_rate_pct:>14.2f}%")
    print(f"Max drawdown        : {result.max_drawdown_pct:>14.2f}%")
    print(f"Kill-switch fires   : {result.kill_switch_triggers:>14d}")
    print(bar)

    if args.csv:
        result.trades_df().to_csv(args.csv, index=False)
        print(f"Wrote {result.num_trades} trades -> {args.csv}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _sweep_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m strategies sweep")
    p.add_argument("--symbols", required=True,
                   help="Comma-separated tickers, e.g. RELIANCE.NS,TCS.NS")
    p.add_argument("--strategies", default=None,
                   help=("Comma-separated strategy names. Default: all. "
                         f"Available: {','.join(sorted(REGISTRY))}"))
    p.add_argument("--interval", default="5m")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--qty-pct", type=float, default=0.10)
    p.add_argument("--max-loss", type=float, default=1.0)
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--csv", default=None,
                   help="Path to write per-row results CSV.")
    args = p.parse_args(argv)

    _setup_logging()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.strategies:
        names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        names = sorted(REGISTRY)
    specs = [(n, {}) for n in names]

    results = run_sweep(
        specs, syms,
        interval=args.interval, days=args.days,
        starting_equity=args.equity, qty_pct=args.qty_pct,
        max_intraday_loss_pct=args.max_loss,
        apply_costs=not args.no_costs, refresh=args.refresh,
    )
    if results.empty:
        print("\n(no results -- check that --symbols are in the cache or "
              "available from yfinance)")
        return 1

    bar = "=" * 90
    print()
    print(bar)
    print(f"PER-(SYMBOL,STRATEGY) RESULTS  ({args.interval}, {args.days}d, "
          f"costs={'off' if args.no_costs else 'on'})")
    print(bar)
    show = results[[
        "symbol", "strategy", "trades", "win_rate_pct",
        "total_return_pct", "max_dd_pct", "kill_fires",
    ]].copy()
    show["win_rate_pct"] = show["win_rate_pct"].round(1)
    show["total_return_pct"] = show["total_return_pct"].round(2)
    show["max_dd_pct"] = show["max_dd_pct"].round(2)
    print(show.to_string(index=False))

    rollup = rollup_by_strategy(results)
    print()
    print(bar)
    print("ROLLUP BY STRATEGY  (win rate is trade-count-weighted)")
    print(bar)
    rshow = rollup.copy()
    for col in ("win_rate_pct", "avg_return_pct", "avg_max_dd_pct"):
        rshow[col] = rshow[col].round(2)
    print(rshow.to_string(index=False))
    print()

    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"Wrote {len(results)} rows -> {args.csv}")
    return 0


_SUBCOMMANDS = {"fetch": _fetch_main, "cache": _cache_main, "sweep": _sweep_main}


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in _SUBCOMMANDS:
        return _SUBCOMMANDS[args_list[0]](args_list[1:])
    return _run_main(args_list)


if __name__ == "__main__":
    sys.exit(main())
