"""Multi-strategy / multi-symbol sweep runner.

Sister of :func:`strategies.backtest.run_backtest` — runs the engine across
the cartesian product of (strategies × symbols), pulls bars via
:mod:`strategies.data`, and returns one row per combo for direct comparison."""
from __future__ import annotations

import logging

import pandas as pd

from . import REGISTRY, run_backtest
from . import data as strat_data

log = logging.getLogger("strategies.sweep")

StrategySpec = tuple[str, dict]
"""``(name, params_dict)``. ``params_dict={}`` uses the strategy's defaults."""


def run_sweep(
    strategy_specs: list[StrategySpec],
    symbols: list[str],
    *,
    interval: str = "5m",
    days: int = 60,
    starting_equity: float = 100_000.0,
    qty_pct: float = 0.10,
    max_intraday_loss_pct: float = 1.0,
    apply_costs: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Run each (strategy, symbol) combo and return one row per result.

    Loads each symbol's bars exactly once (then runs every strategy against
    that DataFrame) so we don't re-read the parquet per strategy."""
    rows: list[dict] = []
    n_combos = len(strategy_specs) * len(symbols)
    log.info("Sweep: %d strategies x %d symbols = %d combos",
             len(strategy_specs), len(symbols), n_combos)

    for sym in symbols:
        try:
            df = strat_data.load(sym, interval=interval, days=days,
                                 refresh=refresh)
        except KeyError as e:
            log.warning("Skipping %s: %s", sym, e)
            continue

        for name, params in strategy_specs:
            cls = REGISTRY.get(name)
            if cls is None:
                raise KeyError(f"Unknown strategy: {name!r}")
            strat = cls(**params)
            r = run_backtest(
                strat, df,
                starting_equity=starting_equity,
                qty_pct=qty_pct,
                max_intraday_loss_pct=max_intraday_loss_pct,
                apply_costs=apply_costs,
            )
            rows.append({
                "symbol": sym,
                "strategy": name,
                "params": str(params) if params else "default",
                "bars": len(df),
                "trades": r.num_trades,
                "wins": r.num_wins,
                "win_rate_pct": r.win_rate_pct,
                "total_return_pct": r.total_return_pct,
                "max_dd_pct": r.max_drawdown_pct,
                "kill_fires": r.kill_switch_triggers,
                "ending_equity": r.ending_equity,
            })
    return pd.DataFrame(rows)


def rollup_by_strategy(results: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy averages across symbols. ``win_rate_pct`` is
    trade-count-weighted (not symbol-equal-weighted) so a symbol with
    600 trades counts more than one with 5."""
    if results.empty:
        return pd.DataFrame(columns=[
            "strategy", "symbols", "trades", "win_rate_pct",
            "avg_return_pct", "avg_max_dd_pct", "kill_fires",
        ])
    out = []
    for (name, params), g in results.groupby(["strategy", "params"], sort=False):
        total_trades = int(g["trades"].sum())
        total_wins = int(g["wins"].sum())
        weighted_wr = (total_wins / total_trades * 100.0) if total_trades else 0.0
        out.append({
            "strategy": name,
            "params": params,
            "symbols": int(len(g)),
            "trades": total_trades,
            "win_rate_pct": weighted_wr,
            "avg_return_pct": float(g["total_return_pct"].mean()),
            "avg_max_dd_pct": float(g["max_dd_pct"].mean()),
            "kill_fires": int(g["kill_fires"].sum()),
        })
    return pd.DataFrame(out)
