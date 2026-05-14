"""Tests for strategies.sweep — multi-strategy / multi-symbol runner."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import rollup_by_strategy, run_sweep
from strategies import data as strat_data

IST = ZoneInfo("Asia/Kolkata")


def _synth(n_bars: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic OHLCV frame with enough variation that BarUpDn /
    BollingerBands will both produce some trades."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz=IST)
    closes = 100.0 + np.cumsum(rng.uniform(-1, 1, n_bars))
    opens = closes + rng.uniform(-0.5, 0.5, n_bars)
    highs = np.maximum(opens, closes) + rng.uniform(0, 0.5, n_bars)
    lows = np.minimum(opens, closes) - rng.uniform(0, 0.5, n_bars)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": rng.integers(1000, 10000, n_bars).astype(float),
    }, index=idx)


@pytest.fixture
def fake_loader(monkeypatch):
    """Replace strategies.data.load with a deterministic synthetic loader."""
    seeds = {"AAA.NS": 1, "BBB.NS": 2, "CCC.NS": 3}

    def fake_load(symbol, **kw):
        if symbol not in seeds:
            raise KeyError(f"No data for {symbol}")
        return _synth(seed=seeds[symbol])

    monkeypatch.setattr(strat_data, "load", fake_load)
    return seeds


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_one_row_per_combo(self, fake_loader):
        specs = [("bar_up_dn", {}), ("bollinger_bands", {"length": 20})]
        symbols = ["AAA.NS", "BBB.NS", "CCC.NS"]
        out = run_sweep(specs, symbols, max_intraday_loss_pct=20.0,
                        apply_costs=False)
        assert len(out) == len(specs) * len(symbols)
        assert set(out["symbol"]) == set(symbols)
        assert set(out["strategy"]) == {"bar_up_dn", "bollinger_bands"}

    def test_columns_present(self, fake_loader):
        specs = [("bar_up_dn", {})]
        out = run_sweep(specs, ["AAA.NS"], apply_costs=False)
        for c in ("symbol", "strategy", "params", "bars", "trades", "wins",
                  "win_rate_pct", "total_return_pct", "max_dd_pct",
                  "kill_fires", "ending_equity"):
            assert c in out.columns

    def test_unknown_strategy_raises(self, fake_loader):
        with pytest.raises(KeyError, match="Unknown strategy"):
            run_sweep([("not_a_strategy", {})], ["AAA.NS"])

    def test_missing_symbol_skipped_with_warning(self, fake_loader, caplog):
        out = run_sweep(
            [("bar_up_dn", {})], ["AAA.NS", "ZZZ.NS"],
            max_intraday_loss_pct=20.0, apply_costs=False,
        )
        # Only AAA produces a row; ZZZ is skipped.
        assert set(out["symbol"]) == {"AAA.NS"}

    def test_strategy_params_propagate(self, fake_loader):
        # Two BB configs with different `length` should both run.
        specs = [
            ("bollinger_bands", {"length": 10}),
            ("bollinger_bands", {"length": 20}),
        ]
        out = run_sweep(specs, ["AAA.NS"], max_intraday_loss_pct=20.0,
                        apply_costs=False)
        assert len(out) == 2
        assert set(out["params"]) == {"{'length': 10}", "{'length': 20}"}


# ---------------------------------------------------------------------------
# rollup_by_strategy
# ---------------------------------------------------------------------------

class TestRollup:
    def test_empty_input_returns_empty_frame(self):
        out = rollup_by_strategy(pd.DataFrame())
        assert out.empty
        for col in ("strategy", "symbols", "trades", "win_rate_pct",
                    "avg_return_pct", "avg_max_dd_pct", "kill_fires"):
            assert col in out.columns

    def test_one_row_per_strategy_params_combo(self, fake_loader):
        specs = [("bar_up_dn", {}), ("bollinger_bands", {})]
        results = run_sweep(specs, ["AAA.NS", "BBB.NS", "CCC.NS"],
                            max_intraday_loss_pct=20.0, apply_costs=False)
        rollup = rollup_by_strategy(results)
        assert len(rollup) == 2
        assert set(rollup["strategy"]) == {"bar_up_dn", "bollinger_bands"}
        # Each strategy aggregated 3 symbols.
        assert (rollup["symbols"] == 3).all()

    def test_win_rate_is_trade_count_weighted(self):
        # Strategy A: 100 trades 60% WR (60 wins), 10 trades 40% WR (4 wins)
        # → weighted = 64/110 ≈ 58.18%, not the simple mean of 50%.
        results = pd.DataFrame([
            {"symbol": "X", "strategy": "A", "params": "default", "bars": 1000,
             "trades": 100, "wins": 60, "win_rate_pct": 60.0,
             "total_return_pct": 1.0, "max_dd_pct": -1.0, "kill_fires": 0,
             "ending_equity": 101000},
            {"symbol": "Y", "strategy": "A", "params": "default", "bars": 1000,
             "trades": 10, "wins": 4, "win_rate_pct": 40.0,
             "total_return_pct": -0.5, "max_dd_pct": -0.8, "kill_fires": 0,
             "ending_equity": 99500},
        ])
        rollup = rollup_by_strategy(results)
        assert len(rollup) == 1
        assert rollup["trades"].iat[0] == 110
        assert rollup["win_rate_pct"].iat[0] == pytest.approx(64 / 110 * 100, abs=1e-6)
