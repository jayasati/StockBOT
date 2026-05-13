"""Phase-8 structure indicators + morning precompute tests.

Four classes:

  1. ``TestPivotFloor``     — hand-computed classic pivot levels
  2. ``TestCpr``            — pivot/BC/TC + narrow/wide classification
  3. ``TestFib``            — retracement + extension (LONG and SHORT)
  4. ``TestBundle``         — bundler behaviour with / without swing data
  5. ``TestPrecompute``     — pure helpers + DB roundtrip + symbol skip
  6. ``TestScoringIntegration`` — score_structure prefers daily_levels
                              over snapshot

All tests build their own daily DataFrames or pre-built bundles; no
network / no live cache.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date as _Date, datetime, timezone

import pandas as pd
import pytest

from indicators.structure import (
    NARROW_CPR_PCT,
    bundle_daily_levels,
    cpr,
    fib_extension,
    fib_retracement,
    pivot_points_floor,
)


# ===========================================================================
# 1. Pivot floor
# ===========================================================================

class TestPivotFloor:
    def test_classic_formula(self):
        """Hand-computed: H=110, L=100, C=105.
            pivot = (110+100+105)/3 = 105
            rng   = 10
            r1 = 2·105 − 100 = 110     s1 = 2·105 − 110 = 100
            r2 = 105 + 10   = 115     s2 = 105 − 10   = 95
            r3 = 110 + 2·(105−100) = 120
            s3 = 100 − 2·(110−105) = 90
        """
        out = pivot_points_floor(110, 100, 105)
        assert out["pivot"] == pytest.approx(105.0)
        assert out["r1"] == pytest.approx(110.0)
        assert out["r2"] == pytest.approx(115.0)
        assert out["r3"] == pytest.approx(120.0)
        assert out["s1"] == pytest.approx(100.0)
        assert out["s2"] == pytest.approx(95.0)
        assert out["s3"] == pytest.approx(90.0)

    def test_matches_legacy_levels_pivot_points_classic(self):
        """The Phase-8 scalar form must produce identical numbers to
        the existing ``indicators.levels.pivot_points`` classic mode,
        otherwise the precompute table and the per-tick snapshot
        path would disagree."""
        from indicators.levels import pivot_points as legacy
        a = pivot_points_floor(110, 100, 105)
        b = legacy({"pdh": 110, "pdl": 100, "pdc": 105}, method="classic")
        for k in ("pivot", "r1", "r2", "r3", "s1", "s2", "s3"):
            assert a[k] == pytest.approx(b[k])


# ===========================================================================
# 2. CPR
# ===========================================================================

class TestCpr:
    def test_basic_values(self):
        """H=110, L=100, C=105.
            pivot = 105
            bc    = (110+100)/2 = 105
            tc    = 2·105 − 105 = 105
            width = 0  (close at midpoint → CPR collapses to a point)
        """
        out = cpr(110, 100, 105)
        assert out["pivot"] == pytest.approx(105.0)
        assert out["bc"] == pytest.approx(105.0)
        assert out["tc"] == pytest.approx(105.0)
        assert out["width"] == pytest.approx(0.0)
        assert out["narrow"] is True  # 0% < 0.5%

    def test_wide_cpr(self):
        """Close near the high → asymmetric pivot, wider CPR."""
        out = cpr(110, 100, 109)
        # pivot = 319/3 ≈ 106.333
        # bc    = 105
        # tc    = 2·106.333 − 105 ≈ 107.667
        # width ≈ 2.667 → 2.5% → wide
        assert out["pivot"] == pytest.approx(106.3333, abs=1e-3)
        assert out["bc"] == pytest.approx(105.0)
        assert out["tc"] == pytest.approx(107.6667, abs=1e-3)
        assert out["width"] == pytest.approx(2.6667, abs=1e-3)
        assert out["narrow"] is False

    def test_narrow_threshold(self):
        """Width just below the 0.5% threshold → narrow=True."""
        # Construct: pivot ~100, width ~0.4% of pivot = 0.4
        # H=100.6, L=100.0, C=100.0 → pivot=66.86? Let me reset.
        # Easier: pivot=100, bc=99.8, tc=100.2 → width=0.4 (0.4% of 100)
        # We need H,L,C such that pivot = 100, bc = 99.8.
        # bc = (H+L)/2 = 99.8 ⇒ H+L = 199.6
        # pivot = (H+L+C)/3 = 100 ⇒ H+L+C = 300 ⇒ C = 100.4
        # Pick H=100.0, L=99.6 (so H+L=199.6).
        out = cpr(100.0, 99.6, 100.4)
        assert out["bc"] == pytest.approx(99.8)
        assert out["width_pct"] == pytest.approx(0.004, abs=1e-4)
        assert out["narrow"] is True

    def test_width_pct_uses_absolute_value(self):
        """When tc < bc (close near high), width_pct still ≥ 0."""
        out = cpr(110, 100, 108)
        assert out["width"] >= 0
        assert out["width_pct"] >= 0


# ===========================================================================
# 3. Fibonacci
# ===========================================================================

class TestFib:
    def test_retracement_endpoints(self):
        """Ratio 0.0 → swing_high; ratio 1.0 → swing_low."""
        levels = fib_retracement(100, 50)
        assert levels["0.000"] == pytest.approx(100.0)
        assert levels["1.000"] == pytest.approx(50.0)

    def test_retracement_golden(self):
        """The 0.618 retracement of 100→50 is at 100 − 0.618·50 = 69.1."""
        levels = fib_retracement(100, 50)
        assert levels["0.618"] == pytest.approx(69.1)

    def test_retracement_full_set(self):
        levels = fib_retracement(100, 50)
        # 0.236 ⇒ 100 − 11.8 = 88.2
        # 0.382 ⇒ 100 − 19.1 = 80.9
        # 0.5   ⇒ 75
        # 0.786 ⇒ 100 − 39.3 = 60.7
        assert levels["0.236"] == pytest.approx(88.2, abs=1e-3)
        assert levels["0.382"] == pytest.approx(80.9, abs=1e-3)
        assert levels["0.500"] == pytest.approx(75.0)
        assert levels["0.786"] == pytest.approx(60.7, abs=1e-3)

    def test_extension_up(self):
        """LONG continuation: extensions above swing_high.
        1.272 ⇒ 100 + 0.272·50 = 113.6
        1.618 ⇒ 100 + 0.618·50 = 130.9"""
        levels = fib_extension(100, 50, "up")
        assert levels["1.272"] == pytest.approx(113.6, abs=1e-3)
        assert levels["1.618"] == pytest.approx(130.9, abs=1e-3)
        assert levels["2.000"] == pytest.approx(150.0)
        assert levels["2.618"] == pytest.approx(180.9, abs=1e-3)

    def test_extension_down(self):
        """SHORT continuation: extensions below swing_low."""
        levels = fib_extension(100, 50, "down")
        # 1.272 ⇒ 50 − 0.272·50 = 36.4
        assert levels["1.272"] == pytest.approx(36.4, abs=1e-3)
        assert levels["2.000"] == pytest.approx(0.0)

    def test_extension_bad_direction_raises(self):
        with pytest.raises(ValueError):
            fib_extension(100, 50, "sideways")  # type: ignore[arg-type]


# ===========================================================================
# 4. Bundler
# ===========================================================================

class TestBundle:
    def test_bundle_without_swing_skips_fib(self):
        b = bundle_daily_levels(110, 100, 105)
        assert b["pdh"] == 110.0
        assert b["pdl"] == 100.0
        assert b["pdc"] == 105.0
        assert "pivot" in b
        assert "cpr" in b
        # No swing data → no fib block
        assert "fib_retracement" not in b
        assert "fib_extension_up" not in b

    def test_bundle_with_swing_includes_fib(self):
        b = bundle_daily_levels(
            110, 100, 105,
            swing_high=120, swing_low=80,
        )
        assert "fib_retracement" in b
        assert "fib_extension_up" in b
        assert "fib_extension_down" in b
        assert b["swing_high"] == 120
        assert b["swing_low"] == 80

    def test_bundle_swallows_degenerate_swing(self):
        """swing_high <= swing_low → fib block omitted, no exception."""
        b = bundle_daily_levels(
            110, 100, 105,
            swing_high=90, swing_low=95,
        )
        assert "fib_retracement" not in b

    def test_bundle_round_trips_through_json(self):
        """The bundle gets stored as a JSON blob in daily_levels —
        every field must survive a round trip."""
        b = bundle_daily_levels(
            110, 100, 105,
            swing_high=120, swing_low=80,
        )
        round_tripped = json.loads(json.dumps(b))
        assert round_tripped["pivot"]["pivot"] == pytest.approx(105.0)
        assert round_tripped["fib_retracement"]["0.618"] == pytest.approx(
            120 - 0.618 * 40, abs=1e-3,
        )


# ===========================================================================
# 5. Precompute
# ===========================================================================

def _build_daily_df(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a yfinance-shaped daily DataFrame."""
    return pd.DataFrame(rows).set_index("date")


class TestPrecompute:
    def test_prev_session_row_returns_strict_prior(self):
        from data.precompute import _prev_session_row
        df = _build_daily_df([
            {"date": _Date(2026, 5, 10), "High": 102.0,
             "Low": 98.0, "Close": 100.0},
            {"date": _Date(2026, 5, 11), "High": 105.0,
             "Low": 100.0, "Close": 104.0},
            {"date": _Date(2026, 5, 12), "High": 108.0,
             "Low": 103.0, "Close": 107.0},
        ])
        prev = _prev_session_row(df, _Date(2026, 5, 13))
        assert prev is not None
        assert float(prev["High"]) == 108.0

    def test_prev_session_row_empty_returns_none(self):
        from data.precompute import _prev_session_row
        assert _prev_session_row(pd.DataFrame(), _Date(2026, 5, 13)) is None

    def test_swing_uses_last_n_bars(self):
        from data.precompute import _swing_high_low
        df = _build_daily_df([
            {"date": _Date(2026, 5, d), "High": float(100 + d),
             "Low": float(95 + d), "Close": float(98 + d)}
            for d in range(1, 13)
        ])
        # Last 20 bars (only 12 available) → high = 112, low = 96
        hi, lo = _swing_high_low(df, _Date(2026, 5, 13))
        assert hi == 112.0
        assert lo == 96.0

    def test_compute_for_symbol_full_bundle(self):
        from data.precompute import compute_for_symbol
        df = _build_daily_df([
            {"date": _Date(2026, 5, d), "High": 100.0 + d,
             "Low": 95.0 + d, "Close": 98.0 + d}
            for d in range(1, 13)
        ])
        bundle = compute_for_symbol("TEST.NS", df, _Date(2026, 5, 13))
        assert bundle is not None
        # Prior day = May 12 → H=112, L=107, C=110
        assert bundle["pdh"] == pytest.approx(112.0)
        assert bundle["pdl"] == pytest.approx(107.0)
        assert bundle["pdc"] == pytest.approx(110.0)
        assert "pivot" in bundle
        assert "cpr" in bundle
        assert "fib_retracement" in bundle

    def test_compute_for_symbol_returns_none_on_empty(self):
        from data.precompute import compute_for_symbol
        assert compute_for_symbol(
            "TEST.NS", pd.DataFrame(), _Date(2026, 5, 13),
        ) is None

    def test_upsert_and_get_roundtrip(self, tmp_path, monkeypatch):
        """Write + read via the public helpers."""
        from data import precompute
        db_path = str(tmp_path / "alerts.db")
        # Initialise schema.
        from bot import db as bot_db
        from bot import config as bot_config
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(precompute, "DB_PATH", db_path)
        bot_db.init_db()

        bundle = {"pdh": 110.0, "pdl": 100.0, "pdc": 105.0}
        precompute._upsert_levels(
            "TEST.NS", _Date(2026, 5, 13), bundle, db_path=db_path,
        )
        got = precompute.get_daily_levels(
            "TEST.NS", _Date(2026, 5, 13), db_path=db_path,
        )
        assert got == bundle

    def test_upsert_overwrites_on_conflict(self, tmp_path, monkeypatch):
        from data import precompute
        db_path = str(tmp_path / "alerts.db")
        from bot import db as bot_db
        from bot import config as bot_config
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(precompute, "DB_PATH", db_path)
        bot_db.init_db()

        precompute._upsert_levels(
            "TEST.NS", _Date(2026, 5, 13), {"v": 1}, db_path=db_path,
        )
        precompute._upsert_levels(
            "TEST.NS", _Date(2026, 5, 13), {"v": 2}, db_path=db_path,
        )
        got = precompute.get_daily_levels(
            "TEST.NS", _Date(2026, 5, 13), db_path=db_path,
        )
        assert got == {"v": 2}

    def test_get_levels_missing_returns_none(self, tmp_path, monkeypatch):
        from data import precompute
        from bot import db as bot_db, config as bot_config
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(precompute, "DB_PATH", db_path)
        bot_db.init_db()
        assert precompute.get_daily_levels(
            "MISSING.NS", _Date(2026, 5, 13), db_path=db_path,
        ) is None

class TestPrecomputeSync:
    """Sync wrapper around run_morning_compute for environments
    without pytest-asyncio. Uses asyncio.run directly."""

    def test_run_morning_mixed_watchlist(self, tmp_path, monkeypatch):
        import asyncio
        from data import precompute
        from bot import db as bot_db, config as bot_config

        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(precompute, "DB_PATH", db_path)
        bot_db.init_db()

        good_df = _build_daily_df([
            {"date": _Date(2026, 5, d), "High": 100.0 + d,
             "Low": 95.0 + d, "Close": 98.0 + d}
            for d in range(1, 13)
        ])
        cache = {
            "OK.NS": good_df,
            "EMPTY.NS": pd.DataFrame(),
        }

        summary = asyncio.run(precompute.run_morning_compute(
            watchlist=["OK.NS", "EMPTY.NS", "ABSENT.NS"],
            daily_cache=cache,
            session_date=_Date(2026, 5, 13),
            db_path=db_path,
        ))
        assert summary["ok"] == 1
        assert set(summary["missing"]) == {"EMPTY.NS", "ABSENT.NS"}
        assert summary["errors"] == []
        # And the OK row landed.
        got = precompute.get_daily_levels(
            "OK.NS", _Date(2026, 5, 13), db_path=db_path,
        )
        assert got is not None
        assert "pivot" in got


# ===========================================================================
# 6. Scoring integration
# ===========================================================================

class TestScoringIntegration:
    def _signal(self, **kw):
        from bot.scoring import StockSignals
        defaults = dict(
            symbol="TEST.NS", price=105.0, rsi=60.0, volume_ratio=2.0,
            above_vwap=True, breakout=False, pct_from_high=-0.5, score=70,
        )
        defaults.update(kw)
        return StockSignals(**defaults)

    def test_score_structure_uses_daily_levels_pivot(self):
        from scoring.components import score_structure
        from scoring.config_loader import DEFAULTS
        # Price = 100 sits ON the pivot.
        bundle = {
            "pdh": 110.0, "pdl": 100.0, "pdc": 105.0,
            "pivot": {"pivot": 100.0, "r1": 110.0, "s1": 95.0,
                      "r2": 115.0, "s2": 90.0},
            "cpr": {"pivot": 100.0, "bc": 99.5, "tc": 100.5,
                    "width_pct": 0.01},
        }
        score = score_structure(
            None, price=100.0, side="LONG",
            weights=DEFAULTS["component_weights"]["structure"],
            daily_levels=bundle,
        )
        # On pivot → near_pivot=100, sr (PDH 10% away) → 50; orb → 30
        # Weighted sums to a healthy non-zero score.
        assert score > 50

    def test_score_structure_falls_back_to_snapshot(self):
        """When daily_levels is None, the snapshot.values path is
        still consulted — backward compat for pre-precompute paths."""
        from datetime import date as _D
        from scoring.components import score_structure
        from scoring.config_loader import DEFAULTS
        from indicators.compute import IndicatorSnapshot
        snap = IndicatorSnapshot(
            symbol="X", session_date=_D(2026, 5, 13),
            computed_at=datetime.now(timezone.utc),
            values={"pivot_classic": 100.0, "pdh": 105.0, "pdl": 95.0},
        )
        score = score_structure(
            snap, price=100.0, side="LONG",
            weights=DEFAULTS["component_weights"]["structure"],
            daily_levels=None,
        )
        # Snapshot pivots feed the path → non-zero score.
        assert score > 0

    def test_master_score_signal_accepts_daily_levels(self):
        """End-to-end: master.score_signal threads daily_levels into
        score_structure and the breakdown reflects it."""
        from scoring import score_signal
        from scoring.config_loader import DEFAULTS
        bundle = {
            "pdh": 110.0, "pdl": 100.0, "pdc": 105.0,
            "pivot": {"pivot": 105.0, "r1": 110.0, "s1": 100.0,
                      "r2": 115.0, "s2": 95.0},
            "cpr": {"pivot": 105.0, "bc": 104.5, "tc": 105.5,
                    "width_pct": 0.01},
        }
        s = self._signal(price=105.0)
        br_with = score_signal(s, config=DEFAULTS, daily_levels=bundle)
        br_without = score_signal(s, config=DEFAULTS, daily_levels=None)
        # With the bundle the structure component should be different
        # from the snapshot-fallback path (the signal has no snapshot,
        # so without daily_levels structure collapses to NEUTRAL).
        assert br_with.components["structure"] != br_without.components["structure"]
