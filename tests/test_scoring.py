"""Phase-7 weighted scoring engine tests.

Five classes:

  1. ``TestConfigLoader``  — yaml load, default-merge, mtime cache
  2. ``TestComponents``    — per-category bullish/bearish polarity
  3. ``TestMaster``        — score_signal end-to-end shape
  4. ``TestAcceptance``    — the three acceptance tests in the prompt
                             (bullish-LONG > 85, ADX-counter-trend < 50
                             post-multiplier, yaml hot-reload)
  5. ``TestAuditWiring``   — filter_audit gets components_json +
                             final_score columns and score_breakdown
                             flows through write_audit
"""
from __future__ import annotations

from datetime import date as _Date, datetime, timezone

import json
import pytest

from bot.scoring import StockSignals
import scoring as scoring_engine
from scoring import components as comp
from scoring import config_loader, master


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_scoring_cache():
    """Every test starts with an empty config cache so an mtime change
    in one test does not poison the next."""
    config_loader.clear_cache()
    yield
    config_loader.clear_cache()


def _snapshot(**values):
    """Build an IndicatorSnapshot with the requested keys populated."""
    from indicators.compute import IndicatorSnapshot
    return IndicatorSnapshot(
        symbol="TEST.NS",
        session_date=_Date(2026, 5, 12),
        computed_at=datetime.now(timezone.utc),
        values=dict(values),
    )


def _signal(
    *,
    side: str = "LONG",
    score: int = 70,
    price: float = 100.0,
    above_vwap: bool = True,
    volume_ratio: float = 2.0,
    snapshot=None,
    market_context=None,
    soft_adjustments=None,
    filing_title=None,
) -> StockSignals:
    s = StockSignals(
        symbol="TEST.NS",
        price=price,
        rsi=65.0,
        volume_ratio=volume_ratio,
        above_vwap=above_vwap,
        breakout=False,
        pct_from_high=-0.5,
        score=score,
        side=side,
    )
    s.snapshot = snapshot
    if market_context is not None:
        s.market_context = market_context
    if soft_adjustments is not None:
        s.soft_adjustments = list(soft_adjustments)
    s.filing_title = filing_title
    return s


def _bullish_long_snapshot():
    """Every indicator the scorer reads, set to its maximally-bullish
    value for a LONG signal."""
    return _snapshot(
        rsi_5m=65.0,
        macd_5m_histogram=0.5,
        stochastic_5m_k=60.0,
        stochastic_5m_d=55.0,
        cci_5m=120.0,
        mfi_5m=65.0,
        adx_5m=35.0,
        adx_5m_di_plus=30.0,
        adx_5m_di_minus=15.0,
        supertrend_5m_direction=1.0,
        atr_5m=2.0,
        bollinger_5m_percent_b=0.65,
        cmf_5m=0.10,
        volume_surge_ratio_5m=2.5,
        ttm_squeeze_5m_in_squeeze=0.0,
        ttm_squeeze_5m_momentum=0.3,
        pdh=101.0,
        pdl=99.0,
        pivot_classic=100.0,
        pivot_classic_r1=101.0,
        pivot_classic_s1=99.0,
        orh_15=100.5,
        orl_15=99.5,
    )


# ===========================================================================
# 1. Config loader
# ===========================================================================

class TestConfigLoader:
    def test_defaults_when_file_missing(self, tmp_path):
        cfg = config_loader.load_config(tmp_path / "absent.yaml")
        assert cfg["alert_threshold"] == 85
        assert cfg["weights"]["trend"] == pytest.approx(0.25)
        assert "vwap_position" in cfg["component_weights"]["trend"]

    def test_user_yaml_overrides_defaults(self, tmp_path):
        p = tmp_path / "scoring.yaml"
        p.write_text(
            "weights:\n"
            "  trend: 0.5\n"
            "alert_threshold: 70\n"
        )
        cfg = config_loader.load_config(p)
        assert cfg["weights"]["trend"] == pytest.approx(0.5)
        # Untouched keys fall back to DEFAULTS
        assert cfg["weights"]["momentum"] == pytest.approx(0.20)
        assert cfg["alert_threshold"] == 70

    def test_mtime_cache_picks_up_edit(self, tmp_path):
        import os
        import time as _time

        p = tmp_path / "scoring.yaml"
        p.write_text("weights:\n  momentum: 0.20\n")
        cfg1 = config_loader.load_config(p)
        assert cfg1["weights"]["momentum"] == pytest.approx(0.20)

        # Bump mtime by a full second so stat().st_mtime visibly
        # advances on filesystems with 1s resolution.
        _time.sleep(0.01)
        p.write_text("weights:\n  momentum: 0.30\n")
        new_mtime = p.stat().st_mtime + 1.0
        os.utime(p, (new_mtime, new_mtime))

        cfg2 = config_loader.load_config(p)
        assert cfg2["weights"]["momentum"] == pytest.approx(0.30)

    def test_get_alert_threshold(self, tmp_path):
        p = tmp_path / "scoring.yaml"
        p.write_text("alert_threshold: 70\n")
        assert config_loader.get_alert_threshold(path=p) == 70.0

    def test_get_alert_threshold_default(self, tmp_path):
        # File exists but doesn't set alert_threshold → DEFAULTS apply
        p = tmp_path / "scoring.yaml"
        p.write_text("weights:\n  trend: 0.5\n")
        assert config_loader.get_alert_threshold(path=p) == 85.0


# ===========================================================================
# 2. Components
# ===========================================================================

class TestComponents:
    def test_trend_long_bullish(self):
        weights = config_loader.DEFAULTS["component_weights"]["trend"]
        snap = _bullish_long_snapshot()
        score = comp.score_trend(
            snap, {}, "LONG", weights=weights,
            price=100.5, above_vwap=True,
        )
        assert score > 80

    def test_trend_long_bearish_snapshot(self):
        weights = config_loader.DEFAULTS["component_weights"]["trend"]
        snap = _snapshot(
            macd_5m_histogram=-0.5,
            supertrend_5m_direction=-1.0,
            adx_5m=5.0,
            pivot_classic=110.0,  # price way below pivot
        )
        score = comp.score_trend(
            snap, {}, "LONG", weights=weights,
            price=100.0, above_vwap=False,
        )
        assert score < 30

    def test_momentum_long_in_zone(self):
        weights = config_loader.DEFAULTS["component_weights"]["momentum"]
        snap = _snapshot(
            rsi_5m=65.0, macd_5m_histogram=0.5,
            stochastic_5m_k=60.0, cci_5m=120.0, mfi_5m=60.0,
        )
        score = comp.score_momentum(snap, "LONG", weights=weights)
        assert score > 80

    def test_momentum_polarity_flips_for_short(self):
        weights = config_loader.DEFAULTS["component_weights"]["momentum"]
        # The exact bearish mirror of the LONG-bullish snapshot
        snap = _snapshot(
            rsi_5m=35.0, macd_5m_histogram=-0.5,
            stochastic_5m_k=40.0, cci_5m=-120.0, mfi_5m=40.0,
        )
        long_score = comp.score_momentum(snap, "LONG", weights=weights)
        short_score = comp.score_momentum(snap, "SHORT", weights=weights)
        # Same snapshot: bullish for SHORT, bearish for LONG
        assert short_score > long_score

    def test_volume_high_volume_score(self):
        weights = config_loader.DEFAULTS["component_weights"]["volume"]
        snap = _snapshot(
            cmf_5m=0.10, volume_surge_ratio_5m=2.5,
        )
        score = comp.score_volume(
            snap, volume_ratio=3.0, above_vwap=True,
            side="LONG", weights=weights,
        )
        assert score > 80

    def test_market_aligned_with_long(self):
        weights = config_loader.DEFAULTS["component_weights"]["market"]
        score = comp.score_market(
            {"nifty_pct": 0.7, "bank_nifty_pct": 0.6, "vix": 13.0},
            "LONG", weights=weights,
        )
        assert score > 90

    def test_market_against_long(self):
        weights = config_loader.DEFAULTS["component_weights"]["market"]
        score = comp.score_market(
            {"nifty_pct": -0.7, "bank_nifty_pct": -0.6, "vix": 22.0},
            "LONG", weights=weights,
        )
        assert score < 25

    def test_missing_snapshot_neutral(self):
        """All-None component → neutral 50 (so unscored stocks don't
        get auto-killed for lack of indicator coverage)."""
        weights = config_loader.DEFAULTS["component_weights"]["trend"]
        score = comp.score_trend(
            None, {}, "LONG", weights=weights,
            price=None, above_vwap=None,
        )
        assert score == pytest.approx(50.0)

    def test_news_with_filing(self):
        weights = config_loader.DEFAULTS["component_weights"]["news"]
        with_filing = comp.score_news("Dividend declared", "LONG", weights=weights)
        without = comp.score_news(None, "LONG", weights=weights)
        assert with_filing > without


# ===========================================================================
# 3. Master
# ===========================================================================

class TestMaster:
    def test_breakdown_has_seven_components(self):
        s = _signal(snapshot=_bullish_long_snapshot())
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert set(br.components) == {
            "trend", "momentum", "volume", "volatility",
            "structure", "market", "news",
        }

    def test_base_is_sum_of_weighted(self):
        s = _signal(snapshot=_bullish_long_snapshot())
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert br.base == pytest.approx(sum(br.weighted.values()))

    def test_multiplier_product_applied(self):
        s = _signal(
            snapshot=_bullish_long_snapshot(),
            soft_adjustments=[("adx_counter_trend", 0.4), ("vix_panic", 0.6)],
        )
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert br.multiplier_product == pytest.approx(0.4 * 0.6)
        assert br.final == pytest.approx(br.base * 0.24)

    def test_no_soft_adjustments_means_product_one(self):
        s = _signal(snapshot=_bullish_long_snapshot())
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert br.multiplier_product == 1.0
        assert br.final == pytest.approx(br.base)

    def test_alert_threshold_captured(self):
        cfg = dict(config_loader.DEFAULTS)
        cfg["alert_threshold"] = 70
        s = _signal(snapshot=_bullish_long_snapshot())
        br = master.score_signal(s, config=cfg)
        assert br.alert_threshold == 70.0


# ===========================================================================
# 4. Acceptance tests — directly mirror the prompt
# ===========================================================================

class TestAcceptance:
    def test_bullish_long_scores_over_85(self):
        """Synthetic signal with all indicators flipped bullish on LONG
        and supportive market context → final > 85 (no multipliers)."""
        s = _signal(
            snapshot=_bullish_long_snapshot(),
            market_context={
                "nifty_pct": 0.8, "bank_nifty_pct": 0.7, "vix": 13.0,
            },
            filing_title="Order win announcement",
            soft_adjustments=[],
        )
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert br.final > 85, (
            f"expected final > 85, got {br.final:.2f}\n"
            f"components={br.components}"
        )

    def test_adx_counter_trend_drops_below_50(self):
        """The same bullish snapshot, but the chain attached the
        Meesho-rule multiplier (×0.4) → final drops below 50."""
        s = _signal(
            snapshot=_bullish_long_snapshot(),
            market_context={
                "nifty_pct": 0.8, "bank_nifty_pct": 0.7, "vix": 13.0,
            },
            filing_title="Order win announcement",
            soft_adjustments=[("adx_counter_trend", 0.4)],
        )
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        assert br.final < 50, (
            f"expected final < 50 after 0.4x demote, got {br.final:.2f}"
        )

    def test_yaml_weights_hot_reload(self, tmp_path):
        """Editing scoring.yaml between scans changes the next
        score_signal output without a restart."""
        import os
        import time as _time

        p = tmp_path / "scoring.yaml"
        p.write_text(
            "weights:\n"
            "  trend: 0.50\n"
            "  momentum: 0.20\n"
            "  volume: 0.15\n"
            "  volatility: 0.05\n"
            "  structure: 0.05\n"
            "  market: 0.04\n"
            "  news: 0.01\n"
            "alert_threshold: 60\n"
        )
        s = _signal(snapshot=_bullish_long_snapshot())
        cfg1 = config_loader.load_config(p)
        br1 = master.score_signal(s, config=cfg1)

        # Bump mtime past 1s resolution.
        _time.sleep(0.01)
        p.write_text(
            "weights:\n"
            "  trend: 0.10\n"
            "  momentum: 0.10\n"
            "  volume: 0.10\n"
            "  volatility: 0.10\n"
            "  structure: 0.10\n"
            "  market: 0.10\n"
            "  news: 0.40\n"   # news dominates now
            "alert_threshold: 60\n"
        )
        new_mtime = p.stat().st_mtime + 1.0
        os.utime(p, (new_mtime, new_mtime))

        cfg2 = config_loader.load_config(p)
        br2 = master.score_signal(s, config=cfg2)

        # The component scores are identical (same snapshot), but the
        # weighting shifted — that should change the base aggregate.
        assert br1.base != pytest.approx(br2.base, abs=0.1), (
            f"weights hot-reload had no effect: {br1.base:.2f} vs {br2.base:.2f}"
        )


# ===========================================================================
# 5. Audit wiring
# ===========================================================================

class TestAuditWiring:
    @pytest.fixture
    def tmp_db(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "alerts.db")
        import bot.config
        import bot.db
        import bot.storage
        import bot.suppression.rules
        monkeypatch.setattr(bot.config, "DB_PATH", db_path)
        monkeypatch.setattr(bot.storage, "DB_PATH", db_path)
        monkeypatch.setattr(bot.db, "DB_PATH", db_path)
        monkeypatch.setattr(bot.suppression.rules, "DB_PATH", db_path)
        bot.db.init_db()
        return db_path

    def test_filter_audit_has_phase7_columns(self, tmp_db):
        import sqlite3
        with sqlite3.connect(tmp_db) as conn:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(filter_audit)"
            )}
        assert "components_json" in cols
        assert "final_score" in cols

    def test_write_audit_persists_breakdown(self, tmp_db):
        import sqlite3
        from filters.chain import write_audit

        s = _signal(snapshot=_bullish_long_snapshot())
        br = master.score_signal(s, config=config_loader.DEFAULTS)
        s.confidence = br.final / 100.0
        s.score_breakdown = br.as_dict()
        write_audit(s, alerted=True)

        with sqlite3.connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT components_json, final_score "
                "FROM filter_audit ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        components_json, final_score = row
        parsed = json.loads(components_json)
        assert set(parsed) == {
            "trend", "momentum", "volume", "volatility",
            "structure", "market", "news",
        }
        assert final_score == pytest.approx(br.final)

    def test_migration_adds_columns_to_legacy_db(self, tmp_path, monkeypatch):
        """A pre-Phase-7 database (filter_audit without the new
        columns) is upgraded in place by init_db without losing
        the old rows."""
        import sqlite3
        db_path = str(tmp_path / "legacy.db")
        # Create the OLD shape directly.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE filter_audit ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  ts TEXT NOT NULL, symbol TEXT NOT NULL,"
                "  side TEXT, score INTEGER NOT NULL,"
                "  kill_reasons TEXT, soft_adjustments_json TEXT,"
                "  final_confidence REAL NOT NULL, alerted INTEGER NOT NULL"
                ")"
            )
            conn.execute(
                "INSERT INTO filter_audit "
                "(ts, symbol, side, score, kill_reasons,"
                " soft_adjustments_json, final_confidence, alerted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("2026-05-12T11:00:00", "X.NS", "LONG", 70,
                 None, "[]", 0.7, 1),
            )

        import bot.config, bot.db, bot.storage, bot.suppression.rules
        monkeypatch.setattr(bot.config, "DB_PATH", db_path)
        monkeypatch.setattr(bot.storage, "DB_PATH", db_path)
        monkeypatch.setattr(bot.db, "DB_PATH", db_path)
        monkeypatch.setattr(bot.suppression.rules, "DB_PATH", db_path)

        bot.db.init_db()

        with sqlite3.connect(db_path) as conn:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(filter_audit)"
            )}
            count = conn.execute(
                "SELECT COUNT(*) FROM filter_audit"
            ).fetchone()[0]
        assert "components_json" in cols
        assert "final_score" in cols
        assert count == 1  # legacy row preserved
