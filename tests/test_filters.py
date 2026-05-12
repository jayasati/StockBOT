"""Phase-6 filter chain tests.

Tests are organised by module-under-test:

  1. StockSignals extension + side inference (this file's first class)
  2. Hard filters     — ``filters/hard.py``
  3. Soft filters     — ``filters/soft.py``
  4. Time filters     — ``filters/time.py``
  5. Event filters    — ``filters/event.py``
  6. Chain orchestration — ``filters/chain.py``
  7. Scanner integration — ``_evaluate_symbol`` with the chain wired in

All tests use ``tmp_path``-backed databases and stub IndicatorSnapshot
fixtures so the bot's real alerts.db is never touched."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Per-test alerts.db with the full schema (alerts_sent + paper_trades
    + filter_audit + everything init_db creates)."""
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


# ---------------------------------------------------------------------------
# StockSignals extension + side inference
# ---------------------------------------------------------------------------

class TestStockSignalsExtension:
    """Phase-6 fields default-initialised so every pre-Phase-6 call
    site keeps working byte-identical."""

    def test_defaults_set(self):
        from bot.scoring import StockSignals
        s = StockSignals(
            symbol="X.NS", price=100.0, rsi=50.0, volume_ratio=1.0,
            above_vwap=True, breakout=False, pct_from_high=-1.0, score=0,
        )
        assert s.side == "LONG"
        assert s.kill_reasons == []
        assert s.soft_adjustments == []
        assert s.confidence == 0.0
        assert s.market_context == {}

    def test_side_long_on_bullish_rsi(self):
        """RSI in mid-band → side='LONG' regardless of VWAP position."""
        from bot.scoring import score_stock
        bars = _bullish_bars()
        sig = score_stock("X.NS", bars["intraday"], bars["daily"])
        assert sig.side == "LONG"

    def test_side_short_only_when_both_bearish(self):
        """RSI<35 AND below VWAP → SHORT. Either alone stays LONG."""
        from bot.scoring import score_stock
        bars = _bearish_bars()                # RSI low + below VWAP
        sig = score_stock("X.NS", bars["intraday"], bars["daily"])
        assert sig.side == "SHORT"

    def test_side_long_when_only_rsi_bearish(self):
        """RSI low but above VWAP → LONG (mixed signals, default direction)."""
        from bot.scoring import score_stock
        bars = _mixed_low_rsi_above_vwap_bars()
        sig = score_stock("X.NS", bars["intraday"], bars["daily"])
        assert sig.side == "LONG"


# ---------------------------------------------------------------------------
# Helpers — synthetic OHLCV builders for scoring-shape tests
# ---------------------------------------------------------------------------

def _bullish_bars():
    """Build intraday + daily DataFrames that score as a bullish setup."""
    idx = pd.date_range("2026-05-12 09:15", periods=30, freq="5min", tz=IST)
    # Rising closes → high RSI, above VWAP, volume normal.
    closes = [100 + i * 0.5 for i in range(30)]
    intraday = pd.DataFrame({
        "Open": closes, "High": [c + 0.3 for c in closes],
        "Low": [c - 0.3 for c in closes], "Close": closes,
        "Volume": [1000.0] * 30,
    }, index=idx)
    daily = pd.DataFrame({
        "Open": [98] * 10, "High": [102] * 10, "Low": [97] * 10,
        "Close": [100] * 10, "Volume": [50000] * 10,
    }, index=pd.date_range("2026-05-01", periods=10, freq="D"))
    return {"intraday": intraday, "daily": daily}


def _bearish_bars():
    """Falling closes + descending → RSI<35, below VWAP."""
    idx = pd.date_range("2026-05-12 09:15", periods=30, freq="5min", tz=IST)
    closes = [100 - i * 0.8 for i in range(30)]
    intraday = pd.DataFrame({
        "Open": closes, "High": [c + 0.2 for c in closes],
        "Low": [c - 0.5 for c in closes], "Close": closes,
        "Volume": [1000.0] * 30,
    }, index=idx)
    daily = pd.DataFrame({
        "Open": [98] * 10, "High": [102] * 10, "Low": [97] * 10,
        "Close": [100] * 10, "Volume": [50000] * 10,
    }, index=pd.date_range("2026-05-01", periods=10, freq="D"))
    return {"intraday": intraday, "daily": daily}


def _mixed_low_rsi_above_vwap_bars():
    """Recently-pulled-back from highs but still above session VWAP.
    RSI < 35 but VWAP situation favours LONG → side stays LONG."""
    idx = pd.date_range("2026-05-12 09:15", periods=30, freq="5min", tz=IST)
    # Up first, then sharp pullback in the last few bars to drop RSI
    # without sinking below VWAP (VWAP catches up slower).
    closes = (
        [100 + i * 0.5 for i in range(25)] +   # rise to ~112
        [112 - i * 1.0 for i in range(1, 6)]   # drop 5 points
    )
    intraday = pd.DataFrame({
        "Open": closes, "High": [c + 0.4 for c in closes],
        "Low": [c - 0.4 for c in closes], "Close": closes,
        "Volume": [1000.0] * 30,
    }, index=idx)
    daily = pd.DataFrame({
        "Open": [98] * 10, "High": [102] * 10, "Low": [97] * 10,
        "Close": [100] * 10, "Volume": [50000] * 10,
    }, index=pd.date_range("2026-05-01", periods=10, freq="D"))
    return {"intraday": intraday, "daily": daily}


def _stub_signals(score: int = 70, side: str = "LONG", **kw):
    """Bare-minimum StockSignals for filter tests — only fields any
    filter might read are set; the rest take their dataclass defaults."""
    from bot.scoring import StockSignals
    return StockSignals(
        symbol=kw.get("symbol", "TEST.NS"),
        price=kw.get("price", 100.0),
        rsi=kw.get("rsi", 50.0),
        volume_ratio=kw.get("volume_ratio", 1.0),
        above_vwap=kw.get("above_vwap", True),
        breakout=kw.get("breakout", False),
        pct_from_high=kw.get("pct_from_high", -1.0),
        score=score,
        side=side,
    )


def _stub_ctx(now_str: str = "2026-05-12T11:00:00+05:30", **kw):
    """Bare-minimum FilterContext for tests. ``now_str`` is parsed to
    a tz-aware IST datetime."""
    from filters import FilterContext
    now = pd.Timestamp(now_str).to_pydatetime()
    return FilterContext(now=now, **kw)


# ---------------------------------------------------------------------------
# Time filters (filters/time.py)
# ---------------------------------------------------------------------------

class TestTimeFilters:
    def test_opening_window_penalty_inside(self):
        from filters.time import opening_window_penalty
        ctx = _stub_ctx("2026-05-12T09:20:00+05:30")
        result = opening_window_penalty(_stub_signals(), ctx)
        assert result == ("opening_window", 0.3)

    def test_opening_window_penalty_outside(self):
        from filters.time import opening_window_penalty
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert opening_window_penalty(_stub_signals(), ctx) is None

    def test_lunch_window_penalty_inside(self):
        from filters.time import lunch_window_penalty
        ctx = _stub_ctx("2026-05-12T13:00:00+05:30")
        result = lunch_window_penalty(_stub_signals(), ctx)
        assert result == ("lunch_window", 0.7)

    def test_lunch_window_penalty_at_end_excluded(self):
        """13:30:00 sharp falls OUTSIDE the lunch window — boundary is
        half-open so the immediate next bar (13:30 close) isn't penalised."""
        from filters.time import lunch_window_penalty
        ctx = _stub_ctx("2026-05-12T13:30:00+05:30")
        assert lunch_window_penalty(_stub_signals(), ctx) is None

    def test_end_of_day_penalty_inside(self):
        from filters.time import end_of_day_penalty
        ctx = _stub_ctx("2026-05-12T15:00:00+05:30")
        result = end_of_day_penalty(_stub_signals(), ctx)
        assert result == ("end_of_day", 0.6)

    def test_end_of_day_penalty_outside(self):
        from filters.time import end_of_day_penalty
        ctx = _stub_ctx("2026-05-12T14:30:00+05:30")
        assert end_of_day_penalty(_stub_signals(), ctx) is None

    def test_midmorning_no_penalty(self):
        from filters.time import (
            end_of_day_penalty, lunch_window_penalty, opening_window_penalty,
        )
        ctx = _stub_ctx("2026-05-12T10:30:00+05:30")
        s = _stub_signals()
        assert opening_window_penalty(s, ctx) is None
        assert lunch_window_penalty(s, ctx) is None
        assert end_of_day_penalty(s, ctx) is None


# ---------------------------------------------------------------------------
# Event filters (filters/event.py)
# ---------------------------------------------------------------------------

class TestEventFilters:
    @pytest.fixture
    def yaml_path(self, tmp_path, monkeypatch):
        """Per-test events.yaml; pointed at via monkeypatch."""
        path = tmp_path / "events.yaml"
        import filters.event as ev
        monkeypatch.setattr(ev, "EVENTS_YAML_PATH", path)
        ev._reset_cache()
        return path

    def test_rbi_policy_day_fires(self, yaml_path):
        yaml_path.write_text(
            "rbi_policy_days:\n  - 2026-04-08\n"
            "budget_days: []\n"
            "election_results_days: []\n"
        )
        from filters.event import rbi_policy_day
        ctx = _stub_ctx("2026-04-08T11:00:00+05:30")
        assert rbi_policy_day(_stub_signals(), ctx) == ("rbi_policy_day", 0.2)

    def test_rbi_policy_day_skips_other_dates(self, yaml_path):
        yaml_path.write_text("rbi_policy_days:\n  - 2026-04-08\n")
        from filters.event import rbi_policy_day
        ctx = _stub_ctx("2026-04-09T11:00:00+05:30")
        assert rbi_policy_day(_stub_signals(), ctx) is None

    def test_budget_day_fires(self, yaml_path):
        yaml_path.write_text("budget_days:\n  - 2026-02-01\n")
        from filters.event import budget_day
        ctx = _stub_ctx("2026-02-01T11:00:00+05:30")
        assert budget_day(_stub_signals(), ctx) == ("budget_day", 0.1)

    def test_election_day_fires(self, yaml_path):
        yaml_path.write_text("election_results_days:\n  - 2026-06-04\n")
        from filters.event import election_results_day
        ctx = _stub_ctx("2026-06-04T11:00:00+05:30")
        assert election_results_day(_stub_signals(), ctx) == (
            "election_results_day", 0.1,
        )

    def test_missing_yaml_is_inert(self, tmp_path, monkeypatch):
        """No events.yaml → every event filter returns None.
        Bot must NOT crash for an unfilled config."""
        import filters.event as ev
        monkeypatch.setattr(ev, "EVENTS_YAML_PATH", tmp_path / "absent.yaml")
        ev._reset_cache()
        ctx = _stub_ctx("2026-04-08T11:00:00+05:30")
        assert ev.rbi_policy_day(_stub_signals(), ctx) is None
        assert ev.budget_day(_stub_signals(), ctx) is None
        assert ev.election_results_day(_stub_signals(), ctx) is None

    def test_malformed_date_skipped(self, yaml_path):
        """Bad entries don't poison the rest of the calendar."""
        yaml_path.write_text(
            "rbi_policy_days:\n"
            "  - not-a-date\n"
            "  - 2026-04-08\n"
        )
        from filters.event import rbi_policy_day
        # Good date still fires.
        ctx = _stub_ctx("2026-04-08T11:00:00+05:30")
        assert rbi_policy_day(_stub_signals(), ctx) == ("rbi_policy_day", 0.2)


# ---------------------------------------------------------------------------
# Chain orchestration (filters/chain.py)
# ---------------------------------------------------------------------------

class TestFilterChain:
    def test_no_filters_registers_and_passes(self, tmp_db, monkeypatch):
        """With every registry empty, signals pass through unchanged
        and confidence = score/100."""
        import filters.chain as ch
        monkeypatch.setattr(ch, "HARD_FILTERS", ())
        monkeypatch.setattr(ch, "SOFT_FILTERS", ())
        monkeypatch.setattr(ch, "TIME_FILTERS", ())
        monkeypatch.setattr(ch, "EVENT_FILTERS", ())

        s = _stub_signals(score=70)
        ctx = _stub_ctx()
        result = ch.apply_filters(s, ctx)
        assert result is s
        assert result.confidence == pytest.approx(0.70)

    def test_hard_kill_short_circuits(self, tmp_db, monkeypatch):
        """First hard filter to return a kill reason short-circuits
        the chain. Downstream filters are NOT called."""
        import filters.chain as ch

        downstream_called = []

        def kill(_, __):
            return "test_kill"

        def downstream(_, __):
            downstream_called.append(True)
            return ("downstream", 0.5)

        monkeypatch.setattr(ch, "HARD_FILTERS", (kill,))
        monkeypatch.setattr(ch, "SOFT_FILTERS", (downstream,))
        monkeypatch.setattr(ch, "TIME_FILTERS", ())
        monkeypatch.setattr(ch, "EVENT_FILTERS", ())

        s = _stub_signals(score=70)
        result = ch.apply_filters(s, _stub_ctx())
        assert result is None
        assert s.kill_reasons == ["test_kill"]
        assert downstream_called == []

    def test_multiplier_product_composes(self, tmp_db, monkeypatch):
        """Three multipliers from three filters compose
        multiplicatively. score=70, mults 0.4 * 0.7 * 1.1 → 0.2156."""
        import filters.chain as ch

        def soft_a(_, __):
            return ("soft_a", 0.4)

        def soft_b(_, __):
            return ("soft_b", 0.7)

        def soft_c(_, __):
            return ("soft_c", 1.1)

        monkeypatch.setattr(ch, "HARD_FILTERS", ())
        monkeypatch.setattr(ch, "SOFT_FILTERS", (soft_a, soft_b))
        monkeypatch.setattr(ch, "TIME_FILTERS", (soft_c,))
        monkeypatch.setattr(ch, "EVENT_FILTERS", ())

        s = _stub_signals(score=70)
        result = ch.apply_filters(s, _stub_ctx())
        assert result is s
        assert result.confidence == pytest.approx(0.7 * 0.4 * 0.7 * 1.1, abs=1e-6)
        # Order preserved in the audit list.
        assert [name for name, _ in s.soft_adjustments] == [
            "soft_a", "soft_b", "soft_c",
        ]

    def test_filter_returning_none_skipped(self, tmp_db, monkeypatch):
        """A filter that returns None doesn't contribute a multiplier
        nor an audit entry."""
        import filters.chain as ch

        def fires(_, __):
            return ("fires", 0.5)

        def silent(_, __):
            return None

        monkeypatch.setattr(ch, "HARD_FILTERS", ())
        monkeypatch.setattr(ch, "SOFT_FILTERS", (silent, fires))
        monkeypatch.setattr(ch, "TIME_FILTERS", ())
        monkeypatch.setattr(ch, "EVENT_FILTERS", ())

        s = _stub_signals(score=70)
        ch.apply_filters(s, _stub_ctx())
        assert s.soft_adjustments == [("fires", 0.5)]
        assert s.confidence == pytest.approx(0.35)

    def test_audit_row_on_hard_kill(self, tmp_db, monkeypatch):
        """Hard-killed signals write an audit row with
        ``alerted=False``, kill reason populated."""
        import sqlite3
        import filters.chain as ch

        def kill(_, __):
            return "earnings_within_3d"

        monkeypatch.setattr(ch, "HARD_FILTERS", (kill,))

        s = _stub_signals(score=70)
        ch.apply_filters(s, _stub_ctx())
        with sqlite3.connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT symbol, kill_reasons, alerted FROM filter_audit "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        symbol, reasons, alerted = row
        assert symbol == "TEST.NS"
        assert reasons == "earnings_within_3d"
        assert alerted == 0

    def test_write_audit_alerted_true(self, tmp_db):
        """Direct call to write_audit with alerted=True for the
        passed-chain path used by the scanner."""
        import sqlite3
        from filters.chain import write_audit

        s = _stub_signals(score=82)
        s.soft_adjustments.append(("mtf_alignment_both_agree", 1.10))
        s.confidence = 0.902
        write_audit(s, alerted=True)
        with sqlite3.connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT score, soft_adjustments_json, final_confidence, alerted "
                "FROM filter_audit ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        score, soft_json, confidence, alerted = row
        assert score == 82
        assert confidence == pytest.approx(0.902)
        assert alerted == 1
        # JSON round-trips
        import json
        parsed = json.loads(soft_json)
        assert parsed == [["mtf_alignment_both_agree", 1.10]]


# ---------------------------------------------------------------------------
# Hard filters (filters/hard.py)
# ---------------------------------------------------------------------------

class TestHardFilters:
    def test_market_open_blocks_pre_open(self):
        from filters.hard import market_open
        ctx = _stub_ctx("2026-05-12T09:00:00+05:30")
        assert "market_closed" in market_open(_stub_signals(), ctx)

    def test_market_open_blocks_after_close(self):
        from filters.hard import market_open
        ctx = _stub_ctx("2026-05-12T15:30:00+05:30")
        assert market_open(_stub_signals(), ctx) is not None

    def test_market_open_allows_intraday(self):
        from filters.hard import market_open
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert market_open(_stub_signals(), ctx) is None

    def test_liquidity_kills_thin_stocks(self):
        from filters.hard import liquidity
        daily = pd.DataFrame({"Volume": [100_000] * 20})
        ctx = _stub_ctx(daily_df=daily)
        result = liquidity(_stub_signals(), ctx)
        assert result is not None
        assert "liquidity" in result

    def test_liquidity_passes_liquid_stocks(self):
        from filters.hard import liquidity
        daily = pd.DataFrame({"Volume": [2_000_000] * 20})
        ctx = _stub_ctx(daily_df=daily)
        assert liquidity(_stub_signals(), ctx) is None

    def test_liquidity_fails_open_no_data(self):
        """No daily data → don't kill (avoid silencing every symbol
        when the daily cache fails to refresh)."""
        from filters.hard import liquidity
        ctx = _stub_ctx()  # daily_df=None by default
        assert liquidity(_stub_signals(), ctx) is None

    def test_ban_period_delegates_to_suppression(self, tmp_db, monkeypatch):
        """Filter is a single delegation — proves the wiring."""
        import filters.hard as hard
        called_with = []

        def fake_is_suppressed(symbol, cooldown):
            called_with.append((symbol, cooldown))
            return True, "test-suppression"

        monkeypatch.setattr(
            "bot.suppression.is_suppressed", fake_is_suppressed,
        )
        s = _stub_signals(symbol="X.NS")
        result = hard.ban_period(s, _stub_ctx())
        assert result == "test-suppression"
        assert called_with[0][0] == "X.NS"

    def test_corporate_action_today_kills_when_dividend_today(self, tmp_db):
        """A binary_high filing today must trigger the filter."""
        import sqlite3
        from filters.hard import corporate_action_today
        today_iso = "2026-05-12T10:00:00"
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("F1", "X.NS", "Dividend declared",
                 "binary_high", today_iso),
            )
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert corporate_action_today(_stub_signals(symbol="X.NS"), ctx) \
               == "corporate_action_today"

    def test_corporate_action_today_passes_other_symbols(self, tmp_db):
        """A binary_high filing on a DIFFERENT symbol today must NOT
        affect this signal."""
        import sqlite3
        from filters.hard import corporate_action_today
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("F1", "OTHER.NS", "Dividend declared",
                 "binary_high", "2026-05-12T10:00:00"),
            )
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert corporate_action_today(_stub_signals(symbol="X.NS"), ctx) is None

    def test_corporate_action_today_ignores_old_filings(self, tmp_db):
        """A binary_high filing from yesterday must NOT trigger today's
        filter — corporate-action drift is one-day only."""
        import sqlite3
        from filters.hard import corporate_action_today
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("F1", "X.NS", "Dividend declared",
                 "binary_high", "2026-05-11T10:00:00"),
            )
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert corporate_action_today(_stub_signals(symbol="X.NS"), ctx) is None

    def test_earnings_within_3d_kills(self, tmp_db):
        import sqlite3
        from filters.hard import earnings_within_3d
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("F1", "X.NS", "Audited Financial Results",
                 "event_unknown", "2026-05-10T16:00:00"),
            )
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert earnings_within_3d(_stub_signals(symbol="X.NS"), ctx) \
               == "earnings_within_3d"

    def test_earnings_within_3d_passes_old(self, tmp_db):
        """A filing older than 3 days must NOT block."""
        import sqlite3
        from filters.hard import earnings_within_3d
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                "INSERT INTO filings_seen "
                "(filing_id, symbol, title, classification, seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("F1", "X.NS", "Audited Financial Results",
                 "event_unknown", "2026-05-01T16:00:00"),
            )
        ctx = _stub_ctx("2026-05-12T11:00:00+05:30")
        assert earnings_within_3d(_stub_signals(symbol="X.NS"), ctx) is None

    def test_fno_ban_list_kills_member(self):
        from filters.hard import fno_ban_list
        ctx = _stub_ctx(fno_banned={"BIOCON", "RBLBANK"})
        assert fno_ban_list(_stub_signals(symbol="BIOCON.NS"), ctx) \
               == "fno_ban (BIOCON)"

    def test_fno_ban_list_passes_non_member(self):
        from filters.hard import fno_ban_list
        ctx = _stub_ctx(fno_banned={"BIOCON"})
        assert fno_ban_list(_stub_signals(symbol="RELIANCE.NS"), ctx) is None

    def test_fno_ban_list_fail_open_empty(self):
        """Empty ban set (e.g. fetcher down) → don't kill everything."""
        from filters.hard import fno_ban_list
        ctx = _stub_ctx(fno_banned=set())
        assert fno_ban_list(_stub_signals(symbol="X.NS"), ctx) is None

    def test_nifty_crash_kills_below_threshold(self):
        from filters.hard import nifty_crash
        ctx = _stub_ctx(nifty_pct=-1.7)
        result = nifty_crash(_stub_signals(), ctx)
        assert result is not None
        assert "nifty_crash" in result

    def test_nifty_crash_passes_normal_session(self):
        from filters.hard import nifty_crash
        ctx = _stub_ctx(nifty_pct=-0.5)
        assert nifty_crash(_stub_signals(), ctx) is None

    def test_nifty_crash_fail_open_missing(self):
        from filters.hard import nifty_crash
        ctx = _stub_ctx()  # nifty_pct=None
        assert nifty_crash(_stub_signals(), ctx) is None

    def test_circuit_proximity_kills_near_upper_band(self):
        """price within 2% of prev_close * 1.10 → kill."""
        from filters.hard import circuit_proximity
        # prev_close=100 → upper band 110. Price 109 is within 2% of 110.
        daily = pd.DataFrame({"Close": [100.0]})
        ctx = _stub_ctx(daily_df=daily)
        s = _stub_signals(price=109.0)
        result = circuit_proximity(s, ctx)
        assert result is not None
        assert "upper" in result

    def test_circuit_proximity_kills_near_lower_band(self):
        from filters.hard import circuit_proximity
        # prev_close=100 → lower band 90. Price 91 is within 2% of 90.
        daily = pd.DataFrame({"Close": [100.0]})
        ctx = _stub_ctx(daily_df=daily)
        s = _stub_signals(price=91.0)
        result = circuit_proximity(s, ctx)
        assert result is not None
        assert "lower" in result

    def test_circuit_proximity_passes_mid_range(self):
        from filters.hard import circuit_proximity
        daily = pd.DataFrame({"Close": [100.0]})
        ctx = _stub_ctx(daily_df=daily)
        s = _stub_signals(price=103.0)  # well inside the band
        assert circuit_proximity(s, ctx) is None

    def test_circuit_proximity_fails_open_no_daily(self):
        from filters.hard import circuit_proximity
        ctx = _stub_ctx()
        assert circuit_proximity(_stub_signals(price=110.0), ctx) is None


# ---------------------------------------------------------------------------
# Soft filters: bank_nifty_opposite + vix_filter (Phase-6 tasks 25, 26)
# ---------------------------------------------------------------------------

class TestBankNiftyOpposite:
    def test_long_with_bank_nifty_falling_demotes(self):
        from filters.soft import bank_nifty_opposite
        ctx = _stub_ctx(bank_nifty_pct=-0.5)
        s = _stub_signals(side="LONG")
        assert bank_nifty_opposite(s, ctx) == ("bank_nifty_opposite", 0.9)

    def test_long_with_bank_nifty_rising_passes(self):
        from filters.soft import bank_nifty_opposite
        ctx = _stub_ctx(bank_nifty_pct=0.4)
        s = _stub_signals(side="LONG")
        assert bank_nifty_opposite(s, ctx) is None

    def test_short_with_bank_nifty_rising_demotes(self):
        from filters.soft import bank_nifty_opposite
        ctx = _stub_ctx(bank_nifty_pct=0.5)
        s = _stub_signals(side="SHORT")
        assert bank_nifty_opposite(s, ctx) == ("bank_nifty_opposite", 0.9)

    def test_flat_market_no_penalty(self):
        from filters.soft import bank_nifty_opposite
        ctx = _stub_ctx(bank_nifty_pct=0.1)
        s = _stub_signals(side="LONG")
        assert bank_nifty_opposite(s, ctx) is None

    def test_missing_data_fails_open(self):
        from filters.soft import bank_nifty_opposite
        ctx = _stub_ctx()  # bank_nifty_pct=None
        s = _stub_signals(side="LONG")
        assert bank_nifty_opposite(s, ctx) is None


class TestVixFilter:
    def test_low_vix_bullish_bonus(self):
        from filters.soft import vix_filter
        ctx = _stub_ctx(vix=11.0)
        assert vix_filter(_stub_signals(), ctx) == ("vix_low", 1.05)

    def test_mid_vix_no_change(self):
        """VIX 12-16 is the median state — no multiplier."""
        from filters.soft import vix_filter
        ctx = _stub_ctx(vix=14.0)
        assert vix_filter(_stub_signals(), ctx) is None

    def test_high_vix_demotes(self):
        from filters.soft import vix_filter
        ctx = _stub_ctx(vix=18.0)
        assert vix_filter(_stub_signals(), ctx) == ("vix_high", 0.85)

    def test_panic_vix_heavy_demote(self):
        from filters.soft import vix_filter
        ctx = _stub_ctx(vix=22.0)
        assert vix_filter(_stub_signals(), ctx) == ("vix_panic", 0.6)

    def test_missing_vix_fails_open(self):
        from filters.soft import vix_filter
        ctx = _stub_ctx()  # vix=None
        assert vix_filter(_stub_signals(), ctx) is None


# ---------------------------------------------------------------------------
# Index feed (data/index_feed.py) — uses mocked yfinance fetch
# ---------------------------------------------------------------------------

class TestIndexFeed:
    @pytest.fixture(autouse=True)
    def _reset(self):
        """Clear caches between tests."""
        from data.index_feed import _reset_caches
        _reset_caches()
        yield
        _reset_caches()

    def test_intraday_pct_math(self, monkeypatch):
        """open=100 → last=102 → +2.0%. Mock the fetch directly."""
        import data.index_feed as feed
        from zoneinfo import ZoneInfo
        IST_local = ZoneInfo("Asia/Kolkata")
        today = pd.Timestamp.now(tz=IST_local).date()
        idx = pd.date_range(f"{today} 09:15", periods=3, freq="5min",
                            tz=IST_local)
        df = pd.DataFrame({
            "Open": [100.0, 100.5, 101.0],
            "High": [100.5, 101.0, 102.0],
            "Low": [99.5, 100.0, 100.5],
            "Close": [100.5, 101.0, 102.0],
            "Volume": [0, 0, 0],
        }, index=idx)
        monkeypatch.setattr(feed, "_fetch_intraday", lambda sym: df)
        pct = feed.get_intraday_pct("^NSEI")
        assert pct == pytest.approx(2.0)

    def test_intraday_pct_returns_none_on_empty(self, monkeypatch):
        import data.index_feed as feed
        monkeypatch.setattr(feed, "_fetch_intraday", lambda sym: pd.DataFrame())
        assert feed.get_intraday_pct("^NSEI") is None

    def test_get_direction_classifies(self, monkeypatch):
        import data.index_feed as feed
        monkeypatch.setattr(feed, "get_intraday_pct", lambda sym: 0.5)
        assert feed.get_direction("^NSEBANK") == "UP"
        monkeypatch.setattr(feed, "get_intraday_pct", lambda sym: -0.5)
        assert feed.get_direction("^NSEBANK") == "DOWN"
        monkeypatch.setattr(feed, "get_intraday_pct", lambda sym: 0.1)
        assert feed.get_direction("^NSEBANK") == "FLAT"
        monkeypatch.setattr(feed, "get_intraday_pct", lambda sym: None)
        assert feed.get_direction("^NSEBANK") == "FLAT"


# ---------------------------------------------------------------------------
# ADX-based soft filters (Phase-6 task 27 — the Meesho rule + 3 others)
# ---------------------------------------------------------------------------

def _stub_snapshot(**values):
    """Bare IndicatorSnapshot whose .values dict mirrors the kwargs."""
    from datetime import date as _Date, datetime, timezone
    from indicators.compute import IndicatorSnapshot
    return IndicatorSnapshot(
        symbol="TEST.NS", session_date=_Date(2026, 5, 12),
        computed_at=datetime.now(timezone.utc),
        values=dict(values),
    )


class TestAdxKillCounterTrend:
    """The Meesho rule: ADX>50 + DI lines opposing trade side → 0.4."""

    def test_meesho_long_strong_downtrend(self):
        """ADX=54, DI- > DI+, side=LONG → demote 0.4."""
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals(side="LONG")
        s.snapshot = _stub_snapshot(
            adx_5m=54.0, adx_5m_di_plus=15.0, adx_5m_di_minus=35.0,
        )
        assert adx_kill_counter_trend(s, _stub_ctx()) == (
            "adx_counter_trend", 0.4,
        )

    def test_short_strong_uptrend_also_fires(self):
        """SHORT against a strong uptrend → also 0.4."""
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals(side="SHORT")
        s.snapshot = _stub_snapshot(
            adx_5m=54.0, adx_5m_di_plus=35.0, adx_5m_di_minus=15.0,
        )
        assert adx_kill_counter_trend(s, _stub_ctx()) == (
            "adx_counter_trend", 0.4,
        )

    def test_trend_aligned_no_penalty(self):
        """LONG in an uptrend (DI+ > DI-, ADX high) → no penalty."""
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals(side="LONG")
        s.snapshot = _stub_snapshot(
            adx_5m=54.0, adx_5m_di_plus=35.0, adx_5m_di_minus=15.0,
        )
        assert adx_kill_counter_trend(s, _stub_ctx()) is None

    def test_below_adx_threshold_no_penalty(self):
        """ADX < 50: the counter-trend rule doesn't kick in. The
        adx_weak filter handles low-ADX separately."""
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals(side="LONG")
        s.snapshot = _stub_snapshot(
            adx_5m=35.0, adx_5m_di_plus=15.0, adx_5m_di_minus=30.0,
        )
        assert adx_kill_counter_trend(s, _stub_ctx()) is None

    def test_missing_snapshot_fails_open(self):
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals()
        s.snapshot = None
        assert adx_kill_counter_trend(s, _stub_ctx()) is None

    def test_missing_adx_keys_fail_open(self):
        """Snapshot exists but ADX wasn't computed (insufficient warmup,
        say) → return None, let downstream filters run."""
        from filters.soft import adx_kill_counter_trend
        s = _stub_signals()
        s.snapshot = _stub_snapshot()  # empty values
        assert adx_kill_counter_trend(s, _stub_ctx()) is None


class TestAdxWeakTrend:
    def test_weak_adx_demotes(self):
        from filters.soft import adx_weak_trend
        s = _stub_signals()
        s.snapshot = _stub_snapshot(adx_5m=15.0)
        assert adx_weak_trend(s, _stub_ctx()) == ("adx_weak", 0.7)

    def test_strong_adx_no_penalty(self):
        from filters.soft import adx_weak_trend
        s = _stub_signals()
        s.snapshot = _stub_snapshot(adx_5m=35.0)
        assert adx_weak_trend(s, _stub_ctx()) is None

    def test_at_threshold_boundary(self):
        """ADX = 20 sharp → no penalty (boundary excluded)."""
        from filters.soft import adx_weak_trend
        s = _stub_signals()
        s.snapshot = _stub_snapshot(adx_5m=20.0)
        assert adx_weak_trend(s, _stub_ctx()) is None


class TestAlreadyExtended:
    def _intraday_df(self, open_price: float, current_price: float):
        """Build a synthetic intraday df with the given open and
        a final bar matching ``current_price``."""
        idx = pd.date_range("2026-05-12 09:15", periods=2, freq="5min",
                            tz=IST)
        return pd.DataFrame({
            "Open": [open_price, current_price],
            "High": [open_price, current_price],
            "Low": [open_price, current_price],
            "Close": [open_price, current_price],
            "Volume": [0, 0],
        }, index=idx)

    def test_long_chasing_parabola_demotes(self):
        from filters.soft import already_extended
        s = _stub_signals(side="LONG", price=106.0)
        ctx = _stub_ctx(intraday_df=self._intraday_df(100.0, 106.0))
        # Up 6% → demote.
        assert already_extended(s, ctx) == ("already_extended", 0.8)

    def test_long_normal_move_passes(self):
        from filters.soft import already_extended
        s = _stub_signals(side="LONG", price=103.0)
        ctx = _stub_ctx(intraday_df=self._intraday_df(100.0, 103.0))
        # Up 3% → fine.
        assert already_extended(s, ctx) is None

    def test_short_already_down_demotes(self):
        from filters.soft import already_extended
        s = _stub_signals(side="SHORT", price=94.0)
        ctx = _stub_ctx(intraday_df=self._intraday_df(100.0, 94.0))
        # Down 6% → demote.
        assert already_extended(s, ctx) == ("already_extended", 0.8)

    def test_no_intraday_data_fails_open(self):
        from filters.soft import already_extended
        s = _stub_signals(side="LONG", price=106.0)
        ctx = _stub_ctx()  # intraday_df=None
        assert already_extended(s, ctx) is None


class TestLowVolume:
    def test_below_threshold_demotes(self):
        from filters.soft import low_volume
        s = _stub_signals(volume_ratio=0.6)
        assert low_volume(s, _stub_ctx()) == ("low_volume", 0.85)

    def test_normal_volume_passes(self):
        from filters.soft import low_volume
        s = _stub_signals(volume_ratio=2.0)
        assert low_volume(s, _stub_ctx()) is None

    def test_at_threshold_boundary(self):
        """volume_ratio = 1.0 sharp → no penalty (boundary excluded)."""
        from filters.soft import low_volume
        s = _stub_signals(volume_ratio=1.0)
        assert low_volume(s, _stub_ctx()) is None


# ---------------------------------------------------------------------------
# MTF trend alignment (Phase-6 task 28)
# ---------------------------------------------------------------------------

class TestMtfTrendAlignment:
    def _snap_with_mtf(self, dir_15m: str, dir_60m: str):
        """Build a snapshot with the requested 15m/60m DI directions.
        DI values are arbitrary so long as the inequality matches."""
        values = {}
        # 15m
        if dir_15m == "UP":
            values["adx_15m_di_plus"] = 30.0
            values["adx_15m_di_minus"] = 15.0
        elif dir_15m == "DOWN":
            values["adx_15m_di_plus"] = 15.0
            values["adx_15m_di_minus"] = 30.0
        # 60m
        if dir_60m == "UP":
            values["adx_60m_di_plus"] = 30.0
            values["adx_60m_di_minus"] = 15.0
        elif dir_60m == "DOWN":
            values["adx_60m_di_plus"] = 15.0
            values["adx_60m_di_minus"] = 30.0
        return _stub_snapshot(**values)

    def test_long_with_both_up_boosts(self):
        from filters.soft import mtf_trend_alignment
        s = _stub_signals(side="LONG")
        s.snapshot = self._snap_with_mtf("UP", "UP")
        assert mtf_trend_alignment(s, _stub_ctx()) == ("mtf_both_agree", 1.10)

    def test_long_with_both_down_heavy_demote(self):
        from filters.soft import mtf_trend_alignment
        s = _stub_signals(side="LONG")
        s.snapshot = self._snap_with_mtf("DOWN", "DOWN")
        assert mtf_trend_alignment(s, _stub_ctx()) == ("mtf_both_disagree", 0.5)

    def test_long_mixed_mild_demote(self):
        from filters.soft import mtf_trend_alignment
        s = _stub_signals(side="LONG")
        s.snapshot = self._snap_with_mtf("UP", "DOWN")
        assert mtf_trend_alignment(s, _stub_ctx()) == ("mtf_one_disagree", 0.85)

    def test_short_mirror(self):
        """SHORT side mirrors LONG logic."""
        from filters.soft import mtf_trend_alignment
        s = _stub_signals(side="SHORT")
        s.snapshot = self._snap_with_mtf("DOWN", "DOWN")
        assert mtf_trend_alignment(s, _stub_ctx()) == ("mtf_both_agree", 1.10)

    def test_partial_data_only_one_tf_fails_open(self):
        """If 60m hasn't reached warmup, only 15m is available.
        Filter returns None to be predictable."""
        from filters.soft import mtf_trend_alignment
        s = _stub_signals(side="LONG")
        s.snapshot = _stub_snapshot(
            adx_15m_di_plus=30.0, adx_15m_di_minus=15.0,
        )
        assert mtf_trend_alignment(s, _stub_ctx()) is None

    def test_no_snapshot_fails_open(self):
        from filters.soft import mtf_trend_alignment
        s = _stub_signals()
        s.snapshot = None
        assert mtf_trend_alignment(s, _stub_ctx()) is None
