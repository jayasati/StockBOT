"""Phase-5 paper-tracker tests.

The suite is built up in the same order as the implementation phases:

1. Schema     — DDL idempotency, FK enforcement, CHECK constraints.
2. Helpers    — pure logic (``_evaluate_bar``, ``_compute_pnl``,
                  ``_flatten_snapshot``, ``_is_timed_out``).
3. Writes     — ``open_trade`` / ``close_manual`` + dup-guard.
4. Reads      — ``journal.list_open`` / ``list_closed`` / ``daily_summary``.
5. CLI        — ``python -m paper.journal report``.
6. Monitor    — async loop end-to-end with a stubbed feed.

All tests use a tmp_path-backed alerts.db so the user's real DB is
never touched. ``bot.config.DB_PATH`` is monkeypatched per-test."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd
import pytest


@dataclass
class FakeBar:
    """Duck-typed stand-in for ``data.realtime_feed.Bar`` — exposes
    just the fields ``_evaluate_bar`` reads (high, low). Used across
    helper + monitor tests."""
    high: float
    low: float
    close: float = 0.0
    open: float = 0.0
    volume: float = 0.0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Per-test alerts.db. Patches ``DB_PATH`` everywhere it was
    bound at import time (``bot.config``, ``bot.storage``, ``bot.db``)
    so every code path resolves to the tmp file, then runs
    ``bot.db.init_db()`` so EVERY table exists — ``alerts_sent`` for
    ``record_alert``, plus ``paper_trades`` + ``signal_indicators``."""
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
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_idempotent(self, tmp_db):
        from paper.schema import connect, ensure_paper_schema

        ensure_paper_schema(tmp_db)
        ensure_paper_schema(tmp_db)  # must not raise on re-create
        with connect(tmp_db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name IN "
                    "  ('paper_trades','signal_indicators')"
                ).fetchall()
            }
        assert tables == {"paper_trades", "signal_indicators"}

    def test_fk_enforced(self, tmp_db):
        """signal_indicators.paper_trade_id must reject orphans —
        proves ``PRAGMA foreign_keys = ON`` actually fires on the
        connection paper/* uses."""
        from paper.schema import connect, ensure_paper_schema

        ensure_paper_schema(tmp_db)
        with connect(tmp_db) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO signal_indicators "
                    "(paper_trade_id, indicator, value, timeframe) "
                    "VALUES (?, ?, ?, ?)",
                    (99999, "rsi", 50.0, "5m"),
                )

    def test_status_check_constraint(self, tmp_db):
        """``status`` must be one of the documented six values."""
        from paper.schema import connect, ensure_paper_schema

        ensure_paper_schema(tmp_db)
        with connect(tmp_db) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO paper_trades "
                    "(symbol, side, entry_ts, entry_price, qty, "
                    " stop_loss, target_1, confidence, status) "
                    "VALUES "
                    "('TEST.NS','LONG','2026-05-12T10:00:00',100,1,"
                    " 98,104,0.7,'BOGUS')"
                )

    def test_side_check_constraint(self, tmp_db):
        from paper.schema import connect, ensure_paper_schema

        ensure_paper_schema(tmp_db)
        with connect(tmp_db) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO paper_trades "
                    "(symbol, side, entry_ts, entry_price, qty, "
                    " stop_loss, target_1, confidence) "
                    "VALUES ('TEST.NS','BUY','t',100,1,98,104,0.7)"
                )


# ---------------------------------------------------------------------------
# Pure helpers: _evaluate_bar / _compute_pnl / _flatten_snapshot / _is_timed_out
# ---------------------------------------------------------------------------

class TestEvaluateBar:
    """Pure rule logic — no DB, no realtime feed."""

    def test_long_sl_only(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=None,
                               bar=FakeBar(high=100, low=97))
        assert result == ("SL", 98, "")

    def test_long_tp1_only(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=110,
                               bar=FakeBar(high=105, low=100))
        assert result == ("TP1", 104, "")

    def test_long_tp2_preferred(self):
        """When the bar reaches both TP1 and TP2, the further target wins."""
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=110,
                               bar=FakeBar(high=111, low=100))
        assert result == ("TP2", 110, "")

    def test_long_sl_and_tp1_same_bar(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=None,
                               bar=FakeBar(high=105, low=97))
        assert result[0] == "TP1"
        assert result[1] == 104
        assert "SL+TP1 same bar" in result[2]

    def test_long_sl_and_tp2_same_bar(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=110,
                               bar=FakeBar(high=111, low=97))
        assert result[0] == "TP2"
        assert result[1] == 110
        assert "SL+TP2 same bar" in result[2]

    def test_short_tp1_only(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("SHORT", sl=102, tp1=96, tp2=None,
                               bar=FakeBar(high=101, low=95))
        assert result == ("TP1", 96, "")

    def test_short_sl_only(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("SHORT", sl=102, tp1=96, tp2=None,
                               bar=FakeBar(high=103, low=99))
        assert result == ("SL", 102, "")

    def test_short_sl_and_tp_same_bar(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("SHORT", sl=102, tp1=96, tp2=90,
                               bar=FakeBar(high=103, low=89))
        assert result[0] == "TP2"
        assert result[1] == 90

    def test_no_fill(self):
        from paper.tracker import _evaluate_bar
        result = _evaluate_bar("LONG", sl=98, tp1=104, tp2=None,
                               bar=FakeBar(high=103, low=99))
        assert result is None

    def test_none_bar(self):
        from paper.tracker import _evaluate_bar
        assert _evaluate_bar("LONG", 98, 104, None, None) is None


class TestComputePnl:
    def test_long_winner(self):
        from paper.tracker import _compute_pnl
        gross, net = _compute_pnl("LONG", entry=100.0, exit_price=104.0, qty=100)
        assert gross == pytest.approx(400.0)
        assert net < gross  # costs eat into the gross

    def test_long_loser_net_more_negative(self):
        from paper.tracker import _compute_pnl
        gross, net = _compute_pnl("LONG", entry=100.0, exit_price=98.0, qty=100)
        assert gross == pytest.approx(-200.0)
        assert net < gross  # costs add to the loss

    def test_short_winner(self):
        from paper.tracker import _compute_pnl
        gross, net = _compute_pnl("SHORT", entry=100.0, exit_price=96.0, qty=100)
        assert gross == pytest.approx(400.0)
        assert net < gross

    def test_costs_equal_round_trip(self):
        """The delta between gross and net must match the cost module
        exactly — proves the right API is being called."""
        from paper.tracker import _compute_pnl
        from trading.costs import round_trip_cost
        gross, net = _compute_pnl("LONG", entry=100.0, exit_price=104.0, qty=100)
        expected_costs = round_trip_cost(100.0 * 100)["total"]
        assert (gross - net) == pytest.approx(expected_costs)


class TestFlattenSnapshot:
    def test_series_keys_split_tf(self):
        from paper.tracker import _flatten_snapshot
        flat = _flatten_snapshot({"rsi_5m": 65.0, "atr_5m": 1.2})
        assert flat == {"rsi": (65.0, "5m"), "atr": (1.2, "5m")}

    def test_frame_columns_keep_column_suffix(self):
        from paper.tracker import _flatten_snapshot
        flat = _flatten_snapshot({
            "macd_5m_macd": 0.5,
            "macd_5m_signal": 0.3,
            "macd_5m_histogram": 0.2,
        })
        assert flat == {
            "macd_macd": (0.5, "5m"),
            "macd_signal": (0.3, "5m"),
            "macd_histogram": (0.2, "5m"),
        }

    def test_adx_components(self):
        from paper.tracker import _flatten_snapshot
        flat = _flatten_snapshot({
            "adx_15m_adx": 28.0,
            "adx_15m_di_plus": 30.0,
            "adx_15m_di_minus": 12.0,
        })
        assert flat["adx_adx"] == (28.0, "15m")
        assert flat["adx_di_plus"] == (30.0, "15m")
        assert flat["adx_di_minus"] == (12.0, "15m")

    def test_levels_use_session_tf(self):
        from paper.tracker import _flatten_snapshot
        flat = _flatten_snapshot({
            "pdh": 1500.0, "pdl": 1450.0,
            "orh_15": 1480.0,                # minutes suffix, not a TF
            "pivot_classic_r1": 1510.0,
        })
        assert flat["pdh"] == (1500.0, "session")
        assert flat["pdl"] == (1450.0, "session")
        assert flat["orh_15"] == (1480.0, "session")
        assert flat["pivot_classic_r1"] == (1510.0, "session")

    def test_drops_none_values(self):
        from paper.tracker import _flatten_snapshot
        flat = _flatten_snapshot({"rsi_5m": None, "atr_5m": 1.2})
        assert flat == {"atr": (1.2, "5m")}

    def test_none_snapshot(self):
        from paper.tracker import _flatten_snapshot
        assert _flatten_snapshot(None) == {}

    def test_indicator_snapshot_object(self):
        """Accepts a real ``IndicatorSnapshot`` (uses ``.values``)."""
        from datetime import date, datetime, timezone

        from indicators.compute import IndicatorSnapshot
        from paper.tracker import _flatten_snapshot

        snap = IndicatorSnapshot(
            symbol="TEST.NS",
            session_date=date(2026, 5, 12),
            computed_at=datetime.now(timezone.utc),
            values={"rsi_5m": 65.0, "pdh": 1500.0},
        )
        flat = _flatten_snapshot(snap)
        assert flat == {"rsi": (65.0, "5m"), "pdh": (1500.0, "session")}


class TestIsTimedOut:
    """Strict-intraday predicate: every trade times out at 15:30 IST
    on its entry day. Same-day rule keeps the simulation honest with
    the MIS cost model in ``trading.costs``."""

    def test_same_day_before_close_not_timed_out(self):
        from paper.tracker import IST, _is_timed_out
        entry = "2026-01-06T10:00:00+05:30"   # Tuesday entry
        now = pd.Timestamp("2026-01-06T14:00:00", tz=IST)
        assert _is_timed_out(entry, now) is False

    def test_same_day_at_close_timed_out(self):
        """15:30:00 IST is the exact deadline — predicate fires."""
        from paper.tracker import IST, _is_timed_out
        entry = "2026-01-06T10:00:00+05:30"
        now = pd.Timestamp("2026-01-06T15:30:00", tz=IST)
        assert _is_timed_out(entry, now) is True

    def test_same_day_after_close_timed_out(self):
        from paper.tracker import IST, _is_timed_out
        entry = "2026-01-06T10:00:00+05:30"
        now = pd.Timestamp("2026-01-06T15:31:00", tz=IST)
        assert _is_timed_out(entry, now) is True

    def test_next_morning_still_timed_out(self):
        """A trade not closed by 15:30 stays timed-out indefinitely
        — the monitor will sweep it on the next tick."""
        from paper.tracker import IST, _is_timed_out
        entry = "2026-01-06T14:00:00+05:30"
        now = pd.Timestamp("2026-01-07T10:00:00", tz=IST)
        assert _is_timed_out(entry, now) is True

    def test_friday_entry_timed_out_at_close(self):
        from paper.tracker import IST, _is_timed_out
        entry = "2026-01-02T10:00:00+05:30"   # Friday entry
        now = pd.Timestamp("2026-01-02T15:31:00", tz=IST)
        assert _is_timed_out(entry, now) is True

    def test_default_now_uses_current_time(self):
        """``now_ist=None`` should fall through to ``pd.Timestamp.now(tz=IST)``."""
        from paper.tracker import _is_timed_out
        entry = "2030-01-01T10:00:00+05:30"   # far future entry
        # Far-future trade can't be timed out NOW, regardless of when "now" is.
        assert _is_timed_out(entry) is False


# ---------------------------------------------------------------------------
# Writes: open_trade + close_manual + dup-guard
# ---------------------------------------------------------------------------

class TestOpenTrade:
    def test_long_open_creates_row(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import open_trade
        trade_id = open_trade(
            symbol="TEST.NS", side="LONG", entry=100.0,
            sl=98.0, tp1=104.0, tp2=110.0, qty=10, confidence=0.7,
            indicator_snapshot={"rsi_5m": 65.0, "atr_5m": 1.5},
        )
        assert trade_id > 0
        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT symbol, side, entry_price, stop_loss, target_1, "
                "       target_2, confidence, status "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert row == ("TEST.NS", "LONG", 100.0, 98.0, 104.0, 110.0, 0.7, "OPEN")

    def test_indicator_snapshot_rows_persisted(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import open_trade
        trade_id = open_trade(
            "TEST.NS", "LONG", 100.0, 98.0, 104.0, None, 10, 0.7,
            indicator_snapshot={
                "rsi_5m": 65.0,
                "atr_5m": 1.5,
                "macd_5m_histogram": 0.4,
                "pdh": 105.0,
            },
        )
        with connect(tmp_db) as conn:
            rows = conn.execute(
                "SELECT indicator, value, timeframe FROM signal_indicators "
                "WHERE paper_trade_id = ?",
                (trade_id,),
            ).fetchall()
        as_dict = {(ind, tf): v for ind, v, tf in rows}
        assert as_dict[("rsi", "5m")] == 65.0
        assert as_dict[("atr", "5m")] == 1.5
        assert as_dict[("macd_histogram", "5m")] == 0.4
        assert as_dict[("pdh", "session")] == 105.0

    def test_duplicate_open_returns_existing_id(self, tmp_db):
        """Second call for the same symbol with an OPEN trade must not
        insert a new row — returns the existing id instead. This is
        the in-tracker dup-guard belt complementing the suppression
        cooldown."""
        from paper.schema import connect
        from paper.tracker import open_trade
        id1 = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        id2 = open_trade("TEST.NS", "LONG", 101, 99, 105, None, 10, 0.8, {})
        assert id1 == id2
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE symbol='TEST.NS'"
            ).fetchone()[0]
        assert count == 1

    def test_two_symbols_concurrent(self, tmp_db):
        """Open trades on different symbols must coexist as separate rows."""
        from paper.schema import connect
        from paper.tracker import open_trade
        id1 = open_trade("AAA.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        id2 = open_trade("BBB.NS", "LONG", 200, 196, 208, None, 5, 0.7, {})
        assert id1 != id2
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE status='OPEN'"
            ).fetchone()[0]
        assert count == 2

    def test_default_entry_ts_is_now(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import IST, open_trade
        trade_id = open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, None,
        )
        with connect(tmp_db) as conn:
            entry_ts_str = conn.execute(
                "SELECT entry_ts FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        ts = pd.Timestamp(entry_ts_str)
        now = pd.Timestamp.now(tz=IST)
        delta = abs((now - ts).total_seconds())
        assert delta < 5.0, f"entry_ts drifted from now by {delta}s"

    def test_none_snapshot_writes_no_indicator_rows(self, tmp_db):
        """A trade with no indicator snapshot is still recorded."""
        from paper.schema import connect
        from paper.tracker import open_trade
        trade_id = open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, None,
        )
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM signal_indicators WHERE paper_trade_id=?",
                (trade_id,),
            ).fetchone()[0]
        assert count == 0


class TestCloseManual:
    def test_basic_close_with_ltp(self, tmp_db, monkeypatch):
        from paper.schema import connect
        from paper.tracker import close_manual, open_trade
        monkeypatch.setattr(
            "data.realtime_feed.get_current_partial",
            lambda symbol: FakeBar(high=100.5, low=99.5, close=100.2),
        )
        trade_id = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        close_manual(trade_id, reason="delisted intraday")
        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, exit_price, pnl_gross, notes "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, exit_price, gross, notes = row
        assert status == "MANUAL"
        assert exit_price == pytest.approx(100.2)
        assert gross == pytest.approx((100.2 - 100) * 10)
        assert notes == "delisted intraday"

    def test_close_no_ltp_nulls_pnl(self, tmp_db, monkeypatch):
        """If get_current_partial returns None, pnl_* stay NULL but
        status='MANUAL' is still applied — the trade exits the OPEN
        pool either way."""
        from paper.schema import connect
        from paper.tracker import close_manual, open_trade
        monkeypatch.setattr(
            "data.realtime_feed.get_current_partial",
            lambda symbol: None,
        )
        trade_id = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        close_manual(trade_id, reason="ticks dried up")
        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, exit_price, pnl_gross, pnl_net, notes "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert row == ("MANUAL", None, None, None, "ticks dried up")

    def test_close_already_closed_is_noop(self, tmp_db, monkeypatch):
        from paper.schema import connect
        from paper.tracker import close_manual, open_trade
        monkeypatch.setattr(
            "data.realtime_feed.get_current_partial",
            lambda symbol: FakeBar(high=100.5, low=99.5, close=100.2),
        )
        trade_id = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        close_manual(trade_id, reason="first close")
        close_manual(trade_id, reason="second close should noop")
        with connect(tmp_db) as conn:
            notes = conn.execute(
                "SELECT notes FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert notes == "first close"

    def test_close_missing_id_raises(self, tmp_db):
        from paper.tracker import close_manual
        with pytest.raises(ValueError, match="not found"):
            close_manual(99999, reason="bogus")


# ---------------------------------------------------------------------------
# Journal reads: list_open / list_closed / daily_summary / win_rate_by_indicator
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_db(tmp_db):
    """Seed a mixed-status dataset: 2 OPEN, 3 closed.

    Closed details (all closed "today" in IST):
      CCC.NS LONG  150 → TP1 156 → +60 gross (winner)
      DDD.NS LONG  100 → SL   98 → -20 gross (loser)
      EEE.NS SHORT 100 → TP1  96 → +40 gross (winner)

    Indicator snapshots are picked so the per-indicator pivot has
    enough variety: 'rsi' appears in 1 winner + 1 loser (50% wins),
    'adx_adx' appears in 1 winner only (100% wins)."""
    from paper.tracker import _close, open_trade

    open_trade("AAA.NS", "LONG", 100, 98, 104, None, 10, 0.7,
               {"rsi_5m": 65.0})
    open_trade("BBB.NS", "LONG", 200, 196, 208, None, 5, 0.8,
               {"rsi_5m": 70.0})
    id3 = open_trade("CCC.NS", "LONG", 150, 147, 156, None, 10, 0.7,
                     {"rsi_5m": 62.0})
    _close(id3, "TP1", 156.0)
    id4 = open_trade("DDD.NS", "LONG", 100, 98, 104, None, 10, 0.7,
                     {"rsi_5m": 55.0})
    _close(id4, "SL", 98.0)
    id5 = open_trade("EEE.NS", "SHORT", 100, 102, 96, None, 10, 0.7,
                     {"adx_5m_adx": 28.0})
    _close(id5, "TP1", 96.0)
    return tmp_db


class TestListOpen:
    def test_returns_only_open(self, populated_db):
        from paper.journal import list_open
        trades = list_open()
        assert len(trades) == 2
        assert all(t.status == "OPEN" for t in trades)
        assert {t.symbol for t in trades} == {"AAA.NS", "BBB.NS"}

    def test_ordered_by_entry_ts(self, populated_db):
        from paper.journal import list_open
        trades = list_open()
        assert trades[0].entry_ts <= trades[1].entry_ts

    def test_empty_returns_empty_list(self, tmp_db):
        """No trades yet → returns []."""
        from paper.journal import list_open
        assert list_open() == []


class TestListClosed:
    def test_all_closed(self, populated_db):
        from paper.journal import list_closed
        trades = list_closed()
        assert len(trades) == 3
        assert all(t.status != "OPEN" for t in trades)

    def test_since_today_returns_all(self, populated_db):
        from datetime import date
        from paper.journal import list_closed
        # All seeded trades closed "today" → since=today returns all.
        from paper.tracker import IST
        today_ist = pd.Timestamp.now(tz=IST).date()
        assert len(list_closed(since=today_ist)) == 3

    def test_since_future_returns_none(self, populated_db):
        from datetime import date
        from paper.journal import list_closed
        future = date(2099, 1, 1)
        assert list_closed(since=future) == []


class TestDailySummary:
    def test_today_aggregate(self, populated_db):
        """Gross: +60, -20, +40 = +80. Winners (net): 2, Loser: 1."""
        from paper.journal import daily_summary
        from paper.tracker import IST
        today_ist = pd.Timestamp.now(tz=IST).date()
        summary = daily_summary(today_ist)

        assert summary["n_trades"] == 3
        assert summary["win_rate"] == pytest.approx(2 / 3)
        assert summary["gross_pnl"] == pytest.approx(80.0)
        # Net is always strictly less than gross because round-trip costs
        # eat into every leg. Exact values are validated in TestComputePnl.
        assert summary["net_pnl"] < summary["gross_pnl"]
        assert summary["avg_winner"] > 0
        assert summary["avg_loser"] < 0
        # Two winners against one loser of similar magnitude → factor > 1.
        assert summary["profit_factor"] > 1.0

    def test_empty_date_returns_zeros(self, populated_db):
        """A date with no closed trades returns a zero-shaped summary."""
        from datetime import date
        from paper.journal import daily_summary
        summary = daily_summary(date(2000, 1, 1))
        assert summary["n_trades"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["gross_pnl"] == 0.0
        assert summary["profit_factor"] == 0.0


class TestWinRateByIndicator:
    def test_pivot_columns(self, populated_db):
        from paper.journal import win_rate_by_indicator
        df = win_rate_by_indicator()
        assert set(df.columns) == {
            "indicator", "timeframe", "n_trades", "win_rate", "avg_pnl_net",
        }

    def test_rsi_appears_in_two_closed_trades(self, populated_db):
        """``rsi_5m`` was on CCC (winner) and DDD (loser) → 50% win-rate."""
        from paper.journal import win_rate_by_indicator
        df = win_rate_by_indicator()
        rsi = df[(df.indicator == "rsi") & (df.timeframe == "5m")]
        assert len(rsi) == 1
        assert rsi["n_trades"].iloc[0] == 2
        assert rsi["win_rate"].iloc[0] == pytest.approx(0.5)

    def test_adx_appears_only_in_winner(self, populated_db):
        """``adx_5m_adx`` was only on EEE → 100% win-rate."""
        from paper.journal import win_rate_by_indicator
        df = win_rate_by_indicator()
        adx = df[(df.indicator == "adx_adx") & (df.timeframe == "5m")]
        assert len(adx) == 1
        assert adx["n_trades"].iloc[0] == 1
        assert adx["win_rate"].iloc[0] == pytest.approx(1.0)

    def test_empty_db_returns_empty_frame(self, tmp_db):
        from paper.journal import win_rate_by_indicator
        df = win_rate_by_indicator()
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Journal CLI: python -m paper.journal report
# ---------------------------------------------------------------------------

class TestJournalCLI:
    def test_report_runs(self, populated_db, capsys):
        from paper.journal import _cli
        assert _cli(["report"]) == 0
        out = capsys.readouterr().out
        assert "Paper Journal Report" in out
        assert "OPEN TRADES (2)" in out
        # Default --since=today → "today's closes" + the seeded ones all
        # closed today, so CLOSED TRADES (3).
        assert "CLOSED TRADES (3)" in out

    def test_report_since_far_past(self, populated_db, capsys):
        from paper.journal import _cli
        assert _cli(["report", "--since", "2000-01-01"]) == 0
        out = capsys.readouterr().out
        assert "CLOSED TRADES (3)" in out

    def test_report_since_future(self, populated_db, capsys):
        from paper.journal import _cli
        assert _cli(["report", "--since", "2099-01-01"]) == 0
        out = capsys.readouterr().out
        assert "CLOSED TRADES (0)" in out

    def test_report_includes_summary(self, populated_db, capsys):
        from paper.journal import _cli
        _cli(["report"])
        out = capsys.readouterr().out
        assert "Today" in out
        assert "profit factor" in out
        assert "win rate" in out

    def test_empty_db_report_runs(self, tmp_db, capsys):
        """No trades, no rows — must not crash."""
        from paper.journal import _cli
        assert _cli(["report"]) == 0
        out = capsys.readouterr().out
        assert "OPEN TRADES (0)" in out
        assert "(no trades)" in out


# ---------------------------------------------------------------------------
# Monitor: per-tick rule evaluation (sync) and async loop shutdown
# ---------------------------------------------------------------------------

class TestMonitorOnce:
    """Drive ``_monitor_once`` directly with stubbed get_partial and
    market_open. Covers the five spec-mandated cases (LONG-SL,
    SHORT-TP1, costs applied, multi-symbol, TIMEOUT) plus the
    SL+TP-same-bar tie-break edge case."""

    def test_long_sl_hit(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import _monitor_once, open_trade

        trade_id = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        bars = {"NSE:TEST-EQ": FakeBar(high=99, low=97, close=98.5)}
        _monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            status, exit_price, pnl_net = conn.execute(
                "SELECT status, exit_price, pnl_net FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "SL"
        assert exit_price == pytest.approx(98.0)
        # LONG sl loss: gross = (98-100)*10 = -20; net is more negative.
        assert pnl_net < -20.0

    def test_short_tp1_hit(self, tmp_db, monkeypatch):
        from bot.config import settings
        from paper.schema import connect
        from paper.tracker import _monitor_once, open_trade

        # Legacy full-close-at-TP1 semantics — partial+trail tested separately.
        monkeypatch.setattr(settings, "trailing_stop_enabled", False)
        trade_id = open_trade("TEST.NS", "SHORT", 100, 102, 96, None, 10, 0.7, {})
        bars = {"NSE:TEST-EQ": FakeBar(high=101, low=95, close=96.5)}
        _monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            status, exit_price, pnl_gross, pnl_net = conn.execute(
                "SELECT status, exit_price, pnl_gross, pnl_net "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TP1"
        assert exit_price == pytest.approx(96.0)
        # SHORT winner: gross = (100-96)*10 = 40; net is gross minus costs.
        assert pnl_gross == pytest.approx(40.0)
        assert 0 < pnl_net < pnl_gross

    def test_sl_and_tp_same_bar_takes_tp(self, tmp_db, monkeypatch):
        from bot.config import settings
        from paper.schema import connect
        from paper.tracker import _monitor_once, open_trade

        monkeypatch.setattr(settings, "trailing_stop_enabled", False)
        trade_id = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        # Both SL (low<=98) and TP1 (high>=104) hit on same bar.
        bars = {"NSE:TEST-EQ": FakeBar(high=105, low=97, close=104.5)}
        _monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            status, exit_price, notes = conn.execute(
                "SELECT status, exit_price, notes FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TP1"
        assert exit_price == pytest.approx(104.0)
        assert notes is not None
        assert "SL+TP1 same bar" in notes

    def test_two_trades_concurrent(self, tmp_db, monkeypatch):
        """Independent trades on different symbols resolve independently.
        ``_is_timed_out`` pinned to False so the test isn't wall-clock
        sensitive under the same-day-TIMEOUT rule."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        id_a = tr.open_trade("AAA.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        id_b = tr.open_trade("BBB.NS", "LONG", 200, 196, 208, None, 5, 0.7, {})
        bars = {
            "NSE:AAA-EQ": FakeBar(high=99, low=97, close=98),     # SL hit
            "NSE:BBB-EQ": FakeBar(high=204, low=199, close=201),  # in-range
        }
        tr._monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            rows = dict(conn.execute(
                "SELECT id, status FROM paper_trades"
            ).fetchall())
        assert rows[id_a] == "SL"
        assert rows[id_b] == "OPEN"

    def test_no_fill_keeps_open(self, tmp_db, monkeypatch):
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        bars = {"NSE:TEST-EQ": FakeBar(high=103, low=99, close=101)}
        tr._monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "OPEN"

    def test_market_closed_skips_eval(self, tmp_db, monkeypatch):
        """When market is closed, SL/TP aren't evaluated even if the
        bar would have hit — protects against stale-feed false fills."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        bars = {"NSE:TEST-EQ": FakeBar(high=110, low=97, close=105)}
        tr._monitor_once(lambda sym: bars.get(sym), lambda: False)

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "OPEN"

    def test_none_bar_keeps_open(self, tmp_db, monkeypatch):
        """Missing LTP during market hours just means 'no fill yet'."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        tr._monitor_once(lambda sym: None, lambda: True)

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "OPEN"

    def test_tp_hit_in_completed_bar_between_polls(self, tmp_db, monkeypatch):
        """Regression for the ASIANPAINT bug: a TP hit landed inside a
        5-min bar that completed between two monitor ticks, then the
        fresh partial of the next bar opened below TP. Without the
        completed-bar replay the monitor saw only the fresh partial
        and the fill was lost.

        Setup: trade opened at 100, TP1 at 104. A completed bar shows
        high=105 (TP1 hit). The current partial is below TP (high=103).
        With the fix, the completed bar's high triggers TP1."""
        import paper.tracker as tr
        from bot.config import settings
        from paper.schema import connect

        monkeypatch.setattr(settings, "trailing_stop_enabled", False)
        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # Completed bar between entry and now: high=105 → hits TP1=104.
        completed = [FakeBar(high=105, low=99, close=103.5)]
        partial = FakeBar(high=103, low=101, close=102.5)  # below TP1
        tr._monitor_once(
            lambda sym: partial,
            lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            status, exit_price = conn.execute(
                "SELECT status, exit_price FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TP1"
        assert exit_price == pytest.approx(104.0)

    def test_completed_bar_sl_takes_precedence_over_partial_tp(self, tmp_db, monkeypatch):
        """Order matters: when a completed bar BEFORE the current
        partial hit the stop, the SL exit wins even if the partial
        would now be hitting TP. We close at the SL price recorded
        from the earlier bar — not the current partial's level."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # First completed bar hits SL (low=97 ≤ 98), second would hit TP.
        completed = [
            FakeBar(high=99, low=97, close=98.2),       # SL bar
            FakeBar(high=105, low=99, close=104.5),     # would TP
        ]
        partial = FakeBar(high=106, low=104, close=105.0)
        tr._monitor_once(
            lambda sym: partial,
            lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            status, exit_price = conn.execute(
                "SELECT status, exit_price FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "SL"
        assert exit_price == pytest.approx(98.0)

    def test_monitor_translates_yf_symbol_to_fyers_for_bars(
        self, tmp_db, monkeypatch,
    ):
        """Regression for the prod bug (the ASIANPAINT case): bars_5m
        and the live ``_current`` dict are keyed by Fyers symbols
        (``NSE:ASIANPAINT-EQ``) while paper_trades stores yfinance
        symbols (``ASIANPAINT.NS``). The monitor must convert before
        hitting the feed, otherwise every SL/TP lookup misses and
        trades only ever exit via TIMEOUT."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "ASIANPAINT.NS", "LONG", 2582, 2566, 2604, 2621, 1, 0.7, {},
        )

        bars_calls: list[str] = []

        def fake_bars(sym, since):
            bars_calls.append(sym)
            return [FakeBar(high=2625, low=2580, close=2622)]

        tr._monitor_once(lambda sym: None, lambda: True, fake_bars)

        # Feed-side lookup must use the Fyers form.
        assert bars_calls == ["NSE:ASIANPAINT-EQ"]
        # TP2 fill landed.
        with connect(tmp_db) as conn:
            status, exit_price = conn.execute(
                "SELECT status, exit_price FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TP2"
        assert exit_price == pytest.approx(2621.0)

    def test_monitor_translates_yf_symbol_to_fyers_for_partial(
        self, tmp_db, monkeypatch,
    ):
        """Same translation for the current-partial path. Uses a
        trade whose completed bars don't hit so the partial check
        actually runs."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "ASIANPAINT.NS", "LONG", 2582, 2566, 2604, 2621, 1, 0.7, {},
        )

        partial_calls: list[str] = []

        def fake_partial(sym):
            partial_calls.append(sym)
            return FakeBar(high=2625, low=2580, close=2622)  # hits TP2

        tr._monitor_once(fake_partial, lambda: True, lambda sym, since: [])

        assert partial_calls == ["NSE:ASIANPAINT-EQ"]
        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "TP2"

    def test_completed_bar_replay_optional(self, tmp_db, monkeypatch):
        """The bar-replay arg is optional — callers that don't supply
        ``get_bars_since`` (the existing TestMonitorOnce cases) keep
        the old partial-only behaviour."""
        import paper.tracker as tr
        from bot.config import settings
        from paper.schema import connect

        monkeypatch.setattr(settings, "trailing_stop_enabled", False)
        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        partial = FakeBar(high=105, low=99, close=104.5)  # hits TP1
        tr._monitor_once(lambda sym: partial, lambda: True)  # no third arg

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "TP1"


class TestPartialAndTrailing:
    """Phase-7b: 50% partial at TP1 + trailing stop on the runner.

    Conventions used across these tests:
      * trade qty=10, entry=100, SL=98, TP1=104, TP2=None
      * trail_distance = entry - SL = 2.0 (default mult 1.0)
      * trailing stop floor = entry (100) — runner can't lose
    """

    def test_partial_books_half_at_tp1_and_runner_stays_open(
        self, tmp_db, monkeypatch,
    ):
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # Single completed bar hits TP1 (high=104.5).
        completed = [FakeBar(high=104.5, low=101, close=103.8)]
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, tp1_filled, tp1_exit_price, tp1_qty, "
                "       runner_qty, trailing_stop, running_high "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, tp1_filled, tp1_price, tp1_qty, runner, trail, rhigh = row
        assert status == "OPEN"
        assert tp1_filled == 1
        assert tp1_price == pytest.approx(104.0)
        assert tp1_qty == 5
        assert runner == 5
        # trail floor = entry = 100, candidate = running_high(104) - 2 = 102
        # So trail = max(100, 102) = 102.
        assert trail == pytest.approx(102.0)
        assert rhigh == pytest.approx(104.0)

    def test_runner_closes_on_trailing_stop_hit(self, tmp_db, monkeypatch):
        """After TP1 partial, a later completed bar dips to the trail
        line → runner exits at the trail price, status='TRAIL'."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # First tick fires TP1 partial; second tick provides a bar
        # that knocks the trail.
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=104.5, low=101, close=103.8)],
        )
        # After partial: trail=102, runner=5 open. Next bar low<=102
        # closes the runner.
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=103.5, low=101.5, close=102.0)],
        )

        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, exit_price, pnl_gross, pnl_net, "
                "       tp1_pnl_gross "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, exit_price, gross, net, tp1_gross = row
        assert status == "TRAIL"
        assert exit_price == pytest.approx(102.0)
        # Total gross = TP1 partial (5 × 4 = 20) + runner (5 × 2 = 10) = 30
        assert tp1_gross == pytest.approx(20.0)
        assert gross == pytest.approx(30.0)
        # Net = gross − round-trip costs (≥ 0 since both legs profitable)
        assert 0 < net < gross

    def test_trail_ratchets_up_on_higher_bars(self, tmp_db, monkeypatch):
        """Each bar's high above (running_high − trail_distance) pushes
        the trail up; the trail never moves backward."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # TP1 + a bar with even higher high in one tick.
        completed = [
            FakeBar(high=104.5, low=101, close=103.8),  # TP1 partial
            FakeBar(high=107.0, low=103, close=106.5),  # pushes trail to 105
            FakeBar(high=106.0, low=105.5, close=105.8),  # stays above trail
        ]
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, running_high, trailing_stop "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, rhigh, trail = row
        assert status == "OPEN"  # runner still alive — no bar dipped to trail
        assert rhigh == pytest.approx(107.0)
        # trail = max(100, 107 - 2) = 105
        assert trail == pytest.approx(105.0)

    def test_runner_floor_at_entry_protects_against_loss(
        self, tmp_db, monkeypatch,
    ):
        """If the runner has only just transitioned to TP1 and price
        immediately retraces, the trailing stop cannot go below
        entry — the runner exits at breakeven, not below."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        # TP1 only 1pt above entry so running_high = TP1 = 101.
        # trail = max(100, 101 - 2) = 100 (floor at entry).
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 101, None, 10, 0.7, {},
        )
        completed = [
            FakeBar(high=101.0, low=100.5, close=100.8),  # TP1 partial
            FakeBar(high=100.5, low=99.5, close=99.7),    # touches trail floor
        ]
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            status, exit_price = conn.execute(
                "SELECT status, exit_price FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TRAIL"
        # Runner closes at entry (breakeven), NOT at the 99.5 low.
        assert exit_price == pytest.approx(100.0)

    def test_tp2_ignored_in_runner_phase(self, tmp_db, monkeypatch):
        """Phase-7b spec: 'will be booked at trailing stop loss hit
        only'. A runner price beyond TP2 must NOT exit — only the
        trailing stop or TIMEOUT do."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, 108, 10, 0.7, {},
        )
        # TP1 partial then a bar shooting THROUGH TP2 — must NOT close.
        completed = [
            FakeBar(high=104.5, low=101, close=103.8),
            FakeBar(high=110.0, low=107, close=109.0),  # past TP2=108
        ]
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: completed,
        )

        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, trailing_stop, running_high "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, trail, rhigh = row
        # Runner still OPEN; trail ratcheted to 108.
        assert status == "OPEN"
        assert rhigh == pytest.approx(110.0)
        assert trail == pytest.approx(108.0)

    def test_partial_disabled_falls_back_to_full_tp1_close(
        self, tmp_db, monkeypatch,
    ):
        """When ``trailing_stop_enabled=False``, TP1 closes the full
        position as before — back-compat path."""
        import paper.tracker as tr
        from bot.config import settings
        from paper.schema import connect

        monkeypatch.setattr(settings, "trailing_stop_enabled", False)
        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=105, low=99, close=104.5)],
        )

        with connect(tmp_db) as conn:
            status, tp1_filled = conn.execute(
                "SELECT status, tp1_filled FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TP1"
        assert tp1_filled == 0  # no partial recorded

    def test_short_partial_then_trail(self, tmp_db, monkeypatch):
        """SHORT mirror: entry 100, SL 102, TP1 96. trail_distance =
        SL - entry = 2. After TP1 partial, running_low = 96, trail =
        min(100, 96 + 2) = 98. A later bar that ticks UP to 98 stops
        the runner out at 98."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "SHORT", 100, 102, 96, None, 10, 0.7, {},
        )
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=99, low=95.5, close=96.5)],
        )
        with connect(tmp_db) as conn:
            tp1_filled, trail, rlow = conn.execute(
                "SELECT tp1_filled, trailing_stop, running_low "
                "FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()
        assert tp1_filled == 1
        assert rlow == pytest.approx(96.0)
        assert trail == pytest.approx(98.0)

        # Runner bar ticks back UP to 98 → trail hit, runner closes.
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=98.5, low=97, close=97.8)],
        )
        with connect(tmp_db) as conn:
            status, exit_price = conn.execute(
                "SELECT status, exit_price FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TRAIL"
        assert exit_price == pytest.approx(98.0)

    def test_runner_timeout_closes_at_ltp(self, tmp_db, monkeypatch):
        """A runner that has not hit its trail by 15:30 IST should
        TIMEOUT at the partial bar's close. Total P&L = TP1 partial +
        runner-at-LTP."""
        import paper.tracker as tr
        from paper.schema import connect

        # TP1 partial first, then force TIMEOUT.
        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: False)
        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        tr._monitor_once(
            lambda sym: None, lambda: True,
            lambda sym, since: [FakeBar(high=104.5, low=101, close=103.8)],
        )
        # Now flip TIMEOUT on and feed a partial bar with close=105.
        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: True)
        tr._monitor_once(
            lambda sym: FakeBar(high=105, low=104, close=105.0),
            lambda: True,
            lambda sym, since: [],
        )

        with connect(tmp_db) as conn:
            status, exit_price, gross = conn.execute(
                "SELECT status, exit_price, pnl_gross "
                "FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()
        assert status == "TIMEOUT"
        assert exit_price == pytest.approx(105.0)
        # Total gross = 5 × 4 (TP1) + 5 × 5 (runner at 105) = 45
        assert gross == pytest.approx(45.0)


class TestMonitorTimeout:
    """TIMEOUT specifically — uses monkeypatch on ``_is_timed_out``
    to avoid having to fabricate "next-session-close" wall-clock."""

    def test_timeout_closes_at_last_close(self, tmp_db, monkeypatch):
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: True)

        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        bars = {"NSE:TEST-EQ": FakeBar(high=100.5, low=99.5, close=100.2)}
        tr._monitor_once(lambda sym: bars.get(sym), lambda: True)

        with connect(tmp_db) as conn:
            status, exit_price, pnl_gross = conn.execute(
                "SELECT status, exit_price, pnl_gross "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert status == "TIMEOUT"
        assert exit_price == pytest.approx(100.2)
        assert pnl_gross == pytest.approx((100.2 - 100) * 10)

    def test_timeout_no_ltp_nulls_pnl(self, tmp_db, monkeypatch):
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: True)

        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        tr._monitor_once(lambda sym: None, lambda: True)

        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT status, exit_price, pnl_gross, pnl_net, notes "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        status, exit_price, gross, net, notes = row
        assert status == "TIMEOUT"
        assert exit_price is None
        assert gross is None
        assert net is None
        assert "no LTP" in (notes or "")

    def test_timeout_fires_even_with_market_closed(self, tmp_db, monkeypatch):
        """TIMEOUT is wall-clock — must fire regardless of market hours."""
        import paper.tracker as tr
        from paper.schema import connect

        monkeypatch.setattr(tr, "_is_timed_out", lambda *a, **kw: True)

        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        # market_open=False AND bar=None — still must close as TIMEOUT.
        tr._monitor_once(lambda sym: None, lambda: False)

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "TIMEOUT"


class TestMonitorAsyncLoop:
    """One coverage test for the actual async ``monitor()`` coroutine
    — proves the stop_event handshake works end-to-end and that a
    real tick reaches ``_monitor_once``. Uses ``asyncio.run`` so the
    test stays sync and doesn't pull in pytest-asyncio."""

    def test_one_tick_then_stops(self, tmp_db, monkeypatch):
        import asyncio

        import paper.tracker as tr
        from paper.schema import connect

        trade_id = tr.open_trade(
            "TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {},
        )
        monkeypatch.setattr("bot.schedule.is_market_open", lambda: True)
        monkeypatch.setattr(
            "data.realtime_feed.get_current_partial",
            lambda sym: FakeBar(high=99, low=97, close=98),  # SL hit
        )
        monkeypatch.setattr(tr, "POLL_INTERVAL_SECONDS", 0.05)

        async def runner():
            stop = asyncio.Event()
            task = asyncio.create_task(tr.monitor(stop))
            await asyncio.sleep(0.02)   # give monitor one tick
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)

        asyncio.run(runner())

        with connect(tmp_db) as conn:
            status = conn.execute(
                "SELECT status FROM paper_trades WHERE id = ?", (trade_id,),
            ).fetchone()[0]
        assert status == "SL"

    def test_immediate_stop_exits_clean(self, tmp_db, monkeypatch):
        """Stop event set BEFORE the loop even runs — coroutine must
        still return (not hang)."""
        import asyncio

        import paper.tracker as tr

        monkeypatch.setattr("bot.schedule.is_market_open", lambda: True)
        monkeypatch.setattr(
            "data.realtime_feed.get_current_partial", lambda sym: None,
        )

        async def runner():
            stop = asyncio.Event()
            stop.set()
            await asyncio.wait_for(tr.monitor(stop), timeout=1.0)

        asyncio.run(runner())


# ---------------------------------------------------------------------------
# SL/TP derivation + from_signal glue
# ---------------------------------------------------------------------------

class TestDeriveSlTp:
    def test_uses_atr_when_available(self):
        from paper.tracker import derive_sl_tp
        sl, tp1, tp2 = derive_sl_tp(price=100.0, atr=2.0)
        # SL = 100 - 1.5*2 = 97.00; TP1 = 100 + 2.0*2 = 104.00;
        # TP2 = 100 + 3.5*2 = 107.00
        assert sl == 97.0
        assert tp1 == 104.0
        assert tp2 == 107.0

    def test_falls_back_when_atr_none(self):
        from paper.tracker import derive_sl_tp
        # Fallback: ATR ≈ 1.5% of price = 1.5
        # SL = 100 - 1.5*1.5 = 97.75; TP1 = 103.0; TP2 = 105.25
        sl, tp1, tp2 = derive_sl_tp(price=100.0, atr=None)
        assert sl == 97.75
        assert tp1 == 103.0
        assert tp2 == 105.25

    def test_falls_back_when_atr_zero(self):
        from paper.tracker import derive_sl_tp
        sl, _, _ = derive_sl_tp(price=100.0, atr=0.0)
        assert sl == 97.75

    def test_falls_back_when_atr_negative(self):
        """Defensive: negative ATR shouldn't be possible but if it
        slips through, treat as missing."""
        from paper.tracker import derive_sl_tp
        sl, _, _ = derive_sl_tp(price=100.0, atr=-1.0)
        assert sl == 97.75

    def test_values_are_rounded_to_paise(self):
        from paper.tracker import derive_sl_tp
        sl, tp1, tp2 = derive_sl_tp(price=123.456, atr=1.234)
        # 2dp rounding for all three
        assert sl == round(sl, 2)
        assert tp1 == round(tp1, 2)
        assert tp2 == round(tp2, 2)


class TestFromSignal:
    """Glue between StockSignals and open_trade. Uses duck-typed
    stub objects so the test doesn't have to construct a full
    StockSignals (which would pull in bot.scoring + its deps).

    The late-session cutoff is disabled class-wide via the autouse
    fixture so these tests stay deterministic regardless of wall-
    clock. The cutoff behavior itself is covered by
    ``TestPastEntryCutoff``."""

    @pytest.fixture(autouse=True)
    def _bypass_cutoff(self, monkeypatch):
        import paper.tracker as tr
        monkeypatch.setattr(tr, "_past_entry_cutoff", lambda *a, **kw: False)

    @staticmethod
    def _stub(**overrides):
        class Stub:
            pass
        s = Stub()
        s.symbol = "TEST.NS"
        s.price = 100.0
        s.sl = 97.0
        s.tp1 = 104.0
        s.tp2 = 107.0
        s.score = 70
        s.snapshot = {"rsi_5m": 65.0}
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_opens_trade(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import from_signal
        trade_id = from_signal(self._stub())
        assert trade_id is not None
        with connect(tmp_db) as conn:
            row = conn.execute(
                "SELECT symbol, side, entry_price, stop_loss, target_1, "
                "       target_2, confidence "
                "FROM paper_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
        assert row == ("TEST.NS", "LONG", 100.0, 97.0, 104.0, 107.0, 0.7)

    def test_persists_indicator_snapshot(self, tmp_db):
        from paper.schema import connect
        from paper.tracker import from_signal
        trade_id = from_signal(self._stub(
            snapshot={"rsi_5m": 65.0, "atr_5m": 2.0, "macd_5m_histogram": 0.4},
        ))
        with connect(tmp_db) as conn:
            rows = conn.execute(
                "SELECT indicator, value, timeframe FROM signal_indicators "
                "WHERE paper_trade_id = ?",
                (trade_id,),
            ).fetchall()
        as_dict = {(ind, tf): v for ind, v, tf in rows}
        assert as_dict[("rsi", "5m")] == 65.0
        assert as_dict[("atr", "5m")] == 2.0
        assert as_dict[("macd_histogram", "5m")] == 0.4

    def test_missing_sl_skips(self, tmp_db):
        from paper.tracker import from_signal
        assert from_signal(self._stub(sl=None)) is None

    def test_missing_tp1_skips(self, tmp_db):
        from paper.tracker import from_signal
        assert from_signal(self._stub(tp1=None)) is None

    def test_zero_risk_skips(self, tmp_db):
        """If entry == sl, risk_per_share=0 → sizing returns qty=0 →
        from_signal returns None without inserting."""
        from paper.schema import connect
        from paper.tracker import from_signal
        assert from_signal(self._stub(sl=100.0)) is None
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades"
            ).fetchone()[0]
        assert count == 0

    def test_dup_returns_existing_id(self, tmp_db):
        from paper.tracker import from_signal
        id1 = from_signal(self._stub())
        id2 = from_signal(self._stub(price=101.0, sl=98.0, tp1=105.0))
        assert id1 == id2

    def test_rejects_after_late_session_cutoff(self, tmp_db, monkeypatch):
        """No new paper trades after 14:30 IST — too little room
        before same-day TIMEOUT for an MIS setup to develop."""
        from paper.schema import connect
        import paper.tracker as tr

        monkeypatch.setattr(tr, "_past_entry_cutoff", lambda: True)
        assert tr.from_signal(self._stub()) is None
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades"
            ).fetchone()[0]
        assert count == 0

    def test_allows_before_cutoff(self, tmp_db, monkeypatch):
        """Sanity check: when cutoff says False, trade opens normally."""
        import paper.tracker as tr

        monkeypatch.setattr(tr, "_past_entry_cutoff", lambda: False)
        assert tr.from_signal(self._stub()) is not None


class TestPastEntryCutoff:
    """Pure predicate for the 14:30 IST entry cutoff."""

    def test_before_cutoff_false(self):
        from paper.tracker import IST, _past_entry_cutoff
        # 14:00 IST is comfortably before 14:30.
        now = pd.Timestamp("2026-05-12T14:00:00", tz=IST)
        assert _past_entry_cutoff(now) is False

    def test_at_cutoff_true(self):
        """14:30:00 IST exactly — predicate fires (no entries from
        the cutoff minute onwards)."""
        from paper.tracker import IST, _past_entry_cutoff
        now = pd.Timestamp("2026-05-12T14:30:00", tz=IST)
        assert _past_entry_cutoff(now) is True

    def test_after_cutoff_true(self):
        from paper.tracker import IST, _past_entry_cutoff
        now = pd.Timestamp("2026-05-12T14:31:00", tz=IST)
        assert _past_entry_cutoff(now) is True

    def test_default_uses_wall_clock(self):
        """``now_ist=None`` falls through to ``pd.Timestamp.now(tz=IST)``.
        Don't assert the boolean (depends on when the test runs) — just
        confirm it executes without error and returns a bool."""
        from paper.tracker import _past_entry_cutoff
        assert isinstance(_past_entry_cutoff(), bool)


# ---------------------------------------------------------------------------
# Phase-5b end-of-session digest
# ---------------------------------------------------------------------------

class TestEODDigest:
    def test_empty_db_returns_none(self, tmp_db):
        """Quiet day → no digest. Caller skips the Telegram send."""
        from paper.journal import build_eod_digest
        assert build_eod_digest() is None

    def test_only_open_trades_returns_none(self, tmp_db):
        """Day with paper-trade rows but none closed → still None.
        The digest summarises closed trades only."""
        from paper.journal import build_eod_digest
        from paper.tracker import open_trade
        open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        assert build_eod_digest() is None

    def test_renders_summary_with_trades(self, tmp_db):
        from paper.journal import build_eod_digest
        from paper.tracker import IST, _close, open_trade
        # 2 winners + 1 loser closed today.
        id1 = open_trade("AAA.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        _close(id1, "TP1", 104.0)
        id2 = open_trade("BBB.NS", "LONG", 200, 196, 208, None, 5, 0.7, {})
        _close(id2, "TP1", 208.0)
        id3 = open_trade("CCC.NS", "LONG", 50, 48, 54, None, 20, 0.7, {})
        _close(id3, "SL", 48.0)

        today = pd.Timestamp.now(tz=IST).date()
        msg = build_eod_digest(today)
        assert msg is not None
        assert "EOD Digest" in msg
        assert "Closed: 3" in msg
        # Win rate 2/3 ≈ 67%
        assert "67%" in msg or "66%" in msg
        # Top winners header + top 3
        assert "Top winners" in msg
        # winner symbols appear (without .NS suffix)
        assert "AAA" in msg
        assert "BBB" in msg

    def test_top_winners_capped_at_three(self, tmp_db):
        """Even with 5 winners, only the top 3 by pnl_net are listed."""
        from paper.journal import build_eod_digest
        from paper.tracker import _close, open_trade
        for i, gain in enumerate([100, 50, 200, 30, 80]):
            tid = open_trade(
                f"SYM{i}.NS", "LONG",
                entry=100.0, sl=98.0, tp1=100.0 + gain / 10, tp2=None,
                qty=10, confidence=0.7, indicator_snapshot={},
            )
            _close(tid, "TP1", 100.0 + gain / 10)
        msg = build_eod_digest()
        assert msg is not None
        # 5 winners total but only 3 lines under "Top winners".
        bullets = [line for line in msg.split("\n") if line.startswith("  • ")]
        assert len(bullets) == 3

    def test_open_trades_surfaces_warning(self, tmp_db):
        """If trades are still OPEN at digest time, surface as a
        ``monitor may be lagging`` warning so the operator notices."""
        from paper.journal import build_eod_digest
        from paper.tracker import _close, open_trade
        id1 = open_trade("DONE.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        _close(id1, "TP1", 104.0)  # closed
        open_trade("STUCK.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        # ↑ left open; digest must flag it
        msg = build_eod_digest()
        assert msg is not None
        assert "still OPEN" in msg
        assert "monitor may be lagging" in msg

    def test_html_safe_ampersand(self, tmp_db):
        """``Net P&L`` must be HTML-escaped (``P&amp;L``) so Telegram
        doesn't choke on the raw ampersand under parse_mode='HTML'."""
        from paper.journal import build_eod_digest
        from paper.tracker import _close, open_trade
        tid = open_trade("X.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        _close(tid, "TP1", 104.0)
        msg = build_eod_digest()
        assert "P&amp;L" in msg
        # Should not appear as bare ampersand outside the entity.
        # (Telegram tolerates &amp; but rejects bare & in HTML mode.)
        between = msg.replace("P&amp;L", "")
        assert "&" not in between


# ---------------------------------------------------------------------------
# Phase-5b dispatch gating: Telegram fires only at score >= telegram_threshold
# ---------------------------------------------------------------------------

class TestDispatchGating:
    """Verify the score-band routing in ``bot.scanner._dispatch``:
       paper trade for composite_threshold..telegram_threshold,
       paper trade + Telegram for >= telegram_threshold."""

    @staticmethod
    def _signals(score: int, confidence: float | None = None):
        """Build a Phase-6-ready StockSignals. Default ``confidence``
        is ``score/100`` (no filter penalty) — most existing tests
        want this. Pass explicit ``confidence`` to test the filtered
        gating cases."""
        from bot.scoring import StockSignals
        if confidence is None:
            confidence = score / 100.0
        return StockSignals(
            symbol="TEST.NS", price=100.0, rsi=65.0, volume_ratio=2.0,
            above_vwap=True, breakout=False, pct_from_high=-2.0,
            score=score, reasons=["VR 2.0x"],
            sl=97.0, tp1=104.0, tp2=110.0,
            confidence=confidence,
        )

    def test_telegram_send_above_threshold(self, tmp_db, monkeypatch):
        import asyncio

        from bot.scanner import _dispatch

        sent = []

        class FakeTelegram:
            async def send(self, text):
                sent.append(text)

        monkeypatch.setattr("bot.scanner.settings.telegram_threshold", 80)
        # Avoid the late-session cutoff so the paper trade gets opened.
        monkeypatch.setattr(
            "paper.tracker._past_entry_cutoff", lambda *a, **kw: False,
        )
        asyncio.run(_dispatch(FakeTelegram(), self._signals(score=85)))
        assert len(sent) == 1, "high-score alert should fire Telegram"

    def test_telegram_silent_below_threshold(self, tmp_db, monkeypatch):
        import asyncio

        from bot.scanner import _dispatch
        from paper.schema import connect

        sent = []

        class FakeTelegram:
            async def send(self, text):
                sent.append(text)

        monkeypatch.setattr("bot.scanner.settings.telegram_threshold", 80)
        monkeypatch.setattr(
            "paper.tracker._past_entry_cutoff", lambda *a, **kw: False,
        )
        asyncio.run(_dispatch(FakeTelegram(), self._signals(score=70)))
        assert sent == [], "mid-band alert must NOT fire Telegram"
        # ...but it MUST have opened a paper trade.
        with connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE symbol='TEST.NS'"
            ).fetchone()[0]
        assert count == 1, "mid-band alert must still open paper trade"

    def test_record_alert_runs_for_both_bands(self, tmp_db, monkeypatch):
        """The cooldown row (alerts_sent) must be written either way so
        a back-to-back periodic + bar-event firing can't open two
        paper trades on the same symbol."""
        import asyncio
        import sqlite3

        from bot.scanner import _dispatch

        class FakeTelegram:
            async def send(self, text):
                pass

        monkeypatch.setattr("bot.scanner.settings.telegram_threshold", 80)
        monkeypatch.setattr(
            "paper.tracker._past_entry_cutoff", lambda *a, **kw: False,
        )
        asyncio.run(_dispatch(FakeTelegram(), self._signals(score=70)))
        with sqlite3.connect(tmp_db) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM alerts_sent WHERE symbol='TEST.NS'"
            ).fetchone()[0]
        assert n == 1


# ---------------------------------------------------------------------------
# format_alert: SL/TP rendering (Phase-5 addition)
# ---------------------------------------------------------------------------

class TestFormatAlertSLTP:
    def test_renders_sl_tp_tp2(self):
        from bot.notifier import format_alert
        from bot.scoring import StockSignals
        s = StockSignals(
            symbol="TEST.NS", price=100.0, rsi=65.0, volume_ratio=2.0,
            above_vwap=True, breakout=False, pct_from_high=-2.0,
            score=75, reasons=["VR 2.0x", "> VWAP"],
            sl=97.0, tp1=104.0, tp2=107.0,
        )
        text = format_alert(s)
        assert "SL ₹97.00" in text
        assert "TP ₹104.00" in text
        assert "TP2 ₹107.00" in text

    def test_renders_sl_tp_without_tp2(self):
        from bot.notifier import format_alert
        from bot.scoring import StockSignals
        s = StockSignals(
            symbol="TEST.NS", price=100.0, rsi=65.0, volume_ratio=2.0,
            above_vwap=True, breakout=False, pct_from_high=-2.0,
            score=75, reasons=["VR 2.0x"],
            sl=97.0, tp1=104.0, tp2=None,
        )
        text = format_alert(s)
        assert "SL ₹97.00" in text
        assert "TP ₹104.00" in text
        assert "TP2" not in text

    def test_no_sl_no_line(self):
        """Pre-Phase-5 callers that don't set sl/tp don't get the line."""
        from bot.notifier import format_alert
        from bot.scoring import StockSignals
        s = StockSignals(
            symbol="TEST.NS", price=100.0, rsi=65.0, volume_ratio=2.0,
            above_vwap=True, breakout=False, pct_from_high=-2.0,
            score=75, reasons=["VR 2.0x"],
        )
        text = format_alert(s)
        assert "🎯" not in text


# ---------------------------------------------------------------------------
# Phase-5b paper-open suppression: re-alerts blocked while paper trade is OPEN
# ---------------------------------------------------------------------------

class TestPaperOpenSuppression:
    """An OPEN paper trade on a symbol must block fresh Telegram
    re-alerts until that trade closes — otherwise the 60-min cooldown
    expires while the trade is still active, producing alerts that
    have no journal counterpart (silently deduped by open_trade)."""

    def test_open_paper_trade_suppresses(self, tmp_db):
        from bot.suppression.rules import is_suppressed
        from paper.tracker import open_trade

        open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        blocked, reason = is_suppressed("TEST.NS", cooldown_minutes=60)
        assert blocked
        assert "paper" in reason.lower()

    def test_closed_paper_trade_does_not_suppress(self, tmp_db):
        """Once the trade closes (SL/TP/TIMEOUT/MANUAL), the suppression
        ends — a new alert later in the day can fire again."""
        from bot.suppression.rules import is_suppressed
        from paper.tracker import _close, open_trade

        tid = open_trade("TEST.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        _close(tid, "TP1", 104.0)
        blocked, reason = is_suppressed("TEST.NS", cooldown_minutes=60)
        # No alerts_sent row was written (we didn't go through _dispatch),
        # so cooldown shouldn't fire either. Pure paper-open test.
        assert not blocked, f"expected no suppression, got: {reason!r}"

    def test_open_on_other_symbol_does_not_suppress(self, tmp_db):
        from bot.suppression.rules import is_suppressed
        from paper.tracker import open_trade

        open_trade("AAA.NS", "LONG", 100, 98, 104, None, 10, 0.7, {})
        blocked, reason = is_suppressed("BBB.NS", cooldown_minutes=60)
        assert not blocked

    def test_no_trade_history_does_not_suppress(self, tmp_db):
        """Fresh DB — symbol never seen — no suppression."""
        from bot.suppression.rules import is_suppressed
        blocked, reason = is_suppressed("UNTOUCHED.NS", cooldown_minutes=60)
        assert not blocked
        assert reason == ""
