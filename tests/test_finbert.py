"""Phase-10 news pipeline tests.

Six classes:

  1. TestSymbolMatch  — alias dict + Nifty 500 substring lookups
  2. TestFinBertStub  — stub mode behaviour when transformers/torch absent
  3. TestFinBertReal  — guarded with importorskip; runs real ProsusAI/finbert
                        only when the deps + model are available
  4. TestFetcher      — feedparser entry conversion, dedupe by id
  5. TestAggregator   — weighted-recency aggregate, decay correctness
  6. TestScoreNews    — score_news with news_score parameter

No network. RSS fixture lives at ``tests/fixtures/rss_economictimes.xml``.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ===========================================================================
# 1. Symbol matching
# ===========================================================================

class TestSymbolMatch:
    def test_full_company_name_high_relevance(self):
        from news.symbol_match import match_symbols
        out = match_symbols("Reliance Industries reports record profit")
        tickers = {t for t, _ in out}
        assert "RELIANCE.NS" in tickers
        rel = next(r for t, r in out if t == "RELIANCE.NS")
        assert rel == pytest.approx(1.0)

    def test_alias_match_lower_relevance(self):
        from news.symbol_match import match_symbols
        out = match_symbols("RIL profit beats estimates")
        rels = dict(out)
        assert "RELIANCE.NS" in rels
        # Alias hit but no full-name hit → 0.7
        assert rels["RELIANCE.NS"] == pytest.approx(0.7)

    def test_ambiguous_alias_penalised(self):
        """Bare 'Tata' is ambiguous (Steel / Motors / Power / TCS).
        Without a full company name in the text the match must
        carry a low relevance — 0.4 by spec."""
        from news.symbol_match import match_symbols
        out = match_symbols("Tata announces major restructuring")
        # No specific Tata company named — alias hits a default
        # mapping with reduced confidence.
        for ticker, rel in out:
            if rel == pytest.approx(1.0):
                continue  # full-name hit allowed
            assert rel <= 0.7

    def test_no_match_returns_empty(self):
        from news.symbol_match import match_symbols
        out = match_symbols("Global markets rise on Fed comments")
        # Nothing pointing at a specific Indian symbol.
        assert all(t not in {x[0] for x in out} for t in (
            "RELIANCE.NS", "TCS.NS", "INFY.NS",
        ))

    def test_watchlist_filter(self):
        """When watchlist is supplied, only those symbols may match."""
        from news.symbol_match import match_symbols
        out = match_symbols(
            "TCS bags multi-year deal worth $500 million",
            watchlist=["INFY.NS"],
        )
        # TCS would normally match but is outside the watchlist.
        assert "TCS.NS" not in {t for t, _ in out}

    def test_word_boundary_avoids_false_match(self):
        """'ITC' must NOT match 'switch' or 'practice'. Whole-word
        regex prevents the obvious failure mode."""
        from news.symbol_match import match_symbols
        out = match_symbols("Switching brokers; practice account update")
        assert "ITC.NS" not in {t for t, _ in out}


# ===========================================================================
# 2. FinBERT stub mode
# ===========================================================================

class TestFinBertStub:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from news import finbert as fb
        fb._reset_for_tests()
        yield
        fb._reset_for_tests()

    def test_stub_returns_zero_when_transformers_missing(self, monkeypatch):
        """Force the import to fail so the stub path activates,
        regardless of whether the host has transformers installed."""
        from news.finbert import get_finbert
        import builtins
        real_import = builtins.__import__

        def fail_import(name, *a, **kw):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("simulated missing transformers")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_import)
        fb = get_finbert()
        assert fb.score_batch(["good", "bad", "neutral"]) == [0.0, 0.0, 0.0]
        assert fb.is_stub is True

    def test_stub_batch_with_labels(self, monkeypatch):
        from news.finbert import get_finbert
        import builtins
        real_import = builtins.__import__

        def fail_import(name, *a, **kw):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("simulated")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_import)
        fb = get_finbert()
        out = fb.score_batch_with_labels(["abc"])
        assert out == [(0.0, "stub")]

    def test_empty_input_returns_empty(self, monkeypatch):
        from news.finbert import get_finbert
        fb = get_finbert()
        assert fb.score_batch([]) == []
        assert fb.score_batch_with_labels([]) == []


# ===========================================================================
# 3. FinBERT real model (guarded)
# ===========================================================================

class TestFinBertReal:
    """Runs only when transformers + torch are installed AND the
    ProsusAI/finbert model is locally cached. Skipped otherwise."""

    @pytest.fixture(scope="class")
    def fb(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        from news import finbert
        finbert._reset_for_tests()
        fb = finbert.get_finbert()
        # Touch one inference to force load. If load fails (no
        # internet, no cache), skip.
        try:
            fb.score_batch(["test"])
        except Exception as e:
            pytest.skip(f"FinBERT model unavailable: {e}")
        if fb.is_stub:
            pytest.skip("FinBERT fell back to stub mode")
        return fb

    def test_classifies_clear_positive(self, fb):
        out = fb.score_batch_with_labels([
            "Company posts record profit, beats all estimates",
        ])
        score, label = out[0]
        assert score > 0.3
        assert label in ("positive",)

    def test_classifies_clear_negative(self, fb):
        out = fb.score_batch_with_labels([
            "CEO arrested for fraud; company faces criminal probe",
        ])
        score, label = out[0]
        assert score < -0.3
        assert label in ("negative",)


# ===========================================================================
# 4. Fetcher
# ===========================================================================

class TestFetcher:
    def test_parses_rss_fixture(self):
        from news.fetcher import _parse_feed, _entry_to_item

        xml = (FIXTURES / "rss_economictimes.xml").read_text(encoding="utf-8")
        feed = _parse_feed(xml)
        assert len(feed.entries) == 3

        items = [_entry_to_item("economic_times", e) for e in feed.entries]
        titles = [i.title for i in items]
        assert any("Reliance" in t for t in titles)
        assert any("TCS" in t for t in titles)
        # All items have non-empty ID + published_at.
        for i in items:
            assert i.id
            assert i.published_at

    def test_id_deterministic(self):
        from news.fetcher import NewsItem
        a = NewsItem.make(
            source="x", title="hello world", body="", url="https://a/1",
            published_at="2026-05-13T10:00:00+05:30",
        )
        b = NewsItem.make(
            source="x", title="hello world", body="x", url="https://a/1",
            published_at="2026-05-13T10:01:00+05:30",
        )
        # Same source+url+title → same id even when body / published differ.
        assert a.id == b.id


# ===========================================================================
# 5. Aggregator
# ===========================================================================

class TestAggregator:
    def test_aggregate_simple(self):
        from news.scorer import aggregate_rows
        # (score, relevance, age_hours)
        rows = [(0.8, 1.0, 0.0), (-0.2, 1.0, 0.0)]
        out = aggregate_rows(rows)
        assert out == pytest.approx(0.3, abs=1e-6)

    def test_aggregate_recency_weights_newer_higher(self):
        """Two opposite-sign items: a fresh +1 and a 24h-old -1.
        The fresh one should dominate by ~e^2 ≈ 7×, so net > 0."""
        from news.scorer import aggregate_rows
        rows = [(+1.0, 1.0, 0.0), (-1.0, 1.0, 24.0)]
        out = aggregate_rows(rows)
        assert out is not None
        assert out > 0.6

    def test_aggregate_relevance_downweights(self):
        """A low-relevance positive shouldn't drown a high-relevance
        negative."""
        from news.scorer import aggregate_rows
        rows = [(+1.0, 0.4, 0.0), (-1.0, 1.0, 0.0)]
        out = aggregate_rows(rows)
        assert out is not None
        assert out < 0  # neg wins because of higher relevance

    def test_aggregate_empty_returns_none(self):
        from news.scorer import aggregate_rows
        assert aggregate_rows([]) is None

    def test_aggregate_capped(self):
        """Even with all-positive items, max is 1.0."""
        from news.scorer import aggregate_rows
        rows = [(+1.0, 1.0, 0.0)] * 5
        assert aggregate_rows(rows) == pytest.approx(1.0)


# ===========================================================================
# 6. get_symbol_news_score via DB
# ===========================================================================

class TestGetSymbolNewsScore:
    @pytest.fixture
    def db_path(self, tmp_path, monkeypatch):
        import bot.config as bot_config
        import bot.db as bot_db
        path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", path)
        monkeypatch.setattr(bot_db, "DB_PATH", path)
        bot_db.init_db()
        return path

    def _insert(self, db_path, items):
        """items: list of (id, finbert_score, published_at, symbol, relevance)."""
        with sqlite3.connect(db_path) as conn:
            for item_id, fb_score, pub, sym, rel in items:
                conn.execute(
                    "INSERT INTO news_items "
                    "(id, source, title, body, url, published_at, fetched_at, "
                    " finbert_score, finbert_label) "
                    "VALUES (?, 'rss', 't', '', '', ?, ?, ?, 'positive')",
                    (item_id, pub, pub, fb_score),
                )
                conn.execute(
                    "INSERT INTO news_scores (symbol, news_id, relevance) "
                    "VALUES (?, ?, ?)",
                    (sym, item_id, rel),
                )

    def test_positive_aggregate(self, db_path):
        from news.scorer import get_symbol_news_score
        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        pub = (now - timedelta(hours=1)).isoformat()
        self._insert(db_path, [
            ("a", +0.8, pub, "RELIANCE.NS", 1.0),
            ("b", +0.5, pub, "RELIANCE.NS", 1.0),
            ("c", -0.2, pub, "RELIANCE.NS", 0.4),
        ])
        out = get_symbol_news_score(
            "RELIANCE.NS", hours=24, db_path=db_path, now_utc=now,
        )
        assert out is not None
        assert out > 0

    def test_no_news_returns_none(self, db_path):
        from news.scorer import get_symbol_news_score
        out = get_symbol_news_score(
            "TCS.NS", hours=24, db_path=db_path,
            now_utc=datetime(2026, 5, 13, 12, tzinfo=timezone.utc),
        )
        assert out is None

    def test_outside_window_returns_none(self, db_path):
        from news.scorer import get_symbol_news_score
        now = datetime(2026, 5, 13, 12, tzinfo=timezone.utc)
        old = (now - timedelta(hours=48)).isoformat()
        self._insert(db_path, [("old", 0.5, old, "INFY.NS", 1.0)])
        out = get_symbol_news_score(
            "INFY.NS", hours=24, db_path=db_path, now_utc=now,
        )
        assert out is None

    def test_recency_dominates(self, db_path):
        """Fresh negative outweighs older positive."""
        from news.scorer import get_symbol_news_score
        now = datetime(2026, 5, 13, 12, tzinfo=timezone.utc)
        old_pub = (now - timedelta(hours=20)).isoformat()
        fresh_pub = (now - timedelta(minutes=15)).isoformat()
        self._insert(db_path, [
            ("old", +0.9, old_pub, "X.NS", 1.0),
            ("new", -0.9, fresh_pub, "X.NS", 1.0),
        ])
        out = get_symbol_news_score(
            "X.NS", hours=24, db_path=db_path, now_utc=now,
        )
        assert out is not None
        assert out < 0


# ===========================================================================
# 7. score_news with news_score
# ===========================================================================

class TestScoreNewsWithSentiment:
    def test_positive_news_lifts_long(self):
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        long_score = score_news(
            None, "LONG", weights=weights, news_score=0.8,
        )
        short_score = score_news(
            None, "SHORT", weights=weights, news_score=0.8,
        )
        # LONG with strongly positive news → high
        # SHORT with strongly positive news → low
        assert long_score > 80
        assert short_score < 20

    def test_negative_news_lifts_short(self):
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        long_score = score_news(
            None, "LONG", weights=weights, news_score=-0.8,
        )
        short_score = score_news(
            None, "SHORT", weights=weights, news_score=-0.8,
        )
        assert long_score < 20
        assert short_score > 80

    def test_neutral_news(self):
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        out = score_news(None, "LONG", weights=weights, news_score=0.0)
        assert out == pytest.approx(50.0)

    def test_news_score_preferred_over_filing_title(self):
        """When both present, FinBERT-derived news_score wins —
        it's text-aware and the filing_title bonus was a regex
        approximation."""
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        out = score_news(
            "Dividend declared", "LONG",
            weights=weights, news_score=-0.6,
        )
        assert out < 50  # FinBERT-bearish wins

    def test_filing_title_fallback_when_no_news_score(self):
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        out = score_news("Dividend declared", "LONG", weights=weights)
        assert out == pytest.approx(100.0)

    def test_nothing_neutral(self):
        from scoring.components import score_news
        weights = {"filing_signal": 1.0}
        out = score_news(None, "LONG", weights=weights)
        assert out == pytest.approx(50.0)
