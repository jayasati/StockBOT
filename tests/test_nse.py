"""Phase-9 NSE client + MarketContext tests. NO live HTTP.

Fixture files under ``tests/fixtures/nse/`` drive every parser test.
The async fetchers are exercised by monkey-patching ``data.nse._get_json``
to return fixture payloads, so the network code path is never touched.

Classes:
  1. TestParsers           — pure parser functions vs fixtures
  2. TestCache             — in-memory + DB snapshot cache behaviour
  3. TestFetchers          — public async fetchers via patched _get_json
  4. TestMarketContext     — aggregation, partial-failure, to_dict
  5. TestScoreMarket       — FII/PCR sub-criteria in score_market
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "nse"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


# ===========================================================================
# 1. Parsers
# ===========================================================================

class TestParsers:
    def test_parse_vix(self):
        from data.nse import parse_vix
        assert parse_vix(_load("all_indices.json")) == pytest.approx(13.45)

    def test_parse_vix_missing_returns_none(self):
        from data.nse import parse_vix
        assert parse_vix({"data": [{"index": "NIFTY 50", "last": 22000}]}) is None

    def test_parse_vix_handles_garbage(self):
        from data.nse import parse_vix
        assert parse_vix(None) is None
        assert parse_vix({}) is None
        assert parse_vix({"data": None}) is None

    def test_parse_index_quote_nifty(self):
        from data.nse import parse_index_quote
        q = parse_index_quote(_load("all_indices.json"), "NIFTY 50")
        assert q is not None
        assert q["last"] == pytest.approx(22513.7)
        assert q["change_pct"] == pytest.approx(0.40)
        assert q["prev_close"] == pytest.approx(22425.0)

    def test_parse_index_quote_banknifty(self):
        from data.nse import parse_index_quote
        q = parse_index_quote(_load("all_indices.json"), "NIFTY BANK")
        assert q is not None
        assert q["change_pct"] == pytest.approx(0.26)

    def test_parse_index_quote_missing(self):
        from data.nse import parse_index_quote
        assert parse_index_quote(_load("all_indices.json"), "NIFTY FMCG") is None

    def test_parse_market_status(self):
        from data.nse import parse_market_status
        m = parse_market_status(_load("market_status.json"))
        assert m["Capital Market"] == "OPEN"
        assert m["Currency"] == "OPEN"

    def test_parse_fii_dii(self):
        from data.nse import parse_fii_dii
        out = parse_fii_dii(_load("fii_dii.json"))
        assert out is not None
        assert out["fii_net_cr"] == pytest.approx(1300.24)
        assert out["dii_net_cr"] == pytest.approx(-550.45)
        assert out["date"] == "13-May-2026"

    def test_parse_fii_dii_empty(self):
        from data.nse import parse_fii_dii
        assert parse_fii_dii([]) is None
        assert parse_fii_dii(None) is None

    def test_parse_option_chain_shape(self):
        from data.nse import parse_option_chain
        df = parse_option_chain(_load("option_chain_nifty.json"))
        assert isinstance(df, pd.DataFrame)
        assert {"strike", "expiry", "ce_oi", "pe_oi"}.issubset(df.columns)
        assert len(df) == 4  # 3 strikes × 1st expiry + 1 strike × 2nd

    def test_parse_option_chain_handles_missing(self):
        from data.nse import parse_option_chain
        df = parse_option_chain(None)
        assert df.empty
        assert "strike" in df.columns

    def test_parse_pcr_nearest_expiry(self):
        """First expiry (16-May): CE total = 50k+120k+75k = 245k,
        PE total = 80k+95k+100k = 275k. PCR = 275/245 ≈ 1.1224."""
        from data.nse import parse_pcr
        pcr = parse_pcr(_load("option_chain_nifty.json"))
        assert pcr == pytest.approx(275000 / 245000)

    def test_parse_pcr_specific_expiry(self):
        from data.nse import parse_pcr
        pcr = parse_pcr(_load("option_chain_nifty.json"), expiry="23-May-2026")
        # CE=30k, PE=40k → 4/3
        assert pcr == pytest.approx(40000 / 30000)

    def test_parse_pcr_zero_ce_returns_none(self):
        from data.nse import parse_pcr
        # All PE only; CE total = 0.
        payload = {
            "records": {
                "expiryDates": ["16-May-2026"],
                "data": [
                    {"strikePrice": 22000, "expiryDate": "16-May-2026",
                     "PE": {"openInterest": 1000}},
                ],
            }
        }
        assert parse_pcr(payload) is None

    def test_parse_fno_ban_dict_shape(self):
        from data.nse import parse_fno_ban
        s = parse_fno_ban(_load("fno_ban.json"))
        assert s == {"BIOCON", "RBLBANK", "GNFC"}

    def test_parse_fno_ban_list_of_strings(self):
        from data.nse import parse_fno_ban
        s = parse_fno_ban({"data": ["BIOCON", "RBLBANK"]})
        assert s == {"BIOCON", "RBLBANK"}

    def test_parse_fno_ban_empty(self):
        from data.nse import parse_fno_ban
        assert parse_fno_ban({}) == set()
        assert parse_fno_ban(None) == set()


# ===========================================================================
# 2. Cache
# ===========================================================================

class TestCache:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        """Each test gets its own DB path + cleared in-memory cache."""
        from data import nse
        import bot.config as bot_config
        import bot.db as bot_db
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(nse, "DB_PATH", db_path)
        bot_db.init_db()
        nse.clear_caches()
        self.db_path = db_path
        yield
        nse.clear_caches()

    def test_cache_hit_skips_fetch(self, monkeypatch):
        """A second call within the TTL returns the cached payload
        without invoking the inner HTTP path."""
        from data import nse
        nse._cache_set("all_indices", None, {"data": [{"index": "INDIA VIX", "last": 13.45}]})

        async def boom(*a, **kw):
            raise AssertionError("HTTP path must not be called on cache hit")

        # Patch the session lookup just in case the function under
        # test bypasses _cache_get.
        monkeypatch.setattr(nse, "_get_session", boom)
        vix = asyncio.run(nse.fetch_vix())
        assert vix == pytest.approx(13.45)

    def test_persist_writes_nse_snapshots(self):
        from data import nse
        nse._persist_snapshot("all_indices", None,
                              {"data": [{"index": "X"}]}, db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT kind, payload_json FROM nse_snapshots"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "all_indices"
        assert "X" in rows[0][1]

    def test_cache_persist_round_trip(self):
        from data import nse
        nse._cache_set("option_chain", {"symbol": "NIFTY"},
                       {"records": {"data": []}}, db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            kind, payload = conn.execute(
                "SELECT kind, payload_json FROM nse_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert kind == "option_chain:symbol=NIFTY"
        assert json.loads(payload) == {"records": {"data": []}}


# ===========================================================================
# 3. Async fetchers via patched _get_json
# ===========================================================================

class TestFetchers:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch, tmp_path):
        """Patch _get_json to return fixture payloads keyed by endpoint."""
        from data import nse
        import bot.config as bot_config
        import bot.db as bot_db
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(nse, "DB_PATH", db_path)
        bot_db.init_db()
        nse.clear_caches()

        responses = {
            "all_indices":   _load("all_indices.json"),
            "market_status": _load("market_status.json"),
            "fii_dii":       _load("fii_dii.json"),
            "option_chain":  _load("option_chain_nifty.json"),
            "fno_ban":       _load("fno_ban.json"),
        }

        async def fake_get_json(endpoint_key, params=None):
            return responses.get(endpoint_key)

        monkeypatch.setattr(nse, "_get_json", fake_get_json)
        yield
        nse.clear_caches()

    def test_fetch_vix(self):
        from data import nse
        assert asyncio.run(nse.fetch_vix()) == pytest.approx(13.45)

    def test_fetch_nifty_quote(self):
        from data import nse
        q = asyncio.run(nse.fetch_nifty_quote())
        assert q["last"] == pytest.approx(22513.7)
        assert q["change_pct"] == pytest.approx(0.40)

    def test_fetch_banknifty_quote(self):
        from data import nse
        q = asyncio.run(nse.fetch_banknifty_quote())
        assert q["change_pct"] == pytest.approx(0.26)

    def test_fetch_fii_dii(self):
        from data import nse
        out = asyncio.run(nse.fetch_fii_dii())
        assert out["fii_net_cr"] == pytest.approx(1300.24)
        assert out["dii_net_cr"] == pytest.approx(-550.45)

    def test_fetch_options_chain_returns_frame(self):
        from data import nse
        df = asyncio.run(nse.fetch_options_chain("NIFTY"))
        assert len(df) == 4

    def test_fetch_pcr(self):
        from data import nse
        pcr = asyncio.run(nse.fetch_pcr("NIFTY"))
        assert pcr == pytest.approx(275000 / 245000)

    def test_fetch_market_status(self):
        from data import nse
        m = asyncio.run(nse.fetch_market_status())
        assert m["Capital Market"] == "OPEN"

    def test_fetch_fno_ban_set(self):
        from data import nse
        s = asyncio.run(nse.fetch_fno_ban_set())
        assert s == {"BIOCON", "RBLBANK", "GNFC"}


# ===========================================================================
# 4. MarketContext aggregation
# ===========================================================================

class TestMarketContext:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch, tmp_path):
        from data import nse
        import bot.config as bot_config
        import bot.db as bot_db
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(nse, "DB_PATH", db_path)
        bot_db.init_db()
        nse.clear_caches()

        responses = {
            "all_indices":   _load("all_indices.json"),
            "fii_dii":       _load("fii_dii.json"),
            "option_chain":  _load("option_chain_nifty.json"),
        }

        async def fake_get_json(endpoint_key, params=None):
            return responses.get(endpoint_key)

        monkeypatch.setattr(nse, "_get_json", fake_get_json)
        yield
        nse.clear_caches()

    def test_get_market_context_populates_all_fields(self):
        from data.market_context import get_market_context
        ctx = asyncio.run(get_market_context())
        assert ctx.nifty_change_pct == pytest.approx(0.40)
        assert ctx.banknifty_change_pct == pytest.approx(0.26)
        assert ctx.vix == pytest.approx(13.45)
        assert ctx.fii_net_cr == pytest.approx(1300.24)
        assert ctx.dii_net_cr == pytest.approx(-550.45)
        assert ctx.pcr_nifty == pytest.approx(275000 / 245000)

    def test_to_dict_keys_match_score_market(self):
        from data.market_context import get_market_context
        ctx = asyncio.run(get_market_context())
        d = ctx.to_dict()
        # Keys the scoring engine reads:
        for k in ("nifty_pct", "bank_nifty_pct", "vix",
                  "fii_net_cr", "dii_net_cr", "pcr_nifty"):
            assert k in d

    def test_partial_failure_returns_partial_context(self, monkeypatch):
        """If one underlying fetch raises, the others still populate."""
        from data import nse
        from data.market_context import get_market_context

        async def half_broken(endpoint_key, params=None):
            if endpoint_key == "fii_dii":
                raise RuntimeError("simulated NSE 503")
            return {
                "all_indices": _load("all_indices.json"),
                "option_chain": _load("option_chain_nifty.json"),
            }.get(endpoint_key)

        monkeypatch.setattr(nse, "_get_json", half_broken)
        nse.clear_caches()
        ctx = asyncio.run(get_market_context())
        assert ctx.vix is not None
        assert ctx.fii_net_cr is None  # only this one failed
        assert ctx.pcr_nifty is not None

    def test_format_for_status(self):
        from data.market_context import (
            MarketContext, format_for_status,
        )
        ctx = MarketContext(
            nifty_change_pct=0.40, banknifty_change_pct=0.26,
            vix=13.45, fii_net_cr=1300.24, dii_net_cr=-550.45,
            pcr_nifty=1.12,
        )
        s = format_for_status(ctx)
        assert "NIFTY" in s
        assert "13.45" in s
        assert "+1300" in s or "1300" in s


# ===========================================================================
# 5. score_market reads FII/PCR
# ===========================================================================

class TestFyersOptionChainFallback:
    """When NSE silent-throttles the option-chain endpoint, the
    Fyers SDK is the fallback. These tests verify:

      1. ``normalise_to_nse_shape`` produces the exact records-data
         shape ``parse_pcr`` already understands.
      2. ``_option_chain_payload`` invokes the fallback only when
         NSE returns empty/None, and caches the result.
    """

    def test_normalise_to_nse_shape(self):
        from fyers_client.options import normalise_to_nse_shape
        fy = _load("fyers_option_chain_nifty.json")
        out = normalise_to_nse_shape(fy["data"])
        rows = out["records"]["data"]
        # 3 strikes × CE+PE = 3 row groups; underlying-spot entry
        # (option_type='') was dropped.
        assert len(rows) == 3
        assert all("CE" in r and "PE" in r for r in rows)
        # Expiry epoch 1747401000 → 16-May-2026 IST.
        assert all(r["expiryDate"] == "16-May-2026" for r in rows)
        # expiryDates is populated, ordered.
        assert "16-May-2026" in out["records"]["expiryDates"]

    def test_normalised_payload_drives_parse_pcr(self):
        """Round-trip: Fyers payload → NSE shape → parse_pcr →
        the same number the NSE-shaped fixture produces."""
        from data.nse import parse_pcr
        from fyers_client.options import normalise_to_nse_shape
        fy = _load("fyers_option_chain_nifty.json")
        nse_shaped = normalise_to_nse_shape(fy["data"])
        pcr = parse_pcr(nse_shaped, expiry="16-May-2026")
        # CE = 50k + 120k + 75k = 245k
        # PE = 80k + 95k + 100k = 275k
        assert pcr == pytest.approx(275000 / 245000)

    def test_normalise_handles_empty(self):
        from fyers_client.options import normalise_to_nse_shape
        out = normalise_to_nse_shape(None)
        assert out["records"]["data"] == []

    def test_option_chain_payload_uses_nse_when_populated(
        self, tmp_path, monkeypatch,
    ):
        """NSE returns valid data → Fyers must NOT be called."""
        from data import nse
        import bot.config as bot_config
        import bot.db as bot_db
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(nse, "DB_PATH", db_path)
        bot_db.init_db()
        nse.clear_caches()

        nse_payload = _load("option_chain_nifty.json")

        async def fake_get_json(endpoint_key, params=None):
            return nse_payload

        async def boom(*a, **kw):
            raise AssertionError("Fyers fallback must not be invoked")

        monkeypatch.setattr(nse, "_get_json", fake_get_json)
        monkeypatch.setattr(
            "fyers_client.options.fetch_option_chain_nse_shape", boom,
        )
        payload = asyncio.run(nse._option_chain_payload("NIFTY"))
        assert payload is nse_payload

    def test_option_chain_payload_falls_back_to_fyers_on_empty(
        self, tmp_path, monkeypatch,
    ):
        """NSE returns ``{}`` → Fyers fetcher is called and its
        normalised payload is returned + cached."""
        from data import nse
        import bot.config as bot_config
        import bot.db as bot_db
        db_path = str(tmp_path / "alerts.db")
        monkeypatch.setattr(bot_config, "DB_PATH", db_path)
        monkeypatch.setattr(bot_db, "DB_PATH", db_path)
        monkeypatch.setattr(nse, "DB_PATH", db_path)
        bot_db.init_db()
        nse.clear_caches()

        fy_normalised = {
            "records": {
                "data": [
                    {"strikePrice": 22500, "expiryDate": "16-May-2026",
                     "CE": {"openInterest": 100},
                     "PE": {"openInterest": 200}},
                ],
                "expiryDates": ["16-May-2026"],
            }
        }

        async def fake_get_json(endpoint_key, params=None):
            return {}  # NSE silent throttle

        async def fake_fyers(symbol, strikecount=20):
            return fy_normalised

        monkeypatch.setattr(nse, "_get_json", fake_get_json)
        monkeypatch.setattr(
            "fyers_client.options.fetch_option_chain_nse_shape", fake_fyers,
        )
        payload = asyncio.run(nse._option_chain_payload("NIFTY"))
        assert payload is fy_normalised
        # PCR over normalised → 200/100 = 2.0
        from data.nse import parse_pcr
        assert parse_pcr(payload) == pytest.approx(2.0)


class TestScoreMarketWithMacro:
    def test_fii_positive_lifts_long(self):
        from scoring.components import score_market
        from scoring.config_loader import DEFAULTS
        weights = DEFAULTS["component_weights"]["market"]
        score = score_market(
            {"nifty_pct": 0.5, "bank_nifty_pct": 0.5, "vix": 14.0,
             "fii_net_cr": 2500.0, "dii_net_cr": -200.0,
             "pcr_nifty": 1.15},
            "LONG", weights=weights,
        )
        assert score > 75

    def test_fii_strong_outflow_kills_long(self):
        from scoring.components import score_market
        from scoring.config_loader import DEFAULTS
        weights = DEFAULTS["component_weights"]["market"]
        score = score_market(
            {"nifty_pct": -0.7, "bank_nifty_pct": -0.7, "vix": 22.0,
             "fii_net_cr": -3000.0, "dii_net_cr": 200.0,
             "pcr_nifty": 0.6},
            "LONG", weights=weights,
        )
        assert score < 30

    def test_pcr_high_favours_long(self):
        """PCR > 1.2 = put-heavy = contrarian-bullish for LONG."""
        from scoring.components import score_market
        from scoring.config_loader import DEFAULTS
        weights = {"pcr": 1.0}  # isolate PCR's contribution
        long_score = score_market(
            {"pcr_nifty": 1.30}, "LONG", weights=weights,
        )
        short_score = score_market(
            {"pcr_nifty": 1.30}, "SHORT", weights=weights,
        )
        assert long_score > short_score

    def test_pcr_low_favours_short(self):
        from scoring.components import score_market
        weights = {"pcr": 1.0}
        long_score = score_market(
            {"pcr_nifty": 0.50}, "LONG", weights=weights,
        )
        short_score = score_market(
            {"pcr_nifty": 0.50}, "SHORT", weights=weights,
        )
        assert short_score > long_score

    def test_missing_fields_neutral(self):
        """When fii/pcr are absent, the score stays usable (falls
        back to the nifty/bn/vix sub-criteria)."""
        from scoring.components import score_market
        from scoring.config_loader import DEFAULTS
        weights = DEFAULTS["component_weights"]["market"]
        score = score_market(
            {"nifty_pct": 0.5, "bank_nifty_pct": 0.5, "vix": 14.0},
            "LONG", weights=weights,
        )
        assert score > 0
