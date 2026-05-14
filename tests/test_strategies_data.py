"""Tests for strategies.data — yfinance loader + parquet cache.

Network is mocked via monkeypatched ``yf.download``; cache root is
redirected per-test to a tmp_path so we never touch the real cache."""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategies import data as strat_data

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fake yfinance responses
# ---------------------------------------------------------------------------

def _fake_yf_frame(symbols: list[str], n_bars: int = 10):
    """Build the multi-index frame yf.download returns for multiple symbols."""
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz="UTC")
    cols = pd.MultiIndex.from_product(
        [symbols, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    rng = np.random.default_rng(42)
    data = rng.uniform(100, 105, size=(n_bars, len(symbols) * 6))
    return pd.DataFrame(data, index=idx, columns=cols)


def _fake_yf_single(symbol: str, n_bars: int = 10):
    """Single-symbol form (single-level columns)."""
    idx = pd.date_range("2026-05-04 09:15", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Open": rng.uniform(100, 105, n_bars),
        "High": rng.uniform(100, 105, n_bars),
        "Low": rng.uniform(100, 105, n_bars),
        "Close": rng.uniform(100, 105, n_bars),
        "Adj Close": rng.uniform(100, 105, n_bars),
        "Volume": rng.integers(1000, 10000, n_bars).astype(float),
    }, index=idx)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect CACHE_ROOT to a tmp dir so tests can't touch the real cache."""
    monkeypatch.setattr(strat_data, "CACHE_ROOT", tmp_path / "cache")
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# cache_path
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_includes_interval_and_days(self, isolated_cache):
        p = strat_data.cache_path("5m", 60)
        assert p.name == "5m_60d.parquet"
        assert p.parent == isolated_cache

    def test_different_combos_different_files(self, isolated_cache):
        assert strat_data.cache_path("5m", 60) != strat_data.cache_path("5m", 30)
        assert strat_data.cache_path("5m", 60) != strat_data.cache_path("15m", 60)


# ---------------------------------------------------------------------------
# fetch + caching
# ---------------------------------------------------------------------------

class TestFetch:
    def test_fetch_writes_parquet_to_cache(self, isolated_cache, monkeypatch):
        calls = []

        def fake_download(tickers, **kw):
            calls.append(list(tickers))
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        out = strat_data.fetch(["RELIANCE.NS", "TCS.NS"], days=60)
        assert set(out.keys()) == {"RELIANCE.NS", "TCS.NS"}
        path = strat_data.cache_path("5m", 60)
        assert path.exists()

    def test_second_fetch_reuses_cache(self, isolated_cache, monkeypatch):
        calls = []

        def fake_download(tickers, **kw):
            calls.append(list(tickers))
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        strat_data.fetch(["RELIANCE.NS"], days=60)
        strat_data.fetch(["RELIANCE.NS"], days=60)
        # Second call should NOT have hit the network.
        assert len(calls) == 1

    def test_fetch_merges_new_symbols_into_existing_cache(
        self, isolated_cache, monkeypatch,
    ):
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        strat_data.fetch(["RELIANCE.NS"], days=60)
        strat_data.fetch(["TCS.NS"], days=60)
        # Both symbols should now be in the cache file.
        from data.yf_fetch import load_parquet
        on_disk = load_parquet(strat_data.cache_path("5m", 60))
        assert set(on_disk) == {"RELIANCE.NS", "TCS.NS"}

    def test_refresh_bypasses_cache(self, isolated_cache, monkeypatch):
        calls = []

        def fake_download(tickers, **kw):
            calls.append(list(tickers))
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        strat_data.fetch(["RELIANCE.NS"], days=60)
        strat_data.fetch(["RELIANCE.NS"], days=60, refresh=True)
        assert len(calls) == 2

    def test_fetch_only_returns_requested_symbols(
        self, isolated_cache, monkeypatch,
    ):
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        # Pre-populate cache with two symbols.
        strat_data.fetch(["A.NS", "B.NS"], days=60)
        # Now ask for only one — should not include the other.
        out = strat_data.fetch(["A.NS"], days=60)
        assert set(out.keys()) == {"A.NS"}


# ---------------------------------------------------------------------------
# load (single-symbol, normalized for the engine)
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_lowercase_ohlcv(self, isolated_cache, monkeypatch):
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        df = strat_data.load("RELIANCE.NS", days=60)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert "adj close" not in df.columns
        assert "Adj Close" not in df.columns

    def test_load_index_is_ist(self, isolated_cache, monkeypatch):
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers))

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        df = strat_data.load("RELIANCE.NS", days=60)
        assert str(df.index.tz) in ("Asia/Kolkata", "IST")

    def test_load_missing_symbol_raises(self, isolated_cache, monkeypatch):
        def fake_download(tickers, **kw):
            # Return an empty frame to simulate "no data".
            return pd.DataFrame()

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        with pytest.raises(KeyError, match="No yfinance data"):
            strat_data.load("NOTASYMBOL.NS", days=60)

    def test_load_output_is_engine_ready(self, isolated_cache, monkeypatch):
        """run_backtest accepts the DataFrame load() returns without modification."""
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers), n_bars=30)

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        from strategies import BarUpDn, run_backtest
        df = strat_data.load("RELIANCE.NS", days=60)
        # Should not raise (column-shape check is the first thing run_backtest does)
        result = run_backtest(BarUpDn(), df, max_intraday_loss_pct=10.0,
                              apply_costs=False)
        assert result is not None


# ---------------------------------------------------------------------------
# cache_summary
# ---------------------------------------------------------------------------

class TestCacheSummary:
    def test_empty_when_no_cache(self, isolated_cache):
        s = strat_data.cache_summary("5m", 60)
        assert s.empty
        assert list(s.columns) == ["symbol", "bars", "first_ts", "last_ts"]

    def test_summarises_cached_symbols(self, isolated_cache, monkeypatch):
        def fake_download(tickers, **kw):
            return _fake_yf_frame(list(tickers), n_bars=42)

        monkeypatch.setattr(strat_data.yf, "download", fake_download)

        strat_data.fetch(["RELIANCE.NS", "TCS.NS"], days=60)
        s = strat_data.cache_summary("5m", 60)
        assert set(s["symbol"]) == {"RELIANCE.NS", "TCS.NS"}
        assert (s["bars"] == 42).all()
