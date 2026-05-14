"""compute_all — snapshot contract: namespacing, insufficient, warnings,
and per-TF/per-indicator coverage."""
from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from indicators import REGISTRY, IndicatorSnapshot, compute_all

IST = ZoneInfo("Asia/Kolkata")


def _intraday_bars(n: int = 200) -> pd.DataFrame:
    """Build a synthetic, deterministic intraday frame at 5m cadence
    covering a full session and a bit, with smooth trend + noise."""
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n)) + np.linspace(0, 5, n)
    highs = closes + rng.uniform(0.1, 0.5, n)
    lows = closes - rng.uniform(0.1, 0.5, n)
    opens = closes - rng.uniform(-0.2, 0.2, n)
    volumes = rng.uniform(800.0, 1500.0, n)
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min", tz=IST)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=idx,
    )


def _daily_bars(n: int = 30) -> pd.DataFrame:
    """Daily frame ending one day before the intraday session date."""
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(1.0, 2.0, n)
    lows = closes - rng.uniform(1.0, 2.0, n)
    idx = pd.date_range("2026-04-04", periods=n, freq="B")   # business days
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows,
         "close": closes, "volume": rng.uniform(1e6, 5e6, n)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------

def test_returns_snapshot_dataclass():
    snap = compute_all(
        symbol="NSE:X-EQ",
        bars=_intraday_bars(),
        daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    assert isinstance(snap, IndicatorSnapshot)
    assert snap.symbol == "NSE:X-EQ"
    assert snap.session_date == date(2026, 5, 4)
    assert isinstance(snap.values, dict)
    assert isinstance(snap.warnings, list)
    assert isinstance(snap.insufficient, list)


def test_namespacing_series_indicator():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(), daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    # RSI is single-output and TF-applicable.
    assert "rsi_5m" in snap.values
    assert "rsi_15m" in snap.values
    assert "rsi_60m" in snap.values


def test_namespacing_frame_indicator_splats_into_per_column_keys():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(), daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    for col in ("macd", "signal", "histogram"):
        assert f"macd_5m_{col}" in snap.values


def test_namespacing_levels_have_no_tf_suffix():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(), daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    # PDH/PDL/PDC come from the daily frame, no TF suffix.
    for key in ("pdh", "pdl", "pdc"):
        assert key in snap.values
    # IB derived from intraday — also flat.
    for key in ("ib_high", "ib_low"):
        assert key in snap.values
    # Pivot points namespaced with method.
    assert "pivot_classic" in snap.values
    assert "pivot_classic_r1" in snap.values
    assert "pivot_classic_s3" in snap.values
    # Opening range namespaced with minutes.
    assert "orh_15" in snap.values
    assert "or_mid_15" in snap.values


# ---------------------------------------------------------------------------
# Insufficient + warnings paths
# ---------------------------------------------------------------------------

def test_short_input_populates_insufficient_list():
    # 50 bars is shorter than ichimoku.warmup (78), so senkou_b should be
    # listed as insufficient.
    snap = compute_all(
        symbol="X", bars=_intraday_bars(50),
        daily_df=_daily_bars(), session_date=date(2026, 5, 4),
    )
    assert "ichimoku_5m_senkou_b" in snap.insufficient
    assert snap.values["ichimoku_5m_senkou_b"] is None


def test_insufficient_indicators_are_none_not_missing():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(50),
        daily_df=_daily_bars(), session_date=date(2026, 5, 4),
    )
    # Even indicators with very long warmups should HAVE a key, just None.
    for k in snap.insufficient:
        assert k in snap.values
        assert snap.values[k] is None


def test_unknown_indicator_raises():
    import pytest
    with pytest.raises(KeyError):
        compute_all(
            symbol="X", bars=_intraday_bars(), daily_df=_daily_bars(),
            session_date=date(2026, 5, 4), indicators=["not_real"],
        )


# ---------------------------------------------------------------------------
# Coverage — every REGISTRY entry shows up in either values or
# insufficient. Phase 7 must never silently miss an indicator.
# ---------------------------------------------------------------------------

def test_every_registry_indicator_appears_in_snapshot():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(200), daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    seen_indicators: set[str] = set()
    # Iterate REGISTRY longest-name-first so that, for example,
    # "vwap_sd_bands_5m_plus_1" matches the longer "vwap_sd_bands"
    # before the shorter "vwap" prefix — names are unique, so this
    # disambiguates without changing semantics.
    sorted_specs = sorted(REGISTRY.items(), key=lambda p: -len(p[0]))
    for key in snap.values:
        # Levels keys won't have a TF suffix; just check the registered
        # indicator's output_keys map into snap.values somehow.
        for name, spec in sorted_specs:
            if spec.category == "level":
                # Level keys are computed; check presence via the
                # registry's output_keys round-trip.
                # We'll do the actual coverage check below.
                pass
            else:
                if key.startswith(f"{name}_"):
                    seen_indicators.add(name)
                    break
    intraday_names = {
        name for name, spec in REGISTRY.items() if spec.category != "level"
    }
    # Every intraday indicator should show up at least once (some TF).
    assert intraday_names <= seen_indicators

    # Levels — verify each output key landed.
    level_specs = [s for s in REGISTRY.values() if s.category == "level"]
    for spec in level_specs:
        for col in spec.output_keys:
            from indicators.compute import _level_key
            key = _level_key(spec, col)
            assert key in snap.values, f"missing level key: {key}"


# ---------------------------------------------------------------------------
# computed_at is UTC
# ---------------------------------------------------------------------------

def test_computed_at_is_tz_aware_utc():
    snap = compute_all(
        symbol="X", bars=_intraday_bars(), daily_df=_daily_bars(),
        session_date=date(2026, 5, 4),
    )
    assert snap.computed_at.tzinfo is not None
    assert snap.computed_at.utcoffset().total_seconds() == 0
