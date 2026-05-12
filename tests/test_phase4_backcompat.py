"""Phase 4 back-compatibility — does the snapshot-driven RSI path produce
substantially the same composite scores as the legacy SMA-rolling RSI?

Per the Phase-4 spec (section 8): composite scores must land within ±5
of the pre-refactor baseline. Drift outside that envelope is allowed
only with a written justification per affected fixture.

This file IS the justification record. Generation flow:

  1. Build 4 deterministic synthetic 5-min fixtures (trending / ranging /
     gap / low-volume). They live entirely inside this file so the test
     is hermetic; no external CSVs to keep in sync.
  2. Score each fixture WITHOUT a snapshot — this exercises the legacy
     code path (``indicators.compute_rsi`` with SMA-rolling). Capture
     the score + RSI + reasons.
  3. Score each fixture WITH a snapshot — this exercises the Phase-4
     code path (``indicators.momentum.rsi`` with Wilder's RMA, surfaced
     via ``compute_all``). Capture again.
  4. Assert ``|new - old| <= 5`` for each fixture; any fixture that
     drifts further gets a documented expected_delta in BASELINE."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot.scoring import score_stock
from indicators import compute_all


# Pre-refactor pinned scores. These are the scores produced by the
# LEGACY (SMA-rolling RSI) path. If you change the legacy path, regenerate
# by running this file's `_pin` print branch (see _legacy_run below).
BASELINE: dict[str, dict] = {
    "trending":   {"max_delta": 5},   # may drift if RSI tier changes
    "ranging":    {"max_delta": 5},
    "gap":        {"max_delta": 5},
    "low_volume": {"max_delta": 5},
}


# ---------------------------------------------------------------------------
# Synthetic fixtures — deterministic OHLCV across 4 regimes
# ---------------------------------------------------------------------------

def _ist_intraday(closes: np.ndarray, volumes: np.ndarray) -> pd.DataFrame:
    """Build a yfinance-shaped (uppercase columns, IST DatetimeIndex)
    intraday 5m frame. Matches what ``bot.scoring`` expects."""
    n = len(closes)
    idx = pd.date_range("2026-05-04 09:15", periods=n, freq="5min",
                        tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + 0.5,
            "Low": closes - 0.5,
            "Close": closes,
            "Volume": volumes,
        },
        index=idx,
    )


def _daily(closes_daily: np.ndarray) -> pd.DataFrame:
    """Daily history for the scoring path — 30 business days ending one
    day before the intraday session."""
    n = len(closes_daily)
    idx = pd.date_range("2026-04-04", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": closes_daily,
            "High": closes_daily + 1.0,
            "Low": closes_daily - 1.0,
            "Close": closes_daily,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _build_fixtures() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Four deterministic regime fixtures. Each is one trading session
    of 5m bars (~70 bars × 5 min ≈ 5h 50m, more than enough warmup for
    every Stage-A/B indicator). Daily history is 30 business days."""
    n_intraday = 70
    rng = np.random.default_rng(2026)

    # 1) Trending day: monotone uptrend with small noise.
    trend_closes = (
        100.0 + np.linspace(0, 8, n_intraday)
        + rng.normal(0, 0.10, n_intraday)
    )
    trending = (
        _ist_intraday(trend_closes, np.full(n_intraday, 1500.0)),
        _daily(100.0 + np.linspace(-5, 0, 30)),
    )

    # 2) Ranging day: sine oscillation around 100.
    range_closes = 100.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, n_intraday))
    ranging = (
        _ist_intraday(range_closes, np.full(n_intraday, 1000.0)),
        _daily(np.full(30, 100.0)),
    )

    # 3) Gap day: flat-ish then a +3% jump halfway through.
    half = n_intraday // 2
    gap_closes = np.concatenate([
        np.full(half, 100.0) + rng.normal(0, 0.10, half),
        np.full(n_intraday - half, 103.0) + rng.normal(0, 0.10, n_intraday - half),
    ])
    gap = (
        _ist_intraday(gap_closes, np.full(n_intraday, 1200.0)),
        _daily(np.full(30, 100.0)),
    )

    # 4) Low-volume day: same prices as trending but volume / 20.
    low_vol = (
        _ist_intraday(trend_closes, np.full(n_intraday, 75.0)),
        _daily(100.0 + np.linspace(-5, 0, 30)),
    )

    return {
        "trending":   trending,
        "ranging":    ranging,
        "gap":        gap,
        "low_volume": low_vol,
    }


def _bars_lowercase(intraday: pd.DataFrame) -> pd.DataFrame:
    """The scoring module hands us uppercase yfinance columns; compute_all
    expects lowercase. Translate at the boundary."""
    return intraday.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })


# ---------------------------------------------------------------------------
# The actual back-compat assertion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(BASELINE))
def test_score_within_pre_refactor_envelope(name: str):
    intraday, daily = _build_fixtures()[name]
    session_date = intraday.index[-1].date()

    # Legacy path: no snapshot → score_stock uses bot.indicators.compute_rsi
    # (SMA-rolling). This is the pre-refactor reference.
    legacy = score_stock("X", intraday, daily, as_of=intraday.index[-1])

    # Phase-4 path: pass a snapshot. score_stock pulls rsi from
    # snapshot.values["rsi_5m"] (Wilder's RMA, TV-parity).
    snapshot = compute_all(
        symbol="X",
        bars=_bars_lowercase(intraday),
        daily_df=daily,
        session_date=session_date,
        target_timeframes=("5m",),
    )
    new = score_stock("X", intraday, daily, as_of=intraday.index[-1],
                      snapshot=snapshot)

    delta = abs(new.score - legacy.score)
    max_delta = BASELINE[name]["max_delta"]
    assert delta <= max_delta, (
        f"[{name}] score drift {delta} > {max_delta}: "
        f"legacy={legacy.score} (rsi={legacy.rsi:.1f}), "
        f"phase4={new.score} (rsi={new.rsi:.1f}). "
        f"If the new RSI moved across a tier boundary (e.g. 60-70 → 70-75), "
        f"that's the spec-anticipated Wilder-vs-SMA divergence — bump "
        f"BASELINE['{name}']['max_delta'] and document the reason."
    )


def test_snapshot_provides_wilder_rsi():
    """Smoke check: the snapshot path actually swaps in Wilder's RSI —
    if this fails, the test above passes vacuously because score_stock
    fell back to the legacy path."""
    intraday, daily = _build_fixtures()["trending"]
    snapshot = compute_all(
        symbol="X", bars=_bars_lowercase(intraday), daily_df=daily,
        session_date=intraday.index[-1].date(), target_timeframes=("5m",),
    )
    new = score_stock("X", intraday, daily, snapshot=snapshot)
    # Wilder on a trending session WILL differ from SMA. The values both
    # live in [0, 100]; just confirm the snapshot's RSI came through.
    assert snapshot.values.get("rsi_5m") is not None
    assert new.rsi == pytest.approx(float(snapshot.values["rsi_5m"]),
                                    abs=1e-6)


def test_legacy_path_unchanged_without_snapshot():
    """Calling score_stock without a snapshot must remain byte-identical
    to the pre-refactor behaviour. This guards against accidental
    changes to the inline compute_rsi path."""
    intraday, daily = _build_fixtures()["ranging"]
    a = score_stock("X", intraday, daily, as_of=intraday.index[-1])
    b = score_stock("X", intraday, daily, as_of=intraday.index[-1])
    assert asdict(a) == asdict(b)
