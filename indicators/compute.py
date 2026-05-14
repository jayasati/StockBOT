"""compute_all — single entry point that returns an ``IndicatorSnapshot``
containing every REGISTRY indicator's value at the latest bar of the
specified timeframes. Phase 7 scoring reads from
``snapshot.values[<namespaced_key>]``.

Failure handling contract (frozen for Phase 7):

  * Indicator returns NaN due to warmup → snapshot.values[key] = None,
    key appended to snapshot.insufficient[]. Not an error.
  * Indicator raises → log + snapshot.values[key] = None, descriptive
    string appended to snapshot.warnings[]. Other indicators still run.
  * Indicator not in REGISTRY → KeyError (caller bug, fail loud).

Phase 7 treats ``None`` as "neutral" (zero contribution to the score),
NOT as "exclude symbol from scoring".

Namespacing convention for ``values`` keys:

  rsi_5m                      single-output indicator at a timeframe
  macd_5m_histogram           multi-output indicator: <name>_<tf>_<column>
  pdh                         session-level (no timeframe suffix)
  orh_15                      opening_range with minutes=15 default
  pivot_classic_r1            pivot_points with method=classic, key=r1

TODO (Phase 8): cache last snapshot per symbol and recompute only the
tail when a new bar closes. Full-recompute every call is acceptable for
current scan sizes (~500 symbols × 36 indicators × 4 timeframes) but
not at 5000+ symbols."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .mtf import resample_ohlcv
from .registry import REGISTRY, IndicatorSpec

log = logging.getLogger("alertbot.indicators")


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Frozen result of one compute_all run for one symbol.

    Treat ``values`` as the contract: keys are the namespaced indicator
    names; values are ``float`` for series/scalar outputs, ``None`` for
    insufficient/erroring. Multi-output indicators (MACD, ADX, Bollinger,
    etc.) flatten into multiple keys.

    ``insufficient`` lists keys whose underlying indicator returned NaN
    because the bar count didn't reach its warmup. ``warnings`` lists
    string descriptions of indicator function exceptions.

    Phase 7 scoring iterates ``REGISTRY.values()`` and pulls each spec's
    value from this snapshot."""
    symbol: str
    session_date: date
    computed_at: datetime
    values: dict[str, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    insufficient: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_scalar(value: Any) -> float | None:
    """Best-effort extract of the LAST bar's value from any indicator
    output (Series, single-column DataFrame, or scalar). Returns None
    for NaN, all-NaN, or empty inputs."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        last = value.iloc[-1]
        return None if pd.isna(last) else float(last)
    raise TypeError(f"unsupported value type for _last_scalar: {type(value)!r}")


def _last_row_dict(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float | None]:
    """Pull the last bar's values for selected columns out of a DataFrame
    output. Missing columns map to None."""
    if frame.empty:
        return {col: None for col in columns}
    last = frame.iloc[-1]
    out: dict[str, float | None] = {}
    for col in columns:
        if col not in frame.columns:
            out[col] = None
            continue
        v = last[col]
        out[col] = None if pd.isna(v) else float(v)
    return out


def _level_key(spec: IndicatorSpec, sub_key: str) -> str:
    """Compose the namespaced key for a session-level value.

    Rules baked in:
      previous_day_hlc → pdh, pdl, pdc           (subkeys passed through)
      opening_range    → orh_15, orl_15, or_mid_15   (suffix with minutes)
      initial_balance  → ib_high, ib_low         (subkeys passed through)
      pivot_points     → pivot_<method>, pivot_<method>_<level>"""
    if spec.name == "previous_day_hlc":
        return sub_key
    if spec.name == "opening_range":
        minutes = spec.default_params.get("minutes", 15)
        return f"{sub_key}_{minutes}"
    if spec.name == "initial_balance":
        return sub_key
    if spec.name == "pivot_points":
        method = spec.default_params.get("method", "classic")
        if sub_key == "pivot":
            return f"pivot_{method}"
        return f"pivot_{method}_{sub_key}"
    # Future levels fall back to {name}_{sub_key}.
    return f"{spec.name}_{sub_key}"


def _series_key(spec: IndicatorSpec, tf: str) -> str:
    return f"{spec.name}_{tf}"


def _frame_key(spec: IndicatorSpec, tf: str, col: str) -> str:
    return f"{spec.name}_{tf}_{col}"


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def compute_all(
    symbol: str,
    bars: pd.DataFrame,
    daily_df: pd.DataFrame,
    session_date: date,
    *,
    source_timeframe: str = "5m",
    target_timeframes: tuple[str, ...] = ("5m", "15m", "60m"),
    indicators: Iterable[str] | None = None,
) -> IndicatorSnapshot:
    """Run every requested REGISTRY indicator at every requested
    timeframe and return one ``IndicatorSnapshot``.

    Parameters
    ----------
    symbol
        Logical symbol (used only for the snapshot field, not for any
        lookup). Matches whatever convention the caller uses.
    bars
        Tz-aware DatetimeIndex OHLCV DataFrame at ``source_timeframe``.
        Lowercase column names. Source 5m is the production default;
        1m will work once the bot starts persisting bars_1m.
    daily_df
        Daily-cadence OHLCV with rows indexed by date (or midnight
        timestamps). Required for ``previous_day_hlc`` / ``pivot_points``;
        intraday-only indicators ignore it.
    session_date
        Trading session being scored. Drives level computations and is
        recorded on the snapshot.
    source_timeframe
        Cadence of ``bars``. Resampling to coarser ``target_timeframes``
        is identity-safe (5m → 5m is a no-op aggregation).
    target_timeframes
        Which intraday TFs to compute MTF-applicable indicators at. A
        registry entry whose ``timeframes`` field doesn't include a
        requested TF is skipped for that TF.
    indicators
        Restrict to a subset of REGISTRY names. ``None`` = all.

    Returns
    -------
    IndicatorSnapshot — see the dataclass docstring."""
    snapshot = IndicatorSnapshot(
        symbol=symbol,
        session_date=session_date,
        computed_at=datetime.now(timezone.utc),
    )
    names = list(indicators) if indicators is not None else list(REGISTRY.keys())
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown indicator: {n!r}")

    # Pre-resample to each target timeframe once per snapshot so each
    # indicator's per-TF call doesn't re-aggregate.
    tf_frames: dict[str, pd.DataFrame] = {}
    for tf in target_timeframes:
        try:
            if tf == source_timeframe:
                tf_frames[tf] = bars
            else:
                tf_frames[tf] = resample_ohlcv(bars, tf)
        except Exception as e:
            snapshot.warnings.append(f"resample {source_timeframe}→{tf}: {e}")

    for name in names:
        spec = REGISTRY[name]

        if spec.category == "level":
            _compute_level(spec, snapshot, bars, daily_df, session_date)
            continue

        for tf in target_timeframes:
            if tf not in spec.timeframes:
                continue
            frame = tf_frames.get(tf)
            if frame is None or frame.empty:
                _mark_insufficient(spec, tf, snapshot)
                continue
            _compute_one(spec, tf, frame, snapshot)

    return snapshot


# ---------------------------------------------------------------------------
# Per-indicator dispatch
# ---------------------------------------------------------------------------

def _compute_one(
    spec: IndicatorSpec,
    tf: str,
    frame: pd.DataFrame,
    snapshot: IndicatorSnapshot,
) -> None:
    try:
        out = spec.func(frame, **spec.default_params)
    except Exception as e:
        msg = f"{spec.name}_{tf}: {type(e).__name__}: {e}"
        log.warning(msg)
        snapshot.warnings.append(msg)
        if spec.output_kind == "series":
            snapshot.values[_series_key(spec, tf)] = None
        else:
            for col in spec.output_keys:
                snapshot.values[_frame_key(spec, tf, col)] = None
        return

    if spec.output_kind == "series":
        key = _series_key(spec, tf)
        v = _last_scalar(out)
        snapshot.values[key] = v
        if v is None:
            snapshot.insufficient.append(key)
        return

    if spec.output_kind == "scalar_dict":
        # Non-level scalar-dict indicators (e.g. visible_average_price)
        # return a flat {col: value} mapping, NOT a bar-indexed frame.
        # Namespace each column under the {name}_{tf}_{col} key so the
        # snapshot stays consistent with frame-output indicators.
        for col in spec.output_keys:
            v = out.get(col) if isinstance(out, dict) else None
            key = _frame_key(spec, tf, col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                snapshot.values[key] = None
                snapshot.insufficient.append(key)
            else:
                snapshot.values[key] = float(v)
        return

    # frame
    cols = _last_row_dict(out, spec.output_keys)
    for col, v in cols.items():
        key = _frame_key(spec, tf, col)
        snapshot.values[key] = v
        if v is None:
            snapshot.insufficient.append(key)


def _compute_level(
    spec: IndicatorSpec,
    snapshot: IndicatorSnapshot,
    bars: pd.DataFrame,
    daily_df: pd.DataFrame,
    session_date: date,
) -> None:
    """Levels return scalar dicts, not bar-indexed Series/Frames. Their
    inputs vary by indicator: PDH/pivots need daily_df, ORB/IB need
    intraday bars. Dispatch by ``spec.name``."""
    try:
        if spec.name == "previous_day_hlc":
            result = spec.func(daily_df, session_date=session_date,
                               **spec.default_params)
        elif spec.name == "pivot_points":
            from .levels import previous_day_hlc as _pdh
            pdhlc = _pdh(daily_df, session_date)
            result = spec.func(pdhlc, **spec.default_params)
        elif spec.name == "opening_range":
            result = spec.func(bars, session_date=session_date,
                               **spec.default_params)
        elif spec.name == "initial_balance":
            result = spec.func(bars, session_date=session_date,
                               **spec.default_params)
        else:
            raise NotImplementedError(
                f"_compute_level missing dispatch for {spec.name}"
            )
    except Exception as e:
        msg = f"{spec.name}: {type(e).__name__}: {e}"
        log.warning(msg)
        snapshot.warnings.append(msg)
        for col in spec.output_keys:
            snapshot.values[_level_key(spec, col)] = None
        return

    for col in spec.output_keys:
        v = result.get(col) if isinstance(result, dict) else None
        key = _level_key(spec, col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            snapshot.values[key] = None
            snapshot.insufficient.append(key)
        else:
            snapshot.values[key] = float(v)


def _mark_insufficient(
    spec: IndicatorSpec, tf: str, snapshot: IndicatorSnapshot,
) -> None:
    if spec.output_kind == "series":
        key = _series_key(spec, tf)
        snapshot.values[key] = None
        snapshot.insufficient.append(key)
    else:
        for col in spec.output_keys:
            key = _frame_key(spec, tf, col)
            snapshot.values[key] = None
            snapshot.insufficient.append(key)
