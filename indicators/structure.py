"""Phase-8 structure indicators — Central Pivot Range, Fibonacci
retracement/extension, and a bundler that packs everything Phase 7's
``score_structure`` needs into one dict for daily caching.

Why a sibling module rather than extending ``indicators/levels.py``:
``levels.py`` is registered in ``indicators/REGISTRY`` and is called
per-bar inside ``compute_all``. Structure values are session-constant —
they only change at the open of a new trading day. Phase 8 moves them
to a 09:00 IST precompute path that writes ``daily_levels`` (SQLite);
keeping them out of the bar-level registry is the point.

Floor-pivot ``pivot_points_floor`` is a scalar-args twin of
``indicators.levels.pivot_points(prev_day_hlc=dict)``. We keep both:
the dict form is what the registry-driven snapshot machinery already
uses; the scalar form is what the precompute path passes in. Output
of both is identical for ``method='classic'``.
"""
from __future__ import annotations

import math
from typing import Literal


# ---------------------------------------------------------------------------
# Floor pivot — scalar-args form (mirrors levels.pivot_points classic mode)
# ---------------------------------------------------------------------------

def pivot_points_floor(
    prev_high: float, prev_low: float, prev_close: float,
) -> dict[str, float]:
    """Classic floor-pivot levels from prior-day H/L/C.

    Formulas (TradingView default):
      pivot = (H + L + C) / 3
      r1 = 2·pivot − L          s1 = 2·pivot − H
      r2 = pivot + (H − L)      s2 = pivot − (H − L)
      r3 = H + 2·(pivot − L)    s3 = L − 2·(H − pivot)

    Returns ``{pivot, r1, r2, r3, s1, s2, s3}`` as floats."""
    pdh, pdl, pdc = float(prev_high), float(prev_low), float(prev_close)
    p = (pdh + pdl + pdc) / 3.0
    rng = pdh - pdl
    return {
        "pivot": p,
        "r1": 2.0 * p - pdl,
        "r2": p + rng,
        "r3": pdh + 2.0 * (p - pdl),
        "s1": 2.0 * p - pdh,
        "s2": p - rng,
        "s3": pdl - 2.0 * (pdh - p),
    }


# ---------------------------------------------------------------------------
# Central Pivot Range (CPR)
# ---------------------------------------------------------------------------

NARROW_CPR_PCT = 0.005
"""(TC − BC) / pivot below this fraction → 'narrow CPR' (commonly read
as a trending-day setup). 0.5% is the conventional Indian-equity
threshold used in popular CPR-trading frameworks."""


def cpr(
    prev_high: float, prev_low: float, prev_close: float,
) -> dict[str, float]:
    """Central Pivot Range.

    Formulas:
      pivot = (H + L + C) / 3      (same as floor pivot)
      bc    = (H + L) / 2          (Bottom-Central — the BC line)
      tc    = 2·pivot − bc          (Top-Central, reflected across pivot)
      width = tc − bc               (may be negative for some HLC
                                     configurations — see note below)

    Width sign quirk: the standard CPR construction yields ``tc > bc``
    in most cases, but when ``close > (high+low)/2`` (close near the
    day's high) the reflection puts ``tc`` BELOW ``bc``. The
    ``width`` field carries the absolute value; ``narrow`` is the
    convention-matching ``|tc − bc| / pivot < 0.5%``.

    Returns ``{pivot, bc, tc, width, width_pct, narrow}``.
    ``width_pct`` is ``|tc − bc| / pivot`` (float, 0..~0.1 typical).
    ``narrow`` is ``True`` when ``width_pct < NARROW_CPR_PCT``."""
    pdh, pdl, pdc = float(prev_high), float(prev_low), float(prev_close)
    p = (pdh + pdl + pdc) / 3.0
    bc = (pdh + pdl) / 2.0
    tc = 2.0 * p - bc
    width = abs(tc - bc)
    width_pct = width / p if p > 0 else float("nan")
    return {
        "pivot": p,
        "bc": bc,
        "tc": tc,
        "width": width,
        "width_pct": width_pct,
        "narrow": width_pct < NARROW_CPR_PCT,
    }


# ---------------------------------------------------------------------------
# Fibonacci retracement / extension
# ---------------------------------------------------------------------------

# Keys are floats to keep the math obvious; serialised to strings for
# JSON storage. Caller indexes by ``levels["0.618"]`` after JSON
# round-trip (see :func:`bundle_daily_levels`).
RETRACEMENT_RATIOS: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXTENSION_RATIOS: tuple[float, ...] = (1.272, 1.618, 2.0, 2.618)


def fib_retracement(
    swing_high: float, swing_low: float,
) -> dict[str, float]:
    """Fibonacci retracement levels between a swing high and swing low.

    Standard ratios: 0.0 (= swing_high), 0.236, 0.382, 0.5, 0.618 (the
    golden retracement), 0.786, 1.0 (= swing_low). Level value at
    ratio r is ``swing_high − r × (swing_high − swing_low)``.

    Returns a dict keyed by stringified ratio so the result round-trips
    through JSON without float-key surprises."""
    hi, lo = float(swing_high), float(swing_low)
    rng = hi - lo
    return {f"{r:.3f}": hi - r * rng for r in RETRACEMENT_RATIOS}


def fib_extension(
    swing_high: float, swing_low: float,
    breakout_direction: Literal["up", "down"] = "up",
) -> dict[str, float]:
    """Fibonacci extension levels projecting past the swing.

    For ``breakout_direction='up'`` (LONG continuation): the level at
    ratio r is ``swing_high + (r − 1) × (swing_high − swing_low)``,
    so ratio 1.272 sits 27.2% of the swing range above the swing high.

    For ``breakout_direction='down'`` (SHORT continuation): mirrored —
    ``swing_low − (r − 1) × (swing_high − swing_low)``.

    Returns ``{ratio_str: level}`` for the standard set 1.272, 1.618,
    2.0, 2.618."""
    if breakout_direction not in ("up", "down"):
        raise ValueError(
            f"breakout_direction must be 'up' or 'down'; "
            f"got {breakout_direction!r}"
        )
    hi, lo = float(swing_high), float(swing_low)
    rng = hi - lo
    if breakout_direction == "up":
        return {f"{r:.3f}": hi + (r - 1.0) * rng for r in EXTENSION_RATIOS}
    return {f"{r:.3f}": lo - (r - 1.0) * rng for r in EXTENSION_RATIOS}


# ---------------------------------------------------------------------------
# Bundler — single dict for caching in daily_levels
# ---------------------------------------------------------------------------

def bundle_daily_levels(
    prev_high: float, prev_low: float, prev_close: float,
    *,
    swing_high: float | None = None,
    swing_low: float | None = None,
) -> dict[str, object]:
    """Pack every Phase-8 level into one JSON-serialisable dict.

    Shape:
      {
        "pdh", "pdl", "pdc":   prior-day H/L/C (top-level scalars)
        "pivot":               classic floor-pivot levels (dict)
        "cpr":                 CPR levels (dict)
        "fib_retracement":     fib retracement (dict, present iff
                               swing_high+swing_low supplied)
        "fib_extension_up":    fib extension (LONG continuation, same
                               condition)
        "fib_extension_down":  fib extension (SHORT continuation)
      }

    The fib block is optional because a brand-new symbol with no
    swing data shouldn't blow up the precompute — the rest of the
    bundle is still useful."""
    bundle: dict[str, object] = {
        "pdh": float(prev_high),
        "pdl": float(prev_low),
        "pdc": float(prev_close),
        "pivot": pivot_points_floor(prev_high, prev_low, prev_close),
        "cpr": cpr(prev_high, prev_low, prev_close),
    }
    if swing_high is not None and swing_low is not None:
        if not math.isfinite(swing_high) or not math.isfinite(swing_low):
            return bundle
        if swing_high <= swing_low:
            # Degenerate swing — skip fib rather than emit garbage.
            return bundle
        bundle["fib_retracement"] = fib_retracement(swing_high, swing_low)
        bundle["fib_extension_up"] = fib_extension(swing_high, swing_low, "up")
        bundle["fib_extension_down"] = fib_extension(swing_high, swing_low, "down")
        bundle["swing_high"] = float(swing_high)
        bundle["swing_low"] = float(swing_low)
    return bundle
