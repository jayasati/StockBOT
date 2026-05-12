"""Indicator + level engine.

Stage A delivery: ~13 hand-rolled indicators (TV-parity formulas) and
the session-level engine (PDH/PDL, ORB, IB, Pivots). The registry
exposes metadata so Phase 7 scoring can iterate indicators without
hardcoding names.

Stage B will add: trend (WMA, HMA, PSAR, Ichimoku, Donchian, Aroon,
Choppiness), momentum (MFI, CCI, ROC, Williams %R, AO, TRIX, Force
Index), volatility (Keltner, TTM Squeeze), volume (Volume Surge Ratio,
RVOL-TOD, A/D Line, VWAP-SD bands, Anchored VWAP).

Stage C will add: MTF resampling (``mtf.py``), the ``compute_all``
snapshot builder, and the bot.py refactor against the new stack."""
from __future__ import annotations

from .compute import IndicatorSnapshot, compute_all
from .mtf import resample_ohlcv
from .registry import (
    REGISTRY,
    Category,
    Direction,
    IndicatorSpec,
    Normalize,
    OutputKind,
    get_indicator,
    list_by_category,
)

__all__ = [
    "REGISTRY",
    "IndicatorSnapshot",
    "IndicatorSpec",
    "Category",
    "Direction",
    "Normalize",
    "OutputKind",
    "compute_all",
    "get_indicator",
    "list_by_category",
    "resample_ohlcv",
]
