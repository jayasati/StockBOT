"""Indicator metadata registry.

REGISTRY is the single source of truth that Phase 7 scoring iterates to
weight every signal. Each entry binds the canonical name to the function,
its default parameters, the shape of its output (series / frame /
scalar_dict), and metadata describing how the score engine should
interpret it (bullish_high / bullish_low / two_sided / binary / level).

This module is the only place that knows how to map "rsi" → the rsi
function and its metadata. To add a new indicator: (1) write the function
in trend/momentum/volatility/volume/levels.py, (2) register it here, and
(3) add a hand-computed test in tests/test_indicators.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

OutputKind = Literal["series", "frame", "scalar_dict"]
Category = Literal["trend", "momentum", "volatility", "volume", "level"]
Direction = Literal[
    "bullish_high",   # higher value = more bullish (e.g. RSI, RS strength)
    "bullish_low",    # lower value = more bullish (e.g. Williams %R near -80)
    "two_sided",      # zero-cross or band-cross matters (e.g. MACD histogram)
    "binary",         # 0/1 signal (e.g. supertrend direction, in_squeeze)
    "level",          # absolute price level (pivots, PDH/PDL)
]
Normalize = Literal["none", "zscore_20", "minmax", "percent"]


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    func: Callable[..., Any]
    default_params: dict[str, Any]
    output_kind: OutputKind
    output_keys: tuple[str, ...]
    warmup_bars: int
    category: Category
    direction: Direction
    bullish_thresh: float | None = None
    bearish_thresh: float | None = None
    normalize: Normalize = "none"
    timeframes: tuple[str, ...] = ("5m", "15m", "60m", "1d")


# REGISTRY is populated at module-import time by _build_registry() to keep
# the canonical wiring in one place. Indicator modules export pure
# functions; the registry attaches metadata.
REGISTRY: dict[str, IndicatorSpec] = {}


def _build_registry() -> dict[str, IndicatorSpec]:
    # Local imports to break the cycle (those modules don't import registry).
    from . import levels, momentum, trend, volatility, volume

    specs: list[IndicatorSpec] = [
        # ── trend ────────────────────────────────────────────────────────
        IndicatorSpec(
            name="sma", func=trend.sma,
            default_params={"period": 20},
            output_kind="series", output_keys=("sma",),
            warmup_bars=20, category="trend", direction="bullish_high",
        ),
        IndicatorSpec(
            name="ema", func=trend.ema,
            default_params={"period": 20},
            output_kind="series", output_keys=("ema",),
            warmup_bars=20, category="trend", direction="bullish_high",
        ),
        IndicatorSpec(
            name="supertrend", func=trend.supertrend,
            default_params={"period": 10, "multiplier": 3.0},
            output_kind="frame", output_keys=("supertrend", "direction"),
            warmup_bars=10, category="trend", direction="binary",
        ),
        IndicatorSpec(
            name="adx", func=trend.adx,
            default_params={"period": 14},
            output_kind="frame", output_keys=("adx", "di_plus", "di_minus"),
            warmup_bars=28, category="trend", direction="bullish_high",
            bullish_thresh=25.0, bearish_thresh=20.0,
        ),
        IndicatorSpec(
            name="wma", func=trend.wma,
            default_params={"period": 20},
            output_kind="series", output_keys=("wma",),
            warmup_bars=20, category="trend", direction="bullish_high",
        ),
        IndicatorSpec(
            name="hull_ma", func=trend.hull_ma,
            default_params={"period": 20},
            output_kind="series", output_keys=("hull_ma",),
            # WMA(20) valid from idx 19, then WMA-of-diff with sqrt(20)≈4
            # needs 4 valid diff values: idx 19,20,21,22 → first valid at 22.
            warmup_bars=23, category="trend", direction="bullish_high",
        ),
        IndicatorSpec(
            name="parabolic_sar", func=trend.parabolic_sar,
            default_params={"af": 0.02, "max_af": 0.2},
            output_kind="series", output_keys=("psar",),
            warmup_bars=2, category="trend", direction="level",
        ),
        IndicatorSpec(
            name="ichimoku", func=trend.ichimoku,
            default_params={"tenkan": 9, "kijun": 26, "senkou": 52},
            output_kind="frame",
            output_keys=("tenkan", "kijun", "senkou_a", "senkou_b", "chikou"),
            # senkou_b needs senkou bars then shift kijun forward = 52 + 26
            warmup_bars=78, category="trend", direction="bullish_high",
        ),
        IndicatorSpec(
            name="donchian", func=trend.donchian,
            default_params={"period": 20},
            output_kind="frame",
            output_keys=("upper", "middle", "lower", "width"),
            warmup_bars=20, category="trend", direction="level",
        ),
        IndicatorSpec(
            name="aroon", func=trend.aroon,
            default_params={"period": 14},
            output_kind="frame",
            output_keys=("aroon_up", "aroon_down", "oscillator"),
            warmup_bars=15, category="trend", direction="bullish_high",
            bullish_thresh=70.0, bearish_thresh=30.0,
        ),
        IndicatorSpec(
            name="choppiness_index", func=trend.choppiness_index,
            default_params={"period": 14},
            output_kind="series", output_keys=("ci",),
            warmup_bars=14, category="trend", direction="bullish_low",
            # ci > 61.8 = choppy (suppress trend), ci < 38.2 = trending
            bullish_thresh=38.2, bearish_thresh=61.8,
        ),
        # ── momentum ─────────────────────────────────────────────────────
        IndicatorSpec(
            name="rsi", func=momentum.rsi,
            default_params={"period": 14},
            output_kind="series", output_keys=("rsi",),
            warmup_bars=15, category="momentum", direction="bullish_high",
            bullish_thresh=60.0, bearish_thresh=40.0, normalize="minmax",
        ),
        IndicatorSpec(
            name="macd", func=momentum.macd,
            default_params={"fast": 12, "slow": 26, "signal": 9},
            output_kind="frame",
            output_keys=("macd", "signal", "histogram"),
            warmup_bars=35, category="momentum", direction="two_sided",
        ),
        IndicatorSpec(
            name="stochastic", func=momentum.stochastic,
            default_params={"k": 14, "d": 3, "smooth": 3},
            output_kind="frame", output_keys=("k", "d"),
            warmup_bars=17, category="momentum", direction="bullish_high",
            bullish_thresh=80.0, bearish_thresh=20.0,
        ),
        IndicatorSpec(
            name="mfi", func=momentum.mfi,
            default_params={"period": 14},
            output_kind="series", output_keys=("mfi",),
            warmup_bars=15, category="momentum", direction="bullish_high",
            bullish_thresh=80.0, bearish_thresh=20.0,
        ),
        IndicatorSpec(
            name="cci", func=momentum.cci,
            default_params={"period": 20},
            output_kind="series", output_keys=("cci",),
            # CCI is sma of |tp - sma(tp, period)|, period — that's TWO
            # nested SMAs, each needing `period` valid inputs, so first
            # valid output is at idx 2*period - 2 (39 for period=20).
            warmup_bars=39, category="momentum", direction="two_sided",
            bullish_thresh=100.0, bearish_thresh=-100.0,
        ),
        IndicatorSpec(
            name="roc", func=momentum.roc,
            default_params={"period": 12},
            output_kind="series", output_keys=("roc",),
            warmup_bars=13, category="momentum", direction="bullish_high",
        ),
        IndicatorSpec(
            name="williams_r", func=momentum.williams_r,
            default_params={"period": 14},
            output_kind="series", output_keys=("williams_r",),
            warmup_bars=14, category="momentum", direction="bullish_low",
            bullish_thresh=-20.0, bearish_thresh=-80.0,
        ),
        IndicatorSpec(
            name="awesome_oscillator", func=momentum.awesome_oscillator,
            default_params={},
            output_kind="series", output_keys=("ao",),
            warmup_bars=34, category="momentum", direction="two_sided",
        ),
        IndicatorSpec(
            name="trix", func=momentum.trix,
            default_params={"period": 15},
            output_kind="series", output_keys=("trix",),
            # e1 valid @14, e2 @28, e3 @42; trix needs e3[t] and e3[t-1]
            # → first valid at idx 43, so 44 bars to make the last bar valid.
            warmup_bars=44, category="momentum", direction="two_sided",
        ),
        IndicatorSpec(
            name="force_index", func=momentum.force_index,
            default_params={"period": 13},
            output_kind="series", output_keys=("force_index",),
            warmup_bars=14, category="momentum", direction="two_sided",
        ),
        # ── volatility ───────────────────────────────────────────────────
        IndicatorSpec(
            name="atr", func=volatility.atr,
            default_params={"period": 14},
            output_kind="series", output_keys=("atr",),
            warmup_bars=14, category="volatility", direction="two_sided",
        ),
        IndicatorSpec(
            name="bollinger", func=volatility.bollinger,
            default_params={"period": 20, "std": 2.0},
            output_kind="frame",
            output_keys=("upper", "middle", "lower", "bandwidth", "percent_b"),
            warmup_bars=20, category="volatility", direction="two_sided",
        ),
        IndicatorSpec(
            name="keltner", func=volatility.keltner,
            default_params={"period": 20, "mult": 2.0, "atr_period": 10},
            output_kind="frame", output_keys=("upper", "middle", "lower"),
            warmup_bars=20, category="volatility", direction="level",
        ),
        IndicatorSpec(
            name="ttm_squeeze", func=volatility.ttm_squeeze,
            default_params={"period": 20, "bb_std": 2.0, "kc_mult": 1.5,
                            "atr_period": 10},
            output_kind="frame", output_keys=("in_squeeze", "momentum"),
            warmup_bars=20, category="volatility", direction="binary",
        ),
        # ── volume ───────────────────────────────────────────────────────
        IndicatorSpec(
            name="obv", func=volume.obv,
            default_params={},
            output_kind="series", output_keys=("obv",),
            warmup_bars=1, category="volume", direction="bullish_high",
        ),
        IndicatorSpec(
            name="cmf", func=volume.cmf,
            default_params={"period": 20},
            output_kind="series", output_keys=("cmf",),
            warmup_bars=20, category="volume", direction="bullish_high",
            bullish_thresh=0.05, bearish_thresh=-0.05,
        ),
        IndicatorSpec(
            name="volume_surge_ratio", func=volume.volume_surge_ratio,
            default_params={"period": 20},
            output_kind="series", output_keys=("vsr",),
            warmup_bars=20, category="volume", direction="bullish_high",
            bullish_thresh=1.5,
        ),
        IndicatorSpec(
            name="rvol_tod", func=volume.rvol_tod,
            default_params={"lookback_days": 20},
            output_kind="series", output_keys=("rvol_tod",),
            # Needs lookback_days+1 sessions; bars-based warmup is a coarse
            # approximation (assumes ~75 5m bars per session).
            warmup_bars=75 * 21, category="volume", direction="bullish_high",
            bullish_thresh=2.0,
        ),
        IndicatorSpec(
            name="ad_line", func=volume.ad_line,
            default_params={},
            output_kind="series", output_keys=("ad_line",),
            warmup_bars=1, category="volume", direction="bullish_high",
        ),
        IndicatorSpec(
            name="vwap_sd_bands", func=volume.vwap_sd_bands,
            default_params={},
            output_kind="frame",
            output_keys=("plus_1", "minus_1", "plus_2", "minus_2",
                         "plus_3", "minus_3"),
            warmup_bars=1, category="volume", direction="level",
            timeframes=("5m",),
        ),
        IndicatorSpec(
            name="anchored_vwap", func=volume.anchored_vwap,
            default_params={},
            output_kind="series", output_keys=("avwap",),
            warmup_bars=1, category="volume", direction="bullish_high",
            timeframes=("5m",),
        ),
        # ── levels (scalar_dict per session) ─────────────────────────────
        IndicatorSpec(
            name="previous_day_hlc", func=levels.previous_day_hlc,
            default_params={}, output_kind="scalar_dict",
            output_keys=("pdh", "pdl", "pdc"),
            warmup_bars=1, category="level", direction="level",
            timeframes=("1d",),
        ),
        IndicatorSpec(
            name="opening_range", func=levels.opening_range,
            default_params={"minutes": 15},
            output_kind="scalar_dict", output_keys=("orh", "orl", "or_mid"),
            warmup_bars=3, category="level", direction="level",
            timeframes=("5m",),
        ),
        IndicatorSpec(
            name="initial_balance", func=levels.initial_balance,
            default_params={}, output_kind="scalar_dict",
            output_keys=("ib_high", "ib_low"),
            warmup_bars=12, category="level", direction="level",
            timeframes=("5m",),
        ),
        IndicatorSpec(
            name="pivot_points", func=levels.pivot_points,
            default_params={"method": "classic"},
            output_kind="scalar_dict",
            output_keys=("pivot", "r1", "r2", "r3", "s1", "s2", "s3"),
            warmup_bars=1, category="level", direction="level",
            timeframes=("1d",),
        ),
    ]

    return {spec.name: spec for spec in specs}


REGISTRY.update(_build_registry())


def get_indicator(name: str) -> Callable[..., Any]:
    """Return the bound function for a registered indicator."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"unknown indicator: {name!r}")
    return spec.func


def list_by_category(category: Category) -> list[IndicatorSpec]:
    """All specs in one category, useful for category-weighted scoring."""
    return [s for s in REGISTRY.values() if s.category == category]
