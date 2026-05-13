"""Combine the seven category scores into a single confidence float
and apply the Phase-6 multiplier product.

The output is a ``ScoreBreakdown`` so the caller can persist the full
component table to ``filter_audit`` — turning every alert into a
debuggable record of WHY it crossed the threshold."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from . import components as comp
from .config_loader import load_config

if TYPE_CHECKING:
    from bot.scoring import StockSignals


@dataclass(frozen=True)
class ScoreBreakdown:
    """Full audit shape for one signal.

    ``components``         — raw 0..100 score per category.
    ``weighted``           — components × top-level weights.
    ``base``               — Σ weighted (the pre-multiplier score).
    ``multiplier_product`` — Π of every soft adjustment on the signal.
    ``final``              — base × product (the alert-gate number).
    ``alert_threshold``    — value used to decide the gate; captured
                             so the audit row records the live
                             threshold at decision time."""
    components: dict[str, float] = field(default_factory=dict)
    weighted: dict[str, float] = field(default_factory=dict)
    base: float = 0.0
    multiplier_product: float = 1.0
    final: float = 0.0
    alert_threshold: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_signal(
    signal: "StockSignals",
    *,
    config: dict[str, Any] | None = None,
    daily_levels: dict[str, Any] | None = None,
    news_score: float | None = None,
) -> ScoreBreakdown:
    """Compute the Phase-7 weighted confidence score for ``signal``.

    Reads from:
      * ``signal.snapshot``           — indicator snapshot
      * ``signal.market_context``     — nifty_pct, bank_nifty_pct, vix
      * ``signal.side``               — LONG / SHORT polarity
      * ``signal.price``              — for VWAP / pivot / S-R checks
      * ``signal.above_vwap``         — boolean trend signal
      * ``signal.volume_ratio``       — Phase-5 microstructure metric
      * ``signal.filing_title``       — Phase-10 news stub input
      * ``signal.soft_adjustments``   — multiplier product from chain

    ``config`` is normally ``None`` — the loader pulls scoring.yaml
    with mtime-cache invalidation. Tests can inject a config dict
    directly to bypass disk.

    ``daily_levels`` is the Phase-8 precomputed structure bundle for
    the signal's symbol. The scanner reads it from the daily_levels
    table once per signal; tests can pass it directly. When None,
    score_structure falls back to the snapshot.values path."""
    cfg = config if config is not None else load_config()

    weights = cfg.get("weights", {}) or {}
    cw = cfg.get("component_weights", {}) or {}
    threshold = float(cfg.get("alert_threshold", 85))

    market_ctx = signal.market_context or {}

    components: dict[str, float] = {
        "trend": comp.score_trend(
            signal.snapshot,
            market_ctx,
            signal.side,
            weights=cw.get("trend", {}),
            price=signal.price,
            above_vwap=signal.above_vwap,
        ),
        "momentum": comp.score_momentum(
            signal.snapshot,
            signal.side,
            weights=cw.get("momentum", {}),
        ),
        "volume": comp.score_volume(
            signal.snapshot,
            signal.volume_ratio,
            signal.above_vwap,
            signal.side,
            weights=cw.get("volume", {}),
        ),
        "volatility": comp.score_volatility(
            signal.snapshot,
            signal.side,
            weights=cw.get("volatility", {}),
            price=signal.price,
        ),
        "structure": comp.score_structure(
            signal.snapshot,
            signal.price,
            signal.side,
            weights=cw.get("structure", {}),
            daily_levels=daily_levels,
        ),
        "market": comp.score_market(
            market_ctx,
            signal.side,
            weights=cw.get("market", {}),
        ),
        "news": comp.score_news(
            signal.filing_title,
            signal.side,
            weights=cw.get("news", {}),
            news_score=news_score,
        ),
    }

    # Normalise top-level weights to 1.0 so users editing the yaml
    # can't accidentally push the base above 100 by setting weights
    # that sum to e.g. 1.5.
    total_w = sum(float(weights.get(k, 0.0)) for k in components) or 1.0
    weighted: dict[str, float] = {
        k: (components[k] * float(weights.get(k, 0.0)) / total_w)
        for k in components
    }
    base = sum(weighted.values())

    product = 1.0
    for _name, mult in signal.soft_adjustments or []:
        try:
            product *= float(mult)
        except (TypeError, ValueError):
            continue
    final = max(0.0, base * product)

    return ScoreBreakdown(
        components=components,
        weighted=weighted,
        base=base,
        multiplier_product=product,
        final=final,
        alert_threshold=threshold,
    )
