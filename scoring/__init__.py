"""Phase-7 weighted confidence scoring.

The engine reads weights from ``config/scoring.yaml`` (hot-reloadable
on mtime change) and composes seven category scores
(trend / momentum / volume / volatility / structure / market / news)
into one [0, 100] number, then applies the Phase-6 filter chain's
multipliers on top.

Public surface — keep narrow:
    score_signal(signal) -> ScoreBreakdown
    get_alert_threshold(default=...) -> float
    load_config(path=None) -> dict
"""
from __future__ import annotations

from .config_loader import (
    DEFAULT_SCORING_PATH,
    clear_cache,
    get_alert_threshold,
    load_config,
)
from .master import ScoreBreakdown, score_signal

__all__ = [
    "DEFAULT_SCORING_PATH",
    "ScoreBreakdown",
    "clear_cache",
    "get_alert_threshold",
    "load_config",
    "score_signal",
]
