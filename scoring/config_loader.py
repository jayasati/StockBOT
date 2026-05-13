"""scoring.yaml loader with mtime-based cache invalidation.

A bot tick may run hundreds of times per scan; re-parsing the yaml
every call is wasteful. We cache the parsed dict keyed on
``(path, mtime)`` and reload only when the file changes — so editing
``config/scoring.yaml`` between scans is picked up without restart.

Missing-file behaviour is intentional: the bundled ``DEFAULTS`` are
returned, so the bot keeps scoring even if the operator deletes the
yaml. A subsequent scan that finds the file restored will pick it up
on the next mtime check.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("alertbot.scoring.config")

DEFAULT_SCORING_PATH = Path("config/scoring.yaml")

# Authoritative defaults. Mirror config/scoring.yaml's shape so the
# bot stays functional without the file. Edits should be reflected
# in both places.
DEFAULTS: dict[str, Any] = {
    "weights": {
        "trend": 0.25,
        "momentum": 0.20,
        "volume": 0.15,
        "volatility": 0.10,
        "structure": 0.10,
        "market": 0.15,
        "news": 0.05,
    },
    "component_weights": {
        "trend": {
            "vwap_position": 0.25,
            "ema_stack": 0.20,
            "supertrend": 0.20,
            "adx": 0.15,
            "price_vs_pivot": 0.20,
        },
        "momentum": {
            "rsi_zone": 0.30,
            "macd_cross": 0.30,
            "stochastic": 0.15,
            "cci": 0.15,
            "mfi": 0.10,
        },
        "volume": {
            "volume_ratio": 0.40,
            "cmf": 0.25,
            "volume_surge_ratio": 0.20,
            "vwap_above": 0.15,
        },
        "volatility": {
            "atr_normalized": 0.40,
            "bollinger_position": 0.35,
            "ttm_squeeze": 0.25,
        },
        "structure": {
            "near_pivot": 0.35,
            "at_support_resistance": 0.35,
            "at_orb": 0.30,
        },
        "market": {
            "nifty_direction": 0.30,
            "bank_nifty_direction": 0.20,
            "vix_regime": 0.15,
            "fii_flow": 0.20,
            "pcr": 0.15,
        },
        "news": {
            "filing_signal": 1.0,
        },
    },
    "alert_threshold": 85,
}


_cache: dict[str, Any] = {"path": None, "mtime": None, "data": None}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the parsed scoring config. Falls back to ``DEFAULTS``
    when the file is missing or unparseable.

    Cache key is ``(absolute path, mtime)``; an mtime change forces a
    reparse. Pass a ``path`` explicitly in tests so each test bills
    its own cache slot."""
    p = Path(path) if path is not None else DEFAULT_SCORING_PATH
    abs_path = str(p.resolve()) if p.exists() else str(p)
    mtime = p.stat().st_mtime if p.exists() else None

    if (
        _cache["path"] == abs_path
        and _cache["mtime"] == mtime
        and _cache["data"] is not None
    ):
        return _cache["data"]

    if not p.exists():
        log.debug("scoring.yaml not found at %s; using DEFAULTS", p)
        data: dict[str, Any] = _merge_defaults({})
        _cache.update(path=abs_path, mtime=None, data=data)
        return data

    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; using DEFAULTS for scoring config")
        data = _merge_defaults({})
        _cache.update(path=abs_path, mtime=mtime, data=data)
        return data

    try:
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        log.exception("Failed to parse %s; using DEFAULTS", p)
        raw = {}

    data = _merge_defaults(raw)
    _cache.update(path=abs_path, mtime=mtime, data=data)
    return data


def get_alert_threshold(
    default: float | None = None, path: str | Path | None = None,
) -> float:
    """Return ``alert_threshold`` from scoring.yaml. ``default`` kicks
    in when the yaml is silent on the key — typically the caller
    passes ``settings.composite_threshold`` so the bot keeps a
    sensible gate even with a stripped-down config file."""
    cfg = load_config(path)
    value = cfg.get("alert_threshold")
    if value is None:
        return float(default) if default is not None else 85.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default) if default is not None else 85.0


def clear_cache() -> None:
    """Test helper — forget the cached config so the next
    ``load_config`` call rereads from disk."""
    _cache.update(path=None, mtime=None, data=None)


def _merge_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``raw`` over ``DEFAULTS`` so a yaml that omits
    a category still gets sensible weights. Top-level keys are
    deep-merged one level down (the only place we need it)."""
    merged: dict[str, Any] = {
        "weights": {**DEFAULTS["weights"], **(raw.get("weights") or {})},
        "component_weights": {},
        "alert_threshold": raw.get("alert_threshold", DEFAULTS["alert_threshold"]),
    }
    raw_cw = raw.get("component_weights") or {}
    for category, defaults in DEFAULTS["component_weights"].items():
        merged["component_weights"][category] = {
            **defaults, **(raw_cw.get(category) or {}),
        }
    return merged
