"""Event-day filters: heavy demotion on days where macro news
dominates microstructure. Multipliers per spec:
  rbi_policy_day        — × 0.2
  budget_day            — × 0.1
  election_results_day  — × 0.1

Dates come from ``config/events.yaml``. The file is loaded once
per Python process (module-level cache) — bot restart picks up
changes. Loader treats a missing file as empty (silent), so the
bot doesn't crash for an unfilled config."""
from __future__ import annotations

import logging
from datetime import date as _Date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.scoring import StockSignals
    from .chain import FilterContext


log = logging.getLogger("alertbot.filters.event")

EVENTS_YAML_PATH = Path("config/events.yaml")

RBI_MULT = 0.2
BUDGET_MULT = 0.1
ELECTION_MULT = 0.1

_EVENT_CACHE: dict[str, set[_Date]] | None = None


def _load_events() -> dict[str, set[_Date]]:
    """Read ``config/events.yaml`` once. Returns a dict of
    ``{event_kind: set_of_dates}``. Empty sets when the file is
    absent or the key is missing — never raises."""
    global _EVENT_CACHE
    if _EVENT_CACHE is not None:
        return _EVENT_CACHE

    empty = {
        "rbi_policy_days": set(),
        "budget_days": set(),
        "election_results_days": set(),
    }
    if not EVENTS_YAML_PATH.exists():
        log.debug("events.yaml not found at %s; event filters inert",
                  EVENTS_YAML_PATH)
        _EVENT_CACHE = empty
        return _EVENT_CACHE

    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; event filters inert")
        _EVENT_CACHE = empty
        return _EVENT_CACHE

    try:
        with EVENTS_YAML_PATH.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        log.exception("Failed to parse events.yaml; event filters inert")
        _EVENT_CACHE = empty
        return _EVENT_CACHE

    parsed = {}
    for key in ("rbi_policy_days", "budget_days", "election_results_days"):
        values = raw.get(key) or []
        dates: set[_Date] = set()
        for v in values:
            if isinstance(v, _Date):
                dates.add(v)
            elif isinstance(v, str):
                try:
                    dates.add(_Date.fromisoformat(v))
                except ValueError:
                    log.warning("events.yaml: skipping unparseable %s entry %r",
                                key, v)
        parsed[key] = dates
    _EVENT_CACHE = parsed
    return _EVENT_CACHE


def _reset_cache() -> None:
    """Clear the cache. Used by tests that swap ``EVENTS_YAML_PATH``."""
    global _EVENT_CACHE
    _EVENT_CACHE = None


def rbi_policy_day(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    today = ctx.now.date()
    if today in _load_events()["rbi_policy_days"]:
        return ("rbi_policy_day", RBI_MULT)
    return None


def budget_day(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    today = ctx.now.date()
    if today in _load_events()["budget_days"]:
        return ("budget_day", BUDGET_MULT)
    return None


def election_results_day(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    today = ctx.now.date()
    if today in _load_events()["election_results_days"]:
        return ("election_results_day", ELECTION_MULT)
    return None


EVENT_FILTERS = (
    rbi_policy_day,
    budget_day,
    election_results_day,
)
