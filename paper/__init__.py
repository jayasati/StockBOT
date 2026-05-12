"""Paper-trading auto-tracker (Phase 5).

Every alert that crosses ``composite_threshold`` is recorded as a
simulated trade and followed via the realtime feed until SL/TP1/TP2
hits or the trade times out at end of the next session. The schema
sub-module owns DDL; ``tracker`` owns writes + the async monitor
loop; ``journal`` owns reads + the CLI report.

Public surface re-exported here is the back-compat seam other modules
should reach for: ``open_trade``, ``monitor``, ``close_manual``,
``ensure_paper_schema``."""
from __future__ import annotations

from .schema import ensure_paper_schema

__all__ = [
    "ensure_paper_schema",
]
