"""Phase-6 filter chain.

Composable hard + soft + time + event filters that run BEFORE the
``_dispatch`` step in :mod:`bot.scanner`. Hard filters kill signals
instantly (early-exit, no scoring effort wasted). Soft filters
accumulate confidence multipliers that demote the signal — the
post-multiplier ``confidence`` field is the Phase-5b threshold's
new comparison target.

Public surface:
  apply_filters(signals, ctx) -> StockSignals | None
      Returns the (mutated) signals if it passes hard filters and
      ``signals.confidence`` is set, or ``None`` when hard-killed.
      Writes a ``filter_audit`` row on hard-kill; the scanner writes
      the audit row for the pass-through case after the threshold
      decision.

  FilterContext
      Per-scan-tick shared state. Built once in ``scan_once`` /
      ``scan_symbol`` so individual filters don't refetch.

Module layout:
  filters/hard.py    — kill-or-pass filters
  filters/soft.py    — multiplier filters tied to indicators
  filters/time.py    — multiplier filters tied to wall-clock IST
  filters/event.py   — multiplier filters tied to a YAML event calendar
  filters/chain.py   — orchestrator + audit writer + FilterContext
"""
from __future__ import annotations

from .chain import FilterContext, apply_filters, register_filters

# Wire the chain's filter tuples from the sibling modules now that
# everything has finished loading. Module-level ``from .hard
# import HARD_FILTERS`` would have been circular; the runtime call
# here is safe.
register_filters()

__all__ = ["FilterContext", "apply_filters"]
