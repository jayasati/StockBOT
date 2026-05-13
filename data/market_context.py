"""Aggregated market context — VIX, FII/DII, PCR, NIFTY/BANKNIFTY
deltas — packaged as a single ``MarketContext`` dataclass.

This is the read-side surface that the scanner calls once per scan
tick (every 5 min in production). Internally it issues the underlying
fetches concurrently via ``asyncio.gather`` so the worst-case latency
is the slowest endpoint, not the sum of them.

Failure handling: every field is independently best-effort. If
``fetch_pcr`` errors, ``pcr_nifty`` is ``None`` and the rest of the
context populates normally. The scoring layer treats every None field
as neutral, never raising.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from core.config import IST

from . import nse

log = logging.getLogger("alertbot.market_context")


@dataclass
class MarketContext:
    """One-tick snapshot of macro inputs to the scoring engine.

    Field semantics:
      ``nifty_change_pct``    — today's %-change for NIFTY 50, from
                                 NSE's ``last`` vs ``previousClose``.
      ``banknifty_change_pct`` — same for BANK NIFTY.
      ``vix``                  — INDIA VIX last value.
      ``fii_net_cr``           — FII/FPI net buy-sell value in ₹ crore.
                                  Positive = net buying.
      ``dii_net_cr``           — DII equivalent.
      ``pcr_nifty``            — NIFTY option Put-Call OI ratio,
                                  nearest expiry. > 1.0 = put-heavy
                                  (oversold / bearish sentiment).
      ``timestamp``            — IST datetime the context was built.

    Every field defaults to ``None``/``[]`` so callers can use a
    partially-populated context the same way as a fully-populated one."""
    nifty_change_pct: float | None = None
    banknifty_change_pct: float | None = None
    vix: float | None = None
    fii_net_cr: float | None = None
    dii_net_cr: float | None = None
    pcr_nifty: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the scoring engine's ``market_context`` arg.
        Keys match the names ``score_market`` reads
        (``nifty_pct``, ``bank_nifty_pct``, ``vix``, ``fii_net_cr``,
        ``dii_net_cr``, ``pcr_nifty``)."""
        return {
            "nifty_pct": self.nifty_change_pct,
            "bank_nifty_pct": self.banknifty_change_pct,
            "vix": self.vix,
            "fii_net_cr": self.fii_net_cr,
            "dii_net_cr": self.dii_net_cr,
            "pcr_nifty": self.pcr_nifty,
            "timestamp": self.timestamp.isoformat(),
        }


async def _safe(coro, name: str):
    """Wrap a fetch coroutine so its exceptions become None. Used
    inside asyncio.gather so one slow/failing endpoint doesn't poison
    the rest of the context."""
    try:
        return await coro
    except Exception:
        log.exception("MarketContext fetch failed: %s", name)
        return None


async def get_market_context() -> MarketContext:
    """Fetch every macro input concurrently and pack into a
    ``MarketContext``. Per-endpoint failures are absorbed — the
    returned context is always usable, possibly with None fields.

    Cache-aware: every NSE fetch hits the 60s in-memory cache in
    ``data.nse`` first, so repeated calls inside a scan tick are
    cheap. Across-tick calls (5 min apart) bypass the cache and
    refresh."""
    # Gather all underlying fetches concurrently.
    (
        nifty_q,
        banknifty_q,
        vix,
        fii_dii,
        pcr,
    ) = await asyncio.gather(
        _safe(nse.fetch_nifty_quote(), "nifty_quote"),
        _safe(nse.fetch_banknifty_quote(), "banknifty_quote"),
        _safe(nse.fetch_vix(), "vix"),
        _safe(nse.fetch_fii_dii(), "fii_dii"),
        _safe(nse.fetch_pcr("NIFTY"), "pcr"),
    )

    return MarketContext(
        nifty_change_pct=(nifty_q or {}).get("change_pct"),
        banknifty_change_pct=(banknifty_q or {}).get("change_pct"),
        vix=vix,
        fii_net_cr=(fii_dii or {}).get("fii_net_cr"),
        dii_net_cr=(fii_dii or {}).get("dii_net_cr"),
        pcr_nifty=pcr,
    )


def format_for_status(ctx: MarketContext) -> str:
    """Build the section of the ``/status`` Telegram reply that
    summarises macro context. None values render as ``—`` so the
    operator immediately sees which endpoint is failing."""
    def _fmt(v: float | None, suffix: str = "", fmt: str = ".2f") -> str:
        return "—" if v is None else f"{v:{fmt}}{suffix}"

    return (
        "📈 <b>Market</b>\n"
        f"NIFTY {_fmt(ctx.nifty_change_pct, '%')}  |  "
        f"BankNifty {_fmt(ctx.banknifty_change_pct, '%')}\n"
        f"VIX {_fmt(ctx.vix)}  |  "
        f"PCR {_fmt(ctx.pcr_nifty)}\n"
        f"FII {_fmt(ctx.fii_net_cr, ' cr', '+.0f')}  |  "
        f"DII {_fmt(ctx.dii_net_cr, ' cr', '+.0f')}"
    )
