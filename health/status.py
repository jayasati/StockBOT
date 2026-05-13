"""Format the /status Telegram reply.

Reads the latest result per check from ``health_log`` and annotates with
the live ``FailureTracker`` state (consecutive-failure count) when it
is provided.

Phase 9: also pulls the cached ``MarketContext`` (VIX / FII / DII / PCR)
and appends a macro-context block so the operator can see the inputs
the scoring engine is currently consuming."""
from __future__ import annotations

import asyncio
import logging

from .alerter import FailureTracker
from .storage import last_failure_ts, latest_per_check

log = logging.getLogger("alertbot.health.status")


def format_status(tracker: FailureTracker | None = None) -> str:
    """Build the /status reply body."""
    rows = latest_per_check()
    lines: list[str]
    if not rows:
        lines = ["📊 <b>Health</b>", "(no checks yet)"]
    else:
        lines = ["📊 <b>Health status</b>"]
        for name in sorted(rows):
            ok, detail, _ts = rows[name]
            emoji = "✅" if ok else "❌"
            line = f"{emoji} <b>{name}</b>"
            if not ok:
                line += f" — {detail}"
                last_fail = last_failure_ts(name)
                if last_fail is not None:
                    line += f" (since {last_fail.strftime('%H:%M:%S')})"
                if tracker is not None:
                    count = tracker.consecutive_failures(name)
                    if count > 0:
                        line += f" [{count}× in a row]"
            lines.append(line)

    # Phase-9 market-context block. Best-effort: a fetch failure
    # silently omits the section so /status stays useful even when
    # NSE is unreachable.
    market_section = _market_section()
    if market_section:
        lines.append("")
        lines.append(market_section)
    return "\n".join(lines)


def _market_section() -> str | None:
    """Pull the latest MarketContext and format. We do this inside
    /status (a Telegram-handler context) rather than caching on the
    health module so the section always reflects current data;
    ``data.nse``'s 60s cache makes that cheap."""
    try:
        from data.market_context import format_for_status, get_market_context
    except Exception:
        log.exception("market_context import failed")
        return None
    try:
        ctx = asyncio.run(get_market_context())
    except RuntimeError:
        # Already inside an event loop — /status is called from the
        # Telegram-commands consumer which IS on-loop. Fall back to
        # a synchronous best-effort path that just hits the cache.
        try:
            from data import nse
            from data.market_context import MarketContext
            cached_quote = nse._cache_get("all_indices", None)
            cached_fii = nse._cache_get("fii_dii", None)
            cached_oc = nse._cache_get("option_chain", {"symbol": "NIFTY"})
            from data import nse as _n
            ctx = MarketContext(
                nifty_change_pct=(_n.parse_index_quote(cached_quote, "NIFTY 50") or {}).get("change_pct"),
                banknifty_change_pct=(_n.parse_index_quote(cached_quote, "NIFTY BANK") or {}).get("change_pct"),
                vix=_n.parse_vix(cached_quote),
                fii_net_cr=(_n.parse_fii_dii(cached_fii) or {}).get("fii_net_cr"),
                dii_net_cr=(_n.parse_fii_dii(cached_fii) or {}).get("dii_net_cr"),
                pcr_nifty=_n.parse_pcr(cached_oc),
            )
        except Exception:
            log.exception("market_context cache read failed")
            return None
    except Exception:
        log.exception("market_context fetch failed")
        return None
    return format_for_status(ctx)
