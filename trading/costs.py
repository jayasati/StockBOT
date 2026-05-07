"""
Single source of truth for transaction costs and slippage (3.5 of
`idea/playbook_architecture.md`).

Live R-multiple math AND the backtester both call this module. If the live
path and the backtest path computed costs separately they would silently
disagree; that's exactly the bug 3.5 was written to prevent.

Rates assume NSE cash equity, intraday MIS, post-2024 charges.
"""

from __future__ import annotations

from datetime import datetime, time

from bot.schedule import SESSION_CLOSE, SESSION_OPEN

# ---------------------------------------------------------------------------
# Statutory / broker rates (§3.5)
# ---------------------------------------------------------------------------

BROKERAGE_FLAT = 20.0           # ₹ per executed order, discount-broker assumption
BROKERAGE_PCT = 0.0003          # 0.03% of leg notional (whichever is lower)
STT_SELL_PCT = 0.00025          # 0.025% on sell leg only
EXCHANGE_TXN_PCT = 0.0000297    # 0.00297% on turnover (both legs)
SEBI_PER_CRORE = 10.0           # ₹10 per crore on turnover (both legs)
STAMP_BUY_PCT = 0.00003         # 0.003% on buy leg only
GST_PCT = 0.18                  # 18% on (brokerage + exchange + SEBI)

# ---------------------------------------------------------------------------
# Slippage assumptions (3.5, in basis points per leg)
# ---------------------------------------------------------------------------

LARGECAP_SLIPPAGE_BPS = 3
MIDCAP_SLIPPAGE_BPS = 8
EDGE_OF_DAY_PENALTY_BPS = 5     # +5 bps in first / last 15 min

LIQUID_OPEN = time(9, 30)       # first-15-min penalty applies before this
LIQUID_CLOSE = time(15, 15)     # last-15-min penalty applies from this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_time(t) -> time:
    if isinstance(t, time):
        return t
    if isinstance(t, datetime):
        return t.time()
    raise TypeError(f"time_of_day must be datetime.time or datetime, got {type(t).__name__}")


def _brokerage_for_leg(notional_inr: float) -> float:
    return min(BROKERAGE_FLAT, BROKERAGE_PCT * notional_inr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def round_trip_cost(notional_inr: float, side: str = "intraday_eq") -> dict:
    """Round-trip statutory + broker charges for a buy + sell at ``notional_inr``.

    Both legs assumed at the same notional (the §3.5 convention). Returns a
    dict with each component and ``total``, all rounded to two decimal paise.
    """
    if side != "intraday_eq":
        raise ValueError(f"unsupported side: {side!r}")
    if notional_inr <= 0:
        raise ValueError(f"notional_inr must be positive, got {notional_inr}")

    turnover = 2.0 * notional_inr
    brokerage = 2.0 * _brokerage_for_leg(notional_inr)
    stt = STT_SELL_PCT * notional_inr
    exchange = EXCHANGE_TXN_PCT * turnover
    sebi = SEBI_PER_CRORE * (turnover / 1.0e7)
    stamp = STAMP_BUY_PCT * notional_inr
    gst = GST_PCT * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
    }


def slippage_bps(symbol_class: str = "largecap", time_of_day=None) -> int:
    """Per-leg slippage in basis points.

    3 bps largecap, 8 bps midcap, +5 bps in the first 15 min (09:15–09:30) or
    last 15 min (15:15–15:30) of the session. ``time_of_day=None`` → no
    edge-of-day penalty (treats the leg as mid-session).
    """
    if symbol_class == "largecap":
        base = LARGECAP_SLIPPAGE_BPS
    elif symbol_class == "midcap":
        base = MIDCAP_SLIPPAGE_BPS
    else:
        raise ValueError(f"unknown symbol_class: {symbol_class!r}")

    if time_of_day is None:
        return base

    t = _to_time(time_of_day)
    in_first_15 = SESSION_OPEN <= t < LIQUID_OPEN
    in_last_15 = LIQUID_CLOSE <= t <= SESSION_CLOSE
    if in_first_15 or in_last_15:
        return base + EDGE_OF_DAY_PENALTY_BPS
    return base


def net_r_multiple(
    entry: float,
    stop: float,
    exit_price: float,
    qty: int,
    symbol_class: str,
    t_entry,
    t_exit,
) -> float:
    """Realised R-multiple after slippage on fills and round-trip costs.

    Long when ``entry > stop``, short when ``entry < stop``. Slippage is
    applied to the *fill prices* (not the signal prices) per §3.5; round-trip
    costs are subtracted from gross P&L.
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share == 0 or qty == 0:
        return 0.0

    is_long = entry > stop
    entry_slip = slippage_bps(symbol_class, t_entry) / 1e4 * entry
    exit_slip = slippage_bps(symbol_class, t_exit) / 1e4 * exit_price

    if is_long:
        eff_entry = entry + entry_slip
        eff_exit = exit_price - exit_slip
        gross = (eff_exit - eff_entry) * qty
    else:
        eff_entry = entry - entry_slip
        eff_exit = exit_price + exit_slip
        gross = (eff_entry - eff_exit) * qty

    notional = entry * qty
    costs = round_trip_cost(notional)["total"]
    net = gross - costs
    return float(net / (risk_per_share * qty))
