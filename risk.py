"""
Position sizing & trade validation (§3 row 7 of `idea/playbook_architecture.md`).

Sizing rule: 1% of account at risk per trade. Reject trades whose notional is
below ₹15k (cost ratio is too punishing) or above the available capital
(over-leveraged for cash MIS — leverage is out of scope for Phase 1).
"""

from __future__ import annotations

MIN_NOTIONAL_INR = 15_000
DEFAULT_RISK_PCT = 0.01


def size_position(
    account_inr: float, entry: float, stop: float, risk_pct: float = DEFAULT_RISK_PCT
) -> int:
    """Integer share qty such that ``|entry - stop| * qty ≈ risk_pct * account``.

    Round-to-nearest, matching the §1 sample card: ₹95k account, entry ₹2845,
    stop ₹2828 → 56 qty (risk per share ₹17, target risk ₹950, 950/17 = 55.88).
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or account_inr <= 0 or risk_pct <= 0:
        return 0
    risk_amount = account_inr * risk_pct
    return int(round(risk_amount / risk_per_share))


def validate_trade(qty: int, entry: float, account_inr: float) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is empty when ``ok`` is True."""
    if qty <= 0:
        return False, "qty must be positive"
    notional = qty * entry
    if notional < MIN_NOTIONAL_INR:
        return (
            False,
            f"notional ₹{notional:,.0f} below ₹{MIN_NOTIONAL_INR:,} floor",
        )
    if notional > account_inr:
        return (
            False,
            f"notional ₹{notional:,.0f} exceeds account capital ₹{account_inr:,.0f}",
        )
    return True, ""
