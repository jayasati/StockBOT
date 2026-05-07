"""Golden tests for risk.py.

The §1 sample-card row is the canonical test: ₹95k account, entry ₹2,845,
stop ₹2,828, 1% risk → 56 qty. If this drifts, the trade card's qty line
disagrees with what the bot actually intends — a silent sizing bug.
"""

import pytest

from trading import risk as R


# ---------------------------------------------------------------------------
# size_position
# ---------------------------------------------------------------------------

def test_size_position_matches_sample_card():
    """1% of ₹95k = ₹950; risk per share = ₹17; 950/17 = 55.88 → 56 (rounded)."""
    assert R.size_position(account_inr=95_000, entry=2845.0, stop=2828.0) == 56


def test_size_position_zero_risk_returns_zero():
    assert R.size_position(95_000, 2845.0, 2845.0) == 0


def test_size_position_inverted_stop_uses_absolute():
    """Shorts have stop > entry; the absolute distance still drives qty."""
    assert R.size_position(95_000, 2828.0, 2845.0) == 56


def test_size_position_explicit_risk_pct():
    # 0.5% of ₹95k = ₹475; 475/17 = 27.94 → 28
    assert R.size_position(95_000, 2845.0, 2828.0, risk_pct=0.005) == 28


def test_size_position_zero_account_returns_zero():
    assert R.size_position(0, 2845.0, 2828.0) == 0


# ---------------------------------------------------------------------------
# validate_trade
# ---------------------------------------------------------------------------

def test_validate_trade_below_floor_rejected():
    """Notional ₹14,000 is below the ₹15k floor — cost ratio would be punishing."""
    ok, reason = R.validate_trade(qty=10, entry=1400.0, account_inr=100_000)
    assert ok is False
    assert "floor" in reason.lower()


def test_validate_trade_above_capital_rejected():
    """Notional ₹2L > ₹1L account capital → leverage required, out of scope."""
    ok, reason = R.validate_trade(qty=200, entry=1000.0, account_inr=100_000)
    assert ok is False
    assert "capital" in reason.lower()


def test_validate_trade_in_range_passes():
    ok, reason = R.validate_trade(qty=50, entry=2000.0, account_inr=200_000)
    assert ok is True
    assert reason == ""


def test_validate_trade_zero_qty_rejected():
    ok, reason = R.validate_trade(qty=0, entry=2845.0, account_inr=100_000)
    assert ok is False


def test_validate_trade_at_15k_floor_passes():
    """Exactly ₹15,000 notional is allowed (boundary inclusive)."""
    ok, reason = R.validate_trade(qty=10, entry=1500.0, account_inr=100_000)
    assert ok is True


def test_validate_trade_at_capital_ceiling_passes():
    """Notional == account capital is allowed (boundary inclusive)."""
    ok, reason = R.validate_trade(qty=100, entry=1000.0, account_inr=100_000)
    assert ok is True
