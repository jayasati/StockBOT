"""Golden tests for costs.py.

The notional ₹1,59,320 is the notional from the §1 sample card (56 × ₹2,845).
Component-by-component arithmetic is verified to the rupee. If any rate in
costs.py drifts, this test fails loudly — which is the whole point of having
costs.py be the single source of truth.
"""

from datetime import time, datetime

import pytest

import costs as C


NOTIONAL = 159_320.0   # 56 × ₹2,845 — §1 sample card


# ---------------------------------------------------------------------------
# round_trip_cost
# ---------------------------------------------------------------------------

def test_round_trip_cost_components_to_the_rupee():
    """Hand-derived breakdown — see §3.5 rate table."""
    out = C.round_trip_cost(NOTIONAL)

    # Brokerage: min(20, 0.03% × 159320 = 47.796) per leg = 20 × 2 = 40
    assert out["brokerage"] == pytest.approx(40.00)
    # STT: 0.025% × 159320 (sell leg only)
    assert out["stt"] == pytest.approx(39.83)
    # Exchange: 0.00297% × turnover (2 × 159320)
    assert out["exchange"] == pytest.approx(9.46)
    # SEBI: ₹10 per crore × turnover (2 × 159320)
    assert out["sebi"] == pytest.approx(0.32)
    # Stamp: 0.003% × 159320 (buy leg only)
    assert out["stamp"] == pytest.approx(4.78)
    # GST: 18% × (brokerage + exchange + SEBI)
    assert out["gst"] == pytest.approx(8.96)
    # Total = sum of components
    assert out["total"] == pytest.approx(103.35)


def test_round_trip_cost_total_equals_sum_of_parts():
    out = C.round_trip_cost(NOTIONAL)
    parts = ["brokerage", "stt", "exchange", "sebi", "stamp", "gst"]
    assert out["total"] == pytest.approx(sum(out[p] for p in parts), abs=0.01)


def test_round_trip_cost_brokerage_caps_at_flat_20_for_large_notional():
    """At ₹6.7L per leg, 0.03% = ₹201 > ₹20 flat → cap kicks in, ₹40 round-trip."""
    out = C.round_trip_cost(670_000.0)
    assert out["brokerage"] == pytest.approx(40.00)


def test_round_trip_cost_brokerage_uses_pct_for_small_notional():
    """At ₹15k per leg, 0.03% = ₹4.5 < ₹20 → percentage branch wins."""
    out = C.round_trip_cost(15_000.0)
    assert out["brokerage"] == pytest.approx(2 * 0.0003 * 15_000.0)


def test_round_trip_cost_rejects_unknown_side():
    with pytest.raises(ValueError):
        C.round_trip_cost(NOTIONAL, side="fno")


# ---------------------------------------------------------------------------
# slippage_bps
# ---------------------------------------------------------------------------

def test_slippage_bps_largecap_midday():
    assert C.slippage_bps("largecap", time(11, 0)) == 3


def test_slippage_bps_midcap_midday():
    assert C.slippage_bps("midcap", time(11, 0)) == 8


def test_slippage_bps_first_15_min_largecap():
    """09:15–09:30 → +5 bps."""
    assert C.slippage_bps("largecap", time(9, 20)) == 8


def test_slippage_bps_last_15_min_largecap():
    """15:15–15:30 → +5 bps."""
    assert C.slippage_bps("largecap", time(15, 20)) == 8


def test_slippage_bps_boundary_at_0930_is_mid_session():
    """09:30 is the start of the liquid window — penalty no longer applies."""
    assert C.slippage_bps("largecap", time(9, 30)) == 3


def test_slippage_bps_first_15_min_midcap():
    assert C.slippage_bps("midcap", time(9, 20)) == 13


def test_slippage_bps_accepts_datetime():
    assert C.slippage_bps("largecap", datetime(2026, 5, 1, 11, 0)) == 3


def test_slippage_bps_no_time_means_no_penalty():
    assert C.slippage_bps("largecap") == 3


def test_slippage_bps_unknown_class_raises():
    with pytest.raises(ValueError):
        C.slippage_bps("smallcap", time(11, 0))


# ---------------------------------------------------------------------------
# net_r_multiple — slippage on FILLS, costs on round-trip
# ---------------------------------------------------------------------------

def test_net_r_multiple_long_winner_at_t1():
    """§1 sample card: entry 2845, stop 2828, T1 2876, qty 56, largecap, mid-session.

    Risk = 17 × 56 = ₹952. Pre-cost gross = (2876-2845) × 56 = ₹1736 (1.82R).
    After 3-bps slippage on fills and round-trip costs, net R drops by a
    quantifiable amount — anything wildly different from this means costs.py
    and the live R math have drifted apart.
    """
    r = C.net_r_multiple(
        entry=2845.0, stop=2828.0, exit_price=2876.0, qty=56,
        symbol_class="largecap", t_entry=time(11, 0), t_exit=time(11, 30),
    )
    # Hand check:
    # entry_slip = 3/10000 * 2845 = 0.85365
    # exit_slip  = 3/10000 * 2876 = 0.86280
    # gross = (2876 - 0.86280 - 2845 - 0.85365) * 56 = 29.28355 * 56 = 1639.879
    # costs = round_trip_cost(2845*56=159320)['total'] = 103.35
    # net = 1639.879 - 103.35 = 1536.529
    # risk = 17 * 56 = 952
    # R = 1536.529 / 952 ≈ 1.6140
    assert r == pytest.approx(1.6140, abs=0.001)


def test_net_r_multiple_long_loser_at_stop():
    """At stop, net R should be roughly -1 minus cost drag — never better than -1."""
    r = C.net_r_multiple(
        entry=2845.0, stop=2828.0, exit_price=2828.0, qty=56,
        symbol_class="largecap", t_entry=time(11, 0), t_exit=time(11, 30),
    )
    assert r < -1.0
    assert r > -1.30   # cost drag, not a catastrophe


def test_net_r_multiple_zero_risk_returns_zero():
    r = C.net_r_multiple(
        entry=100.0, stop=100.0, exit_price=110.0, qty=10,
        symbol_class="largecap", t_entry=time(11, 0), t_exit=time(11, 30),
    )
    assert r == 0.0


def test_net_r_multiple_edge_of_day_slippage_hurts_more():
    """Same trade entered at 09:20 (first 15 min) should net less R than at 11:00
    because slippage is +5 bps each leg in the edge-of-day window."""
    mid = C.net_r_multiple(
        entry=2845.0, stop=2828.0, exit_price=2876.0, qty=56,
        symbol_class="largecap", t_entry=time(11, 0), t_exit=time(11, 30),
    )
    edge = C.net_r_multiple(
        entry=2845.0, stop=2828.0, exit_price=2876.0, qty=56,
        symbol_class="largecap", t_entry=time(9, 20), t_exit=time(15, 20),
    )
    assert edge < mid
