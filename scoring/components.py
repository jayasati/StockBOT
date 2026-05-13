"""Per-category scorers. Each ``score_<category>`` function returns
a float in ``[0, 100]`` by computing named sub-criterion scores and
normalising by the local weight sum.

Missing snapshot keys (e.g. an indicator that hasn't reached warmup
yet) contribute a NEUTRAL 50 to their sub-criterion AND drop that
sub-criterion's weight from the normaliser, so the remaining
sub-criteria carry the full category. This keeps a brand-new symbol
with sparse indicator coverage from collapsing to zero just because
half the snapshot is still warming up.

Bullish/bearish polarity is flipped by ``side``: every sub-scorer
takes ``side`` ("LONG" | "SHORT") and inverts where appropriate
(``rsi`` near 30 is bullish for SHORT, near 70 for LONG, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from indicators.compute import IndicatorSnapshot


NEUTRAL = 50.0


def _val(snapshot: "IndicatorSnapshot | None", key: str) -> float | None:
    """Pull a value from the snapshot, returning None if the
    snapshot is absent, the key is missing, or the value is None."""
    if snapshot is None:
        return None
    return snapshot.values.get(key)


def _normalise(
    sub_scores: dict[str, float | None],
    sub_weights: dict[str, float],
) -> float:
    """Weighted average of available sub-scores. Missing keys (None
    sub-scores) DROP from the normaliser so the remaining weights
    re-sum to 1.0 internally. All-missing → ``NEUTRAL`` (50.0) so a
    category with zero indicator coverage doesn't drag the master
    score to zero."""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, raw in sub_scores.items():
        w = float(sub_weights.get(name, 0.0))
        if w <= 0 or raw is None:
            continue
        weighted_sum += raw * w
        total_weight += w
    if total_weight <= 0:
        return NEUTRAL
    return max(0.0, min(100.0, weighted_sum / total_weight))


# ---------------------------------------------------------------------------
# Helpers — common shape conversions
# ---------------------------------------------------------------------------

def _linear_map(x: float, lo: float, hi: float) -> float:
    """Map ``[lo, hi]`` → ``[0, 100]`` linearly, clipping outside."""
    if hi == lo:
        return 0.0
    pct = (x - lo) / (hi - lo)
    return max(0.0, min(100.0, pct * 100.0))


def _bullish_above(value: float, threshold: float) -> float:
    """1-bit: 100 if above threshold, 0 otherwise."""
    return 100.0 if value > threshold else 0.0


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def score_trend(
    snapshot: "IndicatorSnapshot | None",
    market_ctx: dict[str, Any],
    side: str,
    *,
    weights: dict[str, float],
    price: float | None = None,
    above_vwap: bool | None = None,
) -> float:
    """Trend strength composite. Sub-criteria match the brainstorm
    spec (vwap_position, ema_stack, supertrend, adx, price_vs_pivot)
    with safe proxies where a pure indicator isn't available — e.g.
    ``ema_stack`` uses the MACD histogram sign as a stand-in for
    "9/21/50 EMA ribbon aligned" until those EMAs land in the
    registry."""
    bull = (side == "LONG")

    # vwap_position — distance from session VWAP. The legacy scorer
    # tracks ``above_vwap`` directly on the signal; we use that when
    # provided so SHORT-side signals get the inverted polarity.
    if above_vwap is None:
        vwap_score = None
    elif bull:
        vwap_score = 100.0 if above_vwap else 0.0
    else:
        vwap_score = 0.0 if above_vwap else 100.0

    # ema_stack — proxy: MACD histogram sign (Phase 4 indicator
    # available now). Positive histogram = MACD>signal = fast EMA
    # above slow EMA. When more EMAs land in the registry this
    # becomes a proper 9/21/50 ribbon check.
    macd_hist = _val(snapshot, "macd_5m_histogram")
    if macd_hist is None:
        ema_score = None
    elif bull:
        ema_score = 100.0 if macd_hist > 0 else (50.0 if macd_hist > -0.1 else 0.0)
    else:
        ema_score = 100.0 if macd_hist < 0 else (50.0 if macd_hist < 0.1 else 0.0)

    # supertrend — binary direction
    st_dir = _val(snapshot, "supertrend_5m_direction")
    if st_dir is None:
        st_score = None
    elif bull:
        st_score = 100.0 if st_dir > 0 else 0.0
    else:
        st_score = 100.0 if st_dir < 0 else 0.0

    # adx — same polarity for both sides (trend strength)
    adx = _val(snapshot, "adx_5m")
    if adx is None:
        adx_score = None
    else:
        adx_score = _linear_map(adx, 0.0, 50.0)

    # price_vs_pivot — relative to classic pivot
    pivot = _val(snapshot, "pivot_classic")
    if pivot is None or price is None:
        pivot_score = None
    elif bull:
        pivot_score = 100.0 if price > pivot else 0.0
    else:
        pivot_score = 100.0 if price < pivot else 0.0

    return _normalise(
        {
            "vwap_position": vwap_score,
            "ema_stack": ema_score,
            "supertrend": st_score,
            "adx": adx_score,
            "price_vs_pivot": pivot_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def score_momentum(
    snapshot: "IndicatorSnapshot | None",
    side: str,
    *,
    weights: dict[str, float],
) -> float:
    bull = (side == "LONG")

    rsi = _val(snapshot, "rsi_5m")
    if rsi is None:
        rsi_score = None
    elif bull:
        # 60-70 ideal; 50-60 mild bullish; >75 overbought, <50 bearish
        if 60 <= rsi <= 70:
            rsi_score = 100.0
        elif 50 <= rsi < 60:
            rsi_score = 70.0
        elif 70 < rsi <= 75:
            rsi_score = 60.0
        elif rsi > 80:
            rsi_score = 0.0
        else:
            rsi_score = 25.0
    else:
        # SHORT: mirror — 30-40 ideal
        if 30 <= rsi <= 40:
            rsi_score = 100.0
        elif 40 < rsi <= 50:
            rsi_score = 70.0
        elif 25 <= rsi < 30:
            rsi_score = 60.0
        elif rsi < 20:
            rsi_score = 0.0
        else:
            rsi_score = 25.0

    macd_hist = _val(snapshot, "macd_5m_histogram")
    if macd_hist is None:
        macd_score = None
    elif bull:
        macd_score = 100.0 if macd_hist > 0 else 0.0
    else:
        macd_score = 100.0 if macd_hist < 0 else 0.0

    stoch_k = _val(snapshot, "stochastic_5m_k")
    if stoch_k is None:
        stoch_score = None
    elif bull:
        # 20-80 healthy; >80 overbought
        if stoch_k > 80:
            stoch_score = 30.0
        elif stoch_k >= 50:
            stoch_score = 100.0
        elif stoch_k >= 20:
            stoch_score = 50.0
        else:
            stoch_score = 10.0
    else:
        if stoch_k < 20:
            stoch_score = 30.0
        elif stoch_k <= 50:
            stoch_score = 100.0
        elif stoch_k <= 80:
            stoch_score = 50.0
        else:
            stoch_score = 10.0

    cci = _val(snapshot, "cci_5m")
    if cci is None:
        cci_score = None
    elif bull:
        if cci > 200:
            cci_score = 40.0
        elif cci >= 100:
            cci_score = 100.0
        elif cci >= 0:
            cci_score = 70.0
        else:
            cci_score = 20.0
    else:
        if cci < -200:
            cci_score = 40.0
        elif cci <= -100:
            cci_score = 100.0
        elif cci <= 0:
            cci_score = 70.0
        else:
            cci_score = 20.0

    mfi = _val(snapshot, "mfi_5m")
    if mfi is None:
        mfi_score = None
    elif bull:
        if mfi > 80:
            mfi_score = 30.0
        elif mfi >= 50:
            mfi_score = 100.0
        elif mfi >= 20:
            mfi_score = 50.0
        else:
            mfi_score = 10.0
    else:
        if mfi < 20:
            mfi_score = 30.0
        elif mfi <= 50:
            mfi_score = 100.0
        elif mfi <= 80:
            mfi_score = 50.0
        else:
            mfi_score = 10.0

    return _normalise(
        {
            "rsi_zone": rsi_score,
            "macd_cross": macd_score,
            "stochastic": stoch_score,
            "cci": cci_score,
            "mfi": mfi_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def score_volume(
    snapshot: "IndicatorSnapshot | None",
    volume_ratio: float | None,
    above_vwap: bool | None,
    side: str,
    *,
    weights: dict[str, float],
) -> float:
    bull = (side == "LONG")

    # volume_ratio — directionless: surge is good either way
    if volume_ratio is None:
        vr_score = None
    else:
        vr_score = _linear_map(volume_ratio, 1.0, 3.0)

    cmf = _val(snapshot, "cmf_5m")
    if cmf is None:
        cmf_score = None
    elif bull:
        cmf_score = 100.0 if cmf > 0.05 else (50.0 if cmf > 0 else 0.0)
    else:
        cmf_score = 100.0 if cmf < -0.05 else (50.0 if cmf < 0 else 0.0)

    vsr = _val(snapshot, "volume_surge_ratio_5m")
    if vsr is None:
        vsr_score = None
    else:
        vsr_score = _linear_map(vsr, 1.0, 3.0)

    if above_vwap is None:
        vwap_score = None
    elif bull:
        vwap_score = 100.0 if above_vwap else 0.0
    else:
        vwap_score = 0.0 if above_vwap else 100.0

    return _normalise(
        {
            "volume_ratio": vr_score,
            "cmf": cmf_score,
            "volume_surge_ratio": vsr_score,
            "vwap_above": vwap_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def score_volatility(
    snapshot: "IndicatorSnapshot | None",
    side: str,
    *,
    weights: dict[str, float],
    price: float | None = None,
) -> float:
    # atr_normalized: ATR / price — sweet spot ~1-3%
    atr = _val(snapshot, "atr_5m")
    if atr is None or price is None or price <= 0:
        atr_score = None
    else:
        atr_pct = (atr / price) * 100.0
        # Best around 1.5–2.5%; falls off either side
        if 1.0 <= atr_pct <= 3.0:
            atr_score = 100.0
        elif atr_pct < 1.0:
            atr_score = _linear_map(atr_pct, 0.2, 1.0)
        else:
            # too volatile is bad — penalise above 3%
            atr_score = max(0.0, 100.0 - (atr_pct - 3.0) * 20.0)

    # bollinger %B: 0.5-0.8 ideal for LONG continuation,
    #               0.2-0.5 ideal for SHORT.
    pb = _val(snapshot, "bollinger_5m_percent_b")
    if pb is None:
        bb_score = None
    elif side == "LONG":
        if 0.5 <= pb <= 0.8:
            bb_score = 100.0
        elif 0.4 <= pb < 0.5:
            bb_score = 70.0
        elif 0.8 < pb <= 1.0:
            bb_score = 60.0
        elif pb > 1.0:
            bb_score = 20.0  # breakout — could be parabolic
        else:
            bb_score = 30.0
    else:
        if 0.2 <= pb <= 0.5:
            bb_score = 100.0
        elif 0.5 < pb <= 0.6:
            bb_score = 70.0
        elif 0.0 <= pb < 0.2:
            bb_score = 60.0
        elif pb < 0.0:
            bb_score = 20.0
        else:
            bb_score = 30.0

    # ttm_squeeze: squeeze release in trade direction is bullish
    in_sq = _val(snapshot, "ttm_squeeze_5m_in_squeeze")
    sq_mom = _val(snapshot, "ttm_squeeze_5m_momentum")
    if in_sq is None or sq_mom is None:
        sq_score = None
    elif side == "LONG":
        if in_sq < 0.5 and sq_mom > 0:
            sq_score = 100.0
        elif in_sq > 0.5:
            sq_score = 60.0  # coiled, awaiting break
        else:
            sq_score = 20.0
    else:
        if in_sq < 0.5 and sq_mom < 0:
            sq_score = 100.0
        elif in_sq > 0.5:
            sq_score = 60.0
        else:
            sq_score = 20.0

    return _normalise(
        {
            "atr_normalized": atr_score,
            "bollinger_position": bb_score,
            "ttm_squeeze": sq_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# Structure (pivots / S-R / opening range)
# ---------------------------------------------------------------------------

def score_structure(
    snapshot: "IndicatorSnapshot | None",
    price: float | None,
    side: str,
    *,
    weights: dict[str, float],
) -> float:
    bull = (side == "LONG")
    # near_pivot — within 0.5% of pivot/r1/s1 (for LONG, near s1 or
    # above pivot toward r1 is bullish; for SHORT, the mirror)
    if price is None:
        near_score = None
    else:
        levels: list[float] = []
        for key in ("pivot_classic", "pivot_classic_r1",
                    "pivot_classic_s1", "pivot_classic_r2",
                    "pivot_classic_s2"):
            v = _val(snapshot, key)
            if v is not None:
                levels.append(v)
        if not levels:
            near_score = None
        else:
            dists = [abs(price - lv) / max(price, 1e-9) for lv in levels]
            min_dist = min(dists)
            if min_dist <= 0.005:
                near_score = 100.0
            elif min_dist <= 0.01:
                near_score = 60.0
            else:
                near_score = 20.0

    # at_support_resistance — proximity to PDH/PDL
    if price is None:
        sr_score = None
    else:
        pdh = _val(snapshot, "pdh")
        pdl = _val(snapshot, "pdl")
        if pdh is None and pdl is None:
            sr_score = None
        else:
            best = None
            if bull and pdh is not None:
                # LONG: hovering just under PDH OR breaking above
                dist = (pdh - price) / max(price, 1e-9)
                if -0.005 <= dist <= 0.01:
                    best = 100.0
                elif dist > 0.01:
                    best = 50.0
                else:
                    best = 30.0
            elif (not bull) and pdl is not None:
                dist = (price - pdl) / max(price, 1e-9)
                if -0.005 <= dist <= 0.01:
                    best = 100.0
                elif dist > 0.01:
                    best = 50.0
                else:
                    best = 30.0
            sr_score = best if best is not None else 30.0

    # at_orb — break of opening range high/low in trade direction
    if price is None:
        orb_score = None
    else:
        orh = _val(snapshot, "orh_15")
        orl = _val(snapshot, "orl_15")
        if orh is None and orl is None:
            orb_score = None
        elif bull:
            if orh is not None and price > orh:
                orb_score = 100.0
            elif orh is not None and price >= orh * 0.998:
                orb_score = 70.0
            else:
                orb_score = 30.0
        else:
            if orl is not None and price < orl:
                orb_score = 100.0
            elif orl is not None and price <= orl * 1.002:
                orb_score = 70.0
            else:
                orb_score = 30.0

    return _normalise(
        {
            "near_pivot": near_score,
            "at_support_resistance": sr_score,
            "at_orb": orb_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

def score_market(
    market_ctx: dict[str, Any],
    side: str,
    *,
    weights: dict[str, float],
) -> float:
    bull = (side == "LONG")

    nifty_pct = market_ctx.get("nifty_pct")
    if nifty_pct is None:
        nifty_score = None
    elif bull:
        if nifty_pct >= 0.5:
            nifty_score = 100.0
        elif nifty_pct >= 0:
            nifty_score = 70.0
        elif nifty_pct >= -0.5:
            nifty_score = 40.0
        else:
            nifty_score = 10.0
    else:
        if nifty_pct <= -0.5:
            nifty_score = 100.0
        elif nifty_pct <= 0:
            nifty_score = 70.0
        elif nifty_pct <= 0.5:
            nifty_score = 40.0
        else:
            nifty_score = 10.0

    bn_pct = market_ctx.get("bank_nifty_pct")
    if bn_pct is None:
        bn_score = None
    elif bull:
        if bn_pct >= 0.5:
            bn_score = 100.0
        elif bn_pct >= 0:
            bn_score = 70.0
        elif bn_pct >= -0.5:
            bn_score = 40.0
        else:
            bn_score = 10.0
    else:
        if bn_pct <= -0.5:
            bn_score = 100.0
        elif bn_pct <= 0:
            bn_score = 70.0
        elif bn_pct <= 0.5:
            bn_score = 40.0
        else:
            bn_score = 10.0

    vix = market_ctx.get("vix")
    if vix is None:
        vix_score = None
    elif vix < 12:
        vix_score = 90.0
    elif vix < 16:
        vix_score = 100.0
    elif vix < 20:
        vix_score = 50.0
    else:
        vix_score = 20.0

    return _normalise(
        {
            "nifty_direction": nifty_score,
            "bank_nifty_direction": bn_score,
            "vix_regime": vix_score,
        },
        weights,
    )


# ---------------------------------------------------------------------------
# News — Phase-10 stub. ``filing_title`` set by the filings classifier
# is treated as positive (directional binary_high). With no filing,
# returns NEUTRAL so news doesn't drag the score down.
# ---------------------------------------------------------------------------

def score_news(
    filing_title: str | None,
    side: str,
    *,
    weights: dict[str, float],
) -> float:
    if filing_title:
        # Filings get +30 already in the legacy scorer; here we
        # reflect direction-known catalysts as max news score.
        news_score: float | None = 100.0
    else:
        news_score = NEUTRAL
    return _normalise(
        {"filing_signal": news_score},
        weights,
    )
