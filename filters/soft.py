"""Soft filters: return ``(name, multiplier)`` to apply, ``None`` to
skip. Multipliers compose multiplicatively onto the chain's running
product. Names should be short slugs so the ``filter_audit`` JSON
stays greppable.

Phase-6 inventory built up across tasks:
  adx_kill_counter_trend, adx_weak_trend, already_extended,
  low_volume, wide_spread (stubbed),
  bank_nifty_opposite, mtf_trend_alignment, vix_filter
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.scoring import StockSignals
    from .chain import FilterContext


# ---------------------------------------------------------------------------
# Constants — multipliers + thresholds. Module-level so tests can patch.
# ---------------------------------------------------------------------------

BANK_NIFTY_OPPOSITE_MULT = 0.9
"""Demote 10% when BANKNIFTY direction opposes the trade side.
Banks dominate index gravity in India — if banks are falling,
even non-bank longs face headwinds."""

VIX_LOW_THRESHOLD = 12.0
VIX_HIGH_THRESHOLD = 20.0
VIX_PANIC_THRESHOLD = 25.0
VIX_LOW_MULT = 1.05
"""VIX < 12 means complacent markets — slightly favour trend continuations."""
VIX_HIGH_MULT = 0.85
"""VIX 20-25: elevated vol regime, demote setups. Indian VIX historical
median ~14-18 so the previous 16 cutoff fired on every signal in a
normal-to-mildly-elevated tape."""
VIX_PANIC_MULT = 0.6
"""VIX > 25: panic / news-driven; microstructure signals lose. Bumped
from 20 alongside the VIX_HIGH cutoff move."""


# ---------------------------------------------------------------------------
# Bank Nifty direction (uses ctx.bank_nifty_pct populated by scanner)
# ---------------------------------------------------------------------------

def bank_nifty_opposite(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """Demote if BANKNIFTY direction opposes the trade side.
    ``ctx.bank_nifty_pct`` is filled by the scanner from
    :func:`data.index_feed.get_intraday_pct`. None → skip (fail-open)."""
    if ctx.bank_nifty_pct is None:
        return None
    if signals.side == "LONG" and ctx.bank_nifty_pct <= -0.3:
        return ("bank_nifty_opposite", BANK_NIFTY_OPPOSITE_MULT)
    if signals.side == "SHORT" and ctx.bank_nifty_pct >= 0.3:
        return ("bank_nifty_opposite", BANK_NIFTY_OPPOSITE_MULT)
    return None


# ---------------------------------------------------------------------------
# India VIX (uses ctx.vix populated by scanner via index_feed.get_vix)
# ---------------------------------------------------------------------------

def vix_filter(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """Volatility-regime multiplier. Tiered:

      VIX < 12          → 1.05 (low-vol trend continuation)
      VIX 12-20         → no change (most common state)
      VIX 20-25         → 0.85 (elevated vol regime, demote setups)
      VIX >= 25         → 0.6  (panic / news-driven, microstructure unreliable)

    Thresholds were bumped from the original 16/20 to 20/25 because
    Indian VIX historical median sits ~14-18 — the original 16 cutoff
    fired on every signal in normal-to-mildly-elevated tapes.
    """
    v = ctx.vix
    if v is None:
        return None
    if v < VIX_LOW_THRESHOLD:
        return ("vix_low", VIX_LOW_MULT)
    if v >= VIX_PANIC_THRESHOLD:
        return ("vix_panic", VIX_PANIC_MULT)
    if v >= VIX_HIGH_THRESHOLD:
        return ("vix_high", VIX_HIGH_MULT)
    return None


# ---------------------------------------------------------------------------
# ADX-based: the Meesho rule + weak-trend demotion
# ---------------------------------------------------------------------------

ADX_COUNTER_TREND_THRESHOLD = 50.0
"""ADX above this with DI lines pointing OPPOSITE the trade side is
the brainstorm's 'Meesho rule' — strong downtrend, you're long → 0.4×."""
ADX_COUNTER_TREND_MULT = 0.4

ADX_WEAK_TREND_THRESHOLD = 20.0
"""ADX below this means no clear trend — momentum signals lose
edge in chop. Demote moderately."""
ADX_WEAK_MULT = 0.7

ALREADY_EXTENDED_PCT = 5.0
"""LONG signal on a stock up >5% intraday (or SHORT on -5%) →
likely chasing the parabola top/bottom. Demote 0.8×."""
ALREADY_EXTENDED_MULT = 0.8

LOW_VOLUME_THRESHOLD = 1.0
"""volume_surge_ratio < 1.0 means below-average volume — the
single backtest-validated positive component is weak, so the
whole signal weakens."""
LOW_VOLUME_MULT = 0.85


def adx_kill_counter_trend(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """The brainstorm's 'Meesho rule'. When the 5m ADX exceeds 50
    AND the directional indicators say the dominant trend opposes
    the trade side, demote heavily (0.4×). The intuition: a strong
    counter-trend setup is one of the most common ways to lose money
    in NSE intraday — RSI extremes get bought/sold INTO the trend,
    not against it.

    Reads ``adx_5m``, ``adx_5m_di_plus``, ``adx_5m_di_minus`` from
    the indicator snapshot. Fails-open on missing data."""
    snap = signals.snapshot
    if snap is None:
        return None
    # Registry produces adx_5m_adx (the ADX value column of the ADX
    # multi-output frame), not adx_5m. The plain key never existed,
    # so this filter was silently no-op'd.
    adx = snap.values.get("adx_5m_adx")
    di_plus = snap.values.get("adx_5m_di_plus")
    di_minus = snap.values.get("adx_5m_di_minus")
    if adx is None or di_plus is None or di_minus is None:
        return None
    if adx < ADX_COUNTER_TREND_THRESHOLD:
        return None
    # LONG against a dominant downtrend, or SHORT against an uptrend.
    if signals.side == "LONG" and di_minus > di_plus:
        return ("adx_counter_trend", ADX_COUNTER_TREND_MULT)
    if signals.side == "SHORT" and di_plus > di_minus:
        return ("adx_counter_trend", ADX_COUNTER_TREND_MULT)
    return None


def adx_weak_trend(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """ADX < 20 → no trend in the underlying. Demote the signal —
    breakouts in chop usually fail back."""
    snap = signals.snapshot
    if snap is None:
        return None
    adx = snap.values.get("adx_5m_adx")
    if adx is None or adx >= ADX_WEAK_TREND_THRESHOLD:
        return None
    return ("adx_weak", ADX_WEAK_MULT)


# ---------------------------------------------------------------------------
# Already extended (chasing the parabola)
# ---------------------------------------------------------------------------

def already_extended(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """LONG signal when the stock is already up >5% today, or SHORT
    when it's already down >5% — chasing the move. Demote 0.8×.

    Uses ``ctx.intraday_df`` first bar's Open as the day's open."""
    if ctx.intraday_df is None or ctx.intraday_df.empty:
        return None
    open_col = "Open" if "Open" in ctx.intraday_df.columns else "open"
    try:
        today_open = float(ctx.intraday_df[open_col].iloc[0])
    except (KeyError, IndexError, ValueError):
        return None
    if today_open == 0:
        return None
    pct = (signals.price - today_open) / today_open * 100.0
    if signals.side == "LONG" and pct > ALREADY_EXTENDED_PCT:
        return ("already_extended", ALREADY_EXTENDED_MULT)
    if signals.side == "SHORT" and pct < -ALREADY_EXTENDED_PCT:
        return ("already_extended", ALREADY_EXTENDED_MULT)
    return None


# ---------------------------------------------------------------------------
# Low volume
# ---------------------------------------------------------------------------

def low_volume(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """volume_ratio < 1.0 means today's volume is below the 10-day
    expected. The composite score's only positive-lift component
    (volume) is missing — demote 0.85×."""
    if signals.volume_ratio is None:
        return None
    if signals.volume_ratio < LOW_VOLUME_THRESHOLD:
        return ("low_volume", LOW_VOLUME_MULT)
    return None


WIDE_SPREAD_PCT = 0.003
"""bid-ask spread > 0.3% of price → wide. Liquid NSE stocks usually
sit at 1-3 bps; > 30 bps signals microcap chop / pre-close drift."""
WIDE_SPREAD_MULT = 0.85


def wide_spread(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """Demote when the bid-ask spread is wider than 0.3% of price.

    NOT YET WIRED — the Fyers WebSocket subscription model in
    ``data/realtime_feed.py`` consumes the ``symbolUpdate`` channel,
    which carries LTP and OHLCV but NOT the depth book (top-of-book
    bid/ask + sizes). Fyers DOES expose depth on a separate
    subscription type (``symbolData`` v2 channel) but enabling it
    doubles the WebSocket message volume and exceeds the free-tier
    symbol cap for a 500-stock watchlist.

    Two activation paths for Phase 6b:
      (a) Subscribe depth only for symbols with OPEN paper trades —
          small N, manageable. Useful for monitoring exit-quality.
      (b) Switch the live feed to depth and accept the bandwidth.

    Reads ``signals.market_context['spread_pct']`` when set so a
    partial wiring (depth on only some symbols) still works. With
    no spread data the filter is a pass-through."""
    if not signals.market_context:
        return None
    spread_pct = signals.market_context.get("spread_pct")
    if spread_pct is None:
        return None
    if spread_pct > WIDE_SPREAD_PCT:
        return ("wide_spread", WIDE_SPREAD_MULT)
    return None


MTF_BOTH_AGREE_MULT = 1.10
"""15m + 60m trends both align with 5m signal → reward by 10%."""
MTF_BOTH_DISAGREE_MULT = 0.5
"""Both higher TFs disagree with the 5m signal → heavy demote."""
MTF_ONE_DISAGREE_MULT = 0.85
"""Mixed: one higher TF agrees, one disagrees → mild demote."""


def mtf_trend_alignment(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """Multi-timeframe agreement check. Compares the 5m signal side
    against the dominant direction (DI+ vs DI-) at 15m and 60m.

    - Both 15m and 60m agree with side → 1.10× (lifts confidence)
    - One agrees, one disagrees           → 0.85×
    - Both disagree                       → 0.5× (heavy demote)
    - Partial data (only one TF or none)  → None (fail-open)

    The scanner is responsible for expanding ``target_timeframes`` in
    ``_build_snapshot`` so the snapshot actually contains 15m/60m
    ADX keys. With only today's intraday history, 60m ADX may not
    have enough warmup until mid-session — that's an explicit
    design trade-off documented in :class:`indicators.IndicatorSnapshot`."""
    snap = signals.snapshot
    if snap is None:
        return None

    def trend_dir(tf: str) -> "str | None":
        di_plus = snap.values.get(f"adx_{tf}_di_plus")
        di_minus = snap.values.get(f"adx_{tf}_di_minus")
        if di_plus is None or di_minus is None:
            return None
        return "UP" if di_plus > di_minus else "DOWN"

    d15 = trend_dir("15m")
    d60 = trend_dir("60m")
    if d15 is None and d60 is None:
        return None

    expected = "UP" if signals.side == "LONG" else "DOWN"
    agreed = sum(1 for d in (d15, d60) if d == expected)
    disagreed = sum(1 for d in (d15, d60) if d is not None and d != expected)

    if agreed == 2:
        return ("mtf_both_agree", MTF_BOTH_AGREE_MULT)
    if disagreed == 2:
        return ("mtf_both_disagree", MTF_BOTH_DISAGREE_MULT)
    if disagreed == 1 and agreed == 1:
        return ("mtf_one_disagree", MTF_ONE_DISAGREE_MULT)
    # Only one TF available, and it agrees → boost slightly? Or noop?
    # Current decision: noop (None) to keep partial-data behaviour
    # predictable. This matters when 60m hasn't reached warmup yet.
    return None


ASM_STAGE_ONE_MULT = 0.9
"""ST-ASM/LT-ASM Stage I imposes 100% margin but trading is allowed.
Demote 0.9× rather than hard-killing. Stage II+ still hard-kills via
:func:`bot.suppression.rules._asm_suppresses`."""


def asm_stage_one(
    signals: "StockSignals", ctx: "FilterContext",
) -> "tuple[str, float] | None":
    """Apply a soft demote when the symbol is on ASM Stage I (either
    ST-ASM or LT-ASM). Stage II+ never reaches this filter — the hard
    suppression layer kills those upstream."""
    from bot.suppression.rules import get_asm_stage
    stage = get_asm_stage(signals.symbol)
    if not stage:
        return None
    s = stage.upper()
    if "STAGE I" in s and "STAGE II" not in s and "STAGE III" not in s:
        return ("asm_stage_1", ASM_STAGE_ONE_MULT)
    return None


SOFT_FILTERS = (
    adx_kill_counter_trend,
    adx_weak_trend,
    already_extended,
    low_volume,
    wide_spread,
    bank_nifty_opposite,
    mtf_trend_alignment,
    vix_filter,
    asm_stage_one,
)
