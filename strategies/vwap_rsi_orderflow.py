"""VWAP + RSI + Order Flow — high-conviction scalp strategy.

Read: This is a discretionary-style confluence setup. We only fire when
THREE independent reads line up:

  - **Price location**: bar wicked into VWAP from above (long) or below
    (short) AND closed back on the favourable side. This is the
    "support hold" / "resistance reject" footprint at the institutional
    benchmark.
  - **Momentum read**: RSI in the bullish-pullback zone [rsi_long_low,
    rsi_long_high] and ticking UP — we're catching the bounce as it
    happens, not chasing after RSI has already crossed 60. Mirror logic
    for shorts at the [rsi_short_low, rsi_short_high] rejection zone.
  - **Candlestick confirmation**: bullish (or bearish) engulfing — the
    classic 2-bar reversal where the current body fully eclipses the
    previous body. This is what discretionary scalpers actually wait
    for at the VWAP test.

ORDER FLOW / OI CONFLUENCE — NOT WIRED UP:

The original spec calls for "OI buildup / PCR > 1.2 for extra
conviction". The strategy interface (``Strategy.signal(df, i)``) only
sees OHLCV; F&O Open Interest and Put-Call Ratio live in a different
pipeline (``fyers_client`` / NSE option chain) and aren't merged into
the per-bar DataFrame the backtest engine builds. To wire it up, you
would:

  1. Pre-compute per-bar ``oi_buildup`` / ``pcr`` series in the loader.
  2. Add a ``require_oi_confluence`` param here, and gate entries on
     ``df['pcr'].iat[i] > 1.2`` (long-side example).

For now the strategy is OHLCV-only. The OI gate is a TODO documented
here so future merge of F&O data has an obvious place to plug in.

Exits:
  - Opposite VWAP cross: long closes when close < VWAP (support broke);
    short closes when close > VWAP.
  - 1× ATR target from entry (scalp profile: tighter than 1.5× since
    the setup is meant for quick wins on 3-5 min charts). Snapshot ATR
    at the entry bar so a later ATR expansion doesn't loosen the target.

Reuses ``indicators.volume.vwap``, ``indicators.momentum.rsi``,
``indicators.volatility.atr``."""
from __future__ import annotations

import pandas as pd

from indicators import momentum, volatility, volume

from .base import Signal, SignalKind, Strategy


class VwapRsiOrderflow(Strategy):
    name = "vwap_rsi_orderflow"

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_long_low: float = 40.0,
        rsi_long_high: float = 50.0,
        rsi_short_low: float = 50.0,
        rsi_short_high: float = 60.0,
        atr_period: int = 14,
        atr_target_mult: float = 1.0,
    ) -> None:
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if not (0 < rsi_long_low < rsi_long_high < 100):
            raise ValueError(
                f"need 0 < rsi_long_low ({rsi_long_low}) < rsi_long_high "
                f"({rsi_long_high}) < 100"
            )
        if not (0 < rsi_short_low < rsi_short_high < 100):
            raise ValueError(
                f"need 0 < rsi_short_low ({rsi_short_low}) < rsi_short_high "
                f"({rsi_short_high}) < 100"
            )
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_target_mult <= 0:
            raise ValueError(
                f"atr_target_mult must be > 0, got {atr_target_mult}"
            )
        super().__init__(
            rsi_period=rsi_period,
            rsi_long_low=rsi_long_low, rsi_long_high=rsi_long_high,
            rsi_short_low=rsi_short_low, rsi_short_high=rsi_short_high,
            atr_period=atr_period, atr_target_mult=atr_target_mult,
        )
        self.rsi_period = rsi_period
        self.rsi_long_low = rsi_long_low
        self.rsi_long_high = rsi_long_high
        self.rsi_short_low = rsi_short_low
        self.rsi_short_high = rsi_short_high
        self.atr_period = atr_period
        self.atr_target_mult = atr_target_mult

        # Per-DataFrame indicator caches.
        self._cache_key: tuple[int, int] | None = None
        self._vwap: pd.Series | None = None
        self._rsi: pd.Series | None = None
        self._atr: pd.Series | None = None

        # Open-position tracking — engine doesn't tell us, we mirror.
        self._position: str = "FLAT"   # "FLAT" / "LONG" / "SHORT"
        self._entry_price: float = 0.0
        self._entry_atr: float = 0.0

    def _ensure(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._vwap is not None:
            return
        self._vwap = volume.vwap(df)
        self._rsi = momentum.rsi(df, period=self.rsi_period)
        self._atr = volatility.atr(df, period=self.atr_period)
        self._cache_key = key
        self._position = "FLAT"
        self._entry_price = 0.0
        self._entry_atr = 0.0

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure(df)

        v_curr = self._vwap.iat[i]
        r_prev = self._rsi.iat[i - 1]
        r_curr = self._rsi.iat[i]
        atr_curr = self._atr.iat[i]
        o_prev = float(df["open"].iat[i - 1])
        c_prev = float(df["close"].iat[i - 1])
        o_curr = float(df["open"].iat[i])
        h_curr = float(df["high"].iat[i])
        l_curr = float(df["low"].iat[i])
        c_curr = float(df["close"].iat[i])
        if (pd.isna(v_curr) or pd.isna(r_prev) or pd.isna(r_curr)
                or pd.isna(atr_curr)):
            return None

        # ---- Exits first ----
        if self._position == "LONG":
            target = self._entry_price + self.atr_target_mult * self._entry_atr
            target_hit = c_curr >= target
            vwap_lost = c_curr < v_curr
            if target_hit or vwap_lost:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._entry_atr = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "VwapRsiL_Target" if target_hit else "VwapRsiL_Lost",
                )
        elif self._position == "SHORT":
            target = self._entry_price - self.atr_target_mult * self._entry_atr
            target_hit = c_curr <= target
            vwap_lost = c_curr > v_curr
            if target_hit or vwap_lost:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._entry_atr = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "VwapRsiS_Target" if target_hit else "VwapRsiS_Lost",
                )

        # ---- Entries ----
        if self._position != "FLAT":
            return None

        # Bullish engulfing: prev red, curr green, curr body engulfs prev body.
        bullish_engulf = (
            c_prev < o_prev
            and c_curr > o_curr
            and o_curr <= c_prev
            and c_curr >= o_prev
        )
        # Bearish engulfing: prev green, curr red, curr body engulfs prev body.
        bearish_engulf = (
            c_prev > o_prev
            and c_curr < o_curr
            and o_curr >= c_prev
            and c_curr <= o_prev
        )

        # VWAP support hold: wicked into VWAP from above, closed back above.
        vwap_support = l_curr <= v_curr <= c_curr
        # VWAP resistance reject: wicked into VWAP from below, closed below.
        vwap_resistance = h_curr >= v_curr >= c_curr

        # RSI bounce-up in the long zone (e.g. 40-50 rising).
        rsi_bounce_up = (
            self.rsi_long_low <= r_prev <= self.rsi_long_high
            and r_curr > r_prev
        )
        # RSI rejection-down in the short zone (e.g. 50-60 falling).
        rsi_reject_dn = (
            self.rsi_short_low <= r_prev <= self.rsi_short_high
            and r_curr < r_prev
        )

        if vwap_support and rsi_bounce_up and bullish_engulf:
            self._position = "LONG"
            self._entry_price = c_curr
            self._entry_atr = float(atr_curr)
            return Signal(SignalKind.ENTER_LONG, "VwapRsiOFLE")
        if vwap_resistance and rsi_reject_dn and bearish_engulf:
            self._position = "SHORT"
            self._entry_price = c_curr
            self._entry_atr = float(atr_curr)
            return Signal(SignalKind.ENTER_SHORT, "VwapRsiOFSE")
        return None
