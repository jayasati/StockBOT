"""VWAP + EMA + Volume strategy — trend + mean-reversion hybrid.

Read: VWAP is the institutional benchmark for the day. A close that
reclaims VWAP from below (was at-or-below the prior bar, now above)
while the fast EMA is above the slow EMA AND volume is elevated is
treated as a high-probability institutional footprint — an entry that
combines mean-reversion at VWAP with confirmation from a trend filter
and a volume-spike filter. The short side is the mirror image: a
rejection at VWAP from above + fast EMA below slow + volume spike.

Exits:
  - Opposite VWAP cross: long closes when price crosses BACK below VWAP;
    short closes when price crosses BACK above VWAP.
  - ATR target: when the current bar's close is at-or-beyond
    ``entry_price ± atr_mult * ATR(entry_bar)`` in the favourable
    direction. We snapshot the ATR at the bar of entry (not live ATR)
    so the target is fixed at trade time — matches how a discretionary
    trader sets a 1.5×ATR target on entry.

Reuses ``indicators.volume.vwap``, ``indicators.trend.ema``,
``indicators.volume.volume_surge_ratio`` (RVOL vs SMA), and
``indicators.volatility.atr`` so signal math is shared with the rest of
the stack."""
from __future__ import annotations

import pandas as pd

from indicators import trend, volatility, volume

from .base import Signal, SignalKind, Strategy


class VwapEmaVolume(Strategy):
    name = "vwap_ema_volume"

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        volume_period: int = 20,
        volume_spike: float = 1.5,
        atr_period: int = 14,
        atr_mult: float = 1.5,
    ) -> None:
        if fast < 2:
            raise ValueError(f"fast must be >= 2, got {fast}")
        if slow <= fast:
            raise ValueError(
                f"slow ({slow}) must be > fast ({fast})"
            )
        if volume_period < 1:
            raise ValueError(
                f"volume_period must be >= 1, got {volume_period}"
            )
        if volume_spike <= 0:
            raise ValueError(
                f"volume_spike must be > 0, got {volume_spike}"
            )
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be > 0, got {atr_mult}")
        super().__init__(
            fast=fast, slow=slow,
            volume_period=volume_period, volume_spike=volume_spike,
            atr_period=atr_period, atr_mult=atr_mult,
        )
        self.fast = fast
        self.slow = slow
        self.volume_period = volume_period
        self.volume_spike = volume_spike
        self.atr_period = atr_period
        self.atr_mult = atr_mult

        
        # Per-DataFrame caches — rolling math is expensive in the bar loop.
        self._cache_key: tuple[int, int] | None = None
        self._vwap: pd.Series | None = None
        self._fast_ema: pd.Series | None = None
        self._slow_ema: pd.Series | None = None
        self._rvol: pd.Series | None = None
        self._atr: pd.Series | None = None
        # Open-position state — tracked so EXIT can fire on opposite VWAP
        # cross or ATR target without the engine telling us we're long/short.
        self._position: str = "FLAT"  # "FLAT" / "LONG" / "SHORT"
        self._entry_price: float = 0.0
        self._entry_atr: float = 0.0

    def _ensure(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._vwap is not None:
            return
        self._vwap = volume.vwap(df)
        self._fast_ema = trend.ema(df, period=self.fast)
        self._slow_ema = trend.ema(df, period=self.slow)
        self._rvol = volume.volume_surge_ratio(df, period=self.volume_period)
        self._atr = volatility.atr(df, period=self.atr_period)
        self._cache_key = key
        # New frame ⇒ reset position tracking.
        self._position = "FLAT"
        self._entry_price = 0.0
        self._entry_atr = 0.0

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure(df)
        v_prev = self._vwap.iat[i - 1]
        v_curr = self._vwap.iat[i]
        f_curr = self._fast_ema.iat[i]
        s_curr = self._slow_ema.iat[i]
        r_curr = self._rvol.iat[i]
        a_curr = self._atr.iat[i]
        c_prev = float(df["close"].iat[i - 1])
        c_curr = float(df["close"].iat[i])
        if (pd.isna(v_prev) or pd.isna(v_curr) or pd.isna(f_curr)
                or pd.isna(s_curr) or pd.isna(r_curr) or pd.isna(a_curr)):
            return None

        # ---- Exit logic (runs before entry so we don't immediately re-enter
        # the same direction we just exited from on this bar) ----
        if self._position == "LONG":
            atr_target = self._entry_price + self.atr_mult * self._entry_atr
            crossed_back = c_prev >= v_prev and c_curr < v_curr
            target_hit = c_curr >= atr_target
            if crossed_back or target_hit:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._entry_atr = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "VwapExitL_Target" if target_hit else "VwapExitL_Cross",
                )
        elif self._position == "SHORT":
            atr_target = self._entry_price - self.atr_mult * self._entry_atr
            crossed_back = c_prev <= v_prev and c_curr > v_curr
            target_hit = c_curr <= atr_target
            if crossed_back or target_hit:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._entry_atr = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "VwapExitS_Target" if target_hit else "VwapExitS_Cross",
                )

        # ---- Entry logic ----
        if self._position != "FLAT":
            return None

        vol_spike = r_curr >= self.volume_spike
        # Long: reclaim VWAP from below (prev <= VWAP, now > VWAP) + 9>21 + vol
        if (c_prev <= v_prev and c_curr > v_curr
                and f_curr > s_curr and vol_spike):
            self._position = "LONG"
            self._entry_price = c_curr
            self._entry_atr = float(a_curr)
            return Signal(SignalKind.ENTER_LONG, "VwapEmaVolLE")
        # Short: rejection at VWAP from above (prev >= VWAP, now < VWAP) + 9<21 + vol
        if (c_prev >= v_prev and c_curr < v_curr
                and f_curr < s_curr and vol_spike):
            self._position = "SHORT"
            self._entry_price = c_curr
            self._entry_atr = float(a_curr)
            return Signal(SignalKind.ENTER_SHORT, "VwapEmaVolSE")
        return None
