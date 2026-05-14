"""ADX + Supertrend + ATR — trend-strength-filtered breakout strategy.

Read: ADX is the trend-strength meter that prevents this whole setup
from firing in chop. A Supertrend flip in a sideways tape is the
classic whipsaw trap; gating it on ADX > 25 (with ADX itself rising
on the entry bar) keeps you out of those.

Long entry — all of:
  - Supertrend direction flips from -1 → +1 on this bar (green flip)
  - ADX[i] > ``min_adx`` (default 25)
  - ADX[i] > ADX[i-1]                                (ADX rising)
  - +DI > -DI                                         (bullish DMI bias)

Short entry — mirror image.

Risk:
  - Stop loss = 1× ATR away from entry close (long: entry - ATR; short:
    entry + ATR). The stop is FIXED at entry — we snapshot ATR on the
    entry bar so a later ATR expansion doesn't loosen the stop.
  - Exit also fires on a Supertrend flip back against position (this is
    the classic trail-the-trend exit; the trend has structurally broken
    so we don't wait for the stop).

Reuses ``indicators.trend.supertrend``, ``indicators.trend.adx``,
``indicators.volatility.atr``."""
from __future__ import annotations

import pandas as pd

from indicators import trend, volatility

from .base import Signal, SignalKind, Strategy


class AdxSupertrendAtr(Strategy):
    name = "adx_supertrend_atr"

    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_mult: float = 3.0,
        adx_period: int = 14,
        min_adx: float = 25.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.0,
    ) -> None:
        if supertrend_period < 1:
            raise ValueError(
                f"supertrend_period must be >= 1, got {supertrend_period}"
            )
        if supertrend_mult <= 0:
            raise ValueError(
                f"supertrend_mult must be > 0, got {supertrend_mult}"
            )
        if adx_period < 2:
            raise ValueError(f"adx_period must be >= 2, got {adx_period}")
        if not (0 < min_adx < 100):
            raise ValueError(f"min_adx must be in (0, 100), got {min_adx}")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_stop_mult <= 0:
            raise ValueError(
                f"atr_stop_mult must be > 0, got {atr_stop_mult}"
            )
        super().__init__(
            supertrend_period=supertrend_period,
            supertrend_mult=supertrend_mult,
            adx_period=adx_period, min_adx=min_adx,
            atr_period=atr_period, atr_stop_mult=atr_stop_mult,
        )
        self.supertrend_period = supertrend_period
        self.supertrend_mult = supertrend_mult
        self.adx_period = adx_period
        self.min_adx = min_adx
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

        # Per-DataFrame caches (rolling math is expensive in the bar loop).
        self._cache_key: tuple[int, int] | None = None
        self._dir: pd.Series | None = None
        self._adx: pd.Series | None = None
        self._dip: pd.Series | None = None
        self._dim: pd.Series | None = None
        self._atr: pd.Series | None = None

        # Open-position tracking — engine doesn't notify us, so we mirror.
        self._position: str = "FLAT"   # "FLAT" / "LONG" / "SHORT"
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0

    def _ensure(self, df: pd.DataFrame) -> None:
        key = (id(df), len(df))
        if key == self._cache_key and self._dir is not None:
            return
        st = trend.supertrend(
            df, period=self.supertrend_period, multiplier=self.supertrend_mult,
        )
        adx_df = trend.adx(df, period=self.adx_period)
        self._dir = st["direction"]
        self._adx = adx_df["adx"]
        self._dip = adx_df["di_plus"]
        self._dim = adx_df["di_minus"]
        self._atr = volatility.atr(df, period=self.atr_period)
        self._cache_key = key
        # New frame ⇒ reset position state.
        self._position = "FLAT"
        self._entry_price = 0.0
        self._stop_price = 0.0

    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        if i < 1:
            return None
        self._ensure(df)
        d_prev = self._dir.iat[i - 1]
        d_curr = self._dir.iat[i]
        a_prev = self._adx.iat[i - 1]
        a_curr = self._adx.iat[i]
        dip = self._dip.iat[i]
        dim = self._dim.iat[i]
        atr_curr = self._atr.iat[i]
        c_curr = float(df["close"].iat[i])
        if (pd.isna(d_prev) or pd.isna(d_curr) or pd.isna(a_prev)
                or pd.isna(a_curr) or pd.isna(dip) or pd.isna(dim)
                or pd.isna(atr_curr)):
            return None

        # ---- Exits first ----
        if self._position == "LONG":
            stop_hit = c_curr <= self._stop_price
            st_flipped = d_curr == -1
            if stop_hit or st_flipped:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._stop_price = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "AdxStopL" if stop_hit else "AdxSupertrendL_Flip",
                )
        elif self._position == "SHORT":
            stop_hit = c_curr >= self._stop_price
            st_flipped = d_curr == 1
            if stop_hit or st_flipped:
                self._position = "FLAT"
                self._entry_price = 0.0
                self._stop_price = 0.0
                return Signal(
                    SignalKind.EXIT,
                    "AdxStopS" if stop_hit else "AdxSupertrendS_Flip",
                )

        # ---- Entries ----
        if self._position != "FLAT":
            return None

        adx_strong = a_curr > self.min_adx
        adx_rising = a_curr > a_prev
        if not (adx_strong and adx_rising):
            return None

        # Supertrend green flip (-1 → +1) + bullish DMI.
        if d_prev == -1 and d_curr == 1 and dip > dim:
            self._position = "LONG"
            self._entry_price = c_curr
            self._stop_price = c_curr - self.atr_stop_mult * float(atr_curr)
            return Signal(SignalKind.ENTER_LONG, "AdxSupertrendLE")
        # Red flip (+1 → -1) + bearish DMI.
        if d_prev == 1 and d_curr == -1 and dim > dip:
            self._position = "SHORT"
            self._entry_price = c_curr
            self._stop_price = c_curr + self.atr_stop_mult * float(atr_curr)
            return Signal(SignalKind.ENTER_SHORT, "AdxSupertrendSE")
        return None
