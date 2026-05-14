"""Strategy base class + signal types.

A Strategy is a pure function of price history: given a DataFrame of OHLCV
bars and an index ``i``, return what the strategy wants to do AT THE CLOSE
of bar ``i``. The backtest engine in ``strategies.backtest`` is what turns
those signals into positions and P&L."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class SignalKind(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Signal:
    kind: SignalKind
    reason: str = ""


class Strategy(ABC):
    """Subclass and override :meth:`signal`. Pass any tunables via ``**params``
    on construction; they'll land in ``self.params`` for the subclass to read."""

    name: str = ""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params

    @abstractmethod
    def signal(self, df: pd.DataFrame, i: int) -> Signal | None:
        """Return the action the strategy wants taken AT bar ``i``'s close.

        ``None`` means do nothing. The engine fills the resulting trade on
        bar ``i+1``'s open (Pine ``calc_on_every_tick=false`` semantics)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"
