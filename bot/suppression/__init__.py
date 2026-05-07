"""Risk-flag suppression layer.

Blocks alerts on stocks that are structurally compromised:
  - on NSE ASM stage 2+ (price-band/margin restrictions tightened)
  - on NSE GSM (any stage — graded surveillance, severe restrictions)
  - have promoter pledge % > 50 (financial distress)
  - already alerted within the cooldown window (anti-spam)

Tables (in alerts.db):
  risk_flags   (symbol, flag_type, value, updated_at)
  alerts_sent  (existing — owned by bot/storage; we read for cooldown)

Public API:
  refresh_asm_gsm()      — scrape NSE ASM/GSM lists; persist
  refresh_pledge_data()  — populate from HIGH_PLEDGE_STOCKS dict (TODO: BSE XML)
  is_suppressed(sym, cooldown_minutes) -> (bool, reason)

Schema + init_db live in :mod:`bot.db`.

Diagnostic: ``python -m bot.suppression refresh`` refreshes ASM/GSM and
dumps the resulting tables to stdout."""
from .http import (
    NSE_ASM_API_CANDIDATES,
    NSE_ASM_PAGE,
    NSE_GSM_API_CANDIDATES,
    NSE_GSM_PAGE,
    NSE_HOME,
    USER_AGENT,
)
from .pledge import HIGH_PLEDGE_STOCKS, refresh_pledge_data
from .refresh import refresh_asm_gsm
from .rules import is_suppressed

__all__ = [
    "HIGH_PLEDGE_STOCKS",
    "NSE_ASM_API_CANDIDATES",
    "NSE_ASM_PAGE",
    "NSE_GSM_API_CANDIDATES",
    "NSE_GSM_PAGE",
    "NSE_HOME",
    "USER_AGENT",
    "is_suppressed",
    "refresh_asm_gsm",
    "refresh_pledge_data",
]
