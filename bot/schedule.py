"""Back-compat shim — source-of-truth moved to :mod:`core.session`.

Importing from ``bot.schedule`` continues to work (existing callers,
test fixtures that ``monkeypatch.setattr("bot.schedule.is_market_open",
...)`` are unaffected). New code should import from :mod:`core.session`
directly so it doesn't pull in the bot package."""
from core.session import (  # noqa: F401
    SESSION_CLOSE,
    SESSION_LAST_BAR_OPEN,
    SESSION_OPEN,
    in_session,
    is_market_open,
    seconds_until_market_open,
)

__all__ = [
    "SESSION_CLOSE",
    "SESSION_LAST_BAR_OPEN",
    "SESSION_OPEN",
    "in_session",
    "is_market_open",
    "seconds_until_market_open",
]
