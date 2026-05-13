"""Back-compat shim — source-of-truth moved to :mod:`core.config`.

Importing from ``bot.config`` continues to work (existing callers,
test fixtures that ``monkeypatch.setattr(bot.config, "DB_PATH", ...)``
are unaffected). New code should import from :mod:`core.config`
directly so it doesn't pull in the bot package."""
from core.config import DB_PATH, IST, Settings, settings  # noqa: F401

__all__ = ["DB_PATH", "IST", "Settings", "settings"]
