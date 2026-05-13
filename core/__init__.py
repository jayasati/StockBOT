"""Foundation layer: settings, time zones, session hours.

Everything in ``core`` must be importable without pulling in ``bot``,
``data``, or any other top-level package. This is the bottom of the
dependency graph so leaves like ``data/`` and ``trading/`` can read
shared constants without creating a cycle through ``bot``.
"""
