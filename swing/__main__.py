"""``python -m swing`` — print subcommand usage and exit.

The actual CLIs live in ``swing.alert`` and ``swing.replay``; run them
directly via ``python -m swing.alert`` / ``python -m swing.replay``."""
print("usage: python -m swing.{alert|replay} [args]")
print()
print("  alert   — daily EOD dispatcher (sends to Telegram)")
print("  replay  — backtest the strategy on historical data")
raise SystemExit(1)
