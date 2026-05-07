"""Signal + regime constants and shared paths for the swing strategy."""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from backtest import CACHE_DIR, DAILY_CACHE

IST = ZoneInfo("Asia/Kolkata")

NIFTY_TICKER = "^NSEI"
NIFTY_CACHE = CACHE_DIR / "nifty_daily.parquet"
RESULTS_CSV = Path("swing_results.csv")

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
VOLUME_MULTIPLE = 2.0
RANGE_POSITION_THRESHOLD = 0.90
EMA_PERIOD = 20

# Skip alerts where price is >X% above 20-day EMA (parabolic / exhaustion).
# Backtest showed every disaster trade had emaΔ > 10%. Default 8% is the
# conservative cut; pass --max-extension to override.
MAX_EXTENSION_PCT = 8.0

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
NIFTY_UP_PCT = 0.30
BREADTH_THRESHOLD_PCT = 55.0

# ---------------------------------------------------------------------------
# Holding horizons (days). Used by the backtest for forward-return computation.
# ---------------------------------------------------------------------------
HORIZONS = [1, 3, 5]

# ---------------------------------------------------------------------------
# Alert-path window (live single-day evaluation).
# 60 calendar days ≈ 40 trading days; enough for 20-EMA + a few prior EMA values.
# ---------------------------------------------------------------------------
HISTORY_DAYS = 60
