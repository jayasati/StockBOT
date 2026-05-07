"""Promoter pledge data (manual seed for now).

TODO: scrape BSE's shareholding pattern XML per company and populate
automatically. For now, the user adds known high-pledge stocks here;
``refresh_pledge_data()`` copies them into ``risk_flags`` so
``is_suppressed()`` treats them like ASM/GSM hits.

Sources to look up promoter pledge %:
  - trendlyne.com / screener.in (per-stock fundamentals page)
  - BSE shareholding pattern PDFs (quarterly)
Threshold > 50% is the rule of thumb for "structurally compromised"."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("alerts.db")

HIGH_PLEDGE_STOCKS: dict[str, float] = {
    # "EXAMPLE.NS": 55.0,
}


def refresh_pledge_data() -> int:
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM risk_flags WHERE flag_type = 'pledge_pct'")
        for sym, pct in HIGH_PLEDGE_STOCKS.items():
            conn.execute(
                "INSERT OR REPLACE INTO risk_flags VALUES (?, ?, ?, ?)",
                (sym, "pledge_pct", str(pct), now),
            )
    return len(HIGH_PLEDGE_STOCKS)
