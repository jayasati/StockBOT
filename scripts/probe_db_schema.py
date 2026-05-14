"""Print the schemas + a sample row for tables the dashboard will query."""
import sqlite3
from pathlib import Path

DB = Path("alerts.db")
TABLES = [
    "paper_trades", "nse_snapshots", "news_items", "news_scores",
    "daily_levels", "filter_audit", "alerts_sent", "filings_seen",
    "risk_flags", "signal_indicators", "bars_5m",
]

con = sqlite3.connect(DB)
cur = con.cursor()
for t in TABLES:
    print(f"\n=== {t} ===")
    try:
        for row in cur.execute(f"PRAGMA table_info({t})"):
            print(" ", row)
        cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  rows: {cnt}")
        if cnt > 0:
            cur.execute(f"SELECT * FROM {t} LIMIT 1")
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            print("  sample:")
            for c, v in zip(cols, row):
                v_str = str(v)[:80]
                print(f"    {c}: {v_str}")
    except sqlite3.OperationalError as e:
        print(f"  ERR: {e}")
