"""One-off: print today's filter_audit + alerts for the user-named symbols."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB = Path("alerts.db")
SYMBOLS = [
    "BHARTIARTL.NS", "BSE.NS", "ALKYLAMINE.NS", "CIPLA.NS",
    "GODREJIND.NS", "MCX.NS", "ADANIPOWER.NS", "HFCL.NS",
]

con = sqlite3.connect(DB)
cur = con.cursor()

print("=== TABLES ===")
for (name,) in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
):
    print(f"  {name}")

print("\n=== SCHEMA: filter_audit ===")
try:
    for row in cur.execute("PRAGMA table_info(filter_audit)"):
        print(" ", row)
except sqlite3.OperationalError as exc:
    print(f"  (no filter_audit table: {exc})")

print("\n=== SCHEMA: alerts ===")
for row in cur.execute("PRAGMA table_info(alerts)"):
    print(" ", row)

print("\n=== SCHEMA: alerts_sent ===")
for row in cur.execute("PRAGMA table_info(alerts_sent)"):
    print(" ", row)

print("\n=== TODAY filter_audit for user symbols ===")
cur.execute(
    """
    SELECT symbol, ts, score, final_score, final_confidence, kill_reasons,
           soft_adjustments_json, alerted
    FROM filter_audit
    WHERE date(ts) = date('now', 'localtime')
      AND symbol IN ({})
    ORDER BY ts
    """.format(",".join("?" * len(SYMBOLS))),
    SYMBOLS,
)
rows = cur.fetchall()
if not rows:
    print("  (no audit rows today)")
for r in rows:
    print(" ", r)

print("\n=== Latest 5 audit rows per user symbol (any date) ===")
for sym in SYMBOLS:
    print(f"\n  --- {sym} ---")
    cur.execute(
        """SELECT ts, score, final_score, final_confidence, kill_reasons,
                  alerted
           FROM filter_audit WHERE symbol = ?
           ORDER BY ts DESC LIMIT 5""",
        (sym,),
    )
    rows = cur.fetchall()
    if not rows:
        print("    (no audit rows ever)")
    for r in rows:
        print("   ", r)

print("\n=== alerts_sent today ===")
cur.execute(
    "SELECT * FROM alerts_sent "
    "WHERE date(sent_at) = date('now', 'localtime') ORDER BY sent_at"
)
rows = cur.fetchall()
print(f"  count today: {len(rows)}")
for r in rows[:30]:
    print(" ", r)

print("\n=== alerts_sent: most recent 10 (any date) ===")
cur.execute("SELECT * FROM alerts_sent ORDER BY sent_at DESC LIMIT 10")
for r in cur.fetchall():
    print(" ", r)

print("\n=== filter_audit kill-reason breakdown TODAY ===")
cur.execute(
    """
    SELECT kill_reasons, COUNT(*) AS n
    FROM filter_audit
    WHERE date(ts) = date('now', 'localtime')
      AND kill_reasons IS NOT NULL AND kill_reasons != ''
    GROUP BY kill_reasons
    ORDER BY n DESC
    LIMIT 30
    """
)
for r in cur.fetchall():
    print(" ", r)

print("\n=== filter_audit overall today ===")
cur.execute(
    """SELECT COUNT(*), SUM(alerted)
       FROM filter_audit WHERE date(ts) = date('now', 'localtime')"""
)
print(" ", cur.fetchone())

print("\n=== Top 20 final_score TODAY (passed all hard filters) ===")
cur.execute(
    """SELECT symbol, final_score, final_confidence, soft_adjustments_json
       FROM filter_audit
       WHERE date(ts) = date('now', 'localtime')
         AND (kill_reasons IS NULL OR kill_reasons = '')
       ORDER BY final_score DESC LIMIT 20"""
)
for r in cur.fetchall():
    print(" ", r)

print("\n=== components_json (most recent today) for user symbols ===")
import json
for sym in SYMBOLS:
    cur.execute(
        """SELECT ts, score, final_score, components_json,
                  soft_adjustments_json, kill_reasons
           FROM filter_audit WHERE symbol = ?
             AND date(ts) = date('now', 'localtime')
           ORDER BY ts DESC LIMIT 1""",
        (sym,),
    )
    row = cur.fetchone()
    print(f"\n  --- {sym} ---")
    if not row:
        print("    (no rows today)")
        continue
    ts, score, final_score, comp_json, soft_json, kill = row
    print(f"    ts={ts}")
    print(f"    legacy_score={score}  final_score={final_score}")
    print(f"    soft_adj={soft_json}  kill={kill}")
    if comp_json:
        try:
            comp = json.loads(comp_json)
            for k, v in comp.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        print(f"        {k2}: {v2}")
                else:
                    print(f"    {k}: {v}")
        except Exception as exc:
            print(f"    (json parse failed: {exc})")
            print(f"    raw: {comp_json[:400]}")
