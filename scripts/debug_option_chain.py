r"""Inspect the latest option_chain payload NSE returned.

Pulls from nse_snapshots (where the verify_nse run persisted it) so
we don't hit NSE again. Prints the top-level structure + first row
so we can see whether NSE shifted the schema or returned an empty
envelope.

Run from the project root:
    venv\Scripts\python.exe scripts\debug_option_chain.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from bot.config import DB_PATH

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT ts, kind, payload_json FROM nse_snapshots "
            "WHERE kind LIKE 'option_chain%' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()

    if row is None:
        print("No option_chain snapshot in nse_snapshots. "
              "Run verify_nse.py first.")
        sys.exit(1)

    ts, kind, payload_json = row
    payload = json.loads(payload_json)

    print(f"=== Latest option_chain snapshot ===")
    print(f"timestamp: {ts}")
    print(f"kind:      {kind}")
    print(f"payload bytes: {len(payload_json):,}")
    print()

    if not isinstance(payload, dict):
        print(f"[!] Top-level is not a dict — it's a {type(payload).__name__}")
        print("First 500 chars:")
        print(payload_json[:500])
        return

    print("--- Top-level keys ---")
    for key in payload.keys():
        v = payload[key]
        if isinstance(v, list):
            print(f"  {key:20s}  list[{len(v)}]")
        elif isinstance(v, dict):
            print(f"  {key:20s}  dict{{{', '.join(list(v.keys())[:5])}{'...' if len(v) > 5 else ''}}}")
        else:
            sample = repr(v)
            if len(sample) > 80:
                sample = sample[:77] + "..."
            print(f"  {key:20s}  {type(v).__name__}: {sample}")

    # Look for the strike data in the conventional locations.
    print("\n--- Schema probe ---")
    for path in [
        ("records", "data"),
        ("filtered", "data"),
        ("data",),
    ]:
        cursor = payload
        try:
            for p in path:
                cursor = cursor[p]
        except (KeyError, TypeError):
            print(f"  payload[{']['.join(repr(p) for p in path)}]  ABSENT")
            continue
        if isinstance(cursor, list):
            n = len(cursor)
            print(f"  payload[{']['.join(repr(p) for p in path)}]  list[{n}]")
            if n > 0:
                first = cursor[0]
                if isinstance(first, dict):
                    print(f"    sample keys: {list(first.keys())}")
                    print(f"    sample: {json.dumps(first, indent=2)[:400]}")
        else:
            print(f"  payload[{']['.join(repr(p) for p in path)}]  "
                  f"{type(cursor).__name__}")

    # Schema-shift heuristic
    print("\n--- Conclusion ---")
    records = payload.get("records") or {}
    data = records.get("data") if isinstance(records, dict) else None
    if data is None:
        print("  records.data path is missing — NSE schema HAS shifted.")
    elif data == []:
        print("  records.data exists but is EMPTY.")
        print("  Possible causes:")
        print("    1. Pre-market / no trades yet")
        print("    2. NSE returns envelope-only when symbol unrecognised")
        print("    3. Cookie/auth issue specific to option-chain-indices")
        print(f"  Underlying value reported: "
              f"{records.get('underlyingValue')}")
    else:
        print(f"  records.data has {len(data)} rows — parse_option_chain "
              f"should have returned non-empty.")
        print("  If verify_nse showed 0 rows, the parser has a bug.")


if __name__ == "__main__":
    main()
