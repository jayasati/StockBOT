"""``python -m data.filings check`` — diagnostic dump of current BSE returns.

Replaces the old ``scripts/check_filings.py``.
"""
from __future__ import annotations

import asyncio
import sys

from . import _fetch_announcements, _item_title, match_ticker


async def _check() -> None:
    items = await _fetch_announcements()
    if items is None:
        print("FETCH FAILED (BSE API unreachable or blocked)")
        return
    print(f"items returned: {len(items)}")
    if not items:
        print("(BSE returned 0 announcements — likely off-hours / weekend)")
        return
    print()
    print("company names returned by BSE right now:")
    matched = 0
    for it in items:
        name = (it.get("SLONGNAME") or "").strip()
        sym = match_ticker(_item_title(it))
        flag = sym if sym else "-"
        if sym:
            matched += 1
        print(f"  [{flag:14s}] {name}")
    print()
    print(f"matched watchlist: {matched} / {len(items)}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m data.filings {check}")
        return 1
    cmd = sys.argv[1]
    if cmd == "check":
        asyncio.run(_check())
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
