"""Diagnostic: show what BSE is returning right now and which entries match."""
import asyncio
import filings


async def main() -> None:
    items = await filings._fetch_announcements()
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
        sym = filings.match_ticker(filings._item_title(it))
        flag = sym if sym else "-"
        if sym:
            matched += 1
        print(f"  [{flag:14s}] {name}")
    print()
    print(f"matched watchlist: {matched} / {len(items)}")


if __name__ == "__main__":
    asyncio.run(main())
