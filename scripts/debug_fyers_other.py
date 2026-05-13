r"""Sanity-check that Fyers auth/network is fine by hitting a known-
working endpoint. If quotes() succeeds but optionchain() fails with
-99 Bad request, the issue is option-chain permission on your Fyers
app, not the code.

Run from the project root:
    venv\Scripts\python.exe scripts\debug_fyers_other.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from fyers_apiv3 import fyersModel
    from fyers_client.creds import load_creds
    from fyers_client.token_cache import _load_cached_token

    token = _load_cached_token()
    if not token:
        print("No cached Fyers token.")
        sys.exit(1)

    creds = load_creds()
    fyers = fyersModel.FyersModel(
        client_id=creds.app_id, token=token, is_async=False,
    )

    # 1. Profile — does auth work at all?
    print("--- profile() ---")
    try:
        r = fyers.get_profile()
        print(f"  s={r.get('s')!r}  code={r.get('code')!r}")
        if r.get("s") == "ok":
            data = r.get("data") or {}
            print(f"  name: {data.get('name')!r}")
            print(f"  fy_id: {data.get('fy_id')!r}")
    except Exception as e:
        print(f"  raised: {e}")

    # 2. Quotes on NIFTY index — minimal data call
    print("\n--- quotes(NIFTY50-INDEX) ---")
    try:
        r = fyers.quotes(data={"symbols": "NSE:NIFTY50-INDEX"})
        print(f"  s={r.get('s')!r}  code={r.get('code')!r}")
        if r.get("s") == "ok":
            d = (r.get("d") or [])
            if d:
                print(f"  LTP: {d[0].get('v', {}).get('lp')!r}")
    except Exception as e:
        print(f"  raised: {e}")

    # 3. Option chain with FULL response dump so we see EVERY field
    print("\n--- optionchain (full response dump) ---")
    try:
        r = fyers.optionchain(data={
            "symbol": "NSE:NIFTY50-INDEX",
            "strikecount": 5,
        })
        print(json.dumps(r, indent=2, default=str)[:1500])
    except Exception as e:
        print(f"  raised: {e}")


if __name__ == "__main__":
    main()
