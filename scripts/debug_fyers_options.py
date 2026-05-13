r"""Try several Fyers optionchain parameter combinations live, print
which ones succeed. Fyers -99 'Bad request' means the SDK got through
but Fyers rejected the params; we just need to find the working set.

Run from the project root:
    venv\Scripts\python.exe scripts\debug_fyers_options.py
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
        print("No cached Fyers token — run `python -m fyers_client auth` first.")
        sys.exit(1)

    creds = load_creds()
    fyers = fyersModel.FyersModel(
        client_id=creds.app_id, token=token, is_async=False,
    )

    attempts = [
        # Try 1: current (failing) — empty string timestamp + int strikecount
        ("v1: timestamp=''", {"symbol": "NSE:NIFTY50-INDEX",
                               "strikecount": 20, "timestamp": ""}),
        # Try 2: omit timestamp entirely
        ("v2: no timestamp",  {"symbol": "NSE:NIFTY50-INDEX",
                               "strikecount": 20}),
        # Try 3: strikecount as string
        ("v3: strikecount str", {"symbol": "NSE:NIFTY50-INDEX",
                                  "strikecount": "20", "timestamp": ""}),
        # Try 4: smaller strikecount
        ("v4: strikecount=5",   {"symbol": "NSE:NIFTY50-INDEX",
                                  "strikecount": 5, "timestamp": ""}),
        # Try 5: omit timestamp + str strikecount
        ("v5: no ts + str sc",  {"symbol": "NSE:NIFTY50-INDEX",
                                  "strikecount": "5"}),
        # Try 6: alternative symbol form
        ("v6: NIFTY-INDEX",     {"symbol": "NSE:NIFTY-INDEX",
                                  "strikecount": 5}),
        # Try 7: with greeks param
        ("v7: greeks=1",        {"symbol": "NSE:NIFTY50-INDEX",
                                  "strikecount": 5, "timestamp": "", "greeks": "1"}),
    ]

    for label, data in attempts:
        print(f"\n--- {label} ---")
        print(f"  request: {data}")
        try:
            r = fyers.optionchain(data=data)
        except Exception as e:
            print(f"  raised: {type(e).__name__}: {e}")
            continue
        s = r.get("s")
        code = r.get("code")
        msg = r.get("message")
        print(f"  s={s!r}  code={code!r}  message={msg!r}")
        if s == "ok":
            data_dict = r.get("data") or {}
            chain = data_dict.get("optionsChain") or []
            expiries = data_dict.get("expiryData") or []
            print(f"  optionsChain rows: {len(chain)}")
            print(f"  expiryData entries: {len(expiries)}")
            if chain:
                # First non-INDEX entry
                first_strike = next(
                    (c for c in chain if (c.get("option_type") or "") in ("CE", "PE")),
                    None,
                )
                if first_strike:
                    print(f"  sample CE/PE row keys: {list(first_strike.keys())}")
                    print(f"  sample: {json.dumps({k: first_strike.get(k) for k in ('symbol', 'strike_price', 'option_type', 'oi', 'expiry')}, indent=2)}")
            print(f"  >> WORKING params: {data}")
            return


if __name__ == "__main__":
    main()
