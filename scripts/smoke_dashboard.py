"""Programmatic smoke test for the dashboard pages — uses Streamlit's
AppTest harness to actually execute each page's Python code and report
any exceptions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGES = [
    "dashboard/app.py",
    "dashboard/pages/1_Live.py",
    "dashboard/pages/2_Paper_Trades.py",
    "dashboard/pages/3_Filter_Audit.py",
    "dashboard/pages/4_Performance.py",
    "dashboard/pages/5_Verify.py",
]

failures: list[str] = []
for page in PAGES:
    print(f"\n--- {page} ---", flush=True)
    try:
        at = AppTest.from_file(str(ROOT / page), default_timeout=20)
        at.run()
        if at.exception:
            for exc in at.exception:
                print(f"  EXCEPTION: {exc.value}")
                failures.append(f"{page}: {exc.value}")
        else:
            print(f"  OK — {len(at.markdown)} markdown blocks, "
                  f"{len(at.metric)} metrics, {len(at.dataframe)} dataframes")
    except Exception as e:
        print(f"  CRASH: {type(e).__name__}: {e}")
        failures.append(f"{page}: CRASH {e}")

print("\n" + "=" * 60)
if failures:
    print(f"FAIL — {len(failures)} page(s) errored:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print("ALL PAGES OK")
