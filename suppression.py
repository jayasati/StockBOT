"""
Risk-flag suppression layer.

Blocks alerts on stocks that are structurally compromised:
  - on NSE ASM stage 2+ (price-band/margin restrictions tightened)
  - on NSE GSM (any stage — graded surveillance, severe restrictions)
  - have promoter pledge % > 50 (financial distress)
  - already alerted within the cooldown window (anti-spam)

Tables (in alerts.db):
  risk_flags   (symbol, flag_type, value, updated_at)
  alerts_sent  (existing — owned by bot.py; we read for cooldown)

Public API:
  init_db()
  refresh_asm_gsm()      — scrape NSE ASM/GSM lists; persist
  refresh_pledge_data()  — populate from HIGH_PLEDGE_STOCKS dict (TODO: BSE XML)
  is_suppressed(sym, cooldown_minutes) -> (bool, reason)

Run standalone (`python suppression.py`) to refresh + dump the latest
ASM/GSM tables to stdout. Useful for verifying the scraper.
"""

import asyncio
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import httpx

log = logging.getLogger("alertbot.suppression")

DB_PATH = Path("alerts.db")

NSE_HOME = "https://www.nseindia.com/"
NSE_ASM_PAGE = "https://www.nseindia.com/reports/asm"
NSE_GSM_PAGE = "https://www.nseindia.com/reports/gsm"

# NSE serves the ASM/GSM data as JSON behind these endpoints. Keep as a
# list so we can add fallbacks if NSE renames them again (they have
# historically); reportSurveillance was an older name that 404s today.
NSE_ASM_API_CANDIDATES = ["https://www.nseindia.com/api/reportASM"]
NSE_GSM_API_CANDIDATES = ["https://www.nseindia.com/api/reportGSM"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_flags (
    symbol     TEXT NOT NULL,
    flag_type  TEXT NOT NULL CHECK (flag_type IN ('asm', 'gsm', 'pledge_pct', 'high_pe')),
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, flag_type)
);
CREATE INDEX IF NOT EXISTS idx_risk_flags_type ON risk_flags(flag_type);
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


# ============================================================================
# Pledge data (manual for now)
# ============================================================================
# TODO: scrape BSE's shareholding pattern XML per company and populate
# automatically. For now, the user adds known high-pledge stocks here;
# refresh_pledge_data() copies these into risk_flags so is_suppressed()
# treats them like ASM/GSM hits.
#
# Sources to look up promoter pledge %:
#   - trendlyne.com / screener.in (per-stock fundamentals page)
#   - BSE shareholding pattern PDFs (quarterly)
# Threshold > 50% is the rule of thumb for "structurally compromised".
# ============================================================================

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


# ============================================================================
# NSE ASM / GSM scraper
# ============================================================================
# NSE blocks any request that doesn't first hit the homepage to set
# session cookies (nseappid, bm_sv, etc.). The pattern is:
#   1. GET / (homepage)
#   2. GET /reports/asm (the page the API call would come from)
#   3. GET /api/... with Referer pointing back to the page in step 2
# Without this dance NSE returns 401/403 (and sometimes a 200 with HTML
# instead of JSON). We retry across multiple known endpoint names since
# NSE has renamed them in the past.
# ============================================================================


def _parse_asm(data: dict) -> list[tuple[str, str]]:
    """Extract (symbol, stage_label) from NSE ASM response.

    NSE ASM response shape:
      {"longterm":  {"data": [{symbol, asmSurvIndicator, ...}, ...]},
       "shortterm": {"data": [{symbol, asmSurvIndicator, ...}, ...]}}
    We prefix the stage with LT-ASM / ST-ASM so the suppression rule
    can treat them differently (any ST-ASM stage is restrictive,
    while LT-ASM Stage I is informational only).
    """
    out: list[tuple[str, str]] = []
    if not isinstance(data, dict):
        return out
    for key, prefix in (("longterm", "LT-ASM"), ("shortterm", "ST-ASM")):
        section = data.get(key) or {}
        for row in section.get("data") or []:
            sym = str(row.get("symbol") or "").strip().upper()
            indicator = str(row.get("asmSurvIndicator") or "").strip()
            if sym and indicator:
                out.append((sym, f"{prefix} {indicator}"))
    return out


_GSM_STAGE_RE = re.compile(r"\bGSM\s+(?:stage\s+)?([0-9IVX]+)\b", re.IGNORECASE)


def _parse_gsm(data) -> list[tuple[str, str]]:
    """Extract (symbol, stage_label) from NSE GSM response.

    The GSM stage shows up cleanly inside survCode/survDesc (e.g.
    "...GSM 0 (62)") — the gsmStage field sometimes carries a serial
    number ('LXII') instead of the stage. Prefer regex over the raw
    field, fall back to gsmStage.
    """
    rows_out: list[tuple[str, str]] = []

    def iter_items(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from iter_items(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and any(
                    k.lower() == "symbol" for k in item.keys()
                ):
                    yield item
                else:
                    yield from iter_items(item)

    for item in iter_items(data):
        sym = str(item.get("symbol") or item.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        stage = ""
        for field in ("survDesc", "survCode"):
            text = str(item.get(field) or "")
            m = _GSM_STAGE_RE.search(text)
            if m:
                stage = f"Stage {m.group(1).upper()}"
                break
        if not stage:
            raw = str(item.get("gsmStage") or "").strip()
            if raw:
                # NSE sometimes puts a serial number (e.g. "LXII") here
                # instead of the actual stage. Keep as-is — operationally
                # we suppress on any GSM presence regardless of stage.
                stage = f"Stage {raw}"
        if stage:
            rows_out.append((sym, stage))
    return rows_out


async def _nse_session() -> httpx.AsyncClient:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        # Skip brotli — would need a separate package; gzip is fine.
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    client = httpx.AsyncClient(
        headers=headers, timeout=20, follow_redirects=True
    )
    # Cookie warmup — NSE refuses API calls without these.
    try:
        await client.get(NSE_HOME)
        await client.get(NSE_ASM_PAGE)
    except httpx.RequestError as e:
        log.warning("NSE warmup failed: %s", e)
    return client


async def _fetch_json(
    client: httpx.AsyncClient, url: str, referer: str
) -> dict | list | None:
    try:
        r = await client.get(url, headers={"Referer": referer})
    except httpx.RequestError as e:
        log.warning("NSE %s fetch error: %s", url, e)
        return None
    if r.status_code != 200:
        log.warning("NSE %s returned HTTP %d", url, r.status_code)
        return None
    ct = r.headers.get("content-type", "")
    if "json" not in ct.lower() and not r.text.lstrip().startswith(("{", "[")):
        log.warning("NSE %s returned non-JSON (content-type=%s)", url, ct)
        return None
    try:
        return r.json()
    except ValueError as e:
        log.warning("NSE %s JSON parse failed: %s", url, e)
        return None


async def _fetch_first_working(
    client: httpx.AsyncClient, urls: list[str], referer: str
) -> dict | list | None:
    for url in urls:
        data = await _fetch_json(client, url, referer)
        if data is not None:
            log.info("NSE endpoint OK: %s", url)
            return data
    return None


async def refresh_asm_gsm() -> dict[str, int]:
    """Pull current ASM/GSM lists from NSE and replace the rows in risk_flags."""
    init_db()
    counts = {"asm": 0, "gsm": 0}
    client = await _nse_session()
    try:
        asm_data = await _fetch_first_working(
            client, NSE_ASM_API_CANDIDATES, NSE_ASM_PAGE
        )
        gsm_data = await _fetch_first_working(
            client, NSE_GSM_API_CANDIDATES, NSE_GSM_PAGE
        )
    finally:
        await client.aclose()

    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        # ASM/GSM are authoritative-state lists — wipe and replace each refresh.
        conn.execute("DELETE FROM risk_flags WHERE flag_type IN ('asm', 'gsm')")

        if asm_data is not None:
            for sym, stage in _parse_asm(asm_data):
                conn.execute(
                    "INSERT OR REPLACE INTO risk_flags VALUES (?, ?, ?, ?)",
                    (f"{sym}.NS", "asm", stage, now),
                )
                counts["asm"] += 1
        else:
            log.warning("ASM fetch returned no data — leaving table empty for this refresh")

        if gsm_data is not None:
            for sym, stage in _parse_gsm(gsm_data):
                conn.execute(
                    "INSERT OR REPLACE INTO risk_flags VALUES (?, ?, ?, ?)",
                    (f"{sym}.NS", "gsm", stage, now),
                )
                counts["gsm"] += 1
        else:
            log.warning("GSM fetch returned no data — leaving table empty for this refresh")

    log.info("ASM/GSM refresh: %d ASM flags, %d GSM flags", counts["asm"], counts["gsm"])
    return counts


# ============================================================================
# Suppression check
# ============================================================================

def _asm_suppresses(stage: str) -> bool:
    """Decide whether an ASM stage label warrants suppressing alerts.

    NSE has two ASM tracks:
      - LT-ASM (long-term): Stage I is informational (price band); Stage II
        adds 100% margin. Suppress only Stage II+.
      - ST-ASM (short-term): even Stage I imposes 100% margin and Stage II
        moves to trade-to-trade. Suppress any stage.
    """
    s = stage.upper()
    if "ST-ASM" in s:
        return True
    if "LT-ASM" in s:
        return any(
            tag in s
            for tag in ("STAGE II", "STAGE III", "STAGE IV", "STAGE 2", "STAGE 3", "STAGE 4")
        )
    # Unknown prefix — be conservative: any stage 2+ suppresses.
    return any(
        tag in s
        for tag in ("STAGE II", "STAGE III", "STAGE IV", "STAGE 2", "STAGE 3", "STAGE 4")
    )


def is_suppressed(symbol: str, cooldown_minutes: int = 60) -> tuple[bool, str]:
    """Return (suppressed, reason). Reason is empty when not suppressed."""
    cutoff = (datetime.now() - timedelta(minutes=cooldown_minutes)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        # Cooldown — was alerted in the last N minutes
        row = conn.execute(
            "SELECT 1 FROM alerts_sent WHERE symbol = ? AND sent_at > ? LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        if row:
            return True, f"cooldown ({cooldown_minutes}m)"

        # ASM stage 2+
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'asm'",
            (symbol,),
        ).fetchone()
        if row and _asm_suppresses(row[0]):
            return True, f"ASM {row[0]}"

        # GSM — any stage suppresses
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'gsm'",
            (symbol,),
        ).fetchone()
        if row:
            return True, f"GSM {row[0]}"

        # Promoter pledge > 50%
        row = conn.execute(
            "SELECT value FROM risk_flags WHERE symbol = ? AND flag_type = 'pledge_pct'",
            (symbol,),
        ).fetchone()
        if row:
            try:
                if float(row[0]) > 50:
                    return True, f"pledge {row[0]}%"
            except ValueError:
                pass

    return False, ""


# ============================================================================
# Standalone diagnostic
# ============================================================================

async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    counts = await refresh_asm_gsm()
    print()
    print(f"ASM rows persisted: {counts['asm']}")
    print(f"GSM rows persisted: {counts['gsm']}")
    print()
    with sqlite3.connect(DB_PATH) as conn:
        for flag_type in ("asm", "gsm"):
            rows = conn.execute(
                "SELECT symbol, value FROM risk_flags WHERE flag_type = ? "
                "ORDER BY value, symbol",
                (flag_type,),
            ).fetchall()
            print(f"--- {flag_type.upper()} ({len(rows)} rows) ---")
            for sym, val in rows[:20]:
                print(f"  {sym:18s} {val}")
            if len(rows) > 20:
                print(f"  ... and {len(rows) - 20} more")
            print()


if __name__ == "__main__":
    asyncio.run(_main())
