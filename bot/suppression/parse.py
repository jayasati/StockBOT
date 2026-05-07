"""ASM/GSM JSON → (symbol, stage_label) tuples."""
from __future__ import annotations

import re

_GSM_STAGE_RE = re.compile(r"\bGSM\s+(?:stage\s+)?([0-9IVX]+)\b", re.IGNORECASE)


def _parse_asm(data: dict) -> list[tuple[str, str]]:
    """Extract (symbol, stage_label) from NSE ASM response.

    NSE ASM response shape:
      {"longterm":  {"data": [{symbol, asmSurvIndicator, ...}, ...]},
       "shortterm": {"data": [{symbol, asmSurvIndicator, ...}, ...]}}
    We prefix the stage with LT-ASM / ST-ASM so the suppression rule
    can treat them differently (any ST-ASM stage is restrictive,
    while LT-ASM Stage I is informational only)."""
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


def _parse_gsm(data) -> list[tuple[str, str]]:
    """Extract (symbol, stage_label) from NSE GSM response.

    The GSM stage shows up cleanly inside survCode/survDesc (e.g.
    "...GSM 0 (62)") — the gsmStage field sometimes carries a serial
    number ('LXII') instead of the stage. Prefer regex over the raw
    field, fall back to gsmStage."""
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
