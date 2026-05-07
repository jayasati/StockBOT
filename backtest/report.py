"""Backtest reporting: win-rate, summary, threshold sweep, CSV export."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import COOLDOWN_MINUTES, RESULTS_CSV, SCORE_THRESHOLD, WIN_THRESHOLD_PCT
from .replay import AlertRecord

log = logging.getLogger("backtest")


def compute_win_rate(
    returns: list[Optional[float]], threshold_pct: float = WIN_THRESHOLD_PCT
) -> float:
    valid = [r for r in returns if r is not None and not (isinstance(r, float) and np.isnan(r))]
    if not valid:
        return 0.0
    wins = sum(1 for r in valid if r >= threshold_pct)
    return wins / len(valid) * 100


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:+.2f}%" if x is not None and not np.isnan(x) else "  n/a"


def summarize(records: list[AlertRecord]) -> str:
    if not records:
        return "No alerts generated. Try lowering --threshold or check data coverage."

    df = pd.DataFrame([asdict(r) for r in records])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    out: list[str] = []
    bar = "=" * 76
    out.append("")
    out.append(bar)
    out.append(f"BACKTEST REPORT — {len(records)} alerts "
               f"(score >= {SCORE_THRESHOLD}, cooldown {COOLDOWN_MINUTES}m)")
    out.append(f"Period: {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    out.append(f"Win threshold: >= {WIN_THRESHOLD_PCT}% move")
    out.append(bar)

    out.append("")
    out.append(f"{'Horizon':<10} {'Count':>7} {'WinRate':>10} {'AvgRet':>10} "
               f"{'MedRet':>10} {'StdDev':>10}")
    out.append("-" * 60)
    for label, col in [("+30min", "ret_30m"), ("+1 day", "ret_1d"), ("+5 days", "ret_5d")]:
        rets = df[col].dropna().tolist()
        if not rets:
            out.append(f"{label:<10} {0:>7} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        wr = compute_win_rate(rets)
        avg = float(np.mean(rets))
        med = float(np.median(rets))
        std = float(np.std(rets))
        out.append(f"{label:<10} {len(rets):>7} {wr:>9.1f}% "
                   f"{avg:>9.2f}% {med:>9.2f}% {std:>9.2f}%")

    out.append("")
    out.append("BEST 5 TRADES (by +1d return):")
    have_1d = df[df["ret_1d"].notna()]
    for _, r in have_1d.nlargest(5, "ret_1d").iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        out.append(f"  {r['symbol']:<14} {ts}  score={r['score']:>3}  "
                   f"+1d={_fmt_pct(r['ret_1d'])}  [{r['reasons']}]")

    out.append("")
    out.append("WORST 5 TRADES (by +1d return):")
    for _, r in have_1d.nsmallest(5, "ret_1d").iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        out.append(f"  {r['symbol']:<14} {ts}  score={r['score']:>3}  "
                   f"+1d={_fmt_pct(r['ret_1d'])}  [{r['reasons']}]")

    # Per-component analysis: each row in the reasons string contains tokens
    # like "VR 3.2x", "RSI 64", "> VWAP", "BO +1.5%", "near high". We bucket
    # alerts by which components fired, then compare +1d win rate of "with"
    # vs "without" each component.
    out.append("")
    out.append("PER-COMPONENT CONTRIBUTION  (+1d return, win >= "
               f"{WIN_THRESHOLD_PCT}%):")
    out.append(f"  {'Component':<16} {'With_n':>7} {'WR_with':>9} "
               f"{'Without_n':>10} {'WR_without':>11} {'Lift':>8}")
    out.append("  " + "-" * 70)

    components = {
        "Volume ratio": lambda s: "VR " in s,
        "RSI in zone":  lambda s: "RSI " in s,
        "Above VWAP":   lambda s: "VWAP" in s,
        "Pullback":     lambda s: "pullback" in s,
        "Extended":     lambda s: "extended" in s,
    }
    df_clean = df[df["ret_1d"].notna()].copy()
    for name, predicate in components.items():
        mask = df_clean["reasons"].apply(predicate)
        with_g = df_clean.loc[mask, "ret_1d"].tolist()
        without_g = df_clean.loc[~mask, "ret_1d"].tolist()
        wr_w = compute_win_rate(with_g) if with_g else 0.0
        wr_o = compute_win_rate(without_g) if without_g else 0.0
        lift = wr_w - wr_o
        out.append(f"  {name:<16} {len(with_g):>7} {wr_w:>8.1f}% "
                   f"{len(without_g):>10} {wr_o:>10.1f}% {lift:>+7.1f}pp")

    out.append("")
    out.append(f"All alerts saved to: {RESULTS_CSV}")
    out.append(bar)
    return "\n".join(out)


def save_csv(records: list[AlertRecord], path: Path) -> None:
    if not records:
        log.warning("No records to save.")
        return
    df = pd.DataFrame([asdict(r) for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d alerts to %s", len(df), path)


def apply_cooldown(
    records: list[AlertRecord], threshold: int, cooldown_minutes: int
) -> list[AlertRecord]:
    """Filter records to score >= threshold, then apply per-symbol cooldown."""
    cooldown_td = timedelta(minutes=cooldown_minutes)
    by_symbol: dict[str, list[AlertRecord]] = {}
    for r in records:
        if r.score < threshold:
            continue
        by_symbol.setdefault(r.symbol, []).append(r)
    out: list[AlertRecord] = []
    for rs in by_symbol.values():
        rs.sort(key=lambda r: r.timestamp)
        last: Optional[pd.Timestamp] = None
        for r in rs:
            if last is not None and (r.timestamp - last) < cooldown_td:
                continue
            last = r.timestamp
            out.append(r)
    out.sort(key=lambda r: r.timestamp)
    return out


def sweep_summary(
    all_records: list[AlertRecord], thresholds: list[int]
) -> str:
    """Print a comparison table across score thresholds.

    Sanity check: a well-calibrated score should produce HIGHER win rates as
    the threshold rises, even if absolute win rates are weak.
    """
    out: list[str] = []
    bar = "=" * 78
    out.append("")
    out.append(bar)
    out.append(f"THRESHOLD SWEEP  ({len(all_records)} raw events scored, "
               f"cooldown {COOLDOWN_MINUTES}m applied per threshold)")
    out.append(bar)
    out.append(f"{'Threshold':>10} {'Alerts':>8} {'WR_30m':>9} {'WR_1d':>9} "
               f"{'WR_5d':>9} {'AvgRet_1d':>11} {'AvgRet_5d':>11}")
    out.append("-" * 78)
    for thr in thresholds:
        recs = apply_cooldown(all_records, thr, COOLDOWN_MINUTES)
        if not recs:
            out.append(f"{thr:>10} {0:>8}   (no alerts at this threshold)")
            continue
        ret30 = [r.ret_30m for r in recs if r.ret_30m is not None]
        ret1 = [r.ret_1d for r in recs if r.ret_1d is not None]
        ret5 = [r.ret_5d for r in recs if r.ret_5d is not None]
        wr30 = compute_win_rate(ret30) if ret30 else 0.0
        wr1 = compute_win_rate(ret1) if ret1 else 0.0
        wr5 = compute_win_rate(ret5) if ret5 else 0.0
        avg1 = float(np.mean(ret1)) if ret1 else 0.0
        avg5 = float(np.mean(ret5)) if ret5 else 0.0
        out.append(f"{thr:>10} {len(recs):>8} {wr30:>8.1f}% {wr1:>8.1f}% "
                   f"{wr5:>8.1f}% {avg1:>10.2f}% {avg5:>10.2f}%")
    out.append("")
    out.append("Read: a calibrated score gives rising win rates as threshold")
    out.append("rises. Flat or falling = the score is not ranking trades.")
    out.append(bar)
    return "\n".join(out)
