"""Backtest-mode reporting: summary tables, regime A/B lift, CSV export."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import WIN_THRESHOLD_PCT, compute_win_rate

from .evaluate import SwingAlert

log = logging.getLogger("swing")


def summarize(alerts: list[SwingAlert], label: str) -> str:
    if not alerts:
        return f"\n{label}\n  (no alerts)"

    df = pd.DataFrame([asdict(a) for a in alerts])
    df["signal_date"] = pd.to_datetime(df["signal_date"])

    out: list[str] = []
    bar = "=" * 76
    out.append("")
    out.append(bar)
    out.append(f"SWING BACKTEST  ({label})")
    out.append(f"Alerts: {len(alerts)}   "
               f"Period: {df['signal_date'].min().date()} → "
               f"{df['signal_date'].max().date()}")
    out.append(f"Win threshold: >= {WIN_THRESHOLD_PCT}% move")
    out.append(bar)
    out.append("")
    out.append(f"{'Hold':<8} {'Count':>7} {'WinRate':>10} {'AvgRet':>10} "
               f"{'MedRet':>10} {'StdDev':>10}")
    out.append("-" * 60)
    for h_label, col in [("1-day", "ret_1d"), ("3-day", "ret_3d"), ("5-day", "ret_5d")]:
        rets = df[col].dropna().tolist()
        if not rets:
            continue
        wr = compute_win_rate(rets)
        avg = float(np.mean(rets))
        med = float(np.median(rets))
        std = float(np.std(rets))
        out.append(f"{h_label:<8} {len(rets):>7} {wr:>9.1f}% "
                   f"{avg:>9.2f}% {med:>9.2f}% {std:>9.2f}%")

    have_5 = df[df["ret_5d"].notna()]
    if not have_5.empty:
        out.append("")
        out.append("BEST 5 (5-day return):")
        for _, r in have_5.nlargest(5, "ret_5d").iterrows():
            out.append(f"  {r['symbol']:<14} {r['signal_date'].date()}  "
                       f"vol={r['volume_mult']:>4.1f}x  "
                       f"rp={r['range_position']:.2f}  "
                       f"emaΔ={r['ema_pct']:+.1f}%  "
                       f"ret_5d={r['ret_5d']:+.2f}%")
        out.append("")
        out.append("WORST 5 (5-day return):")
        for _, r in have_5.nsmallest(5, "ret_5d").iterrows():
            out.append(f"  {r['symbol']:<14} {r['signal_date'].date()}  "
                       f"vol={r['volume_mult']:>4.1f}x  "
                       f"rp={r['range_position']:.2f}  "
                       f"emaΔ={r['ema_pct']:+.1f}%  "
                       f"ret_5d={r['ret_5d']:+.2f}%")

    # Conviction analysis: split by volume multiple bucket
    out.append("")
    out.append("BY VOLUME-MULTIPLE BUCKET (5-day return):")
    out.append(f"  {'Bucket':<14} {'n':>5} {'WinRate':>9} {'AvgRet':>9}")
    buckets = [
        ("vol 2.0–3.0x", lambda v: 2.0 <= v < 3.0),
        ("vol 3.0–5.0x", lambda v: 3.0 <= v < 5.0),
        ("vol >= 5.0x",  lambda v: v >= 5.0),
    ]
    for name, pred in buckets:
        sub = have_5[have_5["volume_mult"].apply(pred)]
        if sub.empty:
            out.append(f"  {name:<14} {0:>5}    (none)")
            continue
        rets = sub["ret_5d"].tolist()
        out.append(f"  {name:<14} {len(sub):>5} "
                   f"{compute_win_rate(rets):>8.1f}% "
                   f"{float(np.mean(rets)):>+8.2f}%")

    out.append(bar)
    return "\n".join(out)


def regime_lift(
    alerts_off: list[SwingAlert], alerts_on: list[SwingAlert]
) -> str:
    """Compare 5-day stats with and without regime filter."""
    def stats(alerts):
        if not alerts:
            return (0, 0.0, 0.0)
        rets = [a.ret_5d for a in alerts if a.ret_5d is not None]
        if not rets:
            return (len(alerts), 0.0, 0.0)
        return (len(alerts), compute_win_rate(rets), float(np.mean(rets)))

    n_off, wr_off, av_off = stats(alerts_off)
    n_on, wr_on, av_on = stats(alerts_on)

    out: list[str] = []
    out.append("")
    out.append("=" * 76)
    out.append("REGIME FILTER LIFT  (5-day hold)")
    out.append("=" * 76)
    out.append(f"  {'Mode':<14} {'Alerts':>8} {'WinRate':>10} {'AvgRet':>10}")
    out.append(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    out.append(f"  {'regime OFF':<14} {n_off:>8} {wr_off:>9.1f}% {av_off:>+9.2f}%")
    out.append(f"  {'regime ON':<14} {n_on:>8} {wr_on:>9.1f}% {av_on:>+9.2f}%")
    out.append(f"  {'lift':<14} {n_on - n_off:>+8} {wr_on - wr_off:>+9.1f}pp "
               f"{av_on - av_off:>+9.2f}pp")
    out.append("")
    out.append("Read: positive lift on win rate AND avg return = regime filter")
    out.append("is selecting the up-tape days where momentum signals work.")
    out.append("=" * 76)
    return "\n".join(out)


def save_csv(alerts: list[SwingAlert], path: Path) -> None:
    if not alerts:
        log.warning("No alerts to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(a) for a in alerts]).to_csv(path, index=False)
    log.info("Saved %d alerts to %s", len(alerts), path)
