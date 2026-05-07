"""CLI verifiers: auth, validate, ticks, bars, record."""
from __future__ import annotations

import json
import logging
import sys
import time as _time

import pandas as pd

from .auth import authenticate
from .bars import TICK_STORE
from .creds import load_creds
from .logging import fyers_log_path
from .record import record_ticks
from .token_cache import TOKEN_CACHE
from .websocket import start_data_socket

log = logging.getLogger("alertbot.fyers")


def _cli_auth(force: bool) -> int:
    try:
        token = authenticate(force=force)
    except Exception as e:
        log.exception("Auth failed: %s", e)
        return 2
    print()
    print(f"  Got Fyers access token (length={len(token)})")
    print(f"  Cached at: {TOKEN_CACHE.absolute()}")
    if TOKEN_CACHE.exists():
        data = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        print(f"  Expires:   {data['expiry']} IST")
    return 0


def _cli_validate() -> int:
    try:
        token = authenticate()
        creds = load_creds()
    except Exception as e:
        log.exception("Auth failed: %s", e)
        return 2
    from fyers_apiv3 import fyersModel
    fyers = fyersModel.FyersModel(
        client_id=creds.app_id, token=token, log_path=fyers_log_path()
    )
    profile = fyers.get_profile()
    print(json.dumps(profile, indent=2))
    if profile.get("s") != "ok":
        return 3
    return 0


def _cli_ticks() -> int:
    """Subscribe to 3 symbols and print incoming ticks."""
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("WebSocket test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Ticks flow only during NSE market hours (Mon-Fri 09:15-15:30 IST).")
    print("Press Ctrl+C to stop.")
    print()

    counter = {"n": 0}

    def on_tick(msg):
        counter["n"] += 1
        print(f"  [{counter['n']:5d}] {msg}")

    try:
        start_data_socket(symbols, on_tick)
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopped. {counter['n']} ticks received total.")
        return 0
    except Exception as e:
        log.exception("ticks command failed: %s", e)
        return 2


def _cli_bars() -> int:
    """Subscribe to 3 symbols and dump the aggregated bars on stop."""
    symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
    print("Bar-aggregator test — subscribing to:")
    for s in symbols:
        print(f"   {s}")
    print()
    print("Run during market hours (09:15-15:30 IST) for at least one full")
    print("5-minute boundary crossing. Ctrl+C to stop and dump.")
    print()
    try:
        start_data_socket(symbols, TICK_STORE.on_tick)
        last_log = 0.0
        while True:
            _time.sleep(2)
            now = _time.time()
            if now - last_log >= 10:
                stats = TICK_STORE.stats()
                print(
                    f"  [stats] ticks={stats['ticks']} symbols={stats['symbols_seen']} "
                    f"open_bars={stats['open_bars']} archived={stats['archived_bars']}"
                )
                last_log = now
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print(f"Final stats: {TICK_STORE.stats()}")
        print("=" * 70)
        for sym in symbols:
            df = TICK_STORE.get_bars_df(sym)
            print()
            print(f"--- {sym} ({len(df)} bars) ---")
            if df.empty:
                print("   (no bars built — no qualifying ticks received)")
                continue
            with pd.option_context("display.max_rows", 20, "display.width", 120):
                print(df)
        return 0
    except Exception as e:
        log.exception("bars command failed: %s", e)
        return 2


def _cli_record(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: python -m fyers_client record <output.jsonl> <duration_sec> [SYM1 SYM2 ...]")
        return 1
    out = argv[2]
    try:
        dur = int(argv[3])
    except ValueError:
        print(f"duration must be an integer; got {argv[3]!r}")
        return 1
    syms = argv[4:] or ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ", "NSE:ICICIBANK-EQ"]
    n = record_ticks(syms, out, dur)
    print(f"Captured {n} ticks to {out}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if len(sys.argv) < 2:
        print("usage: python -m fyers_client {auth|validate|ticks|bars|record} [--force]")
        return 1
    cmd = sys.argv[1]
    if cmd == "auth":
        return _cli_auth(force="--force" in sys.argv)
    if cmd == "validate":
        return _cli_validate()
    if cmd == "ticks":
        return _cli_ticks()
    if cmd == "bars":
        return _cli_bars()
    if cmd == "record":
        return _cli_record(sys.argv)
    print(f"unknown command: {cmd}")
    return 1
