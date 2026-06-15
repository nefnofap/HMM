#!/usr/bin/env python3
"""
scan_signals.py
===============

Run the multi-asset scanner (the same logic that paints LONG / WAIT / STAND
ASIDE on the dashboard) and push a Discord alert whenever an instrument flips
to **LONG** — i.e. it enters the bull regime AND meets the entry confirmations
(7 of 8 on the conservative profile).

This is the "the website says LONG -> send a signal" path. It fits its own HMM
per instrument, so it does not depend on web/data/regimes.json.

    python scripts/scan_signals.py
    python scripts/scan_signals.py --dry-run
    python scripts/scan_signals.py --watchlist "Gold (XAUUSD),Bitcoin (BTC)"

Configuration (environment; alerts are a no-op unless DISCORD_WEBHOOK_URL is set)
--------------------------------------------------------------------------------
    DISCORD_WEBHOOK_URL        incoming webhook URL (store as a GitHub secret!)
    SIGNAL_WATCHLIST           csv of display names / aliases, default "Gold (XAUUSD)"
    SIGNAL_PROFILE             conservative | aggressive, default conservative
    SIGNAL_MIN_CONFIDENCE_PCT  0..100 floor on HMM confidence, default 0 (any LONG)
    SIGNAL_WEBHOOK_NAME        username shown in Discord, default "Regime Radar"

Note: confidence is the HMM's posterior probability of the current state and
rarely equals exactly 100. The 7-of-8 confirmations already make a LONG strict;
raise SIGNAL_MIN_CONFIDENCE_PCT (e.g. 95) only if you want near-certain ones.

> Research/education only. Not financial advice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import discord_signals  # noqa: E402

STATE_PATH = os.path.join(ROOT, "web", "data", "long_signal_state.json")
DEFAULT_WATCHLIST = "Gold (XAUUSD)"


def build_watchlist(raw: str) -> dict:
    """Resolve a csv of display names / aliases to a {display: symbol} map."""
    from regime_detection import DISPLAY_TO_SYMBOL, resolve_ticker

    out: dict = {}
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item in DISPLAY_TO_SYMBOL:
            out[item] = DISPLAY_TO_SYMBOL[item]
        else:
            out[item] = resolve_ticker(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Alert Discord on scanner LONG verdicts.")
    ap.add_argument("--watchlist", default=os.getenv("SIGNAL_WATCHLIST", DEFAULT_WATCHLIST),
                    help="csv of instruments to scan (display names or aliases)")
    ap.add_argument("--profile", default=os.getenv("SIGNAL_PROFILE", "conservative"),
                    choices=["conservative", "aggressive"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print what WOULD alert without posting or saving state")
    args = ap.parse_args()

    from backtest import StrategyProfile
    from scanner import scan

    profile = (StrategyProfile.aggressive() if args.profile == "aggressive"
               else StrategyProfile.conservative())
    instruments = build_watchlist(args.watchlist)
    if not instruments:
        print("[scan] empty watchlist; nothing to do.")
        return

    print(f"[scan] scanning {len(instruments)} instrument(s): "
          + ", ".join(instruments) + f"  (profile={args.profile})")
    df = scan(instruments=instruments, profile=profile)
    rows = df.to_dict("records") if not df.empty else []
    for r in rows:
        print(f"[scan]   {r.get('instrument')}: {r.get('verdict')} "
              f"({r.get('confidence', 0):.0f}%, {r.get('confirmations')}/8, "
              f"{r.get('status')})")

    if not os.getenv("DISCORD_WEBHOOK_URL", "").strip() and not args.dry_run:
        print("[discord] DISCORD_WEBHOOK_URL not set; skipping (no alerts sent).")
        # Still advance state so the first configured run doesn't flood.
        discord_signals.notify_scanner_signals(rows, state_path=STATE_PATH)
        return

    sent = discord_signals.notify_scanner_signals(
        rows, state_path=STATE_PATH,
        timestamp=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        dry_run=args.dry_run)

    if not sent:
        print("[discord] no new LONG signals to alert.")
        return

    verb = "would alert" if args.dry_run else "alerted"
    for r in sent:
        print(f"[discord] {verb} LONG: {r.get('instrument')} "
              f"({r.get('confidence', 0):.0f}%)")


if __name__ == "__main__":
    main()
