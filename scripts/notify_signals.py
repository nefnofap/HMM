#!/usr/bin/env python3
"""
notify_signals.py
=================

Read the regime bundle that ``export_regime.py`` just wrote
(``web/data/regimes.json``) and push a Discord alert for any market whose
regime has CHANGED since the previous run.

Run this right after the export step in CI. It is a no-op (exit 0, nothing
sent) unless ``DISCORD_WEBHOOK_URL`` is configured, so it is safe to leave in
the pipeline before you have a webhook set up.

    python scripts/notify_signals.py
    python scripts/notify_signals.py --dry-run        # preview, no posting

Tuning is read from the environment by ``discord_signals`` (see that module's
header): SIGNAL_MIN_CONFIDENCE, SIGNAL_DIRECTIONS, SIGNAL_ALERT_NEUTRAL,
SIGNAL_WEBHOOK_NAME.

> Research/education only. Not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import discord_signals  # noqa: E402

BUNDLE_PATH = os.path.join(ROOT, "web", "data", "regimes.json")
STATE_PATH = os.path.join(ROOT, "web", "data", "signal_state.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Push regime-change alerts to Discord.")
    ap.add_argument("--bundle", default=BUNDLE_PATH,
                    help="path to the regime bundle JSON (default: web/data/regimes.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what WOULD be sent without posting or saving state")
    args = ap.parse_args()

    try:
        with open(args.bundle, "r") as fh:
            bundle = json.load(fh)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[discord] no bundle to read ({exc}); nothing to alert.")
        return

    if not os.getenv("DISCORD_WEBHOOK_URL", "").strip() and not args.dry_run:
        print("[discord] DISCORD_WEBHOOK_URL not set; skipping (no alerts sent).")
        # Still advance state so the first run after you add a webhook does not
        # flood with every market's already-standing regime.
        discord_signals.notify_from_bundle(bundle, state_path=STATE_PATH)
        return

    sent = discord_signals.notify_from_bundle(
        bundle, state_path=STATE_PATH, dry_run=args.dry_run)

    if not sent:
        print("[discord] no new regime changes to alert.")
        return

    verb = "would alert" if args.dry_run else "alerted"
    for s in sent:
        print(f"[discord] {verb}: {s['combo_id']} -> {s['label']} "
              f"({s['confidence'] * 100:.0f}%)")


if __name__ == "__main__":
    main()
