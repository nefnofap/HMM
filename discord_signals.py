"""
discord_signals.py
==================

Push regime SIGNAL ALERTS to a Discord channel via an incoming webhook.

This is deliberately separate from ``discord_auth.py``. That module gates
*login* (Discord OAuth: "is this visitor a member of my server?"). This module
only PUSHES messages OUT: when a tracked market's regime flips into an
actionable state (e.g. turns bullish), we post a formatted embed to a Discord
channel.

It is built to run from the scheduled GitHub Action (``scripts/export_regime.py``)
so alerts fire even when nobody has the dashboard open. De-duplication state is
persisted to ``web/data/signal_state.json`` so each regime change is announced
once - on the run where it flips - not every hour.

Zero extra dependencies: standard-library ``urllib`` only, so it installs and
runs anywhere the rest of the project does (including CI).

Configuration (all via environment; if no webhook is set the module is a no-op)
-------------------------------------------------------------------------------
    DISCORD_WEBHOOK_URL    incoming webhook URL   (store as a GitHub secret!)
    SIGNAL_MIN_CONFIDENCE  float 0..1, default 0.40  - ignore weak regimes
    SIGNAL_DIRECTIONS      csv of {bull,bear},   default "bull,bear"
    SIGNAL_ALERT_NEUTRAL   "1" to also alert when a market returns to neutral
    SIGNAL_WEBHOOK_NAME    username shown in Discord, default "Regime Radar"

> Research/education only. Not financial advice.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

# Discord embed colours (decimal RGB).
COLOR_BULL = 0x54C98C      # green
COLOR_BEAR = 0xF26D5B      # red
COLOR_NEUTRAL = 0x8A93A6   # slate

DEFAULT_MIN_CONFIDENCE = 0.40
DEFAULT_DIRECTIONS = ("bull", "bear")
DEFAULT_USERNAME = "Regime Radar"


# ──────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────
def classify_direction(label: str) -> str:
    """Map a regime label ('Strong bull', 'Soft bear', 'Neutral', ...) to one
    of 'bull' / 'bear' / 'neutral'."""
    low = (label or "").lower()
    if "bull" in low:
        return "bull"
    if "bear" in low:
        return "bear"
    return "neutral"


def _color(direction: str) -> int:
    return {"bull": COLOR_BULL, "bear": COLOR_BEAR}.get(direction, COLOR_NEUTRAL)


def _arrow(direction: str) -> str:
    return {"bull": "🟢 LONG bias", "bear": "🔴 STAND ASIDE",
            "neutral": "⚪ Neutral"}.get(direction, "⚪")


# ──────────────────────────────────────────────────────────────────────────
# Read the export bundle -> actionable signals
# ──────────────────────────────────────────────────────────────────────────
def actionable_signals(bundle: dict, *, min_confidence: float,
                       directions: Tuple[str, ...],
                       alert_neutral: bool) -> Dict[str, dict]:
    """
    Inspect every (asset, source) combo in an export bundle and return the ones
    whose CURRENT regime is worth announcing.

    Returns ``{combo_id: signal_dict}`` where signal_dict carries everything
    ``build_embed`` needs. Non-actionable combos are omitted here but their
    current label is still tracked by ``notify_from_bundle`` for de-dup.
    """
    out: Dict[str, dict] = {}
    payloads = bundle.get("payloads", {}) or {}
    for combo_id, p in payloads.items():
        cur = (p or {}).get("current", {}) or {}
        label = cur.get("label", "—")
        conf = float(cur.get("confidence", 0.0) or 0.0)
        direction = classify_direction(label)

        wanted = direction in directions or (alert_neutral and direction == "neutral")
        if not wanted or conf < min_confidence:
            continue

        # Next-step expected move, if a forecast is present.
        exp_pct = None
        steps = ((p.get("forecast") or {}).get("steps") or [])
        if steps:
            exp_pct = steps[0].get("expectedPct")

        out[combo_id] = {
            "combo_id": combo_id,
            "ticker": p.get("ticker", combo_id),
            "source": p.get("source", "—"),
            "exchange": p.get("exchange", "—"),
            "interval": p.get("interval", "—"),
            "label": label,
            "direction": direction,
            "confidence": conf,
            "expected_pct": exp_pct,
            "generated_at": p.get("generatedAt"),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# De-dup state (persisted JSON: combo_id -> last label we recorded)
# ──────────────────────────────────────────────────────────────────────────
def load_state(path: str) -> Dict[str, str]:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        return dict(data.get("last_label", {})) if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def save_state(path: str, last_label: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"schemaVersion": "1.0", "last_label": last_label}, fh,
                  separators=(",", ":"))


# ──────────────────────────────────────────────────────────────────────────
# Embed formatting
# ──────────────────────────────────────────────────────────────────────────
def build_embed(sig: dict) -> dict:
    direction = sig["direction"]
    conf_pct = f"{sig['confidence'] * 100:.0f}%"
    fields = [
        {"name": "Regime", "value": f"**{sig['label']}**", "inline": True},
        {"name": "Confidence", "value": conf_pct, "inline": True},
        {"name": "Source", "value": f"{sig['source']} · {sig['exchange']}",
         "inline": True},
    ]
    if sig.get("expected_pct") is not None:
        sign = "+" if sig["expected_pct"] >= 0 else ""
        fields.append({"name": "Next-bar expected",
                       "value": f"{sign}{sig['expected_pct']:.2f}%",
                       "inline": True})
    fields.append({"name": "Interval", "value": str(sig["interval"]),
                   "inline": True})

    embed = {
        "title": f"{_arrow(direction)} · {sig['ticker']}",
        "description": "Regime change detected by the HMM Regime Radar.",
        "color": _color(direction),
        "fields": fields,
        "footer": {"text": "HMM Regime Radar · research only, not financial advice"},
    }
    if sig.get("generated_at"):
        embed["timestamp"] = sig["generated_at"]
    return embed


# ──────────────────────────────────────────────────────────────────────────
# Webhook delivery
# ──────────────────────────────────────────────────────────────────────────
def post_webhook(webhook_url: str, embeds: List[dict], *,
                 username: str = DEFAULT_USERNAME,
                 content: Optional[str] = None, timeout: float = 10.0) -> int:
    """
    POST one Discord message carrying up to 10 embeds. Returns the HTTP status
    code (Discord returns 204 on success). Raises urllib errors on transport
    failure - callers should wrap this so a delivery hiccup never sinks the run.
    """
    payload = {"username": username, "embeds": embeds[:10]}
    if content:
        payload["content"] = content
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


# ──────────────────────────────────────────────────────────────────────────
# Scanner LONG-verdict alerts (the "the website says LONG" signal)
# ──────────────────────────────────────────────────────────────────────────
def _fmt_price(x) -> str:
    try:
        x = float(x)
        if x != x:  # NaN
            return "—"
        return f"{x:,.4f}".rstrip("0").rstrip(".") if x < 1000 else f"{x:,.2f}"
    except (TypeError, ValueError):
        return "—"


def build_scanner_embed(row: dict, *, max_confirmations: int = 8,
                        timestamp: Optional[str] = None) -> dict:
    """Build a Discord embed from one scanner row whose verdict is LONG."""
    conf = float(row.get("confidence", 0.0) or 0.0)
    nconf = int(row.get("confirmations", 0) or 0)
    name = row.get("instrument") or row.get("symbol") or "?"
    embed = {
        "title": f"🟢 LONG · {name}",
        "description": ("The scanner is flashing **LONG** — bull regime with "
                        "entry confirmations met."),
        "color": COLOR_BULL,
        "fields": [
            {"name": "Verdict", "value": "**LONG**", "inline": True},
            {"name": "Confirmations", "value": f"{nconf} / {max_confirmations}",
             "inline": True},
            {"name": "Confidence", "value": f"{conf:.0f}%", "inline": True},
            {"name": "Regime", "value": str(row.get("regime", "BULLISH")),
             "inline": True},
            {"name": "Last price", "value": _fmt_price(row.get("last_price")),
             "inline": True},
        ],
        "footer": {"text": "HMM Regime Radar · research only, not financial advice"},
    }
    if timestamp:
        embed["timestamp"] = timestamp
    return embed


def notify_scanner_signals(rows: List[dict], *, webhook_url: Optional[str] = None,
                           state_path: str = "web/data/long_signal_state.json",
                           min_confidence_pct: Optional[float] = None,
                           username: Optional[str] = None,
                           timestamp: Optional[str] = None,
                           dry_run: bool = False) -> List[dict]:
    """
    Alert on the scanner's LONG verdict — i.e. exactly what the website shows.

    ``rows`` is the list of dicts produced by ``scanner.scan(...).to_dict('records')``.
    Fires once per transition INTO ``verdict == 'LONG'`` (optionally gated by a
    confidence floor, percent 0..100). When an instrument leaves LONG, its new
    verdict is recorded so the next LONG re-alerts.

    Returns the rows that were (or, in ``dry_run``, would be) sent.
    """
    webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if min_confidence_pct is None:
        min_confidence_pct = float(os.getenv("SIGNAL_MIN_CONFIDENCE_PCT", "0"))
    if username is None:
        username = os.getenv("SIGNAL_WEBHOOK_NAME", DEFAULT_USERNAME)

    prev = load_state(state_path)
    new_state = dict(prev)
    to_send: List[dict] = []

    for row in rows:
        key = str(row.get("instrument") or row.get("symbol") or "")
        if not key:
            continue
        verdict = str(row.get("verdict", "-")).upper()
        conf = float(row.get("confidence", 0.0) or 0.0)
        new_state[key] = verdict  # record current verdict for de-dup

        is_long = verdict == "LONG" and conf >= min_confidence_pct
        if is_long and prev.get(key) != "LONG":
            to_send.append(row)

    if dry_run:
        return to_send

    if to_send and webhook_url:
        embeds = [build_scanner_embed(r, timestamp=timestamp) for r in to_send]
        for i in range(0, len(embeds), 10):
            try:
                post_webhook(webhook_url, embeds[i:i + 10], username=username)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(f"[discord] delivery failed: {exc}")

    save_state(state_path, new_state)
    return to_send if webhook_url else []


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator (regime-label variant — kept for completeness)
# ──────────────────────────────────────────────────────────────────────────
def notify_from_bundle(bundle: dict, *, webhook_url: Optional[str] = None,
                       state_path: str = "web/data/signal_state.json",
                       min_confidence: Optional[float] = None,
                       directions: Optional[Tuple[str, ...]] = None,
                       alert_neutral: Optional[bool] = None,
                       username: Optional[str] = None,
                       dry_run: bool = False) -> List[dict]:
    """
    The single entry point the scheduled job calls.

    1. Find actionable signals in ``bundle``.
    2. Compare each against the last label we recorded for that combo.
    3. POST an embed for every combo whose label CHANGED (one message, batched).
    4. Persist the new labels so the same regime isn't re-announced next run.

    Returns the list of signal dicts that were (or, in ``dry_run``, would be)
    sent. A no-op returning ``[]`` when no webhook is configured.
    """
    webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if min_confidence is None:
        min_confidence = float(os.getenv("SIGNAL_MIN_CONFIDENCE",
                                          DEFAULT_MIN_CONFIDENCE))
    if directions is None:
        raw = os.getenv("SIGNAL_DIRECTIONS", ",".join(DEFAULT_DIRECTIONS))
        directions = tuple(d.strip().lower() for d in raw.split(",") if d.strip())
    if alert_neutral is None:
        alert_neutral = os.getenv("SIGNAL_ALERT_NEUTRAL", "0").strip() in ("1", "true", "yes")
    if username is None:
        username = os.getenv("SIGNAL_WEBHOOK_NAME", DEFAULT_USERNAME)

    signals = actionable_signals(
        bundle, min_confidence=min_confidence,
        directions=directions, alert_neutral=alert_neutral)

    prev = load_state(state_path)
    to_send: List[dict] = []
    new_state = dict(prev)

    # Record the current label for EVERY combo present (so a dip below the
    # confidence floor, or a swing through neutral, re-arms a later alert).
    for combo_id, p in (bundle.get("payloads", {}) or {}).items():
        cur_label = ((p or {}).get("current", {}) or {}).get("label", "")
        new_state[combo_id] = cur_label

    for combo_id, sig in signals.items():
        if prev.get(combo_id) != sig["label"]:
            to_send.append(sig)

    # dry-run is a pure PREVIEW: no posting, no state mutation.
    if dry_run:
        return to_send

    # Deliver (only if there is something to send AND a webhook is configured).
    if to_send and webhook_url:
        embeds = [build_embed(s) for s in to_send]
        # Discord caps 10 embeds per message; batch if needed.
        for i in range(0, len(embeds), 10):
            try:
                post_webhook(webhook_url, embeds[i:i + 10], username=username)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(f"[discord] delivery failed: {exc}")

    # Always persist the latest labels so a swing through neutral / a dip below
    # the confidence floor re-arms a later alert, and so the first run after a
    # webhook is added doesn't flood with every market's standing regime.
    save_state(state_path, new_state)
    return to_send if webhook_url else []
