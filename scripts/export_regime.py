#!/usr/bin/env python3
"""
export_regime.py
================

Run the HMM regime model and emit web/data/regime.json for the dashboard.

By default this LOADS the saved model (btc_regime_model.json) and decodes the
latest prices — fast, deterministic, the "lightweight backend" path from the
README. Pass --refit to re-fit on the rolling window instead (slower, adapts to
new market structure).

Usage:
    python scripts/export_regime.py
    python scripts/export_regime.py --refit --n-states 7 --ticker BTC-USD
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np

# Make the repo-root modules importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmm_predictor import HMMPredictor  # noqa: E402

OUT_PATH = os.path.join(ROOT, "web", "data", "regime.json")
SAVED_MODEL = os.path.join(ROOT, "btc_regime_model.json")

# Diverging ramp anchors (bear -> bull) used to colour states by rank.
RAMP = [
    (0.00, (0xF2, 0x6D, 0x5B)),
    (0.25, (0xE8, 0xA1, 0x3A)),
    (0.50, (0x8A, 0x93, 0xA6)),
    (0.75, (0x3F, 0xB6, 0xA8)),
    (1.00, (0x54, 0xC9, 0x8C)),
]
NAMES = ["Deep bear", "Bear", "Soft bear", "Neutral", "Soft bull", "Bull", "Strong bull"]
RIBBON_POINTS = 240  # down-sample target for the tape


def hue_for(frac: float) -> str:
    frac = min(max(frac, 0.0), 1.0)
    for i in range(len(RAMP) - 1):
        x0, c0 = RAMP[i]
        x1, c1 = RAMP[i + 1]
        if x0 <= frac <= x1:
            t = 0 if x1 == x0 else (frac - x0) / (x1 - x0)
            r = round(c0[0] + (c1[0] - c0[0]) * t)
            g = round(c0[1] + (c1[1] - c0[1]) * t)
            b = round(c0[2] + (c1[2] - c0[2]) * t)
            return f"#{r:02X}{g:02X}{b:02X}"
    return "#8A93A6"


def label_for(rank: int, n: int) -> str:
    if n <= 1:
        return "Neutral"
    if n == 2:
        return ["Bear", "Bull"][rank]
    if n == 3:
        return ["Bear", "Neutral", "Bull"][rank]
    idx = round(rank * (len(NAMES) - 1) / (n - 1))
    return NAMES[idx]


def fetch_prices(ticker: str, period: str, interval: str):
    """Return (timestamps_iso, close_prices) or exit non-zero on failure."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed; `pip install -r requirements.txt`")

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        sys.exit(f"No price data returned for {ticker} ({period}/{interval}).")
    close = df["Close"].dropna()
    # yfinance can return a single-column frame; squeeze to 1-D.
    prices = np.asarray(close).reshape(-1).astype(float)
    index = close.index
    ts = [t.tz_convert("UTC").isoformat() if t.tzinfo else t.isoformat() for t in index.to_pydatetime()]
    return ts, prices


def bars_per_year(interval: str) -> float:
    table = {"1h": 24 * 365, "60m": 24 * 365, "1d": 365, "30m": 48 * 365, "15m": 96 * 365}
    return float(table.get(interval, 24 * 365))


def annualise(mean: float, std: float, bpy: float):
    # Arithmetic annualisation of log-return drift. Compounding an in-regime
    # conditional mean as if held all year (exp(mean*bpy)) explodes from noisy
    # short-sample means, so we report the more readable annualised drift.
    ann_ret = mean * bpy * 100.0
    ann_vol = std * math.sqrt(bpy) * 100.0
    return ann_ret, ann_vol


def downsample(seq, k):
    if len(seq) <= k:
        return list(seq)
    step = len(seq) / k
    return [seq[min(int(i * step), len(seq) - 1)] for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="BTC-USD")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--n-states", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--refit", action="store_true", help="re-fit instead of loading the saved model")
    args = ap.parse_args()

    ts_all, prices = fetch_prices(args.ticker, args.period, args.interval)
    if len(prices) < 50:
        sys.exit("Not enough price history to decode regimes.")

    log_returns = np.diff(np.log(prices)).reshape(-1, 1)
    # states/returns align to prices[1:] / ts_all[1:]
    ts = ts_all[1:]
    px = prices[1:]

    if args.refit or not os.path.exists(SAVED_MODEL):
        predictor = HMMPredictor(
            n_components=args.n_states,
            covariance_type="diag",
            n_iter=200,
            init_method="kmeans",
            random_state=42,
        )
        predictor.fit(log_returns)
        refit = True
    else:
        predictor = HMMPredictor.load(SAVED_MODEL)
        refit = False

    states = np.asarray(predictor.predict_hidden_states(log_returns)).reshape(-1)
    summary = predictor.api_summary(log_returns, n_steps=args.horizon)
    n = int(summary.get("n_components", predictor.model.n_components))

    # Posterior for "now" = last row of per-timestep state probabilities.
    state_probs = np.asarray(summary["state_probabilities"], dtype=float)
    current_dist = state_probs[-1].tolist()
    current_id = int(np.argmax(current_dist))

    # Empirical per-state stats from the decoded path.
    bpy = bars_per_year(args.interval)
    flat = log_returns.reshape(-1)
    raw_meta = {}
    for sid in range(n):
        mask = states == sid
        grp = flat[mask]
        if grp.size:
            mean = float(grp.mean())
            std = float(grp.std())
        else:
            mean, std = 0.0, 0.0
        ann_ret, ann_vol = annualise(mean, std, bpy)
        raw_meta[sid] = {
            "mean": mean,
            "annReturnPct": round(ann_ret, 1),
            "annVolPct": round(ann_vol, 1),
            "sharePct": round(100.0 * grp.size / states.size, 1),
        }

    # Rank states by mean return (ascending) -> labels + hues.
    order = sorted(range(n), key=lambda s: raw_meta[s]["mean"])
    rank_of = {sid: r for r, sid in enumerate(order)}
    state_meta = []
    for sid in range(n):
        r = rank_of[sid]
        frac = r / (n - 1) if n > 1 else 0.5
        state_meta.append({
            "id": sid,
            "rank": r,
            "label": label_for(r, n),
            "hue": hue_for(frac),
            "meanLogReturn": round(raw_meta[sid]["mean"], 6),
            "annReturnPct": raw_meta[sid]["annReturnPct"],
            "annVolPct": raw_meta[sid]["annVolPct"],
            "sharePct": raw_meta[sid]["sharePct"],
        })
    label_of = {m["id"]: m["label"] for m in state_meta}

    # Forecast -> compact per-step shape.
    steps = []
    for f in summary["forecast"]["forecast"]:
        exp_log = float(f["expected_observation"][0])
        steps.append({
            "step": int(f["step"]),
            "expectedLogReturn": round(exp_log, 6),
            "expectedPct": round((math.exp(exp_log) - 1.0) * 100.0, 3),
            "distribution": [round(float(x), 3) for x in f["state_distribution"]],
        })

    # Transition matrix.
    transmat = np.asarray(predictor.model.transmat_, dtype=float)

    # Down-sample the long series for the ribbon.
    idx = list(range(len(states)))
    keep = downsample(idx, RIBBON_POINTS)
    payload = {
        "schemaVersion": "1.0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "ticker": args.ticker,
        "interval": args.interval,
        "lookbackDays": int(args.period.rstrip("d")) if args.period.endswith("d") else 730,
        "nComponents": n,
        "refit": refit,
        "timestamps": [ts[i] for i in keep],
        "prices": [round(float(px[i]), 2) for i in keep],
        "states": [int(states[i]) for i in keep],
        "current": {
            "stateId": current_id,
            "label": label_of[current_id],
            "confidence": round(float(current_dist[current_id]), 3),
            "distribution": [round(float(x), 3) for x in current_dist],
        },
        "forecast": {"horizon": args.horizon, "steps": steps},
        "stateMeta": state_meta,
        "transitionMatrix": [[round(float(x), 3) for x in row] for row in transmat],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {OUT_PATH}: {label_of[current_id]} "
          f"({payload['current']['confidence']*100:.0f}% conf), "
          f"{n} states, {'re-fit' if refit else 'decoded'}.")


if __name__ == "__main__":
    main()
