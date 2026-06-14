#!/usr/bin/env python3
"""
export_regime.py
================

Run the HMM regime model across several assets and BOTH data sources, then emit
a single bundle (web/data/regimes.json) the dashboard switches between client-
side.

Sources:
  * spot  -> yfinance ("BTC-USD", free, works from CI)
  * perp  -> USDT perpetuals via ccxt public endpoints (free, no API key).
             Defaults to Bybit because Binance returns HTTP 451 from cloud IPs
             (GitHub Actions runners), so a Binance pull would fail in CI.

Each (asset, source) combination is fit independently (regimes differ per
market). If a perp pull fails for one asset, that combo is skipped and the rest
still ship — the run does not hard-fail.

Usage:
    python scripts/export_regime.py
    python scripts/export_regime.py --exchange okx --n-states 6
    python scripts/export_regime.py --assets BTC,ETH --sources spot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmm_predictor import HMMPredictor  # noqa: E402

OUT_PATH = os.path.join(ROOT, "web", "data", "regimes.json")

# asset key -> (display label, yfinance spot ticker, ccxt unified perp symbol)
ASSETS = {
    "BTC": ("Bitcoin", "BTC-USD", "BTC/USDT:USDT"),
    "ETH": ("Ethereum", "ETH-USD", "ETH/USDT:USDT"),
    "SOL": ("Solana", "SOL-USD", "SOL/USDT:USDT"),
    "BNB": ("BNB", "BNB-USD", "BNB/USDT:USDT"),
    "XRP": ("XRP", "XRP-USD", "XRP/USDT:USDT"),
}

RAMP = [
    (0.00, (0xF2, 0x6D, 0x5B)),
    (0.25, (0xE8, 0xA1, 0x3A)),
    (0.50, (0x8A, 0x93, 0xA6)),
    (0.75, (0x3F, 0xB6, 0xA8)),
    (1.00, (0x54, 0xC9, 0x8C)),
]
NAMES = ["Deep bear", "Bear", "Soft bear", "Neutral", "Soft bull", "Bull", "Strong bull"]
RIBBON_POINTS = 200


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


def bars_per_year(interval: str) -> float:
    table = {"1h": 24 * 365, "1d": 365, "30m": 48 * 365, "15m": 96 * 365}
    return float(table.get(interval, 24 * 365))


def downsample(seq, k):
    if len(seq) <= k:
        return list(seq)
    step = len(seq) / k
    return [seq[min(int(i * step), len(seq) - 1)] for i in range(k)]


# ── data sources ─────────────────────────────────────────────────────────

def fetch_spot(ticker: str, period: str, interval: str):
    import yfinance as yf

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no spot data for {ticker}")
    close = df["Close"].dropna()
    prices = np.asarray(close).reshape(-1).astype(float)
    ts = [t.tz_convert("UTC").isoformat() if t.tzinfo else t.isoformat()
          for t in close.index.to_pydatetime()]
    return ts, prices


def fetch_perp(exchange: str, symbol: str, interval: str, days: int):
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = ex.milliseconds() - days * 86400 * 1000
    rows: list = []
    max_bars = days * 24 + 50
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 1000 or len(rows) >= max_bars:
            break
    if not rows:
        raise RuntimeError(f"no perp data for {symbol} on {exchange}")
    seen = set()
    ts, prices = [], []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        ts.append(dt.datetime.fromtimestamp(r[0] / 1000, dt.timezone.utc).isoformat())
        prices.append(float(r[4]))  # close
    return ts, np.asarray(prices, dtype=float)


# ── regime computation for one (asset, source) series ────────────────────────

def compute_regime(ts_all, prices, *, source, exchange, ticker_label,
                   interval, n_states, horizon):
    if len(prices) < 80:
        raise RuntimeError("not enough history")

    log_returns = np.diff(np.log(prices)).reshape(-1, 1)
    ts = ts_all[1:]
    px = prices[1:]

    predictor = HMMPredictor(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        init_method="kmeans",
        random_state=42,
    )
    predictor.fit(log_returns)

    states = np.asarray(predictor.predict_hidden_states(log_returns)).reshape(-1)
    summary = predictor.api_summary(log_returns, n_steps=horizon)
    n = int(summary.get("n_components", predictor.model.n_components))

    state_probs = np.asarray(summary["state_probabilities"], dtype=float)
    current_dist = state_probs[-1].tolist()
    current_id = int(np.argmax(current_dist))

    bpy = bars_per_year(interval)
    flat = log_returns.reshape(-1)
    raw = {}
    for sid in range(n):
        grp = flat[states == sid]
        mean = float(grp.mean()) if grp.size else 0.0
        std = float(grp.std()) if grp.size else 0.0
        raw[sid] = {
            "mean": mean,
            "annReturnPct": round(mean * bpy * 100.0, 1),
            "annVolPct": round(std * math.sqrt(bpy) * 100.0, 1),
            "sharePct": round(100.0 * grp.size / states.size, 1),
        }

    order = sorted(range(n), key=lambda s: raw[s]["mean"])
    rank_of = {sid: r for r, sid in enumerate(order)}
    state_meta = []
    for sid in range(n):
        r = rank_of[sid]
        frac = r / (n - 1) if n > 1 else 0.5
        state_meta.append({
            "id": sid, "rank": r, "label": label_for(r, n), "hue": hue_for(frac),
            "meanLogReturn": round(raw[sid]["mean"], 6),
            "annReturnPct": raw[sid]["annReturnPct"],
            "annVolPct": raw[sid]["annVolPct"],
            "sharePct": raw[sid]["sharePct"],
        })
    label_of = {m["id"]: m["label"] for m in state_meta}

    steps = []
    for f in summary["forecast"]["forecast"]:
        exp_log = float(f["expected_observation"][0])
        steps.append({
            "step": int(f["step"]),
            "expectedLogReturn": round(exp_log, 6),
            "expectedPct": round((math.exp(exp_log) - 1.0) * 100.0, 3),
            "distribution": [round(float(x), 3) for x in f["state_distribution"]],
        })

    transmat = np.asarray(predictor.model.transmat_, dtype=float)
    keep = downsample(list(range(len(states))), RIBBON_POINTS)

    return {
        "schemaVersion": "1.0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "ticker": ticker_label,
        "source": source,                         # "spot" | "perp"
        "exchange": exchange if source == "perp" else "Yahoo Finance",
        "interval": interval,
        "nComponents": n,
        "refit": True,
        "timestamps": [ts[i] for i in keep],
        "prices": [round(float(px[i]), 2) for i in keep],
        "states": [int(states[i]) for i in keep],
        "current": {
            "stateId": current_id,
            "label": label_of[current_id],
            "confidence": round(float(current_dist[current_id]), 3),
            "distribution": [round(float(x), 3) for x in current_dist],
        },
        "forecast": {"horizon": horizon, "steps": steps},
        "stateMeta": state_meta,
        "transitionMatrix": [[round(float(x), 3) for x in row] for row in transmat],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=",".join(ASSETS.keys()),
                    help="comma-separated asset keys, e.g. BTC,ETH,SOL")
    ap.add_argument("--sources", default="spot,perp", help="spot, perp, or both")
    ap.add_argument("--exchange", default="bybit", help="ccxt exchange id for perps (bybit, okx, ...)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--spot-period", default="730d")
    ap.add_argument("--perp-days", type=int, default=365)
    ap.add_argument("--n-states", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()

    asset_keys = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    payloads: dict = {}
    assets_meta: list = []
    combos: list = []

    for key in asset_keys:
        if key not in ASSETS:
            print(f"[skip] unknown asset {key}")
            continue
        label, spot_ticker, perp_symbol = ASSETS[key]
        assets_meta.append({"key": key, "label": label})

        for source in sources:
            combo_id = f"{key}:{source}"
            try:
                if source == "spot":
                    ts, px = fetch_spot(spot_ticker, args.spot_period, args.interval)
                    tick = f"{key} spot"
                    exch = "Yahoo Finance"
                elif source == "perp":
                    ts, px = fetch_perp(args.exchange, perp_symbol, args.interval, args.perp_days)
                    tick = f"{key} perp"
                    exch = args.exchange
                else:
                    print(f"[skip] unknown source {source}")
                    continue

                payloads[combo_id] = compute_regime(
                    ts, px, source=source, exchange=exch, ticker_label=tick,
                    interval=args.interval, n_states=args.n_states, horizon=args.horizon,
                )
                combos.append({"id": combo_id, "asset": key, "source": source})
                cur = payloads[combo_id]["current"]
                print(f"[ok] {combo_id}: {cur['label']} ({cur['confidence']*100:.0f}%)")
            except Exception as exc:  # noqa: BLE001 — one combo failing must not sink the run
                print(f"[fail] {combo_id}: {exc}")

    if not payloads:
        sys.exit("No regimes computed for any asset/source. Aborting.")

    have_assets = {c["asset"] for c in combos}
    assets_meta = [a for a in assets_meta if a["key"] in have_assets]

    bundle = {
        "schemaVersion": "1.0",
        "generatedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "exchange": args.exchange,
        "interval": args.interval,
        "assets": assets_meta,
        "combos": combos,
        "payloads": payloads,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"Wrote {OUT_PATH}: {len(payloads)} combos across {len(assets_meta)} assets.")


if __name__ == "__main__":
    main()
