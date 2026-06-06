"""
scanner.py
==========

Multi-asset scanner: run the regime + confirmation pipeline across many
instruments at once and rank them by their CURRENT signal, so you can see at a
glance which markets are flashing LONG right now.

This is the "command-center radar" feature - one screen, every market.

> Research/education only. Not financial advice.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import StrategyProfile
from hmm_predictor import ConvergenceError, HMMPredictor, SingularCovarianceError
from indicators import ConfirmationConfig, confirmation_matrix
from regime_detection import (DISPLAY_TO_SYMBOL, build_features, load_prices,
                              summarise_states)


def scan_instrument(display_name: str, symbol: str, n_states: int = 7,
                    period: str = "365d", interval: str = "1h",
                    profile: Optional[StrategyProfile] = None) -> Dict[str, object]:
    """
    Fit the HMM on one instrument and return its current signal snapshot.
    Returns a dict (one row of the scanner table). On failure returns a row
    with status set to the error reason.
    """
    profile = profile or StrategyProfile.conservative()
    cfg = profile.confirmation_cfg or ConfirmationConfig()
    base = {"instrument": display_name, "symbol": symbol}
    try:
        df = load_prices(symbol, period=period, interval=interval)
        feats = build_features(df)
        if len(feats) < 200:
            return {**base, "status": "insufficient data", "verdict": "-",
                    "regime": "-", "confirmations": 0, "confidence": 0.0,
                    "last_price": float("nan")}
        model = HMMPredictor(n_components=n_states, covariance_type="diag",
                             n_iter=200, init_method="kmeans", random_state=42)
        model.fit(feats)
        states = model.predict_hidden_states(feats)
        summary = summarise_states(feats, states)
        bull = int(summary["mean_return"].idxmax())
        bear = int(summary["mean_return"].idxmin())

        cur = int(states[-1])
        proba = model.predict_state_proba(feats)[-1]
        confidence = float(proba[cur]) * 100.0
        conf = confirmation_matrix(df.reindex(feats.index), cfg)
        n_conf = int(conf.iloc[-1]["n_confirmations"])
        last_price = float(df["Close"].reindex(feats.index).iloc[-1])

        if cur == bull and n_conf >= profile.min_confirmations:
            verdict, regime = "LONG", "BULLISH"
        elif cur == bull:
            verdict, regime = "WAIT", "BULLISH"
        elif cur == bear:
            verdict, regime = "STAND ASIDE", "BEARISH"
        else:
            verdict, regime = "WAIT", "NEUTRAL"

        # Sort key: LONG first, then by confirmations then confidence.
        rank = (0 if verdict == "LONG" else 1 if verdict == "WAIT" else 2)
        return {**base, "status": "ok", "verdict": verdict, "regime": regime,
                "confirmations": n_conf, "confidence": confidence,
                "last_price": last_price, "_rank": rank}
    except (ConvergenceError, SingularCovarianceError) as exc:
        return {**base, "status": f"model: {exc}", "verdict": "-",
                "regime": "-", "confirmations": 0, "confidence": 0.0,
                "last_price": float("nan"), "_rank": 9}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": f"error: {exc}", "verdict": "-",
                "regime": "-", "confirmations": 0, "confidence": 0.0,
                "last_price": float("nan"), "_rank": 9}


def scan(instruments: Optional[Dict[str, str]] = None, n_states: int = 7,
         period: str = "365d", interval: str = "1h",
         profile: Optional[StrategyProfile] = None,
         progress_cb=None) -> pd.DataFrame:
    """
    Scan a set of instruments (default: the full catalog) and return a ranked
    DataFrame: LONG signals first, then WAIT, then STAND ASIDE.

    progress_cb : optional callable(done, total, name) for UI progress bars.
    """
    instruments = instruments or DISPLAY_TO_SYMBOL
    rows: List[Dict[str, object]] = []
    total = len(instruments)
    for i, (name, sym) in enumerate(instruments.items(), start=1):
        rows.append(scan_instrument(name, sym, n_states, period, interval, profile))
        if progress_cb:
            progress_cb(i, total, name)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_cols = ["_rank", "confirmations", "confidence"]
    asc = [True, False, False]
    df = df.sort_values(sort_cols, ascending=asc).drop(columns=["_rank"])
    return df.reset_index(drop=True)
