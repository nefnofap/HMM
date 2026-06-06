"""
backtest.py
===========

A simple, HONEST regime-based backtester for the HMM strategy.

Strategy logic (intentionally minimal - this is a research baseline, not a
money printer):

    * Identify the BULL regime  = state with the highest mean return.
    * Identify the BEAR/CRASH regime = state with the lowest mean return.
    * Go LONG (optionally leveraged) when the current regime is bullish.
    * Exit to cash immediately when the regime flips to bear/crash.
    * Apply a cooldown after each exit before re-entry is allowed.
    * Apply a per-trade transaction cost.

IMPORTANT honesty notes (read these):
    * We shift signals by one bar so we trade on the NEXT bar's open-ish price,
      never on the same bar the regime was detected. Acting on the same bar is
      lookahead bias and inflates results.
    * The HMM here is fit on the WHOLE history (in-sample). For a truly fair
      test you must re-fit on a rolling/walk-forward window. This baseline is
      for wiring up metrics and comparing assets, not for live deployment.
    * Past performance does not predict future results. This is for research
      and learning only - not financial advice.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def backtest_regime_strategy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    states: np.ndarray,
    leverage: float = 1.0,
    cost_per_trade: float = 0.0004,   # 4 bps round-trip-ish per state change
    cooldown_bars: int = 0,
) -> Dict[str, object]:
    """
    Run the regime-long/cash strategy and return returns + position series.

    Parameters
    ----------
    df : OHLCV frame (must contain 'Close'), indexed like `features`.
    features : feature frame with a 'returns' (log return) column.
    states : decoded hidden state per row of `features`.
    leverage : position multiplier when long (e.g. 2.5 or aggressive 4.0).
    cost_per_trade : fractional cost applied whenever the position changes.
    cooldown_bars : bars to stay in cash after an exit before re-entering.

    Returns
    -------
    dict with:
        strategy_returns : pd.Series of per-bar simple returns (after costs).
        benchmark_returns: pd.Series of buy-and-hold simple returns.
        position         : pd.Series (0 = cash, `leverage` = long).
        bull_state, bear_state : identified regime ids.
    """
    feats = features.copy()
    feats["state"] = states

    # Per-state mean return -> identify bull (max) and bear (min).
    mean_by_state = feats.groupby("state")["returns"].mean()
    bull_state = int(mean_by_state.idxmax())
    bear_state = int(mean_by_state.idxmin())

    # Simple (not log) per-bar returns of the asset.
    close = df["Close"].reindex(feats.index)
    bar_ret = close.pct_change().fillna(0.0)

    # Build raw signal: long while bullish, flat while bearish, hold otherwise.
    # We carry the previous position forward when the regime is neither extreme.
    raw_pos = np.zeros(len(feats), dtype=float)
    in_market = False
    cooldown = 0
    for i, s in enumerate(feats["state"].to_numpy()):
        if cooldown > 0:
            cooldown -= 1
            in_market = False
        elif s == bull_state:
            in_market = True
        elif s == bear_state:
            if in_market:
                cooldown = cooldown_bars  # just exited -> start cooldown
            in_market = False
        # else: neutral regime -> keep current in_market state
        raw_pos[i] = leverage if in_market else 0.0

    position = pd.Series(raw_pos, index=feats.index)

    # Trade on the NEXT bar (shift) to avoid lookahead bias.
    traded_pos = position.shift(1).fillna(0.0)

    # Transaction cost whenever the position changes.
    pos_change = traded_pos.diff().abs().fillna(traded_pos.abs())
    costs = pos_change * cost_per_trade

    strategy_returns = traded_pos * bar_ret - costs
    benchmark_returns = bar_ret  # buy-and-hold

    return {
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "position": traded_pos,
        "bull_state": bull_state,
        "bear_state": bear_state,
    }
