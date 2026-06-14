"""
backtest.py
===========

Regime-based backtester for the HMM strategy, with technical confirmations,
risk management, an aggressive variant, and a per-trade log.

Two entry points
----------------
1. ``backtest_regime_strategy`` - the simple regime-only baseline (long in the
   bull regime, cash otherwise). Kept for quick comparisons.
2. ``backtest_strategy`` - the full strategy from the spec:
       * enter only when regime is BULLISH **and** >= min_confirmations of the
         8 technical signals are true (7-of-8 conservative, 5-of-8 aggressive);
       * exit immediately when the regime flips to BEAR/CRASH;
       * exit on a trailing stop (aggressive mode);
       * enforce a cooldown (e.g. 48 hours) after any exit before re-entry;
       * apply leverage and per-trade transaction costs;
       * produce a trade log (entry/exit time, price, P&L, bars held, reason).

HONESTY NOTES (read these)
--------------------------
* Signals are computed on bar t but ACTED ON at bar t+1 (we shift positions by
  one bar) to avoid lookahead bias.
* ``backtest_strategy`` fits the HMM on the whole window (in-sample). For an
  out-of-sample test use ``walkforward.walkforward_backtest`` which re-fits on a
  rolling window.
* This is a research/simulation tool. Past performance does not predict the
  future. Not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from indicators import ConfirmationConfig, confirmation_matrix


# ======================================================================
# Strategy profiles (conservative vs aggressive)
# ======================================================================
@dataclass
class StrategyProfile:
    """All tunable strategy parameters in one place."""
    min_confirmations: int = 7        # of 8 (use 5 for aggressive)
    leverage: float = 2.5             # base leverage (use 4.0 for aggressive)
    cooldown_bars: int = 48           # 48 hourly bars = 48h cooldown
    cost_per_trade: float = 0.0004    # fractional cost per position change
    use_trailing_stop: bool = False   # aggressive mode turns this on
    trailing_stop_pct: float = 0.08   # exit if price falls 8% from peak-in-trade
    confirmation_cfg: ConfirmationConfig = None  # type: ignore

    @classmethod
    def conservative(cls) -> "StrategyProfile":
        return cls(min_confirmations=7, leverage=2.5, cooldown_bars=48,
                   use_trailing_stop=False,
                   confirmation_cfg=ConfirmationConfig())

    @classmethod
    def aggressive(cls) -> "StrategyProfile":
        """4x leverage, 5-of-8 confirmations, trailing stop (per the spec)."""
        return cls(min_confirmations=5, leverage=4.0, cooldown_bars=48,
                   use_trailing_stop=True, trailing_stop_pct=0.08,
                   confirmation_cfg=ConfirmationConfig())


# ======================================================================
# Simple regime-only baseline (kept for comparison)
# ======================================================================
def backtest_regime_strategy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    states: np.ndarray,
    leverage: float = 1.0,
    cost_per_trade: float = 0.0004,
    cooldown_bars: int = 0,
) -> Dict[str, object]:
    """Regime-long/cash baseline (no technical confirmations). See module doc."""
    feats = features.copy()
    feats["state"] = states
    mean_by_state = feats.groupby("state")["returns"].mean()
    bull_state = int(mean_by_state.idxmax())
    bear_state = int(mean_by_state.idxmin())

    close = df["Close"].reindex(feats.index)
    bar_ret = close.pct_change().fillna(0.0)

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
                cooldown = cooldown_bars
            in_market = False
        raw_pos[i] = leverage if in_market else 0.0

    position = pd.Series(raw_pos, index=feats.index)
    traded_pos = position.shift(1).fillna(0.0)
    pos_change = traded_pos.diff().abs().fillna(traded_pos.abs())
    costs = pos_change * cost_per_trade
    strategy_returns = traded_pos * bar_ret - costs

    return {
        "strategy_returns": strategy_returns,
        "benchmark_returns": bar_ret,
        "position": traded_pos,
        "bull_state": bull_state,
        "bear_state": bear_state,
    }


# ======================================================================
# Full strategy: regime + 7-of-8 confirmations + risk management
# ======================================================================
def backtest_strategy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    states: np.ndarray,
    profile: Optional[StrategyProfile] = None,
) -> Dict[str, object]:
    """
    Full regime + technical-confirmation strategy with risk management and a
    trade log.

    Parameters
    ----------
    df : OHLCV frame (Open/High/Low/Close/Volume), indexed like `features`.
    features : feature frame (must include 'returns'), indexed like `df`.
    states : decoded HMM state per row of `features`.
    profile : StrategyProfile (defaults to conservative).

    Returns
    -------
    dict with:
        strategy_returns, benchmark_returns : pd.Series of per-bar simple returns
        position : pd.Series of the leverage actually applied each bar (shifted)
        trades   : list of dicts (the trade log)
        bull_state, bear_state : identified regime ids
        n_confirmations : pd.Series of confirmation counts per bar
        profile : the StrategyProfile used
    """
    profile = profile or StrategyProfile.conservative()
    cfg = profile.confirmation_cfg or ConfirmationConfig()

    feats = features.copy()
    feats["state"] = states
    mean_by_state = feats.groupby("state")["returns"].mean()
    bull_state = int(mean_by_state.idxmax())
    bear_state = int(mean_by_state.idxmin())

    # Align price + confirmations to the feature index.
    df_aligned = df.reindex(feats.index)
    close = df_aligned["Close"]
    bar_ret = close.pct_change().fillna(0.0)
    conf = confirmation_matrix(df_aligned, cfg).reindex(feats.index)
    n_conf = conf["n_confirmations"].fillna(0).astype(int)

    state_arr = feats["state"].to_numpy()
    close_arr = close.to_numpy()
    nconf_arr = n_conf.to_numpy()
    idx = feats.index

    raw_pos = np.zeros(len(feats), dtype=float)
    trades: List[Dict[str, object]] = []

    in_market = False
    cooldown = 0
    entry_price = 0.0
    entry_i = -1
    peak_price = 0.0

    for i in range(len(feats)):
        s = state_arr[i]
        price = close_arr[i]

        if in_market:
            peak_price = max(peak_price, price)
            exit_reason = None

            if s == bear_state:
                exit_reason = "regime_flip_bear"
            elif (profile.use_trailing_stop and peak_price > 0
                  and price <= peak_price * (1.0 - profile.trailing_stop_pct)):
                exit_reason = "trailing_stop"

            if exit_reason:
                pnl = (price / entry_price - 1.0) * profile.leverage
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "entry_price": float(entry_price), "exit_price": float(price),
                    "bars_held": i - entry_i,
                    "return_pct": float(price / entry_price - 1.0),
                    "leveraged_return_pct": float(pnl),
                    "exit_reason": exit_reason,
                })
                in_market = False
                cooldown = profile.cooldown_bars
                raw_pos[i] = 0.0
                continue
            raw_pos[i] = profile.leverage
            continue

        # Not in market.
        if cooldown > 0:
            cooldown -= 1
            raw_pos[i] = 0.0
            continue

        # Entry condition: bullish regime AND enough confirmations.
        if s == bull_state and nconf_arr[i] >= profile.min_confirmations:
            in_market = True
            entry_price = price
            entry_i = i
            peak_price = price
            raw_pos[i] = profile.leverage
        else:
            raw_pos[i] = 0.0

    # Close any open trade at the final bar (mark-to-market).
    if in_market and entry_i >= 0:
        price = close_arr[-1]
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[-1],
            "entry_price": float(entry_price), "exit_price": float(price),
            "bars_held": (len(feats) - 1) - entry_i,
            "return_pct": float(price / entry_price - 1.0),
            "leveraged_return_pct": float((price / entry_price - 1.0) * profile.leverage),
            "exit_reason": "end_of_data",
        })

    position = pd.Series(raw_pos, index=idx)
    traded_pos = position.shift(1).fillna(0.0)  # act next bar (no lookahead)
    pos_change = traded_pos.diff().abs().fillna(traded_pos.abs())
    costs = pos_change * profile.cost_per_trade
    strategy_returns = traded_pos * bar_ret - costs

    return {
        "strategy_returns": strategy_returns,
        "benchmark_returns": bar_ret,
        "position": traded_pos,
        "trades": trades,
        "bull_state": bull_state,
        "bear_state": bear_state,
        "n_confirmations": n_conf,
        "profile": profile,
    }


def trades_to_frame(trades: List[Dict[str, object]]) -> pd.DataFrame:
    """Convert the trade log to a tidy DataFrame for display/export."""
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "bars_held", "return_pct", "leveraged_return_pct", "exit_reason",
        ])
    return pd.DataFrame(trades)
