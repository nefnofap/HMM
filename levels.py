"""
levels.py
=========

Turns the abstract "it's bullish" signal into CONCRETE numbers a trader can act
on: an entry price, a stop-loss, profit targets, and a position size.

Two pieces
----------
1. ATR-based levels (``compute_levels``)
   ATR = Average True Range = the average size of a bar's price movement. It is
   the standard, volatility-aware way to place stops/targets:
       * a volatile asset gets a WIDER stop (so normal noise doesn't stop you out)
       * a calm asset gets a TIGHTER stop
   We set:
       stop   = entry - stop_atr_mult * ATR
       target = entry + target_atr_mult * ATR     (risk:reward = target/stop mult)

2. Volatility-aware position sizing (``position_size``)
   Risk a FIXED fraction of the account per trade (e.g. 1%). The number of units
   is derived from the dollar risk and the per-unit stop distance, so every trade
   risks the same amount regardless of the asset's volatility. We also shrink the
   size further in high-volatility regimes.

> Research/education only. Not financial advice. Leverage magnifies losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


@dataclass
class LevelConfig:
    atr_window: int = 14
    stop_atr_mult: float = 2.0       # stop = entry - 2*ATR
    target1_atr_mult: float = 3.0    # first target  (1.5 R)
    target2_atr_mult: float = 5.0    # second target (2.5 R)


def compute_levels(df: pd.DataFrame, cfg: Optional[LevelConfig] = None
                   ) -> Dict[str, float]:
    """
    Compute entry/stop/target levels from the latest bar.

    Returns a dict with entry, stop, target1, target2, atr, risk_per_unit,
    and the risk:reward ratios. Entry is the latest close.
    """
    cfg = cfg or LevelConfig()
    a = float(atr(df, cfg.atr_window).iloc[-1])
    entry = float(df["Close"].iloc[-1])
    stop = entry - cfg.stop_atr_mult * a
    target1 = entry + cfg.target1_atr_mult * a
    target2 = entry + cfg.target2_atr_mult * a
    risk = entry - stop
    return {
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "atr": a,
        "risk_per_unit": risk,
        "rr_target1": (target1 - entry) / risk if risk > 0 else 0.0,
        "rr_target2": (target2 - entry) / risk if risk > 0 else 0.0,
        "stop_pct": (stop / entry - 1.0) if entry > 0 else 0.0,
        "target1_pct": (target1 / entry - 1.0) if entry > 0 else 0.0,
        "target2_pct": (target2 / entry - 1.0) if entry > 0 else 0.0,
    }


def position_size(account_equity: float, levels: Dict[str, float],
                  risk_pct: float = 0.01, max_leverage: float = 2.5,
                  vol_scalar: float = 1.0) -> Dict[str, float]:
    """
    Volatility-aware position size.

    Parameters
    ----------
    account_equity : total account value in quote currency (e.g. USD).
    levels : output of ``compute_levels`` (needs entry + risk_per_unit).
    risk_pct : fraction of equity to risk if the stop is hit (e.g. 0.01 = 1%).
    max_leverage : cap on notional / equity.
    vol_scalar : 0-1 multiplier to shrink size in high-vol regimes
                 (e.g. 0.5 halves the size). Compute it from the regime's
                 volatility relative to the calmest regime.

    Returns
    -------
    dict: units, notional, leverage_used, dollar_risk, capped (bool).
    """
    entry = levels["entry"]
    risk_per_unit = levels["risk_per_unit"]
    if entry <= 0 or risk_per_unit <= 0:
        return {"units": 0.0, "notional": 0.0, "leverage_used": 0.0,
                "dollar_risk": 0.0, "capped": False}

    dollar_risk = account_equity * risk_pct * max(min(vol_scalar, 1.0), 0.0)
    units = dollar_risk / risk_per_unit
    notional = units * entry

    # Cap by max leverage.
    max_notional = account_equity * max_leverage
    capped = False
    if notional > max_notional:
        notional = max_notional
        units = notional / entry
        capped = True

    return {
        "units": units,
        "notional": notional,
        "leverage_used": notional / account_equity if account_equity > 0 else 0.0,
        "dollar_risk": min(dollar_risk, units * risk_per_unit),
        "capped": capped,
    }


def vol_scalar_from_regimes(summary: pd.DataFrame, current_state: int) -> float:
    """
    Derive a 0-1 size multiplier: 1.0 in the calmest regime, smaller in more
    volatile regimes. Uses return_volatility from summarise_states().
    """
    if "return_volatility" not in summary.columns or current_state not in summary.index:
        return 1.0
    vols = summary["return_volatility"]
    lo, hi = float(vols.min()), float(vols.max())
    if hi <= lo:
        return 1.0
    cur = float(vols.loc[current_state])
    # Linear: calmest -> 1.0, most volatile -> 0.4 (never fully zero).
    frac = (cur - lo) / (hi - lo)
    return float(1.0 - 0.6 * frac)
