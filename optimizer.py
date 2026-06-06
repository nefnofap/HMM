"""
optimizer.py
============

Parameter optimizer that searches RSI / ADX / leverage / confirmation settings
and ranks them by their WALK-FORWARD (out-of-sample) performance - not in-sample.

Why walk-forward, not in-sample?
--------------------------------
Optimising on in-sample results is how you build a curve-fit fantasy that dies
live. We score every candidate with ``walkforward_backtest`` so the winner is
the one that generalised to unseen data, not the one that memorised the past.

This is a grid search (slow but honest). Keep the grid small or the data window
modest, because each candidate re-fits the HMM many times.

> Research/education only. Not financial advice.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Dict, List, Optional

import pandas as pd

from backtest import StrategyProfile
from indicators import ConfirmationConfig
from metrics import summary as metric_summary
from walkforward import walkforward_backtest


# A small, sensible default search grid. Extend if you have time/compute.
DEFAULT_GRID = {
    "min_confirmations": [5, 6, 7],
    "leverage": [1.0, 2.0, 2.5],
    "adx_min": [15.0, 20.0, 25.0],
    "rsi_max": [80.0, 90.0],
}


def _build_profile(base: StrategyProfile, combo: Dict[str, float]) -> StrategyProfile:
    """Create a StrategyProfile variant from a parameter combination."""
    cfg = base.confirmation_cfg or ConfirmationConfig()
    cfg = replace(cfg, adx_min=combo.get("adx_min", cfg.adx_min),
                  rsi_max=combo.get("rsi_max", cfg.rsi_max))
    return replace(base,
                   min_confirmations=int(combo.get("min_confirmations",
                                                   base.min_confirmations)),
                   leverage=float(combo.get("leverage", base.leverage)),
                   confirmation_cfg=cfg)


def optimize(df: pd.DataFrame, features: pd.DataFrame, periods_per_year: float,
             grid: Optional[Dict[str, list]] = None, n_states: int = 7,
             train_size: int = 2000, test_size: int = 300,
             objective: str = "sharpe", base_profile: Optional[StrategyProfile] = None,
             progress_cb=None) -> pd.DataFrame:
    """
    Grid-search parameters, scoring each by walk-forward metrics.

    Parameters
    ----------
    df, features : aligned OHLCV + feature frames.
    periods_per_year : for annualised metrics.
    grid : dict of param -> list of values (defaults to DEFAULT_GRID).
    objective : metric to rank by ('sharpe', 'calmar', 'total_return').
    progress_cb : optional callable(done, total, label).

    Returns
    -------
    DataFrame of every candidate's params + walk-forward metrics, sorted best
    objective first.
    """
    grid = grid or DEFAULT_GRID
    base_profile = base_profile or StrategyProfile.conservative()
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)

    results: List[Dict[str, float]] = []
    for i, values in enumerate(combos, start=1):
        combo = dict(zip(keys, values))
        profile = _build_profile(base_profile, combo)
        label = ", ".join(f"{k}={v}" for k, v in combo.items())
        try:
            wf = walkforward_backtest(df, features, n_states=n_states,
                                      train_size=train_size, test_size=test_size,
                                      profile=profile)
            m = metric_summary(wf["strategy_returns"], periods_per_year,
                               benchmark_returns=wf["benchmark_returns"])
            row = {**combo,
                   "sharpe": m["sharpe"], "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"], "calmar": m["calmar"],
                   "win_rate": m["win_rate"], "n_trades": len(wf["trades"]),
                   "n_folds": wf["n_folds"]}
        except Exception as exc:  # noqa: BLE001 - skip bad combos
            row = {**combo, "sharpe": float("nan"), "total_return": float("nan"),
                   "max_drawdown": float("nan"), "calmar": float("nan"),
                   "win_rate": float("nan"), "n_trades": 0, "n_folds": 0,
                   "error": str(exc)}
        results.append(row)
        if progress_cb:
            progress_cb(i, total, label)

    out = pd.DataFrame(results)
    if out.empty:
        return out
    obj = objective if objective in out.columns else "sharpe"
    return out.sort_values(obj, ascending=False).reset_index(drop=True)
