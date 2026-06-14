"""
walkforward.py
==============

Walk-forward (rolling re-fit) backtesting - the single most important guard
against fooling yourself.

THE PROBLEM with a normal backtest
-----------------------------------
If you fit the HMM on the ENTIRE history and then "predict" regimes over that
same history, the model has effectively seen the future. The regime labels are
informed by data that, in live trading, would not exist yet. Backtests done
this way look fantastic and then fail in production. This is in-sample bias.

THE FIX: walk forward
----------------------
We slide a window through time:

    [ ---- train window ---- ][ test ]
              fit HMM            decode + trade, using ONLY the model
                                 trained on the earlier data
            then move the window forward by `test_size` and repeat.

At every step the model is trained only on data strictly BEFORE the bars it
trades. Stitching the test segments together gives an out-of-sample equity
curve that is a far more honest estimate of live performance.

This is slower (it re-fits many times), which is exactly why in production you
train OFFLINE on a schedule and serve the saved parameters - never fit per
request.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest import StrategyProfile, backtest_strategy
from hmm_predictor import ConvergenceError, HMMPredictor, SingularCovarianceError


def walkforward_backtest(
    df: pd.DataFrame,
    features: pd.DataFrame,
    n_states: int = 7,
    train_size: int = 2000,
    test_size: int = 250,
    profile: Optional[StrategyProfile] = None,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Run a rolling walk-forward backtest.

    Parameters
    ----------
    df : OHLCV frame aligned to `features`.
    features : feature frame (returns/range/volume_change).
    n_states : HMM regimes.
    train_size : bars used to fit the HMM in each fold.
    test_size : bars traded out-of-sample per fold before re-fitting.
    profile : StrategyProfile (conservative by default).
    covariance_type, random_state : passed to HMMPredictor.

    Returns
    -------
    dict with:
        strategy_returns, benchmark_returns : stitched out-of-sample Series
        states : out-of-sample decoded states (per test bar)
        n_folds : number of folds run
        trades : combined trade log across folds
    """
    profile = profile or StrategyProfile.conservative()
    n = len(features)
    if n < train_size + test_size:
        raise ValueError(
            f"Not enough data: have {n} bars, need at least "
            f"{train_size + test_size}. Reduce train_size/test_size or fetch "
            f"more history."
        )

    oos_strategy: list[pd.Series] = []
    oos_bench: list[pd.Series] = []
    oos_states: list[pd.Series] = []
    all_trades: list[dict] = []
    n_folds = 0

    start = 0
    while start + train_size + test_size <= n:
        train_feats = features.iloc[start:start + train_size]
        test_slice = slice(start + train_size, start + train_size + test_size)
        test_feats = features.iloc[test_slice]
        test_df = df.reindex(features.index).iloc[test_slice]

        try:
            model = HMMPredictor(n_components=n_states,
                                 covariance_type=covariance_type,
                                 n_iter=200, init_method="kmeans",
                                 random_state=random_state)
            model.fit(train_feats)
        except (ConvergenceError, SingularCovarianceError):
            # Skip this fold if the model can't fit cleanly; move on.
            start += test_size
            continue

        # Decode the bull/bear mapping on TRAIN data, then label TEST data
        # with the SAME model (out-of-sample decoding).
        train_states = model.predict_hidden_states(train_feats)
        test_states = model.predict_hidden_states(test_feats)

        # Determine bull/bear from the TRAIN window only (no future info).
        tmp = train_feats.copy()
        tmp["state"] = train_states
        mean_by_state = tmp.groupby("state")["returns"].mean()
        bull_state = int(mean_by_state.idxmax())
        bear_state = int(mean_by_state.idxmin())

        # Remap test states so backtest_strategy's internal bull/bear detection
        # agrees: we pass the test states but the helper recomputes bull/bear on
        # the test window. To keep the train-derived mapping authoritative, we
        # run a lightweight position loop here instead.
        fold = _trade_fold(test_df, test_feats, test_states,
                           bull_state, bear_state, profile)
        oos_strategy.append(fold["strategy_returns"])
        oos_bench.append(fold["benchmark_returns"])
        oos_states.append(pd.Series(test_states, index=test_feats.index))
        all_trades.extend(fold["trades"])
        n_folds += 1
        start += test_size

    if n_folds == 0:
        raise ValueError("No valid folds could be fit. Try more data or fewer states.")

    return {
        "strategy_returns": pd.concat(oos_strategy),
        "benchmark_returns": pd.concat(oos_bench),
        "states": pd.concat(oos_states),
        "n_folds": n_folds,
        "trades": all_trades,
    }


def _trade_fold(test_df, test_feats, test_states, bull_state, bear_state,
                profile: StrategyProfile) -> Dict[str, object]:
    """
    Trade a single out-of-sample fold using a train-derived bull/bear mapping.
    Mirrors backtest_strategy's logic but with fixed regime ids.
    """
    from indicators import confirmation_matrix

    cfg = profile.confirmation_cfg
    close = test_df["Close"]
    bar_ret = close.pct_change().fillna(0.0)
    conf = confirmation_matrix(test_df, cfg).reindex(test_feats.index)
    nconf = conf["n_confirmations"].fillna(0).astype(int).to_numpy()

    close_arr = close.to_numpy()
    idx = test_feats.index
    raw_pos = np.zeros(len(test_feats))
    trades: list[dict] = []

    in_market = False
    cooldown = 0
    entry_price = 0.0
    entry_i = -1
    peak = 0.0

    for i in range(len(test_feats)):
        s = test_states[i]
        price = close_arr[i]
        if in_market:
            peak = max(peak, price)
            reason = None
            if s == bear_state:
                reason = "regime_flip_bear"
            elif (profile.use_trailing_stop and peak > 0
                  and price <= peak * (1 - profile.trailing_stop_pct)):
                reason = "trailing_stop"
            if reason:
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "entry_price": float(entry_price), "exit_price": float(price),
                    "bars_held": i - entry_i,
                    "return_pct": float(price / entry_price - 1.0),
                    "leveraged_return_pct": float((price / entry_price - 1.0) * profile.leverage),
                    "exit_reason": reason,
                })
                in_market = False
                cooldown = profile.cooldown_bars
                raw_pos[i] = 0.0
                continue
            raw_pos[i] = profile.leverage
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if s == bull_state and nconf[i] >= profile.min_confirmations:
            in_market = True
            entry_price = price
            entry_i = i
            peak = price
            raw_pos[i] = profile.leverage

    position = pd.Series(raw_pos, index=idx)
    traded_pos = position.shift(1).fillna(0.0)
    pos_change = traded_pos.diff().abs().fillna(traded_pos.abs())
    costs = pos_change * profile.cost_per_trade
    return {
        "strategy_returns": traded_pos * bar_ret - costs,
        "benchmark_returns": bar_ret,
        "trades": trades,
    }
