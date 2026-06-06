"""
metrics.py
==========

Performance and risk metrics for evaluating a regime-based strategy.

These are the "institutional" numbers you should look at before trusting any
strategy: not just total return, but RISK-ADJUSTED return and worst-case pain.

All functions take a pandas Series of PERIODIC RETURNS (e.g. per-bar simple
returns, not log returns) indexed by time, unless stated otherwise.

`periods_per_year` lets the annualised metrics work for any timeframe:
    * hourly crypto (24/7)      -> 24 * 365 = 8760
    * hourly equities (~6.5h/d) -> ~1638
    * daily crypto              -> 365
    * daily equities            -> 252
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative equity (growth of `initial`) from periodic simple returns."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the whole period (e.g. 0.5 = +50%)."""
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: float) -> float:
    """Compound Annual Growth Rate."""
    n = returns.shape[0]
    if n == 0:
        return 0.0
    growth = (1.0 + returns.fillna(0.0)).prod()
    if growth <= 0:
        return -1.0
    years = n / periods_per_year
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else 0.0


def annualised_volatility(returns: pd.Series, periods_per_year: float) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: float,
                 risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio = (mean excess return / volatility) * sqrt(periods).
    Higher is better; > 1 is decent, > 2 is good, > 3 is excellent (and rare).
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns.fillna(0.0) - rf_per_period
    sd = excess.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float,
                  risk_free_rate: float = 0.0) -> float:
    """
    Like Sharpe, but only penalises DOWNSIDE volatility (losses). A high Sortino
    with a lower Sharpe means most of your volatility is to the upside.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns.fillna(0.0) - rf_per_period
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown: the worst peak-to-trough drop in the equity curve,
    as a NEGATIVE fraction (e.g. -0.35 = the account fell 35% from a high).
    This is the single most important "can I stomach this?" risk number.
    """
    eq = equity_curve(returns)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown series (for plotting the 'underwater' curve)."""
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def calmar_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """CAGR divided by the absolute max drawdown. Return per unit of worst pain."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / mdd)


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades (or periods) that were profitable."""
    r = trade_returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(trade_returns: pd.Series) -> float:
    """Gross profit / gross loss. > 1 means profitable; higher is better."""
    r = trade_returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def alpha_vs_benchmark(strategy_returns: pd.Series,
                       benchmark_returns: pd.Series) -> float:
    """
    Simple excess total return of the strategy over buy-and-hold of the asset
    (e.g. 0.20 = strategy beat buy-and-hold by 20 percentage points overall).
    """
    return total_return(strategy_returns) - total_return(benchmark_returns)


def summary(returns: pd.Series, periods_per_year: float,
            benchmark_returns: pd.Series | None = None,
            risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Compute the full metric bundle in one call. Returns a plain dict that is
    easy to display in a table or serialise to JSON for the dashboard.
    """
    out: Dict[str, float] = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annualised_volatility": annualised_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(returns, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }
    if benchmark_returns is not None:
        out["alpha_vs_buy_hold"] = alpha_vs_benchmark(returns, benchmark_returns)
    return out
