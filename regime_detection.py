"""
regime_detection.py
===================

PHASE 1 - Core HMM logic validation (Colab-friendly).

Market regime detection on Bitcoin (or any Yahoo Finance ticker) using a
7-state Gaussian Hidden Markov Model, built on the validated `HMMPredictor`
engine in hmm_predictor.py.

Specification implemented
-------------------------
* Data source : Yahoo Finance, hourly bars for the past 730 days
                (yfinance caps intraday history at ~730d, so this is the max).
* Features    : 3 columns the model is trained on
                  1. returns       = log return of close
                  2. range         = (high - low) / close   (intrabar range)
                  3. volume_change = log change in volume
* Model       : Gaussian emissions, 7 regimes (states 0..6).
* Output      : (a) summary table of mean return & return volatility per state
                (b) scatter plot of close price colour-coded by detected regime
                    saved to regime_plot.png

Usage
-----
    python regime_detection.py --ticker BTC-USD
    python regime_detection.py --ticker ETH-USD --n-states 7

In Google Colab, run the cells equivalently:
    !pip install yfinance hmmlearn scikit-learn matplotlib pandas numpy
    from regime_detection import run
    run(ticker="BTC-USD")

Notes
-----
* If Yahoo Finance is unreachable (no network / rate limit), the script falls
  back to a synthetic regime-switching series so the pipeline still validates.
* Raw prices are converted to stationary features before fitting; the
  HMMPredictor standardises them internally to avoid numerical instability.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hmm_predictor import HMMPredictor


# ----------------------------------------------------------------------
# Multi-asset support: instrument catalog + aliases -> Yahoo Finance symbols
# ----------------------------------------------------------------------
# The dashboard reads INSTRUMENT_CATALOG to build category + instrument
# dropdowns. The CLI accepts short aliases (e.g. "BTC", "GOLD"). Add more
# instruments by editing INSTRUMENT_CATALOG below.
# INSTRUMENT CATALOG -- organised by category for the dashboard dropdown.
# Each entry maps a friendly display name -> Yahoo Finance symbol.
INSTRUMENT_CATALOG: Dict[str, Dict[str, str]] = {
    "Crypto": {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Solana (SOL)": "SOL-USD",
        "BNB": "BNB-USD",
        "XRP": "XRP-USD",
        "Cardano (ADA)": "ADA-USD",
        "Dogecoin (DOGE)": "DOGE-USD",
        "Avalanche (AVAX)": "AVAX-USD",
        "Polkadot (DOT)": "DOT-USD",
        "Chainlink (LINK)": "LINK-USD",
        "Litecoin (LTC)": "LTC-USD",
        "Polygon (MATIC)": "MATIC-USD",
    },
    "Metals": {
        "Gold (XAUUSD)": "GC=F",
        "Silver (XAGUSD)": "SI=F",
        "Platinum": "PL=F",
        "Copper": "HG=F",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
    },
    "Indices": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Nasdaq Composite": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "VIX (Volatility)": "^VIX",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
    },
    "Commodities": {
        "Crude Oil (WTI)": "CL=F",
        "Natural Gas": "NG=F",
        "Brent Crude": "BZ=F",
    },
    "Stocks": {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "Meta (META)": "META",
        "Alphabet (GOOGL)": "GOOGL",
        "MicroStrategy (MSTR)": "MSTR",
        "Coinbase (COIN)": "COIN",
    },
}

# Flat display-name -> symbol map for quick lookup.
DISPLAY_TO_SYMBOL: Dict[str, str] = {
    name: sym for group in INSTRUMENT_CATALOG.values() for name, sym in group.items()
}

# Short aliases (case-insensitive) for the CLI / text entry.
TICKER_ALIASES: Dict[str, str] = {
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "BNB": "BNB-USD", "XRP": "XRP-USD", "ADA": "ADA-USD",
    "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "DOT": "DOT-USD", "LINK": "LINK-USD",
    "LTC": "LTC-USD", "MATIC": "MATIC-USD",
    "XAU": "GC=F", "XAUUSD": "GC=F", "GOLD": "GC=F",
    "XAG": "SI=F", "XAGUSD": "SI=F", "SILVER": "SI=F",
    "PLATINUM": "PL=F", "COPPER": "HG=F",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "USDCHF": "CHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
    "SPX": "^GSPC", "SP500": "^GSPC", "NDX": "^NDX", "NASDAQ": "^IXIC",
    "DOW": "^DJI", "RUSSELL": "^RUT", "VIX": "^VIX",
    "FTSE": "^FTSE", "NIKKEI": "^N225",
    "OIL": "CL=F", "WTI": "CL=F", "NATGAS": "NG=F", "BRENT": "BZ=F",
}


def resolve_ticker(ticker: str) -> str:
    """Map a friendly alias or display name (case-insensitive) to a symbol."""
    t = ticker.strip()
    if t in DISPLAY_TO_SYMBOL:          # exact dropdown display name
        return DISPLAY_TO_SYMBOL[t]
    return TICKER_ALIASES.get(t.upper(), t)


def is_crypto_symbol(symbol: str) -> bool:
    """True if the resolved symbol is a 24/7 crypto pair."""
    return symbol.upper().endswith("-USD")


def periods_per_year_for(interval: str, ticker: str) -> float:
    """
    Bars per year for annualising metrics. Crypto trades 24/7; gold/equity
    futures and indices trade fewer hours, so we approximate accordingly.
    """
    resolved = resolve_ticker(ticker)
    is_crypto = is_crypto_symbol(resolved)
    if interval == "1h":
        return 24 * 365 if is_crypto else 6.5 * 252  # ~1638 for non-crypto
    if interval == "1d":
        return 365 if is_crypto else 252
    if interval == "1wk":
        return 52
    # default to daily-crypto assumption
    return 365


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_prices(ticker: str = "BTC-USD", period: str = "730d",
                interval: str = "1h") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance. Accepts friendly aliases
    (e.g. "GOLD", "XAUUSD", "BTC"). Falls back to synthetic data if yfinance
    is unavailable or returns nothing.
    """
    resolved = resolve_ticker(ticker)
    try:
        import yfinance as yf

        df = yf.download(
            resolved, period=period, interval=interval,
            auto_adjust=True, progress=False,
        )
        if df is None or df.empty:
            raise RuntimeError("empty response")
        # Flatten possible MultiIndex columns (yfinance >= 0.2 returns these).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        print(f"Downloaded {len(df)} {interval} bars for {ticker} ({resolved}).")
        return df
    except Exception as exc:  # network/import/data issues -> synthetic fallback
        print(f"[load_prices] yfinance unavailable for {ticker} ({exc}); "
              f"using synthetic data.")
        return _synthetic_ohlcv()


def _synthetic_ohlcv(n: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame with several volatility regimes."""
    rng = np.random.default_rng(seed)
    regimes = [(-0.0008, 0.012), (0.0001, 0.004), (0.0010, 0.008),
               (-0.0030, 0.025), (0.0003, 0.006)]
    trans = np.full((5, 5), 0.05)
    np.fill_diagonal(trans, 0.80)
    trans /= trans.sum(axis=1, keepdims=True)
    state, rets = 2, []
    for _ in range(n):
        mu, sig = regimes[state]
        rets.append(rng.normal(mu, sig))
        state = rng.choice(5, p=trans[state])
    close = 30000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.lognormal(mean=10, sigma=0.5, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# ----------------------------------------------------------------------
# Feature engineering (3 features)
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the three stationary features the HMM trains on.

    All are stationary / scale-free so the Gaussian HMM behaves well; the
    HMMPredictor additionally standardises them internally.
    """
    feats = pd.DataFrame(index=df.index)
    feats["returns"] = np.log(df["Close"]).diff()
    feats["range"] = (df["High"] - df["Low"]) / df["Close"]
    # log volume change, guarding against zero volume.
    vol = df["Volume"].replace(0, np.nan).ffill()
    feats["volume_change"] = np.log(vol).diff()
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def summarise_states(features: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """Per-regime summary: count, mean return, return volatility, avg range/volume."""
    tbl = features.copy()
    tbl["state"] = states
    summary = tbl.groupby("state").agg(
        n_obs=("returns", "size"),
        mean_return=("returns", "mean"),
        return_volatility=("returns", "std"),
        mean_range=("range", "mean"),
        mean_volume_change=("volume_change", "mean"),
    )
    # Label the extremes the trading layer cares about.
    summary["label"] = ""
    summary.loc[summary["mean_return"].idxmax(), "label"] = "BULL_RUN (highest return)"
    summary.loc[summary["mean_return"].idxmin(), "label"] = "BEAR/CRASH (lowest return)"
    return summary.sort_values("mean_return")


def plot_regimes(df: pd.DataFrame, features: pd.DataFrame, states: np.ndarray,
                 ticker: str, out_path: str = "regime_plot.png") -> Optional[str]:
    """Scatter of close price colour-coded by detected regime; saved to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / server-safe backend
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot_regimes] matplotlib unavailable ({exc}); skipping plot.")
        return None

    close = df["Close"].reindex(features.index)
    fig, ax = plt.subplots(figsize=(15, 7))
    sc = ax.scatter(features.index, close, c=states, cmap="viridis", s=6)
    ax.set_title(f"{ticker} - Close price coloured by HMM regime")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close price")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Hidden state (regime)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Regime overlay plot saved to {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run(ticker: str = "BTC-USD", n_states: int = 7,
        period: str = "730d", interval: str = "1h") -> pd.DataFrame:
    """Full Phase-1 pipeline: load -> features -> fit 7-state HMM -> report."""
    df = load_prices(ticker, period=period, interval=interval)
    features = build_features(df)

    predictor = HMMPredictor(
        n_components=n_states,
        covariance_type="diag",   # robust with 3 features x 7 states
        n_iter=300,
        init_method="kmeans",
        random_state=42,
    )
    predictor.fit(features)
    states = predictor.predict_hidden_states(features)

    summary = summarise_states(features, states)
    print("\n=== Regime summary (sorted by mean return) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
        print(summary)

    plot_regimes(df, features, states, ticker)

    # Persist learned parameters for the web backend (Phase 2/3 reuse).
    safe = resolve_ticker(ticker).replace("=", "").replace("^", "").replace("-", "_")
    model_path = f"{safe.lower()}_regime_model.json"
    predictor.save(model_path)
    print(f"\nModel parameters saved to {model_path}")
    return summary


def run_multi(tickers: List[str], n_states: int = 7,
              period: str = "730d", interval: str = "1h") -> pd.DataFrame:
    """
    Fit a SEPARATE regime model per asset and return a side-by-side comparison
    of strategy metrics. Each asset gets its own HMM because regimes (and their
    volatility) differ across BTC, ETH and gold - one shared model would blur
    them. Uses the regime backtest from backtest.py.
    """
    from backtest import backtest_regime_strategy
    from metrics import summary as metric_summary

    rows = []
    for tk in tickers:
        try:
            df = load_prices(tk, period=period, interval=interval)
            features = build_features(df)
            model = HMMPredictor(n_components=n_states, covariance_type="diag",
                                 n_iter=300, init_method="kmeans", random_state=42)
            model.fit(features)
            states = model.predict_hidden_states(features)

            bt = backtest_regime_strategy(df, features, states)
            ppy = periods_per_year_for(interval, tk)
            strat = metric_summary(bt["strategy_returns"], ppy,
                                   benchmark_returns=bt["benchmark_returns"])
            strat["asset"] = tk.upper()
            rows.append(strat)
            print(f"  [{tk.upper()}] Sharpe={strat['sharpe']:.2f} "
                  f"MaxDD={strat['max_drawdown']:.1%} "
                  f"Return={strat['total_return']:.1%}")
        except Exception as exc:  # noqa: BLE001 - keep going across assets
            print(f"  [{tk.upper()}] skipped: {exc}")

    if not rows:
        return pd.DataFrame()
    cmp = pd.DataFrame(rows).set_index("asset")
    print("\n=== Multi-asset strategy comparison ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(cmp)
    return cmp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMM market regime detection (Phase 1).")
    p.add_argument("--ticker", default="BTC-USD",
                   help="Single asset, or use --tickers for several.")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated list, e.g. BTC,ETH,GOLD for multi-asset.")
    p.add_argument("--n-states", type=int, default=7)
    p.add_argument("--period", default="730d")
    p.add_argument("--interval", default="1h")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        run_multi(tickers, n_states=args.n_states,
                  period=args.period, interval=args.interval)
    else:
        run(ticker=args.ticker, n_states=args.n_states,
            period=args.period, interval=args.interval)
