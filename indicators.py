"""
indicators.py
=============

Technical-indicator entry confirmations for the regime strategy.

The HMM tells you WHICH regime the market is in (bull / bear / neutral). That
is a coarse, slow signal. To decide WHEN to actually enter inside a bull
regime, we layer a set of classic technical confirmations on top - the
"7-of-8" filter from the strategy spec.

Each confirmation returns a boolean Series (True = condition satisfied at that
bar). We then count how many are True per bar; entry requires at least
`min_confirmations` of them (e.g. 7 of 8 for the conservative profile, 5 of 8
for the aggressive profile).

Everything here is pure pandas/numpy - no extra TA dependency - so it installs
cleanly on Windows with no compiler.

The 8 confirmations
-------------------
    1. RSI not overbought      : RSI(14) < rsi_max  (default 90)
    2. RSI shows strength      : RSI(14) > rsi_min  (default 50)
    3. MACD bullish            : MACD line > signal line
    4. Trend up (price)        : close > SMA(slow)  (default 50)
    5. Fast over slow (MA xover): SMA(fast) > SMA(slow)
    6. Positive momentum       : ROC(momentum_window) > 0
    7. Trend strength (ADX)    : ADX(14) > adx_min  (default 20)
    8. Volume participation    : volume > its SMA(vol_window)

Tune the thresholds to "tighten" or "loosen" the entry, exactly as the
iterative-prompt workflow describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Individual indicator calculations
# ----------------------------------------------------------------------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing approximation)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)  # neutral when undefined


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Dict[str, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {"macd": macd_line, "signal": signal_line,
            "hist": macd_line - signal_line}


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of change (momentum), as a fraction over `window` bars."""
    return close.pct_change(window)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index - measures TREND STRENGTH (not direction).
    Values > 20-25 indicate a trending market worth trading directionally.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index)
                       .ewm(alpha=1.0 / window, adjust=False).mean() / atr)
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index)
                        .ewm(alpha=1.0 / window, adjust=False).mean() / atr)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean().fillna(0.0)


# ----------------------------------------------------------------------
# Confirmation configuration
# ----------------------------------------------------------------------
@dataclass
class ConfirmationConfig:
    """Thresholds for the 8 entry confirmations. Tune to tighten/loosen."""
    rsi_window: int = 14
    rsi_max: float = 90.0      # don't buy when wildly overbought
    rsi_min: float = 50.0      # require some strength
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_fast: int = 20
    sma_slow: int = 50
    momentum_window: int = 12
    adx_window: int = 14
    adx_min: float = 20.0
    vol_window: int = 20


# ----------------------------------------------------------------------
# Build the confirmation matrix
# ----------------------------------------------------------------------
def confirmation_matrix(df: pd.DataFrame,
                        cfg: ConfirmationConfig | None = None) -> pd.DataFrame:
    """
    Compute all 8 boolean confirmations for each bar.

    Returns a DataFrame with 8 boolean columns plus a 'n_confirmations'
    integer column (0..8), aligned to df.index.
    """
    cfg = cfg or ConfirmationConfig()
    close, vol = df["Close"], df["Volume"]

    rsi_v = rsi(close, cfg.rsi_window)
    macd_v = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    sma_fast = close.rolling(cfg.sma_fast).mean()
    sma_slow = close.rolling(cfg.sma_slow).mean()
    mom = roc(close, cfg.momentum_window)
    adx_v = adx(df, cfg.adx_window)
    vol_sma = vol.rolling(cfg.vol_window).mean()

    conf = pd.DataFrame(index=df.index)
    conf["rsi_not_overbought"] = rsi_v < cfg.rsi_max
    conf["rsi_strength"] = rsi_v > cfg.rsi_min
    conf["macd_bullish"] = macd_v["macd"] > macd_v["signal"]
    conf["price_above_sma"] = close > sma_slow
    conf["ma_crossover"] = sma_fast > sma_slow
    conf["positive_momentum"] = mom > 0
    conf["adx_trending"] = adx_v > cfg.adx_min
    conf["volume_participation"] = vol > vol_sma

    bool_cols = conf.columns.tolist()
    # Treat NaN (warm-up period) as False (not confirmed).
    conf[bool_cols] = conf[bool_cols].fillna(False)
    conf["n_confirmations"] = conf[bool_cols].sum(axis=1).astype(int)
    return conf



def latest_readings(df: pd.DataFrame,
                    cfg: ConfirmationConfig | None = None) -> Dict[str, str]:
    """
    Human-readable current values for each confirmation (for the dashboard
    checklist). Returns a dict keyed by the same names as confirmation_matrix.
    """
    cfg = cfg or ConfirmationConfig()
    close, vol = df["Close"], df["Volume"]
    rsi_v = rsi(close, cfg.rsi_window).iloc[-1]
    macd_v = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    macd_line = macd_v["macd"].iloc[-1]
    signal_line = macd_v["signal"].iloc[-1]
    sma_fast = close.rolling(cfg.sma_fast).mean().iloc[-1]
    sma_slow = close.rolling(cfg.sma_slow).mean().iloc[-1]
    mom = roc(close, cfg.momentum_window).iloc[-1] * 100
    adx_v = adx(df, cfg.adx_window).iloc[-1]
    vol_now = vol.iloc[-1]
    vol_sma = vol.rolling(cfg.vol_window).mean().iloc[-1]

    def f(x, p=1):
        try:
            return f"{x:,.{p}f}"
        except Exception:  # noqa: BLE001
            return "-"

    return {
        "rsi_not_overbought": f"RSI {f(rsi_v)}",
        "rsi_strength": f"RSI {f(rsi_v)}",
        "macd_bullish": f"{f(macd_line, 3)} vs {f(signal_line, 3)}",
        "price_above_sma": f"px vs SMA{cfg.sma_slow} {f(sma_slow)}",
        "ma_crossover": f"{f(sma_fast)} / {f(sma_slow)}",
        "positive_momentum": f"ROC {f(mom)}%",
        "adx_trending": f"ADX {f(adx_v)}",
        "volume_participation": f"vol {f(vol_now,0)} vs {f(vol_sma,0)}",
    }
