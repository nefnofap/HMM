# Regime Terminal — In-Depth Guide

This guide covers four things:

1. **What each piece does** (the mental model)
2. **How to use it** (commands, dashboard, walk-forward)
3. **How to actually trade with it** (honestly)
4. **How to make it mainstream** so your community can use it

> **Read this first — honesty notice.** This is a *research and education* tool,
> not financial advice and not a profitable bot out of the box. When I tested the
> default strategy on 2 years of hourly BTC data, it **lost money** in-sample
> (Sharpe -0.22, max drawdown -55%) and out-of-sample in walk-forward
> (-66%). That is normal for a naive long-only regime strategy in a choppy
> market, and it is exactly why the walk-forward and metrics tooling exists: to
> tell you the truth *before* you risk money. Treat green backtests with
> suspicion and never trade money you can't afford to lose.

---

## 1. The mental model

The system has four layers, each in its own file:

| Layer | File | Job |
|------|------|-----|
| **Regime engine** | `hmm_predictor.py` | The HMM. Learns hidden market "regimes" and labels each bar. |
| **Data + features** | `regime_detection.py` | Downloads prices (multi-asset), builds the 3 features, runs detection. |
| **Signals** | `indicators.py` | The 8 technical confirmations (RSI, MACD, ADX, momentum, MAs, volume). |
| **Strategy + evaluation** | `backtest.py`, `walkforward.py`, `metrics.py` | Turns regimes + signals into trades, then scores them honestly. |
| **Dashboard** | `streamlit_app.py` | The web UI that ties it all together. |

**The core idea:** the HMM says *what kind of market* we're in (bull / bear /
neutral). The 8 indicators decide *when* to actually enter inside a bull regime.
You exit the moment the regime flips to bear (or a trailing stop triggers).

---

## 2. How to use it

### 2a. Setup (one time)
Follow `WINDOWS_SETUP.md`. The short version (use **Python 3.13**, not 3.14):
```
cd %USERPROFILE%\Documents\hmm
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2b. Detect regimes on one asset
```
python regime_detection.py --ticker BTC
```
Prints a per-regime summary table (mean return + volatility per state), saves a
regime-colored price chart, and saves the trained model as JSON.

### 2c. Compare multiple assets (BTC / ETH / Gold)
```
python regime_detection.py --tickers BTC,ETH,GOLD
```
Each asset gets its own model and a side-by-side metrics table (Sharpe, max
drawdown, total return). Friendly aliases: `BTC`, `ETH`, `GOLD`/`XAU`/`XAUUSD`,
`SOL`, `SPX`, `NASDAQ`.

### 2d. The dashboard
```
streamlit run streamlit_app.py
```
Opens `http://localhost:8501`. Set the ticker, regimes, history, and **profile**
(Conservative 7-of-8 @ 2.5x, or Aggressive 5-of-8 @ 4x), then click
**RUN DETECTION**. You get:
- regime-overlay price chart + regime distribution
- the regime list (bull/bear/neutral) with stats
- a forecast panel
- a **backtest** block: metric cards (Sharpe, Max DD, Calmar, win rate, alpha),
  an equity curve vs buy-and-hold, an underwater drawdown chart, and a trade log.

### 2e. The honest test: walk-forward
The dashboard backtest is *in-sample* (model saw the whole window). For a
realistic estimate, run walk-forward in code:
```python
import warnings; warnings.filterwarnings("ignore")
from regime_detection import load_prices, build_features, periods_per_year_for
from walkforward import walkforward_backtest
from backtest import StrategyProfile
from metrics import summary

df = load_prices("BTC", period="730d", interval="1h")
feats = build_features(df)
wf = walkforward_backtest(df, feats, n_states=7,
                          train_size=2000, test_size=300,
                          profile=StrategyProfile.conservative())
print(summary(wf["strategy_returns"], periods_per_year_for("1h", "BTC"),
              benchmark_returns=wf["benchmark_returns"]))
```
If the in-sample result is great but walk-forward is bad (as with the defaults),
**trust the walk-forward number.** That gap is the difference between a fantasy
and reality.

### 2f. Tuning (the iterative loop)
Edit thresholds in `indicators.py` (`ConfirmationConfig`) and `StrategyProfile`
in `backtest.py`:
- **Drawdown too high?** Raise `min_confirmations` (e.g. 8/8), raise `adx_min`,
  lower leverage, tighten `rsi_max`.
- **Too few trades?** Lower `min_confirmations`, widen history.
- **Different market behaviour?** Try more/fewer `n_states`, or switch the
  feature set (e.g. add an ATR-based volatility feature).

Re-run walk-forward after every change. Tune on one period, then confirm on a
*different* untouched period before believing anything.

---

## 3. How to actually trade with it

**Important:** this repo does **not** place live orders, and I recommend keeping
it that way until you have a walk-forward edge you trust across multiple periods
and assets. Here is the responsible path from signal to trade:

### Step 1 — Paper trade first (weeks to months)
Run the dashboard daily/hourly and record what it *would* do. Compare the live
signals to what actually happens. No money involved. This catches data issues,
timing bugs, and over-optimistic backtests.

### Step 2 — Generate a live signal programmatically
```python
from regime_detection import load_prices, build_features
from hmm_predictor import HMMPredictor
from indicators import confirmation_matrix, ConfirmationConfig

model = HMMPredictor.load("btc_usd_regime_model.json")  # trained offline
df = load_prices("BTC", period="60d", interval="1h")
feats = build_features(df)
state = model.predict_hidden_states(feats)[-1]          # current regime
conf  = confirmation_matrix(df.reindex(feats.index)).iloc[-1]
print("regime:", state, "| confirmations:", int(conf["n_confirmations"]))
```
Your rule: **enter only if regime == bull AND confirmations >= your threshold;
exit if regime flips to bear.**

### Step 3 — Manual execution (recommended for most people)
Let the tool send you an alert (see 3a below), and you place the order yourself
on your exchange. This keeps a human in the loop and avoids the legal/technical
risks of an automated bot.

### Step 4 — Automated execution (advanced, optional)
Only after a proven paper-trading track record. Use a broker/exchange API
(e.g. **ccxt** for crypto). You'd add an `execution.py` that:
- loads the saved model, fetches the latest bars,
- computes regime + confirmations,
- places/closes orders via the exchange API,
- respects the 48h cooldown and position sizing,
- logs every action and has a kill switch.

> **Risk reality check:** leverage (2.5x–4x) magnifies losses as much as gains. A
> -55% drawdown at 2.5x is a wipe-out for most accounts. Start at 1x, tiny size.
> Never risk rent money. Crypto and leverage are how people get liquidated.

### 3a. Add price/signal alerts (a good first automation)
A safe, useful automation is a notifier (not an order-placer): a small script on
a schedule that posts to **Discord/Telegram** when the regime flips or entry
confirms. This is the natural "community" feature too — see below.

---

## 4. How to make it mainstream (share it with your community)

Goal: anyone in your community can use it **without installing Python**. Options,
easiest first:

### Option A — Streamlit Community Cloud (free, ~10 minutes) ⭐ recommended
1. Make sure the repo is on GitHub (it is: `nefnofap/hmm`).
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. Pick `nefnofap/hmm`, branch `main`, main file `streamlit_app.py`, **Deploy**.
4. You get a public URL like `https://nefnofap-hmm.streamlit.app` to share.

Pros: zero server management, free. Cons: limited compute (fine for this).

### Option B — Hugging Face Spaces (free, also easy)
Create a Space (type: Streamlit), point it at the repo. Similar to A, good if you
expect a bigger audience.

### Option C — A real server (Docker) for full control
Add a `Dockerfile`:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Deploy on Render, Railway, Fly.io, or a $5 VPS. Put it behind HTTPS.

### Making it community-friendly (the polish that drives adoption)
- **Caching & limits:** the dashboard already caches model fits. For many users,
  pre-train models offline (`regime_detection.py`) and ship the JSON so the app
  only *loads* them (milliseconds) instead of fitting per visitor.
- **Presets:** ship buttons for BTC/ETH/Gold so newcomers click, not type.
- **A clear disclaimer banner** (already in the UI) — important for trust and to
  avoid giving the impression of financial advice.
- **Discord/Telegram alert bot:** the single most "sticky" community feature.
  A scheduled job posts "BTC regime flipped to BULL, 7/8 confirmations" to a
  channel. People will come back for that.
- **Contributions:** add a `CONTRIBUTING.md`, label "good first issue" tasks
  (new indicators, new assets), and a `LICENSE` (MIT is common) so others can
  build on it.
- **Versioned models:** the model JSON has a `schema_version`; keep it so old
  shared models keep working as you evolve the code.

### Scaling checklist when more people use it
- Move from per-request fitting to **pre-trained model files** (done via `save`/`load`).
- Add a tiny **FastAPI** JSON endpoint if you want a mobile app or others to build
  on your signals (`/api/regimes?ticker=BTC`).
- Rate-limit Yahoo Finance calls (cache downloads); consider a paid data feed if
  you grow.
- Add basic **logging/metrics** so you can see usage and errors.

---

## File reference (cheat sheet)

| File | Run / import | What you get |
|------|--------------|--------------|
| `regime_detection.py` | `python regime_detection.py --ticker BTC` | regime table, chart, saved model |
| `regime_detection.py` | `--tickers BTC,ETH,GOLD` | multi-asset comparison |
| `streamlit_app.py` | `streamlit run streamlit_app.py` | full web dashboard + backtest |
| `walkforward.py` | `walkforward_backtest(...)` | honest out-of-sample results |
| `backtest.py` | `backtest_strategy(..., profile=...)` | trades + returns, conservative/aggressive |
| `indicators.py` | `confirmation_matrix(df)` | the 8 entry confirmations |
| `metrics.py` | `summary(returns, ppy)` | Sharpe, Sortino, Max DD, Calmar, win rate, alpha |

---

*Research and educational software. Markets involve risk of loss. Nothing here is
financial advice. You are responsible for your own trades.*
