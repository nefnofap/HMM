# HMM Market Regime Detection

A production-grade Hidden Markov Model (HMM) toolkit for financial time-series
regime detection and short-horizon forecasting, designed to plug into a web
backend (FastAPI / Flask / Django).

## Contents

| File | Purpose |
|------|---------|
| `hmm_predictor.py` | Reusable `HMMPredictor` engine: fit, Viterbi decoding, forecasting, JSON save/load |
| `regime_detection.py` | Phase 1 script: Bitcoin (Yahoo Finance, hourly/730d) 7-regime detection + plot |
| `example_usage.py` | End-to-end demo and documented API/model JSON schemas |
| `btc_regime_model.json` | Example serialized 7-state BTC model |
| `regime_plot.png` | Close price colour-coded by detected regime |

## Quick start

```bash
pip install -r requirements.txt

# Validate the core HMM logic on real BTC data (falls back to synthetic if offline)
python regime_detection.py --ticker BTC-USD --n-states 7

# Full engine demo (fit, decode, forecast, JSON round-trip)
python example_usage.py
```

## The `HMMPredictor` engine

```python
from hmm_predictor import HMMPredictor
import numpy as np

log_returns = np.diff(np.log(prices)).reshape(-1, 1)

model = HMMPredictor(n_components=3, covariance_type="diag", init_method="kmeans")
model.fit(log_returns)                      # scaling + EM training (handled internally)
states = model.predict_hidden_states(log_returns)   # Viterbi most-likely path
forecast = model.forecast_next(log_returns, n_steps=3)

model.save("model.json")                    # persist parameters only (lightweight)
reloaded = HMMPredictor.load("model.json")  # reconstructs in milliseconds, no re-fit
```

### Key features
- Configurable `n_components`, `covariance_type` (`full`/`tied`/`diag`/`spherical`), `n_iter`, `tol`.
- K-Means initialisation of emission means for reproducible EM fits.
- Explicit `ConvergenceError` / `SingularCovarianceError` handling.
- JSON serialisation of all parameters (transmat, means, covars, scaler) — no pickle.
- `api_summary()` returns a single front-end-ready payload (states, posteriors, forecast).

## Modelling notes
- Train on **stationary features** (log returns, range, volume change), never raw prices.
- Standardisation to zero-mean/unit-variance is done internally and the scaler is persisted.
- HMM parameters are only locally stationary in markets — re-fit on rolling windows.

> For research/simulation. Keep any live trading behind your own broker keys and risk controls.



## Discord login (members-only access)

The dashboard supports an **optional Discord OAuth gate** so only members of
your Discord server can use the site. If credentials are not configured, the
gate is skipped (great for local development).

### Configure (one time)

1. Create an app at <https://discord.com/developers/applications>.
2. In **OAuth2 -> Redirects**, add your hosted URL exactly
   (e.g. `https://nefnofap-hmm.streamlit.app`).
3. Copy the **Client ID** and **Client Secret**.
4. In Discord enable **Developer Mode** (User Settings -> Advanced),
   right-click your server icon -> **Copy Server ID**.
5. Locally, copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` and fill in the values. On Streamlit Community
   Cloud, paste them into the app's **Secrets** panel.

### How it works
- User clicks **Login with Discord**.
- Discord redirects back with a code; we exchange it for an access token.
- We call `/users/@me/guilds` and check if your server's ID is in the list.
- If yes, they get in; if no, they see a "Join the server" screen.
- The session is cached in Streamlit's `session_state` for the visit.

> Without secrets, the gate is automatically disabled so the site runs free.
> With secrets, only members get in &mdash; verified via Discord's API, no manual list.
