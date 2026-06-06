"""
example_usage.py
================

End-to-end demonstration of HMMPredictor:

    1. Build a synthetic regime-switching return series.
    2. Pre-process raw PRICES -> log returns (numerical stability).
    3. Fit, decode hidden states (Viterbi), and forecast.
    4. Serialise the trained model to JSON and reload it (lightweight backend).
    5. Show the raw Viterbi-decoding snippet and the front-end API schema.

Run:
    python example_usage.py
"""

import json

import numpy as np

from hmm_predictor import (
    ConvergenceError,
    HMMPredictor,
    SingularCovarianceError,
)


# ----------------------------------------------------------------------
# 0. Synthetic data: three regimes (bear / sideways / bull)
# ----------------------------------------------------------------------
def make_synthetic_prices(n: int = 1500, seed: int = 7) -> np.ndarray:
    """Generate a price path that switches between three return regimes."""
    rng = np.random.default_rng(seed)
    # (daily mean return, daily vol) per regime
    regimes = [(-0.0015, 0.020), (0.0002, 0.006), (0.0015, 0.012)]
    trans = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.03, 0.94, 0.03],
            [0.01, 0.04, 0.95],
        ]
    )
    state = 1
    log_returns = []
    for _ in range(n):
        mu, sigma = regimes[state]
        log_returns.append(rng.normal(mu, sigma))
        state = rng.choice(3, p=trans[state])
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return prices


def main() -> None:
    prices = make_synthetic_prices()

    # ------------------------------------------------------------------
    # 1. NUMERICAL STABILITY: prices -> log returns.
    #    Never feed raw prices to a Gaussian HMM: they are non-stationary
    #    and trigger underflow / singular covariances. Standardisation to
    #    zero-mean/unit-variance is handled INSIDE HMMPredictor.fit.
    # ------------------------------------------------------------------
    log_returns = np.diff(np.log(prices)).reshape(-1, 1)

    # ------------------------------------------------------------------
    # 2. Fit with explicit robustness handling.
    # ------------------------------------------------------------------
    predictor = HMMPredictor(
        n_components=3,
        covariance_type="diag",   # robust default for financial data
        n_iter=200,
        init_method="kmeans",     # reproducible, stable EM initialisation
        random_state=42,
    )

    try:
        predictor.fit(log_returns)
    except ConvergenceError as exc:
        print(f"[fit] convergence failure: {exc}")
        return
    except SingularCovarianceError as exc:
        print(f"[fit] singular covariance: {exc}")
        return

    print(f"Fitted: {predictor}")
    print(f"Log-likelihood: {predictor.score_:.2f}")

    # ------------------------------------------------------------------
    # 3. Decode hidden states (Viterbi most-likely path).
    # ------------------------------------------------------------------
    states = predictor.predict_hidden_states(log_returns)
    print(f"Decoded states (last 20): {states[-20:].tolist()}")

    # ------------------------------------------------------------------
    # 4. Forecast next 3 steps.
    # ------------------------------------------------------------------
    forecast = predictor.forecast_next(log_returns, n_steps=3)
    print("Current state distribution:",
          np.round(forecast["current_state_distribution"], 3).tolist())
    for f in forecast["forecast"]:
        print(f"  t+{f['step']}: P(states)={np.round(f['state_distribution'], 3).tolist()} "
              f"E[log-ret]={f['expected_observation'][0]:.5f}")

    # ------------------------------------------------------------------
    # 5. Persistence: save to JSON, reload, verify identical decoding.
    # ------------------------------------------------------------------
    predictor.save("model_params.json")
    reloaded = HMMPredictor.load("model_params.json")
    states_reloaded = reloaded.predict_hidden_states(log_returns)
    assert np.array_equal(states, states_reloaded), "Reload mismatch!"
    print("Round-trip JSON persistence verified (decoding identical).")

    # ------------------------------------------------------------------
    # 6. Front-end API payload.
    # ------------------------------------------------------------------
    payload = predictor.api_summary(log_returns, n_steps=3)
    print(f"API payload keys: {list(payload.keys())}")

    # ------------------------------------------------------------------
    # 7. RAW VITERBI SNIPPET (most-likely-path decoding).
    #    predict_hidden_states wraps exactly this; shown explicitly for
    #    mathematical transparency.
    # ------------------------------------------------------------------
    X_scaled = predictor.scaler.transform(log_returns)
    log_prob, viterbi_path = predictor.model.decode(X_scaled, algorithm="viterbi")
    print(f"Viterbi log-probability of the optimal path: {log_prob:.2f}")
    print(f"Viterbi path (first 10): {viterbi_path[:10].tolist()}")


# ----------------------------------------------------------------------
# FRONT-END API JSON SCHEMA (sample contract for the dashboard)
# ----------------------------------------------------------------------
API_RESPONSE_SCHEMA = {
    "schema_version": "1.0",
    "generated_at": "2026-06-06T12:00:00+00:00",
    "n_components": 3,
    "feature_names": ["log_return"],
    # One decoded regime id per input timestep (Viterbi). Map ids to labels
    # (e.g. 0=bear,1=sideways,2=bull) on the client after sorting by mean.
    "states": [1, 1, 0, 0, 2, "..."],
    # Posterior confidence per state per timestep; rows sum to 1. Use to shade
    # regime certainty in the UI instead of a hard assignment.
    "state_probabilities": [[0.8, 0.15, 0.05], "..."],
    "forecast": {
        "horizon": 3,
        "current_state_distribution": [0.7, 0.2, 0.1],
        "forecast": [
            {
                "step": 1,
                "state_distribution": [0.66, 0.23, 0.11],
                "expected_observation": [0.00012],  # original units (log-ret)
            }
        ],
    },
}

# SERIALISED MODEL SCHEMA (what save()/to_json() store; load into the backend):
MODEL_PARAM_SCHEMA = {
    "schema_version": "1.0",
    "fitted_at": "2026-06-06T12:00:00+00:00",
    "config": {
        "n_components": 3,
        "covariance_type": "diag",
        "n_iter": 200,
        "tol": 1e-4,
        "min_covar": 1e-3,
        "init_method": "kmeans",
        "random_state": 42,
    },
    "metadata": {
        "n_features": 1,
        "feature_names": ["log_return"],
        "converged": True,
        "log_likelihood": 1234.56,
    },
    "scaler": {"mean": [0.0001], "scale": [0.012]},
    "parameters": {
        "startprob": [0.33, 0.34, 0.33],
        "transmat": [[0.95, 0.04, 0.01], [0.03, 0.94, 0.03], [0.01, 0.04, 0.95]],
        "means": [[-0.12], [0.01], [0.10]],          # in SCALED space
        "covars": [[[0.9]], [[0.3]], [[0.6]]],        # (n_states, n_feat, n_feat)
    },
}


if __name__ == "__main__":
    main()
    print("\n--- Sample front-end API schema ---")
    print(json.dumps(API_RESPONSE_SCHEMA, indent=2))
