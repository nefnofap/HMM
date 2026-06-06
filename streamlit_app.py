"""
streamlit_app.py
================

The simplest possible "website" for the HMM regime detector.

Streamlit turns a Python script into a web page automatically - no HTML,
CSS, or JavaScript required. This is the easiest path for a beginner to see
the model running in a browser.

Run it (see WINDOWS_SETUP.md for full beginner steps):

    streamlit run streamlit_app.py

Your browser opens at http://localhost:8501 with an interactive dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hmm_predictor import ConvergenceError, HMMPredictor, SingularCovarianceError
from regime_detection import build_features, load_prices, summarise_states

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="HMM Regime Detector", layout="wide")
st.title("Market Regime Detector (Hidden Markov Model)")
st.caption(
    "Detects hidden market 'regimes' (e.g. bull / bear / sideways) from price "
    "data. For research and learning only - not financial advice."
)

# ----------------------------------------------------------------------
# Sidebar controls (these become interactive widgets on the web page)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="BTC-USD")
    n_states = st.slider("Number of regimes", min_value=2, max_value=8, value=7)
    interval = st.selectbox("Bar interval", ["1h", "1d"], index=0)
    period = st.selectbox("History to download", ["730d", "365d", "180d"], index=0)
    run_button = st.button("Detect regimes", type="primary")


# ----------------------------------------------------------------------
# Caching: avoid re-downloading / re-fitting on every interaction.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return load_prices(ticker, period=period, interval=interval)


@st.cache_resource(show_spinner=False)
def fit_model(ticker: str, period: str, interval: str, n_states: int):
    df = get_prices(ticker, period, interval)
    features = build_features(df)
    model = HMMPredictor(
        n_components=n_states,
        covariance_type="diag",
        n_iter=300,
        init_method="kmeans",
        random_state=42,
    )
    model.fit(features)
    states = model.predict_hidden_states(features)
    return df, features, model, states


# ----------------------------------------------------------------------
# Main action
# ----------------------------------------------------------------------
if run_button:
    try:
        with st.spinner("Downloading data and fitting the model..."):
            df, features, model, states = fit_model(
                ticker, period, interval, n_states
            )
    except ConvergenceError as exc:
        st.error(f"The model did not converge: {exc}")
        st.stop()
    except SingularCovarianceError as exc:
        st.error(f"Covariance problem: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surface anything else to the user
        st.error(f"Something went wrong: {exc}")
        st.stop()

    st.success(
        f"Fitted {n_states} regimes on {len(features)} bars "
        f"(log-likelihood {model.score_:.1f})."
    )

    # --- Price chart coloured by regime --------------------------------
    st.subheader(f"{ticker} price, coloured by detected regime")
    close = df["Close"].reindex(features.index)
    chart_df = pd.DataFrame({"price": close.values, "regime": states},
                            index=features.index)
    st.scatter_chart(chart_df, y="price", color="regime", height=420)

    # --- Regime summary table ------------------------------------------
    st.subheader("Regime summary (sorted by average return)")
    summary = summarise_states(features, states)
    st.dataframe(summary, use_container_width=True)

    # --- Forecast ------------------------------------------------------
    st.subheader("Next-step forecast")
    forecast = model.forecast_next(features, n_steps=3)
    fc_rows = [
        {
            "step": f"t+{f['step']}",
            "expected_return": f["expected_observation"][0],
            "most_likely_regime": int(np.argmax(f["state_distribution"])),
        }
        for f in forecast["forecast"]
    ]
    st.table(pd.DataFrame(fc_rows))

    st.caption(
        "Regime IDs are arbitrary integers. The summary table tells you which "
        "ID is the bull (highest mean return) and which is the bear/crash."
    )
else:
    st.info("Set your options in the sidebar, then click **Detect regimes**.")
