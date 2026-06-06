"""
streamlit_app.py
================

REGIME TERMINAL - a dark "command-center" web dashboard for the HMM market
regime detector, styled after an intelligence-terminal layout:

    * dark, near-black background with monospace type and thin panel borders
    * left column  : model dossier + regime activity stat boxes + volatility tiers
    * center column: regime-overlay price chart + regime distribution bars
    * right column : "REGIME LIST" of detected states with labels (like an ops list)

Streamlit turns this Python file into a web page - no HTML/CSS/JS skills needed
to run it. The look-and-feel is applied with a small injected CSS block.

Run it (see WINDOWS_SETUP.md for beginner steps):

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hmm_predictor import ConvergenceError, HMMPredictor, SingularCovarianceError
from regime_detection import (build_features, load_prices, periods_per_year_for,
                              summarise_states)
from backtest import StrategyProfile, backtest_strategy, trades_to_frame
from metrics import drawdown_series, equity_curve, summary as metric_summary

# ----------------------------------------------------------------------
# Page setup + command-center theme (injected CSS)
# ----------------------------------------------------------------------
st.set_page_config(page_title="REGIME TERMINAL", layout="wide",
                   initial_sidebar_state="collapsed")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #0a0b0d;
  --panel: #0e0f12;
  --border: #23262b;
  --text: #c9ccd1;
  --muted: #6b7078;
  --accent: #c0392b;
  --accent-dim: #7e2b22;
  --green: #4f9d69;
}

html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', 'Share Tech Mono', 'Courier New', monospace !important;
}
.block-container { padding-top: 1.2rem; max-width: 1500px; }
#MainMenu, footer, header { visibility: hidden; }

/* Top title bar */
.term-header {
  display:flex; justify-content:space-between; align-items:center;
  border-bottom: 1px solid var(--border); padding-bottom: .5rem; margin-bottom: 1rem;
}
.term-title { font-size: 1.15rem; letter-spacing: 3px; color: var(--text); }
.term-title b { color: var(--accent); }
.term-sub { font-size: .7rem; color: var(--muted); letter-spacing: 1px; }

/* Panels */
.panel {
  border: 1px solid var(--border); background: var(--panel);
  padding: .8rem .9rem; margin-bottom: .9rem;
}
.panel-h {
  font-size: .72rem; letter-spacing: 2px; color: var(--text);
  text-transform: uppercase; border-bottom: 1px solid var(--border);
  padding-bottom: .35rem; margin-bottom: .6rem;
}
.panel-h span { color: var(--accent); }
.kv { display:flex; justify-content:space-between; font-size:.72rem; padding:.12rem 0; }
.kv .k { color: var(--muted); }
.kv .v { color: var(--text); }

/* Stat boxes (Total / Bullish / Bearish) */
.statgrid { display:flex; gap:.5rem; }
.stat { flex:1; border:1px solid var(--border); padding:.4rem; text-align:center; }
.stat .n { font-size:1.4rem; color: var(--text); }
.stat .l { font-size:.6rem; color: var(--muted); letter-spacing:1px; }
.stat.bull .n { color: var(--green); }
.stat.bear .n { color: var(--accent); }

/* Risk / volatility tier rows */
.risk { display:flex; align-items:center; margin:.3rem 0; }
.risk .box {
  background: var(--accent-dim); color:#fff; font-size:.9rem; width:42px;
  text-align:center; padding:.25rem 0; margin-right:.6rem; border:1px solid var(--accent);
}
.risk .lbl { font-size:.7rem; color: var(--muted); letter-spacing:1px; }

/* Operations / regime list */
.op { border:1px solid var(--border); padding:.5rem .6rem; margin-bottom:.5rem; }
.op .code { font-size:.6rem; color: var(--accent); letter-spacing:1px; }
.op .ttl { font-size:.8rem; color: var(--text); margin:.15rem 0; }
.op .meta { font-size:.65rem; color: var(--muted); }

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: var(--panel) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius:0 !important;
  font-family:'JetBrains Mono', monospace !important;
}
.stSlider label, .stTextInput label, .stSelectbox label {
  color: var(--muted) !important; font-size:.65rem !important; letter-spacing:1px;
}
.stButton button {
  background: var(--accent-dim) !important; color:#fff !important;
  border:1px solid var(--accent) !important; border-radius:0 !important;
  font-family:'JetBrains Mono', monospace !important; letter-spacing:2px;
}
.stButton button:hover { background: var(--accent) !important; }
.stDataFrame { border:1px solid var(--border); }

/* Metric cards (Sharpe / MaxDD / etc.) */
.metricgrid { display:flex; flex-wrap:wrap; gap:.5rem; }
.metric {
  flex:1; min-width:115px; border:1px solid var(--border);
  background:var(--panel); padding:.5rem .6rem;
}
.metric .l { font-size:.56rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }
.metric .n { font-size:1.2rem; color:var(--text); margin-top:.2rem; }
.metric.good .n { color: var(--green); }
.metric.bad .n { color: var(--accent); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="term-header">
      <div>
        <div class="term-title">REGIME <b>TERMINAL</b></div>
        <div class="term-sub">HIDDEN MARKOV MODEL // MARKET REGIME DETECTION</div>
      </div>
      <div class="term-sub">RESEARCH BUILD &mdash; NOT FINANCIAL ADVICE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Control bar (top row instead of a sidebar, to match the terminal look)
# ----------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns([1.7, 1.2, 0.9, 0.9, 1.3, 1.2])
with c1:
    ticker = st.text_input("TICKER", value="BTC-USD")
with c2:
    n_states = st.slider("REGIMES", min_value=2, max_value=8, value=7)
with c3:
    interval = st.selectbox("INTERVAL", ["1h", "1d"], index=0)
with c4:
    period = st.selectbox("HISTORY", ["730d", "365d", "180d"], index=0)
with c5:
    profile_name = st.selectbox("PROFILE", ["CONSERVATIVE (7/8, 2.5x)",
                                            "AGGRESSIVE (5/8, 4x)"], index=0)
with c6:
    st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    run_button = st.button("RUN DETECTION", type="primary", use_container_width=True)


# ----------------------------------------------------------------------
# Caching to avoid re-downloading / re-fitting on every interaction
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fit_model(ticker: str, period: str, interval: str, n_states: int):
    df = load_prices(ticker, period=period, interval=interval)
    features = build_features(df)
    model = HMMPredictor(
        n_components=n_states, covariance_type="diag",
        n_iter=300, init_method="kmeans", random_state=42,
    )
    model.fit(features)
    states = model.predict_hidden_states(features)
    return df, features, model, states


# ----------------------------------------------------------------------
# Themed regime-overlay chart (matplotlib, dark)
# ----------------------------------------------------------------------
def regime_chart(df, features, states, ticker, n_states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    close = df["Close"].reindex(features.index)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor("#0a0b0d")
    ax.set_facecolor("#0a0b0d")
    ax.plot(features.index, close.values, color="#3a3e45", linewidth=0.8, zorder=1)
    sc = ax.scatter(features.index, close.values, c=states, cmap="Spectral",
                    s=7, zorder=2, vmin=0, vmax=max(n_states - 1, 1))
    ax.set_title(f"{ticker}  //  CLOSE PRICE BY DETECTED REGIME",
                 color="#c9ccd1", fontsize=10, loc="left", family="monospace")
    for spine in ax.spines.values():
        spine.set_color("#23262b")
    ax.tick_params(colors="#6b7078", labelsize=7)
    ax.grid(color="#15171a", linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("REGIME", color="#6b7078", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#6b7078", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#6b7078")
    fig.tight_layout()
    return fig


def dist_chart(states, n_states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = pd.Series(states).value_counts().reindex(range(n_states), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 1.7))
    fig.patch.set_facecolor("#0a0b0d")
    ax.set_facecolor("#0a0b0d")
    ax.bar(counts.index, counts.values, color="#7e2b22", edgecolor="#c0392b", width=0.7)
    ax.set_title("REGIME DISTRIBUTION (BARS BY STATE)", color="#6b7078",
                 fontsize=8, loc="left", family="monospace")
    for spine in ax.spines.values():
        spine.set_color("#23262b")
    ax.tick_params(colors="#6b7078", labelsize=7)
    fig.tight_layout()
    return fig


def equity_drawdown_chart(strat_ret, bench_ret):
    """Equity curve (strategy vs buy-and-hold) + underwater drawdown, dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq_s = equity_curve(strat_ret)
    eq_b = equity_curve(bench_ret)
    dd = drawdown_series(strat_ret)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (ax1, ax2):
        fig.patch.set_facecolor("#0a0b0d")
        ax.set_facecolor("#0a0b0d")
        for spine in ax.spines.values():
            spine.set_color("#23262b")
        ax.tick_params(colors="#6b7078", labelsize=7)
        ax.grid(color="#15171a", linewidth=0.5)

    ax1.plot(eq_s.index, eq_s.values, color="#4f9d69", linewidth=1.1,
             label="STRATEGY")
    ax1.plot(eq_b.index, eq_b.values, color="#6b7078", linewidth=0.9,
             label="BUY & HOLD")
    ax1.set_title("EQUITY CURVE (GROWTH OF 1.0)", color="#c9ccd1",
                  fontsize=9, loc="left", family="monospace")
    leg = ax1.legend(loc="upper left", fontsize=7, facecolor="#0e0f12",
                     edgecolor="#23262b")
    for t in leg.get_texts():
        t.set_color("#c9ccd1")

    ax2.fill_between(dd.index, dd.values, 0.0, color="#7e2b22", alpha=0.8)
    ax2.set_title("DRAWDOWN (UNDERWATER)", color="#6b7078", fontsize=8,
                  loc="left", family="monospace")
    fig.tight_layout()
    return fig


def metric_card(label, value, fmt="{:.2f}", good=None):
    """Build one metric card. `good` True->green, False->red, None->neutral."""
    cls = "metric"
    if good is True:
        cls += " good"
    elif good is False:
        cls += " bad"
    val = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    return f"<div class='{cls}'><div class='l'>{label}</div><div class='n'>{val}</div></div>"
def panel(title_html: str, body_html: str) -> str:
    return (f"<div class='panel'><div class='panel-h'>{title_html}</div>"
            f"{body_html}</div>")


def kv(k: str, v: str) -> str:
    return f"<div class='kv'><span class='k'>{k}</span><span class='v'>{v}</span></div>"


# ----------------------------------------------------------------------
# Main render
# ----------------------------------------------------------------------
if run_button:
    try:
        with st.spinner("DOWNLOADING DATA // FITTING MODEL ..."):
            df, features, model, states = fit_model(ticker, period, interval, n_states)
    except ConvergenceError as exc:
        st.error(f"MODEL DID NOT CONVERGE: {exc}")
        st.stop()
    except SingularCovarianceError as exc:
        st.error(f"COVARIANCE FAILURE: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"ERROR: {exc}")
        st.stop()

    summary = summarise_states(features, states)  # sorted by mean_return ascending
    n_bars = len(features)
    bull_state = int(summary["mean_return"].idxmax())
    bear_state = int(summary["mean_return"].idxmin())
    bull_bars = int((states == bull_state).sum())
    bear_bars = int((states == bear_state).sum())

    # Volatility tiers (by return_volatility) -> HIGH / MED / LOW counts
    vol_sorted = summary.sort_values("return_volatility", ascending=False)
    thirds = np.array_split(vol_sorted.index.to_numpy(), 3)
    tier_counts = []
    for grp in thirds:
        tier_counts.append(int(np.isin(states, grp).sum()))
    while len(tier_counts) < 3:
        tier_counts.append(0)

    left, center, right = st.columns([1.1, 2.3, 1.15])

    # ---------------- LEFT: dossier + activity + tiers ----------------
    with left:
        body = (
            kv("TICKER", ticker)
            + kv("REGIMES", str(n_states))
            + kv("INTERVAL", interval)
            + kv("HISTORY", period)
            + kv("BARS FITTED", f"{n_bars:,}")
            + kv("LOG-LIKELIHOOD", f"{model.score_:,.1f}")
            + kv("CONVERGED", "YES" if model.converged_ else "NO")
        )
        st.markdown(panel("MODEL <span>DOSSIER</span>", body), unsafe_allow_html=True)

        act = (
            "<div class='statgrid'>"
            f"<div class='stat'><div class='n'>{n_bars:,}</div><div class='l'>TOTAL</div></div>"
            f"<div class='stat bull'><div class='n'>{bull_bars:,}</div><div class='l'>BULLISH</div></div>"
            f"<div class='stat bear'><div class='n'>{bear_bars:,}</div><div class='l'>BEARISH</div></div>"
            "</div>"
        )
        st.markdown(panel("REGIME <span>ACTIVITY</span>", act), unsafe_allow_html=True)

        risk = ""
        for cnt, lbl in zip(tier_counts, ["HIGH VOLATILITY", "MEDIUM VOLATILITY",
                                          "LOW VOLATILITY"]):
            risk += (f"<div class='risk'><div class='box'>{cnt:02d}</div>"
                     f"<div class='lbl'>{lbl}</div></div>")
        st.markdown(panel("VOLATILITY <span>TIERS</span>", risk), unsafe_allow_html=True)

    # ---------------- CENTER: charts ----------------
    with center:
        st.pyplot(regime_chart(df, features, states, ticker, n_states),
                  use_container_width=True)
        st.pyplot(dist_chart(states, n_states), use_container_width=True)

    # ---------------- RIGHT: regime list (ops list) ----------------
    with right:
        items = ""
        # show highest-return regimes first (most "actionable")
        for state_id, row in summary.sort_values("mean_return", ascending=False).iterrows():
            label = row["label"] or "NEUTRAL REGIME"
            tag = ("BULL" if state_id == bull_state
                   else "BEAR" if state_id == bear_state else "NEUTRAL")
            items += (
                "<div class='op'>"
                f"<div class='code'>REGIME // STATE {int(state_id)} &mdash; {tag}</div>"
                f"<div class='ttl'>{label}</div>"
                f"<div class='meta'>&mu; {row['mean_return']:+.5f} &nbsp; "
                f"&sigma; {row['return_volatility']:.5f} &nbsp; n={int(row['n_obs']):,}</div>"
                "</div>"
            )
        st.markdown(panel(f"REGIME <span>LIST ({n_states})</span>", items),
                    unsafe_allow_html=True)

        # Forecast panel
        forecast = model.forecast_next(features, n_steps=3)
        fc = ""
        for f in forecast["forecast"]:
            ml = int(np.argmax(f["state_distribution"]))
            fc += kv(f"t+{f['step']}",
                     f"E[ret] {f['expected_observation'][0]:+.5f} | state {ml}")
        st.markdown(panel("FORECAST <span>(NEXT 3)</span>", fc), unsafe_allow_html=True)

    # ---------------- BACKTEST (full width, below the 3 columns) -------
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    profile = (StrategyProfile.aggressive() if profile_name.startswith("AGGRESSIVE")
               else StrategyProfile.conservative())
    try:
        bt = backtest_strategy(df, features, states, profile=profile)
        ppy = periods_per_year_for(interval, ticker)
        m = metric_summary(bt["strategy_returns"], ppy,
                           benchmark_returns=bt["benchmark_returns"])

        # Metric cards row
        cards = (
            metric_card("TOTAL RETURN", m["total_return"] * 100, "{:+.1f}%",
                        good=m["total_return"] > 0)
            + metric_card("ALPHA vs HODL", m.get("alpha_vs_buy_hold", 0) * 100,
                          "{:+.1f}%", good=m.get("alpha_vs_buy_hold", 0) > 0)
            + metric_card("SHARPE", m["sharpe"], "{:.2f}", good=m["sharpe"] > 1)
            + metric_card("SORTINO", m["sortino"], "{:.2f}", good=m["sortino"] > 1)
            + metric_card("MAX DRAWDOWN", m["max_drawdown"] * 100, "{:.1f}%",
                          good=m["max_drawdown"] > -0.25)
            + metric_card("CALMAR", m["calmar"], "{:.2f}", good=m["calmar"] > 0.5)
            + metric_card("WIN RATE", m["win_rate"] * 100, "{:.0f}%",
                          good=m["win_rate"] > 0.5)
            + metric_card("TRADES", len(bt["trades"]), "{:d}")
        )
        st.markdown(
            panel(f"BACKTEST <span>// {profile_name}</span>",
                  f"<div class='metricgrid'>{cards}</div>"),
            unsafe_allow_html=True,
        )

        ec, tc = st.columns([2, 1.2])
        with ec:
            st.pyplot(equity_drawdown_chart(bt["strategy_returns"],
                                            bt["benchmark_returns"]),
                      use_container_width=True)
        with tc:
            trades_df = trades_to_frame(bt["trades"])
            if not trades_df.empty:
                show = trades_df[["entry_time", "exit_time", "bars_held",
                                  "leveraged_return_pct", "exit_reason"]].copy()
                show["leveraged_return_pct"] = (show["leveraged_return_pct"]
                                                * 100).round(2)
                show = show.rename(columns={"leveraged_return_pct": "ret_%"})
                st.markdown("<div class='panel-h'>TRADE LOG</div>",
                            unsafe_allow_html=True)
                st.dataframe(show.tail(15), use_container_width=True,
                             height=320, hide_index=True)
            else:
                st.markdown(
                    panel("TRADE LOG",
                          "<div class='kv'><span class='k'>No trades triggered. "
                          "Loosen the profile or widen history.</span></div>"),
                    unsafe_allow_html=True)

        st.caption(
            "In-sample backtest (model fit on full window). For honest "
            "out-of-sample results use walkforward.walkforward_backtest in code. "
            "Research only - not financial advice."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Backtest unavailable: {exc}")

else:
    st.markdown(
        "<div class='panel'><div class='panel-h'>STANDBY</div>"
        "<div class='kv'><span class='k'>Set parameters above and press "
        "<b style='color:#c0392b'>RUN DETECTION</b> to begin.</span></div></div>",
        unsafe_allow_html=True,
    )
