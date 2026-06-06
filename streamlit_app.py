"""
streamlit_app.py
================

REGIME TERMINAL v3 - a dark "command-center" web dashboard for the HMM market
regime detector.

Three tabs
----------
1. ANALYSIS  - single instrument: live signal, confirmation checklist,
               ATR entry/stop/target levels + position sizing, in-sample
               backtest, and an optional honest WALK-FORWARD (out-of-sample) run.
2. SCANNER   - rank ALL instruments by their current signal (LONG first).
3. OPTIMIZER - grid-search RSI/ADX/leverage/confirmations, scored by
               walk-forward (out-of-sample) performance.

Run it (see WINDOWS_SETUP.md for beginner steps):

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from hmm_predictor import ConvergenceError, HMMPredictor, SingularCovarianceError
from regime_detection import (DISPLAY_TO_SYMBOL, INSTRUMENT_CATALOG,
                              build_features, load_prices, periods_per_year_for,
                              resolve_ticker, summarise_states)
from backtest import StrategyProfile, backtest_strategy, trades_to_frame
from indicators import ConfirmationConfig, confirmation_matrix, latest_readings
from metrics import drawdown_series, equity_curve, summary as metric_summary
from levels import (LevelConfig, compute_levels, position_size,
                    vol_scalar_from_regimes)
from walkforward import walkforward_backtest
from scanner import scan
from optimizer import optimize, DEFAULT_GRID

CONFIRMATION_LABELS = {
    "rsi_not_overbought": "RSI not overbought",
    "rsi_strength": "RSI strength (>50)",
    "macd_bullish": "MACD bullish",
    "price_above_sma": "Price above trend",
    "ma_crossover": "Fast MA > Slow MA",
    "positive_momentum": "Positive momentum",
    "adx_trending": "ADX trending",
    "volume_participation": "Volume participation",
}

st.set_page_config(page_title="REGIME TERMINAL", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# Command-center theme
# ----------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg:#0a0b0d; --panel:#0e0f12; --border:#23262b; --text:#c9ccd1;
  --muted:#6b7078; --accent:#c0392b; --accent-dim:#7e2b22; --green:#4f9d69;
  --amber:#c9962b;
}
html, body, [class*="css"], .stApp {
  background-color:var(--bg)!important; color:var(--text)!important;
  font-family:'JetBrains Mono','Share Tech Mono','Courier New',monospace!important;
}
.block-container { padding-top:1rem; max-width:1600px; }
#MainMenu, footer, header { visibility:hidden; }

.term-header { display:flex; justify-content:space-between; align-items:center;
  border-bottom:1px solid var(--border); padding-bottom:.5rem; margin-bottom:.9rem; }
.term-title { font-size:1.2rem; letter-spacing:3px; }
.term-title b { color:var(--accent); }
.term-sub { font-size:.7rem; color:var(--muted); letter-spacing:1px; }

.panel { border:1px solid var(--border); background:var(--panel); padding:.8rem .9rem; margin-bottom:.9rem; }
.panel-h { font-size:.72rem; letter-spacing:2px; text-transform:uppercase;
  border-bottom:1px solid var(--border); padding-bottom:.35rem; margin-bottom:.6rem; }
.panel-h span { color:var(--accent); }
.kv { display:flex; justify-content:space-between; font-size:.72rem; padding:.12rem 0; }
.kv .k { color:var(--muted); } .kv .v { color:var(--text); }

.signal { border:1px solid var(--border); background:var(--panel); padding:1rem 1.2rem;
  margin-bottom:.9rem; display:flex; justify-content:space-between; align-items:center; }
.signal .verdict { font-size:2rem; letter-spacing:3px; font-weight:600; }
.signal.long { border-left:5px solid var(--green); } .signal.long .verdict { color:var(--green); }
.signal.wait { border-left:5px solid var(--amber); } .signal.wait .verdict { color:var(--amber); }
.signal.stand { border-left:5px solid var(--accent); } .signal.stand .verdict { color:var(--accent); }
.signal .sub { font-size:.7rem; color:var(--muted); letter-spacing:1px; margin-top:.2rem; }
.signal .right { text-align:right; } .signal .big { font-size:1.6rem; color:var(--text); }

.statgrid { display:flex; gap:.5rem; }
.stat { flex:1; border:1px solid var(--border); padding:.4rem; text-align:center; }
.stat .n { font-size:1.4rem; } .stat .l { font-size:.6rem; color:var(--muted); letter-spacing:1px; }
.stat.bull .n { color:var(--green); } .stat.bear .n { color:var(--accent); }

.op { border:1px solid var(--border); padding:.5rem .6rem; margin-bottom:.5rem; }
.op .code { font-size:.6rem; color:var(--accent); letter-spacing:1px; }
.op .ttl { font-size:.8rem; margin:.15rem 0; }
.op .meta { font-size:.65rem; color:var(--muted); }

.chk { display:flex; justify-content:space-between; align-items:center;
  font-size:.72rem; padding:.28rem .1rem; border-bottom:1px solid #15171a; }
.chk .name { color:var(--text); } .chk .val { color:var(--muted); font-size:.66rem; margin-left:.5rem; }
.chk .mark { font-weight:600; } .chk.pass .mark { color:var(--green); } .chk.fail .mark { color:var(--accent); }

/* Levels ladder */
.lvl { display:flex; justify-content:space-between; font-size:.74rem; padding:.3rem .1rem;
  border-bottom:1px solid #15171a; }
.lvl .tag { color:var(--muted); letter-spacing:1px; }
.lvl.tgt .px { color:var(--green); } .lvl.entry .px { color:var(--text); }
.lvl.stop .px { color:var(--accent); }

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
  background:var(--panel)!important; color:var(--text)!important;
  border:1px solid var(--border)!important; border-radius:0!important;
  font-family:'JetBrains Mono',monospace!important;
}
.stSlider label, .stTextInput label, .stSelectbox label, .stNumberInput label {
  color:var(--muted)!important; font-size:.65rem!important; letter-spacing:1px;
}
.stButton button { background:var(--accent-dim)!important; color:#fff!important;
  border:1px solid var(--accent)!important; border-radius:0!important;
  font-family:'JetBrains Mono',monospace!important; letter-spacing:2px; }
.stButton button:hover { background:var(--accent)!important; }
.stDataFrame { border:1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { gap:2px; }
.stTabs [data-baseweb="tab"] { background:var(--panel); border:1px solid var(--border);
  color:var(--muted); font-family:'JetBrains Mono',monospace; letter-spacing:2px; font-size:.7rem; }
.stTabs [aria-selected="true"] { background:var(--accent-dim); color:#fff; }

.metricgrid { display:flex; flex-wrap:wrap; gap:.5rem; }
.metric { flex:1; min-width:115px; border:1px solid var(--border); background:var(--panel); padding:.5rem .6rem; }
.metric .l { font-size:.56rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }
.metric .n { font-size:1.2rem; margin-top:.2rem; }
.metric.good .n { color:var(--green); } .metric.bad .n { color:var(--accent); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="term-header">
      <div>
        <div class="term-title">REGIME <b>TERMINAL</b></div>
        <div class="term-sub">HIDDEN MARKOV MODEL // MULTI-ASSET REGIME DETECTION</div>
      </div>
      <div class="term-sub">RESEARCH BUILD &mdash; NOT FINANCIAL ADVICE</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# Shared helpers
# ======================================================================
def panel(title_html, body_html):
    return f"<div class='panel'><div class='panel-h'>{title_html}</div>{body_html}</div>"

def kv(k, v):
    return f"<div class='kv'><span class='k'>{k}</span><span class='v'>{v}</span></div>"

def metric_card(label, value, fmt="{:.2f}", good=None):
    cls = "metric" + (" good" if good is True else " bad" if good is False else "")
    val = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    return f"<div class='{cls}'><div class='l'>{label}</div><div class='n'>{val}</div></div>"


@st.cache_resource(show_spinner=False)
def fit_model(symbol: str, period: str, interval: str, n_states: int):
    df = load_prices(symbol, period=period, interval=interval)
    features = build_features(df)
    model = HMMPredictor(n_components=n_states, covariance_type="diag",
                         n_iter=300, init_method="kmeans", random_state=42)
    model.fit(features)
    states = model.predict_hidden_states(features)
    return df, features, model, states


def regime_chart(df, features, states, label, n_states):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    close = df["Close"].reindex(features.index)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.patch.set_facecolor("#0a0b0d"); ax.set_facecolor("#0a0b0d")
    ax.plot(features.index, close.values, color="#3a3e45", linewidth=0.8, zorder=1)
    sc = ax.scatter(features.index, close.values, c=states, cmap="Spectral",
                    s=7, zorder=2, vmin=0, vmax=max(n_states - 1, 1))
    ax.set_title(f"{label}  //  CLOSE PRICE BY DETECTED REGIME",
                 color="#c9ccd1", fontsize=10, loc="left", family="monospace")
    for s in ax.spines.values(): s.set_color("#23262b")
    ax.tick_params(colors="#6b7078", labelsize=7); ax.grid(color="#15171a", linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("REGIME", color="#6b7078", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#6b7078", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#6b7078")
    fig.tight_layout(); return fig


def equity_drawdown_chart(strat_ret, bench_ret, title="EQUITY CURVE (GROWTH OF 1.0)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq_s = equity_curve(strat_ret); eq_b = equity_curve(bench_ret)
    dd = drawdown_series(strat_ret)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (ax1, ax2):
        fig.patch.set_facecolor("#0a0b0d"); ax.set_facecolor("#0a0b0d")
        for s in ax.spines.values(): s.set_color("#23262b")
        ax.tick_params(colors="#6b7078", labelsize=7); ax.grid(color="#15171a", linewidth=0.5)
    ax1.plot(eq_s.index, eq_s.values, color="#4f9d69", linewidth=1.1, label="STRATEGY")
    ax1.plot(eq_b.index, eq_b.values, color="#6b7078", linewidth=0.9, label="BUY & HOLD")
    ax1.set_title(title, color="#c9ccd1", fontsize=9, loc="left", family="monospace")
    leg = ax1.legend(loc="upper left", fontsize=7, facecolor="#0e0f12", edgecolor="#23262b")
    for t in leg.get_texts(): t.set_color("#c9ccd1")
    ax2.fill_between(dd.index, dd.values, 0.0, color="#7e2b22", alpha=0.8)
    ax2.set_title("DRAWDOWN (UNDERWATER)", color="#6b7078", fontsize=8, loc="left", family="monospace")
    fig.tight_layout(); return fig


def metrics_cards_html(m, n_trades, label):
    cards = (
        metric_card("TOTAL RETURN", m["total_return"]*100, "{:+.1f}%", m["total_return"]>0)
        + metric_card("ALPHA", m.get("alpha_vs_buy_hold",0)*100, "{:+.1f}%", m.get("alpha_vs_buy_hold",0)>0)
        + metric_card("SHARPE", m["sharpe"], "{:.2f}", m["sharpe"]>1)
        + metric_card("MAX DD", m["max_drawdown"]*100, "{:.1f}%", m["max_drawdown"]>-0.25)
        + metric_card("CALMAR", m["calmar"], "{:.2f}", m["calmar"]>0.5)
        + metric_card("WIN %", m["win_rate"]*100, "{:.0f}%", m["win_rate"]>0.5)
        + metric_card("TRADES", n_trades, "{:d}"))
    return panel(label, f"<div class='metricgrid'>{cards}</div>")


# ======================================================================
# TABS
# ======================================================================
tab_analysis, tab_scanner, tab_optimizer = st.tabs(
    ["  ANALYSIS  ", "  SCANNER  ", "  OPTIMIZER  "])


# ----------------------------------------------------------------------
# TAB 1: ANALYSIS
# ----------------------------------------------------------------------
with tab_analysis:
    r1c1, r1c2, r1c3 = st.columns([1.3, 2.0, 1.6])
    with r1c1:
        category = st.selectbox("CATEGORY", list(INSTRUMENT_CATALOG.keys()), index=0)
    with r1c2:
        instrument = st.selectbox("INSTRUMENT", list(INSTRUMENT_CATALOG[category].keys()), index=0)
    with r1c3:
        custom = st.text_input("CUSTOM SYMBOL (optional)", value="",
                               placeholder="e.g. SHIB-USD, ^STOXX50E")

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1.0, 0.9, 0.9, 1.6, 1.3])
    with r2c1:
        n_states = st.slider("REGIMES", 2, 8, 7)
    with r2c2:
        interval = st.selectbox("INTERVAL", ["1h", "1d"], index=0)
    with r2c3:
        period = st.selectbox("HISTORY", ["730d", "365d", "180d", "90d"], index=0)
    with r2c4:
        profile_name = st.selectbox("PROFILE", ["CONSERVATIVE (7/8, 2.5x)",
                                                "AGGRESSIVE (5/8, 4x)"], index=0)
    with r2c5:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        run_button = st.button("RUN DETECTION", type="primary", use_container_width=True)

    r3c1, r3c2, r3c3 = st.columns([1.4, 1.4, 3.0])
    with r3c1:
        account_equity = st.number_input("ACCOUNT EQUITY ($)", min_value=100.0,
                                         value=10000.0, step=100.0)
    with r3c2:
        risk_pct = st.number_input("RISK PER TRADE (%)", min_value=0.1,
                                   max_value=10.0, value=1.0, step=0.1) / 100.0
    with r3c3:
        run_wf = st.checkbox("Also run WALK-FORWARD (honest out-of-sample, slower)",
                             value=False)

    if custom.strip():
        symbol = custom.strip(); display_name = custom.strip()
    else:
        symbol = INSTRUMENT_CATALOG[category][instrument]; display_name = instrument

    if run_button:
        try:
            with st.spinner(f"DOWNLOADING {display_name} // FITTING MODEL ..."):
                df, features, model, states = fit_model(symbol, period, interval, n_states)
        except ConvergenceError as exc:
            st.error(f"MODEL DID NOT CONVERGE: {exc}"); st.stop()
        except SingularCovarianceError as exc:
            st.error(f"COVARIANCE FAILURE: {exc}"); st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"ERROR: {exc}"); st.stop()

        summary = summarise_states(features, states)
        n_bars = len(features)
        bull_state = int(summary["mean_return"].idxmax())
        bear_state = int(summary["mean_return"].idxmin())
        bull_bars = int((states == bull_state).sum())
        bear_bars = int((states == bear_state).sum())

        profile = (StrategyProfile.aggressive() if profile_name.startswith("AGGRESSIVE")
                   else StrategyProfile.conservative())
        cfg = profile.confirmation_cfg or ConfirmationConfig()

        current_state = int(states[-1])
        proba = model.predict_state_proba(features)[-1]
        confidence = float(proba[current_state]) * 100.0
        conf_df = confirmation_matrix(df.reindex(features.index), cfg)
        conf_now = conf_df.iloc[-1]
        n_conf_now = int(conf_now["n_confirmations"])
        df_aligned = df.reindex(features.index)
        last_price = float(df_aligned["Close"].iloc[-1])

        if current_state == bull_state and n_conf_now >= profile.min_confirmations:
            verdict, vclass, vsub = ("LONG", "long", "Bull regime + enough confirmations")
        elif current_state == bull_state:
            verdict, vclass, vsub = ("WAIT", "wait", f"Bull regime, only {n_conf_now}/8 confirmations")
        elif current_state == bear_state:
            verdict, vclass, vsub = ("STAND ASIDE", "stand", "Bear / crash regime - stay out")
        else:
            verdict, vclass, vsub = ("WAIT", "wait", "Neutral regime - no edge")
        regime_tag = ("BULLISH" if current_state == bull_state
                      else "BEARISH" if current_state == bear_state else "NEUTRAL")

        st.markdown(
            f"""<div class="signal {vclass}">
              <div><div class="verdict">{verdict}</div>
                <div class="sub">{display_name.upper()} &nbsp;//&nbsp; {vsub}</div></div>
              <div class="right"><div class="big">{regime_tag} &middot; {n_conf_now}/8</div>
                <div class="sub">STATE {current_state} | CONFIDENCE {confidence:.0f}% | LAST {last_price:,.2f}</div></div>
            </div>""", unsafe_allow_html=True)

        left, center, right = st.columns([1.15, 2.2, 1.25])

        with left:
            body = (kv("INSTRUMENT", display_name) + kv("SYMBOL", resolve_ticker(symbol))
                    + kv("REGIMES", str(n_states)) + kv("INTERVAL", interval)
                    + kv("HISTORY", period) + kv("BARS", f"{n_bars:,}")
                    + kv("LOG-LIK", f"{model.score_:,.0f}")
                    + kv("CONVERGED", "YES" if model.converged_ else "NO"))
            st.markdown(panel("MODEL <span>DOSSIER</span>", body), unsafe_allow_html=True)

            act = ("<div class='statgrid'>"
                   f"<div class='stat'><div class='n'>{n_bars:,}</div><div class='l'>BARS</div></div>"
                   f"<div class='stat bull'><div class='n'>{bull_bars:,}</div><div class='l'>BULLISH</div></div>"
                   f"<div class='stat bear'><div class='n'>{bear_bars:,}</div><div class='l'>BEARISH</div></div></div>")
            st.markdown(panel("REGIME <span>ACTIVITY</span>", act), unsafe_allow_html=True)

            readings = latest_readings(df_aligned, cfg)
            rows = ""
            for key, lbl in CONFIRMATION_LABELS.items():
                ok = bool(conf_now[key]); mark = "PASS" if ok else "FAIL"
                cls = "pass" if ok else "fail"
                rows += (f"<div class='chk {cls}'><span class='name'>{lbl}"
                         f"<span class='val'>{readings.get(key,'')}</span></span>"
                         f"<span class='mark'>{mark}</span></div>")
            st.markdown(panel(f"CONFIRMATIONS <span>({n_conf_now}/8)</span>", rows),
                        unsafe_allow_html=True)

        with center:
            st.pyplot(regime_chart(df, features, states, display_name, n_states),
                      use_container_width=True)
            try:
                bt = backtest_strategy(df, features, states, profile=profile)
                ppy = periods_per_year_for(interval, symbol)
                m = metric_summary(bt["strategy_returns"], ppy,
                                   benchmark_returns=bt["benchmark_returns"])
                st.markdown(metrics_cards_html(m, len(bt["trades"]),
                            f"IN-SAMPLE BACKTEST <span>// {profile_name}</span>"),
                            unsafe_allow_html=True)
                st.pyplot(equity_drawdown_chart(bt["strategy_returns"],
                                                bt["benchmark_returns"]),
                          use_container_width=True)
                st.session_state["_trades"] = trades_to_frame(bt["trades"])
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Backtest unavailable: {exc}")
                st.session_state["_trades"] = pd.DataFrame()

            # ---- Optional WALK-FORWARD (honest out-of-sample) ----
            if run_wf:
                try:
                    with st.spinner("RUNNING WALK-FORWARD (re-fitting across folds) ..."):
                        wf = walkforward_backtest(df, features, n_states=n_states,
                                                  train_size=min(2000, max(500, n_bars//3)),
                                                  test_size=300, profile=profile)
                    ppy = periods_per_year_for(interval, symbol)
                    wm = metric_summary(wf["strategy_returns"], ppy,
                                       benchmark_returns=wf["benchmark_returns"])
                    st.markdown(metrics_cards_html(wm, len(wf["trades"]),
                                f"WALK-FORWARD (OUT-OF-SAMPLE) <span>// {wf['n_folds']} FOLDS</span>"),
                                unsafe_allow_html=True)
                    st.pyplot(equity_drawdown_chart(wf["strategy_returns"],
                                                    wf["benchmark_returns"],
                                                    title="WALK-FORWARD EQUITY (OUT-OF-SAMPLE)"),
                              use_container_width=True)
                    st.caption("This is the HONEST estimate - model never saw the bars it traded. "
                               "Trust this over the in-sample number above.")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Walk-forward unavailable: {exc}")

        with right:
            # ATR LEVELS + POSITION SIZING
            lv = compute_levels(df_aligned, LevelConfig())
            vs = vol_scalar_from_regimes(summary, current_state)
            ps = position_size(account_equity, lv, risk_pct=risk_pct,
                               max_leverage=profile.leverage, vol_scalar=vs)
            levels_html = (
                f"<div class='lvl tgt'><span class='tag'>TARGET 2 ({lv['rr_target2']:.1f}R)</span>"
                f"<span class='px'>{lv['target2']:,.2f} ({lv['target2_pct']*100:+.1f}%)</span></div>"
                f"<div class='lvl tgt'><span class='tag'>TARGET 1 ({lv['rr_target1']:.1f}R)</span>"
                f"<span class='px'>{lv['target1']:,.2f} ({lv['target1_pct']*100:+.1f}%)</span></div>"
                f"<div class='lvl entry'><span class='tag'>ENTRY</span>"
                f"<span class='px'>{lv['entry']:,.2f}</span></div>"
                f"<div class='lvl stop'><span class='tag'>STOP (2x ATR)</span>"
                f"<span class='px'>{lv['stop']:,.2f} ({lv['stop_pct']*100:.1f}%)</span></div>")
            st.markdown(panel("LEVELS <span>(ATR-BASED)</span>", levels_html),
                        unsafe_allow_html=True)

            size_html = (kv("ACCOUNT", f"${account_equity:,.0f}")
                         + kv("RISK/TRADE", f"{risk_pct*100:.1f}%  (${account_equity*risk_pct:,.0f})")
                         + kv("VOL SCALAR", f"{vs:.2f}")
                         + kv("POSITION", f"{ps['units']:.4f} units")
                         + kv("NOTIONAL", f"${ps['notional']:,.0f}")
                         + kv("LEVERAGE", f"{ps['leverage_used']:.2f}x"
                              + (" (capped)" if ps['capped'] else ""))
                         + kv("$ AT RISK", f"${ps['dollar_risk']:,.0f}"))
            st.markdown(panel("POSITION <span>SIZING</span>", size_html),
                        unsafe_allow_html=True)

            items = ""
            for sid, row in summary.sort_values("mean_return", ascending=False).iterrows():
                lab = row["label"] or "NEUTRAL REGIME"
                tag = ("BULL" if sid == bull_state else "BEAR" if sid == bear_state else "NEUTRAL")
                items += ("<div class='op'>"
                          f"<div class='code'>STATE {int(sid)} &mdash; {tag}</div>"
                          f"<div class='ttl'>{lab}</div>"
                          f"<div class='meta'>&mu; {row['mean_return']:+.5f} &nbsp; "
                          f"&sigma; {row['return_volatility']:.5f} &nbsp; n={int(row['n_obs']):,}</div></div>")
            st.markdown(panel(f"REGIME <span>LIST ({n_states})</span>", items),
                        unsafe_allow_html=True)

        trades_df = st.session_state.get("_trades", pd.DataFrame())
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            show = trades_df[["entry_time", "exit_time", "bars_held",
                              "leveraged_return_pct", "exit_reason"]].copy()
            show["leveraged_return_pct"] = (show["leveraged_return_pct"]*100).round(2)
            show = show.rename(columns={"leveraged_return_pct": "ret_%"})
            st.markdown("<div class='panel-h'>TRADE LOG</div>", unsafe_allow_html=True)
            st.dataframe(show.tail(15), use_container_width=True, height=300, hide_index=True)

        st.caption(
            "LEVELS, SIZING and the LIVE SIGNAL describe the latest bar - not a "
            "recommendation. In-sample backtest is optimistic; tick walk-forward "
            "for the honest number. Research only - not financial advice."
        )
    else:
        st.markdown(
            "<div class='panel'><div class='panel-h'>STANDBY</div>"
            "<div class='kv'><span class='k'>Pick CATEGORY + INSTRUMENT, set your "
            "account/risk, then press <b style='color:#c0392b'>RUN DETECTION</b>.</span></div></div>",
            unsafe_allow_html=True)


# ----------------------------------------------------------------------
# TAB 2: SCANNER
# ----------------------------------------------------------------------
with tab_scanner:
    st.markdown("<div class='term-sub'>Scan every instrument and rank by current "
                "signal. LONG first. (Fits one model per instrument - takes a "
                "minute.)</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([1.4, 0.9, 0.9, 1.4])
    with s1:
        scan_category = st.selectbox("SCOPE",
                                     ["ALL"] + list(INSTRUMENT_CATALOG.keys()), index=0,
                                     key="scan_scope")
    with s2:
        scan_states = st.slider("REGIMES", 2, 8, 7, key="scan_states")
    with s3:
        scan_interval = st.selectbox("INTERVAL", ["1h", "1d"], index=0, key="scan_interval")
    with s4:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        scan_button = st.button("RUN SCANNER", type="primary", use_container_width=True)

    if scan_button:
        instruments = (DISPLAY_TO_SYMBOL if scan_category == "ALL"
                       else INSTRUMENT_CATALOG[scan_category])
        prog = st.progress(0.0, text="Scanning...")
        def _cb(done, total, name):
            prog.progress(done/total, text=f"Scanning {name} ({done}/{total})")
        try:
            result = scan(instruments, n_states=scan_states, period="365d",
                          interval=scan_interval,
                          profile=StrategyProfile.conservative(), progress_cb=_cb)
            prog.empty()
            if result.empty:
                st.warning("No results.")
            else:
                longs = int((result["verdict"] == "LONG").sum())
                waits = int((result["verdict"] == "WAIT").sum())
                stands = int((result["verdict"] == "STAND ASIDE").sum())
                cards = (metric_card("LONG", longs, "{:d}", longs>0)
                         + metric_card("WAIT", waits, "{:d}")
                         + metric_card("STAND ASIDE", stands, "{:d}", False if stands else None)
                         + metric_card("SCANNED", len(result), "{:d}"))
                st.markdown(f"<div class='metricgrid'>{cards}</div>", unsafe_allow_html=True)
                disp = result[["instrument", "verdict", "regime", "confirmations",
                               "confidence", "last_price", "status"]].copy()
                disp["confidence"] = disp["confidence"].round(0)
                disp["last_price"] = disp["last_price"].round(2)
                st.dataframe(disp, use_container_width=True, height=560, hide_index=True)
                st.caption("Ranked: LONG first, then by confirmation count and confidence. "
                           "Research only - not financial advice.")
        except Exception as exc:  # noqa: BLE001
            prog.empty(); st.error(f"Scanner error: {exc}")
    else:
        st.markdown("<div class='panel'><div class='panel-h'>STANDBY</div>"
                    "<div class='kv'><span class='k'>Press <b style='color:#c0392b'>"
                    "RUN SCANNER</b> to scan all instruments.</span></div></div>",
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------
# TAB 3: OPTIMIZER
# ----------------------------------------------------------------------
with tab_optimizer:
    st.markdown("<div class='term-sub'>Grid-search strategy parameters, scored by "
                "WALK-FORWARD (out-of-sample) performance - not in-sample. The "
                "winner is the one that generalised, not the one that memorised.</div>",
                unsafe_allow_html=True)
    o1, o2, o3, o4, o5 = st.columns([1.3, 1.7, 0.9, 0.9, 1.2])
    with o1:
        opt_category = st.selectbox("CATEGORY", list(INSTRUMENT_CATALOG.keys()),
                                    index=0, key="opt_cat")
    with o2:
        opt_instrument = st.selectbox("INSTRUMENT",
                                      list(INSTRUMENT_CATALOG[opt_category].keys()),
                                      index=0, key="opt_inst")
    with o3:
        opt_states = st.slider("REGIMES", 2, 8, 7, key="opt_states")
    with o4:
        opt_objective = st.selectbox("RANK BY", ["sharpe", "calmar", "total_return"],
                                     index=0, key="opt_obj")
    with o5:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        opt_button = st.button("RUN OPTIMIZER", type="primary", use_container_width=True)

    st.markdown(f"<div class='term-sub'>Search grid: min_confirmations "
                f"{DEFAULT_GRID['min_confirmations']}, leverage "
                f"{DEFAULT_GRID['leverage']}, adx_min {DEFAULT_GRID['adx_min']}, "
                f"rsi_max {DEFAULT_GRID['rsi_max']} "
                f"({len(DEFAULT_GRID['min_confirmations'])*len(DEFAULT_GRID['leverage'])*len(DEFAULT_GRID['adx_min'])*len(DEFAULT_GRID['rsi_max'])} "
                f"combinations).</div>", unsafe_allow_html=True)

    if opt_button:
        opt_symbol = INSTRUMENT_CATALOG[opt_category][opt_instrument]
        try:
            with st.spinner(f"Loading {opt_instrument} ..."):
                df_o = load_prices(opt_symbol, period="730d", interval="1h")
                feats_o = build_features(df_o)
            prog = st.progress(0.0, text="Optimizing...")
            def _ocb(done, total, label):
                prog.progress(done/total, text=f"[{done}/{total}] {label}")
            res = optimize(df_o, feats_o, periods_per_year_for("1h", opt_symbol),
                           n_states=opt_states, train_size=2000, test_size=400,
                           objective=opt_objective, progress_cb=_ocb)
            prog.empty()
            if res.empty:
                st.warning("No optimizer results.")
            else:
                best = res.iloc[0]
                cards = (metric_card("BEST SHARPE", best["sharpe"], "{:.2f}", best["sharpe"]>1)
                         + metric_card("MAX DD", best["max_drawdown"]*100, "{:.1f}%", best["max_drawdown"]>-0.25)
                         + metric_card("RETURN", best["total_return"]*100, "{:+.1f}%", best["total_return"]>0)
                         + metric_card("MIN CONF", int(best["min_confirmations"]), "{:d}")
                         + metric_card("LEVERAGE", best["leverage"], "{:.1f}x")
                         + metric_card("ADX MIN", best["adx_min"], "{:.0f}"))
                st.markdown(panel("BEST PARAMETERS <span>(BY WALK-FORWARD)</span>",
                                  f"<div class='metricgrid'>{cards}</div>"),
                            unsafe_allow_html=True)
                show = res.copy()
                for c in ["total_return", "max_drawdown", "win_rate"]:
                    if c in show: show[c] = (show[c]*100).round(1)
                for c in ["sharpe", "calmar"]:
                    if c in show: show[c] = show[c].round(2)
                st.dataframe(show, use_container_width=True, height=460, hide_index=True)
                st.caption("Every row scored out-of-sample via walk-forward. "
                           "Even the best may be negative - that is honest. "
                           "Research only - not financial advice.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optimizer error: {exc}")
    else:
        st.markdown("<div class='panel'><div class='panel-h'>STANDBY</div>"
                    "<div class='kv'><span class='k'>Pick an instrument and press "
                    "<b style='color:#c0392b'>RUN OPTIMIZER</b>. This re-fits the model "
                    "many times - give it a minute.</span></div></div>",
                    unsafe_allow_html=True)
