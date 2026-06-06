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
import discord_auth

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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Playfair+Display:ital,wght@1,700;1,800;1,900&display=swap');
:root {
  --bg:#050507; --panel:rgba(20,21,26,0.55); --panel-solid:#101116;
  --border:rgba(255,255,255,0.08); --border-strong:rgba(255,255,255,0.14);
  --text:#e7e9ee; --muted:#8a8f9a; --accent:#6e8bff; --accent2:#9b7bff;
  --green:#3ddc97; --red:#ff5c72; --amber:#ffc24b;
  --mono:'JetBrains Mono','SF Mono','Menlo','Consolas',monospace;
  --display:'Playfair Display','Times New Roman',serif;
}
html, body, [class*="css"], .stApp {
  background:
    radial-gradient(1100px 520px at 78% -8%, rgba(110,139,255,0.12), transparent 60%),
    radial-gradient(900px 520px at 12% 8%, rgba(155,123,255,0.10), transparent 55%),
    var(--bg) !important;
  color:var(--text)!important;
  font-family:var(--mono)!important;
}
.block-container { padding-top:1.2rem; max-width:1620px; }
#MainMenu, footer, header { visibility:hidden; }

/* Top nav bar (Cryptoverse-style) */
.navbar { display:flex; justify-content:space-between; align-items:center;
  padding:.2rem .2rem 1rem; margin-bottom:1.4rem; border-bottom:1px solid var(--border); }
.nav-left { display:flex; align-items:center; gap:2.2rem; }
.brand { font-size:1.18rem; font-weight:800; letter-spacing:-.3px;
  display:flex; align-items:center; gap:.55rem; }
.brand .logo { width:22px; height:22px; display:inline-flex; align-items:center;
  filter:drop-shadow(0 0 10px rgba(110,139,255,0.6)); }
.brand .logo svg { width:22px; height:22px; }
.nav-links { display:flex; gap:1.5rem; }
.nav-links a { color:var(--muted); font-size:.82rem; font-weight:500; text-decoration:none; }
.nav-links a.active, .nav-links a:hover { color:var(--text); }
.nav-right { display:flex; align-items:center; gap:1.1rem; }
.nav-social { display:flex; gap:.85rem; color:var(--muted); font-size:.8rem; }
.nav-social span { opacity:.8; }
.nav-cta { font-size:.8rem; font-weight:600; color:#fff; padding:.5rem 1.15rem;
  border-radius:999px; background:rgba(255,255,255,0.08);
  border:1px solid var(--border-strong); }

/* Giant silver-gradient hero heading */
.hero { text-align:center; padding:2.2rem 0 .6rem; position:relative; }
.hero h1 { font-family:var(--display); font-style:italic; font-size:8rem; line-height:.92;
  font-weight:900; letter-spacing:-3px; margin:0;
  background:linear-gradient(180deg,#ffffff 0%,#dfe2eb 50%,#9aa0ad 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  filter:drop-shadow(0 10px 40px rgba(110,139,255,0.18)); }
.hero .tagline { font-size:1rem; font-weight:500; margin-top:.4rem; color:var(--muted);
  letter-spacing:1px; text-transform:uppercase; }
.hero .lead { color:var(--muted); font-size:.9rem; max-width:560px; margin:.8rem auto 0; line-height:1.5; }
.term-sub { font-size:.68rem; color:var(--muted); letter-spacing:1.5px;
  text-transform:uppercase; font-weight:500; }
.term-badge { font-size:.6rem; color:var(--amber); letter-spacing:1.5px;
  border:1px solid rgba(255,194,75,0.35); border-radius:999px; padding:.3rem .7rem;
  background:rgba(255,194,75,0.06); }

/* Footer (Cryptoverse-style overlapping card) */
.footer { border:1px solid var(--border); border-radius:24px; margin-top:2rem;
  padding:2rem 2.2rem 1.4rem; background:var(--panel); position:relative; overflow:hidden;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); }
.footer:before { content:''; position:absolute; right:-10%; top:-40%; width:60%; height:180%;
  background:radial-gradient(circle at 70% 50%, rgba(110,139,255,0.18), transparent 60%);
  pointer-events:none; }
.footer .f-cta { display:flex; justify-content:space-between; align-items:flex-start;
  gap:1rem; padding-bottom:1.6rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem; }
.footer .f-cta h2 { font-size:1.9rem; font-weight:800; margin:0; max-width:480px; letter-spacing:-.5px; }
.footer .f-cta p { color:var(--muted); font-size:.82rem; margin-top:.6rem; max-width:480px; }
.footer .f-pill { font-size:.82rem; font-weight:600; color:#fff; padding:.6rem 1.3rem;
  border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid var(--border-strong);
  white-space:nowrap; }
.footer .f-cols { display:flex; flex-wrap:wrap; gap:2.5rem; }
.footer .f-col h4 { font-size:.66rem; letter-spacing:1px; text-transform:uppercase;
  color:var(--muted); margin:0 0 .7rem; font-weight:600; }
.footer .f-col a { display:block; color:var(--text); font-size:.8rem; text-decoration:none;
  margin-bottom:.45rem; opacity:.85; }
.footer .f-col a:hover { opacity:1; }
.footer .f-bottom { color:var(--muted); font-size:.72rem; margin-top:1.6rem;
  padding-top:1.1rem; border-top:1px solid var(--border); }

/* Glass panels */
.panel { border:1px solid var(--border); background:var(--panel);
  border-radius:18px; padding:1rem 1.1rem; margin-bottom:1rem;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 8px 30px rgba(0,0,0,0.35); }
.panel-h { font-size:.66rem; letter-spacing:1.6px; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding-bottom:.55rem; margin-bottom:.7rem;
  border-bottom:1px solid var(--border); }
.panel-h span { background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.kv { display:flex; justify-content:space-between; font-size:.76rem; padding:.2rem 0; }
.kv .k { color:var(--muted); } .kv .v { color:var(--text); font-weight:500; font-variant-numeric:tabular-nums; }

/* Live signal banner */
.signal { border:1px solid var(--border); border-radius:20px; padding:1.3rem 1.5rem;
  margin-bottom:1.1rem; display:flex; justify-content:space-between; align-items:center;
  background:var(--panel); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 40px rgba(0,0,0,0.4); position:relative; overflow:hidden; }
.signal:before { content:''; position:absolute; inset:0; opacity:.16; pointer-events:none; }
.signal .verdict { font-size:2.3rem; letter-spacing:-.5px; font-weight:800; }
.signal.long:before { background:radial-gradient(600px 200px at 0% 50%, var(--green), transparent 70%); }
.signal.long { border-color:rgba(61,220,151,0.4); } .signal.long .verdict { color:var(--green); }
.signal.wait:before { background:radial-gradient(600px 200px at 0% 50%, var(--amber), transparent 70%); }
.signal.wait { border-color:rgba(255,194,75,0.4); } .signal.wait .verdict { color:var(--amber); }
.signal.stand:before { background:radial-gradient(600px 200px at 0% 50%, var(--red), transparent 70%); }
.signal.stand { border-color:rgba(255,92,114,0.4); } .signal.stand .verdict { color:var(--red); }
.signal .sub { font-size:.72rem; color:var(--muted); letter-spacing:.5px; margin-top:.25rem; }
.signal .right { text-align:right; } .signal .big { font-size:1.5rem; color:var(--text); font-weight:700; }

/* Stat boxes */
.statgrid { display:flex; gap:.6rem; }
.stat { flex:1; border:1px solid var(--border); border-radius:14px; padding:.6rem .4rem;
  text-align:center; background:rgba(255,255,255,0.02); }
.stat .n { font-size:1.5rem; font-weight:700; font-variant-numeric:tabular-nums; }
.stat .l { font-size:.58rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; margin-top:.15rem; }
.stat.bull .n { color:var(--green); } .stat.bear .n { color:var(--red); }

/* Regime list cards */
.op { border:1px solid var(--border); border-radius:14px; padding:.6rem .75rem;
  margin-bottom:.55rem; background:rgba(255,255,255,0.02); transition:.15s; }
.op:hover { border-color:var(--border-strong); background:rgba(255,255,255,0.04); }
.op .code { font-size:.58rem; color:var(--accent); letter-spacing:1px; font-weight:600; }
.op .ttl { font-size:.82rem; margin:.18rem 0; font-weight:600; }
.op .meta { font-size:.66rem; color:var(--muted); font-variant-numeric:tabular-nums; }

/* Confirmation checklist */
.chk { display:flex; justify-content:space-between; align-items:center;
  font-size:.74rem; padding:.36rem .1rem; border-bottom:1px solid var(--border); }
.chk .name { color:var(--text); } .chk .val { color:var(--muted); font-size:.66rem; margin-left:.5rem; font-variant-numeric:tabular-nums; }
.chk .mark { font-weight:700; font-size:.64rem; padding:.16rem .5rem; border-radius:999px; }
.chk.pass .mark { color:var(--green); background:rgba(61,220,151,0.12); }
.chk.fail .mark { color:var(--red); background:rgba(255,92,114,0.12); }

/* Levels ladder */
.lvl { display:flex; justify-content:space-between; font-size:.78rem; padding:.36rem .1rem;
  border-bottom:1px solid var(--border); font-variant-numeric:tabular-nums; }
.lvl .tag { color:var(--muted); letter-spacing:.5px; }
.lvl.tgt .px { color:var(--green); font-weight:600; } .lvl.entry .px { color:var(--text); font-weight:700; }
.lvl.stop .px { color:var(--red); font-weight:600; }

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
  background:var(--panel-solid)!important; color:var(--text)!important;
  border:1px solid var(--border)!important; border-radius:12px!important;
  font-family:var(--mono)!important;
}
.stSlider label, .stTextInput label, .stSelectbox label, .stNumberInput label {
  color:var(--muted)!important; font-size:.64rem!important; letter-spacing:1px;
  text-transform:uppercase; font-weight:600;
}
.stButton button { background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  color:#fff!important; border:none!important; border-radius:999px!important;
  font-family:var(--mono)!important; font-weight:700!important; letter-spacing:.5px;
  box-shadow:0 6px 20px rgba(110,139,255,0.35); }
.stButton button:hover { filter:brightness(1.1); box-shadow:0 8px 26px rgba(110,139,255,0.5); }
.stDataFrame { border:1px solid var(--border); border-radius:14px; overflow:hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:none; }
.stTabs [data-baseweb="tab"] { background:var(--panel); border:1px solid var(--border);
  border-radius:12px; color:var(--muted); font-family:var(--mono); font-weight:600;
  letter-spacing:1px; font-size:.7rem; padding:.2rem 1rem; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff; border-color:transparent; }

/* Metric cards */
.metricgrid { display:flex; flex-wrap:wrap; gap:.6rem; }
.metric { flex:1; min-width:120px; border:1px solid var(--border); border-radius:14px;
  background:rgba(255,255,255,0.02); padding:.65rem .8rem; }
.metric .l { font-size:.56rem; color:var(--muted); letter-spacing:1px; text-transform:uppercase; font-weight:600; }
.metric .n { font-size:1.35rem; margin-top:.25rem; font-weight:700; font-variant-numeric:tabular-nums; }
.metric.good .n { color:var(--green); } .metric.bad .n { color:var(--red); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

def _make_particle_sphere_svg(size: int = 64, n_dots: int = 320,
                              seed: int = 7) -> str:
    """
    Build a 'stippled globe' logo: white particles scattered inside a circle,
    denser near the rim and along the equator/poles so it reads as a sphere,
    matching the reference image. Pure SVG (no JS), deterministic via seed.
    """
    import math
    import random as _r
    rng = _r.Random(seed)
    c = size / 2.0
    R = c * 0.92
    dots = []
    for _ in range(n_dots):
        # bias radius toward the rim (sqrt pushes points outward)
        rad = R * (rng.random() ** 0.62)
        ang = rng.random() * 2 * math.pi
        x = c + rad * math.cos(ang)
        y = c + rad * math.sin(ang)
        # latitude shading: thin out the mid-band a touch for a 3D feel
        lat = abs(y - c) / R
        if rng.random() > 0.35 + 0.65 * lat and rad < R * 0.55:
            continue
        rr = rng.choice([0.5, 0.6, 0.7, 0.9])
        op = rng.choice([0.55, 0.7, 0.85, 1.0])
        dots.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{rr}' "
            f"fill='#ffffff' fill-opacity='{op}'/>")
    return (
        f"<svg viewBox='0 0 {size} {size}' xmlns='http://www.w3.org/2000/svg'>"
        f"<defs><radialGradient id='glow' cx='50%' cy='42%' r='60%'>"
        f"<stop offset='0' stop-color='#6e8bff' stop-opacity='0.18'/>"
        f"<stop offset='1' stop-color='#6e8bff' stop-opacity='0'/>"
        f"</radialGradient></defs>"
        f"<circle cx='{c}' cy='{c}' r='{R}' fill='url(#glow)'/>"
        + "".join(dots) + "</svg>"
    )


LOGO_SVG = _make_particle_sphere_svg()

st.markdown(
    f"""
    <div class="navbar">
      <div class="nav-left">
        <div class="brand"><span class="logo">{LOGO_SVG}</span>Regime Terminal</div>
        <div class="nav-links">
          <a class="active">Analysis</a>
          <a>Scanner</a>
          <a>Optimizer</a>
        </div>
      </div>
      <div class="nav-right">
        <a class="nav-cta" href="https://discord.gg/MSXdaexYdH" target="_blank"
           style="text-decoration:none;">Join Discord &rarr;</a>
      </div>
    </div>
    <div class="hero">
      <h1>regime</h1>
      <div class="tagline">// detect the market's hidden state, today.</div>
      <div class="lead">A Hidden Markov Model engine that classifies market regimes
      across 43 instruments &mdash; crypto, metals, forex, indices, commodities and
      stocks. Research build &mdash; not financial advice.</div>
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
    fig.patch.set_facecolor("#050507"); ax.set_facecolor("#050507")
    ax.plot(features.index, close.values, color="#3a3e4a", linewidth=0.8, zorder=1)
    sc = ax.scatter(features.index, close.values, c=states, cmap="turbo",
                    s=7, zorder=2, vmin=0, vmax=max(n_states - 1, 1))
    ax.set_title(f"{label}  ·  CLOSE PRICE BY DETECTED REGIME",
                 color="#e7e9ee", fontsize=10, loc="left", family="sans-serif")
    for s in ax.spines.values(): s.set_color("#23262b")
    ax.tick_params(colors="#8a8f9a", labelsize=7); ax.grid(color="#15171a", linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("REGIME", color="#8a8f9a", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#8a8f9a", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8a8f9a")
    fig.tight_layout(); return fig


def equity_drawdown_chart(strat_ret, bench_ret, title="EQUITY CURVE (GROWTH OF 1.0)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq_s = equity_curve(strat_ret); eq_b = equity_curve(bench_ret)
    dd = drawdown_series(strat_ret)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (ax1, ax2):
        fig.patch.set_facecolor("#050507"); ax.set_facecolor("#050507")
        for s in ax.spines.values(): s.set_color("#23262b")
        ax.tick_params(colors="#8a8f9a", labelsize=7); ax.grid(color="#15171a", linewidth=0.5)
    ax1.plot(eq_s.index, eq_s.values, color="#3ddc97", linewidth=1.3, label="STRATEGY")
    ax1.plot(eq_b.index, eq_b.values, color="#8a8f9a", linewidth=0.9, label="BUY & HOLD")
    ax1.set_title(title, color="#e7e9ee", fontsize=9, loc="left", family="sans-serif")
    leg = ax1.legend(loc="upper left", fontsize=7, facecolor="#101116", edgecolor="#23262b")
    for t in leg.get_texts(): t.set_color("#e7e9ee")
    ax2.fill_between(dd.index, dd.values, 0.0, color="#ff5c72", alpha=0.65)
    ax2.set_title("DRAWDOWN (UNDERWATER)", color="#8a8f9a", fontsize=8, loc="left", family="sans-serif")
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
# DISCORD GATE - only members of the configured server can use the site.
# ======================================================================
def _login_screen(authorize_url: str, invite_url: str) -> None:
    """Themed Discord login screen."""
    st.markdown(
        f"""
        <div class='panel' style='max-width:560px; margin:2rem auto; text-align:center;
             padding:2.4rem 2rem;'>
          <div style='font-size:.62rem; letter-spacing:2px; color:var(--accent);
               text-transform:uppercase; font-weight:600;'>members only</div>
          <h2 style='font-family:var(--display); font-style:italic; font-size:3.6rem;
               margin:.4rem 0 .6rem; font-weight:900;
               background:linear-gradient(180deg,#fff 0%,#dfe2eb 50%,#9aa0ad 100%);
               -webkit-background-clip:text; background-clip:text;
               -webkit-text-fill-color:transparent;'>access</h2>
          <p style='color:var(--muted); font-size:.82rem; line-height:1.55; margin-bottom:1.4rem;'>
          The terminal is gated to members of our Discord. Sign in with Discord to verify
          your membership and continue. Research only &mdash; not financial advice.</p>
          <a href='{authorize_url}' style='display:inline-block; text-decoration:none;
             color:#fff; padding:.75rem 1.6rem; border-radius:999px;
             background:linear-gradient(135deg,#5865F2,#7289da);
             box-shadow:0 6px 22px rgba(88,101,242,0.45); font-weight:700; letter-spacing:.5px;'>
             Login with Discord &rarr;</a>
          <div style='margin-top:1.1rem; font-size:.72rem; color:var(--muted);'>
            Not a member yet? <a href='{invite_url}' target='_blank'
              style='color:var(--accent);'>Join the server</a>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _denied_screen(user, invite_url: str) -> None:
    """Themed denied screen for users who aren't in the server."""
    name = discord_auth.display_name(user)
    av = discord_auth.avatar_url(user, 96) or ""
    avatar_html = (f"<img src='{av}' style='width:64px;height:64px;border-radius:50%;"
                   f"border:1px solid var(--border);margin-bottom:.8rem;'/>") if av else ""
    st.markdown(
        f"""
        <div class='panel' style='max-width:560px; margin:2rem auto; text-align:center;
             padding:2.4rem 2rem;'>
          {avatar_html}
          <div style='font-size:.62rem; letter-spacing:2px; color:var(--red);
               text-transform:uppercase; font-weight:600;'>access denied</div>
          <h2 style='font-family:var(--display); font-style:italic; font-size:2.6rem;
               margin:.4rem 0 .6rem; font-weight:900; color:#fff;'>hi, {name}.</h2>
          <p style='color:var(--muted); font-size:.82rem; line-height:1.55; margin-bottom:1.4rem;'>
          You're signed in with Discord, but you're not in our server yet. Join the
          community and refresh this page to continue.</p>
          <a href='{invite_url}' target='_blank' style='display:inline-block;
             text-decoration:none; color:#fff; padding:.75rem 1.6rem; border-radius:999px;
             background:linear-gradient(135deg,#5865F2,#7289da);
             box-shadow:0 6px 22px rgba(88,101,242,0.45); font-weight:700; letter-spacing:.5px;'>
             Join the Discord &rarr;</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


_user = discord_auth.require_login(render_login=_login_screen,
                                   render_denied=_denied_screen)
if _user is not None:
    _name = discord_auth.display_name(_user)
    _av = discord_auth.avatar_url(_user, 32)
    _avatar_html = (f"<img src='{_av}' style='width:18px;height:18px;border-radius:50%;"
                    f"vertical-align:middle;margin-right:.4rem;'/>") if _av else ""
    st.markdown(
        f"<div style='text-align:right;font-size:.72rem;color:var(--muted);"
        f"margin:-.4rem 0 .4rem;'>{_avatar_html}signed in as <span style='color:var(--text);'>"
        f"{_name}</span> &middot; verified Discord member</div>",
        unsafe_allow_html=True,
    )


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



# ======================================================================
# FOOTER (Cryptoverse-style)
# ======================================================================
st.markdown(
    f"""
    <div class="footer">
      <div class="f-cta">
        <div>
          <h2>Read the market's hidden regime, today.</h2>
          <p>An open-source Hidden Markov Model engine for regime detection,
          backtesting and signal scanning across 43 instruments. Research and
          education only &mdash; not financial advice.</p>
        </div>
        <a class="f-pill" href="https://discord.gg/MSXdaexYdH" target="_blank"
           style="text-decoration:none;">Join Discord &rarr;</a>
      </div>
      <div class="f-cols">
        <div class="f-col">
          <div class="brand" style="margin-bottom:.6rem;"><span class="logo">{LOGO_SVG}</span>Regime Terminal</div>
          <p style="color:var(--muted); font-size:.78rem; max-width:220px;">
          HMM-based market regime detection &amp; strategy research toolkit.</p>
        </div>
        <div class="f-col">
          <h4>Tools</h4>
          <a>Analysis</a><a>Scanner</a><a>Optimizer</a><a>Walk-forward</a><a>Position sizing</a>
        </div>
        <div class="f-col">
          <h4>Instruments</h4>
          <a>Crypto</a><a>Metals</a><a>Forex</a><a>Indices</a><a>Stocks</a>
        </div>
        <div class="f-col">
          <h4>Community</h4>
          <a href="https://discord.gg/MSXdaexYdH" target="_blank">Discord server</a>
          <a>Discord: lucii_aaa</a>
          <a href="https://www.tiktok.com/@nogodrai" target="_blank">TikTok: @nogodrai</a>
          <a href="https://www.instagram.com/lucii.flow" target="_blank">Instagram: lucii.flow</a>
        </div>
      </div>
      <div class="f-bottom">
        &copy; 2026 Regime Terminal &middot;
        Join the <a href="https://discord.gg/MSXdaexYdH" target="_blank" style="color:var(--accent);">Discord</a>
        &middot; Research only &mdash; not financial advice.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
