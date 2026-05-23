"""
Forward Forecast — Expectations Investing Engine
Streamlit UI
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import fetch_and_solve

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forward Forecast | Reverse DCF",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0d0d12; }
  .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

  .hero-title {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -1px;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 2rem; }

  .kpi-card {
    background: #13131a; border: 1px solid #1e1e2e;
    border-radius: 12px; padding: 1.1rem 1.4rem;
  }
  .kpi-label { color: #64748b; font-size: 0.72rem; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; }
  .kpi-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700;
                margin-top: 0.15rem; }
  .kpi-sub   { color: #475569; font-size: 0.78rem; margin-top: 0.1rem; }

  .verdict-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #3730a3; border-radius: 14px;
    padding: 1.8rem 2rem; margin: 1.5rem 0;
  }
  .verdict-label { color: #818cf8; font-size: 0.75rem; font-weight: 700;
                    letter-spacing: 0.12em; text-transform: uppercase; }
  .verdict-value { font-size: 3.4rem; font-weight: 800; letter-spacing: -2px;
                    background: linear-gradient(90deg, #818cf8, #c084fc);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .verdict-desc  { color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem; }

  .section-title {
    color: #94a3b8; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    border-bottom: 1px solid #1e1e2e; padding-bottom: 0.5rem;
    margin: 1.8rem 0 1rem 0;
  }

  div[data-testid="stDataFrame"] thead th {
    background: #13131a !important; color: #64748b !important;
    font-size: 0.75rem !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 8px;
    padding: 0.55rem 1.8rem; font-weight: 600;
    font-size: 0.95rem; width: 100%; transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  .stTextInput > div > div > input {
    background: #13131a; border: 1px solid #1e1e2e;
    color: #e2e8f0; border-radius: 8px;
    font-size: 1.05rem; font-weight: 600; text-transform: uppercase;
    padding: 0.55rem 0.9rem;
  }

  .tag {
    display: inline-block; background: #1e1e2e; color: #64748b;
    border-radius: 6px; padding: 0.15rem 0.6rem;
    font-size: 0.72rem; font-weight: 600; margin-right: 0.4rem;
  }

  .signal-bullish  { color: #34d399; font-weight: 700; }
  .signal-bearish  { color: #f87171; font-weight: 700; }
  .signal-neutral  { color: #fbbf24; font-weight: 700; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(v, decimals=1):
    if v is None or np.isnan(v): return "N/A"
    return f"{v*100:+.{decimals}f}%"

def fmt_pct_plain(v, decimals=1):
    if v is None or np.isnan(v): return "N/A"
    return f"{v*100:.{decimals}f}%"

def fmt_bn(v):
    if v is None or np.isnan(v): return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_price(v):
    if v is None or np.isnan(v): return "N/A"
    return f"${v:,.2f}"

def signal_class(implied, hist):
    """Return CSS class based on implied vs historical CAGR."""
    if np.isnan(implied) or np.isnan(hist): return "signal-neutral"
    gap = implied - hist
    if gap > 0.05:  return "signal-bearish"   # market expects much more than history
    if gap < -0.05: return "signal-bullish"    # market is pricing in a slowdown
    return "signal-neutral"

def _is_valid(v):
    return v is not None and not np.isnan(v)

def verdict_text(implied, cagr3, cagr5):
    if not _is_valid(implied): return ""
    refs = [x for x in [cagr3, cagr5] if _is_valid(x)]
    if not refs: return "Insufficient history to benchmark."
    avg = np.mean(refs)
    gap = implied - avg
    if gap > 0.08:
        return "Market is pricing in a significant growth acceleration. High expectations embedded in the current price."
    if gap > 0.03:
        return "Market expects moderate growth improvement over recent history."
    if gap < -0.08:
        return "Market is pricing in material growth deceleration vs. history. Potential value or distress."
    if gap < -0.03:
        return "Implied growth trails historical pace — cautious pricing."
    return "Implied growth is broadly in-line with historical trajectory."


# ── sensitivity chart ─────────────────────────────────────────────────────────

def build_sensitivity_chart(labels, values, base_margin):
    colors = []
    for i, v in enumerate(values):
        if np.isnan(v):
            colors.append("#374151")
        elif labels[i] == "Base TTM":
            colors.append("#818cf8")
        elif v > 0:
            colors.append("#34d399" if v < 0.15 else "#f87171")
        else:
            colors.append("#94a3b8")

    y_vals = [v*100 if not np.isnan(v) else 0 for v in values]
    text_vals = [fmt_pct(v) for v in values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=y_vals,
        text=text_vals,
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(
            title="Required 7-yr Revenue CAGR (%)",
            gridcolor="#1e1e2e", zeroline=True,
            zerolinecolor="#374151", ticksuffix="%",
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=300,
        showlegend=False,
    )
    return fig


def build_cagr_comparison_chart(implied, cagr3, cagr5):
    categories = ["Implied\n(Market)", "3-Year\nActual", "5-Year\nActual"]
    vals       = [implied, cagr3, cagr5]
    cols       = ["#818cf8", "#34d399", "#22d3ee"]
    y_vals     = [v*100 if (v is not None and not np.isnan(v)) else None for v in vals]
    text_vals  = [fmt_pct(v) for v in vals]

    fig = go.Figure()
    for cat, yv, tv, col in zip(categories, y_vals, text_vals, cols):
        if yv is not None:
            fig.add_trace(go.Bar(
                x=[cat], y=[yv], text=[tv],
                textposition="outside",
                marker_color=col, marker_line_width=0,
                showlegend=False,
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(gridcolor="#1e1e2e", zeroline=True,
                   zerolinecolor="#374151", ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=300,
        showlegend=False,
        barmode="group",
    )
    return fig


# ── main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Forward Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Institutional-Grade Expectations Investing · Reverse DCF Engine</div>',
            unsafe_allow_html=True)

col_input, col_btn, col_space = st.columns([2, 1, 4])
with col_input:
    ticker_input = st.text_input("", placeholder="Enter ticker (e.g. AAPL, NVDA, MSFT)",
                                  label_visibility="collapsed")
with col_btn:
    st.markdown("<div style='margin-top:0.15rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("Analyze")

if run_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    with st.spinner(f"Fetching live data and solving Reverse DCF for **{ticker}** …"):
        try:
            r = fetch_and_solve(ticker)
        except Exception as e:
            st.error(f"**Error:** {e}")
            st.stop()

    # ── company header ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='margin:1rem 0 0.3rem 0'>
      <span style='color:#e2e8f0;font-size:1.3rem;font-weight:700'>{r['company']}</span>
      &nbsp;
      <span class='tag'>{r['ticker']}</span>
      <span class='tag'>{r['sector']}</span>
      <span class='tag'>{r['industry']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, label, value, sub=""):
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{value}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Stock Price",      fmt_price(r["price"]),     "Current")
    kpi(k2, "Market Cap",       fmt_bn(r["mkt_cap"]),      "")
    kpi(k3, "Enterprise Value", fmt_bn(r["ev"]),           "Target for model")
    kpi(k4, "TTM Revenue",      fmt_bn(r["ttm_rev"]),      "")
    kpi(k5, "TTM Op. Margin",   fmt_pct_plain(r["op_margin"]), "EBIT / Revenue")
    kpi(k6, "WACC",             fmt_pct_plain(r["wacc"]),  f"Beta {r['beta']:.2f}")

    # ── verdict box ───────────────────────────────────────────────────────
    ig = r["implied_g"]
    st.markdown(f"""
    <div class='verdict-box'>
      <div class='verdict-label'>Core Market Verdict · Implied Revenue CAGR (7-Year Horizon)</div>
      <div class='verdict-value'>{fmt_pct(ig)}</div>
      <div class='verdict-desc'>
        The market is pricing <strong>{fmt_pct(ig)}</strong> annualised revenue growth
        over the next 7 years into <strong>{r['company']}</strong>'s current share price
        at a <strong>{fmt_pct_plain(r['wacc'])}</strong> WACC.
        &nbsp;·&nbsp; S2C Ratio: <strong>{r['s2c']:.2f}x</strong>
        &nbsp;·&nbsp; Tax Rate: <strong>{fmt_pct_plain(r['tax_rate'])}</strong>
      </div>
    </div>""", unsafe_allow_html=True)

    vtext = verdict_text(ig, r["cagr3"], r["cagr5"])
    if vtext:
        valid_hists = [x for x in [r["cagr3"], r["cagr5"]] if _is_valid(x)]
        sc = signal_class(ig, np.nanmean(valid_hists) if valid_hists else np.nan)
        st.markdown(f"<div style='margin:-0.8rem 0 0.5rem 0; color:#94a3b8; font-size:0.88rem'>📌 {vtext}</div>",
                    unsafe_allow_html=True)

    # ── two chart columns ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Cross-Sensitivity & Historical Benchmark</div>",
                unsafe_allow_html=True)
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.markdown("<div style='color:#94a3b8;font-size:0.78rem;font-weight:600;margin-bottom:0.5rem'>MARGIN SENSITIVITY → REQUIRED 7-YR REVENUE CAGR</div>",
                    unsafe_allow_html=True)
        fig_sens = build_sensitivity_chart(r["sens_labels"], r["sens_values"], r["op_margin"])
        st.plotly_chart(fig_sens, use_container_width=True)

    with ch2:
        st.markdown("<div style='color:#94a3b8;font-size:0.78rem;font-weight:600;margin-bottom:0.5rem'>IMPLIED vs HISTORICAL REVENUE CAGR</div>",
                    unsafe_allow_html=True)
        fig_hist = build_cagr_comparison_chart(ig, r["cagr3"], r["cagr5"])
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── sensitivity table ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Sensitivity Matrix — Full Detail</div>",
                unsafe_allow_html=True)

    sens_df = pd.DataFrame({
        "Operating Margin Scenario": r["sens_labels"],
        "Absolute Margin (%)":       [fmt_pct_plain(r["op_margin"] + d)
                                      for d in [-0.04, -0.02, 0.0, 0.02, 0.04]],
        "Required 7-Yr Revenue CAGR": [fmt_pct(v) for v in r["sens_values"]],
    })
    st.dataframe(sens_df, use_container_width=True, hide_index=True)

    # ── historical reality check ──────────────────────────────────────────
    st.markdown("<div class='section-title'>Historical Reality Check</div>",
                unsafe_allow_html=True)

    cagr_data = {
        "Metric":           ["Implied 7-Yr Revenue CAGR (Market)", "3-Yr Historical Revenue CAGR", "5-Yr Historical Revenue CAGR"],
        "Value":            [fmt_pct(ig), fmt_pct(r["cagr3"]), fmt_pct(r["cagr5"])],
        "vs Implied":       ["—",
                             fmt_pct(r["cagr3"] - ig) if _is_valid(r["cagr3"]) else "N/A",
                             fmt_pct(r["cagr5"] - ig) if _is_valid(r["cagr5"]) else "N/A"],
    }
    hist_df = pd.DataFrame(cagr_data)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── model assumptions footer ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Model Assumptions</div>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    def assump(col, label, val):
        col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div style='color:#c084fc;font-size:1.1rem;font-weight:700;margin-top:0.1rem'>{val}</div></div>",
                     unsafe_allow_html=True)
    assump(a1, "Forecast Horizon",    "7 Years")
    assump(a2, "Terminal Growth Cap", "2.5%")
    assump(a3, "Risk-Free Rate",      "4.0%")
    assump(a4, "Equity Risk Premium", "5.0%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#374151;font-size:0.72rem;text-align:center;margin-top:1rem'>
      For informational purposes only. Not investment advice.
      Data sourced from Yahoo Finance via yfinance. Model uses a 7-year FCFF Reverse DCF with terminal value.
    </div>""", unsafe_allow_html=True)

elif not ticker_input.strip() and run_btn:
    st.warning("Please enter a valid US stock ticker.")
else:
    # ── landing state ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin: 4rem auto; max-width: 620px; text-align: center;'>
      <div style='font-size: 3.5rem; margin-bottom: 1.2rem'>📈</div>
      <div style='color: #e2e8f0; font-size: 1.15rem; font-weight: 600; margin-bottom: 0.7rem'>
        Enter any US-listed ticker to decode market expectations
      </div>
      <div style='color: #475569; font-size: 0.88rem; line-height: 1.7'>
        Forward Forecast solves the <strong style='color:#818cf8'>Reverse DCF</strong> —
        the exact revenue growth rate the market is pricing into today's stock price.
        Powered by live data, 7-year FCFF modelling, and cross-sensitivity analysis.
      </div>
      <div style='margin-top: 1.8rem; display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap'>
        <span class='tag' style='font-size:0.82rem;padding:0.3rem 0.8rem'>Reverse DCF</span>
        <span class='tag' style='font-size:0.82rem;padding:0.3rem 0.8rem'>Live yfinance Data</span>
        <span class='tag' style='font-size:0.82rem;padding:0.3rem 0.8rem'>Margin Sensitivity</span>
        <span class='tag' style='font-size:0.82rem;padding:0.3rem 0.8rem'>WACC via CAPM</span>
        <span class='tag' style='font-size:0.82rem;padding:0.3rem 0.8rem'>Historical CAGR Benchmark</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
