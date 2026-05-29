"""
Forward Forecast — Expectations Investing Engine
Streamlit UI
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine import fetch_data, compute_model, fair_value_at_growth

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forward Forecast | Reverse DCF",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0a0a0f;}
.block-container{padding:1.4rem 2rem 3rem 2rem;max-width:1440px;}
.hero-title{font-size:2.4rem;font-weight:800;letter-spacing:-1px;
  background:linear-gradient(135deg,#e2e8f0 0%,#94a3b8 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.15rem;}
.hero-sub{color:#475569;font-size:.88rem;margin-bottom:1.5rem;}
.kpi-card{background:#111118;border:1px solid #1e1e2e;border-radius:10px;
  padding:.9rem 1.1rem;height:100%;}
.kpi-label{color:#4b5563;font-size:.65rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;}
.kpi-value{color:#e2e8f0;font-size:1.45rem;font-weight:700;margin-top:.1rem;line-height:1.2;}
.kpi-sub{color:#374151;font-size:.72rem;margin-top:.15rem;}
.kpi-good{color:#34d399;}
.kpi-warn{color:#fbbf24;}
.kpi-bad{color:#f87171;}
.verdict-box{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);
  border:1px solid #3730a3;border-radius:14px;padding:1.6rem 2rem;margin:.8rem 0;}
.verdict-label{color:#818cf8;font-size:.7rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;}
.verdict-val{font-size:3.2rem;font-weight:800;letter-spacing:-2px;
  background:linear-gradient(90deg,#818cf8,#c084fc);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.verdict-desc{color:#94a3b8;font-size:.85rem;margin-top:.4rem;line-height:1.6;}
.section-title{color:#64748b;font-size:.68rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;
  border-bottom:1px solid #1a1a2e;padding-bottom:.4rem;margin:1.6rem 0 .8rem 0;}
.tag{display:inline-block;background:#1a1a2a;color:#64748b;border-radius:5px;
  padding:.12rem .55rem;font-size:.68rem;font-weight:600;margin-right:.35rem;}
.fv-box{background:#0f1729;border:1px solid #1e3a5f;border-radius:10px;padding:1rem 1.2rem;}
.fv-price{font-size:1.8rem;font-weight:700;color:#38bdf8;}
.fv-upside-pos{color:#34d399;font-size:1.1rem;font-weight:700;}
.fv-upside-neg{color:#f87171;font-size:1.1rem;font-weight:700;}
.stSlider>div>div>div{background:#818cf8;}
div[data-testid="stSidebarContent"]{background:#0c0c14;border-right:1px solid #1e1e2e;}
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed);
  color:white;border:none;border-radius:8px;padding:.5rem 1.6rem;
  font-weight:600;font-size:.95rem;width:100%;transition:opacity .2s;}
.stButton>button:hover{opacity:.85;}
.stTextInput>div>div>input{background:#111118;border:1px solid #1e1e2e;
  color:#e2e8f0;border-radius:8px;font-size:1rem;font-weight:600;
  text-transform:uppercase;padding:.5rem .9rem;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── formatters ────────────────────────────────────────────────────────────────

def _v(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))

def pct(v, d=1, sign=True):
    if not _v(v): return "N/A"
    return f"{'+' if (sign and v>0) else ''}{v*100:.{d}f}%"

def pct0(v, d=1):
    if not _v(v): return "N/A"
    return f"{v*100:.{d}f}%"

def bn(v):
    if not _v(v): return "N/A"
    if abs(v)>=1e12: return f"${v/1e12:.2f}T"
    if abs(v)>=1e9:  return f"${v/1e9:.2f}B"
    if abs(v)>=1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def px(v):
    return f"${v:,.2f}" if _v(v) else "N/A"

def fx(v, d=1):
    return f"{v:.{d}f}x" if _v(v) else "N/A"

def num(v, d=1):
    return f"{v:.{d}f}" if _v(v) else "N/A"


# ── KPI card ──────────────────────────────────────────────────────────────────

def kpi(col, label, value, sub="", color=""):
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value' style='{"color:"+color if color else ""}'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── chart builders ────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
)


def _bar_colors(labels, values, base_label="Base TTM"):
    cols = []
    for lbl, v in zip(labels, values):
        if not _v(v):       cols.append("#374151")
        elif lbl == base_label or "Base" in lbl: cols.append("#818cf8")
        elif v > 0.25:      cols.append("#f87171")
        elif v < 0:         cols.append("#94a3b8")
        else:               cols.append("#34d399")
    return cols


def chart_margin_sensitivity(labels, values):
    colors  = _bar_colors(labels, values)
    y_vals  = [v * 100 if _v(v) else 0 for v in values]
    txt     = [pct(v) for v in values]
    fig = go.Figure(go.Bar(x=labels, y=y_vals, text=txt, textposition="outside",
                           marker_color=colors, marker_line_width=0))
    fig.update_layout(**_CHART_LAYOUT, height=280,
                      yaxis=dict(gridcolor="#1a1a2e", zeroline=True,
                                 zerolinecolor="#374151", ticksuffix="%"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def chart_wacc_sensitivity(labels, values):
    colors  = _bar_colors(labels, values, base_label="Base")
    y_vals  = [v * 100 if _v(v) else 0 for v in values]
    txt     = [pct(v) for v in values]
    fig = go.Figure(go.Bar(x=labels, y=y_vals, text=txt, textposition="outside",
                           marker_color=colors, marker_line_width=0))
    fig.update_layout(**_CHART_LAYOUT, height=280,
                      yaxis=dict(gridcolor="#1a1a2e", zeroline=True,
                                 zerolinecolor="#374151", ticksuffix="%"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def chart_cagr_compare(implied, cagr3, cagr5):
    cats = ["Implied (7yr)", "3-Yr Actual", "5-Yr Actual"]
    vals = [implied, cagr3, cagr5]
    cols = ["#818cf8", "#34d399", "#22d3ee"]
    fig = go.Figure()
    for cat, v, col in zip(cats, vals, cols):
        if _v(v):
            fig.add_trace(go.Bar(x=[cat], y=[v * 100], text=[pct(v)],
                                 textposition="outside",
                                 marker_color=col, marker_line_width=0,
                                 showlegend=False))
    fig.update_layout(**_CHART_LAYOUT, height=280, barmode="group",
                      yaxis=dict(gridcolor="#1a1a2e", zeroline=True,
                                 zerolinecolor="#374151", ticksuffix="%"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def chart_projections(projections):
    years   = [r["Year"] for r in projections]
    fcff    = [r["FCFF"] / 1e9 if r["FCFF"] is not None else 0 for r in projections]
    pv_fcff = [r["PV_FCFF"] / 1e9 if r["PV_FCFF"] is not None else 0 for r in projections]

    colors_fcff = ["#c084fc" if y == "Terminal" else "#818cf8" for y in years]
    colors_pv   = ["#7c3aed" if y == "Terminal" else "#4338ca" for y in years]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="FCFF", x=years, y=fcff, marker_color=colors_fcff,
                         marker_line_width=0, text=[f"${v:.1f}B" for v in fcff],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="PV of FCFF", x=years, y=pv_fcff, marker_color=colors_pv,
                         marker_line_width=0, text=[f"${v:.1f}B" for v in pv_fcff],
                         textposition="outside"))
    fig.update_layout(**_CHART_LAYOUT, height=310, barmode="group",
                      legend=dict(orientation="h", y=1.08, x=0,
                                  font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                      yaxis=dict(gridcolor="#1a1a2e", zeroline=True,
                                 zerolinecolor="#374151", ticksuffix="B"),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def chart_heatmap(m_labels, w_labels, matrix):
    z    = [[round(v, 1) if not np.isnan(v) else None for v in row] for row in matrix]
    txt  = [[f"{v:.1f}%" if v is not None else "N/A" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=m_labels, y=w_labels,
        text=txt, texttemplate="%{text}",
        colorscale=[
            [0.0,  "#166534"], [0.25, "#15803d"],
            [0.45, "#eab308"], [0.65, "#dc2626"],
            [1.0,  "#7f1d1d"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Implied CAGR %", font=dict(size=10, color="#94a3b8")),
            tickfont=dict(size=9, color="#94a3b8"),
            thickness=12, len=0.8,
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=340,
        xaxis=dict(title=dict(text="Operating Margin", font=dict(size=10)),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=dict(text="WACC", font=dict(size=10)),
                   gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── data cache ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch(ticker):
    return fetch_data(ticker)


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(r=None):
    st.sidebar.markdown("## ⚙️ Model Parameters")
    wacc_adj = st.sidebar.slider(
        "WACC Adjustment (bps)", -200, 200, 0, 10,
        help="Shift the computed WACC up or down.") / 10000

    gn = st.sidebar.slider(
        "Terminal Growth Rate (%)", 0.5, 3.0, 2.5, 0.1,
        help="Long-run GDP-bound growth after the forecast horizon.") / 100

    horizon = st.sidebar.selectbox(
        "Forecast Horizon", [5, 7, 10], index=1,
        help="Number of explicit FCFF projection years.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎯 Fair Value Calculator")
    g_user = st.sidebar.slider(
        "Your Growth Assumption (%)", -5.0, 50.0, 10.0, 0.5,
        help="Enter your own revenue CAGR thesis and see the implied fair value per share.") / 100

    if r is not None:
        fv = fair_value_at_growth(g_user, r, r["wacc"], gn=gn, horizon=horizon)
        up = fv["upside"]
        up_cls  = "fv-upside-pos" if (_v(up) and up >= 0) else "fv-upside-neg"
        up_str  = pct(up) if _v(up) else "N/A"
        st.sidebar.markdown(f"""
        <div class='fv-box'>
          <div style='color:#64748b;font-size:.68rem;font-weight:700;
               letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>
            Implied Fair Value · {g_user*100:.1f}% Growth
          </div>
          <div class='fv-price'>{px(fv['price'])}</div>
          <div class='fv-upside-{"pos" if (_v(up) and up>=0) else "neg"}'>
            {up_str} vs current price
          </div>
          <div style='color:#374151;font-size:.7rem;margin-top:.4rem'>
            Implied EV: {bn(fv['ev'])}
          </div>
        </div>""", unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🏗️ EV Bridge")
        st.sidebar.markdown(f"""
        <div style='font-size:.8rem;color:#94a3b8;line-height:2'>
          Market Cap &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#e2e8f0'>{bn(r['mkt_cap'])}</b><br>
          + Total Debt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#f87171'>{bn(r['total_debt'])}</b><br>
          − Cash &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#34d399'>{bn(r['cash'])}</b><br>
          <hr style='border-color:#1e1e2e;margin:.3rem 0'>
          <b>Enterprise Value &nbsp;</b> <b style='color:#818cf8'>{bn(r['ev'])}</b>
        </div>""", unsafe_allow_html=True)

    return wacc_adj, gn, horizon, g_user


# ── main ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Forward Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Institutional-Grade Expectations Investing · Reverse DCF Engine</div>',
            unsafe_allow_html=True)

c1, c2, _ = st.columns([2.2, 1, 4])
with c1:
    ticker_in = st.text_input("", placeholder="Ticker  (e.g. AAPL, NVDA, MSFT)",
                               label_visibility="collapsed",
                               key="ticker_input")
with c2:
    st.markdown("<div style='margin-top:.12rem'></div>", unsafe_allow_html=True)
    go_btn = st.button("Analyze")

# persist fetched raw data across sidebar reruns
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None

if go_btn and ticker_in.strip():
    with st.spinner(f"Fetching live data for **{ticker_in.upper()}** …"):
        try:
            st.session_state.raw_data = _fetch(ticker_in.strip().upper())
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ["too many requests", "rate limit", "429"]):
                st.error(
                    "**Yahoo Finance is temporarily rate-limiting this server.** "
                    "This is common on shared cloud IPs. "
                    "Please wait **30–60 seconds** and click **Analyze** again — "
                    "the second request will be served from cache and will succeed instantly."
                )
            else:
                st.error(f"**Data error:** {err}")
            st.stop()

raw = st.session_state.raw_data

# sidebar (always rendered; shows fair value only when data loaded)
wacc_adj, gn, horizon, g_user = render_sidebar(raw)

if raw is None:
    st.markdown("""
    <div style='margin:5rem auto;max-width:560px;text-align:center'>
      <div style='font-size:3rem;margin-bottom:1rem'>📈</div>
      <div style='color:#e2e8f0;font-size:1.1rem;font-weight:600;margin-bottom:.6rem'>
        Enter any US-listed ticker to decode market expectations
      </div>
      <div style='color:#374151;font-size:.85rem;line-height:1.8'>
        Solves the <b style='color:#818cf8'>Reverse DCF</b> — the exact revenue CAGR the market
        is pricing into today's stock price. Live data, 7-year FCFF model,
        margin × WACC sensitivity heatmap, and fair-value calculator.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# recompute model with current sidebar params (fast — pure math, no network)
try:
    r = compute_model(raw, wacc_adj=wacc_adj, gn=gn, horizon=horizon)
except Exception as e:
    st.error(f"**Model error:** {e}")
    st.stop()

ig = r["implied_g"]

# ── company header ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin:.6rem 0 .4rem 0'>
  <span style='color:#e2e8f0;font-size:1.25rem;font-weight:700'>{r['company']}</span>
  &nbsp;
  <span class='tag'>{r['ticker']}</span>
  <span class='tag'>{r['sector']}</span>
  <span class='tag'>{r['industry']}</span>
</div>""", unsafe_allow_html=True)

# ── KPI row 1 ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Market Snapshot</div>", unsafe_allow_html=True)
cols = st.columns(6)
kpi(cols[0], "Stock Price",       px(r["price"]),          "Current")
kpi(cols[1], "Market Cap",        bn(r["mkt_cap"]),         "")
kpi(cols[2], "Enterprise Value",  bn(r["ev"]),              "Model target")
kpi(cols[3], "TTM Revenue",       bn(r["ttm_rev"]),         "")
kpi(cols[4], "TTM Op. Margin",    pct0(r["op_margin"]),     "EBIT / Revenue")
kpi(cols[5], "WACC",              pct0(r["wacc"]),          f"Beta {r['beta']:.2f}")

# ── KPI row 2 ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Valuation & Quality Metrics</div>",
            unsafe_allow_html=True)
cols2 = st.columns(6)
kpi(cols2[0], "Trailing P/E",    fx(r.get("pe_trail")),    "")
kpi(cols2[1], "Forward P/E",     fx(r.get("pe_fwd")),      "")
kpi(cols2[2], "EV / EBITDA",     fx(r.get("ev_ebitda")),   "")
kpi(cols2[3], "FCF Yield",       pct0(r.get("fcf_yield")), f"FCF {bn(r.get('fcf'))}")
kpi(cols2[4], "Gross Margin",    pct0(r.get("gross_margin")), "")
kpi(cols2[5], "Revenue Growth",  pct(r.get("rev_growth")), "YoY TTM")

# ── analyst row ────────────────────────────────────────────────────────────
if _v(r.get("analyst_tgt")):
    at, al, ah = r.get("analyst_tgt"), r.get("analyst_lo"), r.get("analyst_hi")
    upside_to_target = (at / r["price"] - 1) if (_v(at) and r["price"] > 0) else None
    n_analysts = r.get("analyst_n", "")
    st.markdown(f"""
    <div style='background:#0f1a0f;border:1px solid #14532d;border-radius:8px;
         padding:.6rem 1rem;margin:.2rem 0 0 0;font-size:.8rem;color:#94a3b8'>
      🔍 &nbsp;<b style='color:#e2e8f0'>Analyst Consensus</b>
      &nbsp;·&nbsp; Target: <b style='color:#34d399'>{px(at)}</b>
      &nbsp;·&nbsp; Range: {px(al)} – {px(ah)}
      &nbsp;·&nbsp; Upside to Target: <b style='color:{"#34d399" if (_v(upside_to_target) and upside_to_target>=0) else "#f87171"}'>{pct(upside_to_target)}</b>
      &nbsp;·&nbsp; {f"{n_analysts} analysts" if n_analysts else ""}
    </div>""", unsafe_allow_html=True)

# ── verdict box ────────────────────────────────────────────────────────────
cagr_avg = np.nanmean([x for x in [r["cagr3"], r["cagr5"]] if _v(x)] or [np.nan])
gap = (ig - cagr_avg) if (_v(ig) and _v(cagr_avg)) else None

if _v(gap):
    if gap > 0.08:
        narrative = "High expectations embedded — market demands a significant acceleration vs history."
    elif gap > 0.03:
        narrative = "Market prices in moderate growth improvement over the historical trajectory."
    elif gap < -0.08:
        narrative = "Market is pricing in material deceleration vs history — potential value or distress signal."
    elif gap < -0.03:
        narrative = "Implied growth trails historical pace — cautious pricing relative to track record."
    else:
        narrative = "Implied growth is broadly in-line with the company's historical trajectory."
else:
    narrative = "Insufficient historical data to benchmark against."

tv_str = f"Terminal value = <b>{pct0(r.get('tv_pct'))}</b> of total EV model" if _v(r.get("tv_pct")) else ""
st.markdown(f"""
<div class='verdict-box'>
  <div class='verdict-label'>Core Market Verdict · Implied Revenue CAGR ({horizon}-Year Horizon)</div>
  <div class='verdict-val'>{pct(ig)}</div>
  <div class='verdict-desc'>
    The market prices <b>{pct(ig)}</b> annualised revenue growth into <b>{r['company']}</b>
    at a <b>{pct0(r['wacc'])}</b> WACC.
    &nbsp;·&nbsp; S2C: <b>{r['s2c']:.2f}x</b>
    &nbsp;·&nbsp; Terminal g: <b>{pct0(gn)}</b>
    &nbsp;·&nbsp; Tax: <b>{pct0(r['tax_rate'])}</b>
    {f"&nbsp;·&nbsp; {tv_str}" if tv_str else ""}
    <br><br>
    📌 {narrative}
  </div>
</div>""", unsafe_allow_html=True)

# ── charts row 1 ────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Sensitivity Analysis</div>", unsafe_allow_html=True)
ch1, ch2 = st.columns([3, 2])
with ch1:
    st.markdown("<div style='color:#4b5563;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>MARGIN SENSITIVITY → REQUIRED REVENUE CAGR</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_margin_sensitivity(r["sens_m_lbl"], r["sens_m_val"]),
                    use_container_width=True)
with ch2:
    st.markdown("<div style='color:#4b5563;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>IMPLIED vs HISTORICAL REVENUE CAGR</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_cagr_compare(ig, r["cagr3"], r["cagr5"]),
                    use_container_width=True)

# ── charts row 2 ────────────────────────────────────────────────────────────
ch3, ch4 = st.columns(2)
with ch3:
    st.markdown("<div style='color:#4b5563;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>WACC SENSITIVITY → REQUIRED REVENUE CAGR</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_wacc_sensitivity(r["sens_w_lbl"], r["sens_w_val"]),
                    use_container_width=True)
with ch4:
    st.markdown("<div style='color:#4b5563;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>YEAR-BY-YEAR FCFF PROJECTION (IMPLIED GROWTH)</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_projections(r["projections"]), use_container_width=True)

# ── 2D heatmap ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>2D Sensitivity Heatmap — Implied Revenue CAGR (%) by Margin × WACC</div>",
            unsafe_allow_html=True)
st.markdown("<div style='color:#374151;font-size:.75rem;margin-bottom:.5rem'>Green = low implied growth (conservative market expectations) · Red = high implied growth (demanding expectations)</div>",
            unsafe_allow_html=True)
st.plotly_chart(chart_heatmap(r["m_lbl_2d"], r["w_lbl_2d"], r["matrix_2d"]),
                use_container_width=True)

# ── detailed tables in tabs ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Detailed Output</div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Sensitivity Matrix", "📋 DCF Projections", "📈 Historical Reality Check"])

with tab1:
    m_df = pd.DataFrame({
        "Margin Scenario":      r["sens_m_lbl"],
        "Absolute Margin":      [pct0(r["op_margin"] + d) for d in [-0.04,-0.02,0,0.02,0.04]],
        "Required 7-Yr CAGR":  [pct(v) for v in r["sens_m_val"]],
    })
    w_df = pd.DataFrame({
        "WACC Scenario":       r["sens_w_lbl"],
        "Absolute WACC":       [pct0(r["base_wacc"] + d) for d in [-0.02,-0.01,0,0.01,0.02]],
        "Required 7-Yr CAGR": [pct(v) for v in r["sens_w_val"]],
    })
    tc1, tc2 = st.columns(2)
    with tc1:
        st.caption("Margin sensitivity")
        st.dataframe(m_df, use_container_width=True, hide_index=True)
    with tc2:
        st.caption("WACC sensitivity")
        st.dataframe(w_df, use_container_width=True, hide_index=True)

with tab2:
    proj_rows = []
    for row in r["projections"]:
        proj_rows.append({
            "Year":              row["Year"],
            "Revenue":           bn(row["Revenue"])      if row["Revenue"]      is not None else "—",
            "EBIT":              bn(row["EBIT"])         if row["EBIT"]         is not None else "—",
            "NOPAT":             bn(row["NOPAT"])        if row["NOPAT"]        is not None else "—",
            "Reinvestment":      bn(row["Reinvestment"]) if row["Reinvestment"] is not None else "—",
            "FCFF":              bn(row["FCFF"])         if row["FCFF"]         is not None else "—",
            "PV of FCFF":        bn(row["PV_FCFF"])      if row["PV_FCFF"]      is not None else "—",
        })
    proj_df = pd.DataFrame(proj_rows)
    st.dataframe(proj_df, use_container_width=True, hide_index=True)
    pv_sum, pv_tv = r["pv_fcff_sum"], r["pv_tv"]
    st.markdown(f"""
    <div style='font-size:.78rem;color:#64748b;margin-top:.4rem'>
      PV of explicit FCFFs: <b style='color:#818cf8'>{bn(pv_sum)}</b>
      &nbsp;·&nbsp; PV of Terminal Value: <b style='color:#c084fc'>{bn(pv_tv)}</b>
      &nbsp;·&nbsp; Total Modelled EV: <b style='color:#e2e8f0'>{bn(r['total_ev_model'])}</b>
      &nbsp;·&nbsp; Terminal Value Weight: <b>{pct0(r.get('tv_pct'))}</b>
    </div>""", unsafe_allow_html=True)

with tab3:
    def delta_str(hist_v):
        if not _v(hist_v) or not _v(ig): return "N/A"
        return pct(hist_v - ig)

    hist_df = pd.DataFrame({
        "Metric":         ["Implied Revenue CAGR (Market)", f"3-Yr Historical CAGR", f"5-Yr Historical CAGR"],
        "Value":          [pct(ig), pct(r["cagr3"]), pct(r["cagr5"])],
        "Delta vs Implied": ["—", delta_str(r["cagr3"]), delta_str(r["cagr5"])],
        "Signal":         ["—",
                           "Below implied → deceleration priced in" if (_v(r["cagr3"]) and _v(ig) and r["cagr3"] < ig) else ("Above implied → acceleration priced in" if (_v(r["cagr3"]) and _v(ig)) else "N/A"),
                           "Below implied → deceleration priced in" if (_v(r["cagr5"]) and _v(ig) and r["cagr5"] < ig) else ("Above implied → acceleration priced in" if (_v(r["cagr5"]) and _v(ig)) else "N/A")],
    })
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ── model assumptions ───────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Model Assumptions</div>", unsafe_allow_html=True)
ac = st.columns(5)
for col, lbl, val in zip(ac, ["Horizon", "Terminal Growth", "Risk-Free Rate", "ERP", "Pre-tax Kd"],
                              [f"{horizon} Years", pct0(gn), "4.0%", "5.0%", "5.0%"]):
    col.markdown(f"""<div class='kpi-card'><div class='kpi-label'>{lbl}</div>
    <div style='color:#c084fc;font-size:1rem;font-weight:700;margin-top:.1rem'>{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='color:#1f2937;font-size:.7rem;text-align:center;margin-top:1.5rem'>
  For informational purposes only. Not investment advice.
  Data via Yahoo Finance / yfinance. 7-year FCFF Reverse DCF, Brent-q optimisation.
</div>""", unsafe_allow_html=True)
