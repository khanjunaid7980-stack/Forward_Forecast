"""
Reverse DCF / Expectations Investing engine.
fetch_data()    — one network call, cacheable by Streamlit
compute_model() — pure math, re-runs instantly on param changes
"""
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import yfinance as yf
import requests_cache
from scipy.optimize import brentq

# ── HTTP-level cache ──────────────────────────────────────────────────────────
# Shared across all yf.Ticker calls in this process.
# Memory backend avoids filesystem permission issues on Streamlit Cloud.
# expire_after=1800 means Yahoo is only contacted once per ticker per 30 min.
_YF_SESSION = requests_cache.CachedSession(
    backend="memory",
    expire_after=1800,
    stale_if_error=True,   # serve stale data rather than a 429 error
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(val, default=np.nan):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _ttm_sum(series, n=4):
    arr = series.dropna()
    return float(arr.iloc[-min(n, len(arr)):].sum()) if len(arr) else np.nan


def _cagr(end, start, years):
    if years <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def _row(df, *labels):
    if df is None:
        return None
    for lbl in labels:
        for idx in df.index:
            if lbl.lower() in str(idx).lower():
                return df.loc[idx]
    return None


# ── data layer ────────────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(x in s for x in ["too many requests", "rate limit", "429", "rateerror"])


def fetch_data(ticker: str) -> dict:
    """Fetch with up to 3 retries on rate-limit errors (3 s / 6 s / 12 s backoff)."""
    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(3):
        try:
            return _fetch_data_inner(ticker)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < 2:
                time.sleep(3 * 2 ** attempt)
                continue
            raise
    raise last_exc


def _fetch_data_inner(ticker: str) -> dict:
    tk   = yf.Ticker(ticker.upper().strip(), session=_YF_SESSION)
    info = tk.info or {}

    price  = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    shares = _safe(info.get("sharesOutstanding"))
    debt   = _safe(info.get("totalDebt"), 0.0)
    cash   = _safe(info.get("totalCash") or info.get("cash"), 0.0)
    beta   = _safe(info.get("beta"), 1.0) if info.get("beta") is not None else 1.0

    # ── market cap: prefer the direct field (avoids share-count mismatch) ──
    mkt_cap = _safe(info.get("marketCap"))
    if np.isnan(mkt_cap) and not (np.isnan(price) or np.isnan(shares)):
        mkt_cap = price * shares

    ev = mkt_cap + debt - cash if not np.isnan(mkt_cap) else np.nan

    # ── TTM income statement ──────────────────────────────────────────────
    try:
        inc_q = tk.quarterly_income_stmt
    except Exception:
        inc_q = None

    rev_q    = _row(inc_q, "Total Revenue", "Revenue")
    ebit_q   = _row(inc_q, "EBIT", "Operating Income")
    tax_q    = _row(inc_q, "Tax Provision", "Income Tax")
    pretax_q = _row(inc_q, "Pretax Income", "Pre Tax Income")

    ttm_rev    = _ttm_sum(rev_q)    if rev_q    is not None else np.nan
    ttm_ebit   = _ttm_sum(ebit_q)   if ebit_q   is not None else np.nan
    ttm_tax    = _ttm_sum(tax_q)    if tax_q    is not None else np.nan
    ttm_pretax = _ttm_sum(pretax_q) if pretax_q is not None else np.nan

    if ttm_pretax > 0 and ttm_tax > 0:
        tax_rate = float(np.clip(ttm_tax / ttm_pretax, 0.01, 0.50))
    else:
        tax_rate = 0.21

    op_margin = (ttm_ebit / ttm_rev) if (ttm_rev > 0 and not np.isnan(ttm_ebit)) else np.nan

    # ── cash flow / BS for S2C ────────────────────────────────────────────
    try:
        cf_q = tk.quarterly_cashflow
        bs_q = tk.quarterly_balance_sheet
    except Exception:
        cf_q = bs_q = None

    s2c = _compute_s2c(tk)

    # ── supplemental metrics ──────────────────────────────────────────────
    pe_fwd      = _safe(info.get("forwardPE"))
    pe_trail    = _safe(info.get("trailingPE"))
    gross_marg  = _safe(info.get("grossMargins"))
    net_marg    = _safe(info.get("profitMargins"))
    fcf         = _safe(info.get("freeCashflow"))
    fcf_yield   = fcf / mkt_cap if (not np.isnan(fcf) and not np.isnan(mkt_cap) and mkt_cap > 0) else np.nan
    rev_growth  = _safe(info.get("revenueGrowth"))
    ev_ebitda   = _safe(info.get("enterpriseToEbitda"))
    analyst_tgt = _safe(info.get("targetMeanPrice"))
    analyst_lo  = _safe(info.get("targetLowPrice"))
    analyst_hi  = _safe(info.get("targetHighPrice"))
    analyst_n   = info.get("numberOfAnalystOpinions")
    roe         = _safe(info.get("returnOnEquity"))
    roic        = _safe(info.get("returnOnAssets"))      # proxy
    peg         = _safe(info.get("pegRatio"))
    ps_ratio    = _safe(info.get("priceToSalesTrailing12Months"))

    rev3, rev5 = _historical_cagrs(tk)

    return dict(
        ticker=ticker.upper(),
        company=info.get("longName", ticker.upper()),
        sector=info.get("sector", "N/A"),
        industry=info.get("industry", "N/A"),
        # valuation
        price=price, shares=shares, mkt_cap=mkt_cap,
        total_debt=debt, cash=cash, ev=ev,
        # fundamentals
        ttm_rev=ttm_rev, ttm_ebit=ttm_ebit,
        op_margin=op_margin, tax_rate=tax_rate,
        gross_margin=gross_marg, net_margin=net_marg,
        # model input
        beta=beta, s2c=s2c,
        # market metrics
        pe_fwd=pe_fwd, pe_trail=pe_trail,
        ev_ebitda=ev_ebitda, ps_ratio=ps_ratio, peg=peg,
        fcf=fcf, fcf_yield=fcf_yield,
        rev_growth=rev_growth, roe=roe, roic=roic,
        analyst_tgt=analyst_tgt, analyst_lo=analyst_lo,
        analyst_hi=analyst_hi, analyst_n=analyst_n,
        # history
        rev3=rev3, rev5=rev5,
    )


def _compute_s2c(tk) -> float:
    try:
        inc_a   = tk.income_stmt
        cf_a    = tk.cashflow
        rev_a   = _row(inc_a, "Total Revenue", "Revenue")
        capex_a = _row(cf_a, "Capital Expenditure", "Purchase Of PPE")
        depr_a  = _row(cf_a, "Depreciation")

        if rev_a is None or capex_a is None:
            return 1.5

        rev_a   = rev_a.dropna().sort_index()
        capex_a = capex_a.dropna().sort_index()
        depr_a  = depr_a.dropna().sort_index() if depr_a is not None else None

        years = min(len(rev_a) - 1, len(capex_a), 3)
        if years < 1:
            return 1.5

        vals = []
        for i in range(1, years + 1):
            d_rev   = float(rev_a.iloc[-i])  - float(rev_a.iloc[-i - 1])
            capex_i = abs(float(capex_a.iloc[-i]))
            depr_i  = abs(float(depr_a.iloc[-i])) if depr_a is not None else 0
            reinvest = capex_i - depr_i
            if reinvest > 0 and d_rev > 0:
                vals.append(d_rev / reinvest)

        return float(np.clip(np.median(vals), 0.5, 5.0)) if vals else 1.5
    except Exception:
        return 1.5


def _historical_cagrs(tk):
    try:
        inc_a = tk.income_stmt
        rev_a = _row(inc_a, "Total Revenue", "Revenue")
        if rev_a is None:
            return None, None
        rev_a = rev_a.dropna().sort_index()
        n = len(rev_a)
        def pair(yrs):
            return (float(rev_a.iloc[-1]), float(rev_a.iloc[-1 - yrs]), yrs) if n > yrs else None
        return pair(3), pair(5)
    except Exception:
        return None, None


# ── WACC ─────────────────────────────────────────────────────────────────────

def compute_wacc(data: dict) -> float:
    rf, erp, kd_pre = 0.04, 0.05, 0.05
    ke      = rf + max(data["beta"], 0.5) * erp
    kd      = kd_pre * (1 - data["tax_rate"])
    cap     = _safe(data["mkt_cap"], 1.0)
    debt    = _safe(data["total_debt"], 0.0)
    total   = cap + debt
    if total <= 0:
        return 0.09
    return float(np.clip(ke * cap / total + kd * debt / total, 0.05, 0.20))


# ── DCF core ─────────────────────────────────────────────────────────────────

def dcf_value(g, data: dict, wacc: float,
              op_margin_override=None, gn=0.025, horizon=7) -> float:
    rev0   = data["ttm_rev"]
    margin = op_margin_override if op_margin_override is not None else data["op_margin"]
    tax    = data["tax_rate"]
    s2c    = data["s2c"]
    gn     = min(gn, wacc - 0.005)

    pv, rev_prev = 0.0, rev0
    for t in range(1, horizon + 1):
        rev_t    = rev0 * (1 + g) ** t
        nopat_t  = rev_t * margin * (1 - tax)
        reinvest = (rev_t - rev_prev) / s2c if s2c > 0 else 0
        pv      += (nopat_t - reinvest) / (1 + wacc) ** t
        rev_prev = rev_t

    nopat_T  = rev0 * (1 + g) ** horizon * margin * (1 - tax)
    rr_T     = float(np.clip(gn / (s2c * (wacc - gn)), 0, 0.99)) if s2c * (wacc - gn) > 0 else 0
    tv       = nopat_T * (1 - rr_T) * (1 + gn) / (wacc - gn)
    pv      += tv / (1 + wacc) ** horizon
    return pv


def solve_implied_growth(data: dict, wacc: float,
                          op_margin_override=None, gn=0.025, horizon=7) -> float:
    target = data["ev"]
    d      = {**data}
    if op_margin_override is not None:
        d["op_margin"] = op_margin_override

    def obj(g):
        return dcf_value(g, d, wacc, gn=gn, horizon=horizon) - target

    lo, hi = -0.30, 1.50
    try:
        f_lo, f_hi = obj(lo), obj(hi)
        if f_lo * f_hi > 0:
            return lo if f_lo > 0 else hi
        return float(brentq(obj, lo, hi, xtol=1e-6, maxiter=500))
    except Exception:
        return np.nan


# ── projections ───────────────────────────────────────────────────────────────

def build_projections(g: float, data: dict, wacc: float, gn=0.025, horizon=7):
    rev0, margin = data["ttm_rev"], data["op_margin"]
    tax, s2c     = data["tax_rate"], data["s2c"]
    gn           = min(gn, wacc - 0.005)

    rows, pv_sum, rev_prev = [], 0.0, rev0
    for t in range(1, horizon + 1):
        rev_t    = rev0 * (1 + g) ** t
        ebit_t   = rev_t * margin
        nopat_t  = ebit_t * (1 - tax)
        reinvest = (rev_t - rev_prev) / s2c if s2c > 0 else 0
        fcff_t   = nopat_t - reinvest
        pv_f     = fcff_t / (1 + wacc) ** t
        pv_sum  += pv_f
        rows.append(dict(Year=f"Y{t}", Revenue=rev_t, EBIT=ebit_t,
                         NOPAT=nopat_t, Reinvestment=reinvest,
                         FCFF=fcff_t, PV_FCFF=pv_f))
        rev_prev = rev_t

    nopat_T  = rev0 * (1 + g) ** horizon * margin * (1 - tax)
    rr_T     = float(np.clip(gn / (s2c * (wacc - gn)), 0, 0.99)) if s2c * (wacc - gn) > 0 else 0
    tv       = nopat_T * (1 - rr_T) * (1 + gn) / (wacc - gn)
    pv_tv    = tv / (1 + wacc) ** horizon

    rows.append(dict(Year="Terminal", Revenue=None, EBIT=None, NOPAT=None,
                     Reinvestment=None, FCFF=tv, PV_FCFF=pv_tv))
    return rows, pv_sum, pv_tv


# ── sensitivities ─────────────────────────────────────────────────────────────

def sensitivity_margin(data, wacc, base_margin, gn=0.025, horizon=7):
    deltas = [-0.04, -0.02, 0.0, +0.02, +0.04]
    labels = ["-400bps", "-200bps", "Base TTM", "+200bps", "+400bps"]
    vals = []
    for d in deltas:
        m = base_margin + d
        vals.append(np.nan if m <= 0
                    else solve_implied_growth(data, wacc, op_margin_override=m, gn=gn, horizon=horizon))
    return labels, vals


def sensitivity_wacc(data, base_wacc, gn=0.025, horizon=7):
    deltas = [-0.02, -0.01, 0.0, +0.01, +0.02]
    labels = ["-200bps", "-100bps", "Base", "+100bps", "+200bps"]
    vals = [solve_implied_growth(data, float(np.clip(base_wacc + d, 0.04, 0.25)),
                                  gn=gn, horizon=horizon)
            for d in deltas]
    return labels, vals


def sensitivity_2d(data, base_wacc, base_margin, gn=0.025, horizon=7):
    """Returns (margin_labels, wacc_labels, matrix) where matrix[i][j] = implied CAGR %."""
    m_deltas = [-0.04, -0.02, 0.0, +0.02, +0.04]
    w_deltas = [-0.02, -0.01, 0.0, +0.01, +0.02]
    m_labels = [f"{(base_margin+d)*100:.1f}%" for d in m_deltas]
    w_labels = [f"{(base_wacc+d)*100:.1f}%" for d in w_deltas]

    matrix = []
    for wd in w_deltas:
        w   = float(np.clip(base_wacc + wd, 0.04, 0.25))
        row = []
        for md in m_deltas:
            m = base_margin + md
            g = (np.nan if m <= 0
                 else solve_implied_growth(data, w, op_margin_override=m, gn=gn, horizon=horizon))
            row.append(g * 100 if not np.isnan(g) else np.nan)
        matrix.append(row)
    return m_labels, w_labels, matrix


# ── fair-value calculator ─────────────────────────────────────────────────────

def fair_value_at_growth(g: float, data: dict, wacc: float, gn=0.025, horizon=7) -> dict:
    ev_impl = dcf_value(g, data, wacc, gn=gn, horizon=horizon)
    equity  = ev_impl - data["total_debt"] + data["cash"]
    price   = equity / data["shares"] if (not np.isnan(data.get("shares", np.nan))
                                          and data["shares"] > 0) else np.nan
    upside  = price / data["price"] - 1 if (data.get("price", 0) > 0
                                             and not np.isnan(price)) else np.nan
    return dict(ev=ev_impl, equity=equity, price=price, upside=upside)


# ── main entry ────────────────────────────────────────────────────────────────

def compute_model(data: dict, wacc_adj=0.0, gn=0.025, horizon=7) -> dict:
    base_wacc = compute_wacc(data)
    wacc      = float(np.clip(base_wacc + wacc_adj, 0.04, 0.25))

    implied_g                            = solve_implied_growth(data, wacc, gn=gn, horizon=horizon)
    projections, pv_fcff_sum, pv_tv      = build_projections(implied_g, data, wacc, gn=gn, horizon=horizon)
    sens_m_lbl, sens_m_val               = sensitivity_margin(data, wacc, data["op_margin"], gn=gn, horizon=horizon)
    sens_w_lbl, sens_w_val               = sensitivity_wacc(data, base_wacc, gn=gn, horizon=horizon)
    m_lbl_2d, w_lbl_2d, matrix_2d       = sensitivity_2d(data, base_wacc, data["op_margin"], gn=gn, horizon=horizon)

    def safe_cagr(pair):
        if pair is None: return np.nan
        end, start, yrs = pair
        return _cagr(end, start, yrs)

    total_ev = pv_fcff_sum + pv_tv

    return dict(
        **{k: data[k] for k in data},              # pass all raw fields through
        base_wacc=base_wacc, wacc=wacc, wacc_adj=wacc_adj, gn=gn, horizon=horizon,
        implied_g=implied_g,
        pv_fcff_sum=pv_fcff_sum, pv_tv=pv_tv, total_ev_model=total_ev,
        tv_pct=pv_tv / total_ev if total_ev > 0 else np.nan,
        projections=projections,
        sens_m_lbl=sens_m_lbl, sens_m_val=sens_m_val,
        sens_w_lbl=sens_w_lbl, sens_w_val=sens_w_val,
        m_lbl_2d=m_lbl_2d, w_lbl_2d=w_lbl_2d, matrix_2d=matrix_2d,
        cagr3=safe_cagr(data["rev3"]),
        cagr5=safe_cagr(data["rev5"]),
    )
