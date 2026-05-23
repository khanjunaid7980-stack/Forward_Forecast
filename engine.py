"""
Reverse DCF / Expectations Investing engine.
All heavy computation lives here; app.py only calls fetch_and_solve().
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
from scipy.optimize import brentq


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _ttm_sum(series, n=4):
    """Sum last n periods of a quarterly series."""
    arr = series.dropna()
    if len(arr) == 0:
        return np.nan
    return float(arr.iloc[-min(n, len(arr)):].sum())


def _cagr(end, start, years):
    if years <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


# ── data layer ────────────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> dict:
    tk = yf.Ticker(ticker.upper().strip())

    info        = tk.info or {}
    price       = _safe(info.get("currentPrice") or info.get("regularMarketPrice"), np.nan)
    shares      = _safe(info.get("sharesOutstanding"), np.nan)
    total_debt  = _safe(info.get("totalDebt"), 0)
    cash        = _safe(info.get("totalCash") or info.get("cash"), 0)
    beta_raw    = info.get("beta")
    beta        = _safe(beta_raw, 1.0) if beta_raw is not None else 1.0

    # ── income statement (quarterly for TTM) ──────────────────────────────
    try:
        inc_q = tk.quarterly_income_stmt
    except Exception:
        inc_q = None

    def row(df, *labels):
        if df is None:
            return None
        for lbl in labels:
            for idx in df.index:
                if lbl.lower() in str(idx).lower():
                    return df.loc[idx]
        return None

    rev_q    = row(inc_q, "Total Revenue", "Revenue")
    ebit_q   = row(inc_q, "EBIT", "Operating Income")
    tax_q    = row(inc_q, "Tax Provision", "Income Tax")
    pretax_q = row(inc_q, "Pretax Income", "Pre Tax Income")

    ttm_rev    = _ttm_sum(rev_q)   if rev_q    is not None else np.nan
    ttm_ebit   = _ttm_sum(ebit_q)  if ebit_q   is not None else np.nan
    ttm_tax    = _ttm_sum(tax_q)   if tax_q    is not None else np.nan
    ttm_pretax = _ttm_sum(pretax_q)if pretax_q is not None else np.nan

    if ttm_pretax > 0 and ttm_tax > 0:
        tax_rate = min(max(ttm_tax / ttm_pretax, 0.01), 0.50)
    else:
        tax_rate = 0.21

    op_margin = (ttm_ebit / ttm_rev) if (ttm_rev > 0 and not np.isnan(ttm_ebit)) else np.nan

    # ── cash flow / balance sheet for S2C ────────────────────────────────
    try:
        cf_q  = tk.quarterly_cashflow
        bs_q  = tk.quarterly_balance_sheet
    except Exception:
        cf_q = bs_q = None

    capex_q   = row(cf_q, "Capital Expenditure", "Purchase Of PPE", "Capex")
    depr_q    = row(cf_q, "Depreciation", "Depreciation And Amortization")
    curr_a_q  = row(bs_q, "Current Assets", "Total Current Assets")
    curr_l_q  = row(bs_q, "Current Liabilities", "Total Current Liabilities")
    cash_bs_q = row(bs_q, "Cash And Cash Equivalents", "Cash")
    std_q     = row(bs_q, "Short Term Debt", "Current Portion Of Long Term Debt")

    def _series_to_nwc(ca, cl, c, s):
        """Approximate operating NWC = (CA - cash) - (CL - short-term debt)."""
        try:
            ca_ = ca.dropna(); cl_ = cl.dropna()
            c_  = (c.dropna()  if c  is not None else 0)
            s_  = (s.dropna()  if s  is not None else 0)
            ca_op = ca_ - c_
            cl_op = cl_ - s_
            return ca_op - cl_op
        except Exception:
            return None

    nwc_series = _series_to_nwc(curr_a_q, curr_l_q, cash_bs_q, std_q) \
                 if (curr_a_q is not None and curr_l_q is not None) else None

    # S2C = ΔRevenue / Reinvestment  (use annual to reduce noise)
    s2c = _compute_s2c(tk, capex_q, depr_q, nwc_series)

    # ── annual revenue for CAGR ───────────────────────────────────────────
    rev3, rev5 = _historical_cagrs(tk)

    mkt_cap = price * shares if not (np.isnan(price) or np.isnan(shares)) else np.nan
    ev      = mkt_cap + total_debt - cash if not np.isnan(mkt_cap) else np.nan

    return dict(
        ticker    = ticker.upper(),
        price     = price,
        shares    = shares,
        mkt_cap   = mkt_cap,
        total_debt= total_debt,
        cash      = cash,
        ev        = ev,
        ttm_rev   = ttm_rev,
        ttm_ebit  = ttm_ebit,
        op_margin = op_margin,
        tax_rate  = tax_rate,
        beta      = beta,
        s2c       = s2c,
        rev3      = rev3,   # (end_rev, start_rev, years)
        rev5      = rev5,
        company   = info.get("longName", ticker.upper()),
        sector    = info.get("sector", "N/A"),
        industry  = info.get("industry", "N/A"),
    )


def _compute_s2c(tk, capex_q, depr_q, nwc_series):
    """Try annual S2C; fall back to 1.5."""
    try:
        inc_a = tk.income_stmt
        cf_a  = tk.cashflow

        def arow(df, *lbls):
            if df is None: return None
            for lbl in lbls:
                for idx in df.index:
                    if lbl.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        rev_a   = arow(inc_a, "Total Revenue", "Revenue")
        capex_a = arow(cf_a,  "Capital Expenditure", "Purchase Of PPE")
        depr_a  = arow(cf_a,  "Depreciation")

        if rev_a is None or capex_a is None:
            return 1.5

        rev_a   = rev_a.dropna().sort_index()
        capex_a = capex_a.dropna().sort_index()
        depr_a  = depr_a.dropna().sort_index() if depr_a is not None else None

        years = min(len(rev_a)-1, len(capex_a), 3)
        if years < 1:
            return 1.5

        s2c_vals = []
        for i in range(1, years+1):
            d_rev   = float(rev_a.iloc[-i])   - float(rev_a.iloc[-i-1])
            capex_i = abs(float(capex_a.iloc[-i]))
            depr_i  = abs(float(depr_a.iloc[-i])) if depr_a is not None else 0
            reinvest = capex_i - depr_i           # net capex; ignoring NWC for simplicity
            if reinvest > 0 and d_rev > 0:
                s2c_vals.append(d_rev / reinvest)

        if not s2c_vals:
            return 1.5
        val = np.median(s2c_vals)
        return float(np.clip(val, 0.5, 5.0))
    except Exception:
        return 1.5


def _historical_cagrs(tk):
    try:
        inc_a = tk.income_stmt

        def arow(df, *lbls):
            if df is None: return None
            for lbl in lbls:
                for idx in df.index:
                    if lbl.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        rev_a = arow(inc_a, "Total Revenue", "Revenue")
        if rev_a is None:
            return None, None

        rev_a = rev_a.dropna().sort_index()
        n = len(rev_a)

        def pair(yrs):
            if n > yrs:
                return (float(rev_a.iloc[-1]), float(rev_a.iloc[-1-yrs]), yrs)
            return None

        return pair(3), pair(5)
    except Exception:
        return None, None


# ── WACC ─────────────────────────────────────────────────────────────────────

def compute_wacc(data: dict) -> float:
    rf, erp, kd_pretax = 0.04, 0.05, 0.05
    beta    = max(data["beta"], 0.5)
    ke      = rf + beta * erp
    tax     = data["tax_rate"]
    kd      = kd_pretax * (1 - tax)
    mkt_cap = _safe(data["mkt_cap"], 1)
    debt    = _safe(data["total_debt"], 0)
    total   = mkt_cap + debt
    if total <= 0:
        return 0.09
    wacc = ke * (mkt_cap / total) + kd * (debt / total)
    return float(np.clip(wacc, 0.05, 0.20))


# ── DCF core ─────────────────────────────────────────────────────────────────

def dcf_value(g, data: dict, wacc: float, op_margin_override=None, gn=0.025, horizon=7) -> float:
    """Return modelled EV for a given revenue CAGR g."""
    rev0       = data["ttm_rev"]
    margin     = op_margin_override if op_margin_override is not None else data["op_margin"]
    tax        = data["tax_rate"]
    s2c        = data["s2c"]
    gn         = min(gn, wacc - 0.01)   # ensure gn < wacc

    pv = 0.0
    rev_prev = rev0
    for t in range(1, horizon + 1):
        rev_t    = rev0 * (1 + g) ** t
        ebit_t   = rev_t * margin
        nopat_t  = ebit_t * (1 - tax)
        d_rev    = rev_t - rev_prev
        reinvest = d_rev / s2c if s2c > 0 else 0
        fcff_t   = nopat_t - reinvest
        pv      += fcff_t / (1 + wacc) ** t
        rev_prev = rev_t

    # terminal value (Gordon growth on final NOPAT*(1-reinvestment_rate))
    rev_T    = rev0 * (1 + g) ** horizon
    nopat_T  = rev_T * margin * (1 - tax)
    rr_T     = gn / (s2c * (wacc - gn)) if (s2c * (wacc - gn)) > 0 else 0
    rr_T     = np.clip(rr_T, 0, 0.99)
    fcff_T1  = nopat_T * (1 - rr_T) * (1 + gn)
    tv       = fcff_T1 / (wacc - gn)
    pv      += tv / (1 + wacc) ** horizon

    return pv


def solve_implied_growth(data: dict, wacc: float, op_margin_override=None) -> float:
    """Binary search for g that sets DCF == target EV."""
    target = data["ev"]

    def obj(g):
        return dcf_value(g, data, wacc, op_margin_override) - target

    # bracket search
    lo, hi = -0.30, 1.50
    try:
        # check sign change exists
        f_lo = obj(lo)
        f_hi = obj(hi)
        if f_lo * f_hi > 0:
            # EV might be extreme; widen
            if f_lo > 0:          # model always > target → revenue already too high
                return lo
            else:                 # model always < target → implied g > hi
                return hi
        g_star = brentq(obj, lo, hi, xtol=1e-6, maxiter=500)
        return float(g_star)
    except Exception:
        return np.nan


# ── sensitivity matrix ────────────────────────────────────────────────────────

def sensitivity_matrix(data: dict, wacc: float, base_margin: float):
    deltas = [-0.04, -0.02, 0.0, +0.02, +0.04]
    labels = ["-400bps", "-200bps", "Base TTM", "+200bps", "+400bps"]
    results = []
    for d in deltas:
        m = base_margin + d
        if m <= 0:
            results.append(np.nan)
            continue
        g = solve_implied_growth({**data, "op_margin": m}, wacc)
        results.append(g)
    return labels, results


# ── main entry ────────────────────────────────────────────────────────────────

def fetch_and_solve(ticker: str) -> dict:
    data = fetch_data(ticker)

    if np.isnan(data.get("ev", np.nan)):
        raise ValueError(f"Could not retrieve valid market data for '{ticker}'. Check the ticker symbol.")
    if np.isnan(data.get("ttm_rev", np.nan)) or data["ttm_rev"] <= 0:
        raise ValueError(f"Revenue data unavailable for '{ticker}'.")
    if np.isnan(data.get("op_margin", np.nan)):
        raise ValueError(f"Operating margin could not be computed for '{ticker}'.")

    wacc        = compute_wacc(data)
    implied_g   = solve_implied_growth(data, wacc)
    sens_labels, sens_values = sensitivity_matrix(data, wacc, data["op_margin"])

    # historical CAGRs
    def safe_cagr(pair):
        if pair is None: return np.nan
        end, start, yrs = pair
        return _cagr(end, start, yrs)

    cagr3 = safe_cagr(data["rev3"])
    cagr5 = safe_cagr(data["rev5"])

    return dict(
        # metadata
        ticker   = data["ticker"],
        company  = data["company"],
        sector   = data["sector"],
        industry = data["industry"],
        # market
        price    = data["price"],
        mkt_cap  = data["mkt_cap"],
        ev       = data["ev"],
        # fundamentals
        ttm_rev  = data["ttm_rev"],
        op_margin= data["op_margin"],
        tax_rate = data["tax_rate"],
        # model inputs
        wacc     = wacc,
        beta     = data["beta"],
        s2c      = data["s2c"],
        # results
        implied_g    = implied_g,
        sens_labels  = sens_labels,
        sens_values  = sens_values,
        cagr3        = cagr3,
        cagr5        = cagr5,
    )
