"""
Stock Forward Projection — Single-file Streamlit Web App
Combines models, tracker, and UI into one deployable file.
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import json, uuid, time, warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Projection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
[data-testid="metric-container"] {
    background: white; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #2c3e50;
    padding: 10px 0 6px 0; border-bottom: 2px solid #e9ecef;
    margin-bottom: 16px;
}
.callout {
    background: #eaf4fb; border-left: 4px solid #3498db;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.9rem; margin: 12px 0;
}
.callout-warn {
    background: #fef9e7; border-left: 4px solid #f39c12;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.9rem; margin: 12px 0;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════

@dataclass
class StockConfig:
    ticker:               str   = "AAPL"
    history_years:        int   = 5
    years:                int   = 10
    mc_simulations:       int   = 300
    mc_seed:              int   = 42
    inflation_rate:       float = 0.03
    dividend_yield:       float = 0.02
    initial_price:        float = field(default=None)
    mc_annual_mean:       float = field(default=None)
    mc_annual_volatility: float = field(default=None)
    risk_free_rate:       float = field(default=None)
    bull_cagr:            float = field(default=None)
    base_cagr:            float = field(default=None)
    bear_cagr:            float = field(default=None)
    garch_omega:          float = field(default=None)
    garch_alpha:          float = field(default=None)
    garch_beta:           float = field(default=None)
    jump_lambda:          float = 0.10
    jump_mu:              float = -0.05
    jump_sigma:           float = 0.08
    regime_bull_mu:       float = field(default=None)
    regime_bear_mu:       float = field(default=None)
    regime_bull_sigma:    float = field(default=None)
    regime_bear_sigma:    float = field(default=None)
    regime_p_bull:        float = 0.975
    regime_p_bear:        float = 0.950
    eps:                  float = field(default=None)
    eps_growth_rate:      float = field(default=None)
    terminal_pe:          float = 18.0
    discount_rate:        float = field(default=None)
    rolling_mus:          list  = field(default_factory=list)
    rolling_sigmas:       list  = field(default_factory=list)
    rolling_windows:      list  = field(default_factory=list)


@dataclass
class Transaction:
    ticker: str
    type:   str
    shares: float
    price:  float
    date:   str
    notes:  str = ""
    id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def total_value(self):
        return self.shares * self.price

    def to_dict(self):
        return {"ticker":self.ticker,"type":self.type,"shares":self.shares,
                "price":self.price,"date":self.date,"notes":self.notes,"id":self.id}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "config": None, "sims": None,
        "transactions": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ══════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def calibrate(ticker, history_years, years, mc_simulations,
              inflation_rate, terminal_pe):
    c = StockConfig(ticker=ticker, history_years=history_years,
                    years=years, mc_simulations=mc_simulations,
                    inflation_rate=inflation_rate, terminal_pe=terminal_pe)
    t    = yf.Ticker(ticker)
    hist = t.history(period=f"{history_years}y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data for '{ticker}'.")

    closes  = hist["Close"]
    log_ret = np.log(closes / closes.shift(1)).dropna().values
    TD      = 252

    c.initial_price        = round(float(closes.iloc[-1]), 2)
    c.mc_annual_mean       = round(float(log_ret.mean() * TD), 4)
    c.mc_annual_volatility = round(float(log_ret.std() * np.sqrt(TD)), 4)

    try:
        tbill = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]
        c.risk_free_rate = round(float(tbill / 100), 4)
    except:
        c.risk_free_rate = 0.045

    info = t.info
    dy   = info.get("dividendYield", None)
    if dy and dy > 0:
        c.dividend_yield = round(float(dy), 4)

    c.eps            = info.get("trailingEps", None) or info.get("forwardEps", None)
    c.eps_growth_rate = round(float(info.get("earningsGrowth", None) or 0.08), 4)
    c.discount_rate   = c.risk_free_rate + 0.05

    r2   = log_ret**2
    var  = float(np.var(log_ret))
    acf1 = max(float(np.corrcoef(r2[:-1], r2[1:])[0,1]), 0.01)
    c.garch_alpha = round(min(acf1*0.3,  0.15), 4)
    c.garch_beta  = round(min(acf1*0.65, 0.84), 4)
    c.garch_omega = round(var*(1-c.garch_alpha-c.garch_beta), 8)

    med  = np.median(log_ret)
    bull = log_ret[log_ret >= med]; bear = log_ret[log_ret < med]
    c.regime_bull_mu    = round(float(bull.mean()*TD), 4)
    c.regime_bear_mu    = round(float(bear.mean()*TD), 4)
    c.regime_bull_sigma = round(float(bull.std()*np.sqrt(TD)), 4)
    c.regime_bear_sigma = round(float(bear.std()*np.sqrt(TD)), 4)

    sm  = np.exp(c.mc_annual_mean) - 1
    ss  = np.exp(c.mc_annual_volatility) - 1
    c.base_cagr = round(float(sm), 4)
    c.bull_cagr = round(float(sm + ss*0.5), 4)
    c.bear_cagr = round(float(sm - ss*0.5), 4)

    for w in [1,2,3,5]:
        if w > history_years: continue
        r = log_ret[-int(w*TD):]
        c.rolling_windows.append(w)
        c.rolling_mus.append(round(float(r.mean()*TD), 4))
        c.rolling_sigmas.append(round(float(r.std()*np.sqrt(TD)), 4))

    return c


# ══════════════════════════════════════════════════════════════
# SIMULATIONS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def run_simulations(ticker, history_years, years, mc_simulations,
                    inflation_rate, terminal_pe):
    c = calibrate(ticker, history_years, years, mc_simulations,
                  inflation_rate, terminal_pe)

    def gbm():
        np.random.seed(c.mc_seed)
        n,T,mu,sig,S0 = c.mc_simulations,c.years,c.mc_annual_mean,c.mc_annual_volatility,c.initial_price
        Z  = np.random.normal(0,1,(n,T))
        lr = (mu-0.5*sig**2)+sig*Z
        p  = np.zeros((n,T+1)); p[:,0]=S0
        for t in range(T): p[:,t+1]=p[:,t]*np.exp(lr[:,t])
        return p

    def jump():
        np.random.seed(c.mc_seed+1)
        n,T,S0=c.mc_simulations,c.years,c.initial_price
        mu,sig,lam,mj,sj=c.mc_annual_mean,c.mc_annual_volatility,c.jump_lambda,c.jump_mu,c.jump_sigma
        ma=mu-lam*(np.exp(mj+0.5*sj**2)-1)
        p=np.zeros((n,T+1)); p[:,0]=S0
        for t in range(T):
            Z=np.random.normal(0,1,n); J=np.random.normal(mj,sj,n)*np.random.poisson(lam,n)
            p[:,t+1]=p[:,t]*np.exp((ma-0.5*sig**2)+sig*Z+J)
        return p

    def garch():
        np.random.seed(c.mc_seed+2)
        n,T,S0=c.mc_simulations,c.years,c.initial_price
        mu,om,al,be=c.mc_annual_mean,c.garch_omega,c.garch_alpha,c.garch_beta
        p=np.zeros((n,T+1)); p[:,0]=S0
        h=np.full(n, c.mc_annual_volatility**2/252)
        for yr in range(T):
            ret=np.zeros(n)
            for _ in range(252):
                sd=np.sqrt(np.maximum(h,1e-9))
                z=np.random.normal(0,1,n); r=(mu/252-0.5*h)+sd*z
                h=om+al*r**2+be*h; ret+=r
            p[:,yr+1]=p[:,yr]*np.exp(ret)
        return p

    def regime():
        np.random.seed(c.mc_seed+3)
        n,T,S0=c.mc_simulations,c.years,c.initial_price
        p=np.zeros((n,T+1)); p[:,0]=S0
        reg=np.zeros(n,dtype=int)
        for yr in range(T):
            ret=np.zeros(n)
            for _ in range(252):
                mu_d =np.where(reg==0,c.regime_bull_mu/252,  c.regime_bear_mu/252)
                sig_d=np.where(reg==0,c.regime_bull_sigma/np.sqrt(252),c.regime_bear_sigma/np.sqrt(252))
                r=(mu_d-0.5*sig_d**2)+sig_d*np.random.normal(0,1,n); ret+=r
                stay=np.where(reg==0,c.regime_p_bull,c.regime_p_bear)
                reg=np.where(np.random.random(n)>stay,1-reg,reg)
            p[:,yr+1]=p[:,yr]*np.exp(ret)
        return p

    return {"gbm":gbm(), "jump":jump(), "garch":garch(), "regime":regime()}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def pct(paths):
    return {k:np.percentile(paths,v,axis=0)
            for k,v in [("p5",5),("p25",25),("p50",50),("p75",75),("p95",95)]}

def risk(paths, c):
    ret  = paths[:,-1]/c.initial_price-1
    v95  = float(np.percentile(ret,5))
    cv95 = float(ret[ret<=v95].mean())
    dd   = [float((p/np.maximum.accumulate(p)-1).min()) for p in paths]
    av   = float(np.mean(ret))/c.years
    sv   = float(np.std(ret))/np.sqrt(c.years)
    sh   = (av-c.risk_free_rate)/sv if sv>0 else 0
    return {"var_95":round(v95,4),"cvar_95":round(cv95,4),
            "mean_max_dd":round(float(np.mean(dd)),4),"sharpe":round(sh,4),
            "prob_profit":round(float(np.mean(ret>0)),4),
            "prob_beat_rf":round(float(np.mean(ret>(1+c.risk_free_rate)**c.years-1)),4)}

def fmt_price(ax):
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

COLORS = {"bull":"#27ae60","base":"#2980b9","bear":"#c0392b",
          "dcf":"#8e44ad","history":"#95a5a6",
          "gbm":"#2980b9","jump":"#e67e22","garch":"#8e44ad","regime":"#16a085"}


# ══════════════════════════════════════════════════════════════
# TRACKER HELPERS
# ══════════════════════════════════════════════════════════════

def get_txns():
    return st.session_state.transactions

def save_txns(txns):
    st.session_state.transactions = txns

def compute_positions():
    txns = sorted(get_txns(), key=lambda t: t.date)
    pos  = {}
    for txn in txns:
        tk = txn.ticker
        if tk not in pos:
            pos[tk] = {"shares":0.0,"avg_cost":0.0,"cost_basis":0.0,
                       "realised_pnl":0.0,"lots":[]}
        p = pos[tk]
        if txn.type == "BUY":
            p["lots"].append([txn.shares, txn.price])
            p["cost_basis"] += txn.shares*txn.price
            p["shares"]     += txn.shares
            p["avg_cost"]    = p["cost_basis"]/p["shares"]
        elif txn.type == "SELL":
            rem = txn.shares
            while rem>0 and p["lots"]:
                ls,lp = p["lots"][0]
                if ls<=rem:
                    p["realised_pnl"]+=ls*(txn.price-lp)
                    p["cost_basis"]-=ls*lp; p["shares"]-=ls
                    rem-=ls; p["lots"].pop(0)
                else:
                    p["realised_pnl"]+=rem*(txn.price-lp)
                    p["cost_basis"]-=rem*lp; p["shares"]-=rem
                    p["lots"][0][0]-=rem; rem=0
            p["avg_cost"]=p["cost_basis"]/p["shares"] if p["shares"]>0 else 0
    return {k:v for k,v in pos.items() if v["shares"]>1e-6}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(tickers_tuple):
    prices = {}
    for t in tickers_tuple:
        try:    prices[t] = float(yf.Ticker(t).fast_info.last_price)
        except: prices[t] = None
    return prices

def export_txns_json():
    return json.dumps([t.to_dict() for t in get_txns()], indent=2)

def import_txns_json(raw):
    data = json.loads(raw)
    save_txns([Transaction.from_dict(d) for d in data])


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock Projection")
    st.markdown("---")
    st.markdown("### 🔍 Single Stock")

    ticker = st.text_input("Ticker", value="AAPL",
                           placeholder="e.g. AAPL, MSFT, TSLA").upper().strip()
    c1, c2 = st.columns(2)
    years         = c1.number_input("Horizon (yrs)", 1, 30, 10)
    history_years = c2.number_input("History (yrs)", 1, 20, 5)
    n_sims = st.select_slider("Simulations",
                              options=[100,300,500,1000], value=300,
                              help="300 = fast (~1 min), 1000 = thorough (~4 min)")

    with st.expander("⚙️ Advanced"):
        terminal_pe = st.number_input("Terminal P/E", value=18.0, step=1.0)
        inflation   = st.slider("Inflation (%)", 0.0, 10.0, 3.0, 0.1)

    run_btn = st.button("▶ Run Projection", type="primary",
                        use_container_width=True)
    st.markdown("---")
    st.caption("Data: Yahoo Finance · Not financial advice")


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner(f"Fetching data and running simulations for {ticker}…  "
                    f"(~1–3 min for GARCH engine)"):
        try:
            c    = calibrate(ticker, int(history_years), int(years),
                             int(n_sims), inflation/100, terminal_pe)
            sims = run_simulations(ticker, int(history_years), int(years),
                                   int(n_sims), inflation/100, terminal_pe)
            st.session_state.config = c
            st.session_state.sims   = sims
            st.success(f"✅ {ticker} ready — {n_sims:,} simulations × 4 engines")
        except Exception as e:
            st.error(f"❌ {e}")

c    = st.session_state.config
sims = st.session_state.sims


# ══════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════

if c is None:
    st.markdown("# 📈 Stock Forward Projection")
    st.markdown("""
    <div class="callout">
    Enter a ticker in the sidebar and press <strong>▶ Run Projection</strong> to get started.
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Simulation Engines","4","GBM · Jump · GARCH · Regime")
    col2.metric("Data Source","Live","Yahoo Finance")
    col3.metric("Portfolio Tracker","✅","FIFO cost basis")
    col4.metric("Performance History","✅","Daily reconstruction")

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        | Engine | What it models |
        |--------|---------------|
        | **GBM** | Standard random walk — normally distributed returns |
        | **Jump Diffusion** | GBM + sudden crash events |
        | **GARCH(1,1)** | Volatility that clusters over time |
        | **Regime-Switching** | Markets alternate between bull and bear states |

        All parameters are **auto-calibrated** from real historical data via Yahoo Finance.
        The tracker lets you log real trades and see your actual P&L and portfolio history.

        > ⚠️ For informational purposes only. Not financial advice.
        """)
    st.stop()


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Overview", "🎲 Simulations",
                "📐 Risk Metrics", "💼 Tracker",
                "📈 Performance"])
yrs_arr = np.arange(0, c.years+1)


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown(f"# {c.ticker}  ·  {c.years}-Year Projection")
    st.caption(f"Calibrated from {c.history_years}y of history  ·  "
               f"{c.mc_simulations:,} simulations per engine")

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Price",          f"${c.initial_price:,.2f}")
    m2.metric("Avg Return (μ)", f"{c.mc_annual_mean*100:.2f}%")
    m3.metric("Volatility (σ)", f"{c.mc_annual_volatility*100:.2f}%")
    m4.metric("Risk-Free Rate", f"{c.risk_free_rate*100:.2f}%")
    m5.metric("Excess Return",  f"{(c.mc_annual_mean-c.risk_free_rate)*100:.2f}%")
    m6.metric("Dividend Yield", f"{c.dividend_yield*100:.2f}%")

    st.markdown("""
    <div class="callout">
    <strong>μ</strong> = average annual return from history.
    <strong>σ</strong> = how much it swung around that average — higher means more uncertainty.
    <strong>Excess return</strong> = how much more than a T-bill this stock historically earned.
    </div>
    """, unsafe_allow_html=True)

    # Scenario chart
    st.markdown('<div class="section-header">📐 Scenario Analysis</div>',
                unsafe_allow_html=True)

    scen = {"years": yrs_arr,
            "bull": c.initial_price*(1+c.bull_cagr+c.dividend_yield)**yrs_arr,
            "base": c.initial_price*(1+c.base_cagr+c.dividend_yield)**yrs_arr,
            "bear": c.initial_price*(1+c.bear_cagr+c.dividend_yield)**yrs_arr}

    dcf = None
    if c.eps and c.eps > 0:
        dcf = c.eps*(1+c.eps_growth_rate)**yrs_arr*c.terminal_pe \
              /(1+c.discount_rate)**yrs_arr

    hist_close = yf.Ticker(c.ticker).history(
        period=f"{c.history_years}y", auto_adjust=True)["Close"]

    fig, ax = plt.subplots(figsize=(12,5), facecolor="white")
    for key,label in [("bull","Bull"),("base","Base"),("bear","Bear")]:
        cagr = getattr(c, f"{key}_cagr")
        ax.plot(yrs_arr, scen[key], color=COLORS[key], lw=2.5,
                label=f"{label} ({cagr*100:+.1f}%/yr → ${scen[key][-1]:,.0f})")
        ax.plot(yrs_arr, scen[key]/(1+c.inflation_rate)**yrs_arr,
                color=COLORS[key], lw=1.1, ls="--", alpha=0.5)
    if dcf is not None:
        ax.plot(yrs_arr, dcf, color=COLORS["dcf"], lw=2, ls="-.",
                label=f"DCF Intrinsic (g={c.eps_growth_rate*100:.1f}%)")
    norm = hist_close/hist_close.iloc[-1]*c.initial_price
    hx   = np.linspace(-c.history_years,0,len(norm))
    ax.plot(hx, norm.values, color=COLORS["history"], lw=1.3,
            label=f"Historical ({c.history_years}y)")
    ax.axvline(0,color="#2c3e50",ls=":",lw=1.5,alpha=0.6)
    ax.axhline(c.initial_price,color="gray",ls=":",lw=1)
    ax.annotate("TODAY",xy=(0,c.initial_price),xytext=(0.5,c.initial_price*0.88),
                fontsize=8,arrowprops=dict(arrowstyle="->",color="#2c3e50"))
    ax.set_xlabel("Year (negative = past, positive = forecast)")
    ax.set_ylabel("Price ($)"); fmt_price(ax)
    ax.legend(fontsize=8,title="── Nominal  ╌ Real")
    ax.grid(alpha=0.25); ax.set_facecolor("#fafafa")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    # Scenario table
    st.dataframe(pd.DataFrame([{
        "Scenario": f"{'🟢' if k=='bull' else '🔵' if k=='base' else '🔴'} {k.title()}",
        "CAGR": f"{getattr(c,k+'_cagr')*100:+.2f}%",
        f"Price at Year {c.years}": f"${scen[k][-1]:,.2f}",
        "Total Return": f"{(scen[k][-1]/c.initial_price-1)*100:+.1f}%",
        "Real Value": f"${scen[k][-1]/(1+c.inflation_rate)**c.years:,.2f}",
    } for k in ["bull","base","bear"]]),
    use_container_width=True, hide_index=True)

    # Rolling sensitivity
    if c.rolling_windows:
        st.markdown('<div class="section-header">🔁 Calibration Sensitivity</div>',
                    unsafe_allow_html=True)
        st.caption("Large differences across windows = the stock's behaviour has changed "
                   "over time → projections less reliable.")
        roll_df = pd.DataFrame({
            "Window":    [f"{w}y" for w in c.rolling_windows],
            "μ":         [f"{m*100:+.2f}%" for m in c.rolling_mus],
            "σ":         [f"{s*100:.2f}%"  for s in c.rolling_sigmas],
            "Δμ":        [f"{(m-c.mc_annual_mean)*100:+.2f}%" for m in c.rolling_mus],
            "Stable?":   ["✅" if abs(m-c.mc_annual_mean)<0.05 else "⚠️"
                          for m in c.rolling_mus],
        })
        st.dataframe(roll_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SIMULATIONS
# ══════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown(f"# {c.ticker}  ·  Simulation Engines")
    st.markdown("""
    <div class="callout">
    Four engines, same stock, same history — different mathematical assumptions.
    <strong>Darker bands = where most outcomes landed.</strong>
    Where all four agree, confidence is higher.
    </div>
    """, unsafe_allow_html=True)

    engine_meta = {
        "gbm":    ("① GBM — Standard Random Walk",        COLORS["gbm"],
                   "Assumes returns are normally distributed each year. "
                   "Simple baseline — underestimates crashes."),
        "jump":   ("② Jump Diffusion — Crashes Included",  COLORS["jump"],
                   f"GBM + random sudden shocks (~{c.jump_lambda:.0%}/yr, "
                   f"avg {c.jump_mu*100:.1f}% each). Better at capturing rare disasters."),
        "garch":  ("③ GARCH — Clustering Volatility",      COLORS["garch"],
                   f"Volatility changes over time (α={c.garch_alpha:.3f}, β={c.garch_beta:.3f}). "
                   "Turbulent periods follow turbulent periods — like real markets."),
        "regime": ("④ Regime-Switching — Bull & Bear",     COLORS["regime"],
                   f"Bull: μ={c.regime_bull_mu*100:.1f}%, σ={c.regime_bull_sigma*100:.1f}%  ·  "
                   f"Bear: μ={c.regime_bear_mu*100:.1f}%, σ={c.regime_bear_sigma*100:.1f}%. "
                   "Markets switch between states via Markov chain."),
    }

    base_path = c.initial_price*(1+c.base_cagr+c.dividend_yield)**yrs_arr
    col_l, col_r = st.columns(2)
    cols = [col_l, col_r, col_l, col_r]

    for i,(eng,(title,color,explain)) in enumerate(engine_meta.items()):
        paths = sims[eng]; p = pct(paths); r = risk(paths, c)
        with cols[i]:
            st.markdown(f"### {title}")
            st.caption(explain)

            fig, ax = plt.subplots(figsize=(7,4), facecolor="white")
            s = np.random.choice(paths.shape[0], min(80,paths.shape[0]), replace=False)
            for j in s: ax.plot(yrs_arr,paths[j],color="#bdc3c7",alpha=0.07,lw=0.5)
            ax.fill_between(yrs_arr,p["p5"], p["p95"],color=color,alpha=0.12,
                            label="90% of outcomes")
            ax.fill_between(yrs_arr,p["p25"],p["p75"],color=color,alpha=0.30,
                            label="Middle 50%")
            ax.plot(yrs_arr,p["p50"],color=color,lw=2.5,
                    label=f"Median ${p['p50'][-1]:,.0f}")
            ax.plot(yrs_arr,base_path,color="#c0392b",lw=1.5,ls="--",
                    alpha=0.7,label="Base scenario")
            ax.axhline(c.initial_price,color="gray",ls=":",lw=1)
            ax.set_xlabel("Year"); ax.set_ylabel("Price ($)"); fmt_price(ax)
            ax.legend(fontsize=7); ax.grid(alpha=0.25); ax.set_facecolor("#fafafa")
            st.pyplot(fig,use_container_width=True); plt.close(fig)

            ma,mb,mc_ = st.columns(3)
            ma.metric("Median",          f"${p['p50'][-1]:,.0f}")
            mb.metric("5th Pctile",      f"${p['p5'][-1]:,.0f}",
                      delta=f"{(p['p5'][-1]/c.initial_price-1)*100:.1f}%")
            mc_.metric("95th Pctile",    f"${p['p95'][-1]:,.0f}",
                      delta=f"{(p['p95'][-1]/c.initial_price-1)*100:.1f}%")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# TAB 3 — RISK METRICS
# ══════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown(f"# {c.ticker}  ·  Risk Metrics")
    st.markdown("""
    <div class="callout">
    Same stock, four models. <strong>Where they agree = higher confidence.</strong>
    Where they diverge = be cautious with that metric.
    </div>
    """, unsafe_allow_html=True)

    eng_risks = {e: risk(sims[e], c) for e in ["gbm","jump","garch","regime"]}
    rows = []
    for label,key in [("VaR 95%","var_95"),("CVaR 95%","cvar_95"),
                       ("Avg Max Drawdown","mean_max_dd"),("Sharpe Ratio","sharpe"),
                       ("P(profit)","prob_profit"),("P(beat rf)","prob_beat_rf")]:
        is_sh = key=="sharpe"
        row = {"Metric":label}
        for e in ["gbm","jump","garch","regime"]:
            v = eng_risks[e][key]
            row[e.upper()] = f"{v:.2f}" if is_sh else f"{v*100:.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Metric"),
                 use_container_width=True)

    # Explanations
    st.markdown("---")
    st.markdown("### 📖 What Each Metric Means")
    for label,explanation in [
        ("VaR 95%",          "The total return in the **worst 5%** of scenarios. If -40%, then 5% of simulations lost more than 40%."),
        ("CVaR 95%",         "The **average** return in those worst 5% scenarios — always worse than VaR. Also called Expected Shortfall."),
        ("Avg Max Drawdown", "The typical **largest peak-to-trough decline** across all simulated paths — how painful the ride might be."),
        ("Sharpe Ratio",     "Annualised excess return divided by annualised volatility. **Above 1.0 is generally good.** Higher = better risk-adjusted return."),
        ("P(profit)",        "The % of simulated paths that ended **above today's price** after the full horizon."),
        ("P(beat rf)",       f"The % of paths that earned **more than the T-bill return** ({c.risk_free_rate*100:.2f}%) — the minimum bar for taking on stock risk."),
    ]:
        with st.expander(f"**{label}**"):
            st.markdown(explanation)

    # Terminal distributions
    st.markdown("---")
    st.markdown(f"### Terminal Price Distribution at Year {c.years}")
    st.caption("Wider histogram = more uncertainty about the final price.")
    fig, axes = plt.subplots(1,4,figsize=(16,4),facecolor="white")
    for ax,eng,lbl,col in zip(axes,
        ["gbm","jump","garch","regime"],
        ["GBM","Jump Diffusion","GARCH(1,1)","Regime-Switching"],
        [COLORS[e] for e in ["gbm","jump","garch","regime"]]):
        t = sims[eng][:,-1]
        ax.hist(t,bins=60,color=col,alpha=0.8,edgecolor="white")
        ax.axvline(np.median(t),color="#2c3e50",lw=2,
                   label=f"Median ${np.median(t):,.0f}")
        ax.axvline(c.initial_price,color="red",lw=1.5,ls="--",
                   label=f"Today ${c.initial_price:,.0f}")
        ax.set_title(lbl,fontsize=9,fontweight="bold")
        ax.set_xlabel("Final Price ($)"); ax.set_ylabel("Simulations")
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# TAB 4 — TRACKER
# ══════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("# 💼 Portfolio Tracker")
    st.markdown("""
    <div class="callout">
    Log your real trades. The tracker computes your <strong>FIFO cost basis</strong>,
    live market value, and unrealised P&amp;L — updated live from Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

    # Add trade form
    st.markdown('<div class="section-header">➕ Log a Trade</div>',
                unsafe_allow_html=True)
    with st.form("txn_form", clear_on_submit=True):
        fc1,fc2,fc3,fc4,fc5 = st.columns([2,1,2,2,3])
        tk  = fc1.text_input("Ticker", placeholder="AAPL").upper().strip()
        typ = fc2.selectbox("Type", ["BUY","SELL"])
        sh  = fc3.number_input("Shares", min_value=0.0001, step=0.01,
                               value=1.0, format="%.4f")
        pr  = fc4.number_input("Price ($)", min_value=0.01, step=0.01,
                               value=100.0, format="%.2f")
        dt  = fc5.date_input("Date", value=date.today())
        nt  = st.text_input("Notes (optional)")
        sub = st.form_submit_button("✅ Add Trade", use_container_width=True)
        if sub:
            if not tk:
                st.error("Enter a ticker.")
            else:
                txns = get_txns()
                txns.append(Transaction(tk,typ,float(sh),float(pr),
                                        dt.isoformat(),nt))
                save_txns(txns)
                st.success(f"Added: {typ} {sh:.4f} × {tk} @ ${pr:.2f}")
                st.rerun()

    # Live positions
    st.markdown('<div class="section-header">📊 Live Positions</div>',
                unsafe_allow_html=True)
    positions = compute_positions()
    if not positions:
        st.info("No open positions yet. Log a trade above.")
    else:
        with st.spinner("Fetching live prices…"):
            live = fetch_live(tuple(positions.keys()))

        total_cost=total_mkt=total_unr=0
        rows_pnl = []
        for ticker_p, pos in positions.items():
            lp  = live.get(ticker_p) or 0
            mv  = pos["shares"]*lp
            unr = mv - pos["cost_basis"]
            pct_u = (unr/pos["cost_basis"]*100) if pos["cost_basis"]>0 else 0
            total_cost+=pos["cost_basis"]; total_mkt+=mv; total_unr+=unr
            rows_pnl.append({
                "Ticker":        ticker_p,
                "Shares":        f"{pos['shares']:.4f}",
                "Avg Cost":      f"${pos['avg_cost']:,.2f}",
                "Live Price":    f"${lp:,.2f}",
                "Market Value":  f"${mv:,.2f}",
                "Unrealised P&L":f"{'▲' if unr>=0 else '▼'} ${abs(unr):,.2f}",
                "Return":        f"{pct_u:+.2f}%",
                "Realised P&L":  f"${pos['realised_pnl']:,.2f}",
            })

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Cost Basis",     f"${total_cost:,.2f}")
        m2.metric("Market Value",   f"${total_mkt:,.2f}",
                  delta=f"${total_mkt-total_cost:,.2f}")
        m3.metric("Unrealised P&L", f"${total_unr:,.2f}",
                  delta=f"{(total_unr/total_cost*100) if total_cost>0 else 0:.2f}%")
        m4.metric("Positions",      str(len(positions)))

        st.dataframe(pd.DataFrame(rows_pnl),
                     use_container_width=True, hide_index=True)

        # Pie chart
        col1,col2 = st.columns(2)
        with col1:
            fig_pie,ax_pie = plt.subplots(figsize=(5,5),facecolor="white")
            mv_vals = [positions[t]["shares"]*(live.get(t) or 0) for t in positions]
            wedge_colors = ["#3498db","#2ecc71","#e74c3c","#f39c12",
                            "#9b59b6","#1abc9c","#e67e22","#34495e"]
            ax_pie.pie(mv_vals, labels=list(positions.keys()),
                      autopct="%1.1f%%",
                      colors=wedge_colors[:len(positions)],
                      startangle=90, pctdistance=0.82)
            ax_pie.set_title("Allocation by Market Value", fontsize=10)
            st.pyplot(fig_pie,use_container_width=True); plt.close(fig_pie)

        with col2:
            fig_bar,ax_bar = plt.subplots(figsize=(5,5),facecolor="white")
            unr_vals = [positions[t]["shares"]*(live.get(t) or 0)
                        - positions[t]["cost_basis"] for t in positions]
            bar_cols = ["#27ae60" if v>=0 else "#c0392b" for v in unr_vals]
            ax_bar.barh(list(positions.keys()), unr_vals,
                       color=bar_cols, alpha=0.85)
            ax_bar.axvline(0, color="#2c3e50", lw=1)
            ax_bar.set_xlabel("Unrealised P&L ($)")
            ax_bar.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax_bar.grid(axis="x",alpha=0.25); ax_bar.set_facecolor("#fafafa")
            ax_bar.set_title("P&L by Position", fontsize=10)
            st.pyplot(fig_bar,use_container_width=True); plt.close(fig_bar)

    # Transaction log
    st.markdown('<div class="section-header">📋 Transaction Log</div>',
                unsafe_allow_html=True)
    txns = get_txns()
    if not txns:
        st.info("No transactions yet.")
    else:
        txn_df = pd.DataFrame([{
            "ID":t.id,"Date":t.date,"Ticker":t.ticker,
            "Type":t.type,"Shares":t.shares,
            "Price":f"${t.price:,.2f}","Total":f"${t.total_value:,.2f}",
            "Notes":t.notes,
        } for t in sorted(txns,key=lambda x:x.date,reverse=True)])
        st.dataframe(txn_df,use_container_width=True,hide_index=True)

        col_a,col_b = st.columns(2)
        with col_a:
            csv = txn_df.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", data=csv,
                               file_name="transactions.csv",mime="text/csv")
        with col_b:
            st.download_button("⬇️ Export JSON (backup)",
                               data=export_txns_json().encode(),
                               file_name="portfolio.json",
                               mime="application/json")

        with st.expander("📂 Import trades from JSON backup"):
            uploaded = st.file_uploader("Upload portfolio.json",
                                        type="json")
            if uploaded:
                import_txns_json(uploaded.read().decode())
                st.success("Trades imported!"); st.rerun()

        with st.expander("🗑️ Delete / Clear"):
            del_id = st.selectbox(
                "Delete a transaction",
                options=[t.id for t in txns],
                format_func=lambda i: next(
                    f"{t.date} | {t.type} {t.shares} {t.ticker} @ ${t.price}"
                    for t in txns if t.id==i))
            if st.button("Delete selected"):
                save_txns([t for t in txns if t.id!=del_id])
                st.rerun()
            if st.button("⚠️ Clear ALL transactions"):
                save_txns([]); st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE
# ══════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("# 📈 Portfolio Performance")
    txns = get_txns()
    if not txns:
        st.info("Log trades in the Tracker tab first.")
        st.stop()

    with st.spinner("Reconstructing portfolio history…"):
        positions = compute_positions()
        tickers_h = list(positions.keys())
        start     = min(t.date for t in txns)
        hist_data = {}
        for tk_h in tickers_h:
            try:
                h = yf.Ticker(tk_h).history(start=start,auto_adjust=True)
                if not h.empty: hist_data[tk_h] = h["Close"]
            except: pass

    if not hist_data:
        st.warning("Could not fetch enough historical data.")
        st.stop()

    price_df  = pd.DataFrame(hist_data).ffill().dropna(how="all")
    price_df.index = pd.to_datetime(price_df.index).normalize()
    shares_df = pd.DataFrame(0.0,index=price_df.index,columns=tickers_h)
    for txn in sorted(txns,key=lambda t:t.date):
        if txn.ticker not in shares_df.columns: continue
        mask  = price_df.index>=pd.Timestamp(txn.date)
        delta = txn.shares if txn.type=="BUY" else -txn.shares
        shares_df.loc[mask,txn.ticker]+=delta
    value_df          = (shares_df*price_df).clip(lower=0)
    value_df["Total"] = value_df.sum(axis=1)

    total = value_df["Total"]
    dr    = total.pct_change().dropna()
    n_yrs = len(total)/252
    s_val = float(total.iloc[0]); e_val = float(total.iloc[-1])
    t_ret = (e_val/s_val-1) if s_val>0 else 0
    cagr  = (1+t_ret)**(1/n_yrs)-1 if n_yrs>0 else 0
    av    = float(dr.std()*np.sqrt(252))
    rf_d  = (1.045)**(1/252)-1
    sh    = float(((dr-rf_d).mean()/(dr-rf_d).std()*np.sqrt(252))) if dr.std()>0 else 0
    mdd   = float(((total-total.cummax())/total.cummax()).min()*100)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Start Value",   f"${s_val:,.2f}")
    m2.metric("Current Value", f"${e_val:,.2f}",
              delta=f"${e_val-s_val:,.2f}")
    m3.metric("Total Return",  f"{t_ret*100:+.2f}%")
    m4.metric("CAGR",          f"{cagr*100:+.2f}%")
    m5.metric("Sharpe",        f"{sh:.3f}")
    m6.metric("Max Drawdown",  f"{mdd:.2f}%")

    st.markdown("""
    <div class="callout">
    <strong>CAGR</strong> = smoothed annual growth rate.
    <strong>Sharpe</strong> = return per unit of risk taken (above 1.0 is strong).
    <strong>Max Drawdown</strong> = worst peak-to-trough drop — how bad the worst moment felt.
    </div>
    """, unsafe_allow_html=True)

    ticker_cols = [col for col in value_df.columns if col!="Total"]
    colors_h    = ["#3498db","#2ecc71","#e74c3c","#f39c12",
                   "#9b59b6","#1abc9c","#e67e22","#34495e"]

    fig,axes = plt.subplots(3,1,figsize=(13,11),facecolor="white",
                             gridspec_kw={"height_ratios":[3,1.5,1.5]})

    ax1 = axes[0]
    ax1.stackplot(value_df.index,
                  [value_df[t].values for t in ticker_cols],
                  labels=ticker_cols,
                  colors=colors_h[:len(ticker_cols)],alpha=0.65)
    ax1.plot(value_df.index,value_df["Total"],color="#2c3e50",lw=2,label="Total")
    ax1.set_title("Portfolio Value Over Time",fontsize=11,fontweight="bold")
    ax1.set_ylabel("Value ($)"); fmt_price(ax1)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2); ax1.set_facecolor("#fafafa")

    ax2 = axes[1]
    dd  = (total-total.cummax())/total.cummax()*100
    ax2.fill_between(total.index,dd,0,color="#c0392b",alpha=0.4)
    ax2.plot(total.index,dd,color="#c0392b",lw=1.2)
    ax2.set_title("Drawdown (%)",fontsize=10,fontweight="bold")
    ax2.set_ylabel("%"); ax2.grid(alpha=0.2); ax2.set_facecolor("#fafafa")

    ax3 = axes[2]
    dr_pct = dr*100
    ax3.hist(dr_pct,bins=60,color="#2980b9",alpha=0.8,edgecolor="white")
    ax3.axvline(0,color="#2c3e50",lw=1.2)
    ax3.axvline(float(dr_pct.mean()),color="#27ae60",lw=1.5,ls="--",
                label=f"Mean {dr_pct.mean():.2f}%/day")
    ax3.set_title("Daily Return Distribution",fontsize=10,fontweight="bold")
    ax3.set_xlabel("Daily Return (%)"); ax3.set_ylabel("Days")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2); ax3.set_facecolor("#fafafa")

    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    # Rolling Sharpe
    if len(dr) >= 90:
        st.markdown('<div class="section-header">📐 Rolling 90-Day Sharpe</div>',
                    unsafe_allow_html=True)
        rs = ((dr-rf_d).rolling(90).mean() /
              (dr-rf_d).rolling(90).std() * np.sqrt(252)).dropna()
        fig_rs,ax_rs = plt.subplots(figsize=(13,3.5),facecolor="white")
        ax_rs.plot(rs.index,rs,color="#8e44ad",lw=1.8)
        ax_rs.axhline(0,  color="#7f8c8d",ls="-", lw=0.8)
        ax_rs.axhline(1.0,color="#27ae60",ls="--",lw=1,label="Sharpe = 1.0")
        ax_rs.fill_between(rs.index,rs,0,where=rs>=0,
                           color="#27ae60",alpha=0.12)
        ax_rs.fill_between(rs.index,rs,0,where=rs<0,
                           color="#c0392b",alpha=0.12)
        ax_rs.set_ylabel("Sharpe (annualised)")
        ax_rs.legend(fontsize=8); ax_rs.grid(alpha=0.2)
        ax_rs.set_facecolor("#fafafa")
        st.pyplot(fig_rs,use_container_width=True); plt.close(fig_rs)

    # Allocation drift
    st.markdown('<div class="section-header">🔄 Allocation Drift</div>',
                unsafe_allow_html=True)
    st.caption("How your weights shifted as prices moved. Large drift = consider rebalancing.")
    drift = value_df[ticker_cols].div(value_df["Total"].replace(0,np.nan),axis=0)*100
    fig_d,ax_d = plt.subplots(figsize=(13,4),facecolor="white")
    ax_d.stackplot(drift.index,
                   [drift[t].values for t in ticker_cols],
                   labels=ticker_cols,
                   colors=colors_h[:len(ticker_cols)],alpha=0.75)
    ax_d.set_ylabel("Allocation (%)")
    ax_d.set_ylim(0,100)
    ax_d.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax_d.legend(fontsize=8,loc="upper left")
    ax_d.grid(alpha=0.2); ax_d.set_facecolor("#fafafa")
    ax_d.set_title("Portfolio Allocation Drift",fontsize=10,fontweight="bold")
    st.pyplot(fig_d,use_container_width=True); plt.close(fig_d)

    csv_h = value_df.to_csv().encode()
    st.download_button("⬇️ Export Portfolio History CSV",
                       data=csv_h, file_name="portfolio_history.csv",
                       mime="text/csv")

    st.markdown("""
    <div class="callout-warn">
    ⚠️ <strong>Disclaimer:</strong> For informational purposes only.
    Not financial advice. Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)
