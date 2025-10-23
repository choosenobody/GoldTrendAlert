#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# valuation_backtest_grid_v2.py
# Robust monthly backtest for valuation-only parameter grid
# - Rolling out-of-sample intercept (base) with trailing window
# - Gold: Stooq -> FRED -> Yahoo fallback
# - TIPS: FRED DFII10 (monthly mean)
# - CB YoY: from WGC_CSV_URL; roll=12 (fallback 6); monthly
# - Position rule: pos_t = clamp(score_val_t/20, -1, 1)
# - Returns: monthly gold return * lagged position (avoid lookahead)
# - Sharpe: annualized (sqrt(12)); MDD from equity curve
# - Output: .bot_state/valuation_grid_top20.csv

import os, io, math, sys
import pandas as pd, numpy as np, requests, yfinance as yf
from datetime import datetime, timedelta

TIMEOUT=25
UA={"User-Agent":"ValuationGrid/2.0"}

def env(k, d=None):
    v=os.environ.get(k)
    return v if (v is not None and str(v).strip()!="") else d

def clamp(x,a,b): return max(a, min(b, x))

# ---------------- Data loaders ----------------
def get_stooq_xau_series():
    try:
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
        r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date","Close"]).sort_values("Date")
        s = pd.to_numeric(df["Close"], errors="coerce").dropna(); s.index=df["Date"]
        return s if len(s)>0 else None
    except Exception:
        return None

def get_fred_series(series_id, days=365*12):
    key=env("FRED_API_KEY")
    if not key: return None
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":key,"file_type":"json",
            "observation_start":(datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")}
    try:
        r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        obs=r.json().get("observations",[])
        dates=[o["date"] for o in obs]
        vals=[(None if o.get("value") in (".", None) else float(o["value"])) for o in obs]
        s=pd.Series(vals, index=pd.to_datetime(dates, errors="coerce")).dropna()
        return s
    except Exception:
        return None

def get_gold_daily():
    s=get_stooq_xau_series()
    if s is None:
        sid = env("FRED_GOLD_SERIES_ID","GOLDPMGBD228NLBM")
        s=get_fred_series(sid, days=365*12)
    if s is None:
        try:
            df=yf.download("GC=F", period="10y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception:
            pass
    if s is None:
        try:
            df=yf.download("XAUUSD=X", period="10y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception:
            pass
    return s

def parse_wgc_csv_to_monthly(csv_text):
    df=pd.read_csv(io.StringIO(csv_text))
    # date col
    dcol=None
    for c in df.columns:
        lc=str(c).lower().replace(" ","")
        if lc in ("month","date","period","time","month_end","monthend","monthenddate"):
            dcol=c; break
    if dcol is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise")
                dcol=c; break
            except:
                pass
    # value col
    lower={str(c).lower().replace(" ",""): c for c in df.columns}
    vcol=None
    for name in ["netpurchasetons","net_purchase_tons","net","nettons","netcentralbankpurchases","cb_net_buy_tons","value","amount","netbuy","net_buy_tons","purchase_tons"]:
        if name in lower: vcol=lower[name]; break
    if vcol is None:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums: vcol=nums[0]
    if not dcol or not vcol: return None
    import re
    def to_num(x):
        if isinstance(x,str):
            s=x.strip().replace(",","")
            if re.match(r"^\(.*\)$", s): s="-"+s.strip("()")
            try: return float(s)
            except: return np.nan
        return x
    df=df[[dcol,vcol]].copy()
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df[vcol]=df[vcol].apply(to_num)
    df=df.dropna().sort_values(dcol)
    m=df[dcol].dt.to_period("M").dt.to_timestamp("M")
    s=df.assign(M=m).groupby("M")[vcol].sum()
    return s

def get_cb_yoy_series(rollN=12):
    url=env("WGC_CSV_URL")
    if not url: return None, rollN
    r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
    s=parse_wgc_csv_to_monthly(r.text)
    if s is None or len(s) < (2*rollN + 1):
        if s is not None and len(s) >= (2*6 + 1):
            rollN=6
        else:
            return None, rollN
    sums = s.rolling(rollN).sum().dropna()
    yoy = ((sums - sums.shift(rollN)) / sums.shift(rollN)) * 100.0
    return yoy.dropna(), rollN

def annualized_sharpe(monthly_returns):
    m = monthly_returns.dropna()
    if len(m) < 6: return -999.0
    vol = m.std()
    if vol == 0 or np.isnan(vol): return -999.0
    return float((m.mean() / vol) * math.sqrt(12.0))

def max_drawdown(series):
    s = series.dropna()
    if len(s) < 2: return 0.0
    roll_max = s.cummax()
    dd = (s/roll_max - 1.0)
    return float(dd.min())

def rolling_intercept(y, x_tips, x_cby, window):
    base = []
    idx = y.index
    for i in range(window, len(idx)):
        sl = slice(i-window, i)  # up to i-1
        Y = y.iloc[sl].values.astype(float)
        T = x_tips.iloc[sl].values.astype(float)
        C = x_cby.iloc[sl].values.astype(float)
        X = np.column_stack([np.ones(len(Y)), -T, C])
        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            base.append(coef[0])
        except Exception:
            base.append(np.nan)
    base = pd.Series(base, index=idx[window:])
    return base

def main():
    betas = [float(x) for x in (env("GRID_BETA","0.2,0.3,0.4").split(","))]
    alphas = [float(x) for x in (env("GRID_ALPHA","0.01,0.02,0.03").split(","))]
    gaps = [float(x) for x in (env("GRID_GAP_PCT","8,10,12").split(","))]
    reg_win = int(env("REG_WINDOW_M","36"))
    cb_roll = int(env("CB_YOY_ROLL_M","12"))

    s_gold_d = get_gold_daily()
    if s_gold_d is None or len(s_gold_d) < 400:
        print("[FATAL] gold series unavailable or too short"); sys.exit(2)
    m_gold = pd.Series(s_gold_d).resample("ME").last().dropna()

    s_tips_d = get_fred_series("DFII10", days=365*12)
    if s_tips_d is None or len(s_tips_d) < 200:
        print("[FATAL] TIPS series unavailable"); sys.exit(2)
    m_tips = s_tips_d.resample("ME").mean().dropna()

    cby, used_roll = get_cb_yoy_series(cb_roll)
    if cby is None:
        print("[FATAL] CB YoY series unavailable"); sys.exit(2)

    df = pd.concat([m_gold.rename("gold"), m_tips.rename("tips"), cby.rename("cby")], axis=1).dropna()
    if len(df) < reg_win + 18:
        print("[FATAL] aligned monthly sample too short:", len(df)); sys.exit(2)

    base_series = rolling_intercept(df["gold"], df["tips"], df["cby"], reg_win)
    df2 = pd.concat([df, base_series.rename("base")], axis=1).dropna()
    ret = df2["gold"].pct_change().dropna()

    results = []
    for beta in betas:
        for alpha in alphas:
            for gap_pct in gaps:
                fair = df2["base"] + beta*(0 - df2["tips"]) + alpha*df2["cby"]*df2["base"]
                gap = (df2["gold"] - fair)/fair
                score_val = np.clip(-gap/(gap_pct/100.0) * 20.0, -20.0, 20.0)
                pos = np.clip(score_val / 20.0, -1.0, 1.0)
                pos_lag = pd.Series(pos, index=df2.index).shift(1).reindex(ret.index).dropna()
                rets = ret.loc[pos_lag.index] * pos_lag
                if len(rets) < 12:
                    sharpe = -999.0; mdd = 0.0; n=0
                else:
                    sharpe = annualized_sharpe(rets)
                    eq = (1.0 + rets).cumprod()
                    mdd = max_drawdown(eq)
                    n = len(rets)
                results.append({"gap_pct": float(gap_pct),"beta": float(beta),"alpha": float(alpha),
                                "sharpe": float(sharpe),"mdd": float(mdd),"n": int(n)})

    out = pd.DataFrame(results)
    out = out.sort_values(["sharpe","mdd"], ascending=[False, False]).head(20)
    os.makedirs(".bot_state", exist_ok=True)
    out.to_csv(".bot_state/valuation_grid_top20.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
