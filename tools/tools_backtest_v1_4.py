
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Quick backtest for v1_4_0 scoring bands
import os, io, json, math, sys
from datetime import datetime, timedelta
import pandas as pd, requests
import numpy as np

UA={"User-Agent":"Backtest/1.0"}
TIMEOUT=25

def fred_series(series_id, api_key, start):
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":api_key,"file_type":"json","observation_start":start}
    r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
    obs=r.json().get("observations",[])
    df=pd.DataFrame([{"date":o["date"],"value":None if o["value"] in (".",None) else float(o["value"])} for o in obs])
    df["date"]=pd.to_datetime(df["date"]).dt.date
    return df.dropna()

def farside_daily():
    url="https://farside.co.uk/bitcoin-etf-flow-all-data/"
    r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
    tables=pd.read_html(r.text)
    cand=None; best=0
    for t in tables:
        if t.shape[1]<2: continue
        df=t.copy()
        try:
            pd.to_datetime(df.iloc[:,0], errors="raise")
            if len(df)>best: cand=df; best=len(df)
        except: pass
    if cand is None: return None
    df=cand[[cand.columns[0], cand.columns[-1]]].copy()
    df.columns=["date","netflow"]
    df["date"]=pd.to_datetime(df["date"]).dt.date
    df["netflow"]=pd.to_numeric(df["netflow"], errors="coerce")
    return df.dropna()

def yfinance_xau():
    import yfinance as yf
    df=yf.download("XAUUSD=X", period="3y", interval="1d", progress=False, auto_adjust=False)
    s=df["Close"].dropna().reset_index()
    s["date"]=pd.to_datetime(s["Date"]).dt.date
    return s[["date","Close"]].rename(columns={"Close":"xau"})

def score_row(tips_wow, dxy_30d):
    s_tips = np.clip(-tips_wow/0.10 * 10, -40, 40) if tips_wow is not None else 0
    s_dxy  = np.clip(-dxy_30d/1.5 * 5, -15, 15) if dxy_30d is not None else 0
    return s_tips + s_dxy

def main():
    api_key=os.environ.get("FRED_API_KEY")
    if not api_key:
        print("Set FRED_API_KEY"); sys.exit(2)
    tips=fred_series("DFII10", api_key, "2022-01-01")
    dxy =fred_series("DTWEXBGS", api_key, "2022-01-01")
    etf =farside_daily()
    xau =yfinance_xau()
    if etf is None:
        print("Farside parse failed"); sys.exit(3)
    # build daily features
    df=pd.DataFrame({"date":sorted(list(set(tips["date"])|set(dxy["date"])|set(etf["date"])|set(xau["date"])))})
    df=df.merge(tips, on="date", how="left").rename(columns={"value":"tips"})
    df=df.merge(dxy,  on="date", how="left").rename(columns={"value":"dxy"})
    df=df.merge(etf,  on="date", how="left")
    df=df.merge(xau,  on="date", how="left")
    df=df.sort_values("date").ffill()
    # features
    df["tips_wow"]=df["tips"].diff(5)
    df["dxy_30d"]=df["dxy"].pct_change(30)*100
    # simple etf score proxy: 21>63 and slope>0
    df["ema21"]=df["netflow"].ewm(span=21, adjust=False).mean()
    df["ema63"]=df["netflow"].ewm(span=63, adjust=False).mean()
    df["slope5"]=df["ema21"].diff(5)
    df["etf_score"]=np.where((df["ema21"]>df["ema63"]) & (df["slope5"]>0), 10, -10)  # proxy
    df["raw"]=df.apply(lambda r: score_row(r["tips_wow"], r["dxy_30d"]) + r["etf_score"], axis=1)
    df["comp"]=(df["raw"]+120)*100/240
    # strategy: comp>=70 -> +1 unit; 55-70 -> +0.5; 45-55 -> 0; 30-45 -> -0.5; <30 -> -1
    conds=[df["comp"]>=70, df["comp"]>=55, df["comp"]>=45, df["comp"]>=30]
    choices=[1.0, 0.5, 0.0, -0.5]
    df["pos"]=np.select(conds, choices, default=-1.0)
    df["ret"]=df["xau"].pct_change().fillna(0)
    df["pnl"]=df["pos"].shift(1).fillna(0)*df["ret"]
    sharpe = np.sqrt(252)*df["pnl"].mean()/df["pnl"].std() if df["pnl"].std()!=0 else 0
    mdd = ((df["pnl"].cumsum().cummax()-df["pnl"].cumsum()).max())
    print(f"Sharpe≈{sharpe:.2f}, MaxDrawdown≈{mdd:.3f}")
    print(df[["date","comp","pos","ret","pnl"]].tail(10).to_string(index=False))

if __name__=="__main__":
    main()
