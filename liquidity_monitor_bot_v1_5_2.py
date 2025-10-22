
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Monitor Bot
# Version: v1.5.2
#
# 更新点：
# - 估值阈值支持变量 VAL_GAP_PCT（默认 10），用于把 ±阈值 对应到 ±20 分
# - ETF 自适应窗口：若 (LONG,SHORT) 样本不足，自动尝试候选 "63:21,34:13,21:8"
# - 信号 1–4 维持 0–100 展示；估值维持 ±20 分制参与综合
# - 央行购金 YoY 仍支持 CB_YOY_ROLL_M=12（短样本自动退 6）
import os, sys, html, traceback, io, json, math
from datetime import datetime, timedelta
import requests, pandas as pd, yfinance as yf, numpy as np
from dateutil import tz

TIMEOUT=25
UA={"User-Agent":"LiquidityMonitorBot/1.5.2"}

def env(k,d=None):
    v=os.environ.get(k)
    return v if (v is not None and str(v).strip()!="") else d

def clamp(x,a,b): return max(a, min(b, x))
def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname); return datetime.now(tz=tzinfo)

# ----------------- Market helpers -----------------
def get_yahoo_last_and_wow(symbol, adj_by_ten=False):
    try:
        df = yf.download(symbol, period="14d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty: return None, None
        s = df["Close"].dropna()
        if len(s)<2: return None, None
        try: last = float(pd.to_numeric(s.iloc[-1]).item())
        except: last = float(s.iloc[-1])
        try: base = float(pd.to_numeric(s.iloc[max(0,len(s)-6)]).item())
        except: base = float(s.iloc[max(0,len(s)-6)])
        if adj_by_ten: last/=10.0; base/=10.0
        return round(last,4), round(last-base,4)
    except Exception: return None, None

def get_from_fred(series_id, days=120):
    key=env("FRED_API_KEY")
    if not key: return None, None, f"FRED {series_id}"
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":key,"file_type":"json",
            "observation_start":(datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")}
    try:
        r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        obs=r.json().get("observations",[])
        vals=[o for o in obs if o.get("value") not in (".", None)]
        if len(vals)<2: return None, None, f"FRED {series_id}"
        last=float(vals[-1]["value"]); prev=float(vals[-2]["value"])
        return round(last,4), round(last-prev,4), f"FRED {series_id}"
    except Exception: return None, None, f"FRED {series_id}"

def get_series_from_fred(series_id, days=365*10):
    key=env("FRED_API_KEY"); 
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
    except Exception: return None

def get_tips():
    # UST CSV 优先
    try:
        url="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-real-yield-curve-rates-csv"
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
        col=None
        if "10 Yr" in df.columns: col="10 Yr"
        else:
            for c in df.columns:
                if str(c).lower().replace(" ","") in ("10yr","10y"): col=c; break
        if col:
            s=pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s)>=2: return round(float(s.iloc[-1]),4), round(float(s.iloc[-1]-s.iloc[-2]),4), "UST Real Yields CSV"
    except Exception: pass
    return get_from_fred("DFII10", days=120)

def get_dxy_change(days=30):
    last,_,src = get_from_fred("DTWEXBGS", days=days+60)
    if last is None: return None, None, src
    s = get_series_from_fred("DTWEXBGS", days=days+365)
    if s is None or len(s)<days+1: return last, None, src
    base = s.iloc[-days-1]
    pct = (last-base)/base*100.0
    return last, pct, src

def get_stooq_xau_series():
    try:
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
        r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date","Close"]).sort_values("Date")
        s = pd.to_numeric(df["Close"], errors="coerce").dropna(); s.index=df["Date"]
        return s if len(s)>0 else None
    except Exception: return None

def get_fred_gold_series(days=5*365):
    key=env("FRED_API_KEY")
    if not key: return None
    url="https://api.stlouisfed.org/fred/series/observations"
    prefer = env("FRED_GOLD_SERIES_ID","GOLDPMGBD228NLBM")
    ids = [prefer, "GOLDAMGBD228NLBM"] if prefer!="GOLDAMGBD228NLBM" else [prefer,"GOLDPMGBD228NLBM"]
    for sid in ids:
        params={"series_id":sid,"api_key":key,"file_type":"json",
                "observation_start":(datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")}
        try:
            r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            obs=r.json().get("observations",[])
            vals=[float(o["value"]) for o in obs if o.get("value") not in (".", None)]
            if not vals: continue
            dates=[o["date"] for o in obs if o.get("value") not in (".", None)]
            return pd.Series(vals, index=pd.to_datetime(dates))
        except Exception: continue
    return None

def get_xau_series_daily():
    s=get_stooq_xau_series()
    if s is None:
        s=get_fred_gold_series(days=5*365)
    if s is None:
        try:
            df=yf.download("GC=F", period="5y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception: pass
    if s is None:
        try:
            df=yf.download("XAUUSD=X", period="5y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception: pass
    return s

def regression_intercept_base():
    window=int(env("REG_WINDOW_M","36"))
    s_gold=get_xau_series_daily()
    if s_gold is None: return None
    m_gold=pd.Series(s_gold).resample("ME").last().dropna()
    s_tips=get_series_from_fred("DFII10", days=365*10)
    if s_tips is None or len(s_tips)==0: return None
    m_tips=s_tips.resample("ME").mean().dropna()
    url=env("WGC_CSV_URL"); 
    if not url: return None
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        s_net=parse_wgc_netbuy_csv_to_monthly_series(r.text)
        if s_net is None: return None
        rollN=s_net.rolling(12).sum().dropna()
        if len(rollN)<13: return None
        yoy=((rollN - rollN.shift(12))/rollN.shift(12))*100.0
        m_cby=yoy.dropna()
    except Exception: return None
    df=pd.concat([m_gold.rename("gold"), m_tips.rename("tips"), m_cby.rename("cby")], axis=1).dropna()
    if len(df)<window+3: return None
    df=df.iloc[-window:]
    Y=df["gold"].values.astype(float)
    X=np.column_stack([np.ones(len(df)), -df["tips"].values.astype(float), df["cby"].values.astype(float)])
    try:
        coef,_res,_rank,_s=np.linalg.lstsq(X,Y,rcond=None)
        return float(coef[0])
    except Exception: return None

def get_xau_spot_and_base():
    s=get_xau_series_daily()
    cfg=str(env("GOLD_FAIR_BASE","AUTO")).upper()
    if s is None or len(s)==0:
        return None, (3500.0 if cfg in ("AUTO","REG") else float(env("GOLD_FAIR_BASE")))
    try: spot=float(pd.to_numeric(s.iloc[-1]).item())
    except: spot=float(s.iloc[-1])
    if cfg=="REG":
        base=regression_intercept_base()
        if base is None:
            try: base=float(s.ewm(span=252, adjust=False).mean().iloc[-1])
            except: base=3500.0
    elif cfg=="AUTO":
        try: base=float(s.ewm(span=252, adjust=False).mean().iloc[-1])
        except: base=3500.0
    else:
        base=float(env("GOLD_FAIR_BASE"))
    return spot, base

def parse_wgc_netbuy_csv_to_monthly_series(csv_text):
    df=pd.read_csv(io.StringIO(csv_text))
    dcol=None
    for c in df.columns:
        if str(c).lower() in ("month","date","period","time","month_end","monthend","monthenddate"):
            dcol=c; break
    if dcol is None:
        for c in df.columns:
            try: pd.to_datetime(df[c], errors="raise"); dcol=c; break
            except: pass
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
    df["ym"]=df[dcol].dt.to_period("M").dt.to_timestamp("M")
    s=df.groupby("ym")[vcol].sum()
    return s

def cb_yoy_pct_from_wgc():
    url=env("WGC_CSV_URL")
    if not url: return None, "WGC N/A"
    rollN=int(env("CB_YOY_ROLL_M","12"))
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        s=parse_wgc_netbuy_csv_to_monthly_series(r.text)
        if s is None or len(s)<(rollN+3): 
            rollN=6
        roll = s.rolling(rollN).sum().dropna()
        if len(roll) < (rollN+1):
            return None, "WGC roll too short"
        now = roll.iloc[-1]; prev = roll.iloc[-(rollN+1)]
        if prev==0: yoy = 100.0 if now>0 else -100.0
        else: yoy = (now - prev)/abs(prev)*100.0
        return float(yoy), f"WGC YoY(roll{rollN})"
    except Exception:
        return None, "WGC YoY N/A"

def auctions_quality_score(look_back_days=75):
    base="https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    endpoints=["v1/accounting/od/auctions_query","v1/accounting/od/auction_result"]
    data=None; endp=None
    for ep in endpoints:
        url=f"{base}/{ep}"
        params={"page[number]":"1","page[size]":"5000",
                "filter":f"auction_date:gte:{(datetime.utcnow()-timedelta(days=look_back_days+400)).strftime('%Y-%m-%d')}"}
        try:
            r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            j=r.json(); data=j.get("data", [])
            if data: endp=ep; break
        except Exception: continue
    if not data: return 0, "AuctionAPI N/A"
    df=pd.DataFrame(data)
    cols_lower={c.lower():c for c in df.columns}
    date_col=cols_lower.get("auction_date") or cols_lower.get("auctiondate") or cols_lower.get("auction_dt")
    btc_col =cols_lower.get("bid_to_cover_ratio") or cols_lower.get("bid_to_cover") or cols_lower.get("bidtocover")
    if not (date_col and btc_col): return 0, "Auction fields missing"
    df[date_col]=pd.to_datetime(df[date_col], errors="coerce")
    df=df.dropna(subset=[date_col, btc_col])
    recent=df[df[date_col] >= (datetime.utcnow()-timedelta(days=look_back_days))]
    if recent.empty: return 0, "Auction recent N/A"
    baseline=df[df[date_col] >= (datetime.utcnow()-timedelta(days=365))]
    recent_mean=pd.to_numeric(recent[btc_col], errors="coerce").mean()
    base_mean  =pd.to_numeric(baseline[btc_col], errors="coerce").mean()
    if pd.isna(recent_mean) or pd.isna(base_mean): return 0, "Auction baseline N/A"
    delta=recent_mean - base_mean
    score=clamp(delta/0.05 * 10, -25, 25)
    return round(score,1), f"{endp} ΔBTC={round(delta,3)}"

# ------------- ETF helpers ---------------
def etf_daily_series():
    url=env("BTC_ETF_API_URL")
    if url and (env("BTC_ETF_API_METHOD","GET").upper()=="POST"):
        headers=UA.copy()
        hdr=env("BTC_ETF_API_HEADERS")
        if hdr:
            try: headers.update(json.loads(hdr))
            except: pass
        body=env("BTC_ETF_API_BODY","{}")
        try: jbody=json.loads(body)
        except: jbody={"type":"us-btc-spot"}
        try:
            r=requests.post(url, headers=headers, json=jbody, timeout=TIMEOUT); r.raise_for_status()
            j=r.json(); data=j.get("data",{}); rows=data.get("list",[])
            if rows:
                df=pd.DataFrame(rows)
                dcol = "date" if "date" in df.columns else next((c for c in df.columns if str(c).lower() in ("date","day","datetime")), None)
                fcol = "totalNetInflow" if "totalNetInflow" in df.columns else next((c for c in df.columns if "inflow" in str(c).lower() or str(c).lower().endswith("flow")), None)
                if dcol and fcol:
                    df=df[[dcol,fcol]].rename(columns={dcol:"Date", fcol:"NetFlowUSD"})
                    df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
                    df["NetFlowUSD"]=pd.to_numeric(df["NetFlowUSD"], errors="coerce")
                    return df.dropna(subset=["Date","NetFlowUSD"]).sort_values("Date")
        except Exception: pass
    # fallback CSV
    try:
        csvu=env("BTC_ETF_FLOWS_CSV_URL")
        if csvu:
            r=requests.get(csvu, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            df=pd.read_csv(io.StringIO(r.text))
            c1=next((c for c in df.columns if str(c).lower() in ("date","day","datetime")), None)
            c2=next((c for c in df.columns if "flow" in str(c).lower()), None)
            if c1 and c2:
                df=df[[c1,c2]].rename(columns={c1:"Date", c2:"NetFlowUSD"})
                df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
                df["NetFlowUSD"]=pd.to_numeric(df["NetFlowUSD"], errors="coerce")
                return df.dropna(subset=["Date","NetFlowUSD"]).sort_values("Date")
    except Exception: pass
    # fallback Farside
    try:
        url = env("FARSIDE_ALLDATA_URL","https://farside.co.uk/bitcoin-etf-flow-all-data/")
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        tables=pd.read_html(r.text)
        cand=None; best=0
        for t in tables:
            if t.shape[1] < 2: continue
            df=t.copy()
            try:
                pd.to_datetime(df.iloc[:,0], errors="raise")
                if len(df)>best: cand=df; best=len(df)
            except: pass
        if cand is None: return None
        df=cand[[cand.columns[0], cand.columns[-1]]].copy()
        df.columns=["Date","NetFlowUSD"]
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        df["NetFlowUSD"]=pd.to_numeric(df["NetFlowUSD"], errors="coerce")
        return df.dropna().sort_values("Date")
    except Exception: return None

def choose_etf_windows(n_rows, prefer_long, prefer_short):
    # 若样本不足，尝试候选集
    try: prefer_long=int(prefer_long); prefer_short=int(prefer_short)
    except: prefer_long, prefer_short = 63, 21
    cands = env("ETF_WINDOW_CANDIDATES","63:21,34:13,21:8").split(",")
    pairs=[]
    for c in cands:
        try:
            L,S = c.split(":"); L=int(L); S=int(S)
            if L>S: pairs.append((L,S))
        except: pass
    if (n_rows >= prefer_long + 5) and (prefer_long > prefer_short):
        return prefer_long, prefer_short, "(prefer)"
    for L,S in pairs:
        if n_rows >= L + 5:
            return L, S, f"({L}:{S})"
    if n_rows >= 15:
        L = max(10, int(n_rows*0.6))
        S = max(5, int(L*0.4))
        if L>S:
            return L, S, "(auto)"
    return None, None, "(insufficient)"

def etf_score_from_series(df, short=21, long=63):
    s=df["NetFlowUSD"].astype(float)
    if len(s) < long+5: return 0, "ETF样本不足"
    ema_s=s.ewm(span=short, adjust=False).mean()
    ema_l=s.ewm(span=long, adjust=False).mean()
    slope=ema_s.diff(5).iloc[-1]
    bullish = (ema_s.iloc[-1] > ema_l.iloc[-1]) and (slope > 0)
    mad = (s - s.rolling(63).median()).abs().rolling(63).median().iloc[-1]
    norm = 0 if (pd.isna(mad) or mad==0) else float(slope) / float(mad)
    raw = clamp(norm, -1.5, 1.5) * 20   # → [-30,30]裁到[-20,20]
    score = clamp(raw, -20, +20)
    if bullish: score = min(20, score + 5)
    elif (ema_s.iloc[-1] < ema_l.iloc[-1]) and (slope < 0): score = max(-20, score - 5)
    return round(float(score),1), f"EMA{short}/{long}, slope5={round(float(slope),1)}"

def get_weekly_btc_etf_flows():
    # 简化：聚合 etf_daily_series 的本周（周一→周五，上海时区）
    df = etf_daily_series()
    if df is None or df.empty: return None, "ETF N/A"
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    dow = now_tz("Asia/Shanghai").weekday()
    monday = (now_tz("Asia/Shanghai") - timedelta(days=dow)).date()
    friday = (now_tz("Asia/Shanghai") + timedelta(days=(4-dow))).date()
    val=float(df.loc[(df["Date"]>=monday)&(df["Date"]<=friday), "NetFlowUSD"].sum())
    return val, "API/CSV/Farside"

# ------------- Valuation ---------------
def valuation_gap_score(xau_spot, base, tips, cb_yoy_pct):
    beta=float(env("BETA_TIPS","0.3"))
    alpha=float(env("ALPHA_CB","0.02"))
    cb = 0.0 if cb_yoy_pct is None else float(cb_yoy_pct)
    if None in (xau_spot, base, tips): return 0, "估值缺数据"
    fair = base + beta * (0 - float(tips)) + alpha * cb * base
    gap=(xau_spot - fair)/fair
    gap_pct = float(env("VAL_GAP_PCT","10"))
    score = clamp(-gap/(gap_pct/100.0) * 20, -20, +20)  # ±VAL_GAP_PCT% → ∓20分
    note = f"基准={round(base,1)}；公允={round(fair,1)}；偏离={round(gap*100,2)}%"
    if cb_yoy_pct is None: note += "（CB YoY=0 兜底）"
    if str(env("GOLD_FAIR_BASE","")).upper()=="REG": note += "（基准=回归截距）"
    return round(score,1), note

# ------------- Main --------------------
def main():
    mode=(env("MODE","send") or "send").strip().lower()
    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if mode=="send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试用 MODE=status"); sys.exit(2)

    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)
    tips, tips_wow, tips_src = get_tips()
    dxy, dxy_30d_pct, dxy_src = get_dxy_change(days=int(env("DXY_LOOKBACK_D","30")))

    # ETF
    etf_df_all = etf_daily_series()
    if etf_df_all is not None:
        Lpref = int(env("ETF_LONG_D","63"))
        Spref = int(env("ETF_SHORT_D","21"))
        L,S, tag = choose_etf_windows(len(etf_df_all), Lpref, Spref)
        if L is None:
            etf_score, etf_note = 0, "ETF样本不足"
        else:
            etf_score, etf_note = etf_score_from_series(etf_df_all, short=S, long=L)
            etf_note = f"{etf_note} {tag}"
    else:
        etf_score, etf_note = 0, "ETF不可用"

    etf_weekly, etf_src = (None, "ETF N/A")
    try:
        etf_weekly, etf_src = get_weekly_btc_etf_flows()
    except Exception:
        pass

    # Auctions
    auc_score, auc_note = auctions_quality_score(look_back_days=int(env("AUCTION_WINDOW_D","75")))

    # Valuation
    xau_spot, base = get_xau_spot_and_base()
    cb_yoy, cb_note = cb_yoy_pct_from_wgc()
    val_score, val_note = valuation_gap_score(xau_spot, base, tips, cb_yoy)

    # 原始打分（-40..+40 等区间）
    s_tips = 0 if (tips is None or tips_wow is None) else clamp(-tips_wow/0.10 * 10, -40, 40)
    s_dxy  = 0 if (dxy_30d_pct is None) else clamp(-dxy_30d_pct/1.5 * 5, -15, 15)
    s_etf  = etf_score
    s_auc  = auc_score
    s_val  = val_score

    # 显示层：1-4 → 0–100
    def to100(x, lo, hi):
        if x is None: return "N/A"
        return round((clamp(x, lo, hi) - lo)/(hi - lo) * 100.0, 1)
    s1_100 = to100(s_tips, -40, 40)
    s2_100 = to100(s_dxy,  -15, 15)
    s3_100 = to100(s_etf,  -20, 20)
    s4_100 = to100(s_auc,  -25, 25)

    raw = s_tips + s_dxy + s_etf + s_auc + s_val
    comp = clamp((raw + 120) * 100.0 / 240.0, 0, 100)

    if comp >= 70: view="【偏多持有】保留核心仓位，逢回加一点。"; pos_delta="+3%"
    elif comp >= 55: view="【中性偏多】耐心持有，节奏为先。"; pos_delta="+1%"
    elif comp >= 45: view="【中性】观望，等待信号共振。"; pos_delta="0%"
    elif comp >= 30: view="【偏谨慎】降波动，减轻杠杆。"; pos_delta="-1%"
    else: view="【防守】现金/短债为主，控制回撤。"; pos_delta="-3%"

    ts_str = now_tz("Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
    def fmt(v,d=4): return "N/A" if v is None else f"{float(v):.{d}f}"

    lines=[]
    lines.append(f"<b>盘中快评 / 状态查询</b>   {html.escape(ts_str)}")
    lines.append(f"• <b>黄金现价（USD/oz）</b> ≈ <b>${fmt(xau_spot,2)}</b>  <i>（多源：Stooq/FRED/Yahoo）</i>")
    lines.append("")

    lines.append("<b>信号与评分（展示为 0–100）</b>")
    lines.append(f"1）10Y 实际利率 DFII10：{fmt(tips,3)}（WoW {fmt(tips_wow,3)}，{tips_src}） → <b>{s1_100}</b>")
    lines.append(f"2）美元指数（广义）{env('DXY_LOOKBACK_D','30')}日变动：{fmt(dxy,2)}（{('+' if (dxy_30d_pct or 0)>=0 else '')}{'N/A' if dxy_30d_pct is None else f'{float(dxy_30d_pct):.2f}%'}；{dxy_src}） → <b>{s2_100}</b>")
    lines.append(f"3）ETF 净流向趋势（EMA＋斜率）：{etf_note} → <b>{s3_100}</b>")
    lines.append(f"4）美债拍卖质量（近{env('AUCTION_WINDOW_D','75')}天 vs 近年 Bid-to-Cover）：{auc_note} → <b>{s4_100}</b>")
    lines.append(f"5）估值偏离（Valuation）：{val_note} → <b>{round(s_val,1)}</b>（±20分制）")
    lines.append("")
    lines.append(f"<b>综合分（0–100）：{int(round(comp,0))}</b>")
    lines.append(f"<b>结论：</b>{view}  → 建议战术仓位调整：<b>{pos_delta}</b>")
    lines.append("")

    rows=[
        ("10Y名义(%)", fmt(tnx,3), fmt(tnx_wow,3), "Yahoo ^TNX"),
        ("10Y真实(%)", fmt(tips,3), fmt(tips_wow,3), tips_src),
        ("DXY(30d变动%)", "N/A" if dxy_30d_pct is None else f"{float(dxy_30d_pct):.2f}%", "—", dxy_src),
        ("VIX", fmt(vix,2), fmt(vix_wow,2), "Yahoo ^VIX"),
        ("BTC ETF净流(周)", "N/A" if etf_weekly is None else f"{float(etf_weekly):,.0f}", "—", "API/CSV/Farside"),
        ("央行购金YoY(%)", "N/A" if cb_yoy is None else f"{float(cb_yoy):.1f}%", "—", cb_note),
        ("估值阈值(±%)", str(env("VAL_GAP_PCT","10")), "—", "±20分映射阈值"),
        ("估值基准BASE", fmt(base,1), "—", "REG" if str(env("GOLD_FAIR_BASE","")).upper()=="REG" else ("AUTO" if str(env("GOLD_FAIR_BASE","")).upper()=="AUTO" else "Static")),
    ]
    w1=max(len(r[0]) for r in rows); w2=max(len(str(r[1])) for r in rows); w3=max(len(str(r[2])) for r in rows)
    header=f"{'指标'.ljust(w1)}  {'最新'.ljust(w2)}  {'WoW'.ljust(w3)}  来源"
    sep="-"*len(header)
    table_lines=[header,sep]+[f"{r[0].ljust(w1)}  {str(r[1]).ljust(w2)}  {str(r[2]).ljust(w3)}  {r[3]}" for r in rows]
    table_html = html.escape("\n".join(table_lines))
    lines.append("<b>数据小表</b>：")
    lines.append(f"<pre><code>{table_html}</code></pre>")

    msg="\n".join(lines)

    if mode=="status":
        print(msg); return 0
    url=f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/sendMessage"
    payload={"chat_id":env("TELEGRAM_CHAT_ID"), "text":msg, "parse_mode":"HTML", "disable_web_page_preview":"true"}
    r=requests.post(url, data=payload, timeout=TIMEOUT)
    ok = r.status_code==200 and r.json().get("ok") is True
    if not ok:
        print("[ERROR] 发送Telegram失败：", r.text); return 3
    print("[OK] 已发送Telegram消息。"); return 0

if __name__=="__main__":
    try: sys.exit(main())
    except SystemExit as e: raise
    except Exception as e:
        print("[FATAL]", e); traceback.print_exc(); sys.exit(9)
