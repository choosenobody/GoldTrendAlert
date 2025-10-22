
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Weekly Monitor Bot + Scoring & Position
# Version: v1.5.0
#
# 变更摘要：
#  - 央行购金 YoY：改为 “12个月滚动总净购金” 的同比百分比（更稳健，不依赖月对月严格对齐）
#  - 估值基准：新增 GOLD_FAIR_BASE=REG，用近 REG_WINDOW_M（月，默认36）做 OLS 回归截距作为“基准”
#  - 估值缺数据修复：多源金价+TIPS+WGC容错；CB YoY 不可得时设为0并标注
#  - 消息排版：更清晰的标题+留白，直接展示 0–100 综合分，不再展示 raw→映射公式
#  - 其余：pandas 标量取值、ETF 历史序列容错、FRED 黄金系列可用 FRED_GOLD_SERIES_ID 指定
#
import os, sys, html, traceback, io, json, math
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
from dateutil import tz
import numpy as np

TIMEOUT = 25
UA = {"User-Agent":"Mozilla/5.0 (LiquidityMonitorBot/1.5.0)"}

def env(key, default=None):
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default

def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tz=tzinfo)

def clamp(x, a, b): return max(a, min(b, x))

# ---------- Market helpers ----------
def get_yahoo_last_and_wow(symbol, adj_by_ten=False):
    try:
        df = yf.download(symbol, period="14d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None
        s = df["Close"].dropna()
        if len(s) < 2:
            return None, None
        # 规避 FutureWarning
        try:
            last_scalar = pd.to_numeric(s.iloc[-1]).item()
        except Exception:
            last_scalar = float(s.iloc[-1])
        try:
            base_scalar = pd.to_numeric(s.iloc[max(0, len(s)-6)]).item()
        except Exception:
            base_scalar = float(s.iloc[max(0, len(s)-6)])
        last = float(last_scalar)
        base = float(base_scalar)
        if adj_by_ten:
            last /= 10.0
            base /= 10.0
        return round(last, 4), round(last - base, 4)
    except Exception:
        return None, None

def get_tips_from_treasury_csv():
    url="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-real-yield-curve-rates-csv"
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None, None, None
    col = "10 Yr" if "10 Yr" in df.columns else None
    if not col:
        for c in df.columns:
            if str(c).lower().replace(" ","") in ("10yr","10y"): col=c; break
    if not col: return None, None, None
    s=pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s)<2: return None, None, None
    return round(float(s.iloc[-1]),4), round(float(s.iloc[-1]-s.iloc[-2]),4), "UST Real Yields CSV"

def get_from_fred(series_id, days=120):
    api_key=env("FRED_API_KEY")
    if not api_key: return None, None, f"FRED {series_id}"
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":api_key,"file_type":"json",
            "observation_start":(datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")}
    try:
        r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        obs=r.json().get("observations",[])
        vals=[o for o in obs if o.get("value") not in (".", None)]
        if len(vals)<2: return None, None, f"FRED {series_id}"
        last=float(vals[-1]["value"]); prev=float(vals[-2]["value"])
        return round(last,4), round(last-prev,4), f"FRED {series_id}"
    except Exception:
        return None, None, f"FRED {series_id}"

def get_series_from_fred(series_id, days=365*10):
    api_key=env("FRED_API_KEY")
    if not api_key: return None
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":api_key,"file_type":"json",
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

def get_tips():
    v = get_tips_from_treasury_csv()
    if v[0] is not None: return v
    return get_from_fred("DFII10", days=120)

def get_dxy_change(days=30):
    last, _, src = get_from_fred("DTWEXBGS", days=days+60)
    if last is None: return None, None, src
    s = get_series_from_fred("DTWEXBGS", days=days+365)
    if s is None or len(s)<days+1: return last, None, src
    base = s.iloc[-days-1]
    pct = (last-base)/base*100.0
    return last, pct, src

# ---------- ETF ----------
def week_range_shanghai(now_dt):
    dow = now_dt.weekday()
    monday = now_dt - timedelta(days=dow)
    monday = monday.replace(hour=0,minute=0,second=0,microsecond=0)
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)
    return monday.date(), friday.date()

def parse_week_sum_from_dataframe(df):
    if "Date" not in df.columns or "NetFlowUSD" not in df.columns: return None
    df["Date"]=pd.to_datetime(df["Date"], errors="coerce").dt.date
    df=df.dropna(subset=["Date","NetFlowUSD"])
    start,end = week_range_shanghai(now_tz("Asia/Shanghai"))
    mask=(df["Date"]>=start)&(df["Date"]<=end)
    val=pd.to_numeric(df.loc[mask,"NetFlowUSD"], errors="coerce").sum()
    return float(val)

def etf_from_csv_url(url):
    if not url: return None, "No CSV URL"
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
        col_date=None
        for c in df.columns:
            if str(c).lower() in ("date","day","datetime"): col_date=c; break
        col_flow=None
        for c in df.columns:
            if str(c).lower() in ("netflowusd","flowusd","net_flow_usd","netflow","flow","totalnetinflow"):
                col_flow=c; break
        if not col_date or not col_flow: return None, "CSV(url) columns"
        df=df[[col_date,col_flow]].rename(columns={col_date:"Date", col_flow:"NetFlowUSD"})
        return parse_week_sum_from_dataframe(df), "CSV(url)"
    except Exception as e:
        return None, f"CSV(url) error:{e.__class__.__name__}"

def etf_from_api():
    url = env("BTC_ETF_API_URL")
    if not url: return None, "No API URL"
    method = (env("BTC_ETF_API_METHOD","GET") or "GET").upper()
    headers = UA.copy()
    hdr_json = env("BTC_ETF_API_HEADERS")
    if hdr_json:
        try: headers.update(json.loads(hdr_json))
        except: pass
    body_json = None
    body = env("BTC_ETF_API_BODY")
    if body:
        try: body_json = json.loads(body)
        except: body_json = body
    try:
        if method == "POST":
            r = requests.post(url, headers=headers, json=body_json if isinstance(body_json, dict) else None,
                              data=None if isinstance(body_json, dict) else body_json, timeout=TIMEOUT)
        else:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        data = j.get("data", j) if isinstance(j, dict) else j
        rows = None
        if isinstance(data, dict) and isinstance(data.get("list"), list):
            rows = data["list"]
        elif isinstance(data, list):
            rows = data
        if not rows: return None, "API no rows"
        df = pd.DataFrame(rows)
        date_field = env("BTC_ETF_API_DATE_FIELD","date")
        flow_field = env("BTC_ETF_API_FLOW_FIELD","totalNetInflow")
        if date_field not in df.columns or flow_field not in df.columns:
            dcol=None
            for c in df.columns:
                if str(c).lower() in ("date","day","datetime"): dcol=c; break
            fcol=None
            for c in df.columns:
                lc=str(c).lower()
                if ("inflow" in lc) or lc.endswith("flow") or lc.endswith("net"):
                    fcol=c; break
            if not dcol or not fcol: return None, "API columns missing"
            date_field, flow_field = dcol, fcol
        df = df[[date_field, flow_field]].rename(columns={date_field:"Date", flow_field:"NetFlowUSD"})
        df["NetFlowUSD"]=pd.to_numeric(df["NetFlowUSD"], errors="coerce")
        return parse_week_sum_from_dataframe(df), f"API({method})"
    except Exception as e:
        return None, f"API error:{e.__class__.__name__}"

def get_weekly_btc_etf_flows():
    v, src = etf_from_csv_url(env("BTC_ETF_FLOWS_CSV_URL"))
    if v is not None: return v, src
    v, src = etf_from_api()
    if v is not None: return v, src
    # fallback Farside
    url = env("FARSIDE_ALLDATA_URL","https://farside.co.uk/bitcoin-etf-flow-all-data/")
    try:
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
        if cand is None: return None, "Farside(parse-none)"
        df=cand[[cand.columns[0], cand.columns[-1]]].copy()
        df.columns=["Date","NetFlowUSD"]
        return parse_week_sum_from_dataframe(df), "Farside(HTML)"
    except Exception as e:
        return None, f"Farside(error:{e.__class__.__name__})"

# ---------- GOLD ----------
def get_stooq_xau_series():
    try:
        url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
        r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        s.index = df["Date"]
        return s if len(s)>0 else None
    except Exception:
        return None

def get_fred_gold_series(days=5*365):
    api_key = env("FRED_API_KEY")
    if not api_key: return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    prefer = env("FRED_GOLD_SERIES_ID", "GOLDPMGBD228NLBM")
    series_ids = [prefer, "GOLDAMGBD228NLBM"] if prefer != "GOLDAMGBD228NLBM" else [prefer, "GOLDPMGBD228NLBM"]
    for sid in series_ids:
        params = {"series_id": sid,"api_key": api_key,"file_type":"json",
                  "observation_start": (datetime.utcnow()-timedelta(days=days)).strftime("%Y-%m-%d")}
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            obs = r.json().get("observations", [])
            vals = [float(o["value"]) for o in obs if o.get("value") not in (".", None)]
            if len(vals)==0: 
                continue
            dates = [o["date"] for o in obs if o.get("value") not in (".", None)]
            return pd.Series(vals, index=pd.to_datetime(dates))
        except Exception:
            continue
    return None

def get_xau_series_daily():
    s = get_stooq_xau_series()
    if s is None:
        s = get_fred_gold_series(days=5*365)
    if s is None:
        try:
            df = yf.download("GC=F", period="5y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                s = df["Close"].dropna()
        except Exception:
            s = None
    if s is None:
        try:
            df = yf.download("XAUUSD=X", period="5y", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                s = df["Close"].dropna()
        except Exception:
            s = None
    return s

def get_xau_spot_and_base(base_mode=None):
    s = get_xau_series_daily()
    base_cfg = env("GOLD_FAIR_BASE", "AUTO") if base_mode is None else base_mode

    if s is None or len(s)==0:
        spot = None
        base = 3500.0 if str(base_cfg).upper() in ("AUTO","REG") else float(base_cfg)
        return spot, base

    try:
        spot = float(pd.to_numeric(s.iloc[-1]).item())
    except Exception:
        spot = float(s.iloc[-1])

    mode = str(base_cfg).upper()
    if mode == "AUTO":
        try:
            base = float(s.ewm(span=252, adjust=False).mean().iloc[-1])
        except Exception:
            base = 3500.0
    elif mode == "REG":
        base = regression_intercept_base()
        if base is None:
            try:
                base = float(s.ewm(span=252, adjust=False).mean().iloc[-1])
            except Exception:
                base = 3500.0
    else:
        base = float(base_cfg)

    return spot, base

# ---------- WGC ----------
def parse_wgc_netbuy_csv_to_monthly_series(csv_text):
    df=pd.read_csv(io.StringIO(csv_text))
    dcol=None
    for c in df.columns:
        if str(c).lower() in ("month","date","period","time","month_end"): dcol=c; break
    if dcol is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise"); dcol=c; break
            except: pass
    lower={str(c).lower().replace(" ",""): c for c in df.columns}
    vcol=None
    candidates=["netpurchasetons","net_purchase_tons","net","nettons","netcentralbankpurchases","cb_net_buy_tons","value","amount"]
    for name in candidates:
        if name in lower: vcol=lower[name]; break
    if vcol is None:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums: vcol=nums[0]
    if not dcol or not vcol:
        return None
    df=df[[dcol,vcol]].copy()
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df[vcol]=pd.to_numeric(df[vcol], errors="coerce")
    df=df.dropna().sort_values(dcol)
    df["ym"]=df[dcol].dt.to_period("M").dt.to_timestamp("M")
    s=df.groupby("ym")[vcol].sum()
    return s

def cb_yoy_pct_from_wgc():
    url=env("WGC_CSV_URL")
    if not url:
        return None, "WGC N/A"
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        s = parse_wgc_netbuy_csv_to_monthly_series(r.text)
        if s is None or len(s)<15:
            return None, "WGC columns/len"
        roll12 = s.rolling(12).sum().dropna()
        if len(roll12) < 13:
            return None, "WGC roll12 too short"
        now = roll12.iloc[-1]
        prev = roll12.iloc[-13]
        if prev == 0:
            yoy = 100.0 if now>0 else -100.0
        else:
            yoy = (now - prev)/abs(prev) * 100.0
        return float(yoy), "WGC YoY(roll12)"
    except Exception:
        return None, "WGC YoY N/A"

# ---------- Regression intercept base ----------
def get_series_from_fred(series_id, days=365*10):
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

def regression_intercept_base():
    window = int(env("REG_WINDOW_M","36"))
    s_gold = get_xau_series_daily()
    if s_gold is None: 
        return None
    m_gold = pd.Series(s_gold).resample("M").last().dropna()
    s_tips = get_series_from_fred("DFII10", days=365*10)
    if s_tips is None or len(s_tips)==0:
        return None
    m_tips = s_tips.resample("M").mean().dropna()
    url=env("WGC_CSV_URL")
    if not url:
        return None
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        s_net = parse_wgc_netbuy_csv_to_monthly_series(r.text)
        if s_net is None:
            return None
        roll12 = s_net.rolling(12).sum().dropna()
        if len(roll12) < 13:
            return None
        yoy = (roll12 - roll12.shift(12)) / roll12.shift(12)
        m_cby = (yoy*100.0).dropna()
    except Exception:
        return None
    df = pd.concat([m_gold.rename("gold"), m_tips.rename("tips"), m_cby.rename("cby")], axis=1).dropna()
    if len(df) < window+3:
        return None
    df = df.iloc[-window:]
    y = df["gold"].values.astype(float)
    X = np.column_stack([np.ones(len(df)), -df["tips"].values.astype(float), df["cby"].values.astype(float)])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(coef[0])
        return intercept
    except Exception:
        return None

# ---------- Valuation ----------
def valuation_gap_score(xau_spot, base, tips, cb_yoy_pct):
    try:
        beta=float(env("BETA_TIPS","0.3"))
        alpha=float(env("ALPHA_CB","0.02"))
        cb = 0.0 if cb_yoy_pct is None else float(cb_yoy_pct)
        if None in (xau_spot, base, tips):
            return 0, "估值缺数据"
        fair = base + beta * (0 - float(tips)) + alpha * cb * base
        gap=(xau_spot - fair)/fair
        score = clamp(-gap/0.10 * 20, -20, +20)
        note = f"基准={round(base,1)}；公允={round(fair,1)}；偏离={round(gap*100,2)}%"
        if cb_yoy_pct is None: note += "（CB YoY=0 兜底）"
        if str(env("GOLD_FAIR_BASE","")).upper()=="REG": note += "（基准=回归截距）"
        return round(score,1), note
    except Exception:
        return 0, "估值异常"

# ---------- Auctions ----------
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
        except Exception:
            continue
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

# ---------- ETF series & scoring ----------
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
        except Exception:
            pass
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
    except Exception:
        return None

def etf_score_from_series(df, short=21, long=63):
    s=df["NetFlowUSD"].astype(float)
    if len(s) < long+5: return 0, "ETF样本不足"
    ema_s = s.ewm(span=short, adjust=False).mean()
    ema_l = s.ewm(span=long, adjust=False).mean()
    slope = ema_s.diff(5).iloc[-1]
    bullish = (ema_s.iloc[-1] > ema_l.iloc[-1]) and (slope > 0)
    mad = (s - s.rolling(63).median()).abs().rolling(63).median().iloc[-1]
    norm = 0 if (pd.isna(mad) or mad==0) else slope / mad
    raw = clamp(norm, -1.5, 1.5) * 20
    score = clamp(raw, -20, +20)
    if bullish: score = min(20, score + 5)
    elif (ema_s.iloc[-1] < ema_l.iloc[-1]) and (slope < 0): score = max(-20, score - 5)
    return round(float(score),1), f"EMA{short}/{long}, slope5={round(float(slope),1)}"

# ---------- Regression intercept ----------
def regression_intercept_base():
    window = int(env("REG_WINDOW_M","36"))
    s_gold = get_xau_series_daily()
    if s_gold is None: 
        return None
    m_gold = pd.Series(s_gold).resample("M").last().dropna()
    s_tips = get_series_from_fred("DFII10", days=365*10)
    if s_tips is None or len(s_tips)==0:
        return None
    m_tips = s_tips.resample("M").mean().dropna()
    url=env("WGC_CSV_URL")
    if not url:
        return None
    try:
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        s_net = parse_wgc_netbuy_csv_to_monthly_series(r.text)
        if s_net is None:
            return None
        roll12 = s_net.rolling(12).sum().dropna()
        if len(roll12) < 13:
            return None
        yoy = (roll12 - roll12.shift(12)) / roll12.shift(12)
        m_cby = (yoy*100.0).dropna()
    except Exception:
        return None
    df = pd.concat([m_gold.rename("gold"), m_tips.rename("tips"), m_cby.rename("cby")], axis=1).dropna()
    if len(df) < window+3:
        return None
    df = df.iloc[-window:]
    y = df["gold"].values.astype(float)
    X = np.column_stack([np.ones(len(df)), -df["tips"].values.astype(float), df["cby"].values.astype(float)])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(coef[0])
        return intercept
    except Exception:
        return None

# ---------- Main ----------
def main():
    mode=(env("MODE","send") or "send").strip().lower()
    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if mode=="send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试请使用 MODE=status"); sys.exit(2)

    # Snapshots
    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)
    tips, tips_wow, tips_src = get_tips()
    dxy, dxy_30d_pct, dxy_src = get_dxy_change(days=int(env("DXY_LOOKBACK_D","30")))

    # ETF flows & trend score
    etf_weekly, etf_src = get_weekly_btc_etf_flows()
    etf_df = etf_daily_series()
    etf_score, etf_note = (0,"ETF样本不足") if etf_df is None else etf_score_from_series(etf_df, short=int(env("ETF_SHORT_D","21")), long=int(env("ETF_LONG_D","63")))

    # Auctions
    auc_score, auc_note = auctions_quality_score(look_back_days=int(env("AUCTION_WINDOW_D","75")))

    # Valuation
    xau_spot, base = get_xau_spot_and_base()  # 支持 AUTO/REG
    cb_yoy, cb_note = cb_yoy_pct_from_wgc()   # 12m滚动YoY
    val_score, val_note = valuation_gap_score(xau_spot, base, tips, cb_yoy)

    # Scoring combine → 直接 0–100
    s_tips = 0 if (tips is None or tips_wow is None) else clamp(-tips_wow/0.10 * 10, -40, 40)
    s_dxy  = 0 if (dxy_30d_pct is None) else clamp(-dxy_30d_pct/1.5 * 5, -15, 15)
    s_etf  = etf_score
    s_auc  = auc_score
    s_val  = val_score
    raw = s_tips + s_dxy + s_etf + s_auc + s_val
    comp = clamp((raw + 120) * 100.0 / 240.0, 0, 100)

    if comp >= 70: view="【偏多持有】保留核心仓位，逢回加一点。"; pos_delta="+3%"
    elif comp >= 55: view="【中性偏多】耐心持有，节奏为先。"; pos_delta="+1%"
    elif comp >= 45: view="【中性】观望，等待信号共振。"; pos_delta="0%"
    elif comp >= 30: view="【偏谨慎】降波动，减轻杠杆。"; pos_delta="-1%"
    else: view="【防守】现金/短债为主，控制回撤。"; pos_delta="-3%"

    # ---- Message formatting ----
    ts_str = now_tz("Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
    def fmt(v,d=4): return "N/A" if v is None else f"{float(v):.{d}f}"

    lines=[]
    lines.append(f"<b>盘中快评 / 状态查询</b>   {html.escape(ts_str)}")
    lines.append(f"• <b>黄金现价（USD/oz）</b> ≈ <b>${fmt(xau_spot,2)}</b>  <i>（多源：Stooq/FRED/Yahoo）</i>")
    lines.append("")

    lines.append("<b>信号与评分</b>（TIPS±40，DXY±15，ETF±20，拍卖±25，估值±20）")
    lines.append(f"1）10Y 实际利率 DFII10：{fmt(tips,3)}（WoW {fmt(tips_wow,3)}，{tips_src}） → <b>{round(s_tips,1)}</b>")
    lines.append(f"2）美元指数（广义）{env('DXY_LOOKBACK_D','30')}日变动：{fmt(dxy,2)}（{('+' if (dxy_30d_pct or 0)>=0 else '')}{'N/A' if dxy_30d_pct is None else f'{float(dxy_30d_pct):.2f}%'}；{dxy_src}） → <b>{round(s_dxy,1)}</b>")
    lines.append(f"3）ETF 净流向趋势（21/63 EMA＋斜率）：{etf_note} → <b>{round(s_etf,1)}</b>")
    lines.append(f"4）美债拍卖质量（近{env('AUCTION_WINDOW_D','75')}天 vs 近年 Bid-to-Cover）：{auc_note} → <b>{round(s_auc,1)}</b>")
    lines.append(f"5）估值偏离（Valuation）：{val_note} → <b>{round(s_val,1)}</b>")
    lines.append("")
    lines.append(f"<b>综合分（0–100）：{int(round(comp,0))}</b>")
    lines.append(f"<b>结论：</b>{view}  → 建议战术仓位调整：<b>{pos_delta}</b>")
    lines.append("")

    rows = [
        ("10Y名义(%)", fmt(tnx,3), fmt(tnx_wow,3), "Yahoo ^TNX"),
        ("10Y真实(%)", fmt(tips,3), fmt(tips_wow,3), tips_src),
        ("DXY(30d变动%)", "N/A" if dxy_30d_pct is None else f"{float(dxy_30d_pct):.2f}%", "—", dxy_src),
        ("VIX", fmt(vix,2), fmt(vix_wow,2), "Yahoo ^VIX"),
        ("BTC ETF净流(周)", "N/A" if etf_weekly is None else f"{float(etf_weekly):,.0f}", "—", etf_src),
        ("央行购金YoY(%)", "N/A" if cb_yoy is None else f"{float(cb_yoy):.1f}%", "—", "WGC YoY(roll12)" if cb_yoy is not None else cb_note),
        ("估值基准BASE", fmt(base,1), "—", "REG" if str(env("GOLD_FAIR_BASE","")).upper()=="REG" else ("AUTO" if str(env("GOLD_FAIR_BASE","")).upper()=="AUTO" else "Static")),
    ]
    w1=max(len(r[0]) for r in rows); w2=max(len(str(r[1])) for r in rows); w3=max(len(str(r[2])) for r in rows)
    header=f"{'指标'.ljust(w1)}  {'最新'.ljust(w2)}  {'WoW'.ljust(w3)}  来源"
    sep="-"*len(header)
    table_lines=[header,sep]+[f"{r[0].ljust(w1)}  {str(r[1]).ljust(w2)}  {str(r[2]).ljust(w3)}  {r[3]}" for r in rows]
    lines.append("<b>数据小表</b>：")
    table_html = html.escape("\n".join(table_lines))
    lines.append(f"<pre><code>{table_html}</code></pre>")

    msg="\n".join(lines)

    if mode=="status":
        print(msg); return 0
    url = f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/sendMessage"
    payload = {"chat_id": env("TELEGRAM_CHAT_ID"), "text": msg, "parse_mode": "HTML", "disable_web_page_preview": "true"}
    r = requests.post(url, data=payload, timeout=TIMEOUT)
    ok = r.status_code==200 and r.json().get("ok") is True
    if not ok:
        print("[ERROR] 发送Telegram失败：", r.text); return 3
    print("[OK] 已发送Telegram消息。"); return 0

if __name__=="__main__":
    try: sys.exit(main())
    except SystemExit as e: raise
    except Exception as e:
        print("[FATAL]", e); traceback.print_exc(); sys.exit(9)
