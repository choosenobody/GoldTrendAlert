# -*- coding: utf-8 -*-
"""
tools/etf_scoring_helper.py
- 统一从 SoSoValue API / CSV 兜底 /（可选）Farside 获取 BTC 现货ETF日度净流
- 计算自适应窗口下的 EMA 短长趋势+近10日线性斜率
- 产出：
  - score_etf: 0–100
  - weekly_flow_usd: 最近7天净流合计（float）
  - diag: 诊断文本（源/行数/窗口/斜率等）
依赖环境变量（均可选，带默认）：
- BTC_ETF_API_URL / BTC_ETF_API_METHOD / BTC_ETF_API_HEADERS / BTC_ETF_API_BODY / SOSOVALUE_API_KEY(Secret)
- BTC_ETF_FLOWS_CSV_URL / BTC_ETF_CSV_DATE_FIELD / BTC_ETF_CSV_FLOW_FIELD
- ETF_LONG_D / ETF_SHORT_D / DEBUG
"""

import os, io, json, math, sys, re, traceback
import numpy as np, pandas as pd
import requests
from datetime import datetime, timedelta

TIMEOUT=25
UA={"User-Agent":"ETFScoringHelper/1.0"}

def _env(k, d=None):
    v=os.environ.get(k)
    return v if (v is not None and str(v).strip()!="") else d

def _to_float(x):
    if isinstance(x,str):
        s=x.strip().replace(",","")
        if re.match(r"^\(.*\)$", s): s="-"+s.strip("()")
        try: return float(s)
        except: return math.nan
    return float(x)

def load_from_csv():
    url=_env("BTC_ETF_FLOWS_CSV_URL")
    if not url: return None
    r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
    df=pd.read_csv(io.StringIO(r.text))
    dcol = _env("BTC_ETF_CSV_DATE_FIELD","Date")
    vcol = _env("BTC_ETF_CSV_FLOW_FIELD","NetFlowUSD")
    if dcol not in df.columns or vcol not in df.columns:
        # 尝试自动推断
        lc={c.lower():c for c in df.columns}
        dcol = lc.get("date", list(df.columns)[0])
        vcol = lc.get("netflowusd", list(df.columns)[1] if len(df.columns)>1 else None)
    if vcol is None: return None
    df=df[[dcol,vcol]].copy()
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df[vcol]=df[vcol].apply(_to_float)
    df=df.dropna().sort_values(dcol)
    if df.empty: return None
    s=pd.Series(df[vcol].values, index=df[dcol].values)
    return s

def load_from_sosovalue():
    url=_env("BTC_ETF_API_URL","https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart")
    method=_env("BTC_ETF_API_METHOD","POST").upper()
    headers_json=_env("BTC_ETF_API_HEADERS","")
    body_json=_env("BTC_ETF_API_BODY",'{"type":"us-btc-spot"}')
    try:
        headers = json.loads(headers_json) if headers_json else {}
    except Exception:
        headers = {}
    # 自动补 api-key
    if "x-soso-api-key" not in {k.lower():v for k,v in headers.items()}:
        key=_env("SOSOVALUE_API_KEY")
        if key: headers["x-soso-api-key"]=key
    if "content-type" not in {k.lower():v for k,v in headers.items()}:
        headers["Content-Type"]="application/json"

    try:
        if method=="GET":
            resp=requests.get(url, headers=headers, timeout=TIMEOUT)
        else:
            resp=requests.post(url, headers=headers, data=body_json, timeout=TIMEOUT)
        resp.raise_for_status()
        j=resp.json()
        rows=(j.get("data") or {}).get("list") or []
        if not rows: return None
        df=pd.DataFrame(rows)
        # 猜字段
        dcol=None
        for k in ["date","time","Date","day"]: 
            if k in df.columns: dcol=k; break
        vcol=None
        for k in ["totalNetInflow","netInflow","TotalNetInflow","net_flow","netflow"]:
            if k in df.columns: vcol=k; break
        if not dcol or not vcol: return None
        df=df[[dcol,vcol]].copy()
        df.columns=["Date","NetFlowUSD"]
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
        df["NetFlowUSD"]=df["NetFlowUSD"].apply(_to_float)
        df=df.dropna().sort_values("Date")
        if df.empty: return None
        return pd.Series(df["NetFlowUSD"].values, index=df["Date"].values)
    except Exception:
        return None

def choose_source():
    s = load_from_sosovalue()
    src = "API" if s is not None else None
    if s is None:
        s = load_from_csv()
        src = "CSV" if s is not None else None
    # 不再依赖 Farside（稳定性较差）；如需要可在此添加第三兜底
    return s, src

def etf_score():
    DEBUG = int(_env("DEBUG","0") or "0")
    s, source = choose_source()

    if s is None or len(s) < 3:
        diag = f"ETF源={source or 'N/A'} 行数={0} 窗口=N/A"
        return 50.0, None, diag  # 中性

    # 自适应窗口
    L = int(_env("ETF_LONG_D","34") or 34)
    S = int(_env("ETF_SHORT_D","13") or 13)
    n=len(s)
    if n < max(L,S):
        for (l,ss) in [(21,8),(13,5)]:
            if n >= max(l,ss):
                L,S=l,ss
                break

    # EMA & 斜率
    ema_long = s.ewm(span=L, adjust=False).mean()
    ema_short = s.ewm(span=S, adjust=False).mean()
    trend_ok = (ema_short.iloc[-1] > ema_long.iloc[-1])
    win = min(10, n)
    x = np.arange(win)
    y = s.iloc[-win:].values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0]) if win>=5 else 0.0

    # 0–100 映射（趋势±10，斜率标准化±10）
    score = 50.0 + (10.0 if trend_ok else -10.0) + float(np.clip(slope/1e6, -10, 10))
    score = float(np.clip(score, 0, 100))

    # 最近7天净流合计
    s7 = s[s.index >= (s.index.max() - pd.Timedelta(days=7))]
    weekly_flow = float(s7.sum()) if len(s7)>0 else None

    diag = f"ETF源={source} 行数={n} 窗口={L}/{S} 斜率={slope:.2f}"
    if DEBUG:
        print(diag)
    return score, weekly_flow, diag

if __name__ == "__main__":
    score, wflow, diag = etf_score()
    print("score_etf =", score)
    print("weekly_flow_usd =", wflow)
    print("diag =", diag)
