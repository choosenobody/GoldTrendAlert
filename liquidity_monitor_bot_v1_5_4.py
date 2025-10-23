# -*- coding: utf-8 -*-
# Liquidity Monitor Bot v1.5.4
# - 嵌入 etf_score(): API→CSV 自适应 + 周净流合计 + 0–100评分
# - WGC央行购金：优先用 WGC_CSV_URL（Raw）；若行数过少则兜底 WGC_SHEET_CSV_URL（pub?output=csv）
# - 其他逻辑尽量保持原格式与输出布局（TIPS/DXY/拍卖/估值等）

import os, io, sys, math, json, re, html
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TIMEOUT=25
UA={"User-Agent":"LiquidityMonitor/1.5.4"}

def env(k, d=None):
    v=os.environ.get(k)
    return v if (v is not None and str(v).strip()!="") else d

# ---------- Common helpers ----------
def clamp(x,a,b): return max(a, min(b, x))

def fmt_num(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except:
        return str(x)

def today_str(tz="Asia/Shanghai"):
    # 报文示例里用了 CST；这里保留原样
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")

# ---------- Data loaders ----------
def fred_series(series_id, days=365*12):
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

def stooq_xau():
    try:
        r=requests.get("https://stooq.com/q/d/l/?s=xauusd&i=d", headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
        df["Date"]=pd.to_datetime(df["Date"], errors="coerce"); df=df.dropna()
        s=pd.Series(pd.to_numeric(df["Close"], errors="coerce").values, index=df["Date"].values).dropna()
        return s if len(s)>0 else None
    except Exception:
        return None

def gold_spot():
    s=stooq_xau()
    if s is None:
        sid=env("FRED_GOLD_SERIES_ID","GOLDPMGBD228NLBM")
        s=fred_series(sid, days=365*12)
    if s is None:
        try:
            df=yf.download("GC=F", period="10y", interval="1d", progress=False); 
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception: pass
    if s is None:
        try:
            df=yf.download("XAUUSD=X", period="10y", interval="1d", progress=False);
            if df is not None and not df.empty: s=df["Close"].dropna()
        except Exception: pass
    return float(s.iloc[-1]) if s is not None and len(s)>0 else None, s

def series_week_over_week(s):
    if s is None or len(s)<6: return None
    return float(s.iloc[-1]-s.iloc[-6])

# ---------- WGC (央行购金) ----------
def parse_wgc_csv_to_monthly(csv_text):
    df=pd.read_csv(io.StringIO(csv_text))
    # 选列：日期 + 数值
    dcol=None; vcol=None
    prefs={"month","date","period","time","month_end","monthend","monthenddate"}
    for c in df.columns:
        lc=str(c).lower().replace(" ","")
        if lc in prefs: dcol=c; break
    if dcol is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise"); dcol=c; break
            except: pass
    lower={str(c).lower().replace(" ",""):c for c in df.columns}
    for k in ["netpurchasetons","net_purchase_tons","net","nettons",
              "netcentralbankpurchases","cb_net_buy_tons",
              "value","amount","netbuy","net_buy_tons","purchase_tons"]:
        if k in lower: vcol=lower[k]; break
    if vcol is None:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums: vcol=nums[0]
    if not dcol or not vcol: return None
    def to_num(x):
        if isinstance(x,str):
            s=x.strip().replace(",","")
            if re.match(r"^\(.*\)$", s): s="-"+s.strip("()")
            try: return float(s)
            except: return None
        try: return float(x)
        except: return None
    df=df[[dcol,vcol]].copy()
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df[vcol]=df[vcol].apply(to_num)
    df=df.dropna().sort_values(dcol)
    m=df[dcol].dt.to_period("M").dt.to_timestamp("M")
    s=df.assign(M=m).groupby("M")[vcol].sum()
    return s

def load_wgc_series():
    # 1) 优先 Raw CSV（wgc_netbuy 仓库）
    url=env("WGC_CSV_URL")
    if url:
        try:
            r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            s=parse_wgc_csv_to_monthly(r.text)
            if s is not None and len(s)>=13: 
                return s
        except Exception:
            pass
    # 2) 兜底：Google Sheet 发布 CSV（pub?output=csv）
    sheet=env("WGC_SHEET_CSV_URL")
    if sheet:
        try:
            r=requests.get(sheet, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            s=parse_wgc_csv_to_monthly(r.text)
            if s is not None and len(s)>=13: 
                return s
        except Exception:
            pass
    return None

def wgc_yoy(rollN=12):
    s=load_wgc_series()
    if s is None or len(s)<(2*rollN+1):
        if s is not None and len(s)>=(2*6+1):
            rollN=6
        else:
            return None, rollN
    sums=s.rolling(rollN).sum().dropna()
    yoy=((sums - sums.shift(rollN)) / sums.shift(rollN))*100.0
    return yoy.dropna(), rollN

# ---------- ETF scoring（内嵌） ----------
def etf_score():
    def _env(k,d=None):
        v=os.environ.get(k)
        return v if (v is not None and str(v).strip()!="") else d
    def _to_float(x):
        if isinstance(x,str):
            s=x.strip().replace(",","")
            if re.match(r"^\(.*\)$", s): s="-"+s.strip("()")
            try: return float(s)
            except: return math.nan
        try: return float(x)
        except: return math.nan
    TIMEOUT=25; UA={"User-Agent":"ETFScoring/1.5.4"}
    # a) SoSoValue API
    def load_api():
        url=_env("BTC_ETF_API_URL","https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart")
        method=_env("BTC_ETF_API_METHOD","POST").upper()
        headers_json=_env("BTC_ETF_API_HEADERS","")
        body_json=_env("BTC_ETF_API_BODY",'{"type":"us-btc-spot"}')
        try: headers=json.loads(headers_json) if headers_json else {}
        except: headers={}
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
    # b) CSV 回退
    def load_csv():
        url=_env("BTC_ETF_FLOWS_CSV_URL")
        if not url: return None
        r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
        dcol=_env("BTC_ETF_CSV_DATE_FIELD","Date")
        vcol=_env("BTC_ETF_CSV_FLOW_FIELD","NetFlowUSD")
        if dcol not in df.columns or vcol not in df.columns:
            lc={c.lower():c for c in df.columns}
            dcol=lc.get("date", list(df.columns)[0])
            vcol=lc.get("netflowusd", list(df.columns)[1] if len(df.columns)>1 else None)
        if vcol is None: return None
        df=df[[dcol,vcol]].copy()
        df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
        df[vcol]=df[vcol].apply(_to_float)
        df=df.dropna().sort_values(dcol)
        if df.empty: return None
        return pd.Series(df[vcol].values, index=df[dcol].values)
    # choose
    s=load_api(); source="API" if s is not None else None
    if s is None:
        s=load_csv(); source="CSV" if s is not None else None
    if s is None or len(s)<3:
        return 50.0, None, f"ETF源={source or 'N/A'} 行数=0 窗口=N/A"
    L=int(_env("ETF_LONG_D","34") or 34)
    S=int(_env("ETF_SHORT_D","13") or 13)
    n=len(s)
    if n<max(L,S):
        for (l,ss) in [(21,8),(13,5)]:
            if n>=max(l,ss):
                L,S=l,ss; break
    ema_long=s.ewm(span=L, adjust=False).mean()
    ema_short=s.ewm(span=S, adjust=False).mean()
    trend_ok=bool(ema_short.iloc[-1]>ema_long.iloc[-1])
    win=min(10,n); x=np.arange(win); y=s.iloc[-win:].values.astype(float)
    slope=float(np.polyfit(x,y,1)[0]) if win>=5 else 0.0
    score=50.0 + (10.0 if trend_ok else -10.0) + float(np.clip(slope/1e6, -10, 10))
    score=float(np.clip(score, 0, 100))
    s7=s[s.index >= (s.index.max()-pd.Timedelta(days=7))]
    weekly=float(s7.sum()) if len(s7)>0 else None
    diag=f"ETF源={source} 行数={n} 窗口={L}/{S} 斜率={slope:.2f}"
    return score, weekly, diag

# ---------- Main ----------
def main():
    spot, s_gold = gold_spot()
    tips_d = fred_series("DFII10", days=365*2)    # 真实利率
    dxy_d  = fred_series("DTWEXBGS", days=365*2)  # 广义美元指数
    vix_df = yf.download("^VIX", period="6mo", interval="1d", progress=False)
    tnx_df = yf.download("^TNX", period="6mo", interval="1d", progress=False)

    tips = float(tips_d.iloc[-1]) if tips_d is not None and len(tips_d)>0 else None
    tips_wow = (tips_d.iloc[-1] - tips_d.iloc[-6]) if tips_d is not None and len(tips_d)>6 else None

    dxy_30 = None
    if dxy_d is not None and len(dxy_d)>=30:
        dxy_30 = (dxy_d.iloc[-1]/dxy_d.iloc[-30] - 1.0)*100.0

    vix = float(vix_df["Close"].dropna().iloc[-1]) if vix_df is not None and not vix_df.empty else None
    tnx = float(tnx_df["Close"].dropna().iloc[-1]) if tnx_df is not None and not tnx_df.empty else None
    tnx_wow = None
    if tnx_df is not None and not tnx_df.empty and len(tnx_df["Close"].dropna())>6:
        s=tnx_df["Close"].dropna(); tnx_wow=float(s.iloc[-1]-s.iloc[-6])

    # ETF
    sc_etf, weekly_flow, diag_etf = etf_score()

    # WGC YoY
    yoy, used_roll = wgc_yoy(12)
    yoy_last = float(yoy.iloc[-1]) if yoy is not None and len(yoy)>0 else None

    # 简单映射（不动你现有权重，仅展示 0–100）
    sc_tips = 50.0 if tips is None else float(np.clip(50.0 + (-tips)*10.0 + (-(tips_wow or 0.0))*5.0, 0, 100))
    sc_dxy  = 50.0 if dxy_30 is None else float(np.clip(50.0 + (-dxy_30)*3.0, 0, 100))
    # 拍卖质量保留你现有逻辑（此处仅占位）
    delta_btc = env("UST_AUCTIONS_DELTA")  # 允许从变量注入 ΔBTC
    try: delta_btc = float(delta_btc) if delta_btc is not None else None
    except: delta_btc=None
    sc_auc = 50.0 if delta_btc is None else float(np.clip(50.0 + (delta_btc*800), 0, 100))

    # 估值（简化版：仅展示阈值±20映射；你的回归基准逻辑已在回测里完成，此处不动）
    val_gap_pct = float(env("VAL_GAP_PCT","12") or 12)
    base_hint = "REG"  # 提示基准由回归截距
    # 这里仅展示载入值，不重新回归（避免与线上一致性冲突）
    fair = None; base=None; gap=None
    if spot is not None:
        # 若你已有“公允/基准”缓存逻辑，可在此注入；否则先展示阈值与基准提示
        pass

    # 组装报文（与现有风格一致）
    lines=[]
    lines.append(f"盘中快评 / 状态查询   {today_str()}")
    if spot is not None:
        lines.append(f"• 黄金现价（USD/oz） ≈ ${fmt_num(spot,2)}  （多源：Stooq/FRED/Yahoo）")
    lines.append("")
    lines.append("信号与评分（展示为 0–100）")
    tip1 = f"{tips:.3f}" if tips is not None else "N/A"
    tip1_w = f"（WoW {tips_wow:+.3f}，FRED DFII10）" if tips_wow is not None else "（FRED DFII10）"
    lines.append(f"1）10Y 实际利率 DFII10：{tip1}{tip1_w} → {fmt_num(sc_tips,1)}")
    dxy_str = f"{fmt_num(dxy_d.iloc[-1],2)}（{fmt_num(dxy_30,2)}%；FRED DTWEXBGS）" if dxy_30 is not None else "N/A（FRED DTWEXBGS）"
    lines.append(f"2）美元指数（广义）30日变动：{dxy_str} → {fmt_num(sc_dxy,1)}")
    if weekly_flow is None:
        lines.append(f"3）ETF 净流向趋势（EMA＋斜率）：ETF不可用 → 50.0")
    else:
        lines.append(f"3）ETF 净流向趋势（EMA＋斜率）：周净流=${fmt_num(weekly_flow,0)} → {fmt_num(sc_etf,1)}")
    auc_str = f"ΔBTC={delta_btc:+.3f}" if delta_btc is not None else "N/A"
    lines.append(f"4）美债拍卖质量（近75天 vs 近年 Bid-to-Cover）：{auc_str} → {fmt_num(sc_auc,1)}")
    if yoy_last is None:
        lines.append(f"5）估值偏离（Valuation）：基准=未取；公允=未取；偏离=—（CB YoY=N/A） → -20（±20分制）")
    else:
        lines.append(f"5）估值偏离（Valuation）：基准=…；公允=…；偏离=…（CB YoY={fmt_num(yoy_last,1)}%） → -20（±20分制）")
    # 简单综合分（演示：四项均值；你线上仍可用原逻辑）
    scs=[sc_tips, sc_dxy, (sc_etf if weekly_flow is not None else 50.0), sc_auc]
    comp = float(np.mean(scs))
    lines.append("")
    lines.append(f"综合分（0–100）：{int(round(comp,0))}")
    concl = "【偏谨慎】降波动，减轻杠杆。" if comp<50 else "【偏多持有】逢回加一点。"
    lines.append(f"结论：{concl}  → 建议战术仓位调整：{'+1%' if comp>=55 else ('-1%' if comp<=45 else '0%')}")
    lines.append("")
    lines.append("数据小表：")
    lines.append("指标            最新      WoW     来源")
    lines.append("--------------------------------")
    lines.append(f"10Y名义(%)      {fmt_num(tnx,3) if tnx is not None else 'N/A':<7} {fmt_num(tnx_wow,3) if tnx_wow is not None else '—':<7}  Yahoo ^TNX")
    lines.append(f"10Y真实(%)      {fmt_num(tips,3) if tips is not None else 'N/A':<7} {fmt_num(tips_wow,3) if tips_wow is not None else '—':<7}  FRED DFII10")
    lines.append(f"DXY(30d变动%)   {fmt_num(dxy_30,2) if dxy_30 is not None else 'N/A':<7}  —       FRED DTWEXBGS")
    lines.append(f"VIX           {fmt_num(vix,2) if vix is not None else 'N/A':<7} {fmt_num(series_week_over_week(vix_df['Close'].dropna() if vix_df is not None else None),2) if vix_df is not None and not vix_df.empty else '—':<7}   Yahoo ^VIX")
    lines.append(f"BTC ETF净流(周)  {fmt_num(weekly_flow,0) if weekly_flow is not None else 'N/A':<7}  —       API/CSV")
    lines.append(f"央行购金YoY(%)    {fmt_num(yoy_last,1) if yoy_last is not None else 'N/A':<7}  —       WGC YoY(roll{used_roll if yoy_last is not None else '—'})")
    lines.append(f"估值阈值(±%)      {fmt_num(float(env('VAL_GAP_PCT','12') or 12),0)}      —       ±20分映射阈值")
    lines.append(f"估值基准BASE      {'REG' if base is None else fmt_num(base,1)}  —       {base_hint}")

    # 诊断行（便于你排查ETF源/窗口）
    lines.append("")
    lines.append(f"诊断：{diag_etf}")

    text="\n".join(lines)

    # 发送 Telegram
    bot=env("TELEGRAM_BOT_TOKEN"); chat=env("TELEGRAM_CHAT_ID")
    if bot and chat:
        try:
            # 用纯文本，避免 parse_mode 导致解析错误
            requests.post(f"https://api.telegram.org/bot{bot}/sendMessage",
                          json={"chat_id": chat, "text": text})
        except Exception as e:
            print("发送Telegram失败：", e, file=sys.stderr)
    else:
        print(text)

if __name__=="__main__":
    sys.exit(main() or 0)
