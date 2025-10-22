
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Weekly Monitor Bot + Scoring & Position
# File: liquidity_monitor_bot_v1_4_0.py
#
# New in v1_4_0:
#   - 信号打分（TIPS±40, DXY±15, ETF±20, 拍卖±25, 估值±20）→ 综合分 0–100
#   - DXY（FRED: DTWEXBGS，需要 FRED_API_KEY）
#   - 美债拍卖质量（US Treasury Fiscal Data API，近75天 10Y/30Y Bid-to-Cover vs 近1年均值）
#   - 估值偏离：Gold ~ const + BETA_TIPS*(-TIPS) + ALPHA_CB*(CB YoY%)，映射为±20分
#   - 建议仓位：依据综合分与风险档位生成 +X%/-X%
#
import os, sys, html, traceback, io, json, math
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
from dateutil import tz

TIMEOUT = 25
UA = {"User-Agent":"Mozilla/5.0 (LiquidityMonitorBot/1.4.0)"}

def env(key, default=None):
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default

def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tz=tzinfo)

# ---------- Helpers ----------
def clamp(x, a, b):
    return max(a, min(b, x))

def pct_str(v):
    try: return f"{float(v)*100:.2f}%"
    except: return "N/A"

def number_with_sign(v):
    try:
        f=float(v); s=f"{f:,.0f}"
        return s if f<0 else "+"+s
    except: return "N/A"

# ---------- 1) Markets (TIPS, DXY, VIX, TNX) ----------
def get_yahoo_last_and_wow(symbol, adj_by_ten=False):
    try:
        df = yf.download(symbol, period="14d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty: return None, None
        series = df["Close"].dropna()
        if len(series) < 2: return None, None
        last = float(series.iloc[-1]); base = float(series.iloc[max(0, len(series)-6)])
        if adj_by_ten: last/=10.0; base/=10.0
        return round(last,4), round(last-base,4)
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
    if not api_key: return None, None, None
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

def get_tips():
    v = get_tips_from_treasury_csv()
    if v[0] is not None: return v
    return get_from_fred("DFII10", days=120)

def get_dxy_change(days=30):
    # DTWEXBGS: Broad USD Index
    last, _, src = get_from_fred("DTWEXBGS", days=days+5)
    if last is None: return None, None, src
    # to compute change, fetch rolling window again (reuse above)
    url="https://api.stlouisfed.org/fred/series/observations"
    api_key=env("FRED_API_KEY")
    params={"series_id":"DTWEXBGS","api_key":api_key,"file_type":"json",
            "observation_start":(datetime.utcnow()-timedelta(days=days+60)).strftime("%Y-%m-%d")}
    try:
        r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        obs=r.json().get("observations",[])
        vals=[float(o["value"]) for o in obs if o.get("value") not in (".", None)]
        if len(vals)<days+1: return last, None, src
        delta=last - vals[-days-1]
        pct=delta/vals[-days-1]*100.0
        return last, pct, src
    except Exception:
        return last, None, src

# ---------- 2) BTC ETF flows (daily) ----------
def fetch_farside_daily():
    url = env("FARSIDE_ALLDATA_URL", "https://farside.co.uk/bitcoin-etf-flow-all-data/")
    r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
    tables=pd.read_html(r.text)
    cand=None; best=0
    for t in tables:
        if t.shape[1] < 2: continue
        df=t.copy()
        # assume first column Date-like
        try:
            pd.to_datetime(df.iloc[:,0], errors="raise")
            if len(df)>best: cand=df; best=len(df)
        except Exception:
            continue
    if cand is None: return None
    df=cand[[cand.columns[0], cand.columns[-1]]].copy()
    df.columns=["Date","NetFlowUSD"]
    df["Date"]=pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["NetFlowUSD"]=pd.to_numeric(df["NetFlowUSD"], errors="coerce")
    df=df.dropna(subset=["Date","NetFlowUSD"])
    return df

def etf_score_from_series(df, short=21, long=63):
    # need at least long observations
    df=df.sort_values("Date")
    s=df["NetFlowUSD"].astype(float)
    if len(s) < long+5: return 0, "ETF样本不足"
    ema_s = s.ewm(span=short, adjust=False).mean()
    ema_l = s.ewm(span=long, adjust=False).mean()
    slope = ema_s.diff(5).iloc[-1]  # 5-day slope
    bullish = (ema_s.iloc[-1] > ema_l.iloc[-1]) and (slope > 0)
    # scale score by normalized slope vs MAD
    mad = (s - s.rolling(63).median()).abs().rolling(63).median().iloc[-1]
    norm = 0 if (mad is None or mad==0 or math.isnan(mad)) else slope / mad
    raw = clamp(norm, -1.5, 1.5) * 20  # → [-30, +30] then clamp to ±20 below
    score = clamp(raw, -20, +20)
    # Add bonus if condition strongly bullish/bearish
    if bullish: score = min(20, score + 5)
    elif (ema_s.iloc[-1] < ema_l.iloc[-1]) and (slope < 0): score = max(-20, score - 5)
    return round(score,1), f"EMA{short}/{long}, slope5={round(slope,1)}"

# ---------- 3) Treasury Auctions quality ----------
def auctions_quality_score(look_back_days=75, ref_months=12):
    # API docs: https://fiscaldata.treasury.gov/api-documentation/
    base="https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    # Dataset hint suggests: v1/accounting/od/auctions_query (some docs show 'auctions_query'); fall back to auction_result if needed.
    # We'll try auctions_query first, then a secondary endpoint name.
    endpoints=["v1/accounting/od/auctions_query","v1/accounting/od/auction_result"]
    endp=None; data=None
    for ep in endpoints:
        url=f"{base}/{ep}"
        params={
            "page[number]":"1","page[size]":"5000",
            "filter":f"auction_date:gte:{(datetime.utcnow()-timedelta(days=look_back_days+400)).strftime('%Y-%m-%d')}"
        }
        try:
            r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
            j=r.json(); data=j.get("data", [])
            if data: endp=ep; break
        except Exception:
            continue
    if not data:
        return 0, "AuctionAPI N/A"
    df=pd.DataFrame(data)
    # harmonize columns
    cols_lower={c.lower():c for c in df.columns}
    # choose only 10Y Note and 30Y Bond
    # guess columns for security term/type
    term_col=None
    for k in ("security_term","securityterm","term"):
        if k in cols_lower: term_col=cols_lower[k]; break
    type_col=None
    for k in ("security_type","securitytype","type"):
        if k in cols_lower: type_col=cols_lower[k]; break
    date_col=None
    for k in ("auction_date","auctiondate","auction_dt","auc_dt"):
        if k in cols_lower: date_col=cols_lower[k]; break
    btc_col=None
    for k in ("bid_to_cover_ratio","bid_to_cover","bidtocover","bid_to_cover_rt"):
        if k in cols_lower: btc_col=cols_lower[k]; break
    if not (btc_col and date_col):
        return 0, "Auction fields missing"
    df[date_col]=pd.to_datetime(df[date_col], errors="coerce")
    df=df.dropna(subset=[date_col, btc_col])
    # filter window
    recent = df[df[date_col] >= (datetime.utcnow()-timedelta(days=look_back_days))]
    # select 10Y & 30Y by fuzzy term/type
    def is_10y(row):
        t=str(row.get(term_col,"")).lower()
        ty=str(row.get(type_col,"")).lower()
        return ("10" in t and "year" in t) or ("note" in ty and "10" in t)
    def is_30y(row):
        t=str(row.get(term_col,"")).lower()
        ty=str(row.get(type_col,"")).lower()
        return ("30" in t and "year" in t) or ("bond" in ty and "30" in t)
    recent["is10"]=recent.apply(is_10y, axis=1)
    recent["is30"]=recent.apply(is_30y, axis=1)
    seg = recent[(recent["is10"]) | (recent["is30"])]
    if seg.empty: return 0, "Auction none 10Y/30Y"
    # compute ref baseline from older data (approx 12 months)
    baseline = df[df[date_col] >= (datetime.utcnow()-timedelta(days=365))]
    def mean_btc(filt):
        x=pd.to_numeric(filt[btc_col], errors="coerce").dropna()
        return x.mean() if len(x)>0 else None
    recent_mean = mean_btc(seg)
    baseline_mean = mean_btc(baseline)
    if (recent_mean is None) or (baseline_mean is None): return 0, "Auction baseline N/A"
    delta = recent_mean - baseline_mean
    # map delta → score ±25 using a scale (0.05 ~ 5% of BTC change gets ~10 points)
    score = clamp(delta/0.05 * 10, -25, 25)  # heuristic
    return round(score,1), f"{endp} ΔBTC={round(delta,3)}"

# ---------- 4) Valuation gap ----------
def valuation_gap_score(xau_spot=None, tips=None, cb_yoy_pct=None):
    # fair = BASE + BETA_TIPS * (-TIPS) + ALPHA_CB * cb_yoy_pct
    try:
        base=float(env("GOLD_FAIR_BASE", "3500"))  # 可调基准
        beta=float(env("BETA_TIPS","0.3"))
        alpha=float(env("ALPHA_CB","0.02"))
        if xau_spot is None or tips is None or cb_yoy_pct is None:
            return 0, "估值缺数据"
        fair = base + beta * (0 - float(tips)) + alpha * float(cb_yoy_pct) * base
        gap = (float(xau_spot) - fair) / fair  # +为高估
        # map ±10% → ∓20分（高估减分，低估加分）
        score = clamp(-gap/0.10 * 20, -20, +20)
        return round(score,1), f"fair={round(fair,1)}, gap={round(gap*100,2)}%"
    except Exception:
        return 0, "估值异常"

# ---------- Spot gold proxy ----------
def get_xau_spot():
    # 使用 Yahoo XAUUSD=X 作为近似
    try:
        df=yf.download("XAUUSD=X", period="7d", interval="1d", progress=False, auto_adjust=False)
        px=float(df["Close"].dropna().iloc[-1])
        return px
    except Exception:
        return None

# ---------- Central bank YoY% ----------
def cb_yoy_pct_from_sources():
    # WGC CSV URL/path 或 IMF IFS 最近月，与一年前同月比值
    def load_df():
        p=env("WGC_CSV_PATH")
        if p and os.path.exists(p):
            try: return pd.read_csv(p)
            except: pass
        url=env("WGC_CSV_URL")
        if url:
            try:
                r=requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text))
            except: pass
        # IMF fallback：略（为简洁起见，v1_4_0 在YoY阶段不再调用IMF，若需要可开启）
        return None
    df=load_df()
    if df is None: return None, "WGC N/A"
    # identify date & value
    date_col=None; val_col=None
    for c in df.columns:
        if str(c).strip().lower() in ("month","date","period"): date_col=c; break
    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise"); date_col=c; break
            except: pass
    if date_col is None: return None, "WGC date missing"
    lower={str(c).lower().replace(" ",""): c for c in df.columns}
    for name in ["netpurchasetons","net_purchase_tons","net","nettons","netcentralbankpurchases","cb_net_buy_tons"]:
        if name in lower: val_col=lower[name]; break
    if val_col is None:
        num=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        val_col = num[0] if num else None
    if val_col is None: return None, "WGC value missing"
    df=df[[date_col,val_col]].copy()
    df[date_col]=pd.to_datetime(df[date_col], errors="coerce")
    df=df.dropna()
    df=df.sort_values(date_col)
    latest=df.iloc[-1]; last_month=latest[date_col]
    prev_year_month = (last_month - pd.DateOffset(years=1)).strftime("%Y-%m")
    # find same month last year
    df["ym"]=df[date_col].dt.strftime("%Y-%m")
    try:
        v_now=float(latest[val_col])
        v_prev=float(df.loc[df["ym"]==prev_year_month, val_col].iloc[-1])
        yoy = (v_now - v_prev)/abs(v_prev) if v_prev!=0 else (1.0 if v_now>0 else -1.0)
        return yoy*100.0, "WGC YoY%"
    except Exception:
        return None, "WGC YoY calc fail"

# ---------- Main ----------
def main():
    mode=(env("MODE","send") or "send").strip().lower()
    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if mode=="send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试请使用 MODE=status"); sys.exit(2)

    # TIPS, DXY, TNX, VIX
    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)
    tips, tips_wow, tips_src = get_tips()
    dxy, dxy_30d_pct, dxy_src = get_dxy_change(days=int(env("DXY_LOOKBACK_D","30")))

    # ETF daily series for scoring
    etf_df = fetch_farside_daily()
    etf_score, etf_note = (0,"ETF不可用") if etf_df is None else etf_score_from_series(etf_df, short=int(env("ETF_SHORT_D","21")), long=int(env("ETF_LONG_D","63")))

    # Auctions quality
    auc_score, auc_note = auctions_quality_score(look_back_days=int(env("AUCTION_WINDOW_D","75")), ref_months=12)

    # Valuation gap
    xau_spot = get_xau_spot()
    cb_yoy, cb_note = cb_yoy_pct_from_sources()
    val_score, val_note = valuation_gap_score(xau_spot, tips, cb_yoy)

    # Normalize scores & combine
    s_tips = 0
    if tips is not None and tips_wow is not None:
        # 真实利率下降 → 加分，上升 → 减分；0.10pp ≈ 10分
        s_tips = clamp(-tips_wow/0.10 * 10, -40, 40)
    s_dxy = 0 if dxy_30d_pct is None else clamp(-dxy_30d_pct/1.5 * 5, -15, 15)  # 30日升值1.5%→-5分，贬值→+5分；限制±15
    s_etf = etf_score  # [-20,+20]
    s_auc = auc_score  # [-25,+25]
    s_val = val_score  # [-20,+20]

    raw = s_tips + s_dxy + s_etf + s_auc + s_val  # ∈[-120,+120] 理论范围
    composite = (raw + 120) * 100.0 / 240.0  # 线性映射到 0–100
    composite = clamp(composite, 0, 100)
    # Position suggestion: bands
    if composite >= 70:
        view="【偏多持有】保留核心仓位，逢回加一点。"; pos_delta="+3%"
    elif composite >= 55:
        view="【中性偏多】耐心持有，节奏为先。"; pos_delta="+1%"
    elif composite >= 45:
        view="【中性】观望，等待信号共振。"; pos_delta="0%"
    elif composite >= 30:
        view="【偏谨慎】降波动，减轻杠杆。"; pos_delta="-1%"
    else:
        view="【防守】现金/短债为主，控制回撤。"; pos_delta="-3%"

    # Assemble message
    ts_str = now_tz("Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
    def fmt(v,d=4): return "N/A" if v is None else f"{v:.{d}f}"
    lines=[]
    lines.append(f"盘中快评 / 状态查询  {html.escape(ts_str)}")
    xau_line = f"• LBMA Gold PM 近似（USD/oz）≈ ${fmt(xau_spot,2)}  （Yahoo·XAUUSD=X）"
    lines.append(xau_line)
    lines.append("")
    lines.append("信号与评分（单项：TIPS±40，DXY±15，ETF±20，拍卖±25，估值±20）")
    lines.append(f"1）10Y 实际利率 DFII10：{fmt(tips,3)}（WoW {fmt(tips_wow,3)}，{tips_src}） → {round(s_tips,1)}")
    lines.append(f"2）美元指数（广义）30日变动：{fmt(dxy,2)}（{('+' if (dxy_30d_pct or 0)>=0 else '')}{'N/A' if dxy_30d_pct is None else f'{dxy_30d_pct:.2f}%'}；{dxy_src}） → {round(s_dxy,1)}")
    lines.append(f"3）ETF 净流向代理（21/63 日均线＋斜率）：{etf_note} → {round(s_etf,1)}")
    lines.append(f"4）美债拍卖质量（近{env('AUCTION_WINDOW_D','75')}天 vs 近年 Bid-to-Cover）：{auc_note} → {round(s_auc,1)}")
    lines.append(f"5）估值偏离（Valuation Gap）：{val_note} → {round(s_val,1)}")
    lines.append("")
    lines.append(f"综合分（0–100）：{int(round(composite,0))}  （由原始分 {int(round(raw,0))} ∈[-120,+120] 映射：(raw+120)×100/240；中性=50）")
    lines.append("结论")
    lines.append(f"{view}")
    lines.append(f"→ 建议战术仓位调整：{pos_delta}")

    msg="\n".join(lines)

    # Tele
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
