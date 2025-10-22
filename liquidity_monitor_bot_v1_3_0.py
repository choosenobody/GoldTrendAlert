
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Weekly Monitor Bot
# File: liquidity_monitor_bot_v1_3_0.py
#
# New in v1_3_0:
#   - BTC ETF周度净流(USD)新增“免费数据源”自动抓取：
#       * Farside "Bitcoin ETF Flow – All Data" 页面（HTML表格解析，无需Key）
#       * 保留 CSV(URL/本地) 与 API(JSON) 兼容；仍支持手工覆盖
#   - 央行净购金(吨)：WGC_CSV_URL 优先，其次 IMF IFS 汇总（免费API），仍支持手工覆盖
#   - FRED(DFII10) 作为 TIPS 兜底（需 FRED_API_KEY）
#   - 严格 HTML 转义 & <pre><code> 表格输出；更稳健的错误处理与“来源”标注
#
import os, sys, html, traceback, io, re
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
from dateutil import tz

TIMEOUT = 25
UA = {"User-Agent": "Mozilla/5.0 (LiquidityMonitorBot/1.3.0)"}

def env(key, default=None):
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default

def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tz=tzinfo)

# ------------------ Market Data ------------------
def get_yahoo_last_and_wow(symbol, adj_by_ten=False):
    try:
        df = yf.download(symbol, period="14d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None
        series = df["Close"].dropna()
        if len(series) < 2:
            return None, None
        last = float(series.iloc[-1])
        idx = max(0, len(series) - 6)  # approx 5 trading days back
        wow_base = float(series.iloc[idx])
        if adj_by_ten:
            last /= 10.0; wow_base /= 10.0
        return round(last, 4), round(last - wow_base, 4)
    except Exception:
        return None, None

# ---- 10Y real yield (TIPS) with FRED fallback ----
def get_tips_from_treasury_csv():
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-real-yield-curve-rates-csv"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None, None, None
    col = "10 Yr" if "10 Yr" in df.columns else None
    if not col:
        for c in df.columns:
            if str(c).lower().replace(" ", "") in ("10yr","10y"):
                col = c; break
    if not col:
        return None, None, None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < 2: return None, None, None
    last, prev = float(series.iloc[-1]), float(series.iloc[-2])
    return round(last,4), round(last-prev,4), "UST Real Yields CSV"

def get_tips_from_fred():
    api_key = env("FRED_API_KEY")
    if not api_key: return None, None, None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":"DFII10","api_key":api_key,"file_type":"json",
              "observation_start": (datetime.utcnow()-timedelta(days=60)).strftime("%Y-%m-%d")}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        obs = r.json().get("observations", [])
        vals = [o for o in obs if o.get("value") not in (".", None)]
        if len(vals) < 2: return None, None, None
        last, prev = float(vals[-1]["value"]), float(vals[-2]["value"])
        return round(last,4), round(last-prev,4), "FRED DFII10"
    except Exception:
        return None, None, None

def get_treasury_real_10y():
    v = get_tips_from_treasury_csv()
    if v[0] is not None: return v
    v = get_tips_from_fred()
    if v[0] is not None: return v
    return None, None, "N/A"

# ------------------ BTC ETF Flows ------------------
def week_range_shanghai(now_dt):
    dow = now_dt.weekday()
    monday = now_dt - timedelta(days=dow)
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)
    return monday.date(), friday.date()

def parse_week_sum_from_dataframe(df, date_col_candidates=("Date","Day","Datetime","Month","Period")):
    date_col = None
    for c in df.columns:
        if str(c).lower() in [x.lower() for x in date_col_candidates]:
            date_col = c; break
    if date_col is None:
        # try parseable column
        for c in df.columns:
            try:
                pd.to_datetime(df[c], errors="raise")
                date_col = c; break
            except Exception:
                continue
    if date_col is None: return None
    flow_col = None
    for cand in ("FlowUSD","NetFlowUSD","net_flow_usd","netflow_usd","netflow","flow_usd","flow","net","Total"):
        for c in df.columns:
            if str(c).lower() == cand.lower():
                flow_col = c; break
        if flow_col: break
    if flow_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols)==1: flow_col = num_cols[0]
        else: return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=[date_col])
    start, end = week_range_shanghai(now_tz("Asia/Shanghai"))
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    val = pd.to_numeric(df.loc[mask, flow_col], errors="coerce").sum()
    return float(val)

def fetch_btc_etf_weekly_from_farside():
    # Parse HTML tables from Farside "All Data" page
    url = env("FARSIDE_ALLDATA_URL", "https://farside.co.uk/bitcoin-etf-flow-all-data/")
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        tables = pd.read_html(r.text)  # list of dataframes
        # Heuristic: find the largest table with a 'Date' column or a first column parseable as date
        candidate = None; best_rows = 0
        for t in tables:
            # ensure at least 3 columns
            if t.shape[1] < 2: continue
            # try to normalize header (sometimes first row is header)
            df = t.copy()
            # Find a date column
            has_date = False
            for c in df.columns:
                if str(c).lower() in ("date","day","datetime","period","month"):
                    has_date = True; break
            if not has_date:
                # Try first column as date
                try:
                    _ = pd.to_datetime(df.iloc[:,0], errors="raise")
                    df = df.rename(columns={df.columns[0]:"Date"})
                    has_date = True
                except Exception:
                    pass
            if not has_date: continue
            if len(df) > best_rows:
                candidate, best_rows = df, len(df)
        if candidate is None: return None, "Farside(parse-none)"
        # Try to locate total flow column name heuristically
        # Farside tables often have a "Total" or "Total (US$m)" column.
        flow_col = None
        for c in candidate.columns:
            cl = str(c).lower()
            if "total" in cl or "flow" in cl or "us$m" in cl or "usd" in cl:
                flow_col = c; break
        if flow_col is None:
            # fallback: last numeric column
            numeric_cols = [c for c in candidate.columns if pd.api.types.is_numeric_dtype(candidate[c])]
            if numeric_cols: flow_col = numeric_cols[-1]
        if flow_col is None: return None, "Farside(no-flow-col)"
        # Reduce to Date + flow_col
        df = candidate[[candidate.columns[0], flow_col]].copy()
        df.columns = ["Date","NetFlowUSD"]
        weekly = parse_week_sum_from_dataframe(df, date_col_candidates=("Date",))
        return (weekly, "Farside(HTML)") if weekly is not None else (None, "Farside(parse-failed)")
    except Exception as e:
        return None, f"Farside(error:{e.__class__.__name__})"

def parse_etf_flows_from_csv_url(csv_url):
    if not csv_url: return None, "No CSV URL"
    try:
        r = requests.get(csv_url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        v = parse_week_sum_from_dataframe(df)
        return (v, "CSV(url)") if v is not None else (None, "CSV(url) parse error")
    except Exception as e:
        return None, f"CSV(url) error:{e.__class__.__name__}"

def parse_etf_flows_from_csv_path(csv_path):
    try:
        if not csv_path or not os.path.exists(csv_path):
            return None, "CSV not found"
        df = pd.read_csv(csv_path)
        v = parse_week_sum_from_dataframe(df)
        return (v, "CSV(path)") if v is not None else (None, "CSV(path) parse error")
    except Exception as e:
        return None, f"CSV(path) error:{e.__class__.__name__}"

def parse_etf_flows_from_api(api_url):
    if not api_url: return None, "No API URL"
    try:
        r = requests.get(api_url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        j = r.json()
        # expect [{date: yyyy-mm-dd, net_flow_usd: number}, ...] or {data:[...]} same shape
        rows = j.get("data", j) if isinstance(j, dict) else j
        df = pd.DataFrame(rows)
        # normalize column names
        cols = {c.lower():c for c in df.columns}
        date_col = None
        for k in ("date","day","datetime","ts","time"):
            if k in cols: date_col = cols[k]; break
        if date_col in ("ts","time"):
            # convert epoch ms/s to date
            df["Date"] = pd.to_datetime(df[date_col], unit="ms", errors="coerce")
            if df["Date"].isna().all():
                df["Date"] = pd.to_datetime(df[date_col], unit="s", errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        flow_col = None
        for k in ("net_flow_usd","netflow_usd","net_flow","netflow","flow_usd","flow","net","total"):
            if k in cols: flow_col = cols[k]; break
        if flow_col is None:
            # fallback: last numeric
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            flow_col = num_cols[-1] if num_cols else None
        if flow_col is None:
            return None, "API parse error"
        df = df[["Date", flow_col]].rename(columns={flow_col:"NetFlowUSD"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.dropna(subset=["Date"])
        v = parse_week_sum_from_dataframe(df, date_col_candidates=("Date",))
        return (v, "API") if v is not None else (None, "API parse error")
    except Exception as e:
        return None, f"API error:{e.__class__.__name__}"

def get_weekly_btc_etf_flows():
    manual = env("BTC_ETF_FLOWS_USD")
    if manual is not None:
        try: return float(manual), "Manual"
        except Exception: return None, "Manual parse error"
    # priority: CSV(path) -> CSV(url) -> API(json) -> Farside(HTML)
    v, src = parse_etf_flows_from_csv_path(env("BTC_ETF_FLOWS_CSV_PATH","btc_spot_etf_flows.csv"))
    if v is not None: return v, src
    v, src = parse_etf_flows_from_csv_url(env("BTC_ETF_FLOWS_CSV_URL"))
    if v is not None: return v, src
    v, src = parse_etf_flows_from_api(env("BTC_ETF_API_URL"))
    if v is not None: return v, src
    v, src = fetch_btc_etf_weekly_from_farside()
    if v is not None: return v, src
    return None, "N/A"

# ------------------ Central Bank Gold Purchases ------------------
def load_wgc_latest_from_url(csv_url):
    if not csv_url: return None, "No WGC_CSV_URL"
    try:
        r = requests.get(csv_url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        val = _extract_latest_wgc_val(df)
        return (val, "WGC CSV(url)") if val is not None else (None, "WGC CSV(url) parse error")
    except Exception as e:
        return None, f"WGC(url) error:{e.__class__.__name__}"

def load_wgc_latest_from_path(csv_path):
    try:
        if not csv_path or not os.path.exists(csv_path): return None, "CSV not found"
        df = pd.read_csv(csv_path)
        val = _extract_latest_wgc_val(df)
        return (val, "WGC CSV(path)") if val is not None else (None, "WGC CSV(path) parse error")
    except Exception as e:
        return None, f"WGC(path) error:{e.__class__.__name__}"

def _extract_latest_wgc_val(df):
    date_col, val_col = None, None
    for c in df.columns:
        if str(c).strip().lower() in ("month","date","period"): date_col = c; break
    if not date_col:
        for c in df.columns:
            try: pd.to_datetime(df[c], errors="raise"); date_col = c; break
            except Exception: continue
    lower_map = {str(c).lower().replace(" ",""): c for c in df.columns}
    for name in ["netpurchasetons","net_purchase_tons","net","nettons","netcentralbankpurchases","cb_net_buy_tons"]:
        if name in lower_map: val_col = lower_map[name]; break
    if val_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols: val_col = num_cols[0]
    if not date_col or not val_col: return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[~dates.isna()].copy(); df["__dt"] = dates
    df = df.sort_values("__dt")
    return float(df.iloc[-1][val_col]) if len(df)>0 else None

# IMF IFS fallback (free API)
def imf_latest_sum(countries, indicators, freq="M"):
    base = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"
    latest_month = None; month_vals = {}
    for ind in indicators:
        for c in countries:
            url = f"{base}/{freq}.{c}.{ind}.?startPeriod=2020-01"
            try:
                r = requests.get(url, headers=UA, timeout=TIMEOUT); r.raise_for_status()
                j = r.json(); series = j.get("CompactData",{}).get("DataSet",{}).get("Series")
                if not series: continue
                series_list = series if isinstance(series,list) else [series]
                for s in series_list:
                    obs = s.get("Obs"); 
                    if not obs: continue
                    obs_list = obs if isinstance(obs,list) else [obs]
                    for o in obs_list:
                        m = o.get("@TIME_PERIOD"); v = o.get("@OBS_VALUE")
                        if not m or v in ("",None): continue
                        try: val = float(v)
                        except: continue
                        month_vals.setdefault(m,0.0); month_vals[m]+=val
                        if (latest_month is None) or (m>latest_month): latest_month=m
            except Exception:
                continue
    if latest_month and month_vals.get(latest_month) is not None:
        return float(month_vals[latest_month]), f"IMF IFS ({latest_month})"
    return None, "IMF IFS(N/A)"

def get_central_bank_net_buy_tons():
    manual = env("GOLD_CB_PURCHASE_T")
    if manual is not None:
        try: return float(manual), "Manual"
        except: return None, "Manual parse error"
    v, src = load_wgc_latest_from_path(env("WGC_CSV_PATH","wgc_netbuy.csv"))
    if v is not None: return v, src
    v, src = load_wgc_latest_from_url(env("WGC_CSV_URL"))
    if v is not None: return v, src
    if str(env("IMF_USE_IFS","0")).strip()=="1":
        countries = [x.strip() for x in (env("IMF_COUNTRIES","US,CN,RU")).split(",") if x.strip()]
        indicators = [x.strip() for x in (env("IMF_INDICATOR_CANDIDATES","IRFCL_GOLD_NC,IRMIG_GOLD_NC")).split(",") if x.strip()]
        v, src = imf_latest_sum(countries, indicators, freq=env("IMF_FREQ","M"))
        if v is not None: return v, src
    return None, "N/A"

# ------------------ Helpers ------------------
def pct_str(v):
    try: return f"{float(v)*100:.2f}%"
    except: return "N/A"

def number_with_sign(v):
    try:
        f=float(v); s=f"{f:,.0f}"
        return s if f<0 else "+"+s
    except: return "N/A"

# ------------------ Decision Heuristics ------------------
def infer_tldr_and_actions(data):
    tnx, tips, vix, fr = data.get("tnx"), data.get("tips"), data.get("vix"), data.get("fund_rate")
    tips_wow, vix_wow = data.get("tips_wow"), data.get("vix_wow")
    bullets = []
    if (tips is not None and tips_wow is not None and tips_wow < 0) and (vix is not None and vix < 20):
        bullets.append("🟢 流动性边际改善（真实利率下行 + 波动温和）")
    elif vix is not None and vix >= 25:
        bullets.append("🔴 去杠杆冲击未尽（VIX≥25），注意降仓/降波动")
    else:
        bullets.append("🟠 震荡观察期（等待真实利率与波动率同步转向）")
    if (tips is not None and tips_wow is not None and tips_wow < 0) and (vix is not None and vix <= 22):
        gold_view = "偏多：逢回分批；关注 $3,84x-$3,95x 区域的企稳信号"
    elif vix is not None and vix >= 25:
        gold_view = "观望/降波动：避开事件周追价"
    else:
        gold_view = "中性偏多：以节奏为先，等待信号共振"
    if fr is not None and fr < 0 and (vix is not None and vix >= 22):
        btc_view = "偏谨慎：小步DCA，严控回撤阈值"
    elif fr is not None and fr >= 0 and (vix is not None and vix < 20):
        btc_view = "中性偏多：资金面中性/正向且波动温和，可小幅加速"
    else:
        btc_view = "中性：遵循“三信号”再加速（ETF净流入/稳定币总市值/资金费率）"
    return "；".join(bullets), gold_view, btc_view

# ------------------ Telegram ------------------
def send_telegram(token, chat_id, text, parse_mode="HTML", disable_web_page_preview=True):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode,
               "disable_web_page_preview": "true" if disable_web_page_preview else "false"}
    r = requests.post(url, data=payload, timeout=TIMEOUT)
    ok = r.status_code==200 and r.json().get("ok") is True
    return ok, (r.text if not ok else "ok")

def main():
    mode = (env("MODE","send") or "send").strip().lower()
    token, chat_id = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if mode=="send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试请使用 MODE=status"); sys.exit(2)

    # Markets
    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)
    tips, tips_wow, tips_src = get_treasury_real_10y()
    fr, next_ts, mark_px = (None, None, None)  # 可选：Binance funding；如需启用请解除下行注释
    # fr, next_ts, mark_px = get_binance_btc_funding()

    # ETF flows (auto, free-first)
    etf_weekly, etf_src = get_weekly_btc_etf_flows()

    # Central bank purchases (auto)
    cb_tons, cb_src = get_central_bank_net_buy_tons()

    data = {"tnx":tnx,"tnx_wow":tnx_wow,"tips":tips,"tips_wow":tips_wow,"vix":vix,"vix_wow":vix_wow,"fund_rate":fr}
    tldr, gold_view, btc_view = infer_tldr_and_actions(data)

    ts_str = now_tz("Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
    def fmt(v,d=4): return "N/A" if v is None else f"{v:.{d}f}"

    rows = [
        ("10Y名义(%)", fmt(tnx,3), fmt(tnx_wow,3), "Yahoo ^TNX"),
        ("10Y真实(%)", fmt(tips,3), fmt(tips_wow,3), tips_src),
        ("VIX", fmt(vix,2), fmt(vix_wow,2), "Yahoo ^VIX"),
        ("BTC资金费率", "N/A" if fr is None else f"{float(fr)*100:.2f}%", "—", "Binance Perp Funding"),
        ("BTC ETF净流(周)", "N/A" if etf_weekly is None else number_with_sign(etf_weekly), "—", etf_src),
        ("央行净购金(吨,月)", "N/A" if cb_tons is None else f"{cb_tons:.1f}", "—", cb_src),
    ]
    w1=max(len(r[0]) for r in rows); w2=max(len(str(r[1])) for r in rows); w3=max(len(str(r[2])) for r in rows)
    header=f"{'指标'.ljust(w1)}  {'最新'.ljust(w2)}  {'WoW'.ljust(w3)}  来源"
    sep="-"*len(header)
    table_lines=[header,sep]+[f"{r[0].ljust(w1)}  {str(r[1]).ljust(w2)}  {str(r[2]).ljust(w3)}  {r[3]}" for r in rows]
    table_text="\n".join(table_lines)

    prefix=""
    if (vix is not None and vix>=25) or (tips is not None and tips_wow is not None and tips_wow>0.15):
        prefix="⚠️预警 "

    msg=[]
    msg.append(f"{prefix}📊 <b>每周流动性与对冲监控</b>  •  {html.escape(ts_str)}")
    msg.append("")
    msg.append(f"<b>TL;DR</b>：{html.escape(tldr)}")
    msg.append(f"• <b>黄金</b>：{html.escape(gold_view)}")
    msg.append(f"• <b>比特币</b>：{html.escape(btc_view)}")
    msg.append("")
    msg.append("<b>数据小表</b>：")
    msg.append(f"<pre><code>{html.escape(table_text)}</code></pre>")
    msg.append("")
    msg.append(html.escape("数据源优先级：ETF=CSV(path)->CSV(url)->API(JSON)->Farside(HTML)；央行购金=WGC CSV(path)->WGC CSV(url)->IMF IFS；TIPS=UST CSV->FRED(DFII10)。"))
    msg.append(html.escape("提示：BTC_ETF_FLOWS_USD / GOLD_CB_PURCHASE_T 若填写则最高优先级覆盖自动值。"))

    full_msg="\n".join(msg)
    if mode=="status":
        print(full_msg); return 0
    ok,detail=send_telegram(token, chat_id, full_msg, parse_mode="HTML")
    if not ok:
        print("[ERROR] 发送Telegram失败：", detail); return 3
    print("[OK] 已发送Telegram消息。"); return 0

if __name__=="__main__":
    try: sys.exit(main())
    except SystemExit as e: raise
    except Exception as e:
        print("[FATAL]", e); traceback.print_exc(); sys.exit(9)
