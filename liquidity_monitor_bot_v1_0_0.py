
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Weekly Monitor Bot
# File: liquidity_monitor_bot_v1_1_1.py
#
# Fixes vs v1_1_0:
#   - 全面 HTML 安全：所有含有 "<" 或 ">" 的文本均使用 html.escape()
#   - 表格使用 <pre><code>...</code></pre> 以避免解析错误
#   - 去除消息体中尖括号示例，引导语用方括号替代
#
import os
import sys
import html
import json
import traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
from dateutil import tz

TIMEOUT = 15
UA = {"User-Agent": "Mozilla/5.0 (LiquidityMonitorBot/1.1.1)"}

def env(key, default=None):
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default

def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tz=tzname)

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
        idx = max(0, len(series) - 6)
        wow_base = float(series.iloc[idx])
        if adj_by_ten:
            last = last / 10.0
            wow_base = wow_base / 10.0
        wow = last - wow_base
        return round(last, 4), round(wow, 4)
    except Exception:
        return None, None

def get_treasury_real_10y():
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-real-yield-curve-rates-csv"
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        try:
            df = pd.read_csv(pd.compat.StringIO(r.text))
        except Exception:
            import io
            df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None, None

    col = None
    if "10 Yr" in df.columns:
        col = "10 Yr"
    else:
        candidates = [c for c in df.columns if str(c).lower().strip().replace(" ", "") in ("10yr", "10y")]
        if candidates:
            col = candidates[0]
    if not col:
        return None, None

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < 2:
        return None, None
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    return round(last, 4), round(last - prev, 4)

def get_binance_btc_funding():
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        r = requests.get(url, params={"symbol": "BTCUSDT"}, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        rate = float(j.get("lastFundingRate", 0.0))
        next_ts = int(j.get("nextFundingTime", 0)) // 1000
        mark_price = float(j.get("markPrice", 0.0))
        return rate, next_ts, mark_price
    except Exception:
        return None, None, None

# ------------------ ETF Flows (auto) ------------------
def week_range_shanghai(now_dt):
    dow = now_dt.weekday()  # Monday=0
    monday = now_dt - timedelta(days=dow)
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)
    return monday.date(), friday.date()

def parse_etf_flows_from_csv(csv_path, now_dt):
    try:
        if not os.path.exists(csv_path):
            return None, "CSV not found"
        df = pd.read_csv(csv_path)
        # date column
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "day", "datetime", "month", "period"):
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c; break
                except Exception:
                    continue
        if date_col is None:
            return None, "No date column"
        # flow column
        flow_col = None
        for cand in ("FlowUSD", "NetFlowUSD", "net_flow_usd", "netflow_usd", "netflow", "flow_usd", "flow", "net"):
            for c in df.columns:
                if str(c).lower() == cand.lower():
                    flow_col = c; break
            if flow_col: break
        if flow_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) == 1:
                flow_col = num_cols[0]
            else:
                return None, "No flow column"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df = df.dropna(subset=[date_col])
        week_start, week_end = week_range_shanghai(now_dt)
        mask = (df[date_col] >= week_start) & (df[date_col] <= week_end)
        weekly_sum = pd.to_numeric(df.loc[mask, flow_col], errors="coerce").sum()
        return float(weekly_sum), "CSV"
    except Exception as e:
        return None, f"CSV error: {e}"

def parse_etf_flows_from_api(api_url, now_dt):
    try:
        if not api_url:
            return None, "No API URL"
        r = requests.get(api_url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        rows = j["data"] if isinstance(j, dict) and isinstance(j.get("data"), list) else (j if isinstance(j, list) else [])
        weekly_sum = 0.0
        week_start, week_end = week_range_shanghai(now_dt)
        for row in rows:
            d = None
            for k in ("date", "day", "datetime", "ts", "time"):
                if k in row:
                    try:
                        if k in ("ts", "time") and isinstance(row[k], (int, float)):
                            d = datetime.utcfromtimestamp(int(row[k])//1000 if int(row[k])>1e12 else int(row[k])).date()
                        else:
                            d = pd.to_datetime(row[k], errors="coerce").date()
                    except Exception:
                        d = None
                    if d: break
            if not d or not (week_start <= d <= week_end):
                continue
            flow = None
            for k in ("net_flow_usd", "netflow_usd", "net_flow", "netflow", "flow_usd", "flow", "net"):
                if k in row:
                    try:
                        flow = float(row[k])
                    except Exception:
                        flow = None
                    break
            if flow is None:
                continue
            weekly_sum += flow
        return float(weekly_sum), "API"
    except Exception as e:
        return None, f"API error: {e}"

def get_weekly_btc_etf_flows(now_dt):
    manual = env("BTC_ETF_FLOWS_USD")
    if manual is not None:
        try:
            return float(manual), "Manual"
        except Exception:
            return None, "Manual parse error"
    csv_path = env("BTC_ETF_FLOWS_CSV_PATH", "btc_spot_etf_flows.csv")
    v, src = parse_etf_flows_from_csv(csv_path, now_dt)
    if v is not None:
        return v, src
    api_url = env("BTC_ETF_API_URL")
    v, src = parse_etf_flows_from_api(api_url, now_dt)
    if v is not None:
        return v, src
    return None, "N/A"

# ------------------ Central Bank Gold Purchases (auto) ------------------
def load_wgc_latest_net_buy_tons(csv_path):
    try:
        if not os.path.exists(csv_path):
            return None, "CSV not found"
        df = pd.read_csv(csv_path)
        date_col = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("month", "date", "period"):
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c; break
                except Exception:
                    continue
        if date_col is None:
            return None, "No date column"
        val_col = None
        candidates = ["netpurchasetons", "net_purchase_tons", "net", "nettons", "netcentralbankpurchases", "cb_net_buy_tons"]
        lower_map = {str(c).lower().replace(" ", ""): c for c in df.columns}
        for name in candidates:
            if name in lower_map:
                val_col = lower_map[name]; break
        if val_col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                val_col = num_cols[0]
            else:
                return None, "No numeric column"
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df.loc[~dates.isna()].copy()
        df["__dt"] = dates
        df = df.sort_values("__dt")
        latest = df.iloc[-1]
        val = float(latest[val_col])
        return val, "CSV(WGC)"
    except Exception as e:
        return None, f"CSV error: {e}"

def get_central_bank_net_buy_tons():
    manual = env("GOLD_CB_PURCHASE_T")
    if manual is not None:
        try:
            return float(manual), "Manual"
        except Exception:
            return None, "Manual parse error"
    csv_path = env("WGC_CSV_PATH", "wgc_netbuy.csv")
    v, src = load_wgc_latest_net_buy_tons(csv_path)
    if v is not None:
        return v, src
    return None, "N/A"

# ------------------ Helpers ------------------
def pct_str(v):
    try:
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "N/A"

def number_with_sign(v):
    try:
        f = float(v)
        s = f"{f:,.0f}"
        return s if f < 0 else "+" + s
    except Exception:
        return "N/A"

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
        btc_view = "中性：遵循“三信号”再加速（ETF净流入/稳定币市值/资金费率）"

    tldr = "；".join(bullets)
    return tldr, gold_view, btc_view

# ------------------ Telegram ------------------
def send_telegram(token, chat_id, text, parse_mode="HTML", disable_web_page_preview=True):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true" if disable_web_page_preview else "false",
    }
    r = requests.post(url, data=payload, timeout=TIMEOUT)
    ok = r.status_code == 200 and r.json().get("ok") is True
    return ok, (r.text if not ok else "ok")

# ------------------ Main ------------------
def main():
    mode = (env("MODE", "send") or "send").strip().lower()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if mode == "send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试请使用 MODE=status")
        sys.exit(2)

    # Markets
    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)
    tips, tips_wow = get_treasury_real_10y()
    fr, next_ts, mark_px = get_binance_btc_funding()

    # ETF flows (auto)
    now_local = now_tz("Asia/Shanghai")
    etf_weekly, etf_src = get_weekly_btc_etf_flows(now_local)

    # Central bank purchases (auto)
    cb_tons, cb_src = get_central_bank_net_buy_tons()

    data = {
        "tnx": tnx, "tnx_wow": tnx_wow,
        "tips": tips, "tips_wow": tips_wow,
        "vix": vix, "vix_wow": vix_wow,
        "fund_rate": fr
    }
    tldr, gold_view, btc_view = infer_tldr_and_actions(data)

    ts_str = now_local.strftime("%Y-%m-%d %H:%M %Z")

    def fmt(v, digits=4):
        return "N/A" if v is None else f"{v:.{digits}f}"

    rows = [
        ("10Y名义(%)", fmt(tnx, 3), fmt(tnx_wow, 3), "Yahoo ^TNX"),
        ("10Y真实(%)", fmt(tips, 3), fmt(tips_wow, 3), "UST Real Yields CSV"),
        ("VIX",        fmt(vix, 2),  fmt(vix_wow, 2),  "Yahoo ^VIX"),
        ("BTC资金费率", "N/A" if fr is None else pct_str(fr), "—", "Binance Perp Funding"),
        ("BTC ETF净流(周)", "N/A" if etf_weekly is None else number_with_sign(etf_weekly), "—", etf_src),
        ("央行净购金(吨,月)", "N/A" if cb_tons is None else f"{cb_tons:.1f}", "—", cb_src),
    ]
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(str(r[1])) for r in rows)
    w3 = max(len(str(r[2])) for r in rows)
    header = f"{'指标'.ljust(w1)}  {'最新'.ljust(w2)}  {'WoW'.ljust(w3)}  来源"
    sep = "-" * len(header)
    table_lines = [header, sep] + [
        f"{r[0].ljust(w1)}  {str(r[1]).ljust(w2)}  {str(r[2]).ljust(w3)}  {r[3]}"
        for r in rows
    ]
    table_text = "\n".join(table_lines)

    # 预警前缀
    prefix = ""
    if (vix is not None and vix >= 25) or (tips is not None and tips_wow is not None and tips_wow > 0.15):
        prefix = "⚠️预警 "

    msg = []
    msg.append(f"{prefix}📊 <b>每周流动性与对冲监控</b>  •  {html.escape(ts_str)}")
    msg.append("")
    msg.append(f"<b>TL;DR</b>：{html.escape(tldr)}")
    msg.append(f"• <b>黄金</b>：{html.escape(gold_view)}")
    msg.append(f"• <b>比特币</b>：{html.escape(btc_view)}")
    msg.append("")
    msg.append("<b>数据小表</b>：")
    msg.append(f"<pre><code>{html.escape(table_text)}</code></pre>")
    msg.append("")
    msg.append(html.escape("门槛/阈值：VIX > 25 预警；VIX < 18 安稳；10Y TIPS 周降看多金；资金费率中性转正利于BTC。"))
    msg.append("")
    msg.append(html.escape("配置提示：可通过 Secrets 设定 [BTC_ETF_FLOWS_USD / BTC_ETF_FLOWS_CSV_PATH / BTC_ETF_API_URL / WGC_CSV_PATH / GOLD_CB_PURCHASE_T]。"))
    msg.append(html.escape("API 示例：将 `BTC_ETF_API_URL` 指向你自建的 JSON 接口（不要使用尖括号 <> ）。"))

    full_msg = "\n".join(msg)

    if mode == "status":
        print(full_msg)
        return 0

    ok, detail = send_telegram(token, chat_id, full_msg, parse_mode="HTML")
    if not ok:
        print("[ERROR] 发送Telegram失败：", detail)
        return 3
    print("[OK] 已发送Telegram消息。")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(9)
