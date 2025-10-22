
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Liquidity & Hedge Weekly Monitor Bot
# File: liquidity_monitor_bot_v1_0_0.py
#
# Purpose:
#   - 每周自动抓取关键流动性与风险指标，生成一页版摘要，并通过 Telegram 机器人发送给指定聊天。
#   - 指标：FOMC信号（需人工备注/可选）、10Y美债名义利率、10Y TIPS真实利率、VIX、BTC永续资金费率。
#   - 设计为“可运行即用、可拓展”，对数据源不可用时给出 N/A 并不阻断。
#
# Usage:
#   MODE=send python liquidity_monitor_bot_v1_0_0.py           # 发送到 Telegram
#   MODE=status python liquidity_monitor_bot_v1_0_0.py         # 本地打印，不发送
#
# Env (必须/可选):
#   TELEGRAM_BOT_TOKEN   - Telegram 机器人 Token（必填，建议用 GitHub Secrets）
#   TELEGRAM_CHAT_ID     - 目标 Chat ID（必填）
#   FOMC_NOTE            - 可选，手工备注最新FOMC政策信号（如：'9月降息25bp；10月预期再降；或暂停QT'）
#   BTC_ETF_FLOWS_USD    - 可选，手工录入本周美股现货BTC ETF净流入(USD)，如未知可不填
#   GOLD_CB_PURCHASE_T   - 可选，手工录入全球央行当月净购金(吨)，如未知可不填
#
# Dependencies:
#   pip install requests pandas yfinance python-dateutil
#
# Author: ChatGPT (GPT-5 Thinking)
# Version: v1.0.0
#
import os
import sys
import html
import traceback
from datetime import datetime
import requests
import pandas as pd
import yfinance as yf
from dateutil import tz

TIMEOUT = 15  # seconds
UA = {"User-Agent": "Mozilla/5.0 (LiquidityMonitorBot/1.0)"}

def env(key, default=None):
    v = os.environ.get(key)
    return v if (v is not None and str(v).strip() != "") else default

def now_tz(tzname="Asia/Shanghai"):
    tzinfo = tz.gettz(tzname)
    return datetime.now(tz=tzinfo)

# ------------------------- Data Providers -------------------------
def get_yahoo_last_and_wow(symbol, adj_by_ten=False):
    # 拉取最近14天日线，取最后一个有效数值作为 last，取5个交易日前的数值计算WoW。
    # adj_by_ten: True时对^TNX做 /10 处理（^TNX=收益率*10）
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
    # 从美国财政部CSV抓取10Y真实利率（TIPS）。源：Daily Treasury REAL Yield Curve Rates (CSV)
    # 取最后两个有效值，计算WoW近似（以两个最近工作日替代周频）。
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

    if "10 Yr" not in df.columns:
        candidates = [c for c in df.columns if str(c).lower().strip().replace(" ", "") in ("10yr", "10y")]
        if candidates:
            col = candidates[0]
        else:
            return None, None
    else:
        col = "10 Yr"

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < 2:
        return None, None
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    wow = last - prev
    return round(last, 4), round(wow, 4)

def get_binance_btc_funding():
    # 获取币安BTCUSDT永续合约资金费率与下次结算时间。
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

# ------------------------- Helpers -------------------------
def pct_str(v):
    try:
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "N/A"

# ------------------------- Decision Heuristics -------------------------
def infer_tldr_and_actions(data):
    # 依据阈值生成 TL;DR 与操作倾向。
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

# ------------------------- Telegram -------------------------
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

# ------------------------- Main -------------------------
def main():
    mode = (env("MODE", "send") or "send").strip().lower()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if mode == "send" and (not token or not chat_id):
        print("[ERROR] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置；若仅测试请使用 MODE=status")
        sys.exit(2)

    # 拉数据
    tnx, tnx_wow = get_yahoo_last_and_wow("^TNX", adj_by_ten=True)     # 10Y nominal
    vix, vix_wow = get_yahoo_last_and_wow("^VIX", adj_by_ten=False)    # VIX
    tips, tips_wow = get_treasury_real_10y()                            # 10Y real
    fr, next_ts, mark_px = get_binance_btc_funding()                    # Funding rate

    # 可选外部补充
    fomc_note = env("FOMC_NOTE", "（手工）关注FOMC：降息节奏与QT是否暂停")
    etf_flows = env("BTC_ETF_FLOWS_USD", "N/A")
    gold_cb = env("GOLD_CB_PURCHASE_T", "N/A")

    data = {
        "tnx": tnx, "tnx_wow": tnx_wow,
        "tips": tips, "tips_wow": tips_wow,
        "vix": vix, "vix_wow": vix_wow,
        "fund_rate": fr
    }
    tldr, gold_view, btc_view = infer_tldr_and_actions(data)

    sh_now = now_tz("Asia/Shanghai")
    ts_str = sh_now.strftime("%Y-%m-%d %H:%M %Z")

    def fmt(v, digits=4):
        return "N/A" if v is None else f"{v:.{digits}f}"

    rows = [
        ("10Y名义(%)", fmt(tnx, 3), fmt(tnx_wow, 3), "Yahoo ^TNX"),
        ("10Y真实(%)", fmt(tips, 3), fmt(tips_wow, 3), "UST Real Yields CSV"),
        ("VIX",        fmt(vix, 2),  fmt(vix_wow, 2),  "Yahoo ^VIX"),
        ("BTC资金费率", "N/A" if fr is None else pct_str(fr), "—", "Binance Perp Funding"),
        ("BTC ETF净流(周)", str(etf_flows), "—", "手工/数据源见注"),
        ("央行净购金(吨,月)", str(gold_cb), "—", "WGC月度/手工填"),
    ]
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(str(r[1])) for r in rows)
    w3 = max(len(str(r[2])) for r in rows)
    w4 = max(len(str(r[3])) for r in rows)
    header = f"{'指标'.ljust(w1)}  {'最新'.ljust(w2)}  {'WoW'.ljust(w3)}  来源"
    sep = "-" * len(header)
    table_lines = [header, sep] + [
        f"{r[0].ljust(w1)}  {str(r[1]).ljust(w2)}  {str(r[2]).ljust(w3)}  {r[3]}"
        for r in rows
    ]
    table_text = "\n".join(table_lines)

    msg = []
    msg.append(f"📊 <b>每周流动性与对冲监控</b>  •  {ts_str}")
    msg.append("")
    msg.append(f"<b>TL;DR</b>：{html.escape(tldr)}")
    msg.append(f"• <b>黄金</b>：{html.escape(gold_view)}")
    msg.append(f"• <b>比特币</b>：{html.escape(btc_view)}")
    msg.append("")
    msg.append("<b>数据小表</b>：")
    msg.append(f"<pre>{html.escape(table_text)}</pre>")
    msg.append("")
    msg.append("<b>门槛/阈值</b>：VIX > 25 预警；VIX < 18 安稳；10Y TIPS 周降看多金；资金费率中性转正利于BTC。")
    msg.append("")
    msg.append("<b>信息源</b>：Yahoo Finance (^TNX,^VIX), UST Real Yield CSV, Binance Perp Funding；ETF与央行购金建议人工校验/补充。")
    msg.append("（若任何项为 N/A，属数据源限制或当天无更新，不影响总体推送；可用环境变量手工覆盖。）")

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
