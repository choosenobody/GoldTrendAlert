# -*- coding: utf-8 -*-
"""
tools/etf_scoring_helper.py

用途
----
- 统一从 SoSoValue API / CSV（回退）获取 BTC 现货 ETF 日度净流（美元）
- 在自适应窗口下计算：
  * 短/长 EMA 趋势（默认 13 / 34；样本不足自动退 8/21 或 5/13）
  * 近 10 日线性回归斜率
- 产出：
  * score_etf: 0–100（50 为中性；趋势 +10/-10，斜率标准化 ±10）
  * weekly_flow_usd: 最近 7 天净流合计（float；若不足则为 None）
  * diag: 诊断文本（数据源/行数/窗口/斜率）

依赖
----
- pandas, numpy, requests

环境变量（均可选）
----------------
- SOSOVALUE_API_KEY      : SoSoValue API Key（也可放在 BTC_ETF_API_HEADERS 中）
- BTC_ETF_API_URL        : 默认 "https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart"
- BTC_ETF_API_METHOD     : "GET" 或 "POST"（默认 "POST"）
- BTC_ETF_API_HEADERS    : JSON 字符串，如 {"x-soso-api-key":"<key>","Content-Type":"application/json"}
- BTC_ETF_API_BODY       : JSON 字符串，默认 {"type":"us-btc-spot"}
- BTC_ETF_FLOWS_CSV_URL  : CSV 回退（Raw 链接），两列：Date, NetFlowUSD（或设置列名）
- BTC_ETF_CSV_DATE_FIELD : CSV 中日期列名（默认 "Date"）
- BTC_ETF_CSV_FLOW_FIELD : CSV 中净流列名（默认 "NetFlowUSD"）
- ETF_LONG_D             : 长窗 EMA，默认 34
- ETF_SHORT_D            : 短窗 EMA，默认 13
- DEBUG                  : "1" 打印诊断；默认 "0"

作为独立脚本执行
----------------
python tools/etf_scoring_helper.py
→ 打印：
   score_etf = <float>
   weekly_flow_usd = <float or None>
   diag = <str>
"""

from __future__ import annotations

import os, io, json, math, re
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import requests

TIMEOUT = 25
UA = {"User-Agent": "ETFScoringHelper/1.0"}

# -------------------- utils --------------------

def _env(k: str, d: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(k)
    return v if (v is not None and str(v).strip() != "") else d

def _to_float(x) -> float:
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if re.match(r"^\(.*\)$", s):  # 括号负号
            s = "-" + s.strip("()")
        try:
            return float(s)
        except:
            return math.nan
    try:
        return float(x)
    except:
        return math.nan

def _safe_json_loads(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

# -------------------- data loaders --------------------

def load_from_csv() -> Optional[pd.Series]:
    """
    从 CSV 回退源读取，期望两列（日期/净流），返回 pd.Series(values, index=datetime64)
    """
    url = _env("BTC_ETF_FLOWS_CSV_URL")
    if not url:
        return None
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    dcol = _env("BTC_ETF_CSV_DATE_FIELD", "Date")
    vcol = _env("BTC_ETF_CSV_FLOW_FIELD", "NetFlowUSD")

    # 列名对不上时做一次宽松匹配
    if dcol not in df.columns or vcol not in df.columns:
        low = {c.lower(): c for c in df.columns}
        dcol = low.get("date", list(df.columns)[0])
        vcol = low.get("netflowusd", list(df.columns)[1] if len(df.columns) > 1 else None)
    if vcol is None:
        return None

    df = df[[dcol, vcol]].copy()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[vcol] = df[vcol].apply(_to_float)
    df = df.dropna().sort_values(dcol)
    if df.empty:
        return None

    s = pd.Series(df[vcol].values, index=df[dcol].values)
    return s

def load_from_sosovalue() -> Optional[pd.Series]:
    """
    从 SoSoValue API 拉取，自动补 x-soso-api-key 与 Content-Type。
    兼容少量字段名差异（date/time、totalNetInflow/netInflow）。
    """
    url = _env("BTC_ETF_API_URL", "https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart")
    method = (_env("BTC_ETF_API_METHOD", "POST") or "POST").upper()
    headers = _safe_json_loads(_env("BTC_ETF_API_HEADERS", ""))
    body_json = _env("BTC_ETF_API_BODY", '{"type":"us-btc-spot"}')

    # 自动补 key / content-type
    keys_lower = {k.lower(): v for k, v in headers.items()}
    if "x-soso-api-key" not in keys_lower:
        key = _env("SOSOVALUE_API_KEY")
        if key:
            headers["x-soso-api-key"] = key
    if "content-type" not in keys_lower:
        headers["Content-Type"] = "application/json"

    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    else:
        resp = requests.post(url, headers=headers, data=body_json, timeout=TIMEOUT)
    resp.raise_for_status()
    j = resp.json()
    rows = (j.get("data") or {}).get("list") or []
    if not rows:
        return None

    df = pd.DataFrame(rows)

    # 适配字段名
    dcol = None
    for k in ["date", "time", "Date", "day"]:
        if k in df.columns:
            dcol = k
            break
    vcol = None
    for k in ["totalNetInflow", "netInflow", "TotalNetInflow", "net_flow", "netflow"]:
        if k in df.columns:
            vcol = k
            break
    if not dcol or not vcol:
        return None

    out = df[[dcol, vcol]].copy()
    out.columns = ["Date", "NetFlowUSD"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["NetFlowUSD"] = out["NetFlowUSD"].apply(_to_float)
    out = out.dropna().sort_values("Date")
    if out.empty:
        return None

    s = pd.Series(out["NetFlowUSD"].values, index=out["Date"].values)
    return s

def choose_source() -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    优先 API，失败则 CSV。
    """
    s = None
    src = None

    try:
        s = load_from_sosovalue()
        if s is not None:
            src = "API"
    except Exception:
        s = None

    if s is None:
        try:
            s = load_from_csv()
            if s is not None:
                src = "CSV"
        except Exception:
            s = None

    return s, src

# -------------------- scoring --------------------

def _adaptive_spans(n: int, L: int, S: int) -> Tuple[int, int]:
    """
    样本不足时自适应缩短窗口。
    """
    if n >= max(L, S):
        return L, S
    for (l, ss) in [(21, 8), (13, 5)]:
        if n >= max(l, ss):
            return l, ss
    return L, S  # 仍不足时按原值返回，后续逻辑会给中性分

def _ema_trend_and_slope(s: pd.Series, L: int, S: int) -> Tuple[bool, float, int, int]:
    """
    计算短/长 EMA 趋势与近10日线性斜率（返回 趋势是否向上, 斜率, 实际长窗, 实际短窗）
    """
    n = len(s)
    L, S = _adaptive_spans(n, L, S)
    if n < max(L, S) or n < 3:
        return False, 0.0, L, S

    ema_long = s.ewm(span=L, adjust=False).mean()
    ema_short = s.ewm(span=S, adjust=False).mean()
    trend_ok = bool(ema_short.iloc[-1] > ema_long.iloc[-1])

    win = min(10, n)
    x = np.arange(win)
    y = s.iloc[-win:].values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0]) if win >= 5 else 0.0
    return trend_ok, slope, L, S

def compute_score(trend_ok: bool, slope: float) -> float:
    """
    0–100 映射：趋势 ±10；斜率标准化（除以 1e6）±10；中心 50
    """
    score = 50.0 + (10.0 if trend_ok else -10.0) + float(np.clip(slope / 1e6, -10, 10))
    return float(np.clip(score, 0.0, 100.0))

def etf_score() -> Tuple[float, Optional[float], str]:
    """
    主入口：
    - 返回 (score_etf[0..100], weekly_flow_usd or None, diag_text)
    """
    DEBUG = int(_env("DEBUG", "0") or "0")

    # 选择数据源
    s, source = choose_source()
    if s is None or len(s) < 3:
        diag = f"ETF源={source or 'N/A'} 行数=0 窗口=N/A"
        if DEBUG:
            print(diag)
        return 50.0, None, diag  # 中性

    # 自适应窗口
    L = int(_env("ETF_LONG_D", "34") or 34)
    S = int(_env("ETF_SHORT_D", "13") or 13)

    trend_ok, slope, L2, S2 = _ema_trend_and_slope(s, L, S)
    # 最近 7 天净流合计
    s7 = s[s.index >= (s.index.max() - pd.Timedelta(days=7))]
    weekly_flow = float(s7.sum()) if len(s7) > 0 else None

    score = compute_score(trend_ok, slope)
    diag = f"ETF源={source} 行数={len(s)} 窗口={L2}/{S2} 斜率={slope:.2f}"
    if DEBUG:
        print(diag)
    return score, weekly_flow, diag

# -------------------- main --------------------

if __name__ == "__main__":
    sc, wf, dg = etf_score()
    print("score_etf =", sc)
    print("weekly_flow_usd =", wf)
    print("diag =", dg)
