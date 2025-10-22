
#!/usr/bin/env python3
# ETL: fetch Farside "Bitcoin ETF Flow – All Data" and write simple CSV: Date,NetFlowUSD
import requests, pandas as pd, io, sys
URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
r = requests.get(URL, timeout=20); r.raise_for_status()
tables = pd.read_html(r.text)
# heuristic: largest table with a date-like first column
candidate=None; best=0
for t in tables:
    if t.shape[1] < 2: continue
    df=t.copy()
    try:
        pd.to_datetime(df.iloc[:,0], errors="raise")
        if len(df)>best: candidate=df; best=len(df)
    except Exception:
        continue
if candidate is None:
    raise SystemExit("No parseable table from Farside")
df = candidate[[candidate.columns[0], candidate.columns[-1]]].copy()
df.columns=["Date","NetFlowUSD"]
df.to_csv(".bot_state/btc_spot_etf_flows.csv", index=False)
print("[OK] Wrote .bot_state/btc_spot_etf_flows.csv with", len(df), "rows")
