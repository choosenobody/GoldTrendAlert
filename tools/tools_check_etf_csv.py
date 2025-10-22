
#!/usr/bin/env python3
import sys, pandas as pd
from datetime import datetime
path = sys.argv[1] if len(sys.argv)>1 else "btc_spot_etf_flows.csv"
df = pd.read_csv(path)
assert any(c.lower() in ("date","day","datetime","month","period") for c in df.columns), "No date-like column"
# find numeric column
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
assert len(num_cols)>=1, "No numeric columns"
print("[OK] ETF CSV validated. Rows:", len(df))
