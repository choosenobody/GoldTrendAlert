
#!/usr/bin/env python3
import sys, pandas as pd
path = sys.argv[1] if len(sys.argv)>1 else "wgc_netbuy.csv"
df = pd.read_csv(path)
# date col
date_ok = any(str(c).strip().lower() in ("month","date","period") for c in df.columns)
if not date_ok:
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise"); date_ok=True; break
        except Exception:
            pass
assert date_ok, "No parseable date column"
# numeric col
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
assert len(num_cols)>=1, "No numeric columns"
print("[OK] WGC CSV validated. Rows:", len(df))
