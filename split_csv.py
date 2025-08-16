import os
import pandas as pd

# --- change this to your actual file ---
src = r"ProductKnowledge_HCI_Informa-Selma_16082025_QS.csv"
# how many rows per chunk
size = 1000

outdir = os.path.dirname(src) or "."
df = pd.read_csv(src)
total = len(df)

for i in range(0, total, size):
    out = os.path.join(outdir, f"products_{i//size+1:03}.csv")
    df.iloc[i:i+size].to_csv(out, index=False)
    print(f"Wrote {out} ({min(i+size, total)}/{total})")
