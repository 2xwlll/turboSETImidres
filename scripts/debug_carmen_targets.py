from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('/home/wlll2x/turboSETI')
CSV_PATH = ROOT / 'scripts' / 'carmenes_1688mhz_subset.csv'

TARGETS = ["NGC5638", "SAG_DIR"]

df = pd.read_csv(CSV_PATH)

print("\n=== RAW CSV STATS ===")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

df["Target"] = df["Target"].astype(str).str.strip()

print("\n=== TARGET COUNTS ===")
for t in TARGETS:
    print(t, ":", (df["Target"] == t).sum())

print("\n=== UNIQUE TARGETS (top 20) ===")
print(df["Target"].value_counts().head(20))

print("\n=== FREQUENCY STATS ===")
df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")

for t in TARGETS:
    sub = df[df["Target"] == t]
    print(f"\n{t}:")
    print("  rows:", len(sub))
    print("  freq min:", sub["Frequency"].min())
    print("  freq max:", sub["Frequency"].max())
    print("  NaN freq:", sub["Frequency"].isna().sum())

print("done")