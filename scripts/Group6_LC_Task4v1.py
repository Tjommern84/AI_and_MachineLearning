# Group6_LC_Task4_miRNA.py
# --------------------------------------------------------------
# Task 4 (miRNA version)
# Differential expression analysis between LUAD and HC samples.
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# --------------------------------------------------------------
# Config
# --------------------------------------------------------------
INPUT_FILE = Path("data/LUAD_expression_with_clinical.xlsx")
OUT_DIR = Path("output/Task4_miRNA_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "miRNA_diffexp_summary.xlsx"

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

print(f"📘 Reading {INPUT_FILE.name}")
df = pd.read_excel(INPUT_FILE)

# First column = miRNA names
id_col = df.columns[0]
df[id_col] = df[id_col].astype(str)

# Identify sample columns
luad_cols = [c for c in df.columns if c.lower().startswith("luad")]
hc_cols = [c for c in df.columns if c.lower().startswith("hc")]

print(f"✅ Found {len(luad_cols)} LUAD samples and {len(hc_cols)} HC samples")

# --------------------------------------------------------------
# Differential expression (LUAD vs HC)
# --------------------------------------------------------------
results = []
for _, row in df.iterrows():
    mir = row[id_col]
    luad_vals = row[luad_cols].astype(float)
    hc_vals = row[hc_cols].astype(float)

    if np.all(np.isnan(luad_vals)) or np.all(np.isnan(hc_vals)):
        continue

    mean_luad = np.nanmean(luad_vals)
    mean_hc = np.nanmean(hc_vals)
    log2fc = np.log2((mean_luad + 1e-9) / (mean_hc + 1e-9))

    t_stat, p_val = ttest_ind(luad_vals, hc_vals, nan_policy="omit", equal_var=False)

    results.append({
        "miRNA": mir,
        "mean_LUAD": mean_luad,
        "mean_HC": mean_hc,
        "log2FC": log2fc,
        "p_value": p_val
    })

res_df = pd.DataFrame(results)
res_df["adj_p"] = np.minimum(1, res_df["p_value"] * len(res_df))  # Bonferroni FDR approx
res_df["Significant"] = res_df["adj_p"] < 0.05

# --------------------------------------------------------------
# Split by direction
# --------------------------------------------------------------
up = res_df[(res_df["Significant"]) & (res_df["log2FC"] > 0)].sort_values("adj_p")
down = res_df[(res_df["Significant"]) & (res_df["log2FC"] < 0)].sort_values("adj_p")

summary = pd.DataFrame({
    "Total_miRNA": [len(res_df)],
    "Significant_total": [res_df["Significant"].sum()],
    "Up_in_LUAD": [len(up)],
    "Down_in_LUAD": [len(down)]
})

# --------------------------------------------------------------
# Save results
# --------------------------------------------------------------
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as xlw:
    res_df.to_excel(xlw, sheet_name="All_miRNA", index=False)
    up.to_excel(xlw, sheet_name="Up_in_LUAD", index=False)
    down.to_excel(xlw, sheet_name="Down_in_LUAD", index=False)
    summary.to_excel(xlw, sheet_name="Summary", index=False)

print(f"✅ Differential expression complete.\nResults saved to: {OUT_FILE.resolve()}")
