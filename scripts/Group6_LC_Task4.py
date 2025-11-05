# --------------------------------------------------------------
# Group6_LC_Task4_v2_fixed.py
# --------------------------------------------------------------
# Differential miRNA expression analysis for LUAD vs HC
# Updated to use Benjamini–Hochberg FDR (fdr_bh)
# Reads group info directly from Expression sheet
# Output identical to Task4 but with '_v2' suffix
# --------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------
# File paths
# --------------------------------------------------------------
input_file = os.path.join("data", "LUAD_expression_with_clinical.xlsx")
output_dir = os.path.join( "output", "Task4_miRNA_outputs")
os.makedirs(output_dir, exist_ok=True)

print("📘 Task4_v2 – Starting differential miRNA expression analysis (BH-FDR)")

# --------------------------------------------------------------
# Read Expression data only
# --------------------------------------------------------------
df = pd.read_excel(input_file, sheet_name="Expression")
id_col = df.columns[0]
df[id_col] = df[id_col].astype(str)
df = df.set_index(id_col)

# Identify LUAD and HC columns
luad_cols = [c for c in df.columns if "LUAD" in c.upper()]
hc_cols = [c for c in df.columns if "HC" in c.upper()]

if not luad_cols or not hc_cols:
    raise ValueError("No LUAD/HC columns found in Expression sheet. Check headers.")

print(f"[info] LUAD samples: {len(luad_cols)} | HC samples: {len(hc_cols)} | miRNAs: {df.shape[0]}")

# --------------------------------------------------------------
# Differential expression (Welch’s t-test)
# --------------------------------------------------------------
results = []
for mir, row in df.iterrows():
    x1 = row[luad_cols].astype(float)
    x2 = row[hc_cols].astype(float)
    if x1.var() == 0 and x2.var() == 0:
        continue
    tstat, pval = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    log2fc = np.log2(np.mean(x1) + 1e-9) - np.log2(np.mean(x2) + 1e-9)
    results.append([mir, log2fc, pval])

res_df = pd.DataFrame(results, columns=["miRNA", "log2FC", "p_value"])

# --------------------------------------------------------------
# Multiple testing correction (Benjamini–Hochberg)
# --------------------------------------------------------------
res_df["adj_p"] = multipletests(res_df["p_value"], method="fdr_bh")[1]

# Classify significance and direction
res_df["Direction"] = np.where(res_df["log2FC"] > 0, "Up_in_LUAD", "Down_in_LUAD")
res_df["Significant"] = (res_df["adj_p"] < 0.05) & (abs(res_df["log2FC"]) > 1.0)

# Subsets
up_df = res_df[(res_df["Significant"]) & (res_df["Direction"] == "Up_in_LUAD")]
down_df = res_df[(res_df["Significant"]) & (res_df["Direction"] == "Down_in_LUAD")]

# --------------------------------------------------------------
# Summary table
# --------------------------------------------------------------
summary = pd.DataFrame({
    "Total_miRNA": [len(res_df)],
    "Significant_total": [len(res_df[res_df["Significant"]])],
    "Up_in_LUAD": [len(up_df)],
    "Down_in_LUAD": [len(down_df)],
    "FDR_method": ["Benjamini–Hochberg (fdr_bh)"]
})

# --------------------------------------------------------------
# Save results
# --------------------------------------------------------------
output_file = os.path.join(output_dir, "miRNA_diffexp_summary_v2.xlsx")
with pd.ExcelWriter(output_file) as writer:
    res_df.to_excel(writer, sheet_name="All_miRNA_v2", index=False)
    up_df.to_excel(writer, sheet_name="Up_in_LUAD_v2", index=False)
    down_df.to_excel(writer, sheet_name="Down_in_LUAD_v2", index=False)
    summary.to_excel(writer, sheet_name="Summary_v2", index=False)

print(f"✅ Results saved to: {output_file}")
print(summary.to_string(index=False))

# Optional CSV copies
res_df.to_csv(os.path.join(output_dir, "All_miRNA_v2.csv"), index=False)
up_df.to_csv(os.path.join(output_dir, "Up_in_LUAD_v2.csv"), index=False)
down_df.to_csv(os.path.join(output_dir, "Down_in_LUAD_v2.csv"), index=False)
summary.to_csv(os.path.join(output_dir, "Summary_v2.csv"), index=False)

print("🏁 Task4_v2 completed successfully.")
