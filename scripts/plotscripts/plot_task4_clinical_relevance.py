# --------------------------------------------------------------
# plot_task4_clinical_relevance.py
# --------------------------------------------------------------
# Visualizes differential miRNA expression results from Task 4
# as a heatmap of z-scored values (LUAD vs Healthy Control).
#
# Input:
#   ../../output/Task4_miRNA_outputs/miRNA_diffexp_summary.xlsx
#   ../../data/LUAD_expression_with_clinical.xlsx
#
# Output:
#   ../../plots/overview/miRNA_expression_heatmap.png
#
# If no significant miRNAs pass filters, the script falls back
# to the top-N lowest adjusted p-values.
# --------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# File paths (relative to this script)
# --------------------------------------------------------------
task4_file = os.path.join("..", "..", "output", "Task4_miRNA_outputs", "miRNA_diffexp_summary.xlsx")
expr_file  = os.path.join("..", "..", "data", "LUAD_expression_with_clinical.xlsx")
out_dir    = os.path.join("..", "..", "plots", "overview")

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
TOP_N = 25
ADJ_P_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
CMAP = "RdBu_r"

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------
def read_significant_mirnas(file_path):
    """Return list of significant miRNAs (fallback: top-N by p-value)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task 4 results not found: {file_path}")

    df = pd.read_excel(file_path, sheet_name="All_miRNA")

    required = {"miRNA", "adj_p", "log2FC"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in Task 4 results: {required}")

    sig = df[(df["adj_p"] < ADJ_P_THRESHOLD) & (df["log2FC"].abs() > LOG2FC_THRESHOLD)]
    if sig.empty:
        print(f"[warn] No miRNAs passed filters; using top {TOP_N} by adjusted p-value.")
        sig = df.nsmallest(TOP_N, "adj_p")

    mirnas = sig["miRNA"].dropna().astype(str).unique().tolist()
    if not mirnas:
        raise ValueError("No valid miRNA names found in Task 4 results.")
    print(f"[info] Selected {len(mirnas)} miRNAs for visualization.")
    return mirnas


def load_expression_matrix(file_path):
    """Load LUAD expression matrix."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expression file not found: {file_path}")

    expr = pd.read_excel(file_path, sheet_name="Expression")
    id_col = expr.columns[0]
    expr[id_col] = expr[id_col].astype(str)
    expr = expr.set_index(id_col)
    expr = expr.select_dtypes(include=[np.number])
    return expr


def zscore_matrix(df):
    """Z-score rows (miRNAs) across samples."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    z = z.dropna(how="all")
    return z


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == "__main__":
    print("📘 Generating clinical relevance heatmap...")

    # 1) Read miRNA list
    mir_list = read_significant_mirnas(task4_file)

    # 2) Load expression data
    expr = load_expression_matrix(expr_file)

    # 3) Subset to overlapping miRNAs
    overlap = [m for m in mir_list if m in expr.index]
    if len(overlap) == 0:
        raise ValueError("No overlap between selected miRNAs and the Expression matrix index.")
    expr_sub = expr.loc[overlap].copy()
    print(f"[info] Expression subset: {expr_sub.shape[0]} miRNAs × {expr_sub.shape[1]} samples.")

    # 4) Z-score normalization
    z_data = zscore_matrix(expr_sub)
    if z_data.empty:
        raise ValueError("Z-scored data is empty. Relax thresholds or verify data consistency.")

    # 5) Optional column ordering (LUAD first, HC next)
    def sample_sort_key(col):
        c = str(col).lower()
        if c.startswith("luad"): return (0, c)
        if c.startswith("hc"):   return (1, c)
        return (2, c)

    z_data = z_data.reindex(sorted(z_data.columns, key=sample_sort_key), axis=1)

    # 6) Plot
    fig_height = max(6, 0.3 * len(z_data))
    plt.figure(figsize=(10, fig_height))
    sns.heatmap(z_data, cmap=CMAP, center=0, cbar_kws={"label": "Z-score"})
    plt.title("LUAD vs Healthy Controls — miRNA Expression (Z-scored per miRNA)")
    plt.xlabel("Samples")
    plt.ylabel("miRNA")
    plt.tight_layout()

    # 7) Save plot
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "miRNA_expression_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Heatmap saved to: {out_path}")
