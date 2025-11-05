import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths (two levels up)
input_file = os.path.join("..", "..", "output", "Task1_outputs", "Task1_results_standard.xlsx")
out_dir = os.path.join("..", "..", "plots", "overview")
os.makedirs(out_dir, exist_ok=True)

# Read Excel file and all sheets
xls = pd.ExcelFile(input_file)

# Keep only comparison sheets
compare_sheets = ["Opp_EXup_CAdown", "Opp_EXdown_CAup", "Shared_UP", "Shared_DOWN"]

counts = []
for sheet in compare_sheets:
    if sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        counts.append({"Category": sheet, "GeneCount": len(df)})

counts_df = pd.DataFrame(counts)

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(data=counts_df, x="Category", y="GeneCount", palette="deep")
plt.title("Task 1 – Overlapping and Oppositely Regulated Genes", fontsize=12, pad=10)
plt.ylabel("Number of Genes")
plt.xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Task1_Overlap_Genes.png"), dpi=300)
plt.close()
