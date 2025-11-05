import pandas as pd
import os

# --------------------------------------------------------------
# Script: check_gene_counts.py
# Reads all Task1_results_*.xlsx files in the same folder,
# counts up/downregulated genes per sheet, and exports summary.
# --------------------------------------------------------------

# Find all Excel files in current folder
files = [f for f in os.listdir() if f.startswith("Task1_results_") and f.endswith(".xlsx")]
print(f"📘 Found {len(files)} Task1 files:", files)

summary_rows = []

# Process each file
for f in files:
    try:
        # Read both relevant sheets if they exist
        xls = pd.ExcelFile(f)
        for sheet in ["Exercise_filtered_all", "Cancer_filtered_all"]:
            if sheet in xls.sheet_names:
                df = pd.read_excel(f, sheet_name=sheet)

                # Check that logfc column exists
                if "logfc" not in df.columns:
                    print(f"⚠️  No 'logfc' column in {sheet} of {f}")
                    continue

                # Count
                up = (df["logfc"] > 0).sum()
                down = (df["logfc"] < 0).sum()
                total = len(df)

                summary_rows.append({
                    "File": f,
                    "Sheet": sheet,
                    "Total_genes": total,
                    "Upregulated": up,
                    "Downregulated": down
                })
    except Exception as e:
        print(f"❌ Error reading {f}: {e}")

# Combine results
summary_df = pd.DataFrame(summary_rows)

# Save results
out_file = "Task1_gene_count_summary.xlsx"
summary_df.to_excel(out_file, index=False)

print("\n✅ Summary saved to:", os.path.abspath(out_file))
print(summary_df)

