import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# Reads Task1_gene_count_summary.xlsx and visualizes results
# --------------------------------------------------------------

# Load data
file = "Task1_gene_count_summary.xlsx"
df = pd.read_excel(file)

# Clean up names for readability
df["Filter"] = df["File"].str.extract(r"results_(.*)\.xlsx")
df["Type"] = df["Sheet"].str.replace("_filtered_all", "", regex=False)

# Create grouped bar plot for up/downregulated genes
fig, ax = plt.subplots(figsize=(9, 5))

# Prepare data for plotting
pivot_up = df.pivot(index="Filter", columns="Type", values="Upregulated")
pivot_down = df.pivot(index="Filter", columns="Type", values="Downregulated")

# Plot bars
width = 0.35
x = range(len(pivot_up.index))
ax.bar([i - width/2 for i in x], pivot_up["Exercise"], width, label="Exercise - Upregulated", color="#4CAF50")
ax.bar([i + width/2 for i in x], pivot_up["Cancer"], width, label="Cancer - Upregulated", color="#81C784", alpha=0.7)

# Add second pair for downregulated on new figure
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar([i - width/2 for i in x], pivot_down["Exercise"], width, label="Exercise - Downregulated", color="#2196F3")
ax2.bar([i + width/2 for i in x], pivot_down["Cancer"], width, label="Cancer - Downregulated", color="#64B5F6", alpha=0.7)

# Titles and formatting
for a, title in zip([ax, ax2], ["Upregulated Genes", "Downregulated Genes"]):
    a.set_title(f"Task1 Gene Counts – {title}")
    a.set_xlabel("Filter level")
    a.set_ylabel("Number of genes")
    a.set_xticks(x)
    a.set_xticklabels(pivot_up.index, rotation=0)
    a.legend()
    a.grid(True, axis="y", alpha=0.3)

# Save figures
out1 = "Task1_gene_upregulated_plot.png"
out2 = "Task1_gene_downregulated_plot.png"
fig.tight_layout()
fig2.tight_layout()
fig.savefig(out1, dpi=300)
fig2.savefig(out2, dpi=300)

print(f"✅ Saved plots:\n - {os.path.abspath(out1)}\n - {os.path.abspath(out2)}")
plt.show()
