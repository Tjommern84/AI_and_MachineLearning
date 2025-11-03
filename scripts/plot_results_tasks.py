# plot_results_tasks.py
# --------------------------------------------------------------
# Lager biologiske oversiktsfigurer fra Task 1–4
# og lagrer dem under ./plots/overview/
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

OUT_DIR = Path("plots/overview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- #
#  TASK 1: Overlapp og retning  #
# ----------------------------- #
try:
    t1_path = Path("output/Task1_outputs/Task1_results_standard.xlsx")
    xls = pd.ExcelFile(t1_path)
    counts = {
        "Opp_EXup_CAdown": len(pd.read_excel(xls, "Opp_EXup_CAdown")),
        "Opp_EXdown_CAup": len(pd.read_excel(xls, "Opp_EXdown_CAup")),
        "Shared_UP": len(pd.read_excel(xls, "Shared_UP")),
        "Shared_DOWN": len(pd.read_excel(xls, "Shared_DOWN")),
    }
    plt.figure(figsize=(5, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title("Task 1 – Overlappende og motsatt regulerte gener")
    plt.ylabel("Antall gener")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Task1_gene_overlap.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[Task 1 plot] feilet: {e}")

# ----------------------------- #
#  TASK 2: Pathway-analyse      #
# ----------------------------- #
try:
    t2_path = Path("output/Task2_outputs/Task2_results_standard.xlsx")
    xls = pd.ExcelFile(t2_path)
    sheet = "Summary_all" if "Summary_all" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(t2_path, sheet_name=sheet)
    if "Adjusted P-value" in df.columns and "Term" in df.columns:
        df["-log10(Adj P)"] = -np.log10(df["Adjusted P-value"])
        top10 = df.sort_values("Adjusted P-value").head(10)
        plt.figure(figsize=(6, 4))
        sns.barplot(y="Term", x="-log10(Adj P)", data=top10, hue="Library", dodge=False)
        plt.title("Task 2 – Topp 10 berikede pathways")
        plt.xlabel("-log10(adjusted p-verdi)")
        plt.ylabel("")
        plt.legend(title="Database", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "Task2_top_pathways.png", dpi=300)
        plt.close()
except Exception as e:
    print(f"[Task 2 plot] feilet: {e}")

# ----------------------------- #
#  TASK 3: Immune-pathways      #
# ----------------------------- #
try:
    t3_path = Path("output/Task3_outputs/ImmunePathway_summary.xlsx")
    summary = pd.read_excel(t3_path, sheet_name="Summary")
    if {"Group", "Immune_pathways"}.issubset(summary.columns):
        plt.figure(figsize=(5, 4))
        sns.barplot(data=summary, x="Group", y="Immune_pathways", palette="crest")
        plt.title("Task 3 – Antall immun-pathways per gruppe")
        plt.ylabel("Antall pathways")
        plt.xlabel("Gruppe")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "Task3_immune_pathways.png", dpi=300)
        plt.close()
except Exception as e:
    print(f"[Task 3 plot] feilet: {e}")

# ----------------------------- #
#  TASK 4: miRNA volcano-plot   #
# ----------------------------- #
try:
    t4_path = Path("output/Task4_miRNA_outputs/miRNA_diffexp_summary.xlsx")
    df = pd.read_excel(t4_path, sheet_name="All_miRNA")
    if {"log2FC", "p_value"}.issubset(df.columns):
        df["-log10(p)"] = -np.log10(df["p_value"].replace(0, np.nan))
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df, x="log2FC", y="-log10(p)",
            hue=df["Significant"].map({True: "Signifikant", False: "Ikke signifikant"}),
            palette={"Signifikant": "red", "Ikke signifikant": "gray"},
            alpha=0.7
        )
        plt.axvline(0, color="black", lw=0.8)
        plt.title("Task 4 – Volcano-plot (LUAD vs HC)")
        plt.xlabel("log2 Fold Change")
        plt.ylabel("-log10(p-verdi)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "Task4_volcano.png", dpi=300)
        plt.close()
except Exception as e:
    print(f"[Task 4 plot] feilet: {e}")

print("\n✅ Alle tilgjengelige plott er generert.")
print(f"📁 Lagret i: {OUT_DIR.resolve()}")
