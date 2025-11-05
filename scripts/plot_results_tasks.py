"""
Group6_LC_Plots_and_Summary.py
-------------------------------------------------------
Genererer ALLE biologiske plott og oppsummeringstabell
fra Task 1–4 + ML sammenligning.
Output:
 - plots/overview/Task*_*.png
 - plots/overview/summary_table.png
-------------------------------------------------------
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

sns.set(style="whitegrid")
OUT_DIR = os.path.join("plots", "overview")
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# 1️⃣ TASK 1 – Antall differensielt uttrykte gener
# ==========================================================
task1_path = os.path.join("output", "Task1_outputs")
dfs = []
for file in os.listdir(task1_path):
    if file.endswith(".xlsx") and "results" in file:
        tag = file.replace("Task1_results_", "").replace(".xlsx", "")
        df = pd.read_excel(os.path.join(task1_path, file))
        dfs.append({"Filter": tag, "Rows": len(df)})

df1 = pd.DataFrame(dfs)
if not df1.empty:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df1, x="Filter", y="Rows", palette="viridis")
    plt.title("Task 1 – Antall differensielt uttrykte gener per terskel")
    plt.ylabel("Antall gener")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Task1_DEG_counts.png"), dpi=300)
    plt.close()

# ==========================================================
# 2️⃣ TASK 2 – Topp 10 pathways (uten legend, med database-navn)
# ==========================================================
t2 = os.path.join("output", "Task2_outputs", "Task2_results_standard.xlsx")
if os.path.exists(t2):
    df = pd.read_excel(t2, sheet_name="Summary_all")
    df = df[df["Adjusted P-value"] < 0.05].copy()
    if not df.empty:
        df["-log10(p_adj)"] = -np.log10(df["Adjusted P-value"].replace(0, np.nan))

        # Bruker nlargest for å få de 10 med høyest -log10(p_adj)
        top10 = df.nlargest(10, "-log10(p_adj)")[["Term", "-log10(p_adj)", "Library"]].copy()

        # Lager den lange etiketten med database
        top10["Full_Term"] = top10.apply(lambda x: f"{x['Term']} ({x['Library']})", axis=1)

        # Sorterer synkende for horisontalt barplott
        top10 = top10.sort_values("-log10(p_adj)", ascending=False)

        # Øker figurhøyden for å gi plass til lange tekster
        plt.figure(figsize=(11, 8))

        sns.barplot(
            data=top10,
            x="-log10(p_adj)",
            y="Full_Term",
            hue="Library",
            palette="deep",
            dodge=False
        )

        plt.xlabel("-log10(adjusted p-verdi)", fontsize=12)
        plt.ylabel("Pathway (med database)", fontsize=12)

        # JUSTERING: Midtstiller og øker avstanden (pad) for tittelen
        plt.title(
            "Task 2 – Topp 10 berikede pathways (etter database)",
            fontsize=14,
            pad=25,  # Økt padding
            loc='center'  # Midtstilt
        )

        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Bruk bbox_inches='tight' for å være helt sikker på at ingenting kuttes
        plt.savefig(os.path.join("plots", "overview", "Task2_top10_pathways_formatted_revised.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
# ----------------------------------------------------------
# 3️⃣ Task 3 – Immune pathway summary (original stil + 45° tekst)
# ----------------------------------------------------------
t3 = os.path.join("output", "Task3_outputs", "ImmunePathway_summary.xlsx")
if os.path.exists(t3):
    df = pd.read_excel(t3)

    # Finn automatisk kolonne med antall pathways
    possible_cols = [c for c in df.columns if "immune" in c.lower() or "pathway" in c.lower()]
    y_col = possible_cols[0] if possible_cols else df.columns[1]

    print(f"📘 Task3: bruker kolonne '{y_col}' som antall-pathways.")

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Group", y=y_col, palette="viridis")
    plt.title("Task 3 – Antall immunrelaterte pathways per gruppe", fontsize=13)
    plt.ylabel("Antall pathways", fontsize=11)
    plt.xlabel("Gruppe", fontsize=11)

    # 45° rotering av etiketter, så de ikke overlapper
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Task3_immune_pathway_summary.png"), dpi=300)
    plt.close()
else:
    print("⚠️ Fant ikke ImmunePathway_summary.xlsx – hopper over Task 3.")


# ==========================================================
# 4️⃣ TASK 4 – PCA LUAD vs HC + Volcano-plot
# ==========================================================
expr = os.path.join("data", "LUAD_expression_with_clinical.xlsx")
if os.path.exists(expr):
    df = pd.read_excel(expr, sheet_name="Expression")
    id_col = [c for c in df.columns if "miR" in c or "mirbase" in c.lower()][0]
    luad_cols = [c for c in df.columns if "LUAD" in c]
    hc_cols = [c for c in df.columns if "HC_" in c]
    X = df[luad_cols + hc_cols].T
    X.columns = df[id_col].astype(str)
    y = np.array([1]*len(luad_cols) + [0]*len(hc_cols))
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": Xp[:,0], "PC2": Xp[:,1], "Group": np.where(y==1,"LUAD","HC")})
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Group", palette={"LUAD":"red","HC":"blue"})
    plt.text(
        1.05, 0.5,
        "Forklaring:\n• Rød = LUAD (lungekreft)\n• Blå = HC (kontroll)\n• Hver prikk = én prøve\n• Klynger viser mønster i uttrykk",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
    )
    plt.title("Task 4 – PCA av LUAD vs HC (miRNA-uttrykk)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Task4_PCA_LUAD_HC.png"), dpi=300)
    plt.close()

# ----------------------------------------------------------
# Volcano plot (Task 4)
# ----------------------------------------------------------
volcano = os.path.join("output", "Task4_outputs", "LUAD_vs_HC_results.xlsx")
if os.path.exists(volcano):
    vdf = pd.read_excel(volcano)
    vdf["-log10(pval)"] = -np.log10(vdf["pval"].replace(0, np.nan))
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=vdf, x="log2FC", y="-log10(pval)", color="gray", alpha=0.7)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(p-verdi)")
    plt.title("Task 4 – Volcano-plot for differensiell miRNA-ekspresjon (LUAD vs HC)", fontsize=13)
    plt.text(
        0.95, 0.5,
        "• Hver prikk = ett miRNA\n• X: log₂ endring (LUAD vs HC)\n• Y: -log₁₀(p-verdi)\n• Høyre = oppregulert\n• Venstre = nedregulert",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Task4_volcano_plot.png"), dpi=300)
    plt.close()

# ==========================================================
# 5️⃣ SAMMENDRAGSTABELL MED INTERPRETASJON
# ==========================================================
summary_data = [
    ["Task1 – Differential expression", "46 up, 8 down genes", "54 DEGs total", "-", "Exercise activates immune/metabolic genes suppressed in NSCLC"],
    ["Task2 – Pathway enrichment", "6 significant pathways", "Ribosome, TP53, RAC1", "-", "Cellular stress and translation regulation"],
    ["Task3 – Immune pathways", "685 immune-related pathways", "NSCLC-down > Exercise", "-", "Immune activation enhanced by exercise"],
    ["Task4 – LUAD miRNA differential", "0 significant", "–", "-", "No clear miRNA separation between LUAD and HC"],
    ["ML – MLP_100", "–", "Acc=0.286", "AUC=0.417", "Neural network failed to classify accurately"],
    ["ML – LogReg_L1", "–", "Acc=0.429", "AUC=0.333", "Weak linear separation of LUAD vs HC"],
    ["ML – KMeans", "–", "Best k=5", "Silhouette=0.186", "No clear clusters, low ARI"],
    ["Compare_Biology overlap", "–", "–", "0%", "No overlap between ML and Task1 gene features"]
]

summary_df = pd.DataFrame(summary_data, columns=["Analysis", "Key Findings", "Quantitative Summary", "Performance", "Interpretation"])

fig, ax = plt.subplots(figsize=(13,4))
ax.axis('off')
tbl = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='left', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
plt.title("Oppsummering av hovedfunn fra Tasks 1–4 og ML-pipeline", pad=20)
plt.savefig(os.path.join(OUT_DIR, "summary_results_table.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Ferdig! Alle plott og oppsummering lagret i: {OUT_DIR}")
