# compare_visuals_biology_ml.py
# --------------------------------------------------------------
# Sammenligner resultater fra Task 1–4 med ML-pipelinen (Task 5)
# ved å generere fire side-ved-side-plott:
#   1) Overlapp mellom biologiske gener og ML-features
#   2) Pathway heatmap (Task2 vs ML)
#   3) Immun-andel (Task3 vs ML)
#   4) PCA-visning (Task4 data med faktiske vs ML-etiketter)
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
OUT_DIR = Path("plots/comparative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Task-filer
T1_FILE = Path("output/Task1_outputs/Task1_results_standard.xlsx")
T2_FILE = Path("output/Task2_outputs/Task2_results_standard.xlsx")
T3_FILE = Path("output/Task3_outputs/ImmunePathway_summary.xlsx")
T4_FILE = Path("data/LUAD_expression_with_clinical.xlsx")
ML_FEATURES = Path("output/ML_outputs/Feature_importance.xlsx")
ML_METRICS = Path("output/ML_outputs/Model_performance.xlsx")

# --------------------------------------------------------------
# 1️⃣ Overlapp mellom biologiske gener og ML-features
# --------------------------------------------------------------
try:
    t1 = pd.read_excel(T1_FILE, sheet_name="Opp_EXup_CAdown")
    ml = pd.read_excel(ML_FEATURES)
    bio_genes = set(t1["gene"].astype(str).str.upper())
    ml_genes = set(ml["Feature"].astype(str).str.upper())
    overlap = bio_genes & ml_genes

    counts = {
        "Biologiske gener (Task1)": len(bio_genes),
        "ML-top features": len(ml_genes),
        "Overlapp": len(overlap)
    }

    plt.figure(figsize=(4, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="mako")
    plt.title("Overlapp: Biologiske gener vs ML-features")
    plt.ylabel("Antall gener"); plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Overlap_Task1_ML.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[1] Overlapp-plot feilet: {e}")

# --------------------------------------------------------------
# 2️⃣ Pathway-heatmap (Task2 vs ML)
# --------------------------------------------------------------
try:
    t2 = pd.read_excel(T2_FILE, sheet_name="Summary_all")
    t2 = t2[t2["Adjusted P-value"] < 0.05]
    top_terms = t2.groupby("Library").apply(lambda d: d.nsmallest(5, "Adjusted P-value")).reset_index(drop=True)
    top_terms["-log10P"] = -np.log10(top_terms["Adjusted P-value"])

    pivot = top_terms.pivot_table(index="Term", columns="Library", values="-log10P", fill_value=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.5)
    plt.title("Pathway-heatmap: Task2-signifikante pathways")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Pathway_heatmap_Task2.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[2] Pathway-heatmap feilet: {e}")

# --------------------------------------------------------------
# 3️⃣ Immun-andel: hvor mange ML-features er immunrelaterte
# --------------------------------------------------------------
try:
    t3 = pd.read_excel(T3_FILE, sheet_name="Summary")
    immune_terms = []
    for s in pd.ExcelFile(T3_FILE).sheet_names:
        if s != "Summary":
            df = pd.read_excel(T3_FILE, sheet_name=s)
            immune_terms += list(df["Term"].astype(str))
    immune_terms = [t.lower() for t in immune_terms]

    ml = pd.read_excel(ML_FEATURES)
    ml["is_immune"] = ml["Feature"].astype(str).str.lower().apply(
        lambda x: any(k in x for k in ["immune", "tcell", "bcell", "nk", "mhc", "interleukin", "cytokine"])
    )
    immune_pct = ml["is_immune"].mean() * 100

    plt.figure(figsize=(3, 3))
    sns.barplot(x=["ML-features"], y=[immune_pct], color="skyblue")
    plt.ylim(0, 100)
    plt.ylabel("Andel immun-relaterte features (%)")
    plt.title("Immunandel i ML-features (Task3)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Immun_Feature_Percentage.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[3] Immun-andel plot feilet: {e}")

# --------------------------------------------------------------
# 4️⃣ PCA – Klinisk separasjon (Task4-data vs ML-modell)
# --------------------------------------------------------------
try:
    df = pd.read_excel(T4_FILE, sheet_name="Expression")
    # Forsøk å rekonstruere uttrykksmatrise fra LUAD/HC kolonner
    # Finn LUAD og HC kolonner uansett format
    luad_cols = [c for c in df.columns if "luad" in c.lower()]
    hc_cols = [c for c in df.columns if "hc" in c.lower()]

    print(f"🔍 Fant {len(luad_cols)} LUAD-kolonner og {len(hc_cols)} HC-kolonner.")
    if not luad_cols or not hc_cols:
        raise ValueError("Ingen LUAD/HC kolonner funnet. Sjekk kolonnenavnene i miRNA_diffexp_summary.xlsx.")

    X = df[luad_cols + hc_cols].T

    # Finn kolonnen som inneholder miRNA-navn (uansett hva den heter)
    possible_cols = [c for c in df.columns if "mir" in c.lower()]
    if possible_cols:
        id_col = possible_cols[0]
    else:
        raise ValueError("Fant ingen kolonne som inneholder miRNA-navn (eks. 'miRNA' eller 'miRBaseName').")

    X.columns = df[id_col].astype(str)

    labels = np.array([1] * len(luad_cols) + [0] * len(hc_cols))
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Label": labels})
    df_pca["Group"] = df_pca["Label"].map({1: "LUAD", 0: "HC"})

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Group", palette={"LUAD": "red", "HC": "blue"}, alpha=0.8)
    plt.title("Task4 – PCA av LUAD vs HC (kliniske data)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "PCA_Task4_LUAD_HC.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[4] PCA-plot feilet: {e}")

print("\n✅ Ferdig! Alle sammenligningsplott lagret i:")
print(f"📁 {OUT_DIR.resolve()}")
