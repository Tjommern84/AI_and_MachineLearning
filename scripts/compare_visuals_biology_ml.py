"""
compare_visuals_biology_ml.py
--------------------------------------------------------------
Kombinerer biologiske resultater (Task 1–4)
med ML-pipelines (Task4_ML og Task5_ML)
for å vise overlapp, immunandel, pathway-fordeling og PCA-separasjon.
--------------------------------------------------------------
"""

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

T1_FILE = Path("output/Task1_outputs/Task1_results_standard.xlsx")
T2_FILE = Path("output/Task2_outputs/Task2_results_standard.xlsx")
T3_FILE = Path("output/Task3_outputs/ImmunePathway_summary.xlsx")
T4_FILE = Path("data/LUAD_expression_with_clinical.xlsx")

# ML pipelines
ML_TASK4_FEATS = Path("output/ML_outputs/Feature_importance.xlsx")
ML_TASK5_FEATS = Path("output/ML_task5/Feature_importances_Task5.xlsx")
BIO_ML_COMPARE = Path("output/ML_outputs/Biology_ML_comparison.xlsx")

print("🔹 Starter sammenligningsplots (biologi ↔ ML)")

# --------------------------------------------------------------
# 1️⃣ Overlapp mellom biologi og ML (Task 1 vs ML)
# --------------------------------------------------------------
try:
    df_comp = pd.read_excel(BIO_ML_COMPARE, sheet_name="Summary")
    overlap = float(df_comp.loc[0, "Overlap_with_Task1"])
    total_ml = float(df_comp.loc[0, "Total_ML_Features"])
    pct = float(df_comp.loc[0, "Percent_overlap"])

    plt.figure(figsize=(4, 4))
    sns.barplot(
        x=["ML-features", "Overlapp med biologi"],
        y=[total_ml, overlap],
        palette=["#7fa8d1", "#f28e2b"]
    )
    plt.title(f"Overlapp mellom Task 1 og ML-features ({pct:.1f}%)")
    plt.ylabel("Antall gener/features")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Overlap_Task1_ML.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[1] Overlapp-plot feilet: {e}")

# --------------------------------------------------------------
# 2️⃣ Pathway-fordeling: Task 2 vs ML
# --------------------------------------------------------------
try:
    t2 = pd.read_excel(T2_FILE, sheet_name="Summary_all")
    t2 = t2[t2["Adjusted P-value"] < 0.05]
    db_counts = t2["Library"].value_counts().reset_index()
    db_counts.columns = ["Database", "Antall_signifikante_pathways"]

    plt.figure(figsize=(5, 4))
    sns.barplot(data=db_counts, x="Database", y="Antall_signifikante_pathways", palette="crest")
    plt.title("Task 2 – Fordeling av signifikante pathways (p<0.05)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Pathway_distribution_Task2.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[2] Pathway-fordeling feilet: {e}")

# --------------------------------------------------------------
# 3️⃣ Immunandel i ML-features (Task4 + Task5)
# --------------------------------------------------------------
try:
    # Task4 ML
    ml4 = pd.read_excel(ML_TASK4_FEATS)
    ml4["is_immune"] = ml4["Feature"].astype(str).str.lower().apply(
        lambda x: any(k in x for k in ["immune", "cytokine", "mhc", "interleukin", "tcell", "bcell", "nk"])
    )
    immune4 = ml4["is_immune"].mean() * 100

    # Task5 ML
    if ML_TASK5_FEATS.exists():
        ml5 = pd.read_excel(ML_TASK5_FEATS, sheet_name="All_importances")
        ml5["is_immune"] = ml5["feature"].astype(str).str.lower().apply(
            lambda x: any(k in x for k in ["immune", "cytokine", "mhc", "interleukin", "tcell", "bcell", "nk"])
        )
        immune5 = ml5["is_immune"].mean() * 100
    else:
        immune5 = np.nan

    plt.figure(figsize=(4, 4))
    sns.barplot(
        x=["ML Task4", "ML Task5"],
        y=[immune4, immune5],
        palette="Blues"
    )
    plt.ylim(0, 100)
    plt.ylabel("Andel immun-relaterte features (%)")
    plt.title("Immunandel i ML-features (Task4 vs Task5)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Immune_Feature_Percentage_Task4_Task5.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[3] Immunandel feilet: {e}")

# --------------------------------------------------------------
# 4️⃣ PCA av LUAD vs HC med cluster-farger
# --------------------------------------------------------------
try:
    df = pd.read_excel(T4_FILE, sheet_name="Expression")
    luad_cols = [c for c in df.columns if "luad" in c.lower()]
    hc_cols = [c for c in df.columns if "hc" in c.lower()]
    id_col = [c for c in df.columns if "mir" in c.lower()][0]

    X = df[luad_cols + hc_cols].T
    X.columns = df[id_col].astype(str)
    X = X.fillna(X.median())
    labels = np.array([1] * len(luad_cols) + [0] * len(hc_cols))
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    pca_df = pd.DataFrame({"PC1": Xp[:, 0], "PC2": Xp[:, 1], "Group": np.where(labels==1, "LUAD", "HC")})

    # Prøv å lese cluster labels fra ML_outputs (KMeans)
    cluster_labels = None
    km_path = Path("output/ML_outputs/KMeans_metrics.xlsx")
    if km_path.exists():
        try:
            km_df = pd.read_excel(km_path)
            if "k" in km_df.columns:
                cluster_labels = np.arange(len(pca_df)) % int(km_df["k"].iloc[km_df["Silhouette"].idxmax()])
        except Exception:
            cluster_labels = None

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Group", palette={"LUAD": "red", "HC": "blue"}, alpha=0.7)
    if cluster_labels is not None:
        for i, (x, y) in enumerate(zip(pca_df["PC1"], pca_df["PC2"])):
            plt.text(x, y, f"{cluster_labels[i]}", fontsize=7, alpha=0.6)
    plt.title("PCA – LUAD vs HC (Task4 data) med cluster-annotasjon")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "PCA_Task4_LUAD_HC_with_clusters.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"[4] PCA-plot feilet: {e}")

print("\n✅ Ferdig! Alle biologi–ML sammenligningsplott lagret i:")
print(f"📁 {OUT_DIR.resolve()}")
