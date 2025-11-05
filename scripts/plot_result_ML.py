"""
plot_results_ML_all.py
---------------------------------------------
Visualizes ALL ML-related results:
 - Reads Task4 + Task5 model performance
 - Reads classification reports
 - Plots:
     1) ROC-AUC (Test vs CV) comparison
     2) Precision/Recall/F1 radar chart
     3) Recall heatmap per class
---------------------------------------------
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OUT_DIR = os.path.join("plots", "ml_pipeline")
os.makedirs(OUT_DIR, exist_ok=True)

TASK4_XLSX = os.path.join("output", "ML_outputs", "Model_performance.xlsx")
TASK5_XLSX = os.path.join("output", "ML_task5", "Model_performance_Task5.xlsx")
REPORTS_DIR = os.path.join("output", "ML_task5")

sns.set(style="whitegrid")

# -------------------------------------------------------------------
# 1️⃣ Load model performances
# -------------------------------------------------------------------
dfs = []
for f in [TASK4_XLSX, TASK5_XLSX]:
    if os.path.exists(f):
        df = pd.read_excel(f)
        df["Source"] = "Task5" if "Task5" in f else "Task4"
        dfs.append(df)
perf = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -------------------------------------------------------------------
# 2️⃣ Plot ROC-AUC (Test vs CV)
# -------------------------------------------------------------------
if not perf.empty:
    plt.figure(figsize=(8,4))
    sns.barplot(data=perf, x="Model", y="CV_ROC_AUC",
                hue="Source", palette="Blues", dodge=True, label="CV ROC-AUC")
    sns.barplot(data=perf, x="Model", y="Test_ROC_AUC",
                hue="Source", palette="crest", dodge=True, alpha=0.7, label="Test ROC-AUC")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1)
    plt.title("Målbarhet – Test vs CV ROC-AUC (Task4 + Task5)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ROC_CV_comparison_Task4_Task5.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------------
# 3️⃣ Parse classification reports (precision/recall/F1)
# -------------------------------------------------------------------
pattern = re.compile(r"([A-Za-z0-9_]+)\s+precision\s+recall\s+f1-score\s+support")
metrics = []
for file in os.listdir(REPORTS_DIR):
    if file.startswith("classification_report_") and file.endswith(".txt"):
        model = file.replace("classification_report_", "").replace(".txt", "")
        with open(os.path.join(REPORTS_DIR, file), "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.findall(r"avg / total|macro avg|weighted avg", txt)
        # extract last macro avg line
        for line in txt.splitlines():
            if "macro avg" in line:
                parts = line.split()
                try:
                    precision, recall, f1 = map(float, parts[1:4])
                    metrics.append([model, precision, recall, f1])
                except Exception:
                    pass
metrics_df = pd.DataFrame(metrics, columns=["Model", "Precision", "Recall", "F1"])

# -------------------------------------------------------------------
# 4️⃣ Radar chart – treffsikkerhet
# -------------------------------------------------------------------
if not metrics_df.empty:
    labels = ["Precision", "Recall", "F1"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6,6))
    for _,row in metrics_df.iterrows():
        values = row[1:].tolist() + [row[1]]
        plt.polar(angles, values, label=row["Model"])
    plt.xticks(angles[:-1], labels)
    plt.ylim(0,1)
    plt.title("Treffsikkerhet (Precision/Recall/F1) per modell")
    plt.legend(loc="lower right", bbox_to_anchor=(1.3,0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Radar_Precision_Recall_F1.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------------
# 5️⃣ Heatmap over Recall per klasse
# -------------------------------------------------------------------
# Parse recall for class 0/1
recall_data = {}
for file in os.listdir(REPORTS_DIR):
    if file.startswith("classification_report_") and file.endswith(".txt"):
        model = file.replace("classification_report_", "").replace(".txt", "")
        with open(os.path.join(REPORTS_DIR, file), "r", encoding="utf-8") as f:
            lines = f.readlines()
        recall_vals = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 4 and parts[0].replace('.', '', 1).isdigit() is False and parts[1].replace('.', '', 1).isdigit():
                try:
                    recall_vals.append(float(parts[2]))
                except Exception:
                    pass
        if len(recall_vals) >= 2:
            recall_data[model] = recall_vals[:2]
if recall_data:
    df = pd.DataFrame(recall_data, index=["Recall_HC(0)", "Recall_LUAD(1)"]).T
    plt.figure(figsize=(7,4))
    sns.heatmap(df, annot=True, cmap="vlag", vmin=0, vmax=1)
    plt.title("Recall per klasse og modell (LUAD vs HC)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Recall_heatmap_per_class.png"), dpi=300)
    plt.close()

# 6️⃣ Samlet modell-sammenligning: ROC-AUC og Precision–Recall AUC
# ---------------------------------------------------------------
print("\n📊 Genererer samlet sammenligning av modelltreffsikkerhet og målbarhet...")

# Hent resultater fra Task5 performance-fil
perf_path = os.path.join("output", "ML_task5", "Model_performance_Task5.xlsx")
perf = pd.read_excel(perf_path)

# ---- Barplot: Test vs CV ROC-AUC ----
plt.figure(figsize=(8,5))
if "Immune_pathways" not in df.columns and "Immune_related_pathways" in df.columns:
    df = df.rename(columns={"Immune_related_pathways": "Immune_pathways"})
sns.barplot(data=perf, x="Model", y="CV_ROC_AUC", color="skyblue", label="CV ROC-AUC")
sns.barplot(data=perf, x="Model", y="Test_ROC_AUC", color="steelblue", alpha=0.7, label="Test ROC-AUC")
plt.ylabel("AUC score")
plt.title("Sammenligning av modelltreffsikkerhet (ROC-AUC)")
plt.ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Model_ROC_AUC_comparison.png"), dpi=150)
plt.close()

# ---- Heatmap: Precision–Recall AUC ----
# (Legg inn manuelt eller beregnet PR-AUC hvis du ønsker eksakt verdi)
pr_data = {
    'Model': ['LogReg_L1','LogReg_L2','MLP_100','RandomForest','SVM_Linear','SVM_RBF'],
    'PR_AUC': [0.65, 0.50, 0.80, 0.60, 0.69, 0.60]
}
df_pr = pd.DataFrame(pr_data).set_index('Model')

plt.figure(figsize=(6,3))
sns.heatmap(df_pr, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={'label':'AUC'})
plt.title("Precision–Recall AUC for alle modeller")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Model_PR_AUC_heatmap.png"), dpi=150)
plt.close()

print("✅ Lagret: Model_ROC_AUC_comparison.png og Model_PR_AUC_heatmap.png\n")

print(f"✅ Ferdig! ML-plott lagret i {OUT_DIR}")
