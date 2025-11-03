# Group6_LC_ML_Models_v2.py
# --------------------------------------------------------------
# Task 5 — ML-pipeline som kjører parallelt med Task 1–4
# Bruker LUAD_expression_with_clinical.xlsx (samme som Task 4)
# Modeller:
#   1) Logistic Regression (L1)  [supervised, fra Lecture 3]
#   2) MLPClassifier (nevralt nett) [supervised, fra Lecture 3]
#   3) K-Means (egen implementasjon) [unsupervised, fra Lecture 4 + deres Part 1–3]
#
# Output (kun .xlsx + figurer):
#   ML_outputs/Model_performance.xlsx
#   ML_outputs/Feature_importance.xlsx
#   ML_outputs/KMeans_metrics.xlsx
#   ML_outputs/figures/ROC_LogReg.png, ROC_MLP.png, Elbow.png, Silhouette_vs_k.png
# --------------------------------------------------------------

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    classification_report, adjusted_rand_score
)

# --------------------------------------------------------------
# Konfig
# --------------------------------------------------------------
DATA_FILE = Path("data/LUAD_expression_with_clinical.xlsx")
OUT_DIR = Path("output/ML_outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_METRICS = OUT_DIR / "Model_performance.xlsx"
OUT_FEATURES = OUT_DIR / "Feature_importance.xlsx"
OUT_KMEANS = OUT_DIR / "KMeans_metrics.xlsx"

RANDOM_STATE = 42

# --------------------------------------------------------------
# Utils: standardisering, egen K-Means, silhouette (fra deres Part 1–3)
# --------------------------------------------------------------
def standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def kmeans(X, k, max_iter=100, tol=1e-4, n_init=10, random_state=None):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")
    rng = np.random.default_rng(random_state)

    best = None
    best_inertia = np.inf

    for _ in range(n_init):
        init_idx = rng.choice(n, size=k, replace=False)
        centroid = X[init_idx].copy()

        for i in range(1, max_iter + 1):
            dist = ((X[:, None, :] - centroid[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)

            new_centroid = centroid.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centroid[j] = members.mean(axis=0)
                else:
                    new_centroid[j] = X[rng.integers(0, n)]
            shift = np.linalg.norm(new_centroid - centroid)
            centroid = new_centroid
            if shift < tol:
                break

        final_dist = ((X - centroid[labels]) ** 2).sum(axis=1)
        inertia = float(final_dist.sum()) / n
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels, centroid, i, inertia)
    return best  # labels, centroid, n_iter, inertia(per punkt)

def silhouette_score_euclidean(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    # avstands-matrise
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    clusters = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    s = np.zeros(n, dtype=float)

    for i in range(n):
        ci = labels[i]
        in_ci = clusters[ci]
        # a(i): gj.snitt avstand til egen klynge
        if len(in_ci) > 1:
            a = D[i, in_ci].sum() / (len(in_ci) - 1)
        else:
            a = 0.0
        # b(i): laveste gj.snitt avstand til andre klynger
        b_candidates = [D[i, idxs].mean() for cj, idxs in clusters.items() if cj != ci and len(idxs) > 0]
        b = min(b_candidates) if b_candidates else 0.0
        denom = max(a, b)
        s[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(s.mean())

# --------------------------------------------------------------
# Les data (samme kilde/format som Task 4)
#   - Første kolonne = ID (miR/gene), resten = prøver (LUAD_x, HC_y)
#   - Vi lager X (samples × features) og y (1=LUAD, 0=HC)
# --------------------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {DATA_FILE.resolve()}")

print(f"📘 Reading {DATA_FILE.name}")
xls = pd.ExcelFile(DATA_FILE)
expr_df = pd.read_excel(xls, "Expression")  # [miRBaseName | LUAD_1 ... | HC_1 ...]
clin_df = None
try:
    clin_df = pd.read_excel(xls, "Clinical")
except Exception:
    pass  # klinikk valgfritt her

# Sett indeks og transponer: prøver som rader
id_col = expr_df.columns[0]
expr_df[id_col] = expr_df[id_col].astype(str)
expr = expr_df.set_index(id_col).T
expr.index = expr.index.astype(str)

# Lag labels fra kolonnenavn
labels = []
for s in expr.index:
    s_low = s.lower()
    if s_low.startswith("luad"):
        labels.append(1)
    elif s_low.startswith("hc"):
        labels.append(0)
    else:
        labels.append(np.nan)
expr["label"] = labels
expr = expr.dropna(subset=["label"])
expr["label"] = expr["label"].astype(int)

# Enkel imputering (median) + standardisering
X = expr.drop(columns=["label"]).astype(float)
X = X.fillna(X.median(axis=0))
y = expr["label"].values

Xs, mu, sigma = standardize(X.values)
print(f"✅ Samples: {Xs.shape[0]} | Features: {Xs.shape[1]} | LUAD={int((y==1).sum())} | HC={int((y==0).sum())}")

# --------------------------------------------------------------
# Train/test split (for supervised)
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# --------------------------------------------------------------
# Supervised modeller: Logistic Regression (L1), MLP
# --------------------------------------------------------------
models = {
    "LogReg_L1": LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=3000, random_state=RANDOM_STATE
    ),
    "MLP_100": MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        alpha=1e-4, batch_size="auto", learning_rate_init=1e-3,
        max_iter=400, random_state=RANDOM_STATE
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
roc_data = {}  # for plotting

for name, model in models.items():
    print(f"\n🔹 Trener modell: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for modeller uten predict_proba
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # skaler til [0,1] med min-max for ROC (ikke perfekt, men ok)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_prob = None

    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    try:
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    except Exception:
        cv_auc = np.nan

    print(f"   → Test Accuracy: {acc:.3f} | Test ROC-AUC: {auc_val:.3f}")
    print(f"   → CV Accuracy:   {cv_acc:.3f} | CV ROC-AUC: {cv_auc:.3f}")

    rows.append({
        "Model": name,
        "Test_Accuracy": acc,
        "Test_ROC_AUC": auc_val,
        "CV_Accuracy": cv_acc,
        "CV_ROC_AUC": cv_auc
    })

    # ROC-kurve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

# Lagre metrikker
metrics_df = pd.DataFrame(rows).sort_values("Test_ROC_AUC", ascending=False)
with pd.ExcelWriter(OUT_METRICS, engine="openpyxl") as xlw:
    metrics_df.to_excel(xlw, sheet_name="Supervised", index=False)
print(f"📊 Lagret: {OUT_METRICS.resolve()}")

# Feature-importance: LogReg L1 (absolutte koeffisienter)
feat_imp = []
if "LogReg_L1" in models:
    lr = models["LogReg_L1"]
    coefs = lr.coef_[0]
    abs_coefs = np.abs(coefs)
    top_idx = np.argsort(-abs_coefs)[:50]
    top_feats = X.columns[top_idx]
    top_vals = abs_coefs[top_idx]
    feat_imp = pd.DataFrame({
        "Feature": top_feats,
        "Weight_abs": top_vals,
        "Model": "LogReg_L1"
    })

if len(feat_imp) > 0:
    with pd.ExcelWriter(OUT_FEATURES, engine="openpyxl") as xlw:
        feat_imp.to_excel(xlw, sheet_name="Top50_LogReg_L1", index=False)
    print(f"📈 Lagret: {OUT_FEATURES.resolve()}")

# ROC-plots
for name, (fpr, tpr, aucv) in roc_data.items():
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ROC_{name}.png", dpi=300)
    plt.close()

# --------------------------------------------------------------
# Unsupervised: K-Means (egen) — elbow & silhouette, + ARI mot fasit
# --------------------------------------------------------------
ks = [2, 3, 4, 5]
inertias, silhouettes, aris = [], [], []

for k in ks:
    labels, centroids, n_iter, inertia = kmeans(Xs, k=k, n_init=20, random_state=RANDOM_STATE)
    sil = silhouette_score_euclidean(Xs, labels)
    # juster labels ikke nødvendig for ARI; ARI er invariant for label-permutasjoner
    ari = adjusted_rand_score(y, labels)
    inertias.append(inertia)
    silhouettes.append(sil)
    aris.append(ari)
    print(f"[KMeans] k={k} | iters={n_iter:2d} | inertia={inertia:.4f} | silhouette={sil:.3f} | ARI={ari:.3f}")

kmeans_df = pd.DataFrame({"k": ks, "Inertia_per_point": inertias, "Silhouette": silhouettes, "ARI_vs_labels": aris})
with pd.ExcelWriter(OUT_KMEANS, engine="openpyxl") as xlw:
    kmeans_df.to_excel(xlw, sheet_name="KMeans", index=False)
print(f"🧩 Lagret: {OUT_KMEANS.resolve()}")

# Elbow-plot
plt.figure(figsize=(5,4))
plt.plot(ks, inertias, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia (per punkt)")
plt.title("Elbow (LUAD/HC, standardisert)")
plt.tight_layout()
plt.savefig(FIG_DIR / "Elbow.png", dpi=300)
plt.close()

# Silhouette vs k
plt.figure(figsize=(5,4))
plt.plot(ks, silhouettes, marker="o")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.title("Silhouette vs k")
plt.tight_layout()
plt.savefig(FIG_DIR / "Silhouette_vs_k.png", dpi=300)
plt.close()

print("\n✅ Ferdig. Resultater skrevet til:")
print(f"   - {OUT_METRICS.resolve()}")
print(f"   - {OUT_FEATURES.resolve() if len(feat_imp)>0 else '(ingen Feature_importance for MLP)'}")
print(f"   - {OUT_KMEANS.resolve()}")
print(f"   - Figurer i {FIG_DIR.resolve()}")
