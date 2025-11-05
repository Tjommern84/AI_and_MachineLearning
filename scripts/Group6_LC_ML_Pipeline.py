# --------------------------------------------------------------
# Group6_LC_ML_Pipeline.py
# Machine Learning pipeline for LUAD vs. Healthy Control samples
# --------------------------------------------------------------
# Models implemented:
#   1) Logistic Regression (L1-regularized)
#   2) Multi-Layer Perceptron (neural network)
#   3) Support Vector Machine (RBF kernel)
#   4) Custom K-Means clustering (unsupervised)
#
# Output:
#   - output/ML_outputs/Model_performance.xlsx
#   - output/ML_outputs/Feature_importance.xlsx
#   - output/ML_outputs/KMeans_metrics.xlsx
#   - ROC and clustering figures in output/ML_outputs/figures/
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
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    adjusted_rand_score
)

# --------------------------------------------------------------
# Configuration
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
# Utility functions
# --------------------------------------------------------------
def standardize(X: np.ndarray):
    """Standardize feature matrix (zero mean, unit variance)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


def kmeans(X, k, max_iter=100, tol=1e-4, n_init=10, random_state=None):
    """Custom K-Means implementation using Euclidean distance."""
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
    return best  # labels, centroid, n_iter, inertia


def silhouette_score_euclidean(X, labels):
    """Compute mean silhouette score for a given cluster assignment."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    clusters = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    s = np.zeros(n, dtype=float)

    for i in range(n):
        ci = labels[i]
        in_ci = clusters[ci]
        if len(in_ci) > 1:
            a = D[i, in_ci].sum() / (len(in_ci) - 1)
        else:
            a = 0.0
        b_candidates = [D[i, idxs].mean() for cj, idxs in clusters.items() if cj != ci and len(idxs) > 0]
        b = min(b_candidates) if b_candidates else 0.0
        denom = max(a, b)
        s[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(s.mean())


# --------------------------------------------------------------
# Load data (same structure as Task 4)
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
    pass  # Clinical sheet is optional

# Prepare matrix (samples × features)
id_col = expr_df.columns[0]
expr_df[id_col] = expr_df[id_col].astype(str)
expr = expr_df.set_index(id_col).T
expr.index = expr.index.astype(str)

# Generate binary labels: LUAD = 1, HC = 0
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

# Impute missing values and standardize
X = expr.drop(columns=["label"]).astype(float)
X = X.fillna(X.median(axis=0))
y = expr["label"].values

Xs, mu, sigma = standardize(X.values)
print(f"✅ Samples: {Xs.shape[0]} | Features: {Xs.shape[1]} | LUAD={int((y==1).sum())} | HC={int((y==0).sum())}")

# --------------------------------------------------------------
# Supervised learning: Logistic Regression, MLP, SVM
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

models = {
    "LogisticRegression_L1": LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=3000, random_state=RANDOM_STATE
    ),
    "MLP_100": MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=1e-3,
        max_iter=400, random_state=RANDOM_STATE
    ),
    "SVM_RBF": SVC(
        kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=RANDOM_STATE
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
roc_data = {}

for name, model in models.items():
    print(f"\n🔹 Training model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
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

    print(f"   Test Accuracy: {acc:.3f} | Test ROC-AUC: {auc_val:.3f}")
    print(f"   CV Accuracy:   {cv_acc:.3f} | CV ROC-AUC: {cv_auc:.3f}")

    results.append({
        "Model": name,
        "Test_Accuracy": acc,
        "Test_ROC_AUC": auc_val,
        "CV_Accuracy": cv_acc,
        "CV_ROC_AUC": cv_auc
    })

    # ROC curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

# Save supervised metrics
metrics_df = pd.DataFrame(results).sort_values("Test_ROC_AUC", ascending=False)
with pd.ExcelWriter(OUT_METRICS, engine="openpyxl") as xlw:
    metrics_df.to_excel(xlw, sheet_name="Supervised", index=False)
print(f"📊 Saved metrics to: {OUT_METRICS.resolve()}")

# Feature importance (LogReg L1)
if "LogisticRegression_L1" in models:
    lr = models["LogisticRegression_L1"]
    coefs = lr.coef_[0]
    abs_coefs = np.abs(coefs)
    top_idx = np.argsort(-abs_coefs)[:50]
    top_feats = X.columns[top_idx]
    top_vals = abs_coefs[top_idx]
    feat_imp = pd.DataFrame({
        "Feature": top_feats,
        "Weight_abs": top_vals,
        "Model": "LogisticRegression_L1"
    })
    with pd.ExcelWriter(OUT_FEATURES, engine="openpyxl") as xlw:
        feat_imp.to_excel(xlw, sheet_name="Top50_LogReg_L1", index=False)
    print(f"📈 Saved feature importance to: {OUT_FEATURES.resolve()}")

# ROC plots
for name, (fpr, tpr, aucv) in roc_data.items():
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ROC_{name}.png", dpi=300)
    plt.close()

# --------------------------------------------------------------
# Unsupervised learning: Custom K-Means
# --------------------------------------------------------------
ks = [2, 3, 4, 5]
inertias, silhouettes, aris = [], [], []

for k in ks:
    labels, centroids, n_iter, inertia = kmeans(Xs, k=k, n_init=20, random_state=RANDOM_STATE)
    sil = silhouette_score_euclidean(Xs, labels)
    ari = adjusted_rand_score(y, labels)
    inertias.append(inertia)
    silhouettes.append(sil)
    aris.append(ari)
    print(f"[KMeans] k={k} | iterations={n_iter:2d} | inertia={inertia:.4f} | silhouette={sil:.3f} | ARI={ari:.3f}")

kmeans_df = pd.DataFrame({"k": ks, "Inertia_per_point": inertias, "Silhouette": silhouettes, "ARI_vs_labels": aris})
with pd.ExcelWriter(OUT_KMEANS, engine="openpyxl") as xlw:
    kmeans_df.to_excel(xlw, sheet_name="KMeans", index=False)
print(f"🧩 Saved clustering metrics to: {OUT_KMEANS.resolve()}")

# Elbow plot
plt.figure(figsize=(5, 4))
plt.plot(ks, inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia (per point)")
plt.title("Elbow Curve (LUAD vs HC)")
plt.tight_layout()
plt.savefig(FIG_DIR / "Elbow.png", dpi=300)
plt.close()

# Silhouette vs. k
plt.figure(figsize=(5, 4))
plt.plot(ks, silhouettes, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette vs. k")
plt.tight_layout()
plt.savefig(FIG_DIR / "Silhouette_vs_k.png", dpi=300)
plt.close()

print("\n✅ Pipeline completed successfully.")
print(f"   - Metrics: {OUT_METRICS.resolve()}")
print(f"   - Features: {OUT_FEATURES.resolve()}")
print(f"   - K-Means metrics: {OUT_KMEANS.resolve()}")
print(f"   - Figures saved in: {FIG_DIR.resolve()}")
