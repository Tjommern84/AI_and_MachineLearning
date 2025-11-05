import matplotlib.pyplot as plt
import csv
import numpy as np


def load_iris(path="iris.csv"):
    X, y = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            try:
                feats = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            except ValueError:
                continue
            X.append(feats)
            y.append(row[4])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        return X, y


def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


if __name__ == "__main__":
    X, y = load_iris("iris.csv")
    X_std, mu, sigma = standardize(X)


def kmeans(X, k, max_iter=100, tol=1e-4, n_init=1, random_state=None):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not (1 <= k <= n):
        raise ValueError('k must be between 1 and n (number of observations)')
    rng = np.random.default_rng(random_state)

    best = None
    best_inertia = np.inf

# Random k sentoid
    for _ in range(n_init):
        init_idx = rng.choice(n, size=k, replace=False)
        centoid = X[init_idx].copy()

        for i in range(1, max_iter + 1):
            dist = ((X[:, None, :] - centoid[None, :, :]) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)

            new_centoid = centoid.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    new_centoid[j] = members.mean(axis=0)
                else:
                    new_centoid[j] = X[rng.integers(0, n)]
# Stopping criterion
            shift = np.linalg.norm(new_centoid - centoid)
            centoid = new_centoid
            if shift < tol:
                break

        final_dist = ((X - centoid[labels]) ** 2).sum(axis=1)
        inertia = float(final_dist.sum()) / n

        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels, centoid, i, inertia)

        return best

def silhouette_score_euclidean(X,labels):
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
            a = D[i, in_ci].sum() / (len(in_ci) -1)
        else:
            a = 0.0

        b_candidates = [D[i, idxs].mean() for cj, idxs in clusters.items() if cj != ci]
        b = min(b_candidates) if b_candidates else 0.0
        denom = max(a, b)
        s[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(s.mean())

if __name__ == '__main__':
    X, y = load_iris("iris.csv")
    X_std, mu, sigma = standardize(X)

    ks = [2, 3, 4, 5]
    results = {}
    inertias = []
    silhouettes = []

    for k in ks:
        labels, centoid, n_iter, inertia = kmeans(X_std, k=k, n_init=20, random_state=0)
        sil = silhouette_score_euclidean(X_std, labels)
        results[k] = (labels, centoid, n_iter, inertia, sil)
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"k={k} | iters={n_iter:2d} | inertia(per pt)={inertia:.4f} | silhouette={sil:.3f}")

    # Elbow-plot
    plt.figure(figsize=(5, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (per punkt)")
    plt.title("Elbow (Iris, standardisert)")
    plt.tight_layout()

    # Silhouette vs k
    plt.figure(figsize=(5, 4))
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette vs k (Iris)")
    plt.tight_layout()

    # 2×2 klyngeplott på (petal length, petal width)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    feat_x, feat_y = 2, 3  # petal length / petal width
    for ax, k in zip(axes.ravel(), ks):
        labels, centoid, inertia, sil, n_iter = results[k]
        ax.scatter(X_std[:, feat_x], X_std[:, feat_y], c=labels, s=18, alpha=0.9)
        ax.scatter(centoid[:, feat_x], centoid[:, feat_y], marker="X", s=120, c="k")
        ax.set_xlabel("Petal length (std)")
        ax.set_ylabel("Petal width (std)")
        ax.set_title(f"k={k} | inertia={inertia:.3f} | sil={sil:.3f}")
    plt.tight_layout()
    plt.show()