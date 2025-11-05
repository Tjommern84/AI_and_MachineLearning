from stringprep import b1_set
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
        X= np.asarray(X,dtype=float)
        y= np.asarray(row[4])
        return X, y

def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0,ddof=0)
    sigma[sigma == 0] = 1
    Xs =(X-mu)/sigma
    return Xs, mu, sigma


if __name__=="__main__":
    X, y = load_iris("iris.csv")
    X_std, mu, sigma = standardize(X)


    def kmeans(X, k, max_iter=100, tol=1e-4, n_init=1, random_state=None):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if not (1 <= k <= n):
            raise ValueError('k must be between 1 and n (number of observations')
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


    labels, centoid, n_iter, inertia = kmeans(X_std, k=3, n_init=10, random_state=0)

    print("labels:", labels)
    print("centoid:\n", centoid)
    print("iterasjoner:", n_iter)
    print("inertia:", inertia)

    print(f"[Iris] k=3 | iterasjoner: {n_iter}")
    print(f"Inertia (per punkt): {inertia:.4f}")

    counts = np.bincount(labels,minlength = 3)
    print("klynge-størrelser:", counts.tolist())
    