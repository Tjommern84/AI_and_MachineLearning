from sklearn.linear_model import Perceptron
import numpy as np


# --- Task 3.0: Data (OR problem: output is 1 except when both inputs are 0)
X_or = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_or = np.array([0,1,1,1])


# --- Task 3.1: Train Perceptron ---
clf_perc = Perceptron()
clf_perc.fit(X_or, y_or)

# --- Task 3.2: Predict ---
print("Predictions with default params:")
for x in X_or:
    print(f"Input {x} -> Predicted {clf_perc.predict([x])[0]}")

# --- Task 3.3: Questions ---
# Q1: Change max_iter and/or eta0 and re-train. Do results change?
#når datasettet er så lite skal det mye til før det blir forandring i svar.
#men i større datasett kan endringer i max_iter og eta pårvirke resultatet.

clf_perc2 = Perceptron(max_iter=20, eta0=0.5, tol=1e-3)  # annen konfig
clf_perc2.fit(X_or, y_or)

print("\nPredictions with max_iter=20, eta0=0.5:")
for x in X_or:
    print(f"Input {x} -> Predicted {clf_perc2.predict([x])[0]}")
