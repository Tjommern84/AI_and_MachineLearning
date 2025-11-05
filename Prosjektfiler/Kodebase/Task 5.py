import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets

# --- Task 5.1: Load Iris data (sepal length and width) ---
iris = datasets.load_iris()
X_iris = iris.data[:, :2]  # kun sepal length & sepal width

# --- Task 5.2: Fit NearestNeighbors ---
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_iris)

# --- Task 5.3: Mean distance to neighbors ---
distances, indices = nn.kneighbors(X_iris)
mean_distances = distances.mean(axis=1)

# --- Task 5.4: Threshold = top 10% ---
threshold = np.percentile(mean_distances, 90)  # øverste 10%
outliers = mean_distances > threshold

print("Antall outliers:", np.sum(outliers))

# --- Task 5.5: Plot ---
plt.figure(figsize=(8,6))
plt.scatter(X_iris[:,0], X_iris[:,1], c="blue", label="Normal points", alpha=0.6)
plt.scatter(X_iris[outliers,0], X_iris[outliers,1],
            c="red", marker="x", s=100, label="Outliers")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title(f"Outlier detection with k={k}, threshold=top 10%")
plt.legend()
plt.show()

# --- Questions ---
# 1: Change k to 2 and 10. How does the outlier set change?
#  -> Med lavt k (2) blir metoden mer følsom for små lokale avvik, og flere punkter kan se ut som outliers.
#     Med høyt k (10) glattes distansene ut, og bare de mest ekstreme punktene markeres.
#
# 2: Try a stricter threshold (e.g., top 5%). What happens?
#  -> Da blir færre punkter markert som outliers, kun de aller mest ekstreme beholdes.
