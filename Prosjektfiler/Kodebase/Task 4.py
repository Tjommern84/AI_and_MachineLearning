from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import NearestNeighbors

# --- Task 4.1: Generate synthetic single variable data ---
np.random.seed(42)  # for reproduserbarhet
X = np.linspace(0, 10, 40).reshape(-1, 1)  # 40 samples fra 0 til 10
noise = np.random.normal(0, 1, X.shape[0]) # støy ~ N(0,1)
y = 2.5 * X.flatten() + noise              # lineær sammenheng + støy

# --- Task 4.2: Fit model ---
model = LinearRegression()
model.fit(X, y)

# --- Task 4.3: Metrics ---
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")

# --- Task 4.4: Plot data points and regression line ---
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="blue", label="Data (med støy)")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regresjonslinje")
plt.title("Linear Regression (syntetisk data)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Task 4.5: Questions ---
# Q1: Change the slope in data generation to 3.0 and re-run. What happens to R^2?
# Dataene følger fortsatt en rett linje og modelle finner linjen.


# Q2: Try adding larger noise (e.g., std=3.0). How do MAE/MSE change?
# Punktene sprer seg mer rundt linjen, MAE og MSE blir større, fordi modellen bommer mer?
# R² synker fordi modellen ikke klarer å forklare variasjonen.