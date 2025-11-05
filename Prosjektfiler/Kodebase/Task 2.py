# ---- (Run this first) ----
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn import datasets
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

# --- Task 2.0 Use the provided small dataset below for the following tasks:
X = np.array([[1, 10],
              [2, 20],
              [3, 30],
              [4, 40],
              [5, 50]], dtype=float)
df = pd.DataFrame(X, columns=["feat1","feat2"])
print("Original data:")
print(df)

# --- Task 2.1 StandardScaler and MinMaxScaler:
#StandardScaler
scaler_std = preprocessing.StandardScaler()
data_scaled_std = scaler_std.fit_transform(df)
df_std = pd.DataFrame(data_scaled_std, columns=df.columns)
print("Scaled data:")
print(df_std)

#MinMaxScaler
scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = scaler_minmax.fit_transform(df)
df_mm = pd.DataFrame(data_scaled_minmax, columns=df.columns)
print("MinMax scaled data:")
print(df_mm)

# --- Task 2.2: Normalizer (row-wise to unit norm):
input_data = df.values

data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')

df_l1 = pd.DataFrame(data_normalized_l1, columns=df.columns)
df_l2 = pd.DataFrame(data_normalized_l2, columns=df.columns)

print("Normalized L1:")
print(data_normalized_l1)
print("Normalized L2:")
print(data_normalized_l2)

# --- Task 2.3: Label Encoding

input_labels = ['red', 'green', 'blue', 'yellow', 'purple', 'pink', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#Vis mapping
print("Label Mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, "--> ",i)

#Encode en test liste
test_labels = ['green', 'red', 'white']
encoded_values = encoder.transform(test_labels)
print("Test Values: ", test_labels)
print("Encoded Values: ", encoded_values)

#Decode en liste tilbake
decoded_list = encoder.inverse_transform([3, 0, 4, 1])
print("Encoded Values: [3, 0, 4, 1]")
print("Decoded Values: ",(decoded_list))


# --- Task 2.4: Questions (answer **in text** as commments below each question)
# Q1: When would you prefer StandardScaler vs MinMaxScaler vs Normalizer? Why? Explain briefly.

#Standardscaler fjerner gjennomsnittet og skalerer hver feature til standardaavik = 1.
#Brukes i regresjon og nevrale nett.

#MinMaxScaler skalerer alle verdier til verdier mellom 0 og 1.
#brukes i k-Nearest Neibourgh f,eks.

#Normalizer skalerer hver rad slik at normen (lengden) blir 1.
# Vi bruker den når vi ser på hvordan verdiene fordeler seg relativt i vektoren, ikke hvor stor vektoren er.
# (når hver rad er en vektor, og man kun bryr seg om retningen)

# Q2: Create a small histogram for one feature before and after scaling. Any differences you notice?
plt.figure(figsize=(10,5))
plt.hist(df["feat2"], bins=5, alpha=0.5, label="Original")
plt.hist(df_std["feat2"], bins=5, alpha=0.5, label="StandardScaler")
plt.hist(df_mm["feat2"], bins=5, alpha=0.5, label="MinMaxScaler")
#plt.hist(df_norm["feat2"], bins=5, alpha=0.5, label="Normalizer")
plt.title("Histogram: feat2 før og etter skalering")
plt.xlabel("Verdi")
plt.ylabel("Antall")
plt.legend()
plt.show()
