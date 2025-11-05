import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths (two levels up)
input_file = os.path.join("..", "..", "output", "ML_outputs", "Model_performance.xlsx")
out_dir = os.path.join("..", "..", "plots", "ml_pipeline")
os.makedirs(out_dir, exist_ok=True)

# Read Excel
df = pd.read_excel(input_file)

# Prepare data
melted = df.melt(id_vars="Model", value_vars=["Test_ROC_AUC", "CV_ROC_AUC"],
                 var_name="Metric", value_name="AUC")

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=melted, x="Model", y="AUC", hue="Metric", palette="Blues_d")
plt.ylim(0, 1)
plt.ylabel("ROC-AUC")
plt.title("Model Comparison – Test vs CV ROC-AUC")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Model_ROC_AUC_comparison.png"), dpi=300)
plt.close()
