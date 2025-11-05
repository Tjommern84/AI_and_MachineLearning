🧬 Group 6 – Lung Cancer Project (USN–Pécs COIL)
📄 Overview

This project investigates molecular relationships between exercise-induced PBMC gene expression and non-small cell lung cancer (NSCLC).
The pipeline integrates bioinformatics (Tasks 1–4) with machine learning (Task 5) to explore shared and opposing immune and metabolic signatures.

⚙️ How to Run

Run from the project root directory:

python Group6_LungCancer_PyCharm.py


All tasks execute automatically in sequence, generating results and plots in the output/ and plots/ folders.

📂 Project Structure
project_root/
│
├── data/                # Input Excel datasets (PBMC, LUAD)
├── scripts/             # All analysis scripts
│   ├── Group6_LC_Task1.py
│   ├── Group6_LC_Task2.py
│   ├── Group6_LC_Task3.py
│   ├── Group6_LC_Task4.py
│   ├── Group6_LC_Task4v1.py
│   ├── Group6_LC_ML_Pipeline.py
│   ├── Compare_Biology.py
│   ├── plot_results_tasks.py
│   ├── plot_result_ML.py
│   └── compare_visuals_biology_ml.py
│
├── output/              # Generated data tables and ML metrics
└── plots/               # Figures from all analyses

🧠 Script Summary
Script	Description
Group6_LC_Task1.py	Filters and overlaps differentially expressed genes (DEGs) across exercise and NSCLC datasets.
Group6_LC_Task2.py	Performs pathway enrichment (GO, Reactome, KEGG, Hallmark).
Group6_LC_Task3.py	Identifies immune-related pathways and summarizes immune signatures.
Group6_LC_Task4.py / Task4v1.py	Differential miRNA analysis in LUAD vs healthy controls.
Group6_LC_ML_Pipeline.py	Machine learning workflow (Logistic Regression, MLP, K-Means). Generates model metrics and feature importance tables.
Compare_Biology.py	Compares biological DEGs and pathways with ML-identified features.
plot_results_tasks.py	Creates summary plots for biological results (Tasks 1–4).
plot_result_ML.py	Plots ML performance metrics and confusion matrices.
compare_visuals_biology_ml.py	Generates combined visuals linking biological and ML findings (overlap, immune feature ratio, PCA).
Group6_LungCancer_PyCharm.py	Main pipeline runner. Executes all scripts in sequence and logs progress.
📊 Outputs

output/Task*_outputs/ → Filtered data, pathway tables, immune results.

output/ML_outputs/ → Model metrics, feature importances, comparison tables.

plots/overview/ and plots/comparative/ → Generated figures for all tasks.

🧩 Dependencies

Python ≥ 3.10
Packages: pandas, numpy, matplotlib, seaborn, scikit-learn, gseapy, openpyxl