# Compare_Biology_ML.py
# --------------------------------------------------------------
# Leser resultater fra Task 1 (biologi) og Task 5 (ML)
# og lager en kvantitativ sammenligning.
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd

# Paths
T1_FILE = Path("output/Task1_outputs/Task1_results_standard.xlsx")
FEAT_FILE = Path("output/ML_outputs/Feature_importance.xlsx")
OUT_FILE = Path("output/ML_outputs/Biology_ML_comparison.xlsx")

print(f"📘 Leser {T1_FILE.name} og {FEAT_FILE.name}")

# --- Les data ---
task1 = pd.read_excel(T1_FILE, sheet_name="Opp_EXup_CAdown")
ml_feats = pd.read_excel(FEAT_FILE)

task1_genes = set(task1["gene"].astype(str).str.upper())
ml_genes = set(ml_feats["Feature"].astype(str).str.upper())

# --- Beregn overlapp ---
overlap = sorted(task1_genes & ml_genes)
pct_overlap = len(overlap) / len(ml_genes) * 100 if len(ml_genes) > 0 else 0

summary = pd.DataFrame({
    "Total_ML_Features": [len(ml_genes)],
    "Overlap_with_Task1": [len(overlap)],
    "Percent_overlap": [pct_overlap],
})

# --- Lagre ---
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as xlw:
    summary.to_excel(xlw, sheet_name="Summary", index=False)
    pd.DataFrame(overlap, columns=["Overlapping_Genes"]).to_excel(xlw, sheet_name="Overlapping_Genes", index=False)

print(f"✅ Sammenligning ferdig. {len(overlap)} overlappende gener funnet ({pct_overlap:.1f}%).")
print(f"📄 Resultat lagret til: {OUT_FILE.resolve()}")
