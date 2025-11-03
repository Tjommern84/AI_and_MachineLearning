# Group6_LC_Task3_v1.py
# --------------------------------------------------------------
# Task 3 — Immune composition / marker analysis (standard)
# --------------------------------------------------------------
# Reads Task2_results_standard.xlsx, extracts immune-related
# pathways, and summarizes them by group.
# --------------------------------------------------------------

from pathlib import Path
import pandas as pd
import re

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = Path("output/Task2_outputs/Task2_results_standard.xlsx")
OUT_DIR = Path("output/Task3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "ImmunePathway_summary.xlsx"

# Keywords used to detect immune-related pathways
IMMUNE_KEYWORDS = [
    "immune", "immunity", "t cell", "b cell", "nk cell",
    "macrophage", "monocyte", "leukocyte", "neutrophil",
    "antigen", "mhc", "cytokine", "chemokine", "interferon",
    "interleukin", "complement", "inflammatory", "phagocyt",
    "adaptive", "innate", "lymphocyte"
]

def is_immune(term: str) -> bool:
    """Return True if term contains immune-related words."""
    term_lower = str(term).lower()
    return any(re.search(k, term_lower) for k in IMMUNE_KEYWORDS)

# -----------------------------
# Load data
# -----------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

xls = pd.ExcelFile(INPUT_FILE)
sheets = [s for s in xls.sheet_names if s not in ["Summary_all"]]
results = {s: pd.read_excel(INPUT_FILE, sheet_name=s) for s in sheets}

# -----------------------------
# Extract immune-related pathways
# -----------------------------
immune_results = {}
for name, df in results.items():
    if df.empty or "Term" not in df.columns:
        continue

    # Merk hvilke pathways som er immune-relaterte
    df["is_immune"] = df["Term"].apply(is_immune)
    df_immune = df[df["is_immune"]].copy()

    if not df_immune.empty:
        # Finn kolonne for adjusted p-value (robust søk)
        adj_cols = [c for c in df_immune.columns if "adj" in c.lower() and "p" in c.lower()]
        if adj_cols:
            adj_col = adj_cols[0]
            df_immune = df_immune.sort_values(adj_col).reset_index(drop=True)
        else:
            print(f"[warn] No adjusted p-value column found in {name}")
    else:
        print(f"[warn] No immune pathways found in {name}")

    immune_results[name] = df_immune
    print(f"[{name}] immune pathways: {len(df_immune)} / {len(df)}")

# -----------------------------
# Summary statistics
# -----------------------------
summary_rows = []
for name, df in immune_results.items():
    if df.empty:
        continue

    # Finn kolonne for adjusted p-value (robust søk)
    adj_cols = [c for c in df.columns if "adj" in c.lower() and "p" in c.lower()]
    if adj_cols:
        adj_col = adj_cols[0]
    else:
        print(f"[warn] No adjusted p-value column found in {name}, skipping.")
        continue

    # Sorter og velg topp 10
    df = df.sort_values(adj_col)
    top10 = df.head(10)[["Term", adj_col, "Library"]] if "Library" in df.columns else df.head(10)[["Term", adj_col]]
    mean_p = df[adj_col].mean()

    summary_rows.append({
        "Group": name,
        "Immune_pathways": len(df),
        "Top_terms": "; ".join(top10["Term"].head(5)),
        "Mean_adjP": mean_p
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Immune_pathways", ascending=False)


# -----------------------------
# Save outputs
# -----------------------------
with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as xlw:
    for name, df in immune_results.items():
        df.to_excel(xlw, sheet_name=name, index=False)
    summary_df.to_excel(xlw, sheet_name="Summary", index=False)

print(f"\n✅ Task 3 completed. Results saved to {OUT_FILE.resolve()}")
