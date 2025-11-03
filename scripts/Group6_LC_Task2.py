# Group6_LC_Task2_v3.py
# ---------------------------------------------------------------
# Standalone Task 2: Pathway-level enrichment + Opposite Pathway Detection
# ---------------------------------------------------------------
# Runs enrichment (Enrichr ORA) for Exercise and NSCLC up/down gene lists,
# and detects "Opposite pathways" (up in Exercise, down in NSCLC and vice versa).
# ---------------------------------------------------------------

from pathlib import Path
import pandas as pd
import gseapy as gp

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = Path("output/Task1_outputs/Task1_results_standard.xlsx")
OUT_DIR = Path("output/Task2_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "Hallmark_2020"
]

CUTOFF = 0.05
TOP_N = 200

# -----------------------------
# Helper functions
# -----------------------------
def log(msg):
    print(msg, flush=True)

def get_gene_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    up = df[df["logfc"] > 0]["gene"].dropna().unique().tolist()
    down = df[df["logfc"] < 0]["gene"].dropna().unique().tolist()
    return {"up": up[:TOP_N], "down": down[:TOP_N]}

def run_enrichment(genes: list[str], label: str) -> pd.DataFrame:
    if len(genes) < 5:
        log(f"[skip] {label}: too few genes ({len(genes)})")
        return pd.DataFrame()
    all_results = []
    for lib in LIBRARIES:
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=lib,
                organism="Human",
                outdir=None,
                no_plot=True,
                cutoff=CUTOFF
            )
            df = enr.results
            df["Library"] = lib
            df["Group"] = label
            all_results.append(df)
            log(f"[ok] {label}: {lib} ({len(df)} pathways)")
        except Exception as e:
            log(f"[warn] {label} - {lib}: {e}")
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)

def find_opposites(df_ex_up, df_ca_down, df_ex_down, df_ca_up):
    """Find shared pathways with opposite direction (Exercise up vs NSCLC down, and vice versa)."""
    if df_ex_up.empty or df_ca_down.empty:
        ex_up_ca_down = pd.DataFrame()
    else:
        ex_up_ca_down = pd.merge(
            df_ex_up, df_ca_down,
            on="Term", suffixes=("_ex_up", "_ca_down")
        )
        ex_up_ca_down["Direction"] = "Exercise_up / NSCLC_down"

    if df_ex_down.empty or df_ca_up.empty:
        ex_down_ca_up = pd.DataFrame()
    else:
        ex_down_ca_up = pd.merge(
            df_ex_down, df_ca_up,
            on="Term", suffixes=("_ex_down", "_ca_up")
        )
        ex_down_ca_up["Direction"] = "Exercise_down / NSCLC_up"

    combined = pd.concat([ex_up_ca_down, ex_down_ca_up], ignore_index=True)
    if not combined.empty:
        combined = combined[
            ["Term", "Library_ex_up" if "Library_ex_up" in combined.columns else "Library_ex_down",
             "Adjusted P-value_ex_up" if "Adjusted P-value_ex_up" in combined.columns else "Adjusted P-value_ex_down",
             "Adjusted P-value_ca_down" if "Adjusted P-value_ca_down" in combined.columns else "Adjusted P-value_ca_up",
             "Direction"]
        ]
        combined = combined.rename(columns={
            "Library_ex_up": "Library",
            "Adjusted P-value_ex_up": "AdjP_Exercise",
            "Adjusted P-value_ex_down": "AdjP_Exercise",
            "Adjusted P-value_ca_down": "AdjP_NSCLC",
            "Adjusted P-value_ca_up": "AdjP_NSCLC"
        })
    return combined

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"📘 Reading Task 1 results: {INPUT_FILE.name}")
    ex_df = pd.read_excel(INPUT_FILE, sheet_name="Exercise_filtered_all")
    ca_df = pd.read_excel(INPUT_FILE, sheet_name="Cancer_filtered_all")

    ex_lists = get_gene_lists(ex_df)
    ca_lists = get_gene_lists(ca_df)

    results = {}
    results["Exercise_up"] = run_enrichment(ex_lists["up"], "Exercise_up")
    results["Exercise_down"] = run_enrichment(ex_lists["down"], "Exercise_down")
    results["NSCLC_up"] = run_enrichment(ca_lists["up"], "NSCLC_up")
    results["NSCLC_down"] = run_enrichment(ca_lists["down"], "NSCLC_down")

    # Opposite pathway detection
    opposites = find_opposites(
        results["Exercise_up"],
        results["NSCLC_down"],
        results["Exercise_down"],
        results["NSCLC_up"]
    )

    print(f"\n🔁 Opposite pathway pairs found: {len(opposites)}")

    # Combine summary
    all_combined = pd.concat([df for df in results.values() if not df.empty], ignore_index=True)
    summary = all_combined[all_combined["Adjusted P-value"] < CUTOFF].sort_values("Adjusted P-value")

    # Export all results
    out_path = OUT_DIR / "Task2_results_standard.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xlw:
        for key, df in results.items():
            if not df.empty:
                df.to_excel(xlw, sheet_name=key, index=False)
        opposites.to_excel(xlw, sheet_name="Opposite_pathways", index=False)
        summary.to_excel(xlw, sheet_name="Summary_all", index=False)

    print(f"✅ Done. Results saved to: {out_path.resolve()}")
