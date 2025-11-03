# Group6_LC_Task1_v2.py
# Standalone Task 1: Filter DEGs, compute overlap/opposite sets (Exercise vs NSCLC),
# and export 3 Excel files (less_strict / standard / more_strict) with multiple sheets.
# No plotting. Run from project root.

from pathlib import Path
import pandas as pd

# -----------------------------
# Config: file and sheet names
# -----------------------------
DATA_PATH = Path("data/PBMC_exercise_lung_cancer_datasets.xlsx")

EXERCISE_SHEETS = [
    "GSE111552.top.table",
    "GSE14642.top.table",
    "GSE11761.top.table",
]

CANCER_SHEETS = [
    "GSE13255.top.table",
    "GSE39345.top.table (5)",
]

# p-value threshold
ADJ_PVAL = 0.05

# three log2FC thresholds (abs)
RUNS = [
    ("less_strict", 0.58),   # ~1.5x
    ("standard",   1.00),    # ~2x
    ("more_strict",1.50),    # ~3x
]

OUT_DIR = Path("output/Task1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, underscores, remove dots."""
    x = df.copy()
    x.columns = (
        x.columns.astype(str)
        .str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(".", "_", regex=False)
    )
    return x

def get_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return first column name present in df among candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing any of {candidates}. Available: {list(df.columns)[:10]} ...")

def to_symbol_list(s) -> list[str]:
    """
    Extract gene symbols from gene info fields.
    - Split '///' to multiple items; if '//' exists, take middle (often symbol).
    - Return uppercase alphanumeric tokens.
    """
    if pd.isna(s):
        return []
    out = []
    for chunk in str(s).split("///"):
        chunk = chunk.strip()
        if "//" in chunk:
            parts = [p.strip() for p in chunk.split("//")]
            sym = (parts[1] if len(parts) >= 2 else parts[0]).split()[0]
        else:
            sym = chunk.split()[0]
        sym = "".join(ch for ch in sym if ch.isalnum()).upper()
        if sym:
            out.append(sym)
    return out

def explode_symbols(df: pd.DataFrame, info_col: str, log_col: str) -> pd.DataFrame:
    """Create one row per gene symbol; return ['gene','logfc','source']."""
    tmp = df[[info_col, log_col]].copy()
    tmp["symbols"] = tmp[info_col].apply(to_symbol_list)
    tmp = tmp.explode("symbols")
    tmp = tmp[tmp["symbols"].notna() & (tmp["symbols"] != "")]
    tmp = tmp.rename(columns={"symbols": "gene", log_col: "logfc"})
    return tmp[["gene", "logfc"]]

def process_sheet(path: Path, sheet: str, p_cut: float, abs_logfc_cut: float) -> pd.DataFrame:
    """
    Read ONE sheet, standardize cols, filter by adj.p and |logFC|,
    extract symbols and explode rows. Returns ['gene','logfc','source'].
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    df = std_cols(raw)

    # adj p-value column candidates (after std_cols)
    adj_cands = [
        "adj_p_val", "adj_p_value", "adj_p", "p_adj",
        "p_value_adjusted", "pvalue_adj", "adj_pval", "adj_p_val_",
        "adj_p_val_bh", "adj_pvalue",
        "adj_p_val_bonferroni", "adj_pval_bh"
    ]
    # logFC column candidates
    log_cands = [
        "log2fc", "log_fc", "logfc", "log2_fold_change", "log2_foldchange", "log2_fold",
        "log2_fold_change_", "log2_foldchange_", "log_fold_change", "logfoldchange"
    ]
    # gene info column candidates
    geneinfo_cands = [
        "gene_symbol", "gene_symbols", "symbol", "symbols",
        "gene_assignment", "gene_title", "gene", "gene_name", "geneid", "gene_id"
    ]

    try:
        adj = get_col(df, adj_cands)
    except KeyError:
        # Fallback: sometimes 'adj.p.val' stays as 'adj_p_val' already handled.
        # As last resort, accept raw 'p_value' if no adj found (not ideal, but prevents hard fail)
        try:
            adj = get_col(df, ["p_value", "pvalue", "p_val"])
            log(f"  [warn] No adjusted p-value column found in '{sheet}', using raw p-value '{adj}'.")
        except KeyError as e:
            raise e

    logc = get_col(df, log_cands)
    ginf = get_col(df, geneinfo_cands)

    # Filter
    filt = df[(df[adj] < p_cut) & (df[logc].abs() > abs_logfc_cut)].copy()
    cleaned = explode_symbols(filt, ginf, logc)
    cleaned["source"] = sheet
    return cleaned

def mean_logfc_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("gene")["logfc"]
              .mean()
              .rename("mean_logfc")
              .reset_index())

def make_overlap_tables(ex_all: pd.DataFrame, ca_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build four overlap tables with prioritization columns:
      - Opp_EXup_CAdown
      - Opp_EXdown_CAup
      - Shared_UP
      - Shared_DOWN
    Each includes mean_logFC_ex and mean_logFC_ca, sorted by biologically relevant order.
    """
    ex_up   = set(ex_all.loc[ex_all["logfc"] > 0, "gene"])
    ex_down = set(ex_all.loc[ex_all["logfc"] < 0, "gene"])
    ca_up   = set(ca_all.loc[ca_all["logfc"] > 0, "gene"])
    ca_down = set(ca_all.loc[ca_all["logfc"] < 0, "gene"])

    opp_exup_cadown = sorted(ex_up & ca_down)
    opp_exdown_caup = sorted(ex_down & ca_up)
    shared_up       = sorted(ex_up & ca_up)
    shared_down     = sorted(ex_down & ca_down)

    ex_mean = mean_logfc_per_gene(ex_all).rename(columns={"mean_logfc":"mean_logfc_ex"})
    ca_mean = mean_logfc_per_gene(ca_all).rename(columns={"mean_logfc":"mean_logfc_ca"})

    def as_table(glist, label):
        t = pd.DataFrame({"gene": glist})
        t = t.merge(ex_mean, on="gene", how="left").merge(ca_mean, on="gene", how="left")
        t["label"] = label
        return t

    tab_opp_exup_cadown = as_table(opp_exup_cadown, "EX_up ∩ CA_down")\
        .sort_values(["mean_logfc_ex","mean_logfc_ca"], ascending=[False, True], na_position="last")
    tab_opp_exdown_caup = as_table(opp_exdown_caup, "EX_down ∩ CA_up")\
        .sort_values(["mean_logfc_ex","mean_logfc_ca"], ascending=[True, False], na_position="last")
    tab_shared_up       = as_table(shared_up, "EX_up ∩ CA_up")\
        .sort_values(["mean_logfc_ex","mean_logfc_ca"], ascending=[False, False], na_position="last")
    tab_shared_down     = as_table(shared_down, "EX_down ∩ CA_down")\
        .sort_values(["mean_logfc_ex","mean_logfc_ca"], ascending=[True, True], na_position="last")

    return {
        "Opp_EXup_CAdown": tab_opp_exup_cadown.reset_index(drop=True),
        "Opp_EXdown_CAup": tab_opp_exdown_caup.reset_index(drop=True),
        "Shared_UP":       tab_shared_up.reset_index(drop=True),
        "Shared_DOWN":     tab_shared_down.reset_index(drop=True),
    }

def run_once(tag: str, abs_logfc_cut: float) -> None:
    """Run the full Task 1 once for a given log2FC cutoff and save one Excel file."""
    log(f"\n=== Task 1 run: {tag} | adj.p < {ADJ_PVAL} & |log2FC| > {abs_logfc_cut} ===")

    # Process Exercise sheets
    ex_frames = []
    for s in EXERCISE_SHEETS:
        try:
            df = process_sheet(DATA_PATH, s, ADJ_PVAL, abs_logfc_cut)
            ex_frames.append(df)
            log(f"  [OK] Exercise {s}: {len(df)} rows after filtering/explode.")
        except Exception as e:
            log(f"  [skip] Exercise {s}: {e}")

    # Process Cancer sheets
    ca_frames = []
    for s in CANCER_SHEETS:
        try:
            df = process_sheet(DATA_PATH, s, ADJ_PVAL, abs_logfc_cut)
            ca_frames.append(df)
            log(f"  [OK] Cancer   {s}: {len(df)} rows after filtering/explode.")
        except Exception as e:
            log(f"  [skip] Cancer   {s}: {e}")

    if not ex_frames or not ca_frames:
        log("  [STOP] No data after filtering for either Exercise or Cancer — skipping export.")
        return

    ex_all = pd.concat(ex_frames, ignore_index=True)
    ca_all = pd.concat(ca_frames, ignore_index=True)

    log(f"  -> Exercise total rows: {len(ex_all)}")
    log(f"  -> Cancer   total rows: {len(ca_all)}")

    # Overlap tables
    tabs = make_overlap_tables(ex_all, ca_all)

    # Export to Excel with multiple sheets
    out_path = OUT_DIR / f"Task1_results_{tag}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xlw:
        ex_all.to_excel(xlw, sheet_name="Exercise_filtered_all", index=False)
        ca_all.to_excel(xlw, sheet_name="Cancer_filtered_all", index=False)
        tabs["Opp_EXup_CAdown"].to_excel(xlw, sheet_name="Opp_EXup_CAdown", index=False)
        tabs["Opp_EXdown_CAup"].to_excel(xlw, sheet_name="Opp_EXdown_CAup", index=False)
        tabs["Shared_UP"].to_excel(xlw,       sheet_name="Shared_UP",       index=False)
        tabs["Shared_DOWN"].to_excel(xlw,     sheet_name="Shared_DOWN",     index=False)

    log(f"  [SAVE] {out_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    for tag, cut in RUNS:
        run_once(tag, cut)

    print("\n✅ Done. All outputs written to:", OUT_DIR.resolve())
