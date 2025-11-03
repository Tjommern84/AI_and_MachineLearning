# Group6_LungCancer_PyCharm.py
# --------------------------------------------------------------
# HOVEDFIL – kjør hele pipeline for Group 6 – Lungekreftanalyse
#
# Rekkefølge:
#   1) Task 1–4  → Biologiske analyser
#   2) Task 5    → ML-pipeline (LogReg, MLP, K-Means)
#   3) Compare   → Sammenligner ML- og biologiske funn
#   4) Plot      → Lager biologiske og sammenlignende figurer
#
# Input:   ./data/
# Output:  ./output/
# Scripts: ./scripts/
#
# Kjør fra prosjektroten:
#   ▶ python Group6_LungCancer_PyCharm.py
# --------------------------------------------------------------

import subprocess
from pathlib import Path

# Opprett hovedmapper for utdata og plott
for folder in [
    Path("output/Task1_outputs"),
    Path("output/Task2_outputs"),
    Path("output/Task3_outputs"),
    Path("output/Task4_miRNA_outputs"),
    Path("output/ML_outputs/figures"),
    Path("plots/overview"),
    Path("plots/comparative")
]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
#  Filrekkefølge for kjøring
# --------------------------------------------------------------
SCRIPTS_DIR = Path("scripts")

SCRIPTS = [
    "Group6_LC_Task1.py",          # Task 1 – filtrering og overlapp
    "Group6_LC_Task2.py",          # Task 2 – pathway-analyse
    "Group6_LC_Task3.py",          # Task 3 – immune-signaturer
    "Group6_LC_Task4.py",          # Task 4 – LUAD vs HC differensialanalyse
    "Group6_LC_ML_Pipeline.py",    # Task 5 – maskinlæringspipeline
    "Compare_Biology.py",          # Biologisk ↔ ML-overlapp
    "plot_results_tasks.py",       # Plott fra Task 1–4
    "compare_visuals_biology_ml.py" # Visuell sammenligning Biologi ↔ ML
]

# --------------------------------------------------------------
#  Kjør pipeline
# --------------------------------------------------------------
print("\n🚀 Starter komplett pipeline for Group 6 – Lungekreft og AI\n")

for script in SCRIPTS:
    path = SCRIPTS_DIR / script
    if not path.exists():
        print(f"⚠️  Hopper over {path.name} – filen finnes ikke.")
        continue

    print(f"\n🔹 Kjører: {path.name}")
    result = subprocess.run(["python", str(path)], capture_output=True, text=True,
                            encoding="utf-8",check=True, timeout=1800) # 30 minutes
    print(result.stdout)

    if result.returncode != 0:
        print(f"❌ Feil under kjøring av {path.name}:\n{result.stderr}")
        print("⛔ Pipeline stoppet.\n")
        break

print("\n✅ Hele pipeline er ferdig!\n")
print("📂 Resultater finnes i: ./output/")
print("🎨 Plottene er lagret i: ./plots/")
print("📊 Biologiske oversikter: ./plots/overview/")
print("📈 Sammenligninger ML ↔ Biologi: ./plots/comparative/")
print("🧾 Se README.md for detaljer om hvert steg.\n")