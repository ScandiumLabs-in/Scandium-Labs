#!/usr/bin/env python3
"""Phase 2 E: Compile all Phase 2 results into a publication-quality ablation table."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import numpy as np

REPORTS = Path("experiments/reports")
OUT_DIR = Path("experiments/reports/phase2_e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("  PHASE 2 — PUBLICATION-QUALITY ABLATION TABLE")
print("=" * 100)

# ── 1. Dataset Characterization (A) ──────────────────────────────────
print("\n" + "─" * 100)
print("  TABLE 1: DATASET CHARACTERIZATION")
print("─" * 100)

# Distribution stats — loaded from A.1 console output, hardcoded summary
targets_info = [
    ("Formation Energy", "eV/atom", "2925/355/355", "-0.99/-0.92/-1.02", "0.79/0.85/0.69", "0.00/0.00/0.00", "4.77", "⚠ SHIFT", "Architecture"),
    ("Energy Above Hull", "eV/atom", "2925/355/355", "0.18/0.21/0.17", "0.35/0.42/0.30", "0.20/0.21/0.21", "4.71", "OK", "Target"),
    ("Band Gap", "eV", "2925/355/355", "1.31/1.32/1.22", "1.31/1.38/1.35", "0.29/0.24/0.33", "5.59", "OK", "Data"),
    ("Log σ", "log(S/cm)", "0/0/0", "-/-/-", "-/-/-", "-", "-", "N/A", "No Labels"),
    ("Ea", "eV", "0/0/0", "-/-/-", "-/-/-", "-", "-", "N/A", "No Labels"),
]

print(f"{'Target':<20} {'Unit':<12} {'n(train/val/test)':<20} {'Mean':<22} {'Std':<18} {'Zero Frac':<12} {'Max':<10} {'Shift':<12} {'Limitation'}")
print(f"{'-'*120}")
for r in targets_info:
    print(f"{r[0]:<20} {r[1]:<12} {r[2]:<20} {r[3]:<22} {r[4]:<18} {r[5]:<12} {r[6]:<10} {r[7]:<12} {r[8]}")

print(f"\n  Correlation (Pearson): Ef↔Eah=0.60, Ef↔BG=-0.44, Eah↔BG=-0.32")
print(f"  Correlation (Spearman): Ef↔Eah=0.47, Ef↔BG=-0.44, Eah↔BG=-0.43")
print(f"  Benchmark overlap: 0/13 benchmark materials found in training set (no contamination)")

# ── 2. Embedding Analysis (B) ───────────────────────────────────────
print("\n" + "─" * 100)
print("  TABLE 2: EMBEDDING QUALITY ANALYSIS")
print("─" * 100)

embedding_metrics = [
    ("Embedding dimension", "128"),
    ("PCA 2D variance explained", "75.1%"),
    ("PC1 variance", "40.7%"),
    ("PC2 variance", "34.4%"),
    ("PC1 vs Ef correlation", "-0.746"),
    ("PC1 vs Eah correlation", "-0.288"),
    ("PC1 vs BG correlation", "0.464"),
    ("Train↔Test centroid distance", "0.786 (moderate)"),
    ("Train mean norm", "13.98"),
    ("Test mean norm", "14.09"),
]

for name, val in embedding_metrics:
    print(f"  {name:<35}: {val}")

# ── 3. Uncertainty Calibration (C) ───────────────────────────────────
print("\n" + "─" * 100)
print("  TABLE 3: UNCERTAINTY CALIBRATION (raw predictions)")
print("─" * 100)

cal_data = [
    ("formation_energy", "0.4721", "1.576", "0.992", "0.867", "0.146"),
    ("energy_above_hull", "0.2853", "1.201", "0.983", "0.970", "0.232"),
    ("band_gap", "0.2717", "2.635", "0.690", "1.142", "0.287"),
]

print(f"{'Task':<22} {'ECE↓':<8} {'NLL↓':<8} {'Cover@95':<10} {'Opt T':<8} {'T-scaled ECE':<14}")
print(f"{'-'*70}")
for r in cal_data:
    print(f"{r[0]:<22} {r[1]:<8} {r[2]:<8} {r[3]:<10} {r[4]:<8} {r[5]:<14}")
print(f"\n  ⚠ All tasks have POOR calibration (ECE > 0.2). Temperature scaling improves NLL.")
print(f"  ⚠ Coverage@95 is well above 0.95 for Ef and Eah (overconfidence), below for BG.")

# ── 4. LOFO Results (D) ─────────────────────────────────────────────
print("\n" + "─" * 100)
print("  TABLE 4: LOFO PROXY — PER-FAMILY TEST PERFORMANCE")
print("─" * 100)

# Load from saved JSON
proxy_path = REPORTS / "phase2_d1" / "lofo_proxy_results.json"
if proxy_path.exists():
    with open(proxy_path) as f:
        proxy = json.load(f)
    
    print(f"{'Task':<22} {'Metric':<8} {'All':<12} {'Oxide':<12} {'Sulfide':<12} {'Halide':<12}")
    print(f"{'-'*78}")
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        # Overall
        overall_path = REPORTS / "phase2_d2" / "lofo_sulfide_results.json"
        if overall_path.exists():
            with open(overall_path) as f:
                overall = json.load(f)
            overall_res = overall.get('overall', {}).get(task, {})
            overall_mae = f"{overall_res.get('mae', 0):.4f}"
            overall_r2 = f"{overall_res.get('r2', 0):.4f}"
        else:
            overall_mae = overall_r2 = "—"
        
        fam_maes = []
        fam_r2s = []
        for fam in ["oxide", "sulfide", "halide"]:
            res = proxy.get(fam, {}).get(task, {})
            fam_maes.append(f"{res.get('mae', 0):.4f}" if res else "—")
            fam_r2s.append(f"{res.get('r2', 0):.4f}" if res else "—")
        
        print(f"{task:<22} {'MAE':<8} {overall_mae:<12} {fam_maes[0]:<12} {fam_maes[1]:<12} {fam_maes[2]:<12}")
        print(f"{'':<22} {'R²':<8} {overall_r2:<12} {fam_r2s[0]:<12} {fam_r2s[1]:<12} {fam_r2s[2]:<12}")
        print()

# ── 5. Full LOFO Results (D.2) ──────────────────────────────────────
print("─" * 100)
print("  TABLE 5: FULL LOFO — SULFIDE HELD OUT")
print("─" * 100)

lofo_path = REPORTS / "phase2_d2" / "lofo_sulfide_results.json"
if lofo_path.exists():
    with open(lofo_path) as f:
        lofo = json.load(f)
    
    print(f"{'Task':<22} {'Overall MAE':<14} {'Overall R²':<14} {'Sulfide MAE':<14} {'Sulfide R²':<14} {'Δ MAE':<10}")
    print(f"{'-'*88}")
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        ov = lofo.get('overall', {}).get(task, {})
        ho = lofo.get('holdout_family', {}).get(task, {})
        ov_mae = ov.get('mae', 0)
        ov_r2 = ov.get('r2', 0)
        ho_mae = ho.get('mae', 0)
        ho_r2 = ho.get('r2', 0)
        
        # Compare with proxy
        proxy_res = proxy.get('sulfide', {}).get(task, {})
        proxy_mae = proxy_res.get('mae', 0)
        delta = ho_mae - proxy_mae if ho_mae and proxy_mae else 0
        
        print(f"{task:<22} {ov_mae:<14.4f} {ov_r2:<14.4f} {ho_mae:<14.4f} {ho_r2:<14.4f} {delta:<+10.4f}")
else:
    print("  ⏳ Full LOFO retraining still in progress...")

# ── 6. Key Findings Summary ─────────────────────────────────────────
print("\n" + "═" * 100)
print("  KEY FINDINGS & RECOMMENDATIONS")
print("═" * 100)

findings = [
    ("1. Dataset", "Formation energy has significant train/test distribution shift (KS p<0.001). "
     "Conductivity and activation energy have 0% label coverage — unusable."),
    ("2. Model Capacity", "Embedding quality is reasonable (75% PCA variance), but the 2-layer ALIGNN "
     "architecture limits representational power for Ef."),
    ("3. Calibration", "All tasks show POOR uncertainty calibration (ECE > 0.2). "
     "Temperature scaling provides marginal improvement."),
    ("4. Generalization", "Sulfides are the hardest material family: Ef R²=0.40, Eah R²=0.16, BG R²=0.34. "
     "This is critical since sulfides are the most promising solid electrolyte class."),
    ("5. Benchmark", "13 benchmark materials have zero overlap with training set — benchmark is uncontaminated."),
    ("6. Per-Family Bias", "Ef: halide best (R²=0.69) > oxide (0.64) > sulfide (0.40). "
     "Eah: halide (0.55) > oxide (0.18) > sulfide (0.16). "
     "BG: oxide (0.47) > sulfide (0.34) > halide (0.004)."),
]

for title, desc in findings:
    print(f"\n  ◆ {title}: {title.lower()}")
    print(f"    {desc}")

# ── 7. Recommended Actions ──────────────────────────────────────────
print(f"\n{'─'*100}")
print("  RECOMMENDED NEXT ACTIONS (Priority Order)")
print(f"{'─'*100}")

actions = [
    ("P0 — Fix EAH", "EAH has 20% zeros, high skew (4.77). Switch to log-EAH or quantile-based targets. "
     "The v2_10000_log_eah dataset is already prepared."),
    ("P0 — Expand labels", "Obtain conductivity and activation energy labels. Currently 0% coverage "
     "makes these heads untrainable."),
    ("P1 — Architecture upgrade", "Increase hidden_dim (128→256), ALIGNN layers (2→4), "
     "add pretrained encoder. Current architecture is capacity-limited for Ef."),
    ("P1 — Address train/test shift", "Formation energy shows significant distribution shift. "
     "Use stratified splitting or importance weighting."),
    ("P2 — Sulfide-specific training", "Sulfide performance is consistently worst. "
     "Consider sulfide upsampling, data augmentation, or targeted data collection."),
    ("P2 — Temperature scaling", "Apply T=0.867 (Ef), 0.970 (Eah), 1.142 (BG) at inference time "
     "for better-calibrated uncertainties."),
]

for i, (title, desc) in enumerate(actions):
    print(f"\n  {title}")
    print(f"    {desc}")

# Save report
report_path = OUT_DIR / "ablation_table.txt"
with open(report_path, "w") as f:
    f.write("Scandium Labs — Phase 2 Ablation Report\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")
    f.write("=" * 80 + "\n\n")
    for finding in findings:
        f.write(f"◆ {finding[0]}: {finding[1]}\n\n")
    for action in actions:
        f.write(f"★ {action[0]}: {action[1]}\n\n")

print(f"\nReport saved to {report_path}")
