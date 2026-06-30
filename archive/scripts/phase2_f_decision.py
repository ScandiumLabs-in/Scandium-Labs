#!/usr/bin/env python3
"""Phase 2 F: Decision framework — determine next training actions from Phase 2 results."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

REPORTS = Path("experiments/reports")
OUT_DIR = Path("experiments/reports/phase2_f")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("  PHASE 2 — DECISION FRAMEWORK: WHAT TO DO NEXT")
print("=" * 90)

# ── Load all results ────────────────────────────────────────────────
findings = {}

# A.1 Distribution analysis
findings['train_test_shift'] = {
    'formation_energy': {'ks_p': 7.87e-4, 'severity': 'HIGH', 'action': 'Stratified splitting or reweighting'},
    'energy_above_hull': {'ks_p': 0.137, 'severity': 'LOW', 'action': 'None needed'},
    'band_gap': {'ks_p': 0.058, 'severity': 'LOW', 'action': 'None needed'},
}

findings['label_coverage'] = {
    'formation_energy': {'coverage': 1.0, 'status': 'OK'},
    'energy_above_hull': {'coverage': 1.0, 'status': 'OK'},
    'band_gap': {'coverage': 1.0, 'status': 'OK'},
    'log_ionic_conductivity': {'coverage': 0.0, 'status': 'CRITICAL — no labels'},
    'activation_energy': {'coverage': 0.0, 'status': 'CRITICAL — no labels'},
}

findings['eah_properties'] = {
    'zero_fraction': 0.203,
    'skew': 4.77,
    'r2_score': -1.69,
    'verdict': 'Broken target — switch to log-EAH or robust regression'
}

# B. Embedding analysis
findings['embedding_quality'] = {
    'pca_2d_var': 0.751,
    'pc1_vs_ef': -0.746,
    'pc1_vs_eah': -0.288,
    'pc1_vs_bg': 0.464,
    'verdict': 'Embeddings capture Ef well, Eah/BG moderately'
}

# C. Calibration
findings['calibration'] = {
    'formation_energy': {'ece': 0.472, 'ece_level': 'POOR', 'needs_ts': True},
    'energy_above_hull': {'ece': 0.285, 'ece_level': 'POOR', 'needs_ts': True},
    'band_gap': {'ece': 0.272, 'ece_level': 'POOR', 'needs_ts': True},
}

# D. LOFO
findings['lofo'] = {
    'hardest_family': 'sulfide',
    'proxy_hardest_ef_r2': 0.400,
    'proxy_hardest_eah_r2': 0.161,
    'proxy_hardest_bg_r2': 0.341,
    'verdict': 'Sulfide performance is consistently worst. Need targeted data or architecture improvements.'
}

# ── Decision Tree ────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  DECISION TREE")
print(f"{'='*90}")

print("""
  Q1: Are conductivity/activation energy labels available?
  ├── NO  → P0: Acquire labels (0% coverage → unusable)
  └── YES → Continue

  Q2: Is EAH performance acceptable (R² > 0)?
  ├── NO  → P0: Switch to log-EAH or robust regression target
  │           (v2_10000_log_eah dataset is ready)
  └── YES → Continue

  Q3: Is Ef MAE improving with more data?
  ├── NO (R²=0.46, flat at 4.4× data) → P1: Architecture upgrade
  │   ├── Increase hidden_dim 128→256
  │   ├── Increase ALIGNN layers 2→4
  │   ├── Add pretrained encoder
  │   └── Consider deeper transformer (1→3 layers)
  └── YES → Scale data further

  Q4: Are predictions well-calibrated (ECE < 0.2)?
  ├── NO (ECE=0.27-0.47) → P2: Apply temperature scaling
  │   └── T_ef=0.867, T_eah=0.970, T_bg=1.142
  └── YES → Deploy

  Q5: Is sulfide performance acceptable?
  ├── NO (R²=0.40 Ef, 0.16 Eah, 0.34 BG) → P2: Sulfide improvements
  │   ├── Sulfide upsampling in training
  │   ├── Collect more sulfide data
  │   └── Sulfide-specific fine-tuning
  └── YES → Ready for production
""")

# ── Priority Action Matrix ──────────────────────────────────────────
print(f"{'='*90}")
print("  PRIORITY ACTION MATRIX")
print(f"{'='*90}")
print(f"{'Priority':<10} {'Action':<45} {'Impact':<15} {'Effort':<10} {'Evidence'}")
print(f"{'-'*100}")

actions = [
    ("P0", "Acquire conductivity/Ea labels", "TWO NEW TASKS", "High", "0% coverage"),
    ("P0", "Fix EAH: log-transform or robust target", "R²→positive", "Low", "R²=-1.69"),
    ("P1", "Architecture: hidden_dim 128→256", "Ef R² improvement", "Medium", "Arch-limited"),
    ("P1", "Architecture: ALIGNN 2→4 layers", "Ef R² improvement", "Medium", "Arch-limited"),
    ("P1", "Stratified train/test splitting", "Better generalization", "Low", "KS p<0.001"),
    ("P2", "Temperature scaling at inference", "ECE 0.27-0.47→~0.15", "Low", "ECE>0.2"),
    ("P2", "Sulfide upsampling / targeted data", "Sulfide R² improvement", "Medium", "LOFO proxy"),
    ("P2", "Benchmark expansion (54 materials)", "Broader validation", "Low", "Uncontaminated"),
]

for pri, action, impact, effort, evidence in actions:
    print(f"{pri:<10} {action:<45} {impact:<15} {effort:<10} {evidence}")

# ── Recommendation ──────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  RECOMMENDED NEXT STEPS (EXECUTION ORDER)")
print(f"{'='*90}")

print("""
  Step 1 (IMMEDIATE — < 1 hour):
  ───────────────────────────────
  • Apply temperature scaling (T=0.87 Ef, 0.97 Eah, 1.14 BG)
  • Update EAH target: retrain on v2_10000_log_eah dataset
  • Fix train/test stratification for future splits

  Step 2 (SHORT TERM — 1-3 days):
  ───────────────────────────────
  • Architecture upgrade: config/model_config_v2.yaml → hidden_dim=256, layers=4
  • Acquire/incorporate conductivity and activation energy labels
  • Re-run benchmark suite with upgraded model

  Step 3 (MEDIUM TERM — 1-2 weeks):
  ─────────────────────────────────
  • Collect more sulfide training data
  • Expand benchmark to 54+ materials across all families
  • Final production-ready model with calibrated uncertainties
""")

# ── Save ─────────────────────────────────────────────────────────────
with open(str(OUT_DIR / "decision_framework.json"), "w") as f:
    json.dump(findings, f, indent=2)

with open(str(OUT_DIR / "decision_framework.txt"), "w") as f:
    f.write("Phase 2 — Decision Framework\n")
    f.write("=" * 60 + "\n\n")
    for finding_name, data in findings.items():
        f.write(f"{finding_name}:\n")
        f.write(f"  {json.dumps(data, indent=2)}\n\n")

print(f"\nDecision framework saved to {OUT_DIR}")
