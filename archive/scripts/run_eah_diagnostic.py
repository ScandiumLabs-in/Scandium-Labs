#!/usr/bin/env python3
"""Wire eah_diagnostic.py to the actual v2 dataset and run all three checks."""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
from collections import defaultdict

from src.data.dataset import collate_fn
from scipy import stats
from scripts.eah_diagnostic import (
    diagnose_eah_distribution,
    compare_train_test_distribution,
    correlate_with_eah,
    check_fold_chemistry_overlap,
)

DATA_DIR = Path("datasets/v2_10000")

# ── Load data ─────────────────────────────────────────────────────────
print("Loading dataset...")
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
prebuilt = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)

structures = cache["structures"]
targets = cache["targets"]
all_graphs = prebuilt  # list of (cg, lg) tuples
train_idx = split["train"]
val_idx = split["val"]
test_idx = split["test"]

n = len(structures)
print(f"Loaded {n} structures, {len(train_idx)}/{len(val_idx)}/{len(test_idx)} train/val/test")

# ── Raw target arrays ────────────────────────────────────────────────
eah_raw = np.array(targets["energy_above_hull"], dtype=float)
ef_raw  = np.array(targets["formation_energy"], dtype=float)
bg_raw  = np.array(targets["band_gap"], dtype=float)

train_eah = eah_raw[train_idx]
val_eah   = eah_raw[val_idx]
test_eah  = eah_raw[test_idx]

# ── Compositions ─────────────────────────────────────────────────────
compositions = [s.composition.reduced_formula for s in structures]

# ── Other properties for correlation ─────────────────────────────────
n_atoms = np.array([s.num_sites for s in structures], dtype=float)
volumes = np.array([s.volume for s in structures], dtype=float)
densities = np.array([s.density for s in structures], dtype=float)

other_properties = {
    "formation_energy": ef_raw,
    "band_gap": bg_raw,
    "density": densities,
    "volume": volumes,
    "num_atoms": n_atoms,
}

# ═══════════════════════════════════════════════════════════════════════
# CHECK 1: KEY-MISMATCH AUDIT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. KEY-MISMATCH AUDIT")
print("=" * 70)

# Compare each prebuilt graph's y_energy_above_hull against raw target
mismatches = []
for idx in range(n):
    cg, lg = all_graphs[idx]
    graph_val = float(cg.y_energy_above_hull[0])
    raw_val = float(targets["energy_above_hull"][idx])
    if not np.isclose(graph_val, raw_val, atol=1e-6):
        mismatches.append({"idx": idx, "graph": graph_val, "raw": raw_val})

audit_result = {
    "status": "PASS" if not mismatches else "FAIL",
    "n_checked": n,
    "n_mismatches": len(mismatches),
    "examples": mismatches[:10],
    "all_y_keys_present": (
        "y_energy_above_hull" in dir(cg) and
        "y_formation_energy" in dir(cg) and
        "y_band_gap" in dir(cg)
    ),
}
print(json.dumps(audit_result, indent=2, default=str))

if audit_result["status"] == "FAIL":
    print("\n>>> STOP. Fix key mismatch before proceeding.")
    sys.exit(1)
print("\n>>> Eah values consistent between raw targets and prebuilt graphs. Proceeding.\n")

# ═══════════════════════════════════════════════════════════════════════
# CHECK 2: DISTRIBUTION DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("2a. EAH DISTRIBUTION")
print("-" * 40)

full_diag  = diagnose_eah_distribution(eah_raw, "full (3635)")
train_diag = diagnose_eah_distribution(train_eah, f"train ({len(train_eah)})")
test_diag  = diagnose_eah_distribution(test_eah, f"test ({len(test_eah)})")

print("FULL:")
print(json.dumps(full_diag, indent=2, default=str))
print("\nTRAIN:")
print(json.dumps(train_diag, indent=2, default=str))
print("\nTEST:")
print(json.dumps(test_diag, indent=2, default=str))

print("\n2b. TRAIN vs TEST DISTRIBUTION SHIFT (KS-test)")
print("-" * 40)
ks_result = compare_train_test_distribution(train_eah, test_eah)
print(json.dumps(ks_result, indent=2, default=str))

print("\n2c. CORRELATIONS WITH OTHER PROPERTIES")
print("-" * 40)
corr_result = correlate_with_eah(eah_raw, other_properties)
print(json.dumps(corr_result, indent=2, default=str))

# ═══════════════════════════════════════════════════════════════════════
# CHECK 3: CV FOLD CHEMISTRY OVERLAP
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. CV FOLD CHEMISTRY ANALYSIS")
print("=" * 70)

# Replicate the CV fold creation from cross_validate.py
from sklearn.model_selection import StratifiedKFold
import re

def extract_prefix(formula):
    match = re.match(r'([A-Z][a-z]?)', formula)
    return match.group(1) if match else 'Unknown'

prefixes = [extract_prefix(c) for c in compositions]
prefix_to_id = {p: i for i, p in enumerate(sorted(set(prefixes)))}
labels = np.array([prefix_to_id[p] for p in prefixes])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(n), labels))

# Show fold sizes and prefix distribution per fold
print("\nFold sizes and prefix counts:")
fold_info = {}
for fold_id, (trn, val) in enumerate(folds):
    val_prefixes = [prefixes[i] for i in val]
    prefix_counts = defaultdict(int)
    for p in val_prefixes:
        prefix_counts[p] += 1
    top_prefixes = sorted(prefix_counts.items(), key=lambda x: -x[1])[:5]
    fold_info[f"fold_{fold_id+1}"] = {
        "val_size": len(val),
        "train_size": len(trn),
        "top_val_prefixes": dict(top_prefixes),
    }
    print(f"  Fold {fold_id+1}: train={len(trn)} val={len(val)} | val prefixes: {dict(top_prefixes)}")

# Compute Eah statistics per fold
print("\nEah statistics per fold (validation set):")
for fold_id, (trn, val) in enumerate(folds):
    fold_eah = eah_raw[val]
    zero_frac = float(np.mean(np.isclose(fold_eah, 0.0, atol=1e-4)))
    print(f"  Fold {fold_id+1}: mean={np.mean(fold_eah):.4f} std={np.std(fold_eah):.4f} "
          f"median={np.median(fold_eah):.4f} zero_frac={zero_frac:.3f} "
          f"skew={float(stats.skew(fold_eah)):.2f}" if len(fold_eah) > 2 else "  too few")

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════
output = {
    "key_audit": audit_result,
    "distribution": {
        "full": full_diag,
        "train": train_diag,
        "test": test_diag,
        "train_vs_test_ks": ks_result,
        "correlations": corr_result,
    },
    "fold_analysis": fold_info,
}
out_dir = Path("experiments/v2_3635_first_run")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "eah_diagnostic.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_dir / 'eah_diagnostic.json'}")
