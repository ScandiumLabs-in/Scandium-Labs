#!/usr/bin/env python3
"""Regenerate split to be both composition-aware and Eah-stratified.

Eliminates the train/test Eah distribution shift (KS p ~ 1e-52) while
preserving composition-based grouping (no same-reduced-formula across splits).
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats
from pathlib import Path

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/v2_3635_first_run")

# ── Load ──────────────────────────────────────────────────────────────
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
eah = np.array(cache["targets"]["energy_above_hull"], dtype=float)
structures = cache["structures"]
n = len(structures)
reduced_formulas = [s.composition.reduced_formula for s in structures]

print(f"Loaded {n} structures")

# ── Create Eah bins for stratification ────────────────────────────────
# Use quantile binning — handles the zero-inflated shape robustly
# (pd.qcut handles duplicate edges)
eah_bins = pd.qcut(eah, q=10, labels=False, duplicates="drop")
n_bins = len(np.unique(eah_bins))
print(f"Eah quantile bins: {n_bins} (after merging duplicates)")
print(f"  Per-bin counts: {pd.Series(eah_bins).value_counts().sort_index().tolist()}")

# ── StratifiedGroupKFold ─────────────────────────────────────────────
# Use 10 folds: 1 test + 1 val + 8 train = ~80/10/10
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

all_folds = list(sgkf.split(X=np.zeros(n), y=eah_bins, groups=reduced_formulas))

test_idx = sorted(all_folds[0][1].tolist())
val_idx  = sorted(all_folds[1][1].tolist())
train_idx = sorted([
    idx for i in range(2, len(all_folds)) for idx in all_folds[i][1].tolist()
])

# Sort indices for cleanliness
train_idx = sorted(train_idx)
val_idx = sorted(val_idx)
test_idx = sorted(test_idx)

split = {"train": train_idx, "val": val_idx, "test": test_idx}
print(f"\nSplit: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
print(f"  Train: {len(train_idx)/n*100:.1f}%  Val: {len(val_idx)/n*100:.1f}%  Test: {len(test_idx)/n*100:.1f}%")

# ── Verify composition isolation ──────────────────────────────────────
def get_elem_group(f):
    import re
    return '-'.join(sorted(re.findall(r'[A-Z][a-z]?', f)))

formulas_in_train = set(reduced_formulas[i] for i in train_idx)
formulas_in_test  = set(reduced_formulas[i] for i in test_idx)
overlap = formulas_in_train & formulas_in_test
print(f"\nFormula overlap train vs test: {len(overlap)} compositions")
if len(overlap) > 0:
    print(f"  Examples: {list(overlap)[:5]}")
else:
    print(f"  ✅ Zero formula leakage — composition isolation achieved")

# ── Verify distribution shift is fixed ────────────────────────────────
train_eah = eah[train_idx]
test_eah = eah[test_idx]
ks_stat, ks_p = stats.ks_2samp(train_eah, test_eah)

print(f"\nDistribution balance check:")
print(f"  Train mean={np.mean(train_eah):.4f} zero_frac={np.mean(np.isclose(train_eah,0,atol=1e-4)):.3f}")
print(f"  Test  mean={np.mean(test_eah):.4f} zero_frac={np.mean(np.isclose(test_eah,0,atol=1e-4)):.3f}")
print(f"  KS stat={ks_stat:.4f} p={ks_p:.2e}")
print(f"  {'✅ Shift resolved (p > 0.05)' if ks_p > 0.05 else '⚠️  Shift persists (p < 0.05)'}")

# Verify val too
val_eah = eah[val_idx]
ks_train_val = stats.ks_2samp(train_eah, val_eah)
ks_val_test = stats.ks_2samp(val_eah, test_eah)
print(f"  KS train vs val: p={ks_train_val.pvalue:.2e}")
print(f"  KS val vs test:  p={ks_val_test.pvalue:.2e}")

# ── Save ──────────────────────────────────────────────────────────────
out_path = DATA_DIR / "split_indices_v2.pt"
torch.save(split, str(out_path))
print(f"\nSaved to {out_path}")

# Also save a description
meta = {
    "method": "StratifiedGroupKFold (formula groups + Eah quantile bins)",
    "eah_bins": n_bins,
    "n_structures": n,
    "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
    "zero_formula_overlap_between_splits": len(overlap) == 0,
    "ks_train_vs_test": {"statistic": ks_stat, "pvalue": ks_p},
    "ks_train_vs_val": {"statistic": ks_train_val.statistic, "pvalue": ks_train_val.pvalue},
    "ks_val_vs_test": {"statistic": ks_val_test.statistic, "pvalue": ks_val_test.pvalue},
}
with open(OUT_DIR / "split_regeneration_report.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"Report saved to {OUT_DIR / 'split_regeneration_report.json'}")
