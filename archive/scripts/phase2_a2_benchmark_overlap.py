#!/usr/bin/env python3
"""Phase 2 A.2: Check if benchmark materials overlap with the training set."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/reports/phase2_a2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)

structures = cache["structures"]
formulas = [s.composition.reduced_formula for s in structures]

# Benchmark materials from run_benchmark.py
BENCHMARK_FORMULAS = [
    "Li6PS5Cl", "Li2O", "LiF", "NaCl", "MgO", "LiCoO2", "LiFePO4",
    "Li3PO4", "Li2TiO3", "Li2CO3", "SiO2", "Al2O3", "Li2S"
]

train_idx = set(split["train"])
val_idx = set(split["val"])
test_idx = set(split["test"])

print(f"{'='*80}")
print(f"{'BENCHMARK OVERLAP CHECK':^80}")
print(f"{'='*80}")
print(f"{'Formula':>15} {'In Dataset':>15} {'In Train':>12} {'In Val':>10} {'In Test':>10} {'Count':>8}")
print(f"{'-'*80}")

overlap_counts = defaultdict(int)
for bm in BENCHMARK_FORMULAS:
    matches = [i for i, f in enumerate(formulas) if f.lower() == bm.lower()]
    in_train = sum(1 for i in matches if i in train_idx)
    in_val = sum(1 for i in matches if i in val_idx)
    in_test = sum(1 for i in matches if i in test_idx)
    in_dataset = len(matches) > 0
    print(f"{bm:>15} {'YES' if in_dataset else 'NO':>15} {in_train:>12} {in_val:>10} {in_test:>10} {len(matches):>8}")
    if in_dataset:
        overlap_counts['total'] += 1
        if in_train: overlap_counts['train'] += 1
        if in_val: overlap_counts['val'] += 1
        if in_test: overlap_counts['test'] += 1

print(f"\n{'='*80}")
print(f"Summary: {overlap_counts['total']}/{len(BENCHMARK_FORMULAS)} benchmark materials found in dataset")
print(f"  In train: {overlap_counts.get('train', 0)}")
print(f"  In val:   {overlap_counts.get('val', 0)}")
print(f"  In test:  {overlap_counts.get('test', 0)}")
print(f"\nImplication: Benchmark results may be inflated for materials that appear in training set.")
print(f"{'='*80}")

# Check exact formula matches for benchmark materials
print(f"\nDetailed overlap report saved to {OUT_DIR / 'benchmark_overlap.txt'}")
with open(str(OUT_DIR / "benchmark_overlap.txt"), "w") as f:
    f.write(f"{'Formula':>15} {'Split':>10} {'n_atoms':>10} {'Ef':>10} {'Eah':>10} {'BG':>10}\n")
    f.write(f"{'-'*65}\n")
    for bm in BENCHMARK_FORMULAS:
        matches = [i for i, f in enumerate(formulas) if f.lower() == bm.lower()]
        for i in matches:
            which = "train" if i in train_idx else ("val" if i in val_idx else ("test" if i in test_idx else "unknown"))
            n_at = len(structures[i])
            ef = cache["targets"]["formation_energy"][i]
            eah = cache["targets"]["energy_above_hull"][i]
            bg = cache["targets"]["band_gap"][i]
            f.write(f"{bm:>15} {which:>10} {n_at:>10} {ef:>10.4f} {eah:>10.4f} {bg:>10.4f}\n")

print("Done.")
