#!/usr/bin/env python3
"""Phase 2: Quantify train/test distribution shift with plots and statistics."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from scipy import stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/v2_3635_first_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
eah = np.array(cache["targets"]["energy_above_hull"], dtype=float)

train_eah = eah[split["train"]]
val_eah = eah[split["val"]]
test_eah = eah[split["test"]]

# ── Statistics table ──────────────────────────────────────────────────
def describe(arr, label):
    valid = arr[~np.isnan(arr)]
    pct = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(valid, pct)
    return {
        "split": label, "n": len(valid),
        "mean": float(np.mean(valid)), "std": float(np.std(valid)),
        "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
        "p10": float(q[3]), "p25": float(q[4]), "p50": float(q[5]),
        "p75": float(q[6]), "p90": float(q[7]), "p95": float(q[8]),
        "p99": float(q[9]), "max": float(q[10]),
        "zero_frac": float(np.mean(np.isclose(valid, 0, atol=1e-4))),
        "skew": float(sp_stats.skew(valid)) if len(valid) > 2 else 0,
    }

stats = [describe(train_eah, "train"), describe(val_eah, "val"), describe(test_eah, "test")]

print(f"{'Split':>8} {'n':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'p50':>8} {'p95':>8} {'Max':>8} {'Zeros':>8} {'Skew':>8}")
print("-" * 80)
for s in stats:
    print(f"{s['split']:>8} {s['n']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} "
          f"{s['p50']:>8.4f} {s['p95']:>8.4f} {s['max']:>8.4f} {s['zero_frac']:>8.3f} {s['skew']:>8.2f}")

# ── Figure 1: Histogram + KDE ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Linear histogram
ax = axes[0]
ax.hist(train_eah, bins=80, alpha=0.5, label=f'Train (n={len(train_eah)})', color='black', density=True)
ax.hist(test_eah, bins=80, alpha=0.5, label=f'Test (n={len(test_eah)})', color='red', density=True)
ax.set_xlabel('Eah (eV/atom)')
ax.set_ylabel('Density')
ax.set_title('Eah Distribution — Linear')
ax.legend(fontsize=8)

# Log-scale histogram
ax = axes[1]
ax.hist(train_eah, bins=80, alpha=0.5, label='Train', color='black', density=True)
ax.hist(test_eah, bins=80, alpha=0.5, label='Test', color='red', density=True)
ax.set_yscale('log')
ax.set_xlabel('Eah (eV/atom)')
ax.set_ylabel('Density (log)')
ax.set_title('Eah Distribution — Log Y')
ax.legend(fontsize=8)

# Zoomed to tail
ax = axes[2]
ax.hist(train_eah, bins=80, alpha=0.5, label='Train', color='black', density=True, range=(0, 2))
ax.hist(test_eah, bins=80, alpha=0.5, label='Test', color='red', density=True, range=(0, 2))
ax.set_xlabel('Eah (eV/atom)')
ax.set_ylabel('Density')
ax.set_title('Eah Distribution — Zoom (0–2 eV)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "eah_histogram.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved eah_histogram.png")

# ── Figure 2: CDF —───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for arr, label, color in [(train_eah, 'Train', 'black'), (val_eah, 'Val', 'gray'), (test_eah, 'Test', 'red')]:
    sorted_vals = np.sort(arr)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, label=f'{label} (n={len(arr)})', color=color, linewidth=1.5)
ax.set_xlabel('Eah (eV/atom)')
ax.set_ylabel('CDF')
ax.set_title('Eah CDF — Train vs Val vs Test')
ax.legend()
ax.set_xlim(0, 2)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "eah_cdf.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved eah_cdf.png")

# ── Figure 3: QQ plot train vs test ──────────────────────────────────
from scipy import stats as sp_stats
fig, ax = plt.subplots(figsize=(6, 6))
sp_stats.probplot(test_eah, dist=sp_stats.norm, plot=ax)
ax.set_title('Q-Q Plot — Test Eah vs Normal')
plt.tight_layout()
plt.savefig(str(OUT_DIR / "eah_qqplot.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved eah_qqplot.png")

# ── Figure 4: Train vs Test quantile-quantile ────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
train_sorted = np.sort(train_eah)
test_sorted = np.sort(test_eah)
# Resample to same length
n_qq = min(len(train_sorted), len(test_sorted))
train_qq = np.interp(np.linspace(0, 1, n_qq), np.linspace(0, 1, len(train_sorted)), train_sorted)
test_qq = np.interp(np.linspace(0, 1, n_qq), np.linspace(0, 1, len(test_sorted)), test_sorted)
ax.scatter(train_qq, test_qq, s=3, alpha=0.5, color='black')
ax.plot([0, max(train_qq.max(), test_qq.max())], [0, max(train_qq.max(), test_qq.max())],
        'r--', linewidth=1, label='y=x')
ax.set_xlabel('Train Eah quantile')
ax.set_ylabel('Test Eah quantile')
ax.set_title('Train vs Test — Quantile-Quantile')
ax.legend()
plt.tight_layout()
plt.savefig(str(OUT_DIR / "eah_qq_train_vs_test.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved eah_qq_train_vs_test.png")

# ── Figure 5: Distribution by element group (top 10) ─────────────────
from collections import Counter
import re
formulas = [s.composition.reduced_formula for s in cache["structures"]]
def get_elem_group(f):
    return '-'.join(sorted(re.findall(r'[A-Z][a-z]?', f)))
groups = [get_elem_group(f) for f in formulas]
group_counts = Counter(groups)
top10_groups = set(g for g, _ in group_counts.most_common(10))

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(top10_groups))
width = 0.3
for i, g in enumerate(sorted(top10_groups)):
    train_mask = np.isin(groups, g) & np.isin(np.arange(len(eah)), split['train'])
    test_mask = np.isin(groups, g) & np.isin(np.arange(len(eah)), split['test'])
    train_eah_g = eah[train_mask]
    test_eah_g = eah[test_mask]
    if len(train_eah_g):
        ax.bar(i - width/2, np.mean(train_eah_g), width, color='black', alpha=0.7)
    if len(test_eah_g):
        ax.bar(i + width/2, np.mean(test_eah_g), width, color='red', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(sorted(top10_groups), rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Eah (eV/atom)')
ax.set_title('Mean Eah by Top-10 Element Groups')
ax.legend(['Train', 'Test'], fontsize=8)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "eah_by_element_group.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved eah_by_element_group.png")

print(f"\nAll plots saved to {OUT_DIR}")
