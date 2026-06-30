#!/usr/bin/env python3
"""Phase 2 A.1: Comprehensive dataset distribution analysis for all targets."""
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
OUT_DIR = Path("experiments/reports/phase2_a1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)

TARGETS = ["formation_energy", "energy_above_hull", "band_gap", "log_ionic_conductivity", "activation_energy"]
TARGET_LABELS = {
    "formation_energy": "Formation Energy (eV/atom)",
    "energy_above_hull": "E$_{above\\ hull}$ (eV/atom)",
    "band_gap": "Band Gap (eV)",
    "log_ionic_conductivity": "Log Ionic Conductivity (log S/cm)",
    "activation_energy": "Activation Energy (eV)",
}

def describe(arr, label):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"split": label, "n": 0, "n_nan": len(arr), "mean": float('nan'), "std": float('nan'),
                "min": float('nan'), "p50": float('nan'), "p95": float('nan'), "max": float('nan'),
                "zero_frac": float('nan'), "skew": float('nan'), "kurtosis": float('nan')}
    pct = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    q = np.percentile(valid, pct)
    return {
        "split": label, "n": len(valid), "n_nan": len(arr) - len(valid),
        "mean": float(np.mean(valid)), "std": float(np.std(valid)),
        "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
        "p10": float(q[3]), "p25": float(q[4]), "p50": float(q[5]),
        "p75": float(q[6]), "p90": float(q[7]), "p95": float(q[8]),
        "p99": float(q[9]), "max": float(q[10]),
        "zero_frac": float(np.mean(np.isclose(valid, 0, atol=1e-4))),
        "skew": float(sp_stats.skew(valid)) if len(valid) > 2 else 0,
        "kurtosis": float(sp_stats.kurtosis(valid)) if len(valid) > 2 else 0,
    }

rows = []
for target in TARGETS:
    raw = np.array(cache["targets"][target], dtype=float)
    if target == "log_ionic_conductivity":
        mask = ~np.isnan(raw)
        print(f"[{target}] {mask.sum()} valid / {len(raw)} total")
        continue
    stats_list = [
        describe(raw[split["train"]], "train"),
        describe(raw[split["val"]], "val"),
        describe(raw[split["test"]], "test"),
    ]
    rows.append((target, raw, stats_list))

print(f"\n{'='*90}")
print(f"{'TARGET':>25} {'Split':>6} {'n':>6} {'NaN':>5} {'Mean':>9} {'Std':>8} {'Min':>8} {'p50':>8} {'p95':>8} {'Max':>8} {'Zeros':>7} {'Skew':>7}")
print(f"{'='*90}")
for target, raw, stats_list in rows:
    for s in stats_list:
        print(f"{target:>25} {s['split']:>6} {s['n']:>6} {s['n_nan']:>5} {s['mean']:>9.4f} {s['std']:>8.4f} "
              f"{s['min']:>8.4f} {s['p50']:>8.4f} {s['p95']:>8.4f} {s['max']:>8.4f} {s['zero_frac']:>7.3f} {s['skew']:>7.2f}")
    print()

# Distribution shift tests
print(f"\n{'='*90}")
print(f"{'DISTRIBUTION SHIFT TESTS (Train vs Test)':^90}")
print(f"{'='*90}")
for target, raw, stats_list in rows:
    train_v = raw[split["train"]]
    test_v = raw[split["test"]]
    train_v = train_v[~np.isnan(train_v)]
    test_v = test_v[~np.isnan(test_v)]
    if len(train_v) < 2 or len(test_v) < 2:
        continue
    ks_stat, ks_p = sp_stats.ks_2samp(train_v, test_v)
    epps_stat, epps_p = sp_stats.epps_singleton_2samp(train_v, test_v) if (len(train_v) > 4 and len(test_v) > 4) else (0, 1)
    print(f"  {target:>25}: KS={ks_stat:.4f} (p={ks_p:.2e})  ES={epps_stat:.4f} (p={epps_p:.2e})  {'⚠ SHIFT' if ks_p < 0.05 else 'OK'}")

# ── Figures ──────────────────────────────────────────────────────────
for target, raw, stats_list in rows:
    train_v = raw[split["train"]]
    val_v = raw[split["val"]]
    test_v = raw[split["test"]]
    if np.isnan(train_v).all() or np.isnan(test_v).all():
        print(f"  Skipping {target} — all NaN")
        continue
    label = TARGET_LABELS.get(target, target)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1: Linear histogram
    ax = axes[0, 0]
    ax.hist(train_v, bins=80, alpha=0.5, label=f'Train (n={len(train_v)})', color='black', density=True)
    ax.hist(test_v, bins=80, alpha=0.5, label=f'Test (n={len(test_v)})', color='red', density=True)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.set_title(f'{target} — Linear')
    ax.legend(fontsize=7)

    # 2: Log-scale histogram
    ax = axes[0, 1]
    ax.hist(train_v, bins=80, alpha=0.5, label='Train', color='black', density=True)
    ax.hist(test_v, bins=80, alpha=0.5, label='Test', color='red', density=True)
    ax.set_yscale('log')
    ax.set_xlabel(label)
    ax.set_title(f'{target} — Log Y')
    ax.legend(fontsize=7)

    # 3: CDF
    ax = axes[0, 2]
    for arr, lbl, clr in [(train_v, 'Train', 'black'), (val_v, 'Val', 'gray'), (test_v, 'Test', 'red')]:
        sorted_vals = np.sort(arr)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=lbl, color=clr, linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel('CDF')
    ax.set_title(f'{target} — CDF')
    ax.legend(fontsize=7)

    # 4: QQ train vs test
    ax = axes[1, 0]
    t_sorted = np.sort(train_v)
    e_sorted = np.sort(test_v)
    n_qq = min(len(t_sorted), len(e_sorted))
    t_qq = np.interp(np.linspace(0, 1, n_qq), np.linspace(0, 1, len(t_sorted)), t_sorted)
    e_qq = np.interp(np.linspace(0, 1, n_qq), np.linspace(0, 1, len(e_sorted)), e_sorted)
    ax.scatter(t_qq, e_qq, s=3, alpha=0.5, color='black')
    lims = [min(t_qq.min(), e_qq.min()), max(t_qq.max(), e_qq.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('Train quantile')
    ax.set_ylabel('Test quantile')
    ax.set_title(f'{target} — Q-Q (Train vs Test)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # 5: Boxplot by split
    ax = axes[1, 1]
    bp = ax.boxplot([train_v, val_v, test_v], patch_artist=True)
    ax.set_xticklabels(['Train', 'Val', 'Test'])
    for patch, color in zip(bp['boxes'], ['black', 'gray', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_ylabel(label)
    ax.set_title(f'{target} — Boxplot by Split')

    # 6: Density (KDE) comparison
    ax = axes[1, 2]
    for arr, lbl, clr in [(train_v, 'Train', 'black'), (test_v, 'Test', 'red')]:
        density = sp_stats.gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 300)
        ax.plot(xs, density(xs), label=lbl, color=clr, linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.set_title(f'{target} — KDE')
    ax.legend(fontsize=7)

    plt.tight_layout()
    safe_name = target.replace("_", "_")
    plt.savefig(str(OUT_DIR / f"distribution_{safe_name}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved distribution_{safe_name}.png")

# ── Correlation matrix ──────────────────────────────────────────────
print(f"\n{'─'*40}\nCorrelation Matrix (full dataset, non-NaN pairs):\n")
valid_targets = [t for t in TARGETS if t != "log_ionic_conductivity" and t != "activation_energy"]
mat_data = {}
for t in valid_targets:
    arr = np.array(cache["targets"][t], dtype=float)
    mat_data[t] = arr
stack = np.column_stack([mat_data[t] for t in valid_targets])
corr_mask = ~np.isnan(stack).any(axis=1)
stack_clean = stack[corr_mask]
corr = np.corrcoef(stack_clean.T)
spearman_corr, _ = sp_stats.spearmanr(stack_clean)

print(f"{'':>25}", end="")
for t in valid_targets:
    print(f"{t:>20}", end="")
print()
for i, t in enumerate(valid_targets):
    print(f"{t:>25}", end="")
    for j in range(len(valid_targets)):
        print(f"{corr[i,j]:>20.4f}", end="")
    print()

print(f"\nSpearman Rank Correlation:")
for i, t in enumerate(valid_targets):
    print(f"{t:>25}", end="")
    for j in range(len(valid_targets)):
        print(f"{spearman_corr[i,j]:>20.4f}", end="")
    print()

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, (title, cmat) in enumerate([("Pearson Correlation", corr), ("Spearman Rank Correlation", spearman_corr)]):
    ax = axes[idx]
    im = ax.imshow(cmat, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(valid_targets)))
    ax.set_yticks(range(len(valid_targets)))
    ax.set_xticklabels(valid_targets, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(valid_targets, fontsize=8)
    for i in range(len(valid_targets)):
        for j in range(len(valid_targets)):
            ax.text(j, i, f"{cmat[i,j]:.3f}", ha='center', va='center', fontsize=7)
    ax.set_title(title)
plt.colorbar(im, ax=axes, shrink=0.6)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "target_correlation.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved target_correlation.png")

print(f"\nAll reports saved to {OUT_DIR}")
