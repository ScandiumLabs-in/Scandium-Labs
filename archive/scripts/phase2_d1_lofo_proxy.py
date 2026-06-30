#!/usr/bin/env python3
"""Phase 2 D.1: LOFO Proxy — evaluate per-family performance without retraining."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn

CKPT_PATH = "checkpoints/best_model.pt"
DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/reports/phase2_d1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ckpt = torch.load(CKPT_PATH, map_location='cpu')
cfg = ckpt['config']
model_cfg = cfg['model']

model = ScandiumPINNGNN(
    hidden_dim=model_cfg['hidden_dim'],
    num_alignn_layers=model_cfg['num_alignn_layers'],
    num_transformer_layers=model_cfg['num_transformer_layers'],
    num_attention_heads=model_cfg['num_attention_heads'],
    dropout=model_cfg['dropout'],
    tasks=[t['name'] for t in cfg['tasks']]
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
structures = cache["structures"]

from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer
graph_builder = ALIGNNGraphBuilder(
    cutoff=cfg['graph']['cutoff'],
    max_neighbors=cfg['graph']['max_neighbors'],
    rbf_type=cfg['graph']['rbf_type'],
    num_rbf=cfg['graph']['num_rbf'],
    num_sbf=cfg['graph'].get('num_sbf', 32),
)
feature_engineer = FeatureEngineer()

from src.data.dataset import SolidElectrolyteDataset
dataset = SolidElectrolyteDataset(
    cache['structures'], cache['targets'],
    graph_builder, feature_engineer
)

test_loader = DataLoader(Subset(dataset, split['test']), batch_size=32, collate_fn=collate_fn)

# Get predictions
all_preds = {t: [] for t in model.tasks}
all_targets = {t: [] for t in model.tasks}

print("Running inference on test set...")
with torch.no_grad():
    for cg, lg in test_loader:
        cg = cg.to(device)
        lg_d = lg.to(device) if lg is not None else None
        preds = model(cg, lg_d)
        for task in model.tasks:
            all_preds[task].append(preds[task].cpu())
            attr = f'y_{task}'
            if hasattr(cg, attr):
                all_targets[task].append(getattr(cg, attr).cpu())

for task in model.tasks:
    all_preds[task] = torch.cat(all_preds[task])
    all_targets[task] = torch.cat(all_targets[task]) if all_targets[task] else torch.tensor([])

# Denormalize predictions using normalizer
NORMALIZER_PATH = DATA_DIR / "normalizer.json"
if NORMALIZER_PATH.exists():
    with open(NORMALIZER_PATH) as f:
        normalizer = json.load(f)
    for task in model.tasks:
        if task in normalizer:
            all_preds[task] = all_preds[task] * normalizer[task]['std'] + normalizer[task]['mean']
            print(f"  Denormalized {task}: std={normalizer[task]['std']:.4f}, mean={normalizer[task]['mean']:.4f}")
else:
    print("WARNING: No normalizer found, predictions may be in normalized space")

# Classify test indices by material family
test_indices = split['test']
def classify_family(formula):
    """Classify a formula into material family."""
    f_lower = formula.lower()
    if 's' in f_lower and not ('si' in f_lower or 'se' in f_lower):
        return 'sulfide'
    if 'o' in f_lower:
        return 'oxide'
    if 'f' in f_lower or 'cl' in f_lower or 'br' in f_lower or 'i' in f_lower:
        return 'halide'
    return 'other'

family_map = {}
for i in test_indices:
    formula = structures[i].composition.reduced_formula
    family_map[i] = classify_family(formula)

families = set(family_map.values())
print(f"\nMaterial families in test set: {families}")
for fam in sorted(families):
    count = sum(1 for v in family_map.values() if v == fam)
    print(f"  {fam}: {count}")

# Per-family metrics
TARGETS_TO_EVAL = ['formation_energy', 'energy_above_hull', 'band_gap']
print(f"\n{'='*90}")
print(f"{'LOFO PROXY — PER-FAMILY TEST PERFORMANCE':^90}")
print(f"{'='*90}")

family_results = {}
for fam in sorted(families):
    fam_mask = torch.tensor([1 if family_map[ti] == fam else 0 for ti in split['test']], dtype=torch.bool)
    print(f"\n  ── {fam.upper()} (n={fam_mask.sum().item()}) ──")
    fam_results = {}
    for task in TARGETS_TO_EVAL:
        y_t = all_targets[task]
        y_p = all_preds[task]
        m = ~torch.isnan(y_t)
        mask = fam_mask & m
        if mask.sum() < 3:
            print(f"    {task:>25}: n={mask.sum().item()} — too few samples")
            continue
        yt = y_t[mask].numpy()
        yp = y_p[mask].numpy()
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)
        bias = float(np.mean(yp - yt))
        fam_results[task] = {'mae': mae, 'r2': r2, 'bias': bias, 'n': int(mask.sum().item())}
        print(f"    {task:>25}: MAE={mae:.4f}  R²={r2:.4f}  bias={bias:+.4f}  n={mask.sum().item()}")
    family_results[fam] = fam_results

# Overall metrics
print(f"\n  ── OVERALL (all test) ──")
for task in TARGETS_TO_EVAL:
    y_t = all_targets[task]
    y_p = all_preds[task]
    m = ~torch.isnan(y_t)
    yt = y_t[m].numpy()
    yp = y_p[m].numpy()
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    bias = float(np.mean(yp - yt))
    print(f"    {task:>25}: MAE={mae:.4f}  R²={r2:.4f}  bias={bias:+.4f}  n={m.sum().item()}")

# ── Bar chart comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, task in enumerate(TARGETS_TO_EVAL):
    ax = axes[idx]
    fams = []
    maes = []
    r2s = []
    counts = []
    for fam in sorted(families):
        res = family_results.get(fam, {}).get(task)
        if res:
            fams.append(fam)
            maes.append(res['mae'])
            r2s.append(res['r2'])
            counts.append(res['n'])

    x = np.arange(len(fams))
    width = 0.35
    ax.bar(x - width/2, maes, width, label='MAE', color='black', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, r2s, width, label='R²', color='red', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=45, ha='right')
    ax.set_ylabel('MAE')
    ax2.set_ylabel('R²')
    ax.set_title(f'{task} — Per Family')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "lofo_proxy_per_family.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved lofo_proxy_per_family.png")

# ── Element count vs error ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, task in enumerate(TARGETS_TO_EVAL):
    ax = axes[idx]
    y_t = all_targets[task]
    y_p = all_preds[task]
    m = ~torch.isnan(y_t)
    yt = y_t[m].numpy()
    yp = y_p[m].numpy()
    errors = np.abs(yp - yt)
    n_atoms = np.array([len(structures[ti]) for ti in test_indices])
    n_atoms = n_atoms[m.numpy()]
    ax.scatter(n_atoms, errors, s=5, alpha=0.4, color='black')
    ax.set_xlabel('N atoms')
    ax.set_ylabel('|Error|')
    ax.set_title(f'{task}')
    # Trend line
    if len(n_atoms) > 5:
        z = np.polyfit(n_atoms, errors, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(n_atoms), p(np.sort(n_atoms)), 'r--', linewidth=1)

plt.tight_layout()
plt.savefig(str(OUT_DIR / "error_vs_natoms.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved error_vs_natoms.png")

# ── Save results ────────────────────────────────────────────────────
import json
serializable = {fam: {t: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in res.items()}
                      for t, res in fam_results.items()}
                for fam, fam_results in family_results.items()}
with open(str(OUT_DIR / "lofo_proxy_results.json"), "w") as f:
    json.dump(serializable, f, indent=2)
print(f"Saved results to {OUT_DIR / 'lofo_proxy_results.json'}")
