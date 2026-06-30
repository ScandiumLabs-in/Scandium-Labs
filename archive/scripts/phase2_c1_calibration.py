#!/usr/bin/env python3
"""Phase 2 C.1: Uncertainty calibration analysis (ECE, reliability diagrams)."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn

CKPT_PATH = "checkpoints/best_model.pt"
DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/reports/phase2_c1")
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

# Load data
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)

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

# Get predictions with uncertainty
all_preds = {t: [] for t in model.tasks}
all_uncs = {t: [] for t in model.tasks}
all_targets = {t: [] for t in model.tasks}

print("Running inference with uncertainty on test set...")
with torch.no_grad():
    for cg, lg in test_loader:
        cg = cg.to(device)
        lg_img = lg.to(device) if lg is not None else None
        preds, uncs = model(cg, lg_img, return_uncertainty=True)

        for task in model.tasks:
            all_preds[task].append(preds[task].cpu())
            all_uncs[task].append(uncs[task].cpu())
            attr = f'y_{task}'
            if hasattr(cg, attr):
                all_targets[task].append(getattr(cg, attr).cpu())

for task in model.tasks:
    all_preds[task] = torch.cat(all_preds[task])
    all_uncs[task] = torch.cat(all_uncs[task])
    all_targets[task] = torch.cat(all_targets[task]) if all_targets[task] else torch.tensor([])

# ── Calibration Analysis ─────────────────────────────────────────────
def expected_calibration_error(y_true, y_mean, y_std, n_bins=15):
    mask = ~torch.isnan(y_true)
    y_t, y_m, y_s = y_true[mask], y_mean[mask], y_std[mask]
    if len(y_t) < 10:
        return float('nan'), [], []

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ece = 0
    accuracies = []
    confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = ((y_s >= bin_edges[i]) & (y_s < bin_edges[i + 1])) if i < n_bins - 1 else (y_s >= bin_edges[i])
        count = in_bin.sum().item()
        bin_counts.append(count)
        if count == 0:
            accuracies.append(0)
            confidences.append(bin_centers[i])
            continue

        bin_conf = y_s[in_bin].mean().item()
        confidences.append(bin_conf)

        z = 1.96 * bin_centers[i]
        lower = y_m[in_bin] - z
        upper = y_m[in_bin] + z
        covered = ((y_t[in_bin] >= lower) & (y_t[in_bin] <= upper)).float().mean().item()
        accuracies.append(covered)

    ece = np.mean(np.abs(np.array(confidences) - np.array(accuracies)))
    return ece, confidences, accuracies

def coverage_at_alpha(y_true, y_mean, y_std, alpha=0.95):
    mask = ~torch.isnan(y_true)
    y_t, y_m, y_s = y_true[mask], y_mean[mask], y_std[mask]
    if len(y_t) < 10:
        return float('nan')
    z = 1.96 * alpha
    lower = y_m - z * y_s
    upper = y_m + z * y_s
    return ((y_t >= lower) & (y_t <= upper)).float().mean().item()

def nll_score(y_true, y_mean, y_std):
    mask = ~torch.isnan(y_true)
    y_t, y_m, y_s = y_true[mask], y_mean[mask], y_std[mask]
    if len(y_t) < 10:
        return float('nan')
    var = y_s ** 2 + 1e-8
    nll = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (y_t - y_m) ** 2 / var
    return nll.mean().item()

print(f"\n{'='*70}")
print(f"{'UNCERTAINTY CALIBRATION REPORT':^70}")
print(f"{'='*70}")
print(f"{'Task':>25} {'ECE':>8} {'NLL':>8} {'Cover@95':>10} {'Mean σ':>8} {'Corr|σ|':>8}")
print(f"{'-'*70}")

ece_results = {}
for task in model.tasks:
    y_t = all_targets[task]
    y_m = all_preds[task]
    y_s = all_uncs[task]

    if torch.isnan(y_t).all():
        print(f"{task:>25} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
        continue

    ece, confs, accs = expected_calibration_error(y_t, y_m, y_s)
    nll = nll_score(y_t, y_m, y_s)
    cov = coverage_at_alpha(y_t, y_m, y_s)

    mask = ~torch.isnan(y_t)
    mean_sigma = y_s[mask].mean().item()
    # Correlation between |error| and predicted sigma
    errors = (y_t[mask] - y_m[mask]).abs()
    if len(errors) > 10 and errors.std() > 0 and y_s[mask].std() > 0:
        corr = np.corrcoef(errors.numpy(), y_s[mask].numpy())[0, 1]
    else:
        corr = float('nan')
    ece_results[task] = {'ece': ece, 'nll': nll, 'cov': cov, 'mean_sigma': mean_sigma, 'err_sigma_corr': corr,
                         'confidences': confs, 'accuracies': accs}
    print(f"{task:>25} {ece:>8.4f} {nll:>8.4f} {cov:>10.4f} {mean_sigma:>8.4f} {corr:>8.3f}")

# ── Reliability Diagrams ─────────────────────────────────────────────
n_tasks_with_data = sum(1 for t in model.tasks if not torch.isnan(all_targets[t]).all())
fig, axes = plt.subplots(1, max(n_tasks_with_data, 1), figsize=(6 * max(n_tasks_with_data, 1), 5))
if n_tasks_with_data == 1:
    axes = [axes]

plot_idx = 0
for task in model.tasks:
    if task not in ece_results:
        continue
    res = ece_results[task]
    if res['confidences']:
        ax = axes[plot_idx]
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax.plot(res['confidences'], res['accuracies'], 'o-', color='red', linewidth=1.5, markersize=4)
        ax.fill_between(res['confidences'], res['confidences'], res['accuracies'],
                         alpha=0.2, color='red', label=f"ECE={res['ece']:.3f}")
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task}\nECE={res["ece"]:.4f} | NLL={res["nll"]:.4f}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

plt.tight_layout()
plt.savefig(str(OUT_DIR / "reliability_diagrams.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved reliability_diagrams.png")

# ── Error vs Uncertainty Scatter ─────────────────────────────────────
n_tasks = sum(1 for t in model.tasks if t in ece_results)
if n_tasks > 0:
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]
    plot_idx = 0
    for task in model.tasks:
        if task not in ece_results:
            continue
        ax = axes[plot_idx]
        y_t = all_targets[task]
        y_m = all_preds[task]
        y_s = all_uncs[task]
        mask = ~torch.isnan(y_t)
        errors = (y_t[mask] - y_m[mask]).abs()
        ax.scatter(y_s[mask].numpy(), errors.numpy(), s=4, alpha=0.4, color='black')
        ax.set_xlabel('Predicted σ')
        ax.set_ylabel('|Error|')
        ax.set_title(f'{task}\nρ={ece_results[task]["err_sigma_corr"]:.3f}')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "error_vs_uncertainty.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error_vs_uncertainty.png")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
for task, res in ece_results.items():
    qual = "GOOD" if res['ece'] < 0.1 else ("MODERATE" if res['ece'] < 0.2 else "POOR")
    print(f"  {task:>25}: ECE={res['ece']:.4f} → {qual}  (threshold: <0.1=GOOD, <0.2=MODERATE)")
print(f"{'='*70}")

# Checklist for temperature scaling
needs_ts = any(res['ece'] > 0.3 for res in ece_results.values())
if needs_ts:
    print(f"\n⚠ Some tasks have ECE > 0.3 — temperature scaling recommended (run phase2_c2_temperature_scaling.py)")
else:
    print(f"\n✓ All tasks within acceptable calibration range")

print(f"\nAll outputs saved to {OUT_DIR}")
