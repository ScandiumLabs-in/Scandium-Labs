#!/usr/bin/env python3
"""Phase 2 C.2: Temperature scaling for uncertainty calibration."""
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
OUT_DIR = Path("experiments/reports/phase2_c2")
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

val_loader = DataLoader(Subset(dataset, split['val']), batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(Subset(dataset, split['test']), batch_size=32, collate_fn=collate_fn)

def get_predictions(loader):
    all_preds = {t: [] for t in model.tasks}
    all_uncs = {t: [] for t in model.tasks}
    all_targets = {t: [] for t in model.tasks}
    with torch.no_grad():
        for cg, lg in loader:
            cg = cg.to(device)
            lg_d = lg.to(device) if lg is not None else None
            preds, uncs = model(cg, lg_d, return_uncertainty=True)
            for task in model.tasks:
                all_preds[task].append(preds[task].cpu())
                all_uncs[task].append(uncs[task].cpu())
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    all_targets[task].append(getattr(cg, attr).cpu())
    return {t: torch.cat(all_preds[t]) for t in model.tasks}, \
           {t: torch.cat(all_uncs[t]) for t in model.tasks}, \
           {t: torch.cat(all_targets[t]) if all_targets[t] else torch.tensor([]) for t in model.tasks}

print("Getting val predictions...")
val_preds, val_uncs, val_targets = get_predictions(val_loader)
print("Getting test predictions...")
test_preds, test_uncs, test_targets = get_predictions(test_loader)

def nll_with_temp(y_t, y_m, y_s, T):
    y_s_scaled = y_s * T
    var = y_s_scaled ** 2 + 1e-8
    nll = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (y_t - y_m) ** 2 / var
    return nll.mean()

def ece_with_temp(y_t, y_m, y_s, T, n_bins=15):
    y_s_scaled = y_s * T
    mask = ~torch.isnan(y_t)
    y_t_m, y_m_m, y_s_m = y_t[mask], y_m[mask], y_s_scaled[mask]
    if len(y_t_m) < 10:
        return float('nan')
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ece_vals = []
    for i in range(n_bins):
        in_bin = ((y_s_m >= bin_edges[i]) & (y_s_m < bin_edges[i + 1])) if i < n_bins - 1 else (y_s_m >= bin_edges[i])
        count = in_bin.sum().item()
        if count == 0:
            continue
        z = 1.96 * bin_centers[i]
        lower = y_m_m[in_bin] - z
        upper = y_m_m[in_bin] + z
        covered = ((y_t_m[in_bin] >= lower) & (y_t_m[in_bin] <= upper)).float().mean().item()
        ece_vals.append(abs(bin_centers[i] - covered))
    return np.mean(ece_vals) if ece_vals else float('nan')

print(f"\n{'='*70}")
print(f"{'TEMPERATURE SCALING':^70}")
print(f"{'='*70}")

temperature_results = {}
for task in model.tasks:
    y_t = val_targets[task]
    y_m = val_preds[task]
    y_s = val_uncs[task]
    if torch.isnan(y_t).all() or len(y_t[~torch.isnan(y_t)]) < 10:
        print(f"{task:>25}: N/A (no data)")
        continue

    mask = ~torch.isnan(y_t)
    y_t_v, y_m_v, y_s_v = y_t[mask], y_m[mask], y_s[mask]
    y_t_d = y_t_v.to(device)
    y_m_d = y_m_v.to(device)
    y_s_d = y_s_v.to(device)

    def optimize_temp(yt, ym, ys):
        T = torch.tensor(1.0, requires_grad=True, device=device)
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
        best_T = [1.0]
        best_nll = [float('inf')]

        def closure():
            opt.zero_grad()
            loss = nll_with_temp(yt, ym, ys, T)
            loss.backward()
            if loss.item() < best_nll[0]:
                best_T[0] = T.item()
                best_nll[0] = loss.item()
            return loss

        opt.step(closure)
        return best_T[0]

    best_T = optimize_temp(y_t_d, y_m_d, y_s_d)

    # Apply to test
    y_t_test = test_targets[task]
    y_m_test = test_preds[task]
    y_s_test = test_uncs[task]
    mask_test = ~torch.isnan(y_t_test)
    ece_before = ece_with_temp(y_t_test, y_m_test, y_s_test, 1.0)
    ece_after = ece_with_temp(y_t_test, y_m_test, y_s_test, best_T)
    nll_before = nll_with_temp(y_t_test[mask_test], y_m_test[mask_test], y_s_test[mask_test], 1.0).item()
    nll_after = nll_with_temp(y_t_test[mask_test], y_m_test[mask_test], y_s_test[mask_test] * best_T, 1.0).item()

    temperature_results[task] = {
        'T': best_T,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'nll_before': nll_before,
        'nll_after': nll_after,
    }
    print(f"{task:>25}: T={best_T:.4f} | ECE {ece_before:.4f}→{ece_after:.4f} | NLL {nll_before:.4f}→{nll_after:.4f}")

print(f"\n{'='*70}")
print("Temperature scaling results saved.")
print(f"{'='*70}")

# Save temperature values
import json
with open(str(OUT_DIR / "temperatures.json"), "w") as f:
    json.dump(temperature_results, f, indent=2)
print(f"Saved temperatures to {OUT_DIR / 'temperatures.json'}")
