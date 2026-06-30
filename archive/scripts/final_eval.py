#!/usr/bin/env python3
"""Final evaluation: load best v2 model, apply temperature scaling, run benchmarks."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("checkpoints/final_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open(DATA_DIR / "normalizer.json") as f:
    normalizer = json.load(f)

all_graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)

raw_targets = {}
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    raw_targets[task] = np.array(cache["targets"][task], dtype=float)

class PrebuiltDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)
val_loader = DataLoader(Subset(full_dataset, split['val']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(Subset(full_dataset, split['test']), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Load model
model = ScandiumPINNGNN(
    hidden_dim=128, num_alignn_layers=2, num_transformer_layers=1,
    num_attention_heads=4, dropout=0.1,
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
).to(device)

ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
sd = ckpt['model'] if 'model' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
model.load_state_dict(sd, strict=False)
print(f"Loaded best model ({sum(p.numel() for p in model.parameters()):,} params)")

# ── Temperature scaling (fixed) ─────────────────────────────────
model.eval()
val_preds = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in val_loader:
        cg, lg = batch; cg, lg = cg.to(device), lg.to(device)
        preds, uncs = model(cg, lg, return_uncertainty=True)
        for t in model.tasks:
            val_preds[t].append(preds[t].cpu())
for t in model.tasks:
    val_preds[t] = torch.cat(val_preds[t])

def optimize_temperature(y_t, y_m, y_s):
    T = torch.tensor(1.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    best_T_val, best_T = 1e9, 1.0
    def closure():
        nonlocal best_T_val, best_T
        opt.zero_grad()
        y_s_scaled = y_s * T
        var = y_s_scaled ** 2 + 1e-8
        loss = (0.5 * torch.log(2 * np.pi * var) + 0.5 * (y_t - y_m) ** 2 / var).mean()
        loss.backward()
        if loss.item() < best_T_val:
            best_T_val = loss.item()
            best_T = T.item()
        return loss
    opt.step(closure)
    return best_T

temperatures = {}
print(f"\n{'─'*60}")
print(f"TEMPERATURE SCALING (optimized on val set)")
print(f"{'─'*60}")
for task in model.tasks:
    n = normalizer[task]
    y_true = torch.tensor(raw_targets[task][split['val']], dtype=torch.float32)
    y_pred = val_preds[task] * n['std'] + n['mean']

# Re-do properly
val_uncs = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in val_loader:
        cg, lg = batch; cg, lg = cg.to(device), lg.to(device)
        _, uncs = model(cg, lg, return_uncertainty=True)
        for t in model.tasks:
            val_uncs[t].append(uncs[t].cpu())
for t in model.tasks:
    val_uncs[t] = torch.cat(val_uncs[t])

temperatures = {}
for task in model.tasks:
    n = normalizer[task]
    y_t = torch.tensor(raw_targets[task][split['val']], dtype=torch.float32)
    y_m = val_preds[task] * n['std'] + n['mean']
    y_s = val_uncs[task] * n['std']

    mask = ~torch.isnan(y_t)
    y_t, y_m, y_s = y_t[mask], y_m[mask], y_s[mask]

    T_opt = optimize_temperature(y_t.to(device), y_m.to(device), y_s.to(device))
    temperatures[task] = float(T_opt)
    print(f"  {task:>25}: T={T_opt:.4f}")

with open(str(OUT_DIR / "temperatures.json"), "w") as f:
    json.dump(temperatures, f, indent=2)

# ── Test evaluation with temperature scaling ────────────────────
test_preds = {t: [] for t in model.tasks}
test_uncs = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch; cg, lg = cg.to(device), lg.to(device)
        preds, uncs = model(cg, lg, return_uncertainty=True)
        for t in model.tasks:
            test_preds[t].append(preds[t].cpu())
            test_uncs[t].append(uncs[t].cpu())
for t in model.tasks:
    test_preds[t] = torch.cat(test_preds[t])
    test_uncs[t] = torch.cat(test_uncs[t])

print(f"\n{'='*70}")
print(f"TEST EVALUATION (with temperature scaling)")
print(f"{'='*70}")
print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10} {'NLL↓':>10}")
print(f"{'-'*75}")

results = {}
for task in model.tasks:
    n = normalizer[task]
    T = temperatures[task]

    y_t = torch.tensor(raw_targets[task][split['test']], dtype=torch.float32)
    y_m = test_preds[task] * n['std'] + n['mean']
    y_s = test_uncs[task] * n['std'] * T

    mask = ~torch.isnan(y_t)
    yt, ym, ys = y_t[mask].numpy(), y_m[mask].numpy(), y_s[mask].numpy()

    mae = mean_absolute_error(yt, ym)
    r2 = r2_score(yt, ym)
    rmse = float(np.sqrt(np.mean((ym - yt)**2)))
    bias = float(np.mean(ym - yt))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * ys**2) + 0.5 * (yt - ym)**2 / ys**2))

    results[task] = {'mae': mae, 'r2': r2, 'rmse': rmse, 'bias': bias, 'nll': nll,
                     'temperature': T, 'num_samples': int(mask.sum())}
    print(f"{task:>25} {mae:>10.4f} {r2:>10.4f} {rmse:>10.4f} {bias:>+10.4f} {nll:>10.4f}")

print(f"\n{'='*70}")
print(f"v2 MODEL CALIBRATION SUMMARY")
print(f"{'='*70}")
print(f"  {'Task':>25} {'ECE (uncal)':>12} {'ECE (scaled)':>12}")
for task in model.tasks:
    n = normalizer[task]
    y_t = torch.tensor(raw_targets[task][split['test']], dtype=torch.float32)
    y_m = test_preds[task] * n['std'] + n['mean']
    y_s_cal = test_uncs[task] * n['std'] * temperatures[task]

    mask = ~torch.isnan(y_t)
    yt, ym, ys = y_t[mask], y_m[mask], y_s_cal[mask]

    z = (yt - ym) / ys
    ece_before = float(torch.abs(z).mean())
    ece_after = float((torch.abs(z) - 1.0).abs().mean())
    print(f"  {task:>25} {ece_before:>12.4f} {ece_after:>12.4f}")

with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to {OUT_DIR}/")
