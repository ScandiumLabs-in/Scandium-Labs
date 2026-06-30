#!/usr/bin/env python3
"""Phase 2 D.2: Full LOFO — retrain model with a material family held out."""
import sys, os, warnings, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn
from src.training.losses import PINNLoss

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/reports/phase2_d2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_FAMILY = sys.argv[1] if len(sys.argv) > 1 else "sulfide"
print(f"Holdout family: {HOLDOUT_FAMILY}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
structures = cache["structures"]

# Classify materials by family
def classify_family(formula):
    f_lower = formula.lower()
    if 's' in f_lower and not ('si' in f_lower or 'se' in f_lower):
        return 'sulfide'
    if 'o' in f_lower:
        return 'oxide'
    if 'f' in f_lower or 'cl' in f_lower or 'br' in f_lower or 'i' in f_lower:
        return 'halide'
    return 'other'

all_indices = np.arange(len(structures))
family_map = {i: classify_family(structures[i].composition.reduced_formula) for i in all_indices}

# Identify holdout indices
holdout_idx = set(i for i in all_indices if family_map[i] == HOLDOUT_FAMILY)
train_idx_mask = [i for i in split['train'] if i not in holdout_idx]
val_idx = split['val']
test_idx = split['test']

# Count how many from holdout family in each split
print(f"\nHoldout family '{HOLDOUT_FAMILY}' distribution:")
for name, idx in [("train", split['train']), ("val", val_idx), ("test", test_idx)]:
    n_holdout = sum(1 for i in idx if family_map[i] == HOLDOUT_FAMILY)
    print(f"  {name}: {n_holdout}/{len(idx)} removed")

print(f"  Training with {len(train_idx_mask)} samples (removed {len(split['train']) - len(train_idx_mask)} {HOLDOUT_FAMILY})")

# Load config
with open("config/model_config_v2.yaml") as f:
    import yaml
    cfg = yaml.safe_load(f)

# Use prebuilt graphs
prebuilt_file = DATA_DIR / "prebuilt_graphs.pt"
print(f"Loading prebuilt graphs from {prebuilt_file}...", flush=True)
all_graphs = torch.load(str(prebuilt_file), weights_only=False)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

class PrebuiltDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)

train_loader = DataLoader(Subset(full_dataset, train_idx_mask), batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=8, collate_fn=collate_fn)
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=8, collate_fn=collate_fn)

# Build model
model_cfg = cfg['model']
model = ScandiumPINNGNN(
    hidden_dim=model_cfg['hidden_dim'],
    num_alignn_layers=model_cfg['num_alignn_layers'],
    num_transformer_layers=model_cfg['num_transformer_layers'],
    num_attention_heads=model_cfg['num_attention_heads'],
    dropout=model_cfg['dropout'],
    tasks=[t['name'] for t in cfg['tasks']]
).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

task_weights = {t['name']: t.get('weight', 1.0) for t in cfg['tasks']}
loss_fn = PINNLoss(task_weights=task_weights)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
max_epochs = 60
patience = 15

NORMALIZER_PATH = DATA_DIR / "normalizer.json"
with open(NORMALIZER_PATH) as f:
    normalizer = json.load(f)

print(f"\nTraining for up to {max_epochs} epochs...")
t0 = time.time()

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    n_train = 0

    for batch in train_loader:
        cg, lg = batch
        cg = cg.to(device)
        lg = lg.to(device)
        optimizer.zero_grad()
        preds = model(cg, lg)
        targets = {}
        for task in model.tasks:
            attr = f'y_{task}'
            if hasattr(cg, attr):
                v = getattr(cg, attr)
                if not torch.isnan(v).any():
                    targets[task] = v
        losses = loss_fn(preds, targets, cg, model)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += losses['data'].item()
        n_train += 1

    # Validate
    model.eval()
    val_loss = 0
    n_val = 0
    with torch.no_grad():
        for batch in val_loader:
            cg, lg = batch
            cg = cg.to(device)
            lg = lg.to(device)
            preds = model(cg, lg)
            targets = {}
            for task in model.tasks:
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if not torch.isnan(v).any():
                        targets[task] = v
            losses = loss_fn(preds, targets, cg, model)
            val_loss += losses['data'].item()
            n_val += 1

    avg_train = train_loss / max(1, n_train)
    avg_val = val_loss / max(1, n_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), str(OUT_DIR / f"lofo_{HOLDOUT_FAMILY}_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0 or epoch == max_epochs - 1:
        print(f"  Epoch {epoch:3d}: train_loss={avg_train:.4f} val_loss={avg_val:.4f} ({time.time()-t0:.0f}s)")

print(f"Training done in {time.time()-t0:.0f}s")

# Evaluate on held-out family
model.load_state_dict(torch.load(str(OUT_DIR / f"lofo_{HOLDOUT_FAMILY}_best.pt"), map_location=device))
model.eval()

print(f"\n{'='*60}")
print(f"LOFO EVALUATION: {HOLDOUT_FAMILY.upper()} HELD OUT")
print(f"{'='*60}")

all_preds = {t: [] for t in model.tasks}
all_targets = {t: [] for t in model.tasks}

with torch.no_grad():
    for cg, lg in test_loader:
        cg = cg.to(device)
        lg = lg.to(device)
        preds = model(cg, lg)
        for task in model.tasks:
            all_preds[task].append(preds[task].cpu())
            attr = f'y_{task}'
            if hasattr(cg, attr):
                all_targets[task].append(getattr(cg, attr).cpu())

for task in model.tasks:
    all_preds[task] = torch.cat(all_preds[task])
    all_targets[task] = torch.cat(all_targets[task]) if all_targets[task] else torch.tensor([])

    # Denormalize
    if task in normalizer:
        all_preds[task] = all_preds[task] * normalizer[task]['std'] + normalizer[task]['mean']

TARGETS_TO_EVAL = ['formation_energy', 'energy_above_hull', 'band_gap']
results = {'overall': {}, 'holdout_family': {}}

for task in TARGETS_TO_EVAL:
    y_t = all_targets[task]
    y_p = all_preds[task]
    m = ~torch.isnan(y_t)

    # Overall test
    yt_all = y_t[m].numpy()
    yp_all = y_p[m].numpy()
    mae_all = mean_absolute_error(yt_all, yp_all)
    r2_all = r2_score(yt_all, yp_all)
    results['overall'][task] = {'mae': float(mae_all), 'r2': float(r2_all), 'n': int(m.sum().item())}

    # Held-out family only
    test_indices = split['test']
    fam_mask = torch.tensor([1 if family_map[ti] == HOLDOUT_FAMILY else 0 for ti in test_indices], dtype=torch.bool)
    holdout_mask = fam_mask & m
    if holdout_mask.sum() > 0:
        yt_ho = y_t[holdout_mask].numpy()
        yp_ho = y_p[holdout_mask].numpy()
        mae_ho = mean_absolute_error(yt_ho, yp_ho)
        r2_ho = r2_score(yt_ho, yp_ho)
        results['holdout_family'][task] = {'mae': float(mae_ho), 'r2': float(r2_ho), 'n': int(holdout_mask.sum().item())}
    else:
        results['holdout_family'][task] = {'mae': None, 'r2': None, 'n': 0}

    print(f"\n  {task}:")
    print(f"    Overall test:        MAE={mae_all:.4f}  R²={r2_all:.4f}")
    if task in results['holdout_family'] and results['holdout_family'][task]['n'] > 0:
        r_ho = results['holdout_family'][task]
        print(f"    {HOLDOUT_FAMILY.capitalize()} holdout:  MAE={r_ho['mae']:.4f}  R²={r_ho['r2']:.4f}  (n={r_ho['n']})")

# Compare with proxy (no-holdout) results
print(f"\n{'='*60}")
print(f"COMPARISON: {HOLDOUT_FAMILY.upper()} — Proxy vs Full LOFO")
print(f"{'='*60}")

# Load proxy results
proxy_path = Path("experiments/reports/phase2_d1/lofo_proxy_results.json")
if proxy_path.exists():
    with open(proxy_path) as f:
        proxy = json.load(f)
    for task in TARGETS_TO_EVAL:
        proxy_res = proxy.get(HOLDOUT_FAMILY, {}).get(task, {})
        lofo_res = results['holdout_family'].get(task, {})
        if proxy_res and lofo_res.get('mae') is not None:
            delta_mae = lofo_res['mae'] - proxy_res['mae']
            delta_r2 = lofo_res['r2'] - proxy_res['r2']
            print(f"  {task:>25}: Proxy MAE={proxy_res['mae']:.4f} → LOFO MAE={lofo_res['mae']:.4f} (Δ={delta_mae:+.4f})")
            print(f"  {'':>25}:  Proxy R²={proxy_res['r2']:.4f} → LOFO R²={lofo_res['r2']:.4f} (Δ={delta_r2:+.4f})")

# Save results
with open(str(OUT_DIR / f"lofo_{HOLDOUT_FAMILY}_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {OUT_DIR / f'lofo_{HOLDOUT_FAMILY}_results.json'}")
