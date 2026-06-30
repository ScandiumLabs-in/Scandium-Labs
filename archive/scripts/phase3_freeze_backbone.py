#!/usr/bin/env python3
"""Phase 3b: Freeze backbone, train only EAH head with log-EAH."""
import sys, os, json, time, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings('ignore')
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn

DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("checkpoints/v3_freeze_eah")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_EAH_EPS = 0.001
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config
cfg = {
    'model': {'hidden_dim': 128, 'num_alignn_layers': 2, 'num_transformer_layers': 1,
              'num_attention_heads': 4, 'dropout': 0.1},
    'tasks': [{'name': 'formation_energy'}, {'name': 'energy_above_hull'}, {'name': 'band_gap'}],
    'training': {'learning_rate': 1e-4, 'weight_decay': 1e-5, 'max_epochs': 50, 'patience': 15},
}

with open(DATA_DIR / "normalizer.json") as f:
    normalizer = json.load(f)

# Load prebuilt graphs
print("Loading data...", flush=True)
all_graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

# Raw targets for evaluation
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
raw_ef = np.array(cache["targets"]["formation_energy"], dtype=float)
raw_eah = np.array(cache["targets"]["energy_above_hull"], dtype=float)
raw_bg = np.array(cache["targets"]["band_gap"], dtype=float)

class PrebuiltDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)
train_loader = DataLoader(Subset(full_dataset, split['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Subset(full_dataset, split['val']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(Subset(full_dataset, split['test']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

# Build model and load checkpoint
model = ScandiumPINNGNN(
    hidden_dim=cfg['model']['hidden_dim'], num_alignn_layers=cfg['model']['num_alignn_layers'],
    num_transformer_layers=cfg['model']['num_transformer_layers'],
    num_attention_heads=cfg['model']['num_attention_heads'],
    dropout=cfg['model']['dropout'],
    tasks=[t['name'] for t in cfg['tasks']],
).to(device)

ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
sd = ckpt['model'] if 'model' in ckpt else ckpt
model_sd = model.state_dict()
filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
model_sd.update(filtered_sd)
model.load_state_dict(model_sd)
print(f"Loaded checkpoint ({len(filtered_sd)}/{len(model_sd)} keys)", flush=True)

# Freeze everything except EAH heads
for name, param in model.named_parameters():
    if 'energy_above_hull' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} (EAH head only)", flush=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay']
)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
t0 = time.time()

print(f"\nTraining EAH head with log-EAH for up to {cfg['training']['max_epochs']} epochs...")
for epoch in range(cfg['training']['max_epochs']):
    model.train()
    train_loss = 0
    for batch in train_loader:
        cg, lg = batch
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)

        total_loss = 0
        for task in model.tasks:
            attr = f'y_{task}'
            if hasattr(cg, attr):
                v = getattr(cg, attr)
                if not torch.isnan(v).any():
                    if task == 'energy_above_hull':
                        v_raw = v * normalizer[task]['std'] + normalizer[task]['mean']
                        v = torch.log(v_raw + LOG_EAH_EPS)
                    targets = v
                    total_loss += torch.nn.functional.mse_loss(preds[task], targets)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)
            preds = model(cg, lg)
            for task in model.tasks:
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if not torch.isnan(v).any():
                        if task == 'energy_above_hull':
                            v_raw = v * normalizer[task]['std'] + normalizer[task]['mean']
                            v = torch.log(v_raw + LOG_EAH_EPS)
                        val_loss += torch.nn.functional.mse_loss(preds[task], v).item()

    avg_val = val_loss / len(val_loader)
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': avg_val}, str(OUT_DIR / "best_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= cfg['training']['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 5 == 0 or epoch == cfg['training']['max_epochs'] - 1:
        print(f"  Epoch {epoch:3d}: train_loss={train_loss/len(train_loader):.4f} val_loss={avg_val:.4f} ({time.time()-t0:.0f}s)")

print(f"\nTraining complete in {time.time()-t0:.0f}s")
print(f"Best val loss: {best_val_loss:.4f}")

# ── Test evaluation ──────────────────────────────────────────────
model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)['model'])
model.eval()
all_preds = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)
        for task in model.tasks:
            all_preds[task].append(preds[task].cpu())
for task in model.tasks:
    all_preds[task] = torch.cat(all_preds[task])

print(f"\n{'='*70}")
print(f"TEST EVALUATION")
print(f"{'='*70}")
print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10}")
print(f"{'-'*65}")

results = {}
for task, y_raw in [('formation_energy', raw_ef), ('energy_above_hull', raw_eah), ('band_gap', raw_bg)]:
    if task == 'energy_above_hull':
        pred_raw = torch.exp(all_preds[task]) - LOG_EAH_EPS
    else:
        n = normalizer[task]
        pred_raw = all_preds[task] * n['std'] + n['mean']

    y_true = torch.tensor(y_raw[split['test']], dtype=torch.float32)
    mask = ~torch.isnan(y_true)
    yt, yp = y_true[mask].numpy(), pred_raw[mask].numpy()

    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    rmse = float(np.sqrt(np.mean((yp - yt)**2)))
    bias = float(np.mean(yp - yt))
    results[task] = {'mae': float(mae), 'r2': float(r2), 'rmse': rmse, 'bias': bias}
    print(f"{task:>25} {mae:>10.4f} {r2:>10.4f} {rmse:>10.4f} {bias:>+10.4f}")

# Compare with v2
print(f"\n{'='*70}")
print(f"COMPARISON: v2 (full) vs v3 (frozen backbone + log-EAH EAH head)")
print(f"{'='*70}")
v2_ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
v2_metrics = v2_ckpt.get('metrics', {})
print(f"{'Task':>25} {'v2 MAE':>10} {'v3 MAE':>10} {'Δ':>10} {'v2 R²':>10} {'v3 R²':>10} {'Δ':>10}")
print(f"{'-'*85}")
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    v2_mae = v2_metrics.get(f'{task}_mae', 0)
    v3_mae = results[task]['mae']
    v2_r2 = 0.68 if task == 'formation_energy' else (0.43 if task == 'energy_above_hull' else 0.32)
    v3_r2 = results[task]['r2']
    print(f"{task:>25} {v2_mae:>10.4f} {v3_mae:>10.4f} {v3_mae-v2_mae:>+10.4f} {v2_r2:>10.4f} {v3_r2:>10.4f} {v3_r2-v2_r2:>+10.4f}")

with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll outputs saved to {OUT_DIR}/")
