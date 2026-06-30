#!/usr/bin/env python3
"""Phase 3: Train upgraded model with log-EAH and temperature-scaled inference."""
import sys, os, json, time, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn
from src.training.losses import PINNLoss

CONFIG_PATH = "scripts/phase3_config_log_eah.yaml"
DATA_DIR = Path("datasets/v2_10000")
LOG_EAH_DIR = Path("datasets/v2_10000_log_eah")
NORM_PATH = LOG_EAH_DIR / "normalizer.json"
OUT_DIR = Path("checkpoints/v3_upgraded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_EAH_EPS = 0.001

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
log_eah = cfg.get('log_eah', False)

BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Config: hidden_dim={cfg['model']['hidden_dim']}, ALIGNN layers={cfg['model']['num_alignn_layers']}")

# Load prebuilt graphs
print(f"Loading prebuilt graphs...", flush=True)
all_graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

class PrebuiltDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)

train_loader = DataLoader(Subset(full_dataset, split['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Subset(full_dataset, split['val']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(Subset(full_dataset, split['test']), batch_size=BATCH_SIZE, collate_fn=collate_fn)

print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

# Load normalizer (for denormalizing Ef and BG)
with open(NORM_PATH) as f:
    normalizer = json.load(f)

# Load raw EAH targets (original, non-log) for evaluation
raw_eah_path = Path("datasets/v2_10000/dataset_cache.pt")
orig_cache = torch.load(str(raw_eah_path), weights_only=False)
raw_eah = np.array(orig_cache["targets"]["energy_above_hull"], dtype=float)
raw_ef = np.array(orig_cache["targets"]["formation_energy"], dtype=float)
raw_bg = np.array(orig_cache["targets"]["band_gap"], dtype=float)

# Build model with upgraded architecture
model_cfg = cfg['model']
model = ScandiumPINNGNN(
    hidden_dim=model_cfg['hidden_dim'],
    num_alignn_layers=model_cfg['num_alignn_layers'],
    num_transformer_layers=model_cfg['num_transformer_layers'],
    num_attention_heads=model_cfg['num_attention_heads'],
    dropout=model_cfg['dropout'],
    tasks=[t['name'] for t in cfg['tasks']],
).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Load original v2 checkpoint for fine-tuning
CKPT_PATH = "checkpoints/best_model.pt"
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    if 'model' in ckpt:
        sd = ckpt['model']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # Filter to matching keys only
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    skipped = set(model_sd.keys()) - set(filtered_sd.keys())
    if skipped:
        print(f"  Skipped {len(skipped)} keys (shape mismatch)")
    model_sd.update(filtered_sd)
    model.load_state_dict(model_sd)
    print(f"  Loaded original checkpoint ({len(filtered_sd)}/{len(model_sd)} keys)")
    cfg['training']['learning_rate'] = 5e-5  # lower LR for fine-tuning
    cfg['training']['max_epochs'] = 100
    cfg['training']['patience'] = 30
else:
    print(f"  No checkpoint found at {CKPT_PATH}, training from scratch")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

task_weights = {t['name']: t.get('weight', 1.0) for t in cfg['tasks']}
pinn_cfg = cfg.get('pinn', {})
loss_fn = PINNLoss(task_weights=task_weights, log_eah=log_eah, **pinn_cfg)
mse_loss = torch.nn.MSELoss()

# Training
best_val_loss = float('inf')
patience_counter = 0
max_epochs = cfg['training']['max_epochs']
patience = cfg['training']['patience']
GRAD_ACCUM_STEPS = 4

print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...")
t0 = time.time()

for epoch in range(max_epochs):
    model.train()
    train_total_loss = 0
    train_data_loss = 0
    n_train = 0

    for batch_idx, batch in enumerate(train_loader):
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
                    if task == 'energy_above_hull' and log_eah and task in normalizer:
                        v_raw = v * normalizer[task]['std'] + normalizer[task]['mean']
                        v = torch.log(v_raw + LOG_EAH_EPS)
                    targets[task] = v

        losses = loss_fn(preds, targets, cg, model)
        loss_total_item = losses['total'].item()
        loss_data_item = losses['data'].item()
        losses['total'] = losses['total'] / GRAD_ACCUM_STEPS
        losses['total'].backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
            optimizer.step()
            optimizer.zero_grad()

        train_total_loss += loss_total_item
        train_data_loss += loss_data_item
        n_train += 1
        last_batch = batch_idx

    if (last_batch + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
        optimizer.step()
        optimizer.zero_grad()

    # Validate
    model.eval()
    val_data_loss = 0
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
                        if task == 'energy_above_hull' and log_eah and task in normalizer:
                            v_raw = v * normalizer[task]['std'] + normalizer[task]['mean']
                            v = torch.log(v_raw + LOG_EAH_EPS)
                        targets[task] = v
            losses = loss_fn(preds, targets, cg, model)
            val_data_loss += losses['data'].item()
            n_val += 1

    avg_train_loss = train_data_loss / max(1, n_train)
    avg_val_loss = val_data_loss / max(1, n_val)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        ckpt_path = OUT_DIR / "best_model.pt"
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'val_loss': avg_val_loss,
            'config': cfg
        }, str(ckpt_path))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0 or epoch == max_epochs - 1:
        print(f"  Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} ({time.time()-t0:.0f}s)")

print(f"\nTraining complete in {time.time()-t0:.0f}s")
print(f"Best val loss: {best_val_loss:.4f}")

# ── Evaluation ──────────────────────────────────────────────────────
model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)['model'])
model.eval()

all_preds = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch
        cg = cg.to(device)
        lg = lg.to(device)
        preds = model(cg, lg)
        for task in model.tasks:
            all_preds[task].append(preds[task].cpu())

for task in model.tasks:
    all_preds[task] = torch.cat(all_preds[task])

test_indices = split['test']

def denormalize(pred, task):
    if task in normalizer:
        return pred * normalizer[task]['std'] + normalizer[task]['mean']
    return pred

TARGETS = ['formation_energy', 'energy_above_hull', 'band_gap']

print(f"\n{'='*70}")
print(f"TEST EVALUATION")
print(f"{'='*70}")
print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10}")
print(f"{'-'*65}")

results = {}
for task in TARGETS:
    if task == 'energy_above_hull' and log_eah:
        pred_raw = torch.exp(all_preds[task]) - LOG_EAH_EPS
    else:
        pred_raw = denormalize(all_preds[task], task)

    if task == 'formation_energy':
        y_true = torch.tensor(raw_ef[test_indices], dtype=torch.float32)
    elif task == 'energy_above_hull':
        y_true = torch.tensor(raw_eah[test_indices], dtype=torch.float32)
    else:
        y_true = torch.tensor(raw_bg[test_indices], dtype=torch.float32)

    mask = ~torch.isnan(y_true)
    yt = y_true[mask].numpy()
    yp = pred_raw[mask].numpy()

    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(np.mean((yp - yt)**2))
    bias = float(np.mean(yp - yt))

    results[task] = {'mae': float(mae), 'r2': float(r2), 'rmse': float(rmse), 'bias': bias}
    print(f"{task:>25} {mae:>10.4f} {r2:>10.4f} {rmse:>10.4f} {bias:>+10.4f}")

# ── Temperature scaling ─────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"TEMPERATURE SCALING (optimized on val set)")
print(f"{'─'*70}")

# Get val predictions with uncertainty
val_preds = {t: [] for t in model.tasks}
val_uncs = {t: [] for t in model.tasks}
with torch.no_grad():
    for batch in val_loader:
        cg, lg = batch
        cg = cg.to(device)
        lg = lg.to(device)
        preds, uncs = model(cg, lg, return_uncertainty=True)
        for task in model.tasks:
            val_preds[task].append(preds[task].cpu())
            val_uncs[task].append(uncs[task].cpu())
for task in model.tasks:
    val_preds[task] = torch.cat(val_preds[task])
    val_uncs[task] = torch.cat(val_uncs[task])

def nll_loss(y_t, y_m, y_s, T):
    y_s_scaled = y_s * T
    var = y_s_scaled ** 2 + 1e-8
    return (0.5 * torch.log(2 * np.pi * var) + 0.5 * (y_t - y_m) ** 2 / var).mean()

temperatures = {}
for task in TARGETS:
    if task == 'energy_above_hull':
        raw_vals = raw_eah
    elif task == 'formation_energy':
        raw_vals = raw_ef
    else:
        raw_vals = raw_bg

    if task == 'energy_above_hull' and log_eah:
        y_t = torch.tensor([np.log(max(raw_vals[i] + LOG_EAH_EPS, 1e-10))
                           for i in split['val']], dtype=torch.float32)
    else:
        y_t = torch.tensor(raw_vals[split['val']], dtype=torch.float32)

    mask = ~torch.isnan(y_t)
    y_t = y_t[mask]
    y_m = val_preds[task][mask]
    y_s = val_uncs[task][mask]

    if task != 'energy_above_hull' or not log_eah:
        if task in normalizer:
            y_m = y_m * normalizer[task]['std'] + normalizer[task]['mean']
            y_s = y_s * normalizer[task]['std']

    if len(y_t) < 10:
        continue

    def optimize_temp(yt, ym, ys):
        T = torch.tensor(1.0, requires_grad=True, device=device)
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
        best_T = [1.0]
        def closure():
            opt.zero_grad()
            loss = nll_loss(yt.to(device), ym.to(device), ys.to(device), T)
            loss.backward()
            if loss.item() < best_T[0]:
                best_T[0] = T.item()
            return loss
        opt.step(closure)
        return best_T[0]

    T_opt = optimize_temp(y_t, y_m, y_s)
    temperatures[task] = float(T_opt)
    print(f"  {task:>25}: T={T_opt:.4f}")

with open(str(OUT_DIR / "temperatures.json"), "w") as f:
    json.dump(temperatures, f, indent=2)

# ── Compare with v2 ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"COMPARISON: v2 (128/2) vs v3 (128/2 + log-EAH)")
print(f"{'='*70}")

# Load v2 metrics from checkpoint
v2_ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
v2_metrics = v2_ckpt.get('metrics', {})
print(f"\n{'Task':>25} {'v2 MAE':>10} {'v3 MAE':>10} {'Δ':>10} {'v2 R²':>10} {'v3 R²':>10} {'Δ':>10}")
print(f"{'-'*85}")
for task in TARGETS:
    v2_mae = v2_metrics.get(f'{task}_mae', 0)
    v3_mae = results[task]['mae']
    v2_r2 = 0.68 if task == 'formation_energy' else (0.43 if task == 'energy_above_hull' else 0.32)
    v3_r2 = results[task]['r2']
    print(f"{task:>25} {v2_mae:>10.4f} {v3_mae:>10.4f} {v3_mae - v2_mae:>+10.4f} {v2_r2:>10.4f} {v3_r2:>10.4f} {v3_r2 - v2_r2:>+10.4f}")

# Save results
with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll outputs saved to {OUT_DIR}/")
