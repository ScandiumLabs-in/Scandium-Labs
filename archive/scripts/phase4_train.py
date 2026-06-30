#!/usr/bin/env python3
"""Phase 4: Integrated training with GradNorm + Two-Stage Eah."""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn
from src.training.losses import GradNormLoss, PINNLoss

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("checkpoints/phase4_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 0.001
MAX_EPOCHS = 200
PATIENCE = 50
USE_TWO_STAGE = True
USE_GRADNORM = True
LOG_EAH = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Two-Stage Eah: {USE_TWO_STAGE}, GradNorm: {USE_GRADNORM}", flush=True)

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
all_graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

raw_targets = {}
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    raw_targets[task] = np.array(cache["targets"][task], dtype=float)

class PrebuiltDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)
train_loader = DataLoader(Subset(full_dataset, split['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Subset(full_dataset, split['val']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(Subset(full_dataset, split['test']), batch_size=BATCH_SIZE, collate_fn=collate_fn)
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}", flush=True)

# ── Model ──────────────────────────────────────────────────────────────────────
model = ScandiumPINNGNN(
    hidden_dim=128, num_alignn_layers=2, num_transformer_layers=1,
    num_attention_heads=4, dropout=0.1,
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
    use_two_stage_eah=USE_TWO_STAGE,
).to(device)

# Load v2 checkpoint for initialization
ckpt = torch.load("checkpoints/best_model.pt", map_location='cpu')
sd = ckpt['model'] if 'model' in ckpt else ckpt
model_sd = model.state_dict()
filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
skipped = set(model_sd.keys()) - set(filtered_sd.keys())
model_sd.update(filtered_sd)
model.load_state_dict(model_sd)
print(f"Loaded checkpoint ({len(filtered_sd)}/{len(model_sd)} keys)", flush=True)
if skipped:
    print(f"  New keys (random init): {skipped}", flush=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params", flush=True)

# ── Loss & optimizer ──────────────────────────────────────────────────────────
mse_loss = torch.nn.MSELoss()

task_weights = {'formation_energy': 1.0, 'energy_above_hull': 1.0, 'band_gap': 0.4}

if USE_GRADNORM:
    grad_norm = GradNormLoss(
        tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
        alpha=1.5,
        initial_weights=task_weights,
    ).to(device)
    grad_norm_opt = torch.optim.Adam(grad_norm.parameters(), lr=0.025)
    print(f"GradNorm initialized with alpha=1.5", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# ── Two-Stage Eah Loss ─────────────────────────────────────────────────────────
if USE_TWO_STAGE:
    from src.models.heads.two_stage_eah import TwoStageEahLoss
    eah_two_stage_loss = TwoStageEahLoss(lambda_bce=1.0, lambda_reg=1.0, lambda_stable=0.5)

# ── Training loop ──────────────────────────────────────────────────────────────
best_val_loss = float('inf')
patience_counter = 0
t0 = time.time()

print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE})...", flush=True)

for epoch in range(MAX_EPOCHS):
    model.train()
    train_total_loss = 0
    per_task_losses_sum = {t: 0.0 for t in model.tasks if t != 'p_unstable'}
    n_batches = 0

    optimizer.zero_grad()
    if USE_GRADNORM:
        grad_norm_opt.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        cg, lg = batch
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)

        # Build targets
        targets = {}
        for task in model.tasks:
            if task == 'p_unstable':
                continue
            attr = f'y_{task}'
            if hasattr(cg, attr):
                v = getattr(cg, attr)
                if not torch.isnan(v).any():
                    targets[task] = v

        # Per-task losses
        task_losses = {}
        for task in model.tasks:
            if task == 'p_unstable':
                continue
            if task not in targets:
                continue
            if task == 'energy_above_hull' and USE_TWO_STAGE:
                # Two-stage EAH loss
                eah_true = targets[task]
                eah_out = {
                    'eah_pred': preds['energy_above_hull'],
                    'p_unstable': preds['p_unstable'],
                    'eah_magnitude': preds['eah_magnitude'],
                }
                ts_loss = eah_two_stage_loss(eah_out, eah_true)
                task_losses['energy_above_hull'] = ts_loss['total']
                train_total_loss += ts_loss['total'].item()
            else:
                loss = mse_loss(preds[task], targets[task])
                task_losses[task] = loss
                train_total_loss += loss.item()

            if task in per_task_losses_sum:
                per_task_losses_sum[task] += task_losses[task].item()

        # Combined loss
        if USE_GRADNORM:
            total_loss = grad_norm.compute_total(task_losses)
            total_loss = total_loss / GRAD_ACCUM
            # GradNorm update FIRST (gets gradients before backward clears them)
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                grad_norm.update_weights(task_losses, model.global_combiner, lr=0.025)
                grad_norm_opt.step()
                grad_norm_opt.zero_grad()
            total_loss.backward()
        else:
            total_loss = sum(task_losses.values()) / GRAD_ACCUM
            total_loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        n_batches += 1

    # Handle leftover gradient accumulation
    if n_batches % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_total_loss / n_batches

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_total_loss = 0
    val_per_task = {t: 0.0 for t in model.tasks if t != 'p_unstable'}
    val_n = 0

    with torch.no_grad():
        for batch in val_loader:
            cg, lg = batch
            cg, lg = cg.to(device), lg.to(device)
            preds = model(cg, lg)

            for task in model.tasks:
                if task == 'p_unstable':
                    continue
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if torch.isnan(v).any():
                        continue
                    if task == 'energy_above_hull' and USE_TWO_STAGE:
                        eah_true = v
                        eah_out = {
                            'eah_pred': preds['energy_above_hull'],
                            'p_unstable': preds['p_unstable'],
                            'eah_magnitude': preds['eah_magnitude'],
                        }
                        ts_loss = eah_two_stage_loss(eah_out, eah_true)
                        val_total_loss += ts_loss['total'].item()
                        val_per_task[task] += ts_loss['total'].item()
                    else:
                        l = mse_loss(preds[task], v).item()
                        val_total_loss += l
                        val_per_task[task] += l
            val_n += 1

    avg_val_loss = val_total_loss / max(1, val_n)

    # Best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'val_loss': avg_val_loss,
            'config': {'use_two_stage_eah': USE_TWO_STAGE, 'use_gradnorm': USE_GRADNORM}
        }, str(OUT_DIR / "best_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
        gn_str = ""
        if USE_GRADNORM:
            w = grad_norm.weights
            gn_str = f" w=[{w['formation_energy']:.2f}/{w['energy_above_hull']:.2f}/{w['band_gap']:.2f}]"
        print(f"  Epoch {epoch:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f}{gn_str} ({time.time()-t0:.0f}s)", flush=True)

print(f"\nTraining complete in {time.time()-t0:.0f}s", flush=True)
print(f"Best val loss: {best_val_loss:.4f}", flush=True)

# ── Test Evaluation ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)['model'])
model.eval()

all_preds = {t: [] for t in model.tasks if t != 'p_unstable'}
if USE_TWO_STAGE:
    all_preds['p_unstable'] = []
    all_preds['eah_magnitude'] = []

with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)
        for t in all_preds:
            if t in preds:
                all_preds[t].append(preds[t].cpu())

for t in all_preds:
    if all_preds[t]:
        all_preds[t] = torch.cat(all_preds[t])

test_idx = split['test']

print(f"\n{'='*70}")
print(f"TEST EVALUATION")
print(f"{'='*70}")
print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10}")
print(f"{'-'*65}")

results = {}
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    y_true = torch.tensor(raw_targets[task][test_idx], dtype=torch.float32)
    y_pred = all_preds[task]

    mask = ~torch.isnan(y_true)
    yt, yp = y_true[mask].numpy(), y_pred[mask].numpy()

    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    rmse = float(np.sqrt(np.mean((yp - yt)**2)))
    bias = float(np.mean(yp - yt))
    results[task] = {'mae': float(mae), 'r2': float(r2), 'rmse': rmse, 'bias': bias}
    print(f"{task:>25} {mae:>10.4f} {r2:>10.4f} {rmse:>10.4f} {bias:>+10.4f}")

# Two-stage metrics
if USE_TWO_STAGE and 'p_unstable' in all_preds and len(all_preds['p_unstable']) > 0:
    from src.models.heads.two_stage_eah import two_stage_metrics
    eah_true_raw = raw_targets['energy_above_hull'][test_idx]
    ts_out = {
        'p_unstable': all_preds['p_unstable'],
        'eah_pred': all_preds['energy_above_hull'],
        'eah_magnitude': all_preds['eah_magnitude'],
    }
    ts_metrics = two_stage_metrics(ts_out, eah_true_raw)
    print(f"\n  TWO-STAGE EAH METRICS:")
    print(f"    Stability F1:        {ts_metrics['stability_f1']:.4f}")
    print(f"    Precision:           {ts_metrics['stability_precision']:.4f}")
    print(f"    Recall:              {ts_metrics['stability_recall']:.4f}")
    print(f"    Eah MAE (all):       {ts_metrics['eah_mae_all']:.4f}")
    print(f"    Eah MAE (unstable):  {ts_metrics['eah_mae_unstable']:.4f}")
    results['two_stage_eah'] = ts_metrics

# Comparison with v2 baseline
print(f"\n{'='*70}")
print(f"COMPARISON: v2 baseline vs Phase 4 (GradNorm + Two-Stage Eah)")
print(f"{'='*70}")
v2_results = {}
v2_path = "checkpoints/final_eval/test_results.json"
if os.path.exists(v2_path):
    with open(v2_path) as f:
        v2_results = json.load(f)

print(f"{'':>25} {'v2 MAE':>10} {'v4 MAE':>10} {'Δ':>10} {'v2 R²':>10} {'v4 R²':>10} {'Δ':>10}")
print(f"{'-'*85}")
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    v2 = v2_results.get(task, {})
    v4 = results.get(task, {})
    v2_mae = v2.get('mae', 0)
    v4_mae = v4.get('mae', 0)
    v2_r2 = v2.get('r2', 0)
    v4_r2 = v4.get('r2', 0)
    print(f"{task:>25} {v2_mae:>10.4f} {v4_mae:>10.4f} {v4_mae-v2_mae:>+10.4f} {v2_r2:>10.4f} {v4_r2:>10.4f} {v4_r2-v2_r2:>+10.4f}")

with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll outputs saved to {OUT_DIR}/", flush=True)
