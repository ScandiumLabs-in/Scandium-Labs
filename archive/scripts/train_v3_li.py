#!/usr/bin/env python3
"""Train v3_li_10k from scratch with GradNorm + Two-Stage Eah."""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, numpy as np, yaml
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn
from src.training.losses import GradNormLoss

# ── Config ──
DATA_DIR = Path("datasets/v3_li_10000")
OUT_DIR = Path("checkpoints/v3_li_10k_fresh")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("config/model_config_v3_li.yaml") as f:
    cfg = yaml.safe_load(f)

BATCH_SIZE = cfg['training']['batch_size']
GRAD_ACCUM = cfg['training']['gradient_accumulation_steps']
LR = cfg['training']['learning_rate']
MAX_EPOCHS = cfg['training']['max_epochs']
PATIENCE = cfg['training']['patience']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Data ──
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

# ── Model (fresh init) ──
mc = cfg['model']
model = ScandiumPINNGNN(
    hidden_dim=mc['hidden_dim'], num_alignn_layers=mc['num_alignn_layers'],
    num_transformer_layers=mc['num_transformer_layers'],
    num_attention_heads=mc['num_attention_heads'],
    dropout=mc['dropout'],
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
    use_two_stage_eah=mc['use_two_stage_eah'],
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params (fresh init, no checkpoint)", flush=True)

# ── Loss & optimizer ──
mse_loss = torch.nn.MSELoss()

# GradNorm setup
gc = cfg.get('gradnorm', {'enabled': True, 'alpha': 1.5})
task_weights = {'formation_energy': 1.0, 'energy_above_hull': 1.0, 'band_gap': 0.4}
grad_norm = GradNormLoss(
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
    alpha=gc.get('alpha', 1.5),
    initial_weights=task_weights,
).to(device)
grad_norm_opt = torch.optim.Adam(grad_norm.parameters(), lr=0.025)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

from src.models.heads.two_stage_eah import TwoStageEahLoss
eah_two_stage_loss = TwoStageEahLoss(lambda_bce=1.0, lambda_reg=1.0, lambda_stable=0.5)

# ── Training loop ──
best_val_loss = float('inf')
patience_counter = 0
t0 = time.time()

print(f"\nTraining up to {MAX_EPOCHS} epochs (patience={PATIENCE})...", flush=True)

for epoch in range(MAX_EPOCHS):
    model.train()
    train_total_loss = 0
    per_task_losses_sum = {t: 0.0 for t in model.tasks if t != 'p_unstable'}
    n_batches = 0

    optimizer.zero_grad()
    grad_norm_opt.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        cg, lg = batch
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)

        targets = {}
        for task in model.tasks:
            if task == 'p_unstable': continue
            attr = f'y_{task}'
            if hasattr(cg, attr):
                v = getattr(cg, attr)
                if not torch.isnan(v).any():
                    targets[task] = v

        task_losses = {}
        for task in model.tasks:
            if task == 'p_unstable' or task not in targets: continue
            if task == 'energy_above_hull':
                eah_out = {'eah_pred': preds['energy_above_hull'], 'p_unstable': preds['p_unstable'], 'eah_magnitude': preds['eah_magnitude']}
                ts_loss = eah_two_stage_loss(eah_out, targets[task])
                task_losses[task] = ts_loss['total']
                train_total_loss += ts_loss['total'].item()
            else:
                loss = mse_loss(preds[task], targets[task])
                task_losses[task] = loss
                train_total_loss += loss.item()
            if task in per_task_losses_sum:
                per_task_losses_sum[task] += task_losses[task].item()

        total_loss = grad_norm.compute_total(task_losses) / GRAD_ACCUM
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            grad_norm.update_weights(task_losses, model.global_combiner, lr=0.025)
            grad_norm_opt.step()
            grad_norm_opt.zero_grad()
        total_loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        n_batches += 1

    if n_batches % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_total_loss / n_batches

    # ── Validation ──
    model.eval()
    val_total_loss = 0
    val_n = 0
    with torch.no_grad():
        for batch in val_loader:
            cg, lg = batch; cg, lg = cg.to(device), lg.to(device)
            preds = model(cg, lg)
            for task in model.tasks:
                if task == 'p_unstable': continue
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if torch.isnan(v).any(): continue
                    if task == 'energy_above_hull':
                        eah_out = {'eah_pred': preds['energy_above_hull'], 'p_unstable': preds['p_unstable'], 'eah_magnitude': preds['eah_magnitude']}
                        val_total_loss += eah_two_stage_loss(eah_out, v)['total'].item()
                    else:
                        val_total_loss += mse_loss(preds[task], v).item()
            val_n += 1
    avg_val_loss = val_total_loss / max(1, val_n)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss; patience_counter = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': avg_val_loss, 'config': mc},
                   str(OUT_DIR / "best_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}", flush=True); break

    if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
        w = grad_norm.weights
        print(f"  Epoch {epoch:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} w=[{w['formation_energy']:.2f}/{w['energy_above_hull']:.2f}/{w['band_gap']:.2f}] ({time.time()-t0:.0f}s)", flush=True)

print(f"\nTraining complete in {time.time()-t0:.0f}s", flush=True)
print(f"Best val loss: {best_val_loss:.4f}", flush=True)

# ── Test ──
model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)['model'])
model.eval()

all_preds = {t: [] for t in model.tasks if t != 'p_unstable'}
all_preds['p_unstable'] = []; all_preds['eah_magnitude'] = []

with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch; cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)
        for t in all_preds:
            if t in preds: all_preds[t].append(preds[t].cpu())

for t in all_preds:
    if all_preds[t]: all_preds[t] = torch.cat(all_preds[t])

test_idx = split['test']
print(f"\n{'='*70}")
print(f"TEST EVALUATION"); print(f"{'='*70}")
print(f"{'Task':>25} {'MAE↓':>10} {'R²↑':>10} {'RMSE↓':>10} {'Bias':>10}")
print(f"{'-'*65}")

results = {}
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    y_true = torch.tensor(raw_targets[task][test_idx], dtype=torch.float32)
    y_pred = all_preds[task]
    mask = ~torch.isnan(y_true)
    yt, yp = y_true[mask].numpy(), y_pred[mask].numpy()
    mae = mean_absolute_error(yt, yp); r2 = r2_score(yt, yp)
    rmse = float(np.sqrt(np.mean((yp - yt)**2))); bias = float(np.mean(yp - yt))
    results[task] = {'mae': float(mae), 'r2': float(r2), 'rmse': rmse, 'bias': bias}
    print(f"{task:>25} {mae:>10.4f} {r2:>10.4f} {rmse:>10.4f} {bias:>+10.4f}")

if 'p_unstable' in all_preds and len(all_preds['p_unstable']) > 0:
    from src.models.heads.two_stage_eah import two_stage_metrics
    ts_out = {'p_unstable': all_preds['p_unstable'], 'eah_pred': all_preds['energy_above_hull'], 'eah_magnitude': all_preds['eah_magnitude']}
    ts_metrics = two_stage_metrics(ts_out, raw_targets['energy_above_hull'][test_idx])
    print(f"\n  TWO-STAGE EAH: F1={ts_metrics['stability_f1']:.4f} P={ts_metrics['stability_precision']:.4f} R={ts_metrics['stability_recall']:.4f} MAE_all={ts_metrics['eah_mae_all']:.4f}")
    results['two_stage_eah'] = ts_metrics

with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR}/", flush=True)
