#!/usr/bin/env python3
"""Phase 5: Family-stratified training with adversarial debiasing."""
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
from src.training.losses import GradNormLoss

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("checkpoints/phase5_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
GRAD_ACCUM = 4
LR = 0.0005
MAX_EPOCHS = 200
PATIENCE = 50

# Halide shortcut countermeasures
FAMILY_LOSS_WEIGHTS = {0: 0.5, 1: 1.3, 2: 1.3, 3: 1.5, 4: 1.3}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
all_graphs = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

from pymatgen.core import Composition

def _family_id(formula):
    comp = Composition(formula)
    els = {str(e) for e in comp.elements}
    if any(e in els for e in ['F','Cl','Br','I']):
        return 0
    if 'O' in els and 'P' in els:
        return 3
    if 'O' in els:
        return 1
    if 'S' in els:
        return 2
    return 4

raw_targets = {}
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    raw_targets[task] = np.array(cache["targets"][task], dtype=float)

formulas = [s.formula for s in cache['structures']]
family_ids = np.array([_family_id(f) for f in formulas], dtype=int)

for i in range(len(all_graphs)):
    cg, lg = all_graphs[i]
    cg.family_id = int(family_ids[i])

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
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}", flush=True)

# ── Model (initialize from Phase 4 checkpoint) ─────────────────────────────────
model = ScandiumPINNGNN(
    hidden_dim=128, num_alignn_layers=2, num_transformer_layers=1,
    num_attention_heads=4, dropout=0.1,
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
    use_two_stage_eah=True,
).to(device)

ckpt = torch.load("checkpoints/phase4_final/best_model.pt", map_location='cpu')
sd = ckpt['model']
model_sd = model.state_dict()
filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
skipped = set(model_sd.keys()) - set(filtered_sd.keys())
model_sd.update(filtered_sd)
model.load_state_dict(model_sd)
print(f"Loaded Phase 4 checkpoint ({len(filtered_sd)}/{len(model_sd)} keys)", flush=True)
if skipped:
    print(f"  Skipped: {skipped}", flush=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params", flush=True)

# ── Losses & optimizer ─────────────────────────────────────────────────────────
mse_loss = torch.nn.MSELoss()

grad_norm = GradNormLoss(
    tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
    alpha=1.5,
    initial_weights={'formation_energy': 1.0, 'energy_above_hull': 1.0, 'band_gap': 0.4},
).to(device)
grad_norm_opt = torch.optim.Adam(grad_norm.parameters(), lr=0.025)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

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
    n_batches = 0

    optimizer.zero_grad()
    grad_norm_opt.zero_grad()

    for batch_idx, (cg, lg) in enumerate(train_loader):
        cg, lg = cg.to(device), lg.to(device)
        preds = model(cg, lg)

        raw_fids = cg.family_id
        if isinstance(raw_fids, (list, tuple)):
            batch_family_ids = torch.tensor(raw_fids, device=device, dtype=torch.long)
        elif isinstance(raw_fids, torch.Tensor):
            batch_family_ids = raw_fids.to(device, dtype=torch.long)
        else:
            batch_family_ids = None

        targets = {}
        for task in model.tasks:
            if task == 'p_unstable':
                continue
            attr = f'y_{task}'
            if hasattr(cg, attr):
                v = getattr(cg, attr)
                if not torch.isnan(v).any():
                    targets[task] = v

        task_losses = {}
        for task in model.tasks:
            if task == 'p_unstable':
                continue
            if task not in targets:
                continue
            if task == 'energy_above_hull':
                eah_true = targets[task]
                eah_out = {
                    'eah_pred': preds['energy_above_hull'],
                    'p_unstable': preds['p_unstable'],
                    'eah_magnitude': preds['eah_magnitude'],
                }
                fw = None
                if batch_family_ids is not None:
                    fw = torch.tensor([FAMILY_LOSS_WEIGHTS.get(f.item(), 1.0) for f in batch_family_ids],
                                      device=device, dtype=torch.float32)
                ts_loss = eah_two_stage_loss(eah_out, eah_true, family_weights=fw)
                task_losses['energy_above_hull'] = ts_loss['total']
                train_total_loss += ts_loss['total'].item()
            else:
                loss = mse_loss(preds[task], targets[task])
                task_losses[task] = loss
                train_total_loss += loss.item()

        total_loss = grad_norm.compute_total(task_losses)
        total_loss = total_loss / GRAD_ACCUM
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

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_total_loss = 0
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
                    if task == 'energy_above_hull':
                        eah_true = v
                        eah_out = {
                            'eah_pred': preds['energy_above_hull'],
                            'p_unstable': preds['p_unstable'],
                            'eah_magnitude': preds['eah_magnitude'],
                        }
                        ts_loss = eah_two_stage_loss(eah_out, eah_true)
                        val_total_loss += ts_loss['total'].item()
                    else:
                        l = mse_loss(preds[task], v).item()
                        val_total_loss += l
            val_n += 1

    avg_val_loss = val_total_loss / max(1, val_n)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'val_loss': avg_val_loss,
            'config': {'family_weights': FAMILY_LOSS_WEIGHTS}
        }, str(OUT_DIR / "best_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
        w = grad_norm.weights
        print(f"  Epoch {epoch:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f}"
              f" w=[{w['formation_energy']:.2f}/{w['energy_above_hull']:.2f}/{w['band_gap']:.2f}]"
              f" ({time.time()-t0:.0f}s)", flush=True)

print(f"\nTraining complete in {time.time()-t0:.0f}s", flush=True)
print(f"Best val loss: {best_val_loss:.4f}", flush=True)

# ── Test Evaluation ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device)['model'])
model.eval()

all_preds = {t: [] for t in model.tasks if t != 'p_unstable'}
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
test_formulas = [formulas[i] for i in test_idx]
test_families = [family_ids[i] for i in test_idx]

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
from src.models.heads.two_stage_eah import two_stage_metrics
eah_true_raw = raw_targets['energy_above_hull'][test_idx]
eah_pred = all_preds['energy_above_hull'].numpy()
p_unstable = all_preds['p_unstable'].numpy()
eah_magnitude = all_preds['eah_magnitude'].numpy()

ts_out = {'p_unstable': all_preds['p_unstable'], 'eah_pred': all_preds['energy_above_hull'], 'eah_magnitude': all_preds['eah_magnitude']}
ts_metrics = two_stage_metrics(ts_out, eah_true_raw)
print(f"\n  TWO-STAGE EAH METRICS:")
print(f"    Stability F1:        {ts_metrics['stability_f1']:.4f}")
print(f"    Precision:           {ts_metrics['stability_precision']:.4f}")
print(f"    Recall:              {ts_metrics['stability_recall']:.4f}")
print(f"    Eah MAE (all):       {ts_metrics['eah_mae_all']:.4f}")
print(f"    Eah MAE (unstable):  {ts_metrics['eah_mae_unstable']:.4f}")
results['two_stage_eah'] = ts_metrics

# Prediction distribution summary
print(f"\n  PREDICTION DISTRIBUTION:")
print(f"    Eah pred: mean={eah_pred.mean():.4f} std={eah_pred.std():.4f} min={eah_pred.min():.4f} max={eah_pred.max():.4f}")
print(f"    p_unstable: mean={p_unstable.mean():.4f} std={p_unstable.std():.4f}")
print(f"    magnitude:  mean={eah_magnitude.mean():.4f} std={eah_magnitude.std():.4f}")
print(f"    pred<1e-6:  {(eah_pred < 1e-6).sum()}/{len(eah_pred)}")
print(f"    pred<0.01:  {(eah_pred < 0.01).sum()}/{len(eah_pred)}")
print(f"    pred<0.1:   {(eah_pred < 0.1).sum()}/{len(eah_pred)}")

# ── FAMILY DIVERSITY CHECK ──────────────────────────────────────────────────
top10_idx = np.argsort(eah_pred)[:10]
print(f"\n{'='*70}")
print(f"FAMILY DIVERSITY CHECK (top-10)")
print(f"{'='*70}")
for i in top10_idx:
    fam_name = ['halide', 'oxide', 'sulfide', 'phosphate', 'other'][int(test_families[i])]
    print(f"  {test_formulas[i]:>30} fam={fam_name:>10} pred={eah_pred[i]:.6f} true={eah_true_raw[i]:.4f}")

from collections import Counter
top10_fams = Counter([int(test_families[i]) for i in top10_idx])
fam_names = {0: 'halide', 1: 'oxide', 2: 'sulfide', 3: 'phosphate', 4: 'other'}
print(f"  Family distribution: {', '.join(f'{fam_names[f]}:{c}' for f, c in top10_fams.most_common())}")

# Enrichment@10: fraction of predicted top-10 that are in the true bottom decile
k = 10
decile_threshold = np.percentile(eah_true_raw, k / len(eah_true_raw) * 100)
top10_true = eah_true_raw[top10_idx]
n_in_bottom = (top10_true < decile_threshold).sum()
random_expected = k / len(eah_true_raw) * 10  # ~0.28
enrich10 = n_in_bottom / random_expected if random_expected > 0 else 0.0
print(f"\n  Enrichment@10: {enrich10:.2f}x ({n_in_bottom}/{k} in true bottom decile)")
results['enrichment_at_10'] = float(enrich10)

with open(str(OUT_DIR / "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll outputs saved to {OUT_DIR}/", flush=True)
