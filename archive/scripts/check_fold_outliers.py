#!/usr/bin/env python3
"""Check whether fold-to-fold R² variance is driven by extreme Eah outliers.

If a fold's test/val set happens to catch 2-3 of the worst tail samples,
that fold's R² will collapse even if the model performs identically.
This separates "the target is hard everywhere" from "a few points are
doing all the damage."
"""
import sys, os, json, warnings, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset, Dataset
from collections import defaultdict
from pathlib import Path

from src.data.dataset import collate_fn
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss

DATA_DIR = Path("datasets/v2_10000")
BATCH_SIZE = 8
CFG = {
    'atom_feat_dim': 92, 'edge_feat_dim': 64, 'hidden_dim': 128,
    'num_alignn_layers': 2, 'num_transformer_layers': 1,
    'num_attention_heads': 4, 'dropout': 0.1, 'mc_dropout_samples': 20,
    'use_pretrained_alignn': False,
    'tasks': ['log_ionic_conductivity', 'formation_energy',
              'energy_above_hull', 'activation_energy', 'band_gap'],
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────
print("Loading data...")
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
prebuilt = torch.load(str(DATA_DIR / "prebuilt_graphs.pt"), weights_only=False)
eah_raw = np.array(cache["targets"]["energy_above_hull"], dtype=float)
structures = cache["structures"]
n = len(structures)

# ── Replicate the CV split from cross_validate.py ────────────────────
compositions = [s.composition.reduced_formula for s in structures]
prefixes = [re.match(r'([A-Z][a-z]?)', c).group(1) if re.match(r'([A-Z][a-z]?)', c) else 'Unknown' for c in compositions]
prefix_to_id = {p: i for i, p in enumerate(sorted(set(prefixes)))}
labels = np.array([prefix_to_id[p] for p in prefixes])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(n), labels))

# ── Simple PrebuiltDataset ───────────────────────────────────────────
class PrebuiltDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

dataset = PrebuiltDataset(prebuilt)

# ── Load best model from training ────────────────────────────────────
print("Loading model...")
model = ScandiumPINNGNN(**CFG).to(DEVICE)
model.eval()

best_ckpt = torch.load("checkpoints/best_model.pt", weights_only=False)
model.load_state_dict(best_ckpt['model'])
val_loss = best_ckpt.get('metrics', {}).get('total', '?')
print(f"Loaded epoch {best_ckpt['epoch']} model (val_loss={'?' if val_loss=='?' else f'{val_loss:.4f}'})")

# ── Evaluate per fold ────────────────────────────────────────────────
def get_predictions(model, loader):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            cg, lg = batch
            cg, lg = cg.to(DEVICE), lg.to(DEVICE)
            preds = model(cg, lg)
            eah_pred = preds['energy_above_hull'].cpu().numpy().flatten()
            eah_target = cg.y_energy_above_hull.cpu().numpy().flatten()
            mask = ~np.isnan(eah_target)
            all_preds.extend(eah_pred[mask].tolist())
            all_targets.extend(eah_target[mask].tolist())
    return np.array(all_preds), np.array(all_targets)

per_fold = {
    'fold': [], 'r2': [], 'mae': [], 'n_val': [],
    'n_outliers_gt_1': [], 'n_outliers_gt_2': [], 'n_outliers_gt_3': [],
    'max_eah': [], 'p99_eah': [], 'mean_eah': [],
}

for fold_id, (trn, val) in enumerate(folds):
    val_loader = DataLoader(
        Subset(dataset, val), batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    preds, targets = get_predictions(model, val_loader)

    r2 = r2_score(targets, preds)
    mae = np.abs(preds - targets).mean()
    val_eah = eah_raw[val]

    per_fold['fold'].append(fold_id + 1)
    per_fold['r2'].append(round(r2, 4))
    per_fold['mae'].append(round(mae, 4))
    per_fold['n_val'].append(len(val))
    per_fold['n_outliers_gt_1'].append(int(np.sum(val_eah > 1.0)))
    per_fold['n_outliers_gt_2'].append(int(np.sum(val_eah > 2.0)))
    per_fold['n_outliers_gt_3'].append(int(np.sum(val_eah > 3.0)))
    per_fold['max_eah'].append(round(float(np.max(val_eah)), 3))
    per_fold['p99_eah'].append(round(float(np.percentile(val_eah, 99)), 3))
    per_fold['mean_eah'].append(round(float(np.mean(val_eah)), 4))

    print(f"  Fold {fold_id+1}: R²={r2:.4f} MAE={mae:.4f} | "
          f"val_outliers>2={int(np.sum(val_eah>2.0))} >3={int(np.sum(val_eah>3.0))} | "
          f"max={np.max(val_eah):.3f}")

# ── Report ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FOLD OUTLIER vs R² ANALYSIS")
print("=" * 70)
print(f"{'Fold':>5} {'R²':>8} {'MAE':>8} {'n>1':>5} {'n>2':>5} {'n>3':>5} {'max':>7} {'p99':>7} {'mean':>7}")
print("-" * 65)
for i in range(5):
    print(f"{per_fold['fold'][i]:>5} {per_fold['r2'][i]:>8.4f} {per_fold['mae'][i]:>8.4f} "
          f"{per_fold['n_outliers_gt_1'][i]:>5} {per_fold['n_outliers_gt_2'][i]:>5} "
          f"{per_fold['n_outliers_gt_3'][i]:>5} {per_fold['max_eah'][i]:>7.3f} "
          f"{per_fold['p99_eah'][i]:>7.3f} {per_fold['mean_eah'][i]:>7.4f}")

# Correlation between R² and outlier counts
r2_arr = np.array(per_fold['r2'])
for col in ['n_outliers_gt_1', 'n_outliers_gt_2', 'n_outliers_gt_3', 'max_eah', 'p99_eah']:
    col_arr = np.array(per_fold[col])
    if np.std(col_arr) > 0:
        from scipy import stats
        r, p = stats.pearsonr(r2_arr, col_arr)
        print(f"  R² vs {col}: r={r:.3f} p={p:.4f}")

# ── Save ──────────────────────────────────────────────────────────────
out = {
    "per_fold": per_fold,
    "r2_variance": float(np.var(r2_arr)),
    "r2_std": float(np.std(r2_arr)),
    "mean_r2": float(np.mean(r2_arr)),
}
out_dir = Path("experiments/v2_3635_first_run")
with open(out_dir / "fold_outlier_analysis.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_dir / 'fold_outlier_analysis.json'}")
