#!/usr/bin/env python3
"""Learning curves: train at 100/250/500/817 samples.

Measures MAE, RMSE, R², and training time at each data size.
Saves results to experiments/learning_curves/.
"""
import sys, os, json, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.dataset import collate_fn
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

CFG = {
    'atom_feat_dim': 92,
    'edge_feat_dim': 64,
    'hidden_dim': 128,
    'num_alignn_layers': 2,
    'num_transformer_layers': 1,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'mc_dropout_samples': 20,
    'use_pretrained_alignn': False,
    'tasks': ['formation_energy', 'energy_above_hull', 'band_gap'],
}
TASK_WEIGHTS = {
    'formation_energy': 1.0,
    'energy_above_hull': 0.8,
    'band_gap': 0.4,
}
BATCH_SIZE = 8
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5

SAMPLE_SIZES = [100, 250, 500, 817]
N_TRIALS = 3  # run each size 3x with different subsamples

# Normalizer stats (from data/normalizer.json)
NORM = {
    'formation_energy': {'mean': -1.4734, 'std': 0.6012},
    'energy_above_hull': {'mean': 0.0974, 'std': 0.3087},
    'band_gap': {'mean': 1.7154, 'std': 1.6921},
}


class PrebuiltDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return {'mae': None, 'rmse': None, 'r2': None, 'n': 0}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        'mae': float(mean_absolute_error(yt, yp)),
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'r2': float(r2_score(yt, yp)),
        'n': int(mask.sum()),
    }


def denormalize(val_norm, task):
    """Convert normalized metric back to physical units."""
    s = NORM.get(task)
    if s is None or val_norm is None:
        return None
    return val_norm * s['std']


def train_at_size(n_samples, seed, all_graphs):
    """Train on a random subset of n_samples, evaluate on held-out set."""
    rng = random.Random(seed)
    indices = list(range(len(all_graphs)))
    rng.shuffle(indices)
    # Reserve at least 100 for testing, train on up to n-100
    n_test = min(100, len(all_graphs) // 5)
    n_train = min(n_samples, len(all_graphs) - n_test)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    train_loader = DataLoader(
        Subset(PrebuiltDataset(all_graphs), train_idx),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        Subset(PrebuiltDataset(all_graphs), test_idx),
        batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    model = ScandiumPINNGNN(**CFG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = PINNLoss(task_weights=TASK_WEIGHTS)

    tasks = CFG['tasks']
    best_val_loss = float('inf')
    patience_counter = 0
    t0 = time.time()
    n_epochs_done = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            cg, lg = batch
            cg = cg.to(device)
            lg = lg.to(device)
            optimizer.zero_grad()
            preds = model(cg, lg)
            targets = {}
            for task in tasks:
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    v = getattr(cg, attr)
                    if not torch.isnan(v).any():
                        targets[task] = v
            losses = loss_fn(preds, targets, cg, model)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate on test set
        model.eval()
        val_loss_total = 0
        n_val = 0
        test_preds = {t: [] for t in tasks}
        test_targets = {t: [] for t in tasks}

        with torch.no_grad():
            for batch in test_loader:
                cg, lg = batch
                cg = cg.to(device)
                lg = lg.to(device)
                preds = model(cg, lg)
                targets = {}
                for task in tasks:
                    attr = f'y_{task}'
                    if hasattr(cg, attr):
                        v = getattr(cg, attr)
                        if not torch.isnan(v).any():
                            targets[task] = v
                            test_preds[task].append(preds[task].cpu().numpy())
                            test_targets[task].append(v.cpu().numpy())
                losses = loss_fn(preds, targets, cg, model)
                val_loss_total += losses['data'].item() if 'data' in losses else losses['total'].item()
                n_val += 1

        avg_val = val_loss_total / max(1, n_val)
        n_epochs_done = epoch + 1

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    elapsed = time.time() - t0

    # Compute metrics (in normalized space)
    metrics = {}
    for task in tasks:
        if test_targets[task]:
            yt = np.concatenate(test_targets[task])
            yp = np.concatenate(test_preds[task])
            m = compute_metrics(yt, yp)
            # Also store denormalized values
            m['mae_phys'] = denormalize(m['mae'], task)
            m['rmse_phys'] = denormalize(m['rmse'], task)
            metrics[task] = m

    total_params = sum(p.numel() for p in model.parameters())

    result = {
        'n_samples': n_samples,
        'seed': seed,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'n_epochs': n_epochs_done,
        'time_s': round(elapsed, 1),
        'best_val_loss': round(float(best_val_loss), 6),
        'metrics': metrics,
        'total_params': total_params,
    }
    return result


def main():
    os.makedirs('experiments/learning_curves', exist_ok=True)

    global all_graphs
    print('Loading pre-built graphs...')
    all_graphs = torch.load('data/processed/prebuilt_graphs.pt', weights_only=False)
    print(f'Loaded {len(all_graphs)} graphs')

    print(f'\nLearning curves at {SAMPLE_SIZES}, {N_TRIALS} trials each')
    print(f'Config: {CFG["hidden_dim"]} hidden, epochs={MAX_EPOCHS}, patience={PATIENCE}\n')

    all_results = []
    for n in SAMPLE_SIZES:
        for trial in range(N_TRIALS):
            seed = 42 + trial
            print(f'n={n:4d} trial={trial+1}/{N_TRIALS} seed={seed}...', end=' ', flush=True)
            result = train_at_size(n, seed, all_graphs)
            all_results.append(result)
            # Print quick summary
            ef = result['metrics'].get('formation_energy', {})
            bg = result['metrics'].get('band_gap', {})
            ef_mae = f'{ef.get("mae_phys", 0):.4f}' if ef.get('mae_phys') else 'N/A'
            bg_mae = f'{bg.get("mae_phys", 0):.4f}' if bg.get('mae_phys') else 'N/A'
            print(f'MAE Ef={ef_mae} eV/atom BG={bg_mae} eV | {result["n_epochs"]}ep {result["time_s"]:.0f}s')

    # Aggregate
    print(f'\n{"="*70}')
    print('LEARNING CURVES SUMMARY')
    print(f'{"="*70}')

    summary = {
        'sample_sizes': SAMPLE_SIZES,
        'n_trials': N_TRIALS,
        'config': CFG,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'results': all_results,
        'aggregate': {},
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    for n in SAMPLE_SIZES:
        trials = [r for r in all_results if r['n_samples'] == n]
        agg = {}
        for task in CFG['tasks']:
            maes = [t['metrics'][task]['mae_phys'] for t in trials if t['metrics'].get(task, {}).get('mae_phys') is not None]
            maes_norm = [t['metrics'][task]['mae'] for t in trials if t['metrics'].get(task, {}).get('mae') is not None]
            r2s = [t['metrics'][task]['r2'] for t in trials if t['metrics'].get(task, {}).get('r2') is not None]
            times = [t['time_s'] for t in trials]
            n_epochs = [t['n_epochs'] for t in trials]
            agg[task] = {
                'mae_phys': {'mean': round(float(np.mean(maes)), 4), 'std': round(float(np.std(maes)), 4)} if maes else None,
                'mae_norm': {'mean': round(float(np.mean(maes_norm)), 4), 'std': round(float(np.std(maes_norm)), 4)} if maes_norm else None,
                'r2': {'mean': round(float(np.mean(r2s)), 4), 'std': round(float(np.std(r2s)), 4)} if r2s else None,
            }
        agg['time_s'] = {'mean': round(float(np.mean(times)), 1), 'std': round(float(np.std(times)), 1)}
        agg['n_epochs'] = {'mean': round(float(np.mean(n_epochs)), 1), 'std': round(float(np.std(n_epochs)), 1)}
        summary['aggregate'][str(n)] = agg

        # Print
        print(f'\n--- n={n} ---')
        for task in CFG['tasks']:
            s = agg[task]
            if s['mae_phys']:
                print(f'  {task:25s} MAE={s["mae_phys"]["mean"]:.4f}±{s["mae_phys"]["std"]:.4f} eV | R²={s["r2"]["mean"]:.4f}±{s["r2"]["std"]:.4f}')
        print(f'  {"time":25s} {agg["time_s"]["mean"]:.0f}±{agg["time_s"]["std"]:.0f}s | epochs={agg["n_epochs"]["mean"]:.0f}±{agg["n_epochs"]["std"]:.0f}')

    # Save
    path = 'experiments/learning_curves/results.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved {path}')
    print('Done.')


if __name__ == '__main__':
    main()
