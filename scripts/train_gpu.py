#!/usr/bin/env python3
"""Train using pre-built graphs on GPU."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from src.data.dataset import collate_fn
from src.models.scandium_model import ScandiumPINNGNN
from src.training.losses import PINNLoss
import time

device = torch.device('cuda')

# Load pre-built graphs
print('Loading pre-built graphs...')
all_graphs = torch.load('data/processed/prebuilt_graphs.pt', weights_only=False)
print(f'Loaded {len(all_graphs)} graphs')

# Load split indices
split = torch.load('data/processed/split_indices.pt', weights_only=False)
train_idx = split['train']
val_idx = split['val']
test_idx = split['test']
print(f'Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

# Dataset wrapper over pre-built graphs
class PrebuiltDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

full_dataset = PrebuiltDataset(all_graphs)

train_loader = DataLoader(
    Subset(full_dataset, train_idx), batch_size=8, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    Subset(full_dataset, val_idx), batch_size=8, collate_fn=collate_fn
)
test_loader = DataLoader(
    Subset(full_dataset, test_idx), batch_size=8, collate_fn=collate_fn
)

# Build model
cfg = {
    'atom_feat_dim': 92,
    'edge_feat_dim': 64,
    'hidden_dim': 128,
    'num_alignn_layers': 2,
    'num_transformer_layers': 1,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'mc_dropout_samples': 20,
    'use_pretrained_alignn': False,
    'tasks': ['log_ionic_conductivity', 'formation_energy',
              'energy_above_hull', 'activation_energy', 'band_gap']
}
model = ScandiumPINNGNN(**cfg).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

task_weights = {
    'log_ionic_conductivity': 1.0,
    'formation_energy': 1.0,
    'energy_above_hull': 0.8,
    'activation_energy': 0.6,
    'band_gap': 0.4
}
loss_fn = PINNLoss(task_weights=task_weights)

best_val_loss = float('inf')
patience_counter = 0
max_epochs = 100
patience = 15

print(f'\nTraining for up to {max_epochs} epochs (patience={patience})...')
t0 = time.time()

for epoch in range(max_epochs):
    # Train
    model.train()
    train_data_loss = 0
    train_total_loss = 0
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

        train_data_loss += losses['data'].item()
        train_total_loss += losses['total'].item()
        n_train += 1

    # Validate
    model.eval()
    val_data_loss = 0
    val_total_loss = 0
    val_mae = {t: 0 for t in model.tasks}
    val_n = {t: 0 for t in model.tasks}
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
            val_data_loss += losses['data'].item()
            val_total_loss += losses['total'].item()
            n_val += 1

            for task, pred in preds.items():
                if task in targets:
                    mae = (pred - targets[task]).abs().mean().item()
                    val_mae[task] += mae * len(pred)
                    val_n[task] += len(pred)

    avg_train_data = train_data_loss / max(1, n_train)
    avg_val_data = val_data_loss / max(1, n_val)

    val_mae_str = ' | '.join(
        f'{t}: {val_mae[t]/max(1,val_n[t]):.4f}'
        for t in model.tasks if val_n[t] > 0
    )

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}: train_data={avg_train_data:.4f} | '
          f'val_data={avg_val_data:.4f} | '
          f'MAE [{val_mae_str}] | {elapsed:.0f}s', flush=True)

    # Early stopping
    if avg_val_data < best_val_loss:
        best_val_loss = avg_val_data
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': avg_val_data,
            'config': cfg
        }, 'checkpoints/best_model.pt')
        print(f'  -> saved best model (val_data_loss={avg_val_data:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

print(f'\nTraining complete in {time.time()-t0:.0f}s')

# Test
checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

print('\n=== TEST RESULTS ===')
test_mae = {t: 0 for t in model.tasks}
test_n = {t: 0 for t in model.tasks}
with torch.no_grad():
    for batch in test_loader:
        cg, lg = batch
        cg = cg.to(device)
        lg = lg.to(device)
        preds = model(cg, lg)

        for task, pred in preds.items():
            attr = f'y_{task}'
            if hasattr(cg, attr):
                target = getattr(cg, attr)
                if not torch.isnan(target).any():
                    mae = (pred - target).abs().sum().item()
                    test_mae[task] += mae
                    test_n[task] += len(pred)

for task in model.tasks:
    if test_n[task] > 0:
        print(f'  {task}: MAE = {test_mae[task]/test_n[task]:.4f} (n={test_n[task]})')
    else:
        print(f'  {task}: no labeled test data')
