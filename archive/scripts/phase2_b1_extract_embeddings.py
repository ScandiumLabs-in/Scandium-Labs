#!/usr/bin/env python3
"""Phase 2 B.1: Extract graph embeddings from the model for all splits."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn

CKPT_PATH = "checkpoints/best_model.pt"
DATA_DIR = Path("datasets/v2_10000")
OUT_DIR = Path("experiments/reports/phase2_b1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ckpt = torch.load(CKPT_PATH, map_location='cpu')
cfg = ckpt['config']
model_cfg = cfg['model']

model = ScandiumPINNGNN(
    hidden_dim=model_cfg['hidden_dim'],
    num_alignn_layers=model_cfg['num_alignn_layers'],
    num_transformer_layers=model_cfg['num_transformer_layers'],
    num_attention_heads=model_cfg['num_attention_heads'],
    dropout=model_cfg['dropout'],
    tasks=[t['name'] for t in cfg['tasks']]
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded from {CKPT_PATH} ({sum(p.numel() for p in model.parameters()):,} params)")

# Add hook to extract graph embeddings after pooling
embeddings_cache = {}

def get_hook(name):
    def hook(module, input, output):
        embeddings_cache[name] = output.detach().cpu()
    return hook

handle = model.attention_pool.register_forward_hook(get_hook('pooled'))

# Load data
cache = torch.load(str(DATA_DIR / "dataset_cache.pt"), weights_only=False)
split = torch.load(str(DATA_DIR / "split_indices.pt"), weights_only=False)

from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

graph_builder = ALIGNNGraphBuilder(
    cutoff=cfg['graph']['cutoff'],
    max_neighbors=cfg['graph']['max_neighbors'],
    rbf_type=cfg['graph']['rbf_type'],
    num_rbf=cfg['graph']['num_rbf'],
    num_sbf=cfg['graph'].get('num_sbf', 32),
)
feature_engineer = FeatureEngineer()

from src.data.dataset import SolidElectrolyteDataset
dataset = SolidElectrolyteDataset(
    cache['structures'], cache['targets'],
    graph_builder, feature_engineer
)

batch_size = 32
loaders = {
    'train': DataLoader(Subset(dataset, split['train']), batch_size=batch_size, collate_fn=collate_fn),
    'val': DataLoader(Subset(dataset, split['val']), batch_size=batch_size, collate_fn=collate_fn),
    'test': DataLoader(Subset(dataset, split['test']), batch_size=batch_size, collate_fn=collate_fn),
}

all_embeddings = {}
all_targets = {}
all_indices = {}

for split_name, loader in loaders.items():
    print(f"Extracting embeddings for {split_name} ({len(split[split_name])} samples)...")
    emb_list = []
    tgt_list = {t: [] for t in model.tasks}
    idx_list = []

    with torch.no_grad():
        for batch_idx, (cg, lg) in enumerate(loader):
            cg = cg.to(device)
            lg = lg.to(device)
            _ = model(cg, lg)

            emb = embeddings_cache['pooled'].cpu()
            emb_list.append(emb)

            for task in model.tasks:
                attr = f'y_{task}'
                if hasattr(cg, attr):
                    tgt_list[task].append(getattr(cg, attr).cpu())

    all_embeddings[split_name] = torch.cat(emb_list, dim=0).numpy()
    all_targets[split_name] = {t: torch.cat(tgt_list[t], dim=0).numpy() if tgt_list[t] else np.array([]) for t in model.tasks}
    all_indices[split_name] = np.array(split[split_name])

    print(f"  Embeddings shape: {all_embeddings[split_name].shape}")
    print(f"  Task targets available: {[t for t in model.tasks if len(all_targets[split_name][t]) > 0]}")

handle.remove()

np.savez(str(OUT_DIR / "embeddings.npz"),
         train_emb=all_embeddings['train'], val_emb=all_embeddings['val'], test_emb=all_embeddings['test'],
         train_ef=all_targets['train']['formation_energy'],
         train_eah=all_targets['train']['energy_above_hull'],
         train_bg=all_targets['train']['band_gap'],
         val_ef=all_targets['val']['formation_energy'],
         val_eah=all_targets['val']['energy_above_hull'],
         val_bg=all_targets['val']['band_gap'],
         test_ef=all_targets['test']['formation_energy'],
         test_eah=all_targets['test']['energy_above_hull'],
         test_bg=all_targets['test']['band_gap'])
print(f"\nSaved embeddings to {OUT_DIR / 'embeddings.npz'}")
