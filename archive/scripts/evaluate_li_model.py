#!/usr/bin/env python3
"""
Post-training evaluation for v3_li_2k.
Usage:  python scripts/evaluate_li_model.py \\
            --checkpoint checkpoints/v3_li_2k_fresh/best_model.pt \\
            --data_dir datasets/v3_li_2k
"""
import argparse, torch, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.scandium_model import ScandiumPINNGNN
from src.data.dataset import collate_fn
from torch.utils.data import Dataset, DataLoader, Subset
from scripts.family_id import family_id, has_lithium
from collections import Counter


def main(checkpoint_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = torch.load(f'{data_dir}/dataset_cache.pt', weights_only=False)
    splits = torch.load(f'{data_dir}/split_indices.pt', weights_only=False)
    all_graphs = torch.load(f'{data_dir}/prebuilt_graphs.pt', weights_only=False)
    test_idx = splits['test']
    formulas = [cache['structures'][i].formula for i in test_idx]
    eah_true = np.array(cache['targets']['energy_above_hull'], dtype=float)[test_idx]

    class PD(Dataset):
        def __init__(self, g): self.g = g
        def __len__(self): return len(self.g)
        def __getitem__(self, i): return self.g[i]

    loader = DataLoader(Subset(PD(all_graphs), test_idx), batch_size=16, collate_fn=collate_fn)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = ScandiumPINNGNN(hidden_dim=128, num_alignn_layers=2, num_transformer_layers=1,
                            num_attention_heads=4, dropout=0.15,
                            tasks=['formation_energy', 'energy_above_hull', 'band_gap'],
                            use_two_stage_eah=True).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    preds, ps, mags = [], [], []
    with torch.no_grad():
        for cg, lg in loader:
            cg, lg = cg.to(device), lg.to(device)
            out = model(cg, lg)
            preds.append(out['energy_above_hull'].cpu())
            ps.append(out['p_unstable'].cpu())
            mags.append(out['eah_magnitude'].cpu())
    preds = torch.cat(preds).numpy()
    ps = torch.cat(ps).numpy()
    mags = torch.cat(mags).numpy()

    # ── Metrics ──
    mae = np.mean(np.abs(preds - eah_true))
    r2 = 1 - np.sum((preds - eah_true)**2) / np.sum((eah_true - eah_true.mean())**2)
    print(f'MAE:  {mae:.4f}  R²:  {r2:.4f}')
    print(f'Pred: mean={preds.mean():.4f} min={preds.min():.6f} max={preds.max():.4f}')

    # ── Enrichment@10 ──
    k = 10
    N = len(eah_true)
    pred_topk = set(np.argsort(preds)[:k])
    true_topk = set(np.argsort(eah_true)[:k])
    found = len(pred_topk & true_topk)
    random_exp = k * k / N
    enrich = found / random_exp if random_exp > 0 else 0.0
    print(f'Enrichment@10: {enrich:.2f}x ({found}/{k} in true top-{k})')

    # ── Bucketed MAE ──
    buckets = [
        ('stable (<0.001)',      eah_true < 0.001),
        ('marginal (0.001-0.05)', (eah_true >= 0.001) & (eah_true < 0.05)),
        ('elevated (0.05-0.2)',   (eah_true >= 0.05) & (eah_true < 0.2)),
        ('unstable (>0.2)',       eah_true >= 0.2),
    ]
    print('\n=== Bucketed MAE ===')
    for name, mask in buckets:
        if mask.sum() == 0: continue
        print(f'  {name:25s} n={mask.sum():4d} MAE={np.mean(np.abs(preds[mask] - eah_true[mask])):.4f}')

    # ── Top-10 family + Li check ──
    top10 = np.argsort(preds)[:10]
    print('\n=== Top-10 predicted-stable ===')
    li_count = 0
    for i in top10:
        fam = family_id(formulas[i])
        li = has_lithium(formulas[i])
        li_count += li
        print(f'  {formulas[i]:>25} fam={fam:>12} pred={preds[i]:.4f} true={eah_true[i]:.4f} Li={"yes" if li else "no"}')
    print(f'Li in top-10: {li_count}/{k}')
    top_fams = Counter([family_id(formulas[i]) for i in top10])
    print(f'Family dist: {dict(top_fams)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', required=True)
    args = p.parse_args()
    main(args.checkpoint, args.data_dir)
