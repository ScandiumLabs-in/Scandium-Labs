# Evaluation Results

This directory previously contained `test_results.json` from an untracked training run.

## Traceable Results

All experiment results are tracked in `runs/index.csv` and can be found in the corresponding run directories under `runs/`.

### Best tracked run — SL-20260701-007 (GradNorm ON, no scheduler, 150 epochs)

Test set evaluation (1,104 samples):

| Task | MAE | R² | Stability F1 |
|------|-----|----|--------------|
| Formation Energy | 0.327 eV/atom | 0.553 | — |
| Energy Above Hull | 0.103 eV/atom | 0.184 | 0.954 |
| Band Gap | 1.249 eV | 0.037 | — |

Full results: `runs/SL-20260701-007/test_results.json`

### Ablation: SL-20260708-001 (no GradNorm + CosineAnnealingWarmRestarts, 138 epochs)

| Task | MAE | R² | Stability F1 |
|------|-----|----|--------------|
| Formation Energy | 0.315 eV/atom | 0.512 | — |
| Energy Above Hull | 0.097 eV/atom | 0.177 | 0.954 |
| Band Gap | 1.234 eV | 0.069 | — |

Full results: `runs/SL-20260708-001/test_results.json`

## Generating Test Results

```bash
python scripts/evaluate/evaluate.py \
  --checkpoint runs/SL-YYYYMMDD-NNN/checkpoints/best_val_loss.pt \
  --dataset datasets/v3_li_10000 \
  --output results.json
```
