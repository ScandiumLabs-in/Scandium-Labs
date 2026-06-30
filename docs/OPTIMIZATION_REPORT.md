# Performance Optimization Report

## System
- GPU: NVIDIA GTX 1650 (4 GB VRAM, Turing architecture, no Tensor Cores)
- RAM: 14 GB (10 GB baseline, 4 GB available)
- Python: 3.14
- PyTorch: 2.8+

## Bottleneck Analysis

Sorted by impact (highest first):

| Priority | Issue | Impact | Status |
|----------|-------|--------|--------|
| 1 | DataLoader workers=0 (no parallelism) | 58% slower throughput | **Fixed** — workers=4 (13.2 vs 5.7 graphs/s) |
| 2 | Graph building on-the-fly (91% of first epoch) | ~29 min overhead | **Caching** — building 8133 remaining graphs |
| 3 | MSELoss created per-batch | Python overhead | **Fixed** — moved to instance var |
| 4 | Gradient accumulation opportunity | VRAM-limited | **Enabled** — batch=16, accum=2 |
| 5 | Small model underutilizes GPU | 177 MB VRAM (4.3%) | **Scaled** — 1.28M params, 470 MB VRAM |

## Current Throughput

**Config:** hidden_dim=128, 4x ALIGNN layers, 2x Transformer, batch=16, accum=2, GC=enabled

| Metric | Value |
|--------|-------|
| Parameters | 1,281,321 |
| Model size (fp32) | 4.9 MB |
| Step time (with GC) | 1,253 ms |
| Step time (no GC) | 943 ms |
| Throughput (with GC) | 12.8 graphs/s |
| Throughput (no GC) | 17.0 graphs/s |
| Peak VRAM (with GC) | 470 MB (11.5%) |
| Peak VRAM (no GC) | 1,127 MB (27.5%) |

GC saves **2.4x VRAM** at **33% speed cost** — well worth it on 4 GB card.

## Remaining Bottlenecks (post-cache)

Once all 10k graphs are cached, DataLoader will be near-instant (disk reads). The bottleneck shifts to GPU compute:

- **1,253 ms/step** = 28% forward + 67% backward + 5% optimizer
- `torch.compile` may reduce forward pass (28% of step) by ~30-50%
- With `torch.compile`, estimated step time: ~1,050 ms → ~15.3 graphs/s

## Next Optimizations

1. **`torch.compile`** — evaluate after cache complete (forward pass reduction)
2. **Optuna HPO** — hyperparameter search (lr, dropout, hidden_dim, weight_decay)
3. **CUDA graphs** — reduce kernel launch overhead for fixed-size batches
4. **Mixed precision refinements** — ensure `grad_scaler` is optimal
5. **Architecture search** — compare GCN vs GAT vs ALIGNN for this dataset

## Resource Profiles

See `RESOURCE_PROFILES.md` for configs targeting different GPU tiers.
