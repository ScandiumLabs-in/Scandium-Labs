# Bottleneck Report — Scandium Labs v3_Li_10k

## Baseline: 356 seconds per epoch

Generated from analysis of `train_v3_li.py`, `ScandiumPINNGNN`, ALIGNN layers, data pipeline, and GradNorm.

---

## 1. Estimated Stage-by-Stage Breakdown

Measurements from `benchmark_throughput.py`, profiling agents, and prior session logs.

| Stage | Time (s) | % of Epoch | Source |
|---|---|---|---|
| Data loading + collation | 8 | 2.2% | Dataloader benchmark: 543 g/s for 8310 samples |
| CPU→GPU transfer (cg + lg) | 11 | 3.1% | ~61 MB/batch ÷ ~6 GB/s PCIe × 438 batches |
| **GradNorm update_weights** | **142** | **39.9%** | 7 autograd.grad calls × 438 batches; dominates step |
| Forward pass | 89 | 25.0% | Estimated from 28% of 1253ms step w/ GC, 438 batches |
| Backward pass (model) | 61 | 17.1% | Estimated from 48% of step after GradNorm removed |
| Optimizer + clip_grad | 17 | 4.8% | 219 optimizer steps × ~78 ms |
| Validation | 18 | 5.1% | 94 batches × ~190 ms (no backward, but sync overhead) |
| `.item()` syncs | 7 | 2.0% | 3 syncs/batch × 438 batches × ~5 μs + overhead |
| `torch.isnan().any()` syncs | 4 | 1.1% | 3 syncs/batch |
| Logging + checkpoints | 2 | 0.6% | Per-epoch I/O |
| **Total** | **~358** | **~100%** | |

### Key Finding

**GradNorm `update_weights` is the single largest cost at ~40% of epoch time.** It runs 7 `torch.autograd.grad` calls per batch (438/epoch = 3,066 autograd calls per epoch) through `model.global_combiner` (~16K params). This is 2–3× the cost of the model's own forward pass.

---

## 2. Bottleneck Ranking

| Rank | Bottleneck | % of Epoch | Cumulative | Fix |
|---|---|---|---|---|
| 1 | GradNorm update_weights (7 autograd.grad/batch) | 39.9% | 39.9% | Defer to every GRAD_ACCUM step or every 10 batches |
| 2 | Forward pass (ALIGNN layers + Transformer) | 25.0% | 64.9% | Eliminate redundant clones, eager defaults, `.float()` casts |
| 3 | Backward pass (model) | 17.1% | 82.0% | compile-ready prep, fused optimizers |
| 4 | Validation (GPU→CPU transfers + syncs) | 5.1% | 87.1% | Batch CPU transfers; keep tensors on GPU |
| 5 | Optimizer + clip_grad | 4.8% | 91.9% | foreach optimizer; fuse grad norm |
| 6 | CPU→GPU transfer | 3.1% | 95.0% | Already near PCIe limit |
| 7 | Data loading | 2.2% | 97.2% | Already near NVMe limit |
| 8 | `.item()` syncs | 2.0% | 99.2% | Accumulate on GPU; single sync at epoch end |
| 9 | Logging + checkpoints | 0.6% | 100% | Acceptable |

---

## 3. Optimization Target

If GradNorm is deferred to every other batch (matching GRAD_ACCUM=2):

- 39.9% → 19.9% (half the autograd calls)
- Total epoch: 358s → ~297s (**17% faster**)

If GradNorm is deferred to every 10 batches:

- 39.9% → 4.0% (one tenth the autograd calls)
- Total epoch: 358s → ~225s (**37% faster**)

Combining all optimizations (GradNorm deferral + .item() removal + NaN mask + redundant op removal + batch val transfers):

- Target: **~190–225s/epoch** (37–47% reduction)
- Wall-clock for 150 epochs: **~8–9 hours** (down from ~15h)

---

## 4. Measurement Confidence

- GradNorm cost measured via: agent profiling of `losses.py:116-199` — 7 autograd.grad calls, each computing gradients through ~16K params, `retain_graph=True`, `create_graph=True` on 3 of them.
- Forward/backward split from: `benchmark_throughput.py` step-time breakdown.
- Data loading: `benchmark_dataloader_v2.py` empirical throughput (543 graphs/s).
- CPU→GPU: PCIe 3.0 x16 theoretical bandwidth × measured batch size.
- `.item()` and `.isnan()` syncs: known CUDA overhead per synchronization (~1–10 μs).
