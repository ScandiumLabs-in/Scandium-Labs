# Training Speedup Plan — Scandium Labs

**Goal:** Reduce wall-clock training time by 30–70% while maintaining identical or better validation performance.

**Status:** Baseline training running (PID 109865, SL-20260630-002, ~358s/epoch)

---

## Phase 0: Bottleneck Analysis ✓ (Complete)

**Key measurement:** GradNorm `update_weights` = **40% of epoch time** (7 autograd.grad calls/batch)

| Bottleneck | Time/epoch | % |
|---|---|---|
| GradNorm update_weights | 142s | 40% |
| Forward pass | 89s | 25% |
| Backward pass | 61s | 17% |
| Validation | 18s | 5% |
| Optimizer | 17s | 5% |
| CPU→GPU transfer | 11s | 3% |
| Data loading | 8s | 2% |
| Other (syncing, logging) | 13s | 4% |

---

## Phase 1: Immediate Code Changes (Apply Now — Safe)

These changes are safe to apply to the codebase immediately. They have no scientific impact and require no re-training.

| # | Change | Est. Speedup | Files | Lines |
|---|---|---|---|---|
| 1 | Defer GradNorm `update_weights` to match GRAD_ACCUM | **20–25%** | `train_v3_li.py` | 4 |
| 2 | Remove `.item()` syncs; accumulate on GPU | **2–3%** | `train_v3_li.py` | 6 |
| 3 | Precompute NaN masks | **1–2%** | `train_v3_li.py` | 10 |
| 4 | Batch validation CPU transfers | **2–3%** | `train_v3_li.py` | 8 |
| 5 | Remove `edge_feats.clone()` | 1–2% | `scandium_model.py` | 1 |
| 6 | Fix eager `.get()` defaults | 2–3% | `scandium_model.py` | 10 |
| 7 | Remove redundant `.float()` casts | <1% | `train_v3_li.py` | 3 |
| 8 | Update DataLoader to workers=3 | 1–2% | `train_v3_li.py` | 1 |

**Total estimated speedup: 30–40%**

---

## Phase 2: Architectural Changes (After Baseline)

| # | Change | Est. Speedup | Risk |
|---|---|---|---|
| 9 | Remove gradient checkpointing (3.5 GB headroom) | 15–20% epoch | Low; verify VRAM fits |
| 10 | Share TwoStageEahHead initial projection | 1–2% | Low |
| 11 | Replace `unsqueeze/squeeze` loop with single pair | <1% | Low |
| 12 | Implement `torch.compile` after fixing graph breaks | 10–20% forward | Moderate; needs testing |
| 13 | Replace `dict.get()` in task head loop with conditional | 1–2% | Low |

---

## Phase 3: Learning Efficiency (After Baseline Converges)

| # | Change | Est. Speedup | Risk |
|---|---|---|---|
| 14 | Cosine warmup scheduler | Fewer epochs needed | Low |
| 15 | EMA on model weights | Same epochs, better final | Low |
| 16 | Larger effective batch (batch=32, accum=1) | 10–15% throughput | Low (VRAM allows) |

---

## Implementation Order

1. Apply Phase 1 changes to `train_v3_li.py` and supporting files ← **NOW**
2. Wait for baseline (PID 109865) to complete or test changes on a short validation run
3. Launch optimized training run with Phase 1 + Phase 2 changes
4. When that converges, apply Phase 3 learning efficiency improvements
5. Profile again, measure actual speedup vs baseline
6. Evaluate `torch.compile` readiness and apply

---

## Expected Results

| Metric | Baseline (SL-20260630-002) | Optimized | Improvement |
|---|---|---|---|
| Epoch time | ~358s | ~200–250s | 30–44% |
| Time to 150 epochs | ~15h | ~8–10h | 33–47% |
| Val MAE Ef | TBD | Same or better | — |
| Val MAE EaH | TBD | Same or better | — |
| Val MAE BG | TBD | Same or better | — |
