# torch.compile Readiness — Scandium Labs

---

## 1. Identified Graph Breaks

`torch.compile` partitions the model graph at graph break points. Each break reduces optimization potential.

| Break Point | File:Line | Severity | Fix |
|---|---|---|---|
| `torch.utils.checkpoint.checkpoint()` | `scandium_model.py:128,137,152` | **High** | Remove GC or use `torch.compile`'s own rematerialization |
| `isinstance(head_out, dict)` | `scandium_model.py:178` | **High** | Refactor heads to return consistent types |
| `if mask.sum() > 0:` (Python conditional on tensor) | `two_stage_eah.py:106,115` | **Medium** | Replace with `mask * loss` pattern |
| `.detach()` in loss forward | `two_stage_eah.py:131-133` | **Low** | Only in loss return dict, not in compute graph |
| `.numpy()` in `two_stage_metrics` | `two_stage_eah.py:139-140` | **Medium** | Replace with torch-native metric computation |
| Dynamic shapes (variable node/edge counts per graph) | `scandium_model.py:152,156` | **Low** | `torch.compile` handles dynamic shapes with guards |
| `for layer in self.alignn_layers` | `scandium_model.py:126` | **Low** | Python loops over modules are traceable |

---

## 2. Fix Priority

### Must Fix Before Compile

1. **Remove gradient checkpointing** — Hard graph break by design. If memory permits, remove GC entirely
2. **Refactor task heads for consistent return types** — Eliminate `isinstance` check

### Should Fix for Best Performance

3. **Replace conditional masks in TwoStageEahLoss** — Prevent unnecessary recompilation
4. **Replace `.detach()` in loss dict** — Use separate tracking tensors

### Nice to Have

5. **Replace `.numpy()` in metrics** — Keep compute on GPU
6. **Mark `for layer in self.layers` with `torch.no_grad()` where applicable** — Already implicit

---

## 3. Expected Benefit

Without any graph break fixes, `torch.compile` would still provide some benefit:
- Kernels inside each ALIGNN layer would be fused
- Python interpreter overhead for `torch.cat`, `Linear`, `SiLU` calls would be reduced
- `MultiheadAttention` would benefit from fused kernels

With all graph breaks fixed:
- Full end-to-end fusion across ALIGNN layers
- Potential 2–3× forward-pass speedup (based on PyG + compile benchmarks from Kumo.ai)
- But: dynamic shapes (variable nodes/edges) limit fusion

**Estimated speedup with compile:** 15–30% forward pass (after fixes), or ~5–10% total step time (since GradNorm dominates).

---

## 4. Recommendation

**Do not enable `torch.compile` until GradNorm is deferred and the model backward is the dominant cost.** Currently GradNorm is 40% of epoch time, so optimizing the forward+backward by 20% is only 8% of epoch time. After GradNorm deferral, forward+backward will be ~62% of epoch, making compile optimization more impactful.

**Prerequisite order:**
1. ✓ Data caching (done)
2. ✓ DataLoader optimization (workers=3 benchmarked)
3. GradNorm deferral (highest priority)
4. Remove redundant ops (clone, .float(), .item())
5. Fix graph breaks for compile readiness
6. Enable `torch.compile`
