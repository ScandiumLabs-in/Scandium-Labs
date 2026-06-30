# torch.compile Graph Break Report

**Date:** 2026-06-30
**PyTorch:** 2.12.0+cu130
**GPU:** NVIDIA GTX 1650
**Model:** ScandiumPINNGNN (1.28M params)

---

## 1. Eval Mode — fullgraph=True SUCCEEDS

The core model forward pass (`encode` + `pool` + task heads) compiles with **zero graph breaks** under `fullgraph=True`. This is excellent — the underlying architecture is `torch.compile`-compatible.

## 2. Training Mode — One Graph Break

During training (`model.train()`), a single graph break occurs:

**Location:** `torch_geometric/utils/_softmax.py:83` → `maybe_num_nodes()`

```python
return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
```

The `int()` call on a GPU tensor is a `.item()` that `torch.compile` cannot trace. This is in PyG's `softmax` function called by `AttentionGlobalPool`.

## 3. Fix Options

### Option A: Set capture_scalar_outputs (Recommended)

```python
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
```

PyTorch 2.12 supports this. The warning explicitly recommends it. This allows `int(tensor)` and `tensor.item()` to be captured in the compiled graph without breaking.

### Option B: Replace PyG softmax with torch-native

```python
# Instead of:
from torch_geometric.utils import softmax
gates = softmax(gates, batch)

# Use:
gates = gates - gates.max()
gates = gates.exp() / scatter(gates.exp(), batch, dim=0, reduce='sum').index_select(0, batch)
```

This avoids the `maybe_num_nodes` call entirely but requires testing for numerical equivalence.

## 4. Speed Benchmark

| Mode | Forward ms/step | vs Eager |
|---|---|---|
| Eager | 503.8 | 1.0× |
| `compile(reduce-overhead)` | 435.2 | **+13.6%** |
| `compile(max-autotune)` | timed out (300s) | — |

**13.6% forward speedup** with no code changes. With the graph break fix, performance may improve further.

The backward pass is not compiled here — `torch.compile` currently optimizes forward only. In the training loop with backward, the expected total step speedup is **~5-8%** (since forward is ~28% of step time, and 13.6% of 28% ≈ 3.8%, plus some additional fusion benefits).

## 5. Recommendation

Apply the `capture_scalar_outputs` fix and benchmark again. Then decide whether to enable `torch.compile` by default:

- **Default: off** — compile adds overhead (cold start, recompilation on shape changes) and the 5-8% speedup is modest
- **Enable if the 3.8% total speedup matters** — low risk, no scientific impact
- **Stronger case after GradNorm rewrite** — when GradNorm drops from 40% to ~15%, the forward/backward becomes a larger fraction of total time, making compile more impactful
