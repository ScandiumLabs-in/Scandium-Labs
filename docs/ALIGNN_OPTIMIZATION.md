# ALIGNN Optimization — Scandium Labs

Files examined: `src/models/gnn/alignn.py`, `src/models/gnn/layers.py`, `src/models/scandium_model.py`, `src/models/heads/two_stage_eah.py`

---

## 1. Redundant Edge Feats Clone

**File:** `scandium_model.py:119`

```python
lg_feats = edge_feats.clone()
```

- Full tensor copy of `[num_edges, 64]` every forward pass
- The clone is **never modified** — `lg_feats` is passed into `ALIGNNLayer.lg_conv` as `lg_node_feats`, but the line graph conv never writes back to it
- Only `updated_edge_feats` is updated (as return value from `alignn.py:22`)

**Fix:** Replace with `lg_feats = edge_feats` (shared reference).

---

## 2. Eager Default Evaluation in Dict `.get()`

**File:** `scandium_model.py:181-189`

```python
log_var = head_out.get("log_var", self.uncertainty_heads[task](graph_feats).squeeze(-1))
```

- Python evaluates the default argument **eagerly** before `.get()` is called
- When `TwoStageEahHead` is active, `head_out` contains `"log_var"`, so the default is never used
- But the full MLP forward `self.uncertainty_heads[task](graph_feats).squeeze(-1)` executes **every forward pass** anyway
- Same pattern for `predictions[task].clone()` on line 189

**Fix:** Replace with conditional check or `None`-sentinel pattern.

---

## 3. Redundant `.float()` Casts

**File:** `train_v3_li.py:209-214`

```python
eah_out = {
    "eah_pred": preds["energy_above_hull"].float(),    # already float32
    "p_unstable": preds["p_unstable"].float(),          # already float32
    "eah_magnitude": preds["eah_magnitude"].float(),    # already float32
}
```

- Model outputs are `float32` (all `nn.Linear` layers output float32)
- `.float()` is a no-op that still invokes CUDA kernel scheduler

**Fix:** Remove `.float()` calls.

---

## 4. `unsqueeze(0)` / `squeeze(0)` Thrashing for Transformer

**File:** `scandium_model.py:152-156`

```python
node_feats.unsqueeze(0)  # → [1, N, D]
...
.squeeze(0)              # → [N, D]
```

- Runs 2× per forward (once per transformer layer)
- Inside gradient checkpointing, these run 4× (recomputed in backward)
- Forces shape changes for `nn.MultiheadAttention` batch_first=False

**Fix:** Single unsqueeze before transformer loop, single squeeze after.

---

## 5. Subnetwork Redundancy in TwoStageEahHead

**File:** `two_stage_eah.py:60-64`

Three separate MLPs on the same input `x`:
- `stability_head`: 3 Linear layers (128→64→32→1)
- `eah_magnitude_head`: 3 Linear layers (128→64→32→1)
- `uncertainty_head`: 2 Linear layers (128→32→1)
- Total: 8 Linear layers, all on the same 128-dim input

**Fix:** Share an initial projection layer, reducing to ~6 layers:
```python
shared = F.silu(self.shared_proj(x))  # Linear(128, 64)
p_unstable = self.stability_final(shared)
eah_magnitude = self.magnitude_final(shared)
log_var = self.uncertainty_final(shared)
```
Savings: ~22% FLOPs reduction for this head (~23K params → ~18K).

---

## 6. Scalar Tensor Allocation in TwoStageEahLoss

**File:** `two_stage_eah.py:105, 114`

```python
reg_loss = torch.tensor(0.0, device=eah_true.device)
stable_loss = torch.tensor(0.0, device=eah_true.device)
```

- Allocates CUDA scalar tensors even when masks are non-empty
- Immediately overwritten in the conditional branches

**Fix:** Initialize to `None`; handle in `total` computation.

---

## 7. Graph Break: Python `isinstance` Check

**File:** `scandium_model.py:178`

```python
if isinstance(head_out, dict):
```

- Data-dependent control flow prevents `torch.compile` from tracing through
- TwoStageEahHead returns dict; standard heads return tensor

**Fix:** Refactor heads to return consistent types (dict with `"pred"` key).

---

## 8. Graph Break: Conditional Loss Masks

**File:** `two_stage_eah.py:106, 115`

```python
if unstable_mask.sum() > 0:
if stable_mask.sum() > 0:
```

- Python-level conditionals on tensor data
- Prevent `torch.compile` fusion

**Fix:** Always compute losses; use `mask * loss` pattern with automatic zeroing.

---

## Priority Ranking

| Priority | Optimization | Est. Speedup | Risk | Effort |
|---|---|---|---|---|
| P0 | Remove `edge_feats.clone()` | 1–2% forward | None | 1 line |
| P0 | Remove `.float()` casts | <1% | None | 3 lines |
| P0 | Fix eager `.get()` defaults | 2–3% forward | Low | 10 lines |
| P1 | Share TwoStageEahHead initial projection | 1–2% forward | Low | 15 lines |
| P1 | Fix TwoStageEahLoss scalar tensors | <1% | None | 3 lines |
| P2 | Single unsqueeze/squeeze for transformer | <1% | Low | 4 lines |
| P2 | Refactor heads for consistent return type | Prep only | Moderate | 30 lines |
| P2 | Replace conditional masks with `mask*value` | Prep only | Low | 6 lines |
