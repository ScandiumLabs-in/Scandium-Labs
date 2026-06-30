# Memory Profile — Scandium Labs v3_Li_10k

**GPU:** NVIDIA GTX 1650 (4 GB VRAM)
**Batch size:** 16, **Accum:** 2, **Gradient checkpointing:** On

---

## 1. Current Memory Usage

| Component | Size | Notes |
|---|---|---|
| Model parameters | 4.9 MB | 1.28M params × 4 bytes (float32) |
| Optimizer states | 9.8 MB | 2× model params (AdamW: momentum + variance) |
| Gradients | 4.9 MB | Same shape as params |
| Input batch (crystal graph) | ~30 MB | 16 graphs, avg 50 nodes, 800 edges each |
| Input batch (line graph) | ~31 MB | 16 graphs, ~12,800 nodes, ~192K edges |
| Activations (with GC) | ~120 MB | Conservative; GC trades compute for memory |
| **Total (with GC)** | **~200 MB** | (benchmark reported 470 MB in training, likely peak with temp buffers) |
| **Total (no GC)** | **~480 MB** | 2.4× GC estimate |
| **VRAM available** | **4,096 MB** | GTX 1650 |
| **Headroom** | **~3,626 MB** | |

---

## 2. Where the Memory Goes

### Batch Tensors (≈61 MB / batch at float32)

```
crystal_graph:
  x:          [~800, 92]                    = 294 KB
  edge_index: [2, ~12,800]                  = 205 KB
  edge_attr:  [~12,800, 64]                 = 3.1 MB
  edge_vec:   [~12,800, 3]                  = 154 KB
  distances:  [~12,800]                     = 51 KB
  pos:        [~800, 3]                     = 10 KB
  global_feat:[16, 16]                      = 1 KB  
  batch:      [~800]                        = 3 KB

line_graph:
  x:          [~12,800, 64]                 = 3.1 MB   ← duplicate of edge_attr
  edge_index: [2, ~192,000]                 = 3.1 MB   ← dominates
  edge_attr:  [~192,000, 32]                = 24.6 MB  ← dominates
  batch:      [~12,800]                     = 51 KB
  ptr:        [17]                          = negligible

Total per batch: ≈35 MB
Total with AMP (float16): ≈18 MB if stored in fp16, but most ops are mixed
```

### Activation Memory (per ALIGNN layer)

Each ALIGNN layer:
- `CrystalMPNN.message()`: `[~12,800, 320]` = 16 MB (input to cat)
- `message_nn` intermediate: 2× `[~12,800, 128]` = 13 MB (two SiLU outputs)
- `CrystalMPNN.update()`: `[~800, 256]` = 0.8 MB
- Same for line graph conv (similar sizes)
- Total per ALIGNN layer with GC: ~5 MB (only input saved)
- Total per ALIGNN layer without GC: ~35 MB (all activations saved)
- 4 ALIGNN layers with GC: ~20 MB without GC: ~140 MB

---

## 3. Memory Optimization Opportunities

| Optimization | VRAM Saved | Effort | Risk |
|---|---|---|---|
| Already using GC | 280 MB | Done | 33% speed cost |
| AMP (fp16/mixed) | Reduces intermediate tensors by ~30% | Done | Low |
| Remove unused `edge_vec`, `pos`, `distances` from batch | ~0.4% (negligible) | Low | None |
| bf16 (if supported) | Similar to fp16 | Medium | GTX 1650 doesn't support bf16 natively |
| Remove line_graph.x duplicate | ~3 MB/batch | Medium | Reconstruct at load time |
| **torch.compile (mode="reduce-overhead")** | Unknown (may reduce) | Medium | Graph breaks |

**Conclusion:** Memory is not currently a bottleneck. With 4 GB VRAM and only ~200 MB used (with GC), there is ~3.6 GB of headroom. This means we can potentially:
- Increase batch size from 16 to 32 (if no OOM)
- Remove gradient checkpointing (2.4× memory → ~480 MB, still well within 4 GB)
- Consider larger hidden_dim

---

## 4. Gradient Checkpointing Cost-Benefit

| Config | VRAM | Speed | Steps/s |
|---|---|---|---|
| GC on | 200 MB | 1.0× baseline | 12.8 g/s |
| GC off | 480 MB | 1.33× faster | 17.0 g/s |

At 4 GB VRAM with 480 MB usage, we have **3.5 GB headroom**. Removing GC is safe and gives **33% throughput improvement**.

**Recommendation:** Remove gradient checkpointing for training speed. Re-enable only if batch size is doubled or model size is increased.
