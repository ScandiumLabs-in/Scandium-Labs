# Graph Pipeline Analysis — Scandium Labs

## Data Flow

```
build_dataset.py ──► dataset_cache.pt  (structures + targets)
                        │
                        ▼
                 cache_graphs.py ──► graphs/{idx}.pt  (prebuilt crystal + line graphs)
                        │
                        ▼
                 LazyGraphDataset.__getitem__()
                        │  torch.load(path, map_location="cpu")
                        ▼
                 collate_fn() ──► Batch.from_data_list() × 2
                        │
                        ▼
                 train loop: cg.to(device), lg.to(device)
```

---

## Cache State

- 10,000 `.pt` files in `datasets/v3_li_10000/graphs/`
- Avg file: 1.3 MB, Median: 1.1 MB, Total: 12.7 GB
- Max single file: 7.7 MB (large unit cell with 500+ atoms)

### Per-Graph Storage Breakdown

| Component | Shape | Size | % of File |
|---|---|---|---|
| Node features | [N, 92] | ~32 KB | 0.8% |
| Edge index | [2, E] | ~23 KB | 0.6% |
| Edge attr (RBF) | [E, 64] | 360 KB | 9.4% |
| Edge vec | [E, 3] | 17 KB | **UNUSED** |
| Distances | [E] | 6 KB | **UNUSED** |
| Positions | [N, 3] | 1 KB | **UNUSED** |
| Line graph x | [E, 64] | 360 KB | **DUPLICATE of edge_attr** |
| Line graph edge_index | [2, A] | 338 KB | 8.8% |
| Line graph edge_attr | [A, 32] | 2,700 KB | **70.4%** (dominant) |

**Key finding:** Line graph accounts for ~89% of disk footprint. Angle adjacency (A=21,600 for a 90-atom graph) is 15× larger than original edges (E=1,440).

**Disk waste:** ~3.5 GB from duplicate data + unused fields.

---

## DataLoader Benchmark (Updated)

Benchmark from `scripts/maintenance/benchmark_dataloader_v2.py`

| Workers | Best Config | Throughput (graphs/s) |
|---|---|---|
| 0 | PF=4, PM=False | 265.9 |
| 1 | PF=4, PM=False, PW=True, fork | 333.1 |
| 2 | PF=None, PM=True, PW=True, fork | 511.8 |
| **3** | **PF=2, PM=True, PW=True, fork** | **543.2** |
| 4 | PF=4, PM=True, PW=True, fork | 442.9 |

**Optimal: workers=3, prefetch_factor=2, pin_memory=True, persistent_workers=True, fork**

**Current config:** workers=4, PF=2, PM=True, PW=True — suboptimal by ~100 graphs/s.

---

## Redundant Storage

| Item | Size | Reason |
|---|---|---|
| `prebuilt_graphs.pt` | 13 GB | Duplicates individual `graphs/` files. Should be deleted after sharding. |
| `edge_vec`, `distances`, `pos` per graph | ~24 KB each (~240 MB total) | Never read by the model |
| `line_graph.x = edge_attr` duplicate | 360 KB per graph (~3.5 GB total) | Could reconstruct at load time |

**Total reclaimable: ~16.7 GB**

---

## Optimization Recommendations

1. **Change DataLoader to workers=3** — saves ~100 graphs/s (train loop parameter)
2. **Purge `prebuilt_graphs.pt`** — saves 13 GB disk
3. **Remove unused fields from graph cache** — saves 240 MB (if cache is ever rebuilt)
4. **Don't store line_graph.x in cache** — reconstructs it as `edge_attr` clone, saves 3.5 GB
5. **Add `mmap` option for line graph** — lazy-loads large angle tensors only when needed
