# Refactoring Plan (Phase A → Complete)
# Performance Optimization (Phase B → In Progress)

## Key Decisions
- `src/training/trainer.py` → split into: `trainer.py`, `loaders.py`, `distributed.py`, `pretrained.py`, `scheduler.py`
- `src/training/engine.py` → split into: `engine.py`, `recommend.py`, `coverage.py`, `activation.py`
- `src/models/losses.py` → 5 unused loss classes removed (160 lines)
- All imports standardized to absolute package imports across 10 files
- All 5 `src/` packages have proper `__init__.py` exports (28 public symbols)
- Evidence-before-changes mandate: all optimizations follow from real profiling data
- DataLoader `multiprocessing_context='fork'` required on Python 3.14 + CUDA
- Gradient checkpointing saves 2.4x VRAM at 33% speed cost
- Cache build runs single-process (6.1 graphs/s) on CPU to avoid CUDA multiprocessing issues

## Phase B Status

### Completed
- Profiling: model = 1.28M params, 4.9 MB, 470 MB VRAM, 12.8 graphs/s with GC
- DataLoader benchmark: workers=4 gives 13.2 graphs/s vs 5.7 graphs/s (132% faster)
- Config upgrade: hidden_dim=128, 4x ALIGNN layers, 2x Transformer, batch=16, accum=2
- `MSELoss` moved from per-batch creation to `ScandiumTrainer.__init__` instance var
- `use_gradient_checkpointing=true` in config
- `pin_memory=True`, `multiprocessing_context='fork'` added to DataLoaders
- `train_v3_li.py` updated with optimized config
- `docs/OPTIMIZATION_REPORT.md` — bottleneck analysis with before/after metrics
- `docs/RESOURCE_PROFILES.md` — Small/Medium/Large config templates
- Cache builder: single-process CPU builder running reliably (PID 87119, 6.1 graphs/s)

### Running
- Cache build: 2295/10000 → expected completion in ~21 min
- `nohup setsid ./venv/bin/python -u scripts/preprocess/cache_graphs.py > /tmp/cache_graphs.log 2>&1 &`

### Deferred (post-cache)
- Full training run with optimized config (workers=4, pin_memory, GC, larger model)
- `torch.compile` evaluation for forward-pass speedup
- Optuna hyperparameter search (lr, dropout, hidden_dim, weight_decay)
- CUDA graphs for reduced kernel launch overhead
- Architecture comparison (GCN vs GAT vs ALIGNN)

## Key Metrics
- Before: 5.7 graphs/s (DataLoader bottleneck), 728K params, 177 MB VRAM
- After: 12.8 graphs/s (GPU-bound), 1.28M params, 470 MB VRAM
- Cache eliminates 29-min first-epoch overhead
- GC tradeoff: 33% speed for 2.4x VRAM savings (essential on 4 GB GPU)

## Relevant Files
- `scripts/preprocess/cache_graphs.py` — single-process graph cache builder
- `scripts/maintenance/benchmark_throughput.py` — throughput benchmark with GC comparison
- `scripts/maintenance/profile_training.py` — torch profiler + parameter/memory profiling
- `scripts/maintenance/profile_dataloader.py` — DataLoader num_workers benchmark
- `configs/model_config_v3_li.yaml` — optimized config (hidden_dim=128, 4 layers, GC=true)
- `scripts/train/train_v3_li.py` — training script with optimized DataLoader
- `docs/OPTIMIZATION_REPORT.md` — bottleneck analysis and throughput data
- `docs/RESOURCE_PROFILES.md` — resource profiles for different GPU tiers
