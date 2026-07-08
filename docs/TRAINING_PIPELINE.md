# Training Pipeline — Scandium Labs

**Primary script:** `scripts/train/train_v3_li.py`
**Config:** `configs/model_config_v3_li.yaml`
**Dataset:** v3_li_10000 (10,000 Li-containing structures)
**Status:** Experiment B currently running (GradNorm ON + CosineAnnealingWarmRestarts)

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  build_dataset.py                                                        │
│  • Downloads from Materials Project                                      │
│  • Filters Li ≥ 5 at.%                                                    │
│  • Cleans + splits (80/10/10)                                            │
│  • Normalizes targets                                                     │
│  Output: dataset_cache.pt, split_indices.pt, normalizer.json             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  cache_graphs.py (single-process CPU)                                    │
│  • Builds graphs for each structure                                      │
│  • Saves as individual graphs/{idx}.pt files                             │
│  • 6.1 graphs/s throughput                                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  train_v3_li.py                                                          │
│                                                                           │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐                │
│  │ LazyGraph    │ → │ DataLoader   │ → │ Scandium        │                │
│  │ Dataset       │   │ (workers=4)  │   │ PINNGNN         │                │
│  └─────────────┘   └──────────────┘   └───────┬────────┘                │
│                                                │                          │
│                                                ▼                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  Training Loop (manual, per-epoch)                                 │   │
│  │                                                                     │   │
│  │  foreach batch:                                                    │   │
│  │    forward(AMP) → task_losses → GradNorm → backward(AMP)          │   │
│  │    if accum_steps reached: unscale → clip → optimizer.step()       │   │
│  │                                                                     │   │
│  │  validation → metrics → ExperimentTracker → checkpoint             │   │
│  │                                                                     │   │
│  │  scheduler.step() → early_stop check → repeat                      │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  Output: checkpoints/*.pt, runs/SL-YYYYMMDD-NNN/*                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entry Point: `main()`

**File:** `scripts/train/train_v3_li.py:52`

### 2.1 Argument Parsing

```bash
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \   # Model + training config
    --resume path/to/checkpoint.pt \              # Resume from checkpoint
    --data-dir datasets/v3_li_10000 \             # Dataset path
    --out-dir checkpoints/v3_li_10k_fresh \       # Output directory
    --no-gradnorm                                  # Disable GradNorm (use fixed weights)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/model_config_v3_li.yaml` | YAML config path |
| `--resume` | `None` | Checkpoint path for resuming |
| `--data-dir` | `datasets/v3_li_10000` | Dataset directory |
| `--out-dir` | `checkpoints/v3_li_10k_fresh` | Output directory |
| `--no-gradnorm` | `False` | Disable GradNorm adaptive weighting |

### 2.2 Initialization Sequence

```
1. Parse args
2. Load config YAML
3. Detect device (CUDA/CPU)
4. Load dataset_cache.pt + split_indices.pt
5. Initialize ALIGNNGraphBuilder + FeatureEngineer
6. Build LazyGraphDataset
7. Precompute graph sizes for bucketing
8. Build DataLoaders (train, val, test)
9. Initialize ScandiumPINNGNN
10. Initialize ExperimentTracker
11. Initialize loss functions + GradNorm
12. Initialize optimizer + scheduler
13. Restore state if resuming
14. Enter training loop
```

---

## 3. Data Loading

### 3.1 Dataset Files

```
datasets/v3_li_10000/
├── dataset_cache.pt          # {"structures": list, "targets": dict}
├── split_indices.pt          # {"train": [...], "val": [...], "test": [...]}
├── normalizer.json           # Per-task mean, std, min, max
├── metadata.json             # Version, stats
├── dataset_report.json       # Per-task distribution report
└── graphs/                   # Pre-cached graph files
    ├── 0.pt                  # (crystal_graph, line_graph) pickle
    ├── 1.pt
    └── ... (up to 9999.pt)
```

### 3.2 LazyGraphDataset

**File:** `src/data/dataset.py:60`

```python
full_dataset = LazyGraphDataset(
    structure_list=cache["structures"],
    targets=cache["targets"],
    graph_dir=graph_dir,           # Path to graphs/*.pt
    graph_builder=builder,          # ALIGNNGraphBuilder (fallback if graph missing)
    feature_engineer=fe,            # FeatureEngineer
    cache_dir=graph_dir,            # Where to save newly built graphs
)
```

Loading priority:
1. **Memory cache** — `self._cache[idx]` if previously loaded
2. **Disk cache** — `graphs/{idx}.pt` if file exists
3. **On-the-fly build** — `builder.build(structure)` as fallback (saves to disk)

### 3.3 SizeBucketedBatchSampler

**File:** `src/data/samplers.py:28`

Groups graphs by size (num_nodes + line_graph_nodes) to minimize padding:

```python
class SizeBucketedBatchSampler:
    def __init__(self, indices, sizes, batch_size=16, bucket_size_mult=2.0):
        # Sort by size
        sorted_indices = sorted(zip(sizes, indices))
        # Partition into buckets (capacity = batch_size × bucket_size_mult)
        # Each bucket contains similarly-sized graphs
        # Within each bucket: shuffle + draw contiguous chunks of batch_size
```

```
Example: batch_size=16, bucket_size_mult=2.0

Sorted graphs by size:
[0, 1, 2, 3, ..., 15, 16, 17, ..., 31, 32, 33, ...]
│<─── bucket 0 (32 graphs) ──>│<── bucket 1 (32) ──>│<── ...

Within bucket 0: shuffle → [5, 12, 3, 0, ..., 15] → chunk into batches of 16
```

Benefits:
- Reduces padding waste from size heterogeneity
- Improves GPU utilization (fewer wasted FLOPs on padding)
- Enables larger effective batch sizes on memory-constrained GPUs

### 3.4 DataLoader Configuration

```python
loader_kwargs = dict(
    collate_fn=collate_fn,              # Batch.from_data_list (PyG)
    pin_memory=True,                    # Faster CPU→GPU transfer
    multiprocessing_context="fork",     # Required for Python 3.14 + CUDA
)

train_loader = DataLoader(
    full_dataset,
    batch_sampler=batch_sampler,        # SizeBucketedBatchSampler
    num_workers=4,                      # Parallel data loading
    **loader_kwargs,
)

val_loader = DataLoader(
    Subset(full_dataset, split["val"]),
    batch_size=16,
    shuffle=False,
    num_workers=4,
    **loader_kwargs,
)

test_loader = DataLoader(
    Subset(full_dataset, split["test"]),
    batch_size=16,
    shuffle=False,
    num_workers=4,
    **loader_kwargs,
)
```

| Setting | Value | Rationale |
|---------|-------|-----------|
| `num_workers` | 4 | Parallel data loading (2.3× speedup vs workers=0) |
| `pin_memory` | True | Enables async GPU transfer |
| `multiprocessing_context` | `fork` | Required for Python 3.14 CUDA compatibility |
| `batch_sampler` | SizeBucketedBatchSampler | Groups similarly-sized graphs |

### 3.5 DataLoader Throughput Comparison

| Workers | Context | Throughput | vs Baseline |
|:-------:|:-------:|:----------:|:-----------:|
| 0 | N/A | 5.7 graphs/s | 1.0× (baseline) |
| 4 | `fork` | **13.2 graphs/s** | **2.3× faster** |
| 4 | `spawn` | — | Slower (CUDA incompatibility) |

Benchmark from `scripts/maintenance/profile_dataloader.py`.

---

## 4. Dataset Split

| Split | Count | Ratio |
|:-----:|:-----:|:-----:|
| Train | 8,000 | 80% |
| Validation | 1,000 | 10% |
| Test | 1,000 | 10% |
| **Total** | **10,000** | **100%** |

**File:** `split_indices.pt` (pre-computed by `build_dataset.py`)

---

## 5. Training Loop

### 5.1 Complete Loop Flowchart

```
┌──────────────────────┐
│  FOR epoch in range  │
│  (start, MAX_EPOCHS) │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  tracker.start_epoch │
│  model.train()       │
│  optimizer.zero_grad │
└──────────┬───────────┘
           ▼
    ┌──────────────┐     ┌───────────────────────┐
    │ FOR batch in │────→│ cg, lg = batch         │
    │ train_loader │     │ cg, lg → device        │
    └──────────────┘     └──────────┬────────────┘
           ▲                        ▼
           │              ┌───────────────────────┐
           │              │ AMP autocast           │
           │              │ preds = model(cg, lg)  │
           │              └──────────┬────────────┘
           │                        ▼
           │              ┌───────────────────────┐
           │              │ Compute task_losses    │
           │              │ • ef: MSE              │
           │              │ • eah: TwoStageEahLoss │
           │              │ • bg: MSE              │
           │              └──────────┬────────────┘
           │                        ▼
           │              ┌───────────────────────┐
           │              │ GradNorm (every 50)   │
           │              │ total_loss =           │
           │              │   Σ w_t · L_t / accum  │
           │              └──────────┬────────────┘
           │                        ▼
           │              ┌───────────────────────┐
           │              │ scaler.scale(total)    │
           │              │ total_loss.backward()  │
           │              └──────────┬────────────┘
           │                        ▼
           │              ┌───────────────────────┐
           │    NO        │ batch_idx % accum == 0│───YES──┐
           │    ◄─────────└───────────────────────┘       │
           │                                               ▼
           │                                    ┌───────────────────────┐
           │                                    │ scaler.unscale_(opt)  │
           │                                    │ clip_grad_norm_(1.0)  │
           │                                    │ scaler.step(optimizer)│
           │                                    │ scaler.update()       │
           │                                    │ optimizer.zero_grad() │
           │                                    └───────────────────────┘
           │                                               │
           └───────────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────┐
│  End of epoch                                            │
│                                                           │
│  Validation: model.eval() → val_total_loss + metrics     │
│  System: epoch_time, throughput, grad_norm, GPU mem      │
│  GradNorm weights logged                                  │
│                                                           │
│  ExperimentTracker.log_epoch(...)                         │
│  tracker.save_checkpoint(epoch, model, optimizer, ...)    │
│  Save best_model.pt (if val_loss improved)                │
│                                                           │
│  Check early stopping (patience=40)                       │
│  scheduler.step() (if CosineAnnealingWarmRestarts)        │
│                                                           │
│  Console log every 5 epochs                               │
└─────────────────────────────────────────────────────────┘
           ▼
    ┌──────────────┐     YES
    │ epoch < MAX  │────────► next epoch
    │ & not stopped│
    └──────┬───────┘
           │ NO
           ▼
┌──────────────────────┐
│  Test Evaluation      │
│  Load best_model.pt   │
│  Run on test_loader   │
│  Compute per-task     │
│  metrics + two_stage  │
│  Save test_results.json│
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  tracker.finalize()   │
│  • Save metadata      │
│  • Generate plots     │
│  • Write reports      │
│  • Update leaderboard │
└──────────────────────┘
```

### 5.2 Key Hyperparameters

| Parameter | Value | Source |
|-----------|:-----:|--------|
| `hidden_dim` | 128 | `config.model.hidden_dim` |
| `num_alignn_layers` | 4 | `config.model.num_alignn_layers` |
| `num_transformer_layers` | 2 | `config.model.num_transformer_layers` |
| `num_attention_heads` | 4 | `config.model.num_attention_heads` |
| `dropout` | 0.15 | `config.model.dropout` |
| `mc_dropout_samples` | 20 | `config.model.mc_dropout_samples` |
| `use_two_stage_eah` | True | `config.model.use_two_stage_eah` |
| `gradient_checkpointing` | auto | `config.model.use_gradient_checkpointing` |
| `batch_size` | 16 | `config.training.batch_size` |
| `gradient_accumulation_steps` | 2 | `config.training.gradient_accumulation_steps` |
| `effective_batch_size` | 32 | `batch_size × accum_steps` |
| `learning_rate` | 0.0005 | `config.training.learning_rate` |
| `weight_decay` | 1e-5 | Hardcoded in `train_v3_li.py:235` |
| `max_epochs` | 150 | `config.training.max_epochs` |
| `patience` | 40 | `config.training.patience` |
| `gradient_clip` | 1.0 | `torch.nn.utils.clip_grad_norm_` |
| `scheduler` | CosineAnnealingWarmRestarts | `config.training.scheduler` |
| `scheduler.T_0` | 10 | Hardcoded in `train_v3_li.py:243` |
| `scheduler.T_mult` | 2 | Hardcoded in `train_v3_li.py:243` |
| `scheduler.eta_min` | 1e-6 | Hardcoded in `train_v3_li.py:243` |
| `mixed_precision` | fp16 | `torch.amp.autocast` + `GradScaler` |
| `num_workers` | 4 | Hardcoded in `train_v3_li.py:88` |
| `gradnorm_alpha` | 1.5 | `config.gradnorm.alpha` |
| `gradnorm_lr` | 0.025 | Hardcoded in `train_v3_li.py:352` |
| `gradnorm_update_freq` | 50 batches | Hardcoded in `train_v3_li.py:346` |
| `bucketing_enabled` | True | `config.bucketing.enabled` |
| `bucket_size_mult` | 2.0 | `config.bucketing.bucket_size_mult` |
| `cutoff` | 8.0 Å | `config.graph.cutoff` |
| `max_neighbors` | 16 | `config.graph.max_neighbors` |
| `num_rbf` | 64 | `config.graph.num_rbf` |
| `num_sbf` | 32 | `config.graph.num_sbf` |
| `random_seed` | 42 | Hardcoded in training loop restoration |
| `wandb` | False | `config.logging.wandb` |
| `save_interval` | 10 epochs | `config.logging.save_epoch_checkpoints` |

### 5.3 Optimizer

**Type:** AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-5,
)
```

| Hyperparameter | Value |
|---------------|:-----:|
| $\beta_1$ | 0.9 (default) |
| $\beta_2$ | 0.999 (default) |
| $\epsilon$ | 1e-8 (default) |
| weight_decay | 1e-5 |

### 5.4 Scheduler

**Type:** CosineAnnealingWarmRestarts

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,       # Initial restart period: 10 epochs
    T_mult=2,     # Double the period after each restart
    eta_min=1e-6, # Minimum LR floor
)
```

LR schedule pattern:
```
Epoch  0-10:   cosine decay from 5e-4 to ~1.12e-5
Epoch 10-30:   cosine decay from 5e-4 to ~3.9e-6  (period=20)
Epoch 30-70:   cosine decay from 5e-4 to ~1e-6    (period=40)
Epoch 70-150:  cosine decay from 5e-4 to ~1e-6    (period=80)
```

Alternative: `CosineAnnealingLR` (single cosine decay, no restarts), activated when `scheduler: "cosine"` in config.

### 5.5 Mixed Precision (AMP)

```python
scaler = torch.amp.GradScaler("cuda")
use_amp = scaler is not None

# Forward pass
with torch.amp.autocast("cuda", enabled=use_amp):
    preds = model(cg, lg)
    # ... loss computation ...

# Backward
scaler.scale(total_loss).backward()

# Optimizer step
if (batch_idx + 1) % GRAD_ACCUM == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5.6 Gradient Accumulation

```
Batch size:      16
Accumulation:     2
Effective batch: 32

Time per step: ~1,253 ms (with GC)
Throughput:     ~12.8 graphs/s
```

### 5.7 Early Stopping

```python
if tracker.should_stop(PATIENCE):  # PATIENCE = 40
    print(tracker.early_stop_report(epoch, PATIENCE))
    break
```

Stops when validation loss has not improved for 40 consecutive epochs. Best model saved independently.

---

## 6. Checkpointing & Resume

### 6.1 Checkpoint Contents

**Full checkpoint** (saved via `ExperimentTracker.save_checkpoint()`):

```python
{
    "epoch": int,                     # Training epoch
    "model": state_dict,              # Model weights
    "optimizer": state_dict,          # AdamW state (momentum + variance)
    "val_metrics": dict,              # Per-task validation metrics
    "config": {
        "config": mc,                 # Model config dict
        "gradnorm_weights": dict,     # Current GradNorm weights
        "optimizer": "AdamW",
        "scaler_state": dict,         # GradScaler state
        "rng_states": {
            "torch_cpu": list,        # torch CPU RNG state
            "torch_cuda": list,       # torch CUDA RNG state
            "numpy": tuple,           # numpy RNG state
            "random": tuple,          # Python random RNG state
        },
        "best_val_loss": float,
        "train_samples": int,
        "val_samples": int,
        "test_samples": int,
    },
}
```

**Best model** (simple save for easy loading):

```python
{
    "epoch": int,
    "model": state_dict,
    "val_loss": float,
    "config": mc,
}
```

### 6.2 Checkpoint Files

```
runs/SL-YYYYMMDD-NNN/
├── checkpoints/
│   ├── last.pt                    # Always latest epoch
│   ├── best_val_loss.pt           # Best val_loss checkpoint
│   ├── best_{task}_mae.pt         # Per-task best (if improved)
│   ├── best_{task}_r2.pt          # Per-task best (if improved)
│   ├── epoch_{000}.pt             # Periodic (save_interval=10)
│   ├── epoch_{010}.pt
│   └── ...
├── epoch_metrics.json             # All epoch metrics
├── epoch_metrics.csv              # Flattened metrics
├── config.yaml                    # Training config copy
├── run_metadata.json              # Environment + run info
├── test_results.json              # Final test evaluation
├── plots/
│   ├── loss_curve.png
│   ├── mae_curve.png
│   ├── r2_curve.png
│   ├── lr.png
│   ├── grad_norm.png
│   ├── gradnorm_weights.png
│   ├── throughput.png
│   ├── gpu_memory_mb.png
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   └── calibration.png
├── BEST_MODEL_REPORT.md
├── MODEL_CARD.md
├── TRAINING_SUMMARY.md
├── EXPERIMENT_LEADERBOARD.md
├── STOP_REPORT.md
└── tables/
    ├── benchmark.md
    ├── benchmark.csv
    └── benchmark.tex
```

And legacy format at `checkpoints/`:
```
checkpoints/v3_li_10k_fresh/
├── best_model.pt
└── test_results.json
```

### 6.3 Resume Flow

```bash
python scripts/train/train_v3_li.py --resume runs/SL-20260315-001/checkpoints/last.pt
```

Resume restores:
1. **Model weights** — `model.load_state_dict(resume_data["model"])`
2. **Optimizer state** — momentum buffers, learning rate, etc.
3. **GradScaler state** — scale factor, growth/backoff counters
4. **GradNorm weights** — `log_weights` parameters
5. **RNG states** — torch CPU/CUDA, numpy, random (for deterministic resume)
6. **Best val_loss** — early stopping comparison
7. **Experiment directory** — `resume_run_dir` detects and continues existing run
8. **Epoch counter** — `start_epoch = resume_epoch + 1`

```python
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    # Restore GradScaler
    # Restore GradNorm weights
    # Restore RNG states
    # Restore best_val_loss
    start_epoch = ckpt["epoch"] + 1
```

---

## 7. Experiment Tracking

### 7.1 RunRegistry

**File:** `src/training/experiment_tracker.py:46`

SQLAlchemy-free CSV-based run registry:

```python
registry = RunRegistry(runs_dir="runs")

# Generate unique run ID: SL-{YYYYMMDD}-{NNN}
run_id = registry.allocate_run_id()

# Register new run
registry.register(run_id, {"dataset": "v3_li_10000", ...})

# Update status
registry.update_status(run_id, best_mae_ef=0.123, gpu_hours=5.4)
```

```
runs/index.csv:
run_id,date,dataset,architecture,hidden_dim,alignn_layers,transformer_layers,...
SL-20260315-001,2026-03-15T10:00:00,v3_li_10000,ScandiumPINNGNN,128,4,2,...
```

### 7.2 MetricsStore

**File:** `src/training/experiment_tracker.py:169`

```python
metrics = MetricsStore(run_dir)

# Add epoch metrics
metrics.add_epoch({
    "epoch": 42,
    "train_loss": 0.1234,
    "val_loss": 0.2345,
    "tasks": {
        "formation_energy": {"mae": 0.05, "r2": 0.95, "rmse": 0.08, ...},
        "energy_above_hull": {"mae": 0.02, "r2": 0.85, ...},
        "band_gap": {"mae": 0.15, "r2": 0.90, ...},
    },
    "system": {"lr": 0.0005, "grad_norm": 0.8, "epoch_time_s": 120.0, ...},
    "gradnorm_weights": {"formation_energy": 0.85, "energy_above_hull": 1.20, "band_gap": 0.35},
})

# Query best
best_val, best_ep = metrics.get_best("val_loss")
```

Persisted to:
- `epoch_metrics.json` — Full JSON array
- `epoch_metrics.csv` — Flattened, latest-first for quick inspection

### 7.3 CheckpointManager

**File:** `src/training/experiment_tracker.py:258`

```python
checkpoints = CheckpointManager(run_dir, save_interval=10)

checkpoints.save(epoch, model, optimizer, {
    "val_loss": 0.2345,
    "tasks": {task: metrics},
}, extra={...})
```

Saves:
- `last.pt` — Always the latest epoch
- `epoch_{NNN}.pt` — Every `save_interval` epochs (default 10)
- `best_{metric}.pt` — Per-task best metrics (e.g., `best_formation_energy_mae.pt`)
- `best_val_loss.pt` — Overall best validation loss

### 7.4 ExperimentTracker (Orchestrator)

**File:** `src/training/experiment_tracker.py:510`

```python
tracker = ExperimentTracker(
    config=cfg,
    save_epoch_checkpoints=10,
    enable_plots=True,
    resume_from=resume_run_dir,
)
tracker.register_model(model)

# Per epoch
tracker.start_epoch()
# ... training ...
tracker.log_epoch(epoch, train_loss, val_loss, val_metrics, system, gradnorm_weights)
tracker.save_checkpoint(epoch, model, optimizer, val_metrics, extra)
if tracker.should_stop(patience):
    break

# Finalize
tracker.finalize(test_results=results)
```

### 7.5 Logging Output

**Console (every 5 epochs):**

```
  Epoch  42: train=0.1234 val=0.2345 w=[0.85/1.20/0.35] (3600s)
```

**Per-epoch metrics** logged to `epoch_metrics.json`:

```json
{
  "epoch": 42,
  "train_loss": 0.1234,
  "val_loss": 0.2345,
  "timestamp": "2026-03-15T12:00:00",
  "epoch_time_s": 120.5,
  "system": {
    "lr": 0.0005,
    "grad_norm": 0.8,
    "epoch_time_s": 120.5,
    "throughput": 13.2,
    "gpu_memory_mb": 470.0
  },
  "tasks": {
    "formation_energy": {"mae": 0.0523, "rmse": 0.0812, "r2": 0.9512, "pearson": 0.9754, "spearman": 0.9687, "bias": -0.0023},
    "energy_above_hull": {"mae": 0.0215, "rmse": 0.0421, "r2": 0.8534, "pearson": 0.9241, "spearman": 0.9012, "bias": 0.0012},
    "band_gap": {"mae": 0.1543, "rmse": 0.2312, "r2": 0.9034, "pearson": 0.9501, "spearman": 0.9432, "bias": -0.0101}
  },
  "gradnorm_weights": {
    "formation_energy": 0.8532,
    "energy_above_hull": 1.2014,
    "band_gap": 0.3518
  }
}
```

### 7.6 Generated Plots

- `loss_curve.png` — Train + val loss trajectories
- `mae_curve.png` — Per-task MAE
- `r2_curve.png` — Per-task R²
- `lr.png` — Learning rate schedule
- `grad_norm.png` — Global gradient norm
- `gradnorm_weights.png` — GradNorm weight evolution
- `throughput.png` — Graphs/second over epochs
- `gpu_memory_mb.png` — Peak GPU memory
- `confusion_matrix.png` — Stability classification confusion
- `roc_pr_curves.png` — ROC + PR curves for stability
- `calibration.png` — Reliability diagram for uncertainty

---

## 8. Time Budget Breakdown

### 8.1 Per-Step Timing (GC enabled, batch=16)

| Phase | Time (ms) | % of Step |
|-------|:---------:|:---------:|
| Forward pass | 313 ms | 25% |
| Backward pass | 213 ms | 17% |
| GradNorm computation | 501 ms | 40% |
| Optimizer + overhead | 226 ms | 18% |
| **Total per step** | **1,253 ms** | **100%** |

```
Forward       ████████████████████████░ 25%
Backward      █████████████████░░░░░░░░ 17%
GradNorm      ████████████████████████████████████████░░ 40%
Overhead      ██████████████████░░░░░░░░░░░░░░░░░░░░░░░ 18%
```

GradNorm dominates at **40%** of step time. This is because:
- 3 `torch.autograd.grad()` calls per update (50-batch frequency)
- `retain_graph=True` for each grad call
- Additional overhead for backbone parameter filtering

### 8.2 Per-Epoch Timing

| Phase | Time |
|-------|:----:|
| Data loading | ~15 s (500 batches × 30 ms/load) |
| Forward + backward + GradNorm | ~625 s (500 batches × 1,253 ms) |
| Validation | ~30 s |
| Metrics computation + logging | ~5 s |
| **Total per epoch** | **~675 s (~11 min)** |
| **Total for 150 epochs** | **~28 hours** |

### 8.3 GradNorm-Specific Cost

| Operation | Frequency | Cost per Call | Total per Epoch |
|-----------|:---------:|:-------------:|:---------------:|
| GradNorm update | 10 times (every 50/500) | ~500 ms | ~5 s |
| GradNorm compute_total | 500 times | ~2 ms | ~1 s |
| **GradNorm overhead** | | | **~6 s per epoch (~1%)** |

---

## 9. Throughput Analysis

### 9.1 DataLoader Impact

| Configuration | Throughput | vs Baseline |
|:-------------:|:----------:|:-----------:|
| workers=0 | 5.7 graphs/s | 1.0× |
| workers=4, fork | **13.2 graphs/s** | **2.3×** |

### 9.2 Gradient Checkpointing Impact

| Configuration | Step Time | Throughput | VRAM |
|:-------------:|:---------:|:----------:|:----:|
| GC enabled | 1,253 ms | 12.8 graphs/s | 470 MB |
| GC disabled | 943 ms | 17.0 graphs/s | 1,127 MB |

### 9.3 Combined Throughput

| Config | Throughput | VRAM |
|:------:|:----------:|:----:|
| workers=0, no GC | 7.6 graphs/s | 1,127 MB |
| workers=4, GC on | **12.8 graphs/s** | **470 MB** |
| workers=4, no GC | 17.0 graphs/s | 1,127 MB |

### 9.4 `multiprocessing_context` Comparison

| Context | CUDA Compatible | Throughput (workers=4) |
|:-------:|:---------------:|:----------------------:|
| `fork` | Yes (Python 3.14) | 13.2 graphs/s |
| `spawn` | No (CUDA reinit) | Fails on Python 3.14 |

---

## 10. Reproducibility

### 10.1 Seed Management

```python
# Initial seed set in build_dataset.py
SEED = 42

# Full state capture in checkpoints
rng_state = {
    "torch_cpu": torch.get_rng_state().tolist(),
    "torch_cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else [],
    "numpy": np.random.get_state(),
    "random": random.getstate(),
}
```

### 10.2 State Restoration on Resume

```python
# Restore all RNG states from checkpoint
torch.set_rng_state(torch.ByteTensor(rng["torch_cpu"]))
torch.cuda.set_rng_state(torch.ByteTensor(rng["torch_cuda"]))
np.random.set_state(rng["numpy"])
random.setstate(rng["random"])
```

This ensures:
- DataLoader shuffling is deterministic
- Dropout masks are reproducible from the point of resume
- Weight initialization is consistent

### 10.3 Limitations

- PyTorch CUDA convolution ops are **not fully deterministic** on all GPU architectures
- `torch.use_deterministic_algorithms(True)` is **not enabled** (would slow training)
- Results may differ slightly between GPU architectures (GTX 1650 vs RTX 4090)
- Results are **bitwise reproducible** on the same GPU architecture with the same CUDA version

---

## 11. Experiment B: Current Status

### 11.1 Configuration

| Setting | Value |
|---------|-------|
| GradNorm | ON (alpha=1.5) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Gradient checkpointing | auto (enabled on 4 GB GPU) |
| Two-stage EaH | Yes |
| Bucketed batching | Yes |
| DataLoader workers | 4 |
| Mixed precision | fp16 |
| Initial GradNorm weights | ef=1.0, eah=1.0, bg=0.4 |

### 11.2 Expected Results

| Metric | Expected Range | Notes |
|--------|:--------------:|-------|
| Ef MAE | ~0.05–0.10 eV/atom | Materials Project baseline ~0.1 |
| EaH MAE | ~0.02–0.06 eV/atom | Two-stage helps with near-zero values |
| BG MAE | ~0.15–0.30 eV | Band gaps are inherently harder |
| Stability F1 | ~0.70–0.85 | Depends on classification threshold (1e-3) |

*Populate from `test_results.json` when experiment completes.*

---

## 12. Hardware Requirements

### 12.1 Minimum (Current)

| Component | Specification |
|-----------|:-------------|
| GPU | NVIDIA GTX 1650 (4 GB VRAM) |
| CPU | 4+ cores (for DataLoader workers) |
| RAM | 16 GB (10 GB baseline + dataset overhead) |
| Storage | 20 GB for dataset + cache + checkpoints |
| OS | Linux (Python 3.14, CUDA 12.x) |

### 12.2 Recommended

| Component | Specification |
|-----------|:-------------|
| GPU | RTX 3060+ (12 GB VRAM) for larger models |
| CPU | 8+ cores |
| RAM | 32 GB |
| Storage | 50 GB SSD |

---

## 13. Related Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train/train.py` | Config-based `ScandiumTrainer` (uses `PINNLoss`, 5 tasks) |
| `scripts/train/train_v3_li.py` | **Active** — 3-task Li training with GradNorm (this document) |
| `scripts/train/experiment_sweep.py` | Automated experiment orchestration |
| `scripts/preprocess/build_dataset.py` | Download + clean + split + normalize |
| `scripts/preprocess/cache_graphs.py` | Pre-build graph files |
| `scripts/maintenance/profile_training.py` | Parameter count, throughput, step timing |
| `scripts/maintenance/profile_dataloader.py` | DataLoader worker benchmarks |
| `scripts/maintenance/benchmark_throughput.py` | GC vs no-GC throughput |
| `scripts/maintenance/benchmark_gradnorm_ab.py` | GradNorm on/off ablation |
