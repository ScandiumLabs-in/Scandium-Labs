# Operations Manual

> Engineering handbook for operating, maintaining, and troubleshooting the
> Scandium Labs solid-state electrolyte discovery platform.
>
> **Last updated:** July 2026
> **Target audience:** ML Engineers, Research Scientists, DevOps

---

## Table of Contents

1. [Daily Workflows](#1-daily-workflows)
2. [Data Workflows](#2-data-workflows)
3. [Release Workflows](#3-release-workflows)
4. [Backup Strategy](#4-backup-strategy)
5. [Recovery Procedures](#5-recovery-procedures)
6. [Troubleshooting](#6-troubleshooting)
7. [System Architecture Reference](#7-system-architecture-reference)
8. [Performance Baseline](#8-performance-baseline)

---

## 1. Daily Workflows

### 1.1 Morning Checklist

```bash
# 1. Check active training processes
ps aux | grep train_   # Should show 0-2 training processes
watch -n 5 nvidia-smi   # GPU utilization, memory, temperature

# 2. Check latest logs
tail -f logs/training.log | head -20

# 3. Check active experiment directories
ls -lt runs/ | head -5

# 4. Verify data integrity
ls datasets/v3_li_10000/graphs/ | wc -l      # Should be ~10000
ls datasets/v3_li_10000/*.pt                  # dataset_cache.pt, split_indices.pt

# 5. Check disk usage
df -h /    # Should have > 20 GB free
du -sh datasets/ checkpoints/ runs/ logs/
```

### 1.2 Weekly Review

```bash
# 1. Review experiment leaderboard
cat runs/index.csv | column -t -s,

# 2. Run cross-experiment comparison
for run in runs/SL-*; do
    if [ -f "$run/test_results.json" ]; then
        echo "=== $(basename $run) ==="
        python -c "
import json
with open('$run/test_results.json') as f:
    d = json.load(f)
for t in ['formation_energy','energy_above_hull','band_gap']:
    print(f'  {t}: MAE={d[t][\"mae\"]:.4f} R²={d[t][\"r2\"]:.4f}')
"
    fi
done

# 3. Run benchmarks
python scripts/maintenance/benchmark_throughput.py
python scripts/maintenance/profile_training.py

# 4. Check for outdated datasets / configs
git diff --name-only
```

### 1.3 Production Monitoring

If the API and workers are deployed:

```bash
# API health
curl http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":true}

# Celery worker status
celery -A api.tasks.celery_app inspect active
celery -A api.tasks.celery_app inspect stats

# Flower dashboard (if deployed)
# http://localhost:5555

# Database status
psql -h localhost -U user -d scandium -c "SELECT status, count(*) FROM jobs GROUP BY status;"

# Redis status
redis-cli INFO | grep -E "connected_clients|used_memory_human|keyspace"
```

---

## 2. Data Workflows

### 2.1 Rebuilding the Dataset

When new data is added or cleaning rules change:

```bash
# Full rebuild from Materials Project
python scripts/preprocess/build_dataset.py \
    --api-key $MP_API_KEY \
    --output datasets/v4_li_20000 \
    --max-structures 20000 \
    --min-li-fraction 0.05 \
    --families halides,oxides,sulfides,phosphates \
    --test-split 0.1 \
    --val-split 0.1 \
    --seed 42
```

**What `build_dataset.py` does:**

1. Queries Materials Project via API (Li-containing structures, Li ≥ 5 at.%)
2. Filters and cleans data (removes duplicates, invalid structures)
3. Computes chemical family labels (halide, oxide, sulfide, phosphate)
4. Performs family-balanced train/val/test split
5. Saves `dataset_cache.pt` (structures + targets) and `split_indices.pt`
6. Saves `metadata.json` with version info

**Output structure:**

```
datasets/v4_li_20000/
├── dataset_cache.pt       # List[structures], dict[targets]
├── split_indices.pt       # dict with train/val/test keys
├── metadata.json          # Version, date, filter params, statistics
└── graphs/                # (After caching) individual graph .pt files
    ├── 0.pt
    ├── 1.pt
    └── ...
```

### 2.2 Caching Graphs

```bash
# Single-process CPU caching (recommended)
nohup python scripts/preprocess/cache_graphs.py \
    --data-dir datasets/v3_li_10000 \
    > /tmp/cache_graphs.log 2>&1 &

# Monitor progress
tail -f /tmp/cache_graphs.log  # Expected: ~6.1 graphs/s

# Verify completion
ls datasets/v3_li_10000/graphs/ | wc -l   # Should match dataset size
```

**Important:** Cache builds must run single-process on CPU. Do NOT use multiprocessing with CUDA — it causes deadlocks. The cache builder runs at ~6.1 graphs/s and completes ~10k graphs in ~27 minutes.

### 2.3 Adding External Data Sources

```python
# Example: Combine MP + JARVIS data
from src.data.collectors import MaterialsProjectCollector, JARVISCollector
from src.data.cleaner import DataCleaner
import pandas as pd

# Collect from both sources
mp_col = MaterialsProjectCollector(api_key="your_key")
jarvis_col = JARVISCollector()

mp_data = mp_col.collect(elements=["Li"], max_results=5000)
jarvis_data = jarvis_col.collect(dataset_name="dft_3d")

# Clean and merge
cleaner = DataCleaner()
mp_clean = cleaner.clean(mp_data)
jarvis_clean = cleaner.clean(jarvis_data)

combined = pd.concat([mp_clean, jarvis_clean]).drop_duplicates(
    subset=["formula_pretty"]
)
```

### 2.4 Data Verification

```bash
# Check dataset statistics
python -c "
import torch
cache = torch.load('datasets/v3_li_10000/dataset_cache.pt')
split = torch.load('datasets/v3_li_10000/split_indices.pt')
print(f'Structures: {len(cache[\"structures\"])}')
print(f'Train: {len(split[\"train\"])}, Val: {len(split[\"val\"])}, Test: {len(split[\"test\"])}')
for task, vals in cache['targets'].items():
    n_nan = sum(1 for v in vals if v is None or (hasattr(v, '__len__') and len(v) == 0))
    print(f'{task}: {len(vals)} total, {n_nan} NaN')
"
```

---

## 3. Release Workflows

### 3.1 Version Scheme

The project uses [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

| Component | Current | Next | Criteria |
|-----------|---------|------|----------|
| MAJOR | 0 | 1 | First production release |
| MINOR | 5 | 6 | New features, dataset version bumps |
| PATCH | 0 | 1 | Bug fixes, performance improvements |

### 3.2 Making a Release

```bash
# 1. Update CHANGELOG.md
vim CHANGELOG.md
# Add entry under [Unreleased] → [v0.6.0]

# 2. Update version in pyproject.toml
vim pyproject.toml  # version = "0.5.0" → "0.6.0"

# 3. Tag the release
git add CHANGELOG.md pyproject.toml
git commit -m "chore: bump version to v0.6.0"
git tag -a v0.6.0 -m "v0.6.0: GradNorm + cosine scheduler release"
git push origin main --tags

# 4. (Optional) Build and push Docker images
docker compose build
docker tag scandium-labs-api:latest ghcr.io/scandium-labs/api:v0.6.0
docker push ghcr.io/scandium-labs/api:v0.6.0
```

### 3.3 CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [v0.6.0] - 2026-07-08

### Added
- GradNorm adaptive loss balancing with analytical gradient updates
- Cosine annealing with warm restarts scheduler
- Size-bucketed batch sampler (30% less padding)
- Experiment tracker with auto-generated reports and leaderboards
- Gradient checkpointing (auto-detect based on VRAM)

### Changed
- Default config: hidden_dim 128, 4 ALIGNN layers, 2 Transformer layers
- DataLoader: pin_memory=True, multiprocessing_context='fork'
- Two-stage EaH head now default for energy_above_hull

### Performance
- 12.8 graphs/s training throughput (132% improvement)
- 470 MB peak VRAM with gradient checkpointing (2.4x savings)
- 1.28M parameter model (76% more capacity)

### Fixed
- CUDA multiprocessing deadlock on Python 3.14
- NaN propagation in per-task loss computation
- Normalizer path resolution in InferenceEngine
```

### 3.4 Pre-Release Checklist

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Linter clean: `ruff check src/`
- [ ] Type checker: `mypy src/` (if configured)
- [ ] Training runs end-to-end on a small dataset
- [ ] Inference produces sensible results
- [ ] API health check passes
- [ ] CHANGELOG updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Model checkpoint archived (if this is a model release)

---

## 4. Backup Strategy

### 4.1 What to Back Up

| Asset | Location | Size | Backup Frequency | Retention |
|-------|----------|------|------------------|-----------|
| Trained checkpoints | `runs/`, `checkpoints/` | ~10 MB each | After each experiment | Indefinite |
| Experiment configs | `runs/SL-*/config.yaml` | < 1 KB | Per experiment | Indefinite |
| Run registry | `runs/index.csv` | < 100 KB | Per experiment | Indefinite |
| Epoch metrics | `runs/SL-*/epoch_metrics.json` | ~100 KB | Per experiment | Indefinite |
| Dataset | `datasets/` | ~2 GB | After each rebuild | Keep last 2 versions |
| Graphs | `datasets/*/graphs/` | ~5-10 GB | Optional (re-buildable) | Keep last version |
| Source code | `git` | ~10 MB | Per commit | Indefinite |

### 4.2 What Does NOT Need Backing Up

- **Datasets are rebuildable** — The Materials Project API can re-collect the same data using the same parameters. Metadata JSON captures exact query parameters.
- **Graphs are regeneratable** — Given the source dataset and config, `cache_graphs.py` produces identical graphs.
- **Logs are archival but not critical** — Metrics are captured in structured JSON.

### 4.3 Backup Commands

```bash
# Backup all experiment results
tar czf backups/experiments_$(date +%Y%m%d).tar.gz runs/ checkpoints/

# Backup dataset (if not rebuilding from API)
tar czf backups/dataset_v3_li_10000_$(date +%Y%m%d).tar.gz datasets/v3_li_10000/

# Offsite: copy to S3-compatible storage
aws s3 sync runs/ s3://scandium-labs-backups/runs/
aws s3 sync checkpoints/ s3://scandium-labs-backups/checkpoints/
```

### 4.4 Dataset Rebuild from Metadata

```bash
# The metadata.json in each dataset directory contains everything needed to rebuild:
cat datasets/v3_li_10000/metadata.json
{
  "version": "v3_li_10000",
  "created": "2026-06-15T10:30:00Z",
  "source": "MaterialsProject",
  "elements": ["Li"],
  "min_li_fraction": 0.05,
  "max_results": 10000,
  "families": ["halides", "oxides", "sulfides", "phosphates"],
  "n_structures": 10000,
  "n_train": 8310,
  "n_val": 586,
  "n_test": 1104,
  "seed": 42
}
```

---

## 5. Recovery Procedures

### 5.1 Resume Training from Checkpoint

If training crashes (OOM, power loss, timeout):

```bash
# 1. Find the most recent checkpoint
ls -lt runs/SL-20260708-001/checkpoints/
# last.pt or epoch_NNN.pt

# 2. Resume
python scripts/train/train_v3_li.py \
    --resume runs/SL-20260708-001/checkpoints/last.pt
```

**What is preserved:**

- Model weights, optimizer state, LR scheduler, GradScaler, RNG
- GradNorm weights and per-task loss history
- Best val loss tracking (continues across resume)
- Experiment directory (all artifacts in one place)

### 5.2 Recover from Corrupted Checkpoint

If a checkpoint `.pt` file is corrupted:

```bash
# 1. Try loading with strict=False to see which components are intact
python -c "
import torch
ckpt = torch.load('runs/SL-20260708-001/checkpoints/last.pt', map_location='cpu', weights_only=False)
print('Keys:', ckpt.keys())
print('Model keys:', len(ckpt.get('model', {})))
print('Optimizer keys:', ckpt.get('optimizer'))
"

# 2. Use the best alternative checkpoint
ls runs/SL-20260708-001/checkpoints/best_*.pt

# 3. If nothing works, restart from scratch with the same config
# (You lose training progress but not reproducibility)
python scripts/train/train_v3_li.py \
    --config runs/SL-20260708-001/config.yaml
```

### 5.3 CUDA Out-of-Memory (OOM) Recovery

```bash
# 1. Clear GPU memory
sudo fuser -v /dev/nvidia*   # Find processes using GPU
kill -9 <PID>                 # Kill offending process

# 2. Or reset all GPU states
sudo nvidia-smi --gpu-reset

# 3. Apply memory optimizations before retrying:
#    - Enable gradient checkpointing (use_gradient_checkpointing: true)
#    - Reduce batch_size to 8
#    - Reduce hidden_dim to 64
#    - Disable mixed precision? (No — AMP saves memory)
```

### 5.4 Dataset Recovery

If dataset files are corrupted or deleted:

```bash
# Option 1: Rebuild from API (requires MP_API_KEY)
python scripts/preprocess/build_dataset.py \
    --output datasets/v3_li_10000_restored \
    --max-structures 10000

# Option 2: Restore from backup
tar xzf backups/dataset_v3_li_10000_20260701.tar.gz

# Option 3: If only split_indices.pt is missing
python -c "
import torch
cache = torch.load('datasets/v3_li_10000/dataset_cache.pt')
n = len(cache['structures'])
# Create a random 80/10/10 split
import numpy as np
np.random.seed(42)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
split = {
    'train': perm[:n_train].tolist(),
    'val': perm[n_train:n_train+n_val].tolist(),
    'test': perm[n_train+n_val:].tolist(),
}
torch.save(split, 'datasets/v3_li_10000/split_indices.pt')
print('Recovered split_indices.pt')
"
```

---

## 6. Troubleshooting

### 6.1 OOM (Out of Memory)

**Symptom:** `CUDA out of memory. Tried to allocate ... MiB`

**Immediate fix:**

```bash
# 1. Reduce batch size
# Edit config: training.batch_size = 8 (was 16)

# 2. Ensure gradient checkpointing is on
# Edit config: model.use_gradient_checkpointing = true or "auto"

# 3. Clear cached memory
python -c "import torch; torch.cuda.empty_cache()"

# 4. Reduce model size
# Edit config: model.hidden_dim = 64, model.num_alignn_layers = 2
```

**Root causes and permanent fixes:**

| Cause | Diagnostic | Fix |
|-------|-----------|-----|
| Batch too large | OOM on first batch | Reduce batch_size, increase gradient_accumulation_steps |
| GC disabled | OOM after a few batches | Set `use_gradient_checkpointing: true` |
| Memory leak | VRAM grows over epochs | Check for un-freed tensors; use `torch.cuda.empty_cache()` periodically |
| Too many workers | CUDA multiprocessing issues | Set `num_workers=0` or `multiprocessing_context='fork'` |
| Model too large | OOM on model init | Reduce hidden_dim, num_alignn_layers, num_transformer_layers |

### 6.2 NaN Loss

**Symptom:** Training loss becomes `nan` or `inf`

**Diagnostics:**

```bash
# Check which component produces NaN
python -c "
import torch
# Run a single batch through the model and check
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
"
```

**Common causes and fixes:**

| Cause | Diagnostic | Fix |
|-------|-----------|-----|
| Learning rate too high | Loss diverges → NaN | Reduce `learning_rate` (e.g., 5e-4 → 1e-4) |
| Data not normalized | Targets have extreme values | Enable `normalize_targets: true` |
| Log of negative | Arrhenius term: `log10(sigma)` | Add epsilon: `log10(sigma + 1e-10)` |
| Exploding gradients | Grad norm spikes before NaN | Reduce `gradient_clip` or reduce LR |
| Numeric instability in AMP | NaN only with mixed precision | Set `mixed_precision: false` |

**Quick fix checklist:**

1. Check that `normalize_targets: true` is set
2. Reduce `training.learning_rate` by 10×
3. Enable `gradient_clip: 1.0`
4. Set `mixed_precision: false` (debug only)
5. Check for NaN in input data: `torch.isnan(input).any()`

### 6.3 CUDA Errors

**Symptom:** `RuntimeError: CUDA error: ...`

| Error | Most Likely Cause | Fix |
|-------|------------------|-----|
| `out of memory` | Batch too large / model too big | See OOM section |
| `illegal memory access` | Tensor on wrong device | Check `.to(device)` calls |
| `device-side assert triggered` | Index out of bounds in embedding | Check vocabulary sizes |
| `CUDNN_STATUS_NOT_SUPPORTED` | Batch size too large for cuDNN | Reduce batch size |
| `no kernel image is available` | PyTorch/CUDA version mismatch | Reinstall matching versions |
| `driver version is insufficient` | NVIDIA driver too old | Update driver ≥ 525 |

### 6.4 DataLoader Deadlocks

**Symptom:** Training freezes after first epoch, no GPU activity

**Cause:** CUDA multiprocessing incompatibility on Python 3.14+.

**Fix:**

```python
# In training scripts, before DataLoader creation:
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass  # Already set

# In DataLoader:
loader = DataLoader(
    ...,
    num_workers=4,
    multiprocessing_context="fork",
)
```

**If deadlock persists:**

```bash
# 1. Reduce workers
num_workers=2  # or 0 for no multiprocessing

# 2. Use single-process caching to pre-build all graphs
python scripts/preprocess/cache_graphs.py --data-dir datasets/v3_li_10000

# 3. Verify fork method is set
python -c "import multiprocessing as mp; print(mp.get_start_method())"
# Should print: fork
```

### 6.5 Poor Validation Metrics

**Symptom:** Model trains (loss decreases) but validation metrics are poor.

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Train loss ↓, val loss ↑ | Overfitting | Increase dropout, reduce model size, add regularization |
| Both train/val loss plateau | Underfitting | Increase model capacity, reduce dropout, increase LR |
| Val metrics oscillate | LR too high | Reduce LR, add LR scheduler |
| EaH always near zero | Unstable two-stage training | Adjust `lambda_bce`, `lambda_stable` in `TwoStageEahLoss` |
| Band gap always ~0 | Not enough band gap data | Check band gap coverage in dataset |
| GradNorm weights extreme | Alpha too high | Reduce `gradnorm.alpha` to 0.5-1.0 |

### 6.6 Slow Training

**Symptom:** Throughput below 5 graphs/s on GPU

**Diagnostics:**

```bash
# Profile DataLoader
python scripts/maintenance/profile_dataloader.py

# Profile full training step
python scripts/maintenance/profile_training.py

# Benchmark throughput
python scripts/maintenance/benchmark_throughput.py
```

**Optimization hierarchy (in order of impact):**

| Optimization | Speedup | Difficulty |
|-------------|---------|------------|
| Increase `num_workers` to 4 | 132% | Easy |
| Enable `pin_memory=True` | 10-15% | Already default |
| Pre-cache graphs | 29 min saved first epoch | One-time |
| Gradient checkpointing trade-off | 33% slower for 2.4× VRAM | Config toggle |
| `torch.compile` (planned) | ~30% | Medium |
| ONNX / TensorRT (planned) | ~5× inference | Hard |

### 6.7 Graph Building Failures

**Symptom:** Errors during `cache_graphs.py` or dataset iteration

```
RuntimeError: Structure has no sites
ValueError: No atoms in unit cell
KeyError: 'element not found in periodic table'
```

**Fixes:**

```bash
# Skip problematic structures (they'll be filtered during cleaning)
# The cleaner filters structures with < 2 or > 200 atoms automatically

# If specific CIF files fail:
python -c "
from pymatgen.core import Structure
try:
    s = Structure.from_file('path/to/problematic.cif')
    print(f'OK: {s.composition}')
except Exception as e:
    print(f'FAIL: {e}')
"
```

### 6.8 API / Deployment Issues

| Issue | Diagnostic | Fix |
|-------|-----------|-----|
| Model not loading | Health check: `model_loaded: false` | Check `MODEL_PATH`, checkpoint format |
| Celery tasks not processing | `celery inspect active` is empty | Check Redis connection, worker logs |
| Database connection refused | API logs show connection error | Check `DATABASE_URL`, Postgres status |
| Token auth fails | 401 on all endpoints | Check `JWT_SECRET_KEY` matches |
| Slow API responses | Request duration > 5s | Scale workers, check GPU availability |
| File upload fails | 400 on `/screen/upload` | Check file format (.cif/.poscar), file size |

---

## 7. System Architecture Reference

### 7.1 Directory Layout

```
scandium-labs/
├── src/               # Source code (packages)
│   ├── data/          #   Dataset, cleaners, collectors, splitters
│   ├── models/        #   ScandiumPINNGNN, GNN layers, heads
│   ├── training/      #   Trainer, losses, schedulers, experiment tracker
│   ├── inference/     #   Engine, ranking, stability, validation
│   ├── evaluation/    #   Metrics, OOD detection
│   ├── graphs/        #   Graph construction, feature engineering
│   ├── chemistry/     #   Chemical featurization
│   └── utils/         #   Config loaders, logging, I/O
├── scripts/           # Entry points
│   ├── train/         #   train_v3_li.py, experiment_sweep.py
│   ├── inference/     #   screen_candidates.py
│   ├── analyze/       #   analyze_training.py
│   ├── preprocess/    #   build_dataset.py, cache_graphs.py
│   ├── evaluate/      #   cross_validate.py, benchmarks
│   └── maintenance/   #   Profiling, maintenance scripts
├── api/               # FastAPI + Celery
├── configs/           # YAML/JSON configuration files
├── tests/             # Pytest suite
├── datasets/          # Preprocessed datasets
├── checkpoints/       # Model checkpoints
├── runs/              # Experiment tracker artifacts
├── logs/              # Training logs
└── docs/              # Documentation
```

### 7.2 Data Flow

```
Data Collection
  MaterialsProjectCollector  ──▶  DataFrame
  JARVISCollector             ──▶  DataFrame
  OQMDCollector               ──▶  DataFrame
         │
         ▼
Data Cleaning
  DataCleaner.clean()  ──▶  Deduplicated, filtered DataFrame
  PropertyNormalizer.fit()  ──▶  Normalization stats
         │
         ▼
Dataset Building
  build_dataset.py  ──▶  dataset_cache.pt + split_indices.pt
         │
         ▼
Graph Caching
  cache_graphs.py  ──▶  graphs/0.pt, graphs/1.pt, ...
         │
         ▼
Training
  LazyGraphDataset + DataLoader  ──▶  ScandiumPINNGNN
  ScandiumTrainer.train() / train_v3_li.py
         │
         ├── runs/SL-*/ (experiment tracker)
         └── checkpoints/*.pt (model weights)
         │
         ▼
Inference
  InferenceEngine.predict_single()  ──▶  Property predictions
  ParetoRanker.rank()               ──▶  Ranked candidates
  resolve_stability()               ──▶  Stability check
         │
         ▼
API
  FastAPI /screen, /screen/upload   ──▶  JSON response
  Celery workers                     ──▶  Async batch screening
```

---

## 8. Performance Baseline

### 8.1 Known Performance Metrics

Record these metrics after any significant change to detect regressions.

| Metric | Baseline (v0.5.0) | Measurement Method |
|--------|-------------------|-------------------|
| Training throughput | 12.8 graphs/s | `profile_training.py` |
| Inference throughput (GPU, MC=20) | 5.5 struct/s | `screen_candidates.py` |
| Inference throughput (CPU) | 1.2 struct/s | `screen_candidates.py` |
| Peak VRAM (GC on) | 470 MB | `nvidia-smi` during training |
| Epoch time (8310 samples) | ~353 s | Training log |
| Parameters | 1,281,321 | `profile_training.py` |
| Test Ef MAE | 0.042 eV/atom | `test_results.json` |
| Test EaH MAE | 0.089 eV/atom | `test_results.json` |
| Test BG MAE | 0.215 eV | `test_results.json` |
| Test Ef R² | 0.962 | `test_results.json` |
| Test EaH R² | 0.873 | `test_results.json` |
| Test BG R² | 0.754 | `test_results.json` |
| Stability F1 | 0.871 | `test_results.json` |

### 8.2 Running Performance Checks

```bash
# Quick throughput benchmark
python scripts/maintenance/benchmark_throughput.py

# Full profiling
python scripts/maintenance/profile_training.py --epochs 3

# DataLoader benchmark
python scripts/maintenance/profile_dataloader.py --workers 0,2,4

# Torch compile benchmark (experimental)
python scripts/maintenance/benchmark_torch_compile.py
```

### 8.3 When to Investigate Performance

Investigate if any metric degrades by more than 15% from baseline:

- Throughput drop → DataLoader or GPU issues
- VRAM increase → Memory leak or config change
- Metric regression → Model architecture or data issue
- Epoch time increase → Graph building overhead (uncached graphs)
