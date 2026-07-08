# Reproducibility Guide

**Last Verified:** 2026-07-08
**Software Version:** v0.3.0
**Target Platform:** Consumer GPU (4 GB VRAM)

This document provides a complete, step-by-step guide to reproducing all Scandium Labs research results — from a bare system to fully trained models with analysis.

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Requirements](#2-software-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Random Seed Management](#4-random-seed-management)
5. [Data Pipeline](#5-data-pipeline)
6. [Configuration Files](#6-configuration-files)
7. [Training Pipeline](#7-training-pipeline)
8. [Expected Outputs](#8-expected-outputs)
9. [Expected Runtime](#9-expected-runtime)
10. [Checkpoint Format](#10-checkpoint-format)
11. [Evaluation and Analysis](#11-evaluation-and-analysis)
12. [Verification Checklist](#12-verification-checklist)
13. [Reproducing Specific Experiments](#13-reproducing-specific-experiments)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Hardware Requirements

### Minimum Hardware (Verified)

| Component | Requirement | Notes |
|---|---|---|
| **GPU** | NVIDIA GTX 1650 (4 GB VRAM) or compatible | CUDA 12.4+ required; 4 GB is the hard minimum |
| **CPU** | x86_64, 4+ cores | Graph caching benefits from more cores |
| **RAM** | 14 GB minimum | 16 GB recommended for dataset operations |
| **Storage** | 10 GB free | ~2 GB for dataset + ~1 GB for checkpoints + logs |
| **Internet** | Required for data download | Materials Project API access needed |

### Recommended Hardware (for Faster Iteration)

| Component | Recommendation | Speedup |
|---|---|---|
| **GPU** | RTX 3060 (12 GB) or better | 2-4× training speedup; enables larger models |
| **RAM** | 32 GB | Enables in-memory dataset cache |
| **Storage** | SSD (NVMe) | Faster graph loading from disk |

### Hardware Test

Verify GPU availability:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else 'CPU')"
```

Expected output:
```
CUDA: True, Device: NVIDIA GeForce GTX 1650, VRAM: 4.0 GB
```

---

## 2. Software Requirements

### Core Dependencies (Exact Versions)

| Package | Version | Purpose |
|---|---|---|
| Python | 3.12.13 | Verified runtime |
| PyTorch | 2.6.0+cu124 | Deep learning framework |
| torch-geometric | 2.6.1+ | Graph neural network library |
| pymatgen | 2024.0+ | Materials data handling |
| mp-api | 0.44.0+ | Materials Project API client |
| numpy | 1.24+ | Numerical computing |
| pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.2+ | ML utilities (splits, metrics) |
| scipy | 1.10+ | Scientific computing |
| pyyaml | 6.0+ | Configuration loading |
| tqdm | 4.64+ | Progress bars |
| wandb | — | (Optional) experiment logging |

### Development Dependencies

| Package | Version | Purpose |
|---|---|---|
| pytest | 7.4+ | Testing framework |
| pytest-cov | 4.1+ | Test coverage |
| ruff | 0.1+ | Linting and formatting |
| pre-commit | 3.5+ | Git hook management |
| mypy | — | (Optional) type checking |

### Full Dependency Tree

```
torch 2.6.0+cu124
  └── torch-geometric 2.6.1+
        ├── scikit-learn
        ├── numpy
        └── scipy
pymatgen 2024.0+
  ├── mp-api 0.44.0+    (Materials Project REST)
  └── ase (for SOAP features, optional)
streamlit 1.20+         (dashboard)
fastapi 0.100+           (REST API)
celery 5.3+              (async tasks)
redis 4.5+               (message broker)
sqlalchemy 2.0+          (database)
```

---

## 3. Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/scandium-labs/scandium-labs.git
cd scandium-labs
```

### Step 2: Create Virtual Environment

```bash
# Option A: venv (recommended for pip users)
python -m venv venv
source venv/bin/activate

# Option B: conda (if using conda)
conda env create -f environment.yml
conda activate scandium-labs
```

### Step 3: Install PyTorch

```bash
# CUDA 12.4 (matches development environment)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU only (for development/testing)
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install torch-geometric

```bash
pip install torch_geometric==2.6.1

# Optional: install additional torch-geometric dependencies
# pip install pyg_lib torch_scatter torch_sparse \
#   --index-url https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### Step 5: Install Package

```bash
# Minimal install (training only)
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With development tools
pip install -e ".[dev]"

# Full install (all extras)
pip install -e ".[gpu,dev]"
```

### Step 6: Verify Installation

```bash
# Quick smoke test
make test

# Verify all imports
python -c "
from src.data.collectors import MaterialsProjectCollector, OQMDCollector
from src.data.cleaner import DataCleaner, PropertyNormalizer
from src.data.splitter import composition_based_split
from src.graphs.builder import ALIGNNGraphBuilder, CrystalGraphBuilder, FeatureEngineer
from src.models.scandium_model import ScandiumPINNGNN
from src.training.trainer import ScandiumTrainer
from src.training.losses import PINNLoss, GradNormLoss
print('All imports successful')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

---

## 4. Random Seed Management

### Default Seed

The default random seed across the codebase is **42** (used in `splitter.py`, `build_dataset.py`, and configuration defaults).

### Seed Application Points

| Component | Seed | Location |
|---|---|---|
| Data splitting | `random_state=42` | `src/data/splitter.py:16,21` |
| Dataset build | `seed=42` | `datasets/v3_li_10000/metadata.json:config.seed` |
| Model initialization | None (Xavier uniform) | `src/models/scandium_model.py:109` |
| DataLoader shuffle | None (Python hash seed) | `scripts/train/train_v3_li.py:154` |
| NumPy random | Restored from checkpoint | `scripts/train/train_v3_li.py:276-279` |
| PyTorch random | Restored from checkpoint | `scripts/train/train_v3_li.py:274-275` |
| Python random | Restored from checkpoint | `scripts/train/train_v3_li.py:278-279` |

### Full Seed Setting for Reproducibility

For maximal reproducibility, set seeds before any operation:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Note**: Full bitwise reproducibility is not guaranteed across PyTorch versions, CUDA versions, or GPU architectures. The checkpoint system saves RNG states to allow exact resumption.

---

## 5. Data Pipeline

### 5.1 Materials Project API Key

All data collection requires a Materials Project API key.

1. Register at https://materialsproject.org/api
2. Set the key as an environment variable:

```bash
export MP_API_KEY="your_api_key_here"
# or
export MATERIALS_PROJECT_API_KEY="your_api_key_here"
```

### 5.2 Building the Dataset

```bash
# Full pipeline: collect, clean, split, normalize
python scripts/preprocess/build_dataset.py \
    --sources mp \
    --target 10000 \
    --output datasets/v3_li_10000 \
    --elements None \
    --min-atoms 2 \
    --max-atoms 200 \
    --ef-range -10 5 \
    --seed 42 \
    --normalize \
    --no-deduplicate
```

**Expected output:**

```
datasets/v3_li_10000/
├── dataset_cache.pt       # ~200 MB (structures + targets)
├── dataset_report.json    # Summary statistics
├── metadata.json           # Build configuration
├── normalizer.json         # Z-score parameters
├── raw/                    # Raw source data
└── split_indices.pt        # Train/val/test indices
```

### 5.3 Pre-caching Graphs (Optional, Recommended)

Building graphs on the fly during training adds ~29 minutes overhead to the first epoch. Pre-caching eliminates this:

```bash
python scripts/preprocess/cache_graphs.py \
    --data-dir datasets/v3_li_10000 \
    --num-workers 1 \
    --overwrite
```

**Expected behavior:**
- Creates `datasets/v3_li_10000/graphs/` directory.
- Writes one `.pt` file per structure: `graphs/0.pt`, `graphs/1.pt`, ..., `graphs/9999.pt`.
- Single-process CPU mode (~6.1 graphs/s) to avoid CUDA multiprocessing issues.
- Estimated time: ~27 minutes for 10,000 structures.

**Important notes:**
- Cache builder uses `multiprocessing_context='fork'` — required on Python 3.14+ with CUDA.
- Build runs on CPU only (GPU memory cannot be shared across processes).
- Each `.pt` file is ~50-200 KB (varies with structure size).

---

## 6. Configuration Files

### Active Configuration: `configs/model_config_v3_li.yaml`

This is the primary configuration for the current best model (SL-20260708-001):

```yaml
model:
  name: "ScandiumPINNGNN-v3-Li"
  hidden_dim: 128
  num_alignn_layers: 4
  num_transformer_layers: 2
  num_attention_heads: 4
  dropout: 0.15
  mc_dropout_samples: 20
  use_two_stage_eah: true
  use_gradient_checkpointing: auto

graph:
  cutoff: 8.0
  max_neighbors: 16
  num_rbf: 64
  num_sbf: 32

tasks:
  - name: "formation_energy"
    weight: 1.0
  - name: "energy_above_hull"
    weight: 1.0
    two_stage: true
  - name: "band_gap"
    weight: 1.0

gradnorm:
  enabled: true
  alpha: 1.5

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 0.0005
  max_epochs: 150
  patience: 40
  optimizer: "AdamW"
  weight_decay: 0.00001
  gradient_clip: 1.0
  mixed_precision: true
  normalize_targets: true
  dataset: "v3_li_10000"

bucketing:
  enabled: true
  bucket_size_mult: 2.0

logging:
  tensorboard: false
  wandb: false
  plot: true
  save_epoch_checkpoints: 10
```

### Alternative Configurations

| Configuration File | Purpose |
|---|---|
| `configs/model_config_v3_li.yaml` | **Active** — GradNorm enabled, optimized for 4 GB GPU |
| `configs/model_config_v3_li_no_gradnorm.yaml` | Same model, GradNorm disabled (fixed equal weights) |
| `configs/model_config_v3_li_with_scheduler.yaml` | Same model, cosine LR scheduler |
| `configs/model_config_v3.yaml` | Pre-Li-specific architecture (legacy) |
| `configs/model_config_v2.yaml` | v2 architecture (legacy) |
| `configs/phase3_config_log_eah.yaml` | Log-transformed EaH targets (experiment) |

---

## 7. Training Pipeline

### 7.1 Training from Scratch

```bash
# Standard training (active config, GradNorm enabled)
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000 \
    --out-dir checkpoints/v3_li_10k_fresh
```

### 7.2 Training with GradNorm Disabled

```bash
# Ablation: fixed task weights, no adaptive balancing
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000 \
    --out-dir checkpoints/v3_li_no_gradnorm \
    --no-gradnorm
```

### 7.3 Resuming from Checkpoint

```bash
# Resume training from a checkpoint
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000 \
    --resume runs/SL-20260708-001/checkpoints/last.pt
```

### 7.4 Programmatic Training

```python
from src.training.trainer import ScandiumTrainer

trainer = ScandiumTrainer(
    config_path="configs/model_config_v3_li.yaml",
    data_dir="datasets/v3_li_10000",
)
model, test_metrics = trainer.train()
print(f"Test metrics: {test_metrics}")
```

### 7.5 Training Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/model_config_v3_li.yaml` | Path to YAML config |
| `--resume` | `None` | Path to .pt checkpoint for resumption |
| `--data-dir` | `datasets/v3_li_10000` | Dataset directory |
| `--out-dir` | `checkpoints/v3_li_10k_fresh` | Output directory (legacy) |
| `--no-gradnorm` | `False` | Disable GradNorm adaptive weighting |

### 7.6 What Happens During Training

```
1. Load configuration from YAML.
2. Load dataset_cache.pt (10,000 structures + targets).
3. Load split_indices.pt (train=8310, val=586, test=1104).
4. Initialize LazyGraphDataset with graph_dir (or on-the-fly builder).
5. Initialize SizeBucketedBatchSampler (if bucketing enabled).
6. Create DataLoaders with 4 workers, pin_memory, fork context.
7. Build ScandiumPINNGNN model (1.28M params).
8. Initialize optimizer (AdamW, lr=5e-4, wd=1e-5).
9. Initialize GradNormLoss (initial weights: Ef=1.0, EaH=1.0, BG=0.4).
10. Initialize ExperimentTracker (run ID: SL-YYYYMMDD-NNN).
11. Begin epoch loop:
    a. Train: forward → loss → backward (AMP) → clip → step.
    b. Every 50 batches: update GradNorm weights.
    c. Validate: full validation set, per-task metrics.
    d. Save checkpoint (periodic + best-per-metric).
    e. Early stopping check (patience=40).
12. Final test evaluation.
13. Generate reports and plots.
```

---

## 8. Expected Outputs

### Directory Structure After Training

```
runs/SL-YYYYMMDD-NNN/                    # Experiment run directory
├── config.yaml                           # Copy of input config
├── run_metadata.json                     # System + experiment metadata
├── TRAINING_SUMMARY.md                   # Per-epoch summary (updated live)
├── FINAL_REPORT.md                       # Final training report
├── BEST_MODEL_REPORT.md                  # Best model analysis
├── MODEL_CARD.md                         # Auto-generated model card
├── EXPERIMENT_LEADERBOARD.md             # All experiments ranked
├── STOP_REPORT.md                        # Stop reason + best epoch
├── SCORECARD.md                          # 5/6 checks passed
├── epoch_metrics.json                    # All epochs (JSON)
├── epoch_metrics.csv                     # All epochs (CSV)
├── test_results.json                     # Final test metrics
├── checkpoints/                          # Model weights
│   ├── last.pt                           # Most recent epoch
│   ├── epoch_000.pt ... epoch_130.pt     # Periodic (every 10 epochs)
│   ├── best_val_loss.pt                  # Lowest validation loss
│   ├── best_formation_energy_mae.pt      # Best EF MAE
│   ├── best_formation_energy_r2.pt       # Best EF R²
│   ├── best_energy_above_hull_mae.pt     # Best EaH MAE
│   ├── best_energy_above_hull_r2.pt      # Best EaH R²
│   ├── best_band_gap_mae.pt             # Best BG MAE
│   └── best_band_gap_r2.pt             # Best BG R²
├── plots/                                # Training visualizations
│   ├── loss_curve.png                    # Train/val loss over epochs
│   ├── mae_curve.png                     # Per-task MAE curves
│   ├── r2_curve.png                      # Per-task R² curves
│   ├── lr.png                            # Learning rate schedule
│   ├── grad_norm.png                     # Gradient norm over epochs
│   ├── gradnorm_weights.png             # GradNorm weight evolution
│   ├── gpu_memory_mb.png                 # GPU memory usage
│   ├── epoch_time_s.png                  # Epoch duration
│   ├── throughput.png                    # Graphs per second
│   ├── confusion_matrix.png              # Stability classification
│   ├── roc_pr_curves.png                 # ROC + PR curves
│   └── calibration.png                   # Reliability diagram
├── analysis/                             # Analysis outputs
│   ├── per_task_mae.png                  # Task comparison
│   ├── per_task_r2.png                   # Task comparison
│   ├── per_task_rmse.png                 # Task comparison
│   ├── system_metrics.png                # System resource metrics
│   ├── training_timeline.png             # Timeline visualization
│   ├── gradcam.png                       # (future) attention maps
│   ├── SCORECARD.md                      # Experiment health check
│   ├── BEST_MODEL_REPORT.md              # Best model report
│   └── FINAL_REPORT.md                   # Final analysis report
├── tables/                               # Benchmark tables
│   ├── benchmark.csv                     # CSV format
│   ├── benchmark.md                      # Markdown format
│   └── benchmark.tex                     # LaTeX format
└── analysis/                             # Additional analysis
    ├── FINAL_REPORT.md                   # Full analysis
    └── SCORECARD.md                      # Health check
```

### Expected Metric Values (Approximate)

| Metric | Expected Value | Acceptable Range |
|---|---|---|
| Final train loss | ~1.86 | 1.5 - 3.0 |
| Best val loss | ~3.09 | 2.5 - 5.0 |
| Epochs to completion | ~139 | 100 - 150 (early stop) |
| Formation energy MAE | ~0.52 | 0.4 - 0.8 |
| Formation energy R² | ~0.58 | 0.4 - 0.7 |
| Energy above hull MAE | ~0.13 | 0.10 - 0.20 |
| Energy above hull R² | ~0.38 | 0.2 - 0.5 |
| Band gap MAE | ~1.03 | 0.8 - 1.3 |
| Band gap R² | ~0.34 | 0.2 - 0.4 |
| Throughput | ~41 g/s | 30 - 60 g/s |
| GPU memory | ~1536 MB | 1000 - 2000 MB |

---

## 9. Expected Runtime

### Phase Breakdown (~16 hours total)

| Phase | Time | % of Total |
|---|---|---|
| Dataset download + cleaning | ~15 min | 1.5% |
| Graph pre-caching (10k graphs) | ~27 min | 2.8% |
| Training (139 epochs at ~353 s/epoch) | ~13.6 hr | 86.0% |
| Test evaluation | ~5 min | 0.5% |
| Analysis + plotting | ~3 min | 0.3% |
| **Total (with all steps)** | **~15.8 hr** | **100%** |

### Per-Epoch Timing

| Component | Time |
|---|---|
| Training (8310 samples, batch=16, accum=2) | ~350 s |
| Validation (586 samples) | ~25 s |
| Checkpoint save (every 10 epochs) | ~5 s |
| Logging + plotting (every epoch) | ~1 s |
| **Total per epoch** | **~381 s** (6.4 min) |

### Scaling with Dataset Size

| Dataset Size | Epoch Time | Total Time (150 epochs) |
|---|---|---|
| 1,000 (smoketest) | ~45 s | ~1.9 hr |
| 3,635 (mid) | ~155 s | ~6.5 hr |
| 10,000 (full) | ~381 s | ~15.9 hr |

---

## 10. Checkpoint Format

Checkpoints are Python `dict` serialized with `torch.save()`:

```python
checkpoint = {
    # Training state
    "epoch": int,                        # Current epoch number
    "model": OrderedDict,                # model.state_dict()
    "optimizer": dict,                   # optimizer.state_dict()
    "scheduler": dict,                   # scheduler.state_dict() (optional)

    # Metrics
    "val_metrics": dict,                 # Validation metrics at this epoch
    "train_metrics": dict,               # Training metrics at this epoch
    "metrics": dict,                     # Legacy: combined metrics

    # Configuration
    "config": dict,                      # Full model configuration
    "gradnorm_weights": dict,            # GradNorm weights {task: weight}
    "optimizer": str,                    # Optimizer name
    "normalize_targets": bool,           # Normalization flag

    # Reproducibility
    "rng_states": {
        "torch_cpu": list,               # torch.get_rng_state() (as list)
        "torch_cuda": list,              # torch.cuda.get_rng_state() (as list)
        "numpy": tuple,                  # np.random.get_state()
        "random": tuple,                 # random.getstate()
    },
    "scaler_state": dict,               # GradScaler.state_dict() (optional)
    "best_val_loss": float,             # Best validation loss so far

    # Dataset info
    "train_samples": int,               # Number of training samples
    "val_samples": int,                 # Number of validation samples
    "test_samples": int,                # Number of test samples
}
```

### Checkpoint Loading Example

```python
import torch

# Load checkpoint
ckpt = torch.load("runs/SL-20260708-001/checkpoints/best_val_loss.pt", map_location="cuda")

# Extract configuration
model_cfg = ckpt["config"]["model"]      # Model hyperparameters
val_metrics = ckpt["val_metrics"]        # Validation metrics
epoch = ckpt["epoch"]                    # Epoch number

# Reconstruct model
from src.models.scandium_model import ScandiumPINNGNN
model = ScandiumPINNGNN(**model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

# Restore RNG (for exact reproduction)
rng = ckpt.get("rng_states", {})
if rng:
    import numpy as np
    import random
    torch.set_rng_state(bytes(rng["torch_cpu"]))
    if "torch_cuda" in rng and rng["torch_cuda"]:
        torch.cuda.set_rng_state(bytes(rng["torch_cuda"]))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["random"])
```

---

## 11. Evaluation and Analysis

### 11.1 Test Evaluation

After training completes, test metrics are automatically computed and saved to `test_results.json`:

```bash
python -c "
import json
with open('runs/SL-20260708-001/test_results.json') as f:
    results = json.load(f)
for task, metrics in results.items():
    print(f'{task}:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
"
```

### 11.2 Cross-Validation

```bash
# 5-fold cross-validation
python scripts/evaluate/cross_validate.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000 \
    --output outputs/cross_val_results
```

### 11.3 Throughput Benchmark

```bash
python scripts/maintenance/benchmark_throughput.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000
```

### 11.4 Inference on Custom Structures

```bash
# Single structure prediction
python -c "
from src.inference.engine import InferenceEngine
from pymatgen.core import Structure

engine = InferenceEngine(
    model_path='runs/SL-20260708-001/checkpoints/best_val_loss.pt',
    device='cuda',
    use_mc_dropout=True,
)

# Known SSE: Li6PS5Cl
structure = Structure.from_file('examples/Li6PS5Cl.cif')
result = engine.predict_single(structure)

for task, pred in result.items():
    if isinstance(pred, dict):
        print(f'{task}: {pred.get(\"value\", \"N/A\"):.4f} ± {pred.get(\"uncertainty\", \"N/A\")}')
"
```

### 11.5 Experiment Leaderboard

View all experimental runs ranked by composite score:

```bash
cat runs/SL-20260708-001/EXPERIMENT_LEADERBOARD.md
```

---

## 12. Verification Checklist

Use this checklist to confirm each step of the reproducibility pipeline produces expected results.

### Data Pipeline

- [ ] Materials Project API key is set (`echo $MP_API_KEY`).
- [ ] Dataset download completes without errors.
- [ ] `dataset_cache.pt` contains 10,000 entries.
- [ ] `split_indices.pt` has train=8310, val=586, test=1104.
- [ ] `normalizer.json` has mean/std for 3 targets.
- [ ] Graph cache directory has 10,000 .pt files.
- [ ] All targets have correct dtypes (float32).

### Environment

- [ ] `python --version` shows 3.12.13.
- [ ] `python -c "import torch; print(torch.__version__)"` shows 2.6.0.
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` is `True`.
- [ ] `make test` passes (all tests green).
- [ ] `make lint` passes (no ruff errors).

### Training

- [ ] Training script starts without import errors.
- [ ] DataLoader loads batches correctly (no graph construction errors).
- [ ] Model forward pass completes without NaNs.
- [ ] Loss decreases over first 10 epochs.
- [ ] GradNorm weights update every 50 batches.
- [ ] Checkpoints are saved at the configured interval.
- [ ] Validation metrics are computed every epoch.
- [ ] Training completes or early stops within 150 epochs.

### Expected Metric Verification

| Check | Expected | Your Result | Pass? |
|---|---|---|---|
| Final train loss | ~1.86 | | |
| Best val loss | ~3.09 | | |
| Ef MAE | ~0.52 | | |
| Ef R² | ~0.58 | | |
| EaH MAE | ~0.13 | | |
| BG MAE | ~1.03 | | |
| Training time | ~16 hr | | |
| Epochs | ~139 | | |

### Outputs

- [ ] `TRAINING_SUMMARY.md` shows epoch-by-epoch progress.
- [ ] `FINAL_REPORT.md` has final metrics and analysis.
- [ ] `test_results.json` contains test set metrics.
- [ ] Plots are generated in `plots/` directory.
- [ ] Scorecard shows 5/6 or 6/6 checks passed.

---

## 13. Reproducing Specific Experiments

### Experiment SL-20260708-001 (Current Best)

```bash
# Step 1: Verify dataset
ls datasets/v3_li_10000/dataset_cache.pt
ls datasets/v3_li_10000/split_indices.pt

# Step 2: Train (will produce SL-20260708-NNN)
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000

# Note: New run will have a different ID (SL-20260708-002, etc.)
# Metrics should be comparable to SL-20260708-001
```

### Experiment SL-20260707-001 (No GradNorm)

```bash
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --data-dir datasets/v3_li_10000 \
    --no-gradnorm
```

### Experiment with Cosine Scheduler

```bash
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li_with_scheduler.yaml \
    --data-dir datasets/v3_li_10000
```

### Full Automated Reproduction

```bash
# Runs the complete pipeline: install → verify → test → train → evaluate
bash reproduce.sh datasets/v3_li_10000 configs/model_config_v3_li.yaml
```

---

## 14. Troubleshooting

### Common Issues and Solutions

#### CUDA Out of Memory

```bash
# Symptom: torch.cuda.OutOfMemoryError during training
# Solution: Reduce batch size or enable gradient checkpointing
```

Edit config:
```yaml
training:
  batch_size: 8           # Reduced from 16
  gradient_accumulation_steps: 4  # Increased from 2 (effective batch = 32)
model:
  use_gradient_checkpointing: true  # Must be enabled for 4 GB GPUs
```

#### DataLoader Multiprocessing Error

```bash
# Symptom: RuntimeError: DataLoader worker (pid X) is killed by signal
# Solution: Set multiprocessing_context='fork'
```

This is already configured in the training script:
```python
loader_kwargs = dict(
    collate_fn=collate_fn,
    pin_memory=True,
    multiprocessing_context="fork",  # Required for Python 3.14+
)
```

#### NaN Loss During Training

```bash
# Symptom: Loss becomes NaN after some epochs
# Solutions:
# 1. Reduce learning rate
# 2. Enable gradient clipping (already set to 1.0)
# 3. Check for NaN in training data
# 4. Reduce AMP mixed precision (AMP can cause NaNs on some GPUs)
```

#### Missing Normalizer File

```bash
# Symptom: FileNotFoundError: normalizer.json
# Solution: Ensure normalizer.json is in the data directory
ls datasets/v3_li_10000/normalizer.json
```

If missing, run the dataset build pipeline again with `--normalize` flag.

#### Slow Graph Building

```bash
# Symptom: First epoch takes >30 minutes
# Solution: Pre-cache graphs
python scripts/preprocess/cache_graphs.py \
    --data-dir datasets/v3_li_10000 \
    --num-workers 1
```

#### W&B Logging Not Working

```bash
# Symptom: wandb.init() fails
# Solution: Disable W&B in config
```

```yaml
logging:
  wandb: false    # Disable Weights & Biases
  tensorboard: false
```

#### Checkpoint Loading Mismatch

```bash
# Symptom: RuntimeError: Error(s) in loading state_dict
# Solutions:
# 1. Ensure config matches checkpoint (same hidden_dim, layers, etc.)
# 2. Use strict=False to load partial weights
# 3. Verify checkpoint was saved with compatible model version
```

---

## Appendix A: Environment Snapshot (Development Machine)

```yaml
OS: Linux 7.0.0-22-generic (Fedora-like)
GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM, CUDA 12.4)
CPU: x86_64, 8 cores
RAM: 14 GB
Python: 3.12.13 (main, May 6 2026)
PyTorch: 2.6.0+cu124
CUDA: 12.4
torch-geometric: 2.6.1
mp-api: 0.44.0
pymatgen: 2025.0
numpy: 1.26.0
scikit-learn: 1.5.0
```

## Appendix B: Reproducibility Statement

The Scandium Labs team is committed to reproducible research. To the best of our knowledge, the results presented in this repository can be reproduced by following this guide on equivalent hardware. However, we note the following sources of non-determinism:

1. **GPU non-determinism**: CUDA operations are not bitwise reproducible across different GPU architectures or CUDA versions.
2. **DataLoader shuffle**: Python's hash randomization affects DataLoader shuffle order.
3. **AMP (mixed precision)**: FP16 tensor operations have lower precision and may lead to slightly different loss values.
4. **MP API updates**: The Materials Project database is updated periodically, changing the available data.
5. **PyTorch version**: Operations may produce slightly different results across PyTorch versions.

To mitigate these issues, all checkpoints include RNG state snapshots, and the primary training script (`train_v3_li.py`) supports exact resumption from any checkpoint.
