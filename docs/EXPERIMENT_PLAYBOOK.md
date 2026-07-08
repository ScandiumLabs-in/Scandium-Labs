# Experiment Playbook

> Standard operating procedures for running, monitoring, comparing, and
> reproducing training experiments at Scandium Labs.
>
> **Last updated:** July 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Step-by-Step Experiment Workflow](#2-step-by-step-experiment-workflow)
3. [Monitoring Experiments](#3-monitoring-experiments)
4. [Comparing Runs](#4-comparing-runs)
5. [Resuming Interrupted Training](#5-resuming-interrupted-training)
6. [Testing the Best Model](#6-testing-the-best-model)
7. [Best Practices](#7-best-practices)
8. [Quick Reference Cards](#8-quick-reference-cards)

---

## 1. Overview

Every experiment at Scandium Labs follows a standard lifecycle:

```
1. Create Config ──▶ 2. Train ──▶ 3. Monitor ──▶ 4. Analyze ──▶ 5. Test
       │                                                           │
       └────────────────── 6. Compare (← previous runs) ────────────┘
```

The experiment management system (`src/training/experiment_tracker.py`) produces:

| Artifact | Location | Format |
|----------|----------|--------|
| Per-epoch metrics | `runs/SL-YYYYMMDD-NNN/epoch_metrics.json` | JSON array |
| Checkpoints | `runs/SL-YYYYMMDD-NNN/checkpoints/*.pt` | PyTorch serialized |
| Plots | `runs/SL-YYYYMMDD-NNN/plots/*.png` | matplotlib PNG |
| Reports | `runs/SL-YYYYMMDD-NNN/*_REPORT.md` | Markdown |
| Run registry | `runs/index.csv` | CSV leaderboard |

---

## 2. Step-by-Step Experiment Workflow

### Step 1: Create a Configuration YAML

Every experiment needs a configuration file. Start from the base config and create a derivative:

```bash
# Copy the base config
cp configs/model_config_v3_li.yaml configs/experiments/exp_b_gradnorm.yaml

# Edit hyperparameters
vim configs/experiments/exp_b_gradnorm.yaml
```

**Common configuration changes:**

```yaml
# Experiment B: GradNorm with cosine scheduler
model:
  hidden_dim: 128
  num_alignn_layers: 4
  num_transformer_layers: 2
  dropout: 0.15
  use_gradient_checkpointing: true
  use_two_stage_eah: true

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 0.0005
  max_epochs: 150
  patience: 40
  scheduler: "cosine_with_restarts"   # ← Enable scheduler

gradnorm:
  enabled: true                        # ← GradNorm enabled
  alpha: 1.5

bucketing:
  enabled: true
  bucket_size_mult: 2.0
```

**Config validation checklist:**

- [ ] `model.hidden_dim` is set (default 128)
- [ ] `model.num_alignn_layers` ≥ 1 (default 4)
- [ ] `training.batch_size` × `gradient_accumulation_steps` = effective batch (32)
- [ ] `training.max_epochs` × patience ratio ≥ 3 (e.g., 150/40)
- [ ] `gradnorm.alpha` between 0.5 and 2.0
- [ ] Dataset path exists in `training.dataset` or `--data-dir`

### Step 2: Run the Training Script

```bash
# Basic training
python scripts/train/train_v3_li.py \
    --config configs/experiments/exp_b_gradnorm.yaml \
    --out-dir checkpoints/exp_b \
    --data-dir datasets/v3_li_10000
```

**What happens:**

1. `ExperimentTracker` allocates a run ID (`SL-YYYYMMDD-NNN`)
2. Dataset is loaded from cache (`datasets/v3_li_10000/dataset_cache.pt`)
3. Graphs are split into train/val/test using `split_indices.pt`
4. Model is initialized (1.28M params, 4.9 MB)
5. Training loop begins with logging every epoch
6. Checkpoints saved to `runs/SL-YYYYMMDD-NNN/checkpoints/`
7. On completion, test evaluation runs and results are saved

**Expected console output:**

```
Device: cuda  Workers: 4
Loading dataset_cache.pt...
Structures: 10000
Dataset: 10000 samples
Precomputing graph sizes for bucketed batching...
Train: 8310, Val: 586, Test: 1104
  GC=auto: VRAM=3.9GB → enable
Model: 1,281,321 params (fresh init)
Experiment: SL-20260708-001 → runs/SL-20260708-001
  GradNorm enabled (alpha=1.5)
  Using CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2)

Training from epoch 0 to 149 (patience=40)...
  Epoch   0: train=1.2345 val=0.9876 w=[1.00/1.00/0.40] (45s)
  Epoch   5: train=0.4567 val=0.3456 w=[1.05/0.98/0.38] (240s)
  Epoch  10: train=0.3345 val=0.2890 w=[1.12/0.95/0.35] (435s)
  ...
  Epoch  50: train=0.1234 val=0.1089 w=[1.45/0.82/0.28] (2205s)
  ...

Stopped at epoch 87
Reason: Validation loss did not improve for 40 epochs.
Best epoch: 47

Training complete in 3880s

TEST EVALUATION
======================================================================
                 Task       MAE↓        R²↑      RMSE↓       Bias
-------------------------------------------------------------------
      formation_energy     0.0421     0.9623     0.0612    -0.0012
      energy_above_hull    0.0893     0.8734     0.1342     0.0031
              band_gap     0.2154     0.7541     0.3456    -0.0089

  TWO-STAGE EAH: F1=0.8712 MAE=0.0921
Results saved to checkpoints/exp_b/
Experiment tracker finalized: runs/SL-20260708-001
```

### Step 3: Optional Arguments

```bash
# Resume from checkpoint
python scripts/train/train_v3_li.py \
    --resume runs/SL-20260708-001/checkpoints/last.pt

# Disable GradNorm for ablation
python scripts/train/train_v3_li.py \
    --config configs/model_config_v3_li.yaml \
    --out-dir checkpoints/no_gn_ablation \
    --no-gradnorm

# Custom dataset path
python scripts/train/train_v3_li.py \
    --data-dir /mnt/data/datasets/v3_li_extended \
    --out-dir checkpoints/extended_data

# Debug: small run (override config via CLI — edit config file instead for proper tracking)
```

---

## 3. Monitoring Experiments

### 3.1 Live Console Logging

```bash
# Watch the training log
tail -f logs/training.log
```

**Sample log lines:**

```
2026-07-08 14:30:22,123 - __main__ - INFO - Epoch   5: train=0.4567 val=0.3456 w=[1.05/0.98/0.38] (195s)
2026-07-08 14:33:57,456 - __main__ - INFO - Epoch   6: train=0.4231 val=0.3210 w=[1.08/0.96/0.36] (215s)
```

### 3.2 Live Epoch Metrics

```bash
# Watch the metrics file grow
tail -f runs/SL-20260708-001/epoch_metrics.json | jq '.[-1]'
```

```json
{
  "epoch": 5,
  "train_loss": 0.4567,
  "val_loss": 0.3456,
  "tasks": {
    "formation_energy": { "mae": 0.089, "r2": 0.85, "rmse": 0.12 },
    "energy_above_hull": { "mae": 0.145, "r2": 0.72, "rmse": 0.21 },
    "band_gap": { "mae": 0.312, "r2": 0.58, "rmse": 0.42 }
  },
  "gradnorm_weights": {
    "formation_energy": 1.05,
    "energy_above_hull": 0.98,
    "band_gap": 0.38
  },
  "system": {
    "lr": 0.0005,
    "grad_norm": 1.234,
    "epoch_time_s": 195.3,
    "throughput": 12.8,
    "gpu_memory_mb": 470.0
  }
}
```

### 3.3 Real-Time Plot Monitoring

Plots are generated automatically at the end of training at `runs/SL-YYYYMMDD-NNN/plots/`:

| Plot | File | What to Look For |
|------|------|------------------|
| Loss curve | `loss_curve.png` | Train/val loss converging; gap not widening (overfitting) |
| MAE curve | `mae_curve.png` | Per-task MAE decreasing and stabilizing |
| R² curve | `r2_curve.png` | Per-task R² increasing toward 1.0 |
| GradNorm weights | `gradnorm_weights.png` | Weights adapting; should stabilize, not oscillate |
| Throughput | `throughput.png` | Should be stable; drops indicate I/O or GPU bottleneck |
| GPU memory | `gpu_memory_mb.png` | Should be stable; spikes indicate memory leak |
| Confusion matrix | `confusion_matrix.png` | Stability classifier performance |
| ROC/PR curves | `roc_pr_curves.png` | AUC > 0.85 desirable for stability |

### 3.4 GPU Monitoring

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi

# Expected:
# GPU-Util: 85-99%  (GPU-bound = good)
# Mem-Usage: 470MB / 4096MB  (with GC enabled)
# Volatile GPU-Util: 85-99%
```

**Signs of trouble:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU-Util < 50% | DataLoader bottleneck | Increase `num_workers`, check I/O |
| VRAM OOM | Too large batch/model | Reduce batch size, enable GC |
| Temperature > 85°C | Thermal throttling | Clean fans, reduce ambient temp |
| Power limit reached | Power supply limit | Reduce power limit with `nvidia-smi -pl` |

### 3.5 WandB Dashboard (if enabled)

When `logging.wandb: true` in config, metrics are streamed to Weights & Biases:

```bash
# Set your API key
export WANDB_API_KEY=your_key_here

# The training script will auto-create a W&B run
# View at: https://wandb.ai/scandium-labs/scandium-labs
```

---

## 4. Comparing Runs

### 4.1 Using the Analysis Script

```bash
# Basic analysis of a completed run
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --output reports/exp_b_analysis

# Compare against a baseline
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --baseline runs/SL-20260701-007/test_results.json \
    --output reports/exp_b_vs_baseline

# Full analysis with prediction diagnostics
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --checkpoint runs/SL-20260708-001/checkpoints/best_val_loss.pt \
    --output reports/exp_b_deep_analysis
```

**Analysis outputs (`reports/exp_b_analysis/`):**

| File | Description |
|------|-------------|
| `FINAL_REPORT.md` | Comprehensive 10-section report |
| `SCORECARD.md` | Pass/fail checklist (training completed, NaN checks, etc.) |
| `learning_curves.png` | Train/val loss with best epoch and patience window |
| `per_task_mae.png` | Per-task MAE over epochs |
| `per_task_r2.png` | Per-task R² over epochs |
| `gradnorm_weights.png` | GradNorm weight trajectories |
| `gradnorm_correlation.png` | Weight vs performance scatter plots |
| `system_metrics.png` | Throughput, GPU memory, epoch time |
| `resume_audit.png` | Only if run was resumed — consistency check |
| `training_timeline.png` | Visual timeline of training events |
| `pred_vs_actual.png` | Scatter plots of predicted vs true values (requires `--checkpoint`) |
| `residual_histograms.png` | Residual distributions (requires `--checkpoint`) |

### 4.2 Using the Experiment Leaderboard

The experiment tracker maintains a running leaderboard at `runs/index.csv` and per-run markdown at `runs/SL-YYYYMMDD-NNN/EXPERIMENT_LEADERBOARD.md`:

```bash
# View the leaderboard
cat runs/SL-20260708-001/EXPERIMENT_LEADERBOARD.md
```

**Sample output:**

```
| Rank | Run ID | Ef MAE ↓ | Ef R² ↑ | EaH MAE ↓ | EaH R² ↑ | BG MAE ↓ | BG R² ↑ | Score ↑ |
|------|--------|----------|---------|-----------|---------|----------|---------|---------|
| 1 | SL-20260708-001 | 0.0421 | 0.9623 | 0.0893 | 0.8734 | 0.2154 | 0.7541 | 0.8633 |
| 2 | SL-20260701-007 | 0.0489 | 0.9512 | 0.0945 | 0.8610 | 0.2234 | 0.7412 | 0.8511 |
```

### 4.3 Manual Comparison

```bash
# Compare test results across runs
for run in runs/SL-*; do
    if [ -f "$run/test_results.json" ]; then
        echo "=== $(basename $run) ==="
        python -c "
import json
with open('$run/test_results.json') as f:
    data = json.load(f)
for task in ['formation_energy', 'energy_above_hull', 'band_gap']:
    t = data.get(task, {})
    print(f'  {task}: MAE={t.get(\"mae\"):.4f}  R²={t.get(\"r2\"):.4f}')
"
    fi
done
```

---

## 5. Resuming Interrupted Training

### 5.1 When to Resume

Resume training if:

- Training crashed due to OOM, power loss, or timeout
- You want to extend training beyond `max_epochs`
- You want to continue with a lower LR after convergence

### 5.2 How to Resume

```bash
# Find the latest checkpoint
ls runs/SL-20260708-001/checkpoints/last.pt

# Resume training
python scripts/train/train_v3_li.py \
    --resume runs/SL-20260708-001/checkpoints/last.pt
```

**What gets restored:**

| Component | Restored? | Details |
|-----------|-----------|---------|
| Model weights | Yes | Full state_dict |
| Optimizer state | Yes | AdamW momentum, variance |
| LR scheduler state | Yes | Cosine restart phase |
| GradScaler state | Yes | AMP scale factor |
| GradNorm weights | Yes | Per-task adaptive weights |
| RNG state | Yes | torch, CUDA, numpy, random |
| Epoch counter | Yes | Continues from `checkpoint.epoch + 1` |
| Best val loss | Yes | Tracked across resume boundary |
| Experiment directory | Yes | Same `runs/SL-*` — all artifacts collocated |

### 5.3 Resume Verification

After resuming, verify continuity by checking:

1. **Loss continuity** — val_loss should not jump significantly after resume
2. **LR schedule** — learning rate should continue from where it left off
3. **GradNorm weights** — weights should continue their trajectory
4. **Metrics** — MAE/R² should continue improving without regression

The analysis script automatically detects resume points and produces a resume audit:

```bash
python scripts/analyze/analyze_training.py --run runs/SL-20260708-001 --output reports/audit
```

Look for `resume_audit.png` — it shows a green checkmark for each consistency check.

### 5.4 Resume from Legacy Checkpoints

You can also resume from the simpler checkpoints saved to `--out-dir`:

```bash
python scripts/train/train_v3_li.py \
    --resume checkpoints/my_exp/epoch_10.pt
```

Note: legacy checkpoints may not have the full optimizer/scheduler state. The training loop will handle gracefully.

---

## 6. Testing the Best Model

### 6.1 Test Evaluation During Training

The training script automatically evaluates the best model on the held-out test set:

```bash
# Results are saved to
cat checkpoints/exp_b/test_results.json
```

### 6.2 Standalone Inference on Candidates

```bash
# Screen new candidates
python scripts/inference/screen_candidates.py \
    --model runs/SL-20260708-001/checkpoints/best_val_loss.pt \
    --input data/candidate_structures.json \
    --output reports/screening_results.json \
    --top_k 20 \
    --temperature 300.0
```

### 6.3 Programmatic Testing

```python
from src.inference.engine import InferenceEngine
from src.inference.ranking import ParetoRanker
from pymatgen.core import Structure

engine = InferenceEngine(
    model_path="runs/SL-20260708-001/checkpoints/best_val_loss.pt",
)

# Test known compounds
test_compounds = [
    "Li6PS5Cl",    # Known superionic conductor
    "Li3YCl6",     # Halide electrolyte
    "Li7La3Zr2O12" # Garnet electrolyte (LLZO)
]

for formula in test_compounds:
    # You'd need the structure file; here assuming .cif path
    struct = Structure.from_file(f"data/test_structures/{formula}.cif")
    result = engine.predict_single(struct)
    print(f"{formula}: σ={result['ionic_conductivity']['value']:.2e} S/cm, "
          f"EaH={result['energy_above_hull']['value']:.3f} eV/atom, "
          f"rec={result['recommendation']}")
```

---

## 7. Best Practices

### 7.1 Experiment Hygiene

| Rule | Rationale |
|------|-----------|
| **Always use a new out-dir for new experiments** | Prevents checkpoint confusion and accidental overwrites. Each run should have a unique `--out-dir` or rely on the auto-generated run directory. |
| **Save config alongside results** | The experiment tracker automatically copies `config.yaml` to `runs/SL-*/`. Always verify it's correct before training. |
| **Log all metrics** | The experiment tracker logs everything. Don't suppress logging. If adding custom metrics, pass them through `tracker.log_epoch()`. |
| **Run analysis after every experiment** | Always run `analyze_training.py` after completion. It generates the final report, leaderboard, and visualizations. |
| **Commit config changes to git** | Tag experiments with git commit hashes. The experiment tracker captures this automatically. |
| **One hypothesis per experiment** | Change only one hyperparameter or feature at a time for clean ablations. |

### 7.2 Configuration Management

```bash
# Name configs meaningfully
configs/experiments/
├── baseline.yaml           # Current best config
├── exp_b_scheduler.yaml    # Test scheduler effect
├── exp_c_lr_1e-3.yaml      # Test learning rate
└── exp_d_no_gradnorm.yaml  # Ablate GradNorm

# Version all configs in git
git add configs/experiments/
git commit -m "experiments: add baseline and scheduler sweep configs"
```

### 7.3 Dataset Hygiene

- **Never modify a dataset in place.** If you need to change the dataset, create a new version (e.g., `v3_li_10000`, `v4_li_20000`).
- **Rebuild the dataset with `scripts/preprocess/build_dataset.py`** when adding new data or changing filters.
- **Pre-cache graphs** with `scripts/preprocess/cache_graphs.py` to avoid the 29-minute first-epoch overhead.
- **Document dataset versions** in `datasets/<version>/metadata.json`.

### 7.4 Reproducibility

The experiment tracker ensures full reproducibility by capturing:

```yaml
# run_metadata.json captures:
- git_commit: "abc1234"
- git_branch: "main"
- python_version: "3.10.12"
- pytorch_version: "2.1.0"
- cuda_version: "12.1"
- gpu_name: "NVIDIA GeForce GTX 1650"
- config_fingerprint: "a1b2c3d4e5f6"  # SHA-256 of config
```

To reproduce an experiment:

```bash
# Checkout the exact commit
git checkout $(cat runs/SL-20260708-001/run_metadata.json | python -c "import sys,json; print(json.load(sys.stdin)['git_commit'])")

# Run with the exact config
python scripts/train/train_v3_li.py \
    --config runs/SL-20260708-001/config.yaml \
    --data-dir datasets/v3_li_10000
```

### 7.5 Experiment Naming Convention

Run IDs are auto-generated: `SL-YYYYMMDD-NNN`

| Component | Meaning |
|-----------|---------|
| `SL` | Scandium Labs prefix |
| `YYYYMMDD` | Date of experiment start |
| `NNN` | Sequential number within that day |

For named experiments (via `experiment_sweep.py`), use meaningful names:

```
experiments/
├── v2_baseline/
├── v2_gradnorm_alpha_1.5/
├── v3_li_two_stage_eah/
└── v3_li_cosine_scheduler/
```

### 7.6 When to Stop Training

| Signal | Action |
|--------|--------|
| Val loss has not improved for `patience` epochs | Let early stopping fire (automatic) |
| Val loss starts increasing for 5+ epochs | Stop early (overfitting) |
| GradNorm weights oscillate wildly | Reduce `gradnorm.alpha` or disable GradNorm |
| GPU memory full | Reduce batch size, enable gradient checkpointing |
| `NaN` in loss or gradients | Reduce LR, check data normalization |
| Throughput drops below 5 graphs/s | Optimize DataLoader (increase workers, check I/O) |

---

## 8. Quick Reference Cards

### 8.1 Training

```bash
# From scratch (default config)
python scripts/train/train_v3_li.py

# Custom config
python scripts/train/train_v3_li.py \
    --config configs/experiments/exp_b.yaml \
    --out-dir checkpoints/exp_b

# Resume
python scripts/train/train_v3_li.py \
    --resume runs/SL-20260708-001/checkpoints/last.pt

# GradNorm ablation
python scripts/train/train_v3_li.py --no-gradnorm

# Named experiment
python scripts/train/experiment_sweep.py \
    --config configs/model_config_v3.yaml \
    --data_dir datasets/v3_li_10000 \
    --name v3_two_stage_eah
```

### 8.2 Monitoring

```bash
tail -f logs/training.log
watch -n 1 nvidia-smi
tail -f runs/SL-*/epoch_metrics.json | jq '.[-1]'
```

### 8.3 Analysis

```bash
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --output reports/exp_b_analysis \
    --baseline runs/SL-20260701-007/test_results.json
```

### 8.4 Testing

```bash
# Screening
python scripts/inference/screen_candidates.py \
    --model runs/SL-20260708-001/checkpoints/best_val_loss.pt \
    --input candidates.json \
    --output results.json
```

### 8.5 Data

```bash
# Rebuild dataset
python scripts/preprocess/build_dataset.py \
    --api-key $MP_API_KEY \
    --output datasets/v4_li_20000

# Cache graphs
python scripts/preprocess/cache_graphs.py \
    --data-dir datasets/v3_li_10000
```
