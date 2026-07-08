# Experiment Tracking System

> Version 1.0 | Last updated: 2026-07-08

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. RunRegistry — Experiment Database](#2-runregistry--experiment-database)
  - [2.1 Run ID Format](#21-run-id-format)
  - [2.2 CSV Index](#22-csv-index)
  - [2.3 Registration Lifecycle](#23-registration-lifecycle)
  - [2.4 Cross-Experiment Search](#24-cross-experiment-search)
  - [2.5 W&B Integration](#25-wb-integration)
- [3. MetricsStore — Per-Run Persistence](#3-metricsstore--per-run-persistence)
  - [3.1 epoch_metrics.json](#31-epoch_metricsjson)
  - [3.2 epoch_metrics.csv](#32-epoch_metricscsv)
  - [3.3 Config Snapshots](#33-config-snapshots)
  - [3.4 Resume-Aware Loading](#34-resume-aware-loading)
  - [3.5 Best-Metric Tracking](#35-best-metric-tracking)
- [4. CheckpointManager — Model Persistence](#4-checkpointmanager--model-persistence)
  - [4.1 Checkpoint Types](#41-checkpoint-types)
  - [4.2 Full Resume State](#42-full-resume-state)
  - [4.3 Legacy Checkpoint Format](#43-legacy-checkpoint-format)
  - [4.4 Periodic Checkpointing](#44-periodic-checkpointing)
- [5. PlotGenerator — Automated Visualization](#5-plotgenerator--automated-visualization)
  - [5.1 Standard Plot Suite](#51-standard-plot-suite)
  - [5.2 Classification Diagnostics](#52-classification-diagnostics)
  - [5.3 Plot Configuration](#53-plot-configuration)
- [6. ExperimentTracker — The Orchestrator](#6-experimenttracker--the-orchestrator)
  - [6.1 Initialization](#61-initialization)
  - [6.2 Training Loop Integration](#62-training-loop-integration)
  - [6.3 Early Stopping](#63-early-stopping)
  - [6.4 Finalization](#64-finalization)
  - [6.5 Automatic Report Generation](#65-automatic-report-generation)
- [7. Directory Structure](#7-directory-structure)
- [8. How to Create and Track Experiments](#8-how-to-create-and-track-experiments)
- [9. How to Compare Experiments](#9-how-to-compare-experiments)
- [10. How to Resume Experiments](#10-how-to-resume-experiments)
- [11. Analysis Pipeline](#11-analysis-pipeline)
- [12. Best Practices](#12-best-practices)
- [13. Troubleshooting](#13-troubleshooting)

---

## 1. Overview

The Scandium Labs experiment tracking system is a research-grade, file-based experiment manager built for reproducibility, transparency, and minimal infrastructure dependencies. Unlike cloud-based trackers (MLflow, W&B) that require server infrastructure, this system uses a flat-file database with CSV indexing, JSON metrics dumps, and PyTorch checkpoint serialization — all stored under a local `runs/` directory.

The system comprises four core components:

| Component | File | Responsibility |
|-----------|------|----------------|
| `RunRegistry` | `src/training/experiment_tracker.py:46` | Run ID allocation, CSV index management, cross-experiment search |
| `MetricsStore` | `src/training/experiment_tracker.py:169` | Per-epoch metric accumulation, JSON + CSV persistence, best-value tracking |
| `CheckpointManager` | `src/training/experiment_tracker.py:258` | Checkpoint I/O, best-model selection, full-resume state management |
| `PlotGenerator` | `src/training/experiment_tracker.py:314` | Automated matplotlib visualization, classification diagnostics |
| `ExperimentTracker` | `src/training/experiment_tracker.py:510` | Orchestrator tying all components together, report generation |

The design philosophy prioritizes:

- **Zero infrastructure**: No databases, no servers, no API keys. Everything is files.
- **Human readability**: All artifacts are plain text (YAML, JSON, CSV, Markdown).
- **Resume safety**: Every component is designed to survive interruption and resume gracefully.
- **Self-containment**: Every run directory is a complete, portable artifact.

---

## 2. RunRegistry — Experiment Database

`RunRegistry` manages experiment identity and maintains a searchable index of all runs. It is initialized with a path to the `runs/` directory.

```python
from src.training.experiment_tracker import RunRegistry

registry = RunRegistry("runs")
run_id = registry.allocate_run_id()  # e.g., "SL-20260708-003"
```

### 2.1 Run ID Format

Run IDs follow the format `SL-YYYYMMDD-NNN`:

- `SL` — Scandium Labs prefix
- `YYYYMMDD` — Date of run creation
- `NNN` — Zero-padded sequence number within the day (001, 002, ...)

The `allocate_run_id()` method scans the `runs/` directory for existing runs matching today's date and assigns the next sequential number:

```python
def allocate_run_id(self) -> str:
    today = datetime.now().strftime("%Y%m%d")
    existing = list(self.runs_dir.glob(f"SL-{today}-*"))
    n = len(existing) + 1
    return f"SL-{today}-{n:03d}"
```

This format guarantees uniqueness within a day and provides human-readable chronological ordering. Cross-day collisions are impossible due to the date component.

### 2.2 CSV Index

The index is stored at `runs/index.csv` with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | str | Unique experiment identifier |
| `date` | ISO datetime | Registration timestamp |
| `dataset` | str | Dataset name (e.g., `v3_li_10000`) |
| `architecture` | str | Model architecture name |
| `hidden_dim` | int | Hidden dimension |
| `alignn_layers` | int | Number of ALIGNN layers |
| `transformer_layers` | int | Number of Transformer layers |
| `batch_size` | int | Training batch size |
| `best_mae_ef` | float | Best formation energy MAE |
| `best_r2_ef` | float | Best formation energy R² |
| `best_mae_eah` | float | Best energy-above-hull MAE |
| `best_r2_eah` | float | Best energy-above-hull R² |
| `best_mae_bg` | float | Best band gap MAE |
| `best_r2_bg` | float | Best band gap R² |
| `gpu_hours` | float | Total GPU-hours consumed |
| `status` | str | `running`, `completed`, `failed` |

The index is updated incrementally during training via `update_status()`, which rewrites the CSV in place. This is safe for concurrent reads but not for concurrent writes (a mutex is planned for distributed runs).

### 2.3 Registration Lifecycle

1. **Allocation**: `allocate_run_id()` returns a new unique ID.
2. **Registration**: `register(run_id, metadata)` appends a row with `status=running`.
3. **Live updates**: `update_status(run_id, **updates)` is called at each epoch to update best metrics and GPU-hours.
4. **Finalization**: `finalize()` sets `status=completed` and records final GPU-hours.
5. **Failure**: On crash, `status=running` indicates an incomplete run. The `load_all_results()` method only considers runs with valid `epoch_metrics.json`.

### 2.4 Cross-Experiment Search

`RunRegistry.load_all_results()` scans every subdirectory in `runs/` for `epoch_metrics.json`, extracts the best per-task metrics, and optionally scans `checkpoints/` for legacy `test_results.json` files. This powers the leaderboard and comparison tables in reports.

```python
def load_all_results(self) -> list[dict]:
    """Load test_results.json from every experiment run and checkpoint."""
    results = []
    for run_dir in sorted(self.runs_dir.iterdir()):
        metrics_path = run_dir / "epoch_metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text())
            best = self._best_from_epochs(data)
            results.append({"run_id": run_dir.name, **best})
    # Also scans checkpoints/ for legacy test_results.json
    ...
    return results
```

The `_best_from_epochs()` static method iterates all epochs to find the minimum MAE and maximum R² for each task, returning a flat dictionary suitable for leaderboard construction.

### 2.5 W&B Integration

The system includes hooks for Weights & Biases integration, controlled by `logging.wandb` in the config. When enabled:

- Run metadata (config, hyperparameters) is logged to W&B at init
- Per-epoch metrics (loss, MAE, R², throughput) are streamed to W&B
- Checkpoint artifacts can be logged to W&B for model registry
- Plots are uploaded as W&B images

W&B is optional and disabled by default (`logging.wandb: false` in config). The integration is lightweight: W&B calls are gated behind `if config.get("logging", {}).get("wandb", False)` checks, so there is no import-time dependency. To enable, install `wandb` and set `logging.wandb: true` in the config.

TensorBoard support follows the same pattern via `logging.tensorboard`.

---

## 3. MetricsStore — Per-Run Persistence

`MetricsStore` accumulates per-epoch training and validation metrics and persists them in two complementary formats: JSON for programmatic access and CSV for spreadsheet/analysis consumption.

### 3.1 epoch_metrics.json

The JSON file stores a list of epoch records, each containing:

```json
{
  "epoch": 98,
  "train_loss": 1.8623,
  "val_loss": 3.0941,
  "timestamp": "2026-07-08T14:32:15.123456",
  "epoch_time_s": 408.7,
  "system": {
    "lr": 0.00005,
    "grad_norm": 0.85,
    "epoch_time_s": 408.7,
    "throughput": 41.1,
    "gpu_memory_mb": 1535.9
  },
  "tasks": {
    "formation_energy": {
      "mae": 0.5508,
      "rmse": 0.7551,
      "r2": 0.5511,
      "pearson": 0.7421,
      "spearman": 0.5723,
      "bias": 0.0894
    },
    "energy_above_hull": {
      "mae": 0.1313,
      "rmse": 0.3628,
      "r2": 0.3735,
      "pearson": 0.6112,
      "spearman": 0.2891,
      "bias": -0.0412
    },
    "band_gap": {
      "mae": 1.0252,
      "rmse": 1.4109,
      "r2": 0.3385,
      "pearson": 0.5838,
      "spearman": 0.5720,
      "bias": 0.2104
    }
  },
  "gradnorm_weights": {
    "band_gap": 0.4,
    "energy_above_hull": 1.0,
    "formation_energy": 1.0
  }
}
```

Each record captures the complete training state at a given epoch. The JSON file grows linearly with epochs (typically ~3-5 KB per epoch) and is appended to via `add_epoch()`, which calls `_save()` after every append.

### 3.2 epoch_metrics.csv

The CSV format is a flattened version of the JSON, produced by the `_flatten()` method:

```python
@staticmethod
def _flatten(ep: dict) -> dict:
    flat = {}
    for k, v in ep.items():
        if k == "tasks":
            for task_name, task_data in v.items():
                for mk, mv in task_data.items():
                    flat[f"{task_name}_{mk}"] = mv
        elif isinstance(v, dict):
            for mk, mv in v.items():
                flat[f"{k}_{mk}"] = mv
        else:
            flat[k] = v
    return flat
```

This produces columns like:

```
epoch, train_loss, val_loss, epoch_time_s,
formation_energy_mae, formation_energy_r2, formation_energy_rmse, formation_energy_bias,
energy_above_hull_mae, energy_above_hull_r2, ...,
band_gap_mae, band_gap_r2, ...,
system_lr, system_grad_norm, system_throughput, system_gpu_memory_mb
```

The CSV is rewritten in full at every epoch (not appended), which is safe because the dataset is small (typically <200 rows). The header is generated from the most recent epoch's key set.

### 3.3 Config Snapshots

Every experiment stores a complete config snapshot at `runs/<run_id>/config.yaml`. This is written at initialization by `ExperimentTracker.__init__()`:

```python
config_path = self.run_dir / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

The config is also fingerprinted using SHA-256 in reports for quick reproducibility checks:

```python
def config_fingerprint(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
```

### 3.4 Resume-Aware Loading

When resuming, `MetricsStore.__init__()` loads the existing `epoch_metrics.json` and reconstructs the in-memory epoch list and best-metric cache:

```python
if self._json_path.exists():
    try:
        data = json.loads(self._json_path.read_text())
        if isinstance(data, list):
            self.epochs = data
            self._update_best_from_epochs()
    except (json.JSONDecodeError, ValueError):
        pass
```

This ensures that resumed training seamlessly appends to the existing metric history rather than starting from scratch.

### 3.5 Best-Metric Tracking

`MetricsStore` maintains an internal `_best` dictionary mapping metric names to `(value, epoch)` tuples. Updates follow a consistent comparison rule:

- **Lower-is-better** (MAE, RMSE, loss): `val < best_val`
- **Higher-is-better** (R², Pearson, Spearman): `val > best_val`

The `best` property and `get_best(key)` method expose live best values throughout training, which are consumed by the registry (for index updates), the checkpoint manager (for best-model selection), and the report generator.

---

## 4. CheckpointManager — Model Persistence

`CheckpointManager` maintains model checkpoints with a focus on research workflows: multiple best-metric trackpoints, periodic snapshots, and full resume state.

### 4.1 Checkpoint Types

The manager produces four categories of checkpoints:

| File Pattern | Trigger | Purpose |
|-------------|---------|---------|
| `last.pt` | Every epoch | Always-available latest state for resume |
| `epoch_NNN.pt` | Every `save_interval` epochs | Periodic snapshots for analysis |
| `best_{metric}.pt` | New best metric | Per-metric best models |
| `best_val_loss.pt` | New best val_loss | Primary best model |

For each task-metric pair (e.g., `formation_energy_mae`, `formation_energy_r2`), a separate best checkpoint is maintained:

```python
for task_name, task_data in val_metrics.get("tasks", {}).items():
    for metric in ["mae", "r2"]:
        val = task_data.get(metric)
        if val is None: continue
        key = f"{task_name}_{metric}"
        higher_better = metric == "r2"
        best_val, best_ep = self._best_metrics.get(key, (None, -1))
        is_better = (
            best_val is None
            or (higher_better and val > best_val)
            or (not higher_better and val < best_val)
        )
        if is_better:
            self._best_metrics[key] = (val, epoch)
            torch.save(state, str(self.ckpt_dir / f"best_{key}.pt"))
```

This produces up to 7 best checkpoints per run (1 val_loss + 3 tasks × 2 metrics each).

### 4.2 Full Resume State

Every checkpoint contains a complete snapshot for bitwise-exact resume:

```python
state = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "val_metrics": val_metrics,
    "config": extra or {},
}
```

The `config` field (populated from `extra`) stores the training configuration used for that checkpoint, enabling self-describing checkpoints that can be loaded without the original config file. This field also stores auxiliary state:

- **GradScaler state**: AMP gradient scaling parameters
- **RNG state**: `torch.random.get_rng_state()` and `numpy.random` state for deterministic resume
- **GradNorm weights**: Current GradNorm task weights and initial loss references
- **LR scheduler state**: Current learning rate and scheduler internal counters

The `ExperimentTracker.finalize()` method calls `save_checkpoint()` with this extra state:

```python
extra = {
    "config": self.config,
    "scaler_state_dict": self.scaler.state_dict(),
    "rng_state": torch.random.get_rng_state(),
    "np_rng_state": np.random.get_state(),
    "gradnorm_weights": {t: w.item() for t, w in gradnorm.weights.items()},
    "gradnorm_initial_losses": {t: v.item() for t, v in gradnorm._initial_losses.items()},
    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
}
```

### 4.3 Legacy Checkpoint Format

The system supports a legacy checkpoint format at `checkpoints/<name>/` (outside the `runs/` directory). These are organized as:

```
checkpoints/
  phase5_final/
    best_model.pt
    test_results.json
  final_eval/
    best_model.pt
    test_results.json
```

The `load_all_results()` method scans this directory for `test_results.json` and includes results in the leaderboard for backward compatibility with pre-tracking-system experiments.

### 4.4 Periodic Checkpointing

Controlled by `save_epoch_checkpoints` in the config (default: 0 = disabled). When set to a positive integer (e.g., 10), checkpoints are saved every N epochs:

```python
if self.save_interval > 0 and epoch % self.save_interval == 0:
    torch.save(state, str(self.ckpt_dir / f"epoch_{epoch:03d}.pt"))
```

Periodic checkpoints are useful for:
- Analyzing model behavior at specific training stages
- Creating ensemble models from different epochs
- Debugging training anomalies by examining intermediate states

---

## 5. PlotGenerator — Automated Visualization

`PlotGenerator` produces a comprehensive suite of diagnostic plots at the end of training. All plots are saved to `runs/<run_id>/plots/` as 150 DPI PNG files.

### 5.1 Standard Plot Suite

The `generate_all()` method calls five internal plot generators:

**Loss Curve (`loss_curve.png`)**
- Train loss and validation loss over epochs
- Green dotted vertical line at best val_loss epoch
- Green star marker at best val_loss value
- Optional orange shaded patience window

**MAE Curve (`mae_curve.png`)**
- Per-task MAE evolution for all three tasks (Ef, EaH, BG)
- Each task plotted in its own color with star markers at best value

**R² Curve (`r2_curve.png`)**
- Per-task R² evolution, mirroring the MAE layout
- Higher-is-better, star markers at maximum

**System Metrics (individual PNGs)**
- `lr.png`: Learning rate schedule
- `grad_norm.png`: Gradient norm history
- `epoch_time_s.png`: Per-epoch duration
- `throughput.png`: Graphs/second throughput
- `gpu_memory_mb.png`: GPU memory consumption

**GradNorm Weights (`gradnorm_weights.png`)**
- Trajectories of GradNorm task weights over training
- One line per task, with 5-epoch moving average overlay
- Vertical line at best validation epoch

### 5.2 Classification Diagnostics

When validation epoch data is provided via `log_val_epoch_data()`, three additional plots are generated:

**Confusion Matrix (`confusion_matrix.png`)**
- 2×2 matrix for stable/unstable classification
- Derived from the two-stage EaH head's binary predictions

**ROC/PR Curves (`roc_pr_curves.png`)**
- Side-by-side ROC curve (with AUC) and Precision-Recall curve (with AUC)
- Computed from continuous `p_unstable` scores

**Reliability Diagram (`calibration.png`)**
- 10-bin calibration curve with Expected Calibration Error (ECE)
- Perfect calibration diagonal shown as dashed line
- Red fill highlights miscalibration regions

### 5.3 Plot Configuration

Plot generation is configurable:

- **Enable/disable**: `enable_plots` constructor parameter
- **Matplotlib backend**: Automatically set to `Agg` for headless environments
- **Import guard**: Gracefully degrades if matplotlib is not installed
- **Resolution**: 150 DPI, `tight_layout()` for clean margins

---

## 6. ExperimentTracker — The Orchestrator

`ExperimentTracker` ties together all components and exposes a high-level API for the training loop.

### 6.1 Initialization

```python
tracker = ExperimentTracker(
    config=config,                    # Full experiment configuration dict
    run_dir="runs",                   # Base directory for all runs
    save_epoch_checkpoints=10,        # Save periodic checkpoints every 10 epochs
    enable_plots=True,                # Generate plots on finalize()
    primary_metric="avg_r2",          # Metric for best-model selection
    resume_from=None,                 # Path to existing run dir for resume
)
```

On initialization:

1. **Allocate or resume**: If `resume_from` is `None`, allocate a new run ID and create the run directory. If `resume_from` is a path, load the existing run.
2. **Initialize components**: Create `MetricsStore`, `CheckpointManager`, and `PlotGenerator` instances.
3. **Collect metadata**: Record Python version, PyTorch version, CUDA version, GPU name, git commit, and config.
4. **Write config snapshot**: Save `config.yaml` to the run directory.
5. **Write metadata**: Save `run_metadata.json` with environment and configuration.
6. **Load previous results**: Scan all existing runs for leaderboard comparison.

For resumed runs, the `_training_t0` timer is restored from the existing `run_metadata.json` so that total GPU-hour accounting is cumulative.

### 6.2 Training Loop Integration

The tracker integrates with the training loop through four methods:

```python
# Per-epoch hook
tracker.start_epoch()   # Start the epoch timer

# End-of-epoch logging
tracker.log_epoch(
    epoch=epoch,
    train_loss=train_loss.item(),
    val_loss=val_loss.item(),
    val_metrics=metrics,           # dict of {task: {mae, rmse, r2, ...}}
    system={                       # System metrics dict
        "lr": scheduler.get_last_lr()[0],
        "grad_norm": grad_norm_val,
        "throughput": throughput_val,
        "gpu_memory_mb": mem_used,
    },
    gradnorm_weights=gradnorm_weights,  # Optional dict of {task: weight}
)

# Checkpoint saving
tracker.save_checkpoint(epoch, model, optimizer, val_metrics, extra={...})

# Val epoch data (for plots)
tracker.log_val_epoch_data({
    "eah_true": y_true_numpy,
    "p_unstable": p_unstable_numpy,
    "eah_pred_binary": binary_predictions,
})
```

### 6.3 Early Stopping

`ExperimentTracker.should_stop(patience, metric)` implements patience-based early stopping:

```python
def should_stop(self, patience: int, metric: str = "val_loss") -> bool:
    best_val, best_ep = self.metrics.get_best(metric)
    if best_val is None:
        return False
    last_epoch = self.metrics.epochs[-1]["epoch"] if self.metrics.epochs else 0
    return (last_epoch - best_ep) >= patience
```

The `early_stop_report()` method generates a human-readable stopping explanation.

The `_write_stop_report()` method in finalize generates `STOP_REPORT.md` with stop reason, epoch, and best val_loss.

### 6.4 Finalization

`finalize(test_results)` is called after training completes:

1. Compute total elapsed time and store in metadata
2. Save final `run_metadata.json`
3. Save `test_results.json` if provided
4. Generate all plots via `PlotGenerator.generate_all()`
5. Write reports: `BEST_MODEL_REPORT.md`, `MODEL_CARD.md`, `EXPERIMENT_LEADERBOARD.md`, benchmark tables, `STOP_REPORT.md`
6. Update registry with completion status and final GPU-hours

### 6.5 Automatic Report Generation

The tracker produces six automatic reports:

| Report | Trigger | Content |
|--------|---------|---------|
| `TRAINING_SUMMARY.md` | Every epoch (via `_write_summary`) | Current and best metrics, comparison table |
| `BEST_MODEL_REPORT.md` | Finalization | Test set results, best epochs, training summary |
| `MODEL_CARD.md` | Finalization | Full model description, intended use, limitations |
| `EXPERIMENT_LEADERBOARD.md` | Finalization | Ranked comparison of all experiments |
| `STOP_REPORT.md` | Finalization | Stop reason, best epoch, val_loss |
| `tables/benchmark.*` | Finalization | Benchmark tables in MD, CSV, and LaTeX formats |

The `_write_leaderboard()` method computes a composite score (average R² across tasks) and ranks all experiments:

```python
def composite_score(r):
    r2s = [r.get(f"{t}_r2") for t in ["formation_energy", "energy_above_hull", "band_gap"]]
    r2s = [v for v in r2s if v is not None]
    return sum(r2s) / len(r2s) if r2s else -999
```

Benchmark tables are exported in three formats:
- **Markdown**: For GitHub rendering
- **CSV**: For spreadsheet import
- **LaTeX**: For paper inclusion (`booktabs` format)

---

## 7. Directory Structure

Every experiment produces a self-contained directory with the following structure:

```
runs/
  index.csv                                         # Global experiment index
  SL-YYYYMMDD-NNN/                                  # Run directory
    config.yaml                                     # Experiment configuration
    run_metadata.json                               # Environment and runtime metadata
    epoch_metrics.json                              # Per-epoch metrics (JSON)
    epoch_metrics.csv                               # Per-epoch metrics (CSV)
    TRAINING_SUMMARY.md                             # Live training summary
    BEST_MODEL_REPORT.md                            # Best model analysis
    MODEL_CARD.md                                   # Model documentation
    EXPERIMENT_LEADERBOARD.md                       # Cross-experiment ranking
    STOP_REPORT.md                                  # Training stop reason
    test_results.json                               # Test set evaluation results
    checkpoints/
      last.pt                                       # Latest epoch checkpoint
      best_val_loss.pt                              # Best validation loss checkpoint
      best_formation_energy_mae.pt                  # Per-metric best checkpoints
      best_formation_energy_r2.pt
      best_energy_above_hull_mae.pt
      best_energy_above_hull_r2.pt
      best_band_gap_mae.pt
      best_band_gap_r2.pt
      epoch_000.pt                                  # Periodic checkpoints
      epoch_010.pt
      epoch_020.pt
      ...
    plots/
      loss_curve.png                                # Loss curves
      mae_curve.png                                 # Per-task MAE
      r2_curve.png                                  # Per-task R²
      lr.png                                        # Learning rate schedule
      grad_norm.png                                 # Gradient norm
      epoch_time_s.png                              # Epoch duration
      throughput.png                                # Throughput
      gpu_memory_mb.png                             # GPU memory
      gradnorm_weights.png                          # GradNorm weight trajectories
      confusion_matrix.png                          # Stability confusion matrix
      roc_pr_curves.png                             # ROC and PR curves
      calibration.png                               # Reliability diagram
    tables/
      benchmark.md                                  # Benchmark tables (MD)
      benchmark.csv                                 # Benchmark tables (CSV)
      benchmark.tex                                 # Benchmark tables (LaTeX)
    analysis/                                       # Created by analyze_training.py
      FINAL_REPORT.md                               # Comprehensive analysis
      SCORECARD.md                                  # Experiment quality scorecard
      learning_curves.png                           # Post-hoc analysis plots
      per_task_mae.png
      per_task_rmse.png
      per_task_r2.png
      gradnorm_weights.png
      gradnorm_correlation.png
      system_metrics.png
      training_timeline.png
      resume_audit.png                              # Resume consistency check
      pred_vs_actual.png                            # Prediction diagnostics
      residual_histograms.png
```

---

## 8. How to Create and Track Experiments

### Basic Usage

```python
from src.training.experiment_tracker import ExperimentTracker
import yaml

# Load configuration
with open("configs/model_config_v3_li.yaml") as f:
    config = yaml.safe_load(f)

# Initialize tracker
tracker = ExperimentTracker(
    config=config,
    run_dir="runs",
    save_epoch_checkpoints=10,
    enable_plots=True,
)

print(f"Run ID: {tracker.run_id}")  # e.g., "SL-20260708-003"

# Register model (adds parameter count to metadata)
tracker.register_model(model)

# Training loop
for epoch in range(config["training"]["max_epochs"]):
    tracker.start_epoch()

    # ... training code ...
    train_loss = ...
    val_loss, val_metrics = ...

    # Log epoch
    tracker.log_epoch(
        epoch=epoch,
        train_loss=train_loss.item(),
        val_loss=val_loss.item(),
        val_metrics=val_metrics,
        system={
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm_val,
            "throughput": throughput_val,
            "gpu_memory_mb": torch.cuda.max_memory_allocated() // 1024**2,
        },
        gradnorm_weights=gradnorm.weights_dict(),
    )

    # Save checkpoint
    tracker.save_checkpoint(epoch, model, optimizer, val_metrics, extra={...})

    # Early stopping
    if tracker.should_stop(patience=config["training"]["patience"]):
        print(tracker.early_stop_report(epoch, config["training"]["patience"]))
        break

# Finalize with test results
tracker.finalize(test_results=test_metrics)
```

### Command-Line Workflow

The recommended workflow uses `scripts/train/train_v3_li.py`:

```bash
# Run a new experiment
./venv/bin/python scripts/train/train_v3_li.py

# Monitor progress
cat runs/SL-20260708-003/TRAINING_SUMMARY.md

# Watch plots update
ls -la runs/SL-20260708-003/plots/
```

### Sweep-Based Workflow

Use `scripts/train/experiment_sweep.py` for hyperparameter sweeps:

```bash
# Launch a sweep over learning rates
./venv/bin/python scripts/train/experiment_sweep.py \
    --config configs/model_config_v3_li.yaml \
    --lr 0.0001,0.0005,0.001 \
    --output runs/sweep_lr_001
```

Each sweep experiment gets its own run ID and directory, and all results are aggregated in the central `index.csv`.

---

## 9. How to Compare Experiments

### Using the Leaderboard

The `EXPERIMENT_LEADERBOARD.md` file is generated automatically at finalization and ranks all experiments by composite R² score:

```markdown
| Rank | Run ID | Ef MAE ↓ | Ef R² ↑ | EaH MAE ↓ | EaH R² ↑ | BG MAE ↓ | BG R² ↑ | Score ↑ |
|------|--------|----------|---------|-----------|---------|----------|---------|---------|
| 1 | **SL-20260708-001** | 0.5222 | 0.5871 | 0.1280 | 0.3854 | 1.0252 | 0.3385 | 0.4370 |
| 2 | SL-20260701-007 | 0.5181 | 0.5897 | 0.1252 | 0.4227 | 1.0453 | 0.3060 | 0.4395 |
| 3 | SL-20260630-002 | 0.5684 | 0.5359 | 0.1256 | 0.3750 | 1.0479 | 0.2924 | 0.4011 |
```

### Using the CSV Index

The `runs/index.csv` file can be imported into any spreadsheet or analysis tool:

```python
import pandas as pd

df = pd.read_csv("runs/index.csv")
print(df.groupby("architecture")["best_r2_ef"].max())
```

### Using the Analysis Script

The `scripts/analyze/analyze_training.py` script provides head-to-head comparison with baselines:

```bash
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --baseline runs/SL-20260701-007/test_results.json
```

This generates a `FINAL_REPORT.md` with a "Improvement vs Baseline" section showing deltas and percentage changes.

### Manual Comparison

Each run directory is self-contained. To compare two runs manually:

```bash
# Compare best test results
diff -u \
    <(python3 -c "import json; d=json.load(open('runs/SL-20260701-007/test_results.json')); print(d)") \
    <(python3 -c "import json; d=json.load(open('runs/SL-20260708-001/test_results.json')); print(d)")

# Compare learning curves
python3 -c "
import json, numpy as np
a = json.load(open('runs/SL-20260701-007/epoch_metrics.json'))
b = json.load(open('runs/SL-20260708-001/epoch_metrics.json'))
print('Prev val_loss (min):', min(m['val_loss'] for m in a))
print('ExpA val_loss (min):', min(m['val_loss'] for m in b))
"
```

---

## 10. How to Resume Experiments

The tracking system is designed for crash-resilient training. Resume is handled entirely through the `ExperimentTracker` constructor — no manual checkpoint loading is needed.

### Automatic Resume from Crash

If training crashes (e.g., SLURM timeout, OOM, power loss), resume by pointing to the existing run directory:

```python
tracker = ExperimentTracker(
    config=config,
    run_dir="runs",
    resume_from="runs/SL-20260708-001",  # Path to existing run
)
```

The tracker will:

1. **Detect existing data**: Load `epoch_metrics.json` to reconstruct metric history
2. **Find last checkpoint**: Determine the last completed epoch from the metrics
3. **Restore state**: Load `checkpoints/last.pt` for model, optimizer, scaler, and RNG state
4. **Recover timer**: Restore `_training_t0` from metadata for cumulative GPU-hour accounting
5. **Recreate best-metric cache**: Rebuild `MetricsStore._best` from existing epochs

The resume process can be verified with `analyze_training.py`:

```bash
./venv/bin/python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001
```

The analysis script detects resume points by identifying timestamp gaps >3× the median epoch time and >600 seconds, and generates a `resume_audit.png` plot showing before/after consistency.

### Resume Contract

For a successful resume, the following must hold:

| Requirement | Check |
|------------|-------|
| Same config | Config fingerprint should match (logged in reports) |
| Same code | Git commit hash should match |
| Same data | Dataset must be unchanged |
| Same device | GPU type should match (or CPU fallback) |
| Last checkpoint exists | `checkpoints/last.pt` must be present |
| Metrics file intact | `epoch_metrics.json` must be valid JSON |

The `Resume Audit` section in `FINAL_REPORT.md` verifies:

```markdown
| Check | Status |
|-------|--------|
| Same experiment directory | ✓ |
| Optimizer restored from checkpoint | ✓ |
| GradScaler state restored | ✓ |
| RNG state restored | ✓ |
| Training continues from ep N+1 | ✓ |
| Best val_loss before resume: X.XXXX @ ep Y | → |
| Best val_loss after resume: X.XXXX @ ep Z | ✓ |
```

### Manual Resume Example

```python
import torch
from src.training.experiment_tracker import ExperimentTracker

# Load config (must match original)
config = yaml.safe_load(open("configs/model_config_v3_li.yaml"))

# Resume
tracker = ExperimentTracker(
    config=config,
    resume_from="runs/SL-20260708-001",
)

# Load checkpoint
ckpt = torch.load("runs/SL-20260708-001/checkpoints/last.pt")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
start_epoch = ckpt["epoch"] + 1

# Continue training
for epoch in range(start_epoch, config["training"]["max_epochs"]):
    # ... training loop ...
```

---

## 11. Analysis Pipeline

The post-hoc analysis pipeline is implemented in `scripts/analyze/analyze_training.py` and produces comprehensive analysis for any completed run.

### Usage

```bash
# Basic analysis
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --output reports/final_analysis

# With baseline comparison
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --baseline runs/SL-20260701-007/test_results.json \
    --output reports/final_analysis

# With prediction diagnostics (requires checkpoint + dataset)
python scripts/analyze/analyze_training.py \
    --run runs/SL-20260708-001 \
    --checkpoint runs/SL-20260708-001/checkpoints/best_val_loss.pt \
    --output reports/final_analysis
```

### Analysis Components

**1. Metrics Loading (`load_metrics`)**
- Reads `epoch_metrics.json`, `run_metadata.json`, `config.yaml`
- Returns structured data for downstream analysis

**2. Resume Detection (`find_resume_epoch`)**
- Analyzes epoch timestamp gaps to identify resume points
- A gap >3× median epoch time and >600 seconds indicates a resume

**3. Plot Generation (8+ figures)**

| Plot | Description |
|------|-------------|
| `learning_curves.png` | Train/val loss with moving average, best epoch marker, patience window |
| `per_task_mae.png` | Per-task MAE evolution with best markers |
| `per_task_rmse.png` | Per-task RMSE evolution |
| `per_task_r2.png` | Per-task R² evolution |
| `gradnorm_weights.png` | GradNorm weight trajectories with moving average |
| `gradnorm_correlation.png` | Weight vs. performance scatter (2×3 grid) |
| `system_metrics.png` | Epoch time, throughput, GPU memory, grad norm (2×2 grid) |
| `resume_audit.png` | Consistency table for resumed runs |
| `training_timeline.png` | Event timeline visualization |
| `pred_vs_actual.png` | Scatter plots for each task (requires checkpoint) |
| `residual_histograms.png` | Residual distributions (requires checkpoint) |

**4. Report Generation**

`FINAL_REPORT.md` contains 9 sections:

1. **Training Summary**: Total epochs, GPU-hours, best/latest metrics
2. **Best vs Final Comparison**: Delta analysis for all metrics
3. **Best Checkpoint**: Detailed best-epoch metrics
4. **Improvement vs Baseline**: Delta and percentage change from baseline
5. **GradNorm Analysis**: Initial/final/mean/std/trend for each task
6. **System Performance**: Mean/min/max epoch time, throughput, memory
7. **Training Timeline**: ASCII art event chain
8. **Resume Audit** (if applicable): Consistency verification
9. **Configuration**: Environment, full config fingerprint and YAML

**5. Scorecard Generation**

`SCORECARD.md` provides a quality checklist:

- Training completed
- Resume verified (if applicable)
- No NaN losses
- No exploding gradients
- Best checkpoint saved
- Metrics improved from start
- Baseline comparison complete (if applicable)

The scorecard includes a visual progress bar:

```
**5/6 checks passed**

`████████████████░░░░`  **83%**
```

### Prediction Diagnostics

When a checkpoint is provided, the analysis script runs full inference on the test set:

1. **Model reconstruction**: Builds model with matching architecture from checkpoint config
2. **Data loading**: Loads test set with `batch_size=16`, `num_workers=0`
3. **Inference**: Runs model in eval mode, collects predictions
4. **Diagnostic plots**: Predicted vs actual scatter, residual histograms with normality tests

The prediction pipeline requires:
- Original dataset (for test split and graph construction)
- Checkpoint file (`.pt` with model state dict)
- ~600 MB GPU memory (single batch inference)

---

## 12. Best Practices

### Before Training

- **Commit your code**: The tracker records git commit hash. An uncommitted run is unreproducible.
- **Pin dependencies**: Use `requirements.txt` or `environment.yml`. Record Python and PyTorch versions.
- **Set a seed**: The tracker records `torch.manual_seed` and `numpy.random.seed` in checkpoints.
- **Use a unique config name**: Config fingerprints enable quick cross-reference.

### During Training

- **Monitor summaries**: `TRAINING_SUMMARY.md` updates every epoch with best-so-far metrics.
- **Check early stopping**: The patience mechanism prevents wasted GPU hours.
- **Inspect plots**: Watch for NaN losses, exploding gradients, or plateauing metrics.
- **Back up runs**: The `runs/` directory is the canonical experiment record. Include it in backups.

### After Training

- **Run analysis**: Always run `analyze_training.py` for the full report and scorecard.
- **Tag significant runs**: Use the run ID in experiment notebooks and papers.
- **Archive checkpoints**: Best checkpoints are tiny (~5 MB). Archive them for reproducibility.
- **Clean up failures**: Remove or merge duplicate `index.csv` entries from crash-recovered runs.

### Naming Conventions

- Run IDs are auto-generated. Do not rename directories manually.
- Config files should be versioned (e.g., `model_config_v3_li.yaml`).
- Analysis outputs go in `analysis/` subdirectory of the run.
- Reports use UPPERCASE filenames (`FINAL_REPORT.md`, `SCORECARD.md`, `MODEL_CARD.md`).

---

## 13. Troubleshooting

### Duplicate index entries

The CSV index can accumulate duplicate entries from crash-recovery cycles. Deduplicate with:

```bash
python3 -c "
import csv
rows = {}
with open('runs/index.csv', 'r') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        rows[row['run_id']] = row  # Last entry wins
with open('runs/index.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows.values())
"
```

### Missing best checkpoints

If `best_val_loss.pt` is missing but training completed, the checkpoint might not have been saved due to a crash during `finalize()`. Check `checkpoints/last.pt` and manually determine the best epoch from `epoch_metrics.json`.

### Corrupted epoch_metrics.json

The JSON file is written atomically (Python `json.dump` creates the file in one write), but a crash during write can corrupt it. Restore from the CSV backup:

```python
import pandas as pd
df = pd.read_csv("runs/SL-20260708-001/epoch_metrics.csv")
df.to_json("runs/SL-20260708-001/epoch_metrics.json", orient="records")
```

### Resume fails with shape mismatch

If the model architecture changed between the original run and the resume attempt, checkpoint loading will fail with a shape mismatch. This is intentional: resume requires identical architecture. Use `strict=False` in `load_state_dict()` for partial loading:

```python
model.load_state_dict(ckpt["model"], strict=False)
```

### GPU-hour mismatch

The GPU-hour counter is based on `time.perf_counter()` and does not account for GPU idle time (e.g., DataLoader bottlenecks). For accurate GPU-hour accounting, use `torch.cuda.synchronize()` boundaries around the training loop and measure CUDA time directly with CUDA events.

---

*Generated by the Scandium Labs Experiment Tracking System. For questions, refer to the source at `src/training/experiment_tracker.py`.*
