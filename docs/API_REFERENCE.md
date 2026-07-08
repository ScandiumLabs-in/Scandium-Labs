# API Reference

> Comprehensive reference for all public interfaces in the Scandium Labs platform.
>
> **Last updated:** July 2026
> **Codebase:** `scandium-labs` (v1.0.0)

---

## Table of Contents

1. [Training API](#1-training-api)
   - [ScandiumTrainer](#scandiumtrainer)
   - [CLI: train_v3_li.py](#cli-train_v3_lipy)
   - [experiment_sweep.py](#experiment_sweeppy)
2. [Inference API](#2-inference-api)
   - [InferenceEngine](#inferenceengine)
   - [ParetoRanker](#parétoranker)
   - [OODDetector](#ooddetector)
   - [Stability Utilities](#stability-utilities)
3. [REST API](#3-rest-api)
   - [Endpoints](#endpoints)
   - [Authentication](#authentication)
4. [Data API](#4-data-api)
   - [Datasets](#datasets)
   - [Data Cleaning & Normalization](#data-cleaning--normalization)
   - [Collectors](#collectors)
5. [Configuration API](#5-configuration-api)
   - [Config Loading / Merging](#config-loading--merging)
   - [Configuration Reference Table](#configuration-reference-table)

---

## 1. Training API

### ScandiumTrainer

**File:** `src/training/trainer.py`  
The primary orchestrator for model training: builds model, optimizer, loss, runs train/validation loops with mixed-precision, GradScaler, checkpointing, and WandB logging.

---

#### `ScandiumTrainer.__init__(config_path, data_dir)`

| Item | Description |
|------|-------------|
| **Signature** | `__init__(self, config_path: str, data_dir: str = "data/processed")` |
| **Args** | `config_path` — Path to YAML config file; `data_dir` — Directory containing cached dataset and split files. |
| **Side Effects** | Loads YAML config; detects CUDA device; initializes `GradScaler`; loads or creates `PropertyNormalizer` from `data_dir/normalizer.json`. |
| **Raises** | `FileNotFoundError` if config path does not exist. |

**Example:**

```python
from src.training.trainer import ScandiumTrainer

trainer = ScandiumTrainer(
    config_path="configs/model_config_v3_li.yaml",
    data_dir="datasets/v3_li_10000",
)
```

---

#### `build_model()`

| Item | Description |
|------|-------------|
| **Signature** | `build_model(self) -> ScandiumPINNGNN` |
| **Returns** | A `ScandiumPINNGNN` instance moved to `self.device`. Hidden dim, layer counts, dropout, MC samples, and task list are read from `self.config["model"]`. |
| **Side Effects** | If `self.config["model"].get("use_pretrained_alignn")` is truthy, loads a pretrained ALIGNN encoder via `PretrainedEncoder`. |

**Example:**

```python
model = trainer.build_model()
print(model)  # ScandiumPINNGNN(
               #   (atom_encoder): Linear(92 → 128)
               #   (edge_encoder): Sequential(...)
               #   (alignn_layers): ModuleList(4× ALIGNNLayer)
               #   (transformer_layers): ModuleList(2× GraphTransformerLayer)
               #   ...
               # )
```

---

#### `build_optimizer(model)`

| Item | Description |
|------|-------------|
| **Signature** | `build_optimizer(self, model: ScandiumPINNGNN) -> torch.optim.AdamW` |
| **Args** | `model` — The model whose parameters will be optimized. |
| **Returns** | `AdamW` optimizer; parameter groups are created by `get_param_groups()` which applies differential LR / weight decay per module. |
| **Config Used** | `training.weight_decay`, `training.learning_rate` (via `get_param_groups`). |

**Example:**

```python
optimizer = trainer.build_optimizer(model)
# optimizer.param_groups[0]['lr'] = config['training']['learning_rate']
```

---

#### `build_loss()`

| Item | Description |
|------|-------------|
| **Signature** | `build_loss(self) -> PINNLoss` |
| **Returns** | A `PINNLoss` instance with task weights, lambda coefficients, and `log_eah` flag sourced from config. |
| **Config Used** | `tasks[].weight`, `pinn.*`, `log_eah`. |

**Example:**

```python
loss_fn = trainer.build_loss()
# PINNLoss(task_weights={"formation_energy": 1.0, "energy_above_hull": 1.0, "band_gap": 1.0})
```

---

#### `train_epoch(model, loader, optimizer, scheduler, loss_fn)`

| Item | Description |
|------|-------------|
| **Signature** | `train_epoch(self, model, loader, optimizer, scheduler, loss_fn) -> dict` |
| **Args** | `model` — PyTorch module; `loader` — DataLoader yielding `(crystal_graph, line_graph)` tuples; `optimizer` — AdamW; `scheduler` — LR scheduler; `loss_fn` — `PINNLoss`. |
| **Returns** | `dict` with keys: `"data"`, `"arrhenius"`, `"thermodynamic"`, `"total"` (mean losses), `"task_data"` (per-task MSE), `"grad_norms"` (per-task gradient norms). |
| **Side Effects** | Mixed-precision forward/backward via `GradScaler`; gradient clipping at `config["training"]["gradient_clip"]`. |

---

#### `validate(model, loader, loss_fn)`

| Item | Description |
|------|-------------|
| **Signature** | `validate(self, model, loader, loss_fn) -> dict` |
| **Args** | `model` — PyTorch module in eval mode; `loader` — DataLoader; `loss_fn` — unused in validation (only MAE computed). |
| **Returns** | `dict` of `{task}_mae` values. Predictions are denormalized before computing MAE if the normalizer has stats for the task. |
| **Decorator** | `@torch.no_grad()` |

---

#### `train(resume_from=None)`

| Item | Description |
|------|-------------|
| **Signature** | `train(self, resume_from: str | None = None) -> tuple[ScandiumPINNGNN, dict]` |
| **Args** | `resume_from` — Optional path to a `.pt` checkpoint to resume from. |
| **Returns** | `(model, test_metrics)` where `test_metrics` is the dict of per-task MAE on the held-out test set. |
| **Side Effects** | Builds model/optimizer/loss; loads data splits via `load_data()`; saves per-epoch checkpoints; logs to WandB if enabled; best model saved as `checkpoints/best_model.pt`. |
| **Early Stopping** | Triggers when `patience_counter >= config["training"]["patience"]`. |

**Example:**

```python
# Full training run
model, metrics = trainer.train()
print(metrics)
# {'formation_energy_mae': 0.042, 'energy_above_hull_mae': 0.089, 'band_gap_mae': 0.215}

# Resume from checkpoint
trainer.train(resume_from="runs/SL-20260701-007/checkpoints/last.pt")
```

---

### CLI: train_v3_li.py

**File:** `scripts/train/train_v3_li.py`  
Standalone end-to-end training script. Does **not** use `ScandiumTrainer`; has its own manual training loop with GradNorm, TwoStageEaH, ExperimentTracker, size-bucketed batching, cosine scheduler, and AMP.

```
usage: train_v3_li.py [-h] [--config CONFIG] [--resume RESUME]
                      [--data-dir DATA_DIR] [--out-dir OUT_DIR]
                      [--no-gradnorm]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | `str` | `configs/model_config_v3_li.yaml` | Path to training config YAML |
| `--resume` | `str` | `None` | Path to a checkpoint `.pt` to resume from |
| `--data-dir` | `str` | `datasets/v3_li_10000` | Dataset directory containing `dataset_cache.pt`, `split_indices.pt`, and `graphs/` |
| `--out-dir` | `str` | `checkpoints/v3_li_10k_fresh` | Output directory for checkpoints |
| `--no-gradnorm` | flag | `False` | Disable GradNorm adaptive weighting, use fixed equal weights |

**Examples:**

```bash
# Train from scratch
python scripts/train/train_v3_li.py --config configs/model_config_v3_li.yaml --out-dir checkpoints/my_exp

# Resume interrupted run
python scripts/train/train_v3_li.py --resume runs/SL-20260701-007/checkpoints/last.pt

# Disable GradNorm
python scripts/train/train_v3_li.py --no-gradnorm --out-dir checkpoints/no_gn_exp
```

**Output structure:**

```
checkpoints/my_exp/
├── best_model.pt      # Best model (lowest val loss)
├── epoch_10.pt        # Periodic checkpoints (every 10 epochs)
└── test_results.json  # Final test metrics

runs/SL-YYYYMMDD-NNN/
├── config.yaml
├── epoch_metrics.json
├── epoch_metrics.csv
├── run_metadata.json
├── checkpoints/
│   ├── last.pt
│   ├── best_val_loss.pt
│   ├── best_*_*.pt
│   └── epoch_*.pt
├── plots/
│   ├── loss_curve.png
│   ├── mae_curve.png
│   ├── r2_curve.png
│   ├── gradnorm_weights.png
│   └── ...
├── TRAINING_SUMMARY.md
├── BEST_MODEL_REPORT.md
├── MODEL_CARD.md
├── EXPERIMENT_LEADERBOARD.md
└── STOP_REPORT.md
```

---

### experiment_sweep.py

**File:** `scripts/train/experiment_sweep.py`  
Structured experiment runner that creates versioned directories with full reproducibility artifacts.

```
usage: experiment_sweep.py --config CONFIG --data_dir DATA_DIR
                           [--name NAME] [--gpus GPUS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | `str` | **required** | Model config YAML |
| `--data_dir` | `str` | **required** | Dataset directory |
| `--name` | `str` | auto-generated | Experiment name (e.g., `v2_3635_first_run`) |
| `--gpus` | `int` | 1 | Number of GPUs |

**Example:**

```bash
python scripts/train/experiment_sweep.py \
    --config configs/model_config_v2.yaml \
    --data_dir datasets/v2_10000 \
    --name v2_3635_first_run
```

**Output (`experiments/<name>/`):**

```
config.yaml          — Exact config copy
metrics.json         — Test metrics
train.log            — Full training log
checkpoint.pt        — Best model
parity_plot.png      — Predicted vs actual
benchmark.csv        — Benchmark suite results
git_commit.txt       — Git commit hash
dataset_version.txt  — Dataset version
start_time.txt       — ISO timestamp
end_time.txt         — ISO timestamp
```

---

## 2. Inference API

### InferenceEngine

**File:** `src/inference/engine.py`  
The main entry point for running trained models on new crystal structures.

---

#### `InferenceEngine.__init__(model_path, device, use_mc_dropout, mc_samples, log_eah)`

| Item | Description |
|------|-------------|
| **Signature** | `__init__(self, model_path: str, device: str = "cuda", use_mc_dropout: bool = True, mc_samples: int = 20, log_eah: bool = False)` |
| **Args** | `model_path` — Path to trained `.pt` checkpoint; `device` — `"cuda"` or `"cpu"`; `use_mc_dropout` — Enable MC Dropout uncertainty estimation; `mc_samples` — Number of forward passes; `log_eah` — If `True`, energy-above-hull was trained in log space. |
| **Side Effects** | Loads model checkpoint; builds `ALIGNNGraphBuilder` and `FeatureEngineer`; loads `PropertyNormalizer` from checkpoint parent or `data/`; generates coverage report. |

**Example:**

```python
from src.inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="checkpoints/best_model.pt",
    device="cpu",
    use_mc_dropout=True,
    mc_samples=50,
)
```

---

#### `predict_single(structure, temperature)`

| Item | Description |
|------|-------------|
| **Signature** | `predict_single(self, structure: pymatgen.core.Structure, temperature: float = 300.0) -> dict` |
| **Args** | `structure` — A `pymatgen` Structure object; `temperature` — Temperature in Kelvin for Arrhenius calculation. |
| **Returns** | Nested dict with predictions, uncertainties, stability check, recommendation, and optional OOD and conductivity-derived activation energy. |
| **Decorator** | `@torch.no_grad()` |

**Returns structure:**

```json
{
  "formation_energy": {
    "value": -0.423,
    "uncertainty": 0.031
  },
  "energy_above_hull": {
    "value": 0.012,
    "uncertainty": 0.008
  },
  "band_gap": {
    "value": 3.21,
    "uncertainty": 0.15
  },
  "log_ionic_conductivity": {
    "value": -3.12,
    "uncertainty": 0.42,
    "status": "insufficient training data"
  },
  "ionic_conductivity": {
    "value": 0.00076,
    "uncertainty": 7.4e-05,
    "unit": "S/cm"
  },
  "stability_check": {
    "formation_energy": -0.423,
    "energy_above_hull": 0.012,
    "suspicious": false,
    "reason": null,
    "mp_hull_data": null
  },
  "recommendation": "HIGH PRIORITY",
  "recommendation_detail": "Excellent candidate — σ=7.58e-04 S/cm, stable Eah=0.012 eV/atom",
  "recommendation_confidence": "high",
  "recommended_actions": [
    "Proceed to experimental validation",
    "Prepare sample via known synthesis route",
    "Measure ionic conductivity via EIS"
  ]
}
```

---

#### `batch_predict(structures, batch_size)`

| Item | Description |
|------|-------------|
| **Signature** | `batch_predict(self, structures: list, batch_size: int = 32) -> list[dict]` |
| **Args** | `structures` — List of `pymatgen` Structure objects; `batch_size` — Number of structures per batch. |
| **Returns** | List of prediction dicts (same format as `predict_single`). |

**Note:** Currently iterates `predict_single` in a loop. True batching across the model forward pass is planned.

---

#### `screen_candidates(candidates, top_k, temperature)`

*Alias for batch prediction + ranking.* The `screen_candidates` entry point in `scripts/inference/screen_candidates.py` handles JSON-based batch screening:

```
usage: screen_candidates.py --input INPUT [--config CONFIG]
                            [--output OUTPUT] [--model MODEL]
                            [--top_k TOP_K] [--temperature TEMPERATURE]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | `str` | **required** | Path to candidate list JSON |
| `--config` | `str` | `configs/model_config.yaml` | Config path |
| `--output` | `str` | `screening_results.json` | Output JSON path |
| `--model` | `str` | `checkpoints/best_model.pt` | Model checkpoint |
| `--top_k` | `int` | 10 | Number of top candidates to output |
| `--temperature` | `float` | 300.0 | Temperature for Arrhenius calculation |

**Example:**

```bash
python scripts/inference/screen_candidates.py \
    --input candidates.json \
    --model checkpoints/my_exp/best_model.pt \
    --top_k 5 \
    --output results.json
```

---

### ParetoRanker

**File:** `src/inference/ranking.py`  
Multi-objective ranking of candidates using Pareto optimality + weighted composite score.

---

#### `rank(candidates)`

| Item | Description |
|------|-------------|
| **Signature** | `rank(self, candidates: list[dict]) -> list[dict]` |
| **Args** | `candidates` — List of prediction dicts (must contain `ionic_conductivity`, `energy_above_hull`, `ood` keys). |
| **Returns** | Ranked list of candidates (mutated in place) with added keys: `rank`, `pareto_rank`, `composite_score`. |
| **Objective Matrix** | `[log10(σ), -EaH, ood_score]` weighted `[0.5, 0.3, 0.2]`. |
| **Algorithm** | Non-dominated sorting (Pareto front layering), then normalized weighted sum for tie-breaking. |

**Example:**

```python
from src.inference.ranking import ParetoRanker

ranker = ParetoRanker()
ranked = ranker.rank(candidates)
for c in ranked[:5]:
    print(f"#{c['rank']}  {c['formula']}  score={c['composite_score']:.3f}")
```

---

### OODDetector

**File:** `src/inference/validation.py` (OOD detection is integrated into `InferenceEngine`; the OODDetector class is used internally)

---

#### `score(embedding)`

| Item | Description |
|------|-------------|
| **Signature** | `score(self, embedding: np.ndarray) -> dict` |
| **Args** | `embedding` — Latent representation (pooled graph embedding) from the model. |
| **Returns** | `{"ood_score": float, "is_ood": bool, "threshold": float}` |
| **Method** | Distance to training set embedding distribution (Mahalanobis or L2-based). |

#### `predict(structure)`

| Item | Description |
|------|-------------|
| **Signature** | `predict(self, structure: Structure) -> dict` |
| **Returns** | Same as `score()` but accepts a Structure directly (builds graph and extracts embedding internally). |

---

### Stability Utilities

**File:** `src/inference/stability.py`  
Cross-checks model predictions against the Materials Project convex hull and resolves inconsistencies.

---

#### `compute_hull_energy(composition, predicted_formation_energy)`

| Item | Description |
|------|-------------|
| **Signature** | `compute_hull_energy(composition: Composition, predicted_formation_energy: float) -> dict` |
| **Args** | `composition` — `pymatgen` Composition; `predicted_formation_energy` — Model-predicted Ef (eV/atom). |
| **Returns** | `{"energy_above_hull": float | None, "source": str, "num_competing_phases": int, "available": bool}` |
| **Note** | Requires `MP_API_KEY` environment variable. Falls back to `{"available": false}` if no key. |

#### `resolve_stability(predictions, composition)`

| Item | Description |
|------|-------------|
| **Signature** | `resolve_stability(predictions: dict, composition: Composition | None = None) -> dict` |
| **Args** | `predictions` — Dict from `InferenceEngine.predict_single`; `composition` — Optional `pymatgen` Composition. |
| **Returns** | `{"formation_energy": float, "energy_above_hull": float, "suspicious": bool, "reason": str | None, "mp_hull_data": dict | None}` |
| **Logic** | If `|Ef| < 0.1` and `EaH > 0.25`, flags as suspicious and optionally queries the MP convex hull API. |

---

## 3. REST API

**File:** `api/main.py`  
FastAPI application served via Uvicorn, protected by JWT Bearer authentication.

### Endpoints

---

#### `GET /health`

| Item | Description |
|------|-------------|
| **Signature** | `GET /health` |
| **Auth** | None |
| **Response** | `{"status": "healthy", "model_loaded": true\|false}` |

**Example:**

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}
```

---

#### `POST /screen`

| Item | Description |
|------|-------------|
| **Signature** | `POST /screen` |
| **Auth** | Bearer JWT token |
| **Body** | `ScreeningRequest` JSON |
| **Response** | `ScreeningResult` — `{"job_id": "...", "status": "queued", "created_at": "..."}` |
| **Note** | Dispatches to Celery async worker. |

**Request body:**

```json
{
  "material_ids": ["mp-123", "mp-456"],
  "formulas": ["Li6PS5Cl", "Li10GeP2S12"],
  "temperature": 300.0,
  "tasks": ["log_ionic_conductivity", "formation_energy", "energy_above_hull"],
  "top_k": 10
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/screen \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{"formulas": ["Li6PS5Cl"], "top_k": 5}'
```

---

#### `POST /screen/upload`

| Item | Description |
|------|-------------|
| **Signature** | `POST /screen/upload` |
| **Auth** | Bearer JWT token |
| **Params** | `file` (UploadFile) — CIF or POSCAR file; `temperature` (float, default 300.0) |
| **Response** | Full prediction dict (same as `InferenceEngine.predict_single`) plus `material`, `formula`, `spacegroup`. |
| **Status Codes** | `400` — Invalid file format or structure parsing failure; `200` — Successful prediction. |

**Example:**

```bash
curl -X POST http://localhost:8000/screen/upload \
  -H "Authorization: Bearer $JWT" \
  -F "file=@Li6PS5Cl.cif" \
  -F "temperature=300.0"
```

---

#### `GET /job/{job_id}`

| Item | Description |
|------|-------------|
| **Signature** | `GET /job/{job_id}` |
| **Auth** | Bearer JWT token |
| **Response** | Job status + results if completed, progress if in-flight. |

**Response (in progress):**

```json
{
  "job_id": "abc-123",
  "status": "processing",
  "progress": 45.0,
  "n_materials": 100,
  "completed_materials": 45
}
```

**Response (completed):**

```json
{
  "job_id": "abc-123",
  "status": "completed",
  "results": [ ... ],
  "top_k": [ ... ]
}
```

---

### Authentication

**File:** `api/auth.py`

| Function | Description |
|----------|-------------|
| `create_access_token(user_id, expires_delta)` | Creates a JWT with `sub=user_id`, `exp`, `iat`. Default 7-day expiry. |
| `verify_token(credentials)` | FastAPI dependency; extracts `sub` from Bearer token. Raises 401 on expiry or invalid token. |

**Config:** `JWT_SECRET_KEY` env var (default `"dev-secret-key-not-for-production"`).

**Example token generation:**

```python
from api.auth import create_access_token
token = create_access_token("user_alice")
# "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## 4. Data API

### Datasets

**File:** `src/data/dataset.py`

---

#### `LazyGraphDataset`

Memory-efficient dataset that loads pre-cached graphs from disk on demand.

| Item | Description |
|------|-------------|
| **Signature** | `LazyGraphDataset(structure_list=None, targets=None, graph_dir=None, graph_builder=None, feature_engineer=None, cache_dir=None, memory_cache=True, prebuilt_list=None)` |
| **Args** | `structure_list` — Raw structures for on-the-fly building; `targets` — Dict of task → array; `graph_dir` — Directory of `{idx}.pt` files; `graph_builder` — ALIGNNGraphBuilder; `feature_engineer` — FeatureEngineer; `cache_dir` — Where to cache built graphs; `memory_cache` — Cache graphs in RAM after first load; `prebuilt_list` — Pre-loaded monolithic graph list. |
| **Returns** | `(crystal_graph, line_graph)` tuple on `__getitem__`. |
| **Lookup Order** | (1) Memory cache → (2) Graph dir individual files → (3) Prebuilt list → (4) On-the-fly building. |

**Example:**

```python
from src.data.dataset import LazyGraphDataset, collate_fn
from src.graphs.builder import ALIGNNGraphBuilder, FeatureEngineer

ds = LazyGraphDataset(
    structure_list=structures,
    targets=targets,
    graph_dir="datasets/v3_li_10000/graphs",
    graph_builder=ALIGNNGraphBuilder(cutoff=8.0, max_neighbors=16, num_sbf=32),
    feature_engineer=FeatureEngineer(),
)

loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn, num_workers=4)
```

---

#### `SolidElectrolyteDataset`

In-memory dataset that builds graphs on-the-fly during iteration (legacy; prefer `LazyGraphDataset`).

| Item | Description |
|------|-------------|
| **Signature** | `SolidElectrolyteDataset(structures, targets, graph_builder, feature_engineer, transform=None)` |
| **Args** | `structures` — List of pymatgen Structure objects; `targets` — Dict of task arrays; `graph_builder` — ALIGNNGraphBuilder; `feature_engineer` — FeatureEngineer; `transform` — Optional callable. |

---

#### `collate_fn(batch)`

| Item | Description |
|------|-------------|
| **Signature** | `collate_fn(batch: list[tuple]) -> tuple[Batch, Batch]` |
| **Args** | `batch` — List of `(crystal_graph, line_graph)` tuples. |
| **Returns** | `(Batch.from_data_list(crystal_graphs), Batch.from_data_list(line_graphs))` |

---

### Data Cleaning & Normalization

**File:** `src/data/cleaner.py`

---

#### `DataCleaner`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(ltol=0.2, stol=0.3, angle_tol=5)` | Initializes pymatgen `StructureMatcher` for deduplication. |
| `clean` | `(raw_data: list \| pd.DataFrame) -> pd.DataFrame` | Drops nulls; filters `formation_energy_per_atom ∈ [-10, 5]`, `energy_above_hull >= 0`, structure size `[2, 200]`; deduplicates via structure matching; normalizes units. |

#### `PropertyNormalizer`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(stats=None)` | Initialize with optional pre-computed stats dict. |
| `fit` | `(df, columns)` | Computes mean/std/min/max for each column. |
| `transform` | `(df) -> DataFrame` | Z-score normalization. |
| `inverse_transform` | `(values, col) -> ndarray` | Denormalize values. |
| `normalize` | `(raw_targets: dict) -> dict` | Normalize per-task tensor targets. |
| `denormalize` | `(predictions: dict) -> dict` | Denormalize prediction dicts. |
| `save` | `(path: str)` | Save stats to JSON. |
| `load` | `(path: str) -> PropertyNormalizer` | Load from JSON. |

**Example:**

```python
normalizer = PropertyNormalizer()
normalizer.fit(df, ["formation_energy", "energy_above_hull", "band_gap"])
normalizer.save("data/normalizer.json")

normalized = normalizer.transform(df)
restored = normalizer.inverse_transform(normalized["formation_energy"], "formation_energy")
```

---

### Collectors

**File:** `src/data/collectors.py`

---

#### `MaterialsProjectCollector`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(api_key=None)` | Accepts MP API key; falls back to `MP_API_KEY` or `MATERIALS_PROJECT_API_KEY` env vars. |
| `collect` | `(elements=None, fields=None, max_results=50000, num_chunks=None) -> pd.DataFrame` | Queries Materials Project via `mp_api.client.MPRester`. Default elements: `["Li"]`. Default fields include `material_id`, `formula_pretty`, `structure`, `formation_energy_per_atom`, `energy_above_hull`, `band_gap`, `volume`, `density`, `symmetry`, `is_stable`. |

**Example:**

```python
collector = MaterialsProjectCollector()
df = collector.collect(elements=["Li", "S"], max_results=10000)
# Columns: material_id, formula_pretty, structure, formation_energy_per_atom, ...
```

---

#### Other Collectors

| Collector | Method | Source |
|-----------|--------|--------|
| `JARVISCollector` | `collect(dataset_name="dft_3d")` | JARVIS-DFT figshare |
| `OQMDCollector` | `collect(limit=100000, offset=0)` | OQMD REST API |
| `AFLOWCollector` | `collect(elements=None, max_results=50000)` | AFLOW XML-RPC API |
| `NOMADCollector` | `collect(elements=None, page_size=100, max_entries=10000)` | NOMAD Archive API |

---

### Splitter

**File:** `src/data/splitter.py`

#### `composition_based_split(dataset, val_ratio, test_ratio)`

| Item | Description |
|------|-------------|
| **Signature** | `composition_based_split(dataset, val_ratio=0.1, test_ratio=0.1) -> tuple[list[int], list[int], list[int]]` |
| **Args** | `dataset` — Must have `.structures` attribute; `val_ratio` — Validation fraction; `test_ratio` — Test fraction. |
| **Returns** | `(train_indices, val_indices, test_indices)` |
| **Method** | Groups structures by element combination (e.g., `"Li-La-Zr-O"`), then uses `GroupShuffleSplit` to prevent element leakage across splits. |

**Example:**

```python
from src.data.splitter import composition_based_split

train_idx, val_idx, test_idx = composition_based_split(dataset, val_ratio=0.1, test_ratio=0.1)
```

---

### Samplers

**File:** `src/data/samplers.py`

#### `SizeBucketedBatchSampler`

| Item | Description |
|------|-------------|
| **Signature** | `SizeBucketedBatchSampler(indices, sizes=None, graph_dir=None, batch_size=16, bucket_size_mult=2.0, shuffle=True, drop_last=False)` |
| **Args** | `indices` — Dataset indices; `sizes` — Precomputed graph sizes; `graph_dir` — Alternative to sizes (loads from disk); `bucket_size_mult` — Bucket capacity as multiple of batch_size. |
| **Method** | Sorts indices by graph size, splits into buckets of `bucket_size_mult × batch_size`, shuffles within buckets each epoch. Reduces padding waste by ~30% vs random batching. |

#### `precompute_graph_sizes(graph_dir, indices)`

| Item | Description |
|------|-------------|
| **Signature** | `precompute_graph_sizes(graph_dir: str, indices: list[int]) -> list[int]` |
| **Returns** | List of `num_nodes(crystal_graph) + num_nodes(line_graph)` per index for bucketing. |

---

## 5. Configuration API

### Config Loading / Merging

**File:** `src/utils/config.py`

---

#### `load_config(path)`

| Item | Description |
|------|-------------|
| **Signature** | `load_config(path: str \| Path) -> dict` |
| **Args** | `path` — Path to YAML file. |
| **Returns** | Parsed config dict via `yaml.safe_load`. |
| **Raises** | `FileNotFoundError` if path does not exist. |

#### `merge_configs(base, override)`

| Item | Description |
|------|-------------|
| **Signature** | `merge_configs(base: dict, override: dict) -> dict` |
| **Args** | `base` — Base config dict; `override` — Override config dict. |
| **Returns** | Deep-merged dict. Supports nested dict merging. |

**Example:**

```python
from src.utils.config import load_config, merge_configs

base = load_config("configs/model_config_v3_li.yaml")
override = {"training": {"learning_rate": 0.001, "max_epochs": 200}}
merged = merge_configs(base, override)
```

---

### Configuration Reference Table

The following table documents every top-level and second-level key in the Scandium Labs YAML configuration format. Defaults shown are from `configs/model_config_v3_li.yaml`.

| Key | Sub-key | Type | Default | Description |
|-----|---------|------|---------|-------------|
| **model** | | | | Model architecture |
| | `name` | `str` | `"ScandiumPINNGNN-v3-Li"` | Model identifier |
| | `hidden_dim` | `int` | `128` | Hidden dimension throughout the network |
| | `num_alignn_layers` | `int` | `4` | Number of ALIGNN message-passing layers |
| | `num_transformer_layers` | `int` | `2` | Number of Graph Transformer layers |
| | `num_attention_heads` | `int` | `4` | Multi-head attention heads per transformer layer |
| | `dropout` | `float` | `0.15` | Dropout rate (also used for MC Dropout) |
| | `mc_dropout_samples` | `int` | `20` | Monte Carlo Dropout forward passes at inference |
| | `use_two_stage_eah` | `bool` | `true` | Enable two-stage EaH head (classifier + regressor) |
| | `use_gradient_checkpointing` | `str\|bool` | `"auto"` | `true`/`false` or `"auto"` (enabled if VRAM < 6GB) |
| | `use_pretrained_alignn` | `str\|bool` | _(not set)_ | Pretrained ALIGNN checkpoint path |
| **graph** | | | | Graph construction parameters |
| | `cutoff` | `float` | `8.0` | Neighbor search cutoff (Å) |
| | `max_neighbors` | `int` | `16` | Maximum neighbors per atom |
| | `num_rbf` | `int` | `64` | Radial basis function expansion size |
| | `num_sbf` | `int` | `32` | Spherical basis function expansion size |
| **tasks** | | `list[dict]` | _(see below)_ | Task definitions |
| | `[].name` | `str` | — | Task identifier (`formation_energy`, `energy_above_hull`, `band_gap`) |
| | `[].weight` | `float` | `1.0` | Task weight in loss |
| | `[].scale` | `str` | `"linear"` | Target scale |
| | `[].two_stage` | `bool` | `false` | Use two-stage EaH head for this task |
| **gradnorm** | | | | GradNorm adaptive loss balancing |
| | `enabled` | `bool` | `true` | Enable GradNorm |
| | `alpha` | `float` | `1.5` | GradNorm restoring force strength |
| **training** | | | | Training hyperparameters |
| | `batch_size` | `int` | `16` | Per-device batch size |
| | `gradient_accumulation_steps` | `int` | `2` | Accumulate gradients over N steps (effective batch: 32) |
| | `learning_rate` | `float` | `0.0005` | AdamW initial learning rate |
| | `max_epochs` | `int` | `150` | Maximum training epochs |
| | `patience` | `int` | `40` | Early stopping patience |
| | `optimizer` | `str` | `"AdamW"` | Optimizer type |
| | `weight_decay` | `float` | `0.00001` | AdamW weight decay |
| | `gradient_clip` | `float` | `1.0` | Max gradient norm for clipping |
| | `mixed_precision` | `bool` | `true` | Enable AMP fp16 training |
| | `normalize_targets` | `bool` | `true` | Z-score normalize targets |
| | `dataset` | `str` | `"v3_li_10000"` | Dataset identifier |
| | `scheduler` | `str` | _(none)_ | LR scheduler type (`"cosine"`, `"cosine_with_restarts"`, or absent for constant) |
| | `warmup_steps` | `int` | `500` | LR warmup steps |
| **bucketing** | | | | Size-based batching |
| | `enabled` | `bool` | `true` | Enable size-bucketed batch sampler |
| | `bucket_size_mult` | `float` | `2.0` | Bucket size as multiple of batch_size |
| **logging** | | | | Logging and monitoring |
| | `tensorboard` | `bool` | `false` | Enable TensorBoard logging |
| | `wandb` | `bool` | `false` | Enable Weights & Biases logging |
| | `plot` | `bool` | `true` | Generate training plots |
| | `save_epoch_checkpoints` | `int` | `10` | Save checkpoint every N epochs |
| **pinn** | | | | Physics-informed loss weights |
| | `lambda_data` | `float` | `1.0` | Data MSE loss weight |
| | `lambda_physics` | `float` | `0.1` | Physics residual loss weight |
| | `lambda_arrhenius` | `float` | `0.05` | Arrhenius constraint weight |
| | `lambda_thermodynamic` | `float` | `0.05` | Thermodynamic constraint weight |
| | `log_eah` | `bool` | `false` | Train EaH in log space |
