# Development Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, but recommended)
- Redis (for Celery task queue)
- PostgreSQL (for API persistence)

---

## 1. Setting Up the Development Environment

### 1.1 Clone & Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
# Editable install (package + core deps)
pip install -e .

# Development extras (pytest, ruff, pre-commit)
pip install -e ".[dev]"

# Or use requirements.txt for the full pinned set
pip install -r requirements.txt
```

### 1.3 GPU Setup (Optional)

If you have a GPU, install the CUDA-enabled torch variants separately:

```bash
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cu121
```

Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 1.4 One-Command Setup

```bash
make install      # pip install -e . + requirements.txt
make install-dev  # + dev deps + pre-commit hooks
```

---

## 2. Project Layout

```
src/
├── chemistry/       # Chemical family classification utilities
├── data/            # Dataset building, cleaning, splitting
├── evaluation/      # Metrics, OOD scoring
├── graphs/          # Crystal/line graph construction
├── inference/       # Prediction engine, ranking, stability
├── models/          # GNN architecture (ALIGNN, transformers, heads)
├── training/        # Trainer, losses, curriculum, data audit
└── utils/           # Shared utilities

scripts/
├── train/           # Training entrypoints
├── preprocess/      # Dataset building
├── evaluate/        # Cross-validation, benchmarking
├── inference/       # Screening commands
├── benchmark/       # Benchmark suite
└── maintenance/     # Housekeeping scripts

configs/             # YAML/JSON model & deployment configs
tests/               # pytest suite
experiments/         # Training runs, CV results, reports
checkpoints/         # Saved model weights
datasets/            # Cached graph datasets (v1_817, v2_10000, v3_li_10000)
```

---

## 3. Running Tests

```bash
# Run the full test suite
make test
# or: python -m pytest tests/ -v --tb=short

# Run with coverage
make test-coverage
# or: python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_models.py -v

# Run a specific test by name
python -m pytest tests/ -k "test_forward_pass"
```

Tests use `pytest` and live in `tests/`. Fixtures (model instances, sample graphs) are in `tests/conftest.py`.

---

## 4. Code Quality

### 4.1 Linting

Ruff enforces PEP 8, import sorting (`I`), naming (`N`), and pyupgrade (`UP`):

```bash
make lint
# or: ruff check src/ scripts/ tests/
```

### 4.2 Formatting

```bash
make format
# or: ruff format src/ scripts/ tests/ && ruff check --fix src/ scripts/ tests/
```

### 4.3 Pre-commit

```bash
pre-commit install
# Now runs ruff on every `git commit`
```

Config is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

Always run `make lint && make test` before pushing.

---

## 5. Adding a New Model

### 5.1 Create the Model Class

Add a file in `src/models/` (e.g., `src/models/my_model.py`):

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(92, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, crystal_graph, line_graph):
        x = self.encoder(crystal_graph.x)
        out = self.head(x.mean(dim=0, keepdim=True))
        return {"my_target": out.squeeze(-1)}
```

### 5.2 Register the Model

Edit `src/models/__init__.py`:

```python
from src.models.my_model import MyModel
```

### 5.3 Add a Config

Create `configs/model_config_my.yaml`:

```yaml
model:
  name: "MyModel-v1"
  hidden_dim: 128
  dropout: 0.1

tasks:
  - name: "my_target"
    weight: 1.0
    scale: "linear"

training:
  batch_size: 16
  learning_rate: 0.001
  max_epochs: 100
  patience: 30
```

### 5.4 Wire into the Trainer

If the new model has different forward/output conventions, add a builder branch in `ScandiumTrainer.build_model()` (`src/training/trainer.py`).

### 5.5 Add Tests

```python
# tests/test_my_model.py
def test_my_model_forward(graph_fixture):
    model = MyModel(hidden_dim=128)
    crystal, line = graph_fixture
    out = model(crystal, line)
    assert "my_target" in out
    assert out["my_target"].ndim == 1
```

### 5.6 Files to Modify

| File | Action |
|---|---|
| `src/models/my_model.py` | Create |
| `src/models/__init__.py` | Add import |
| `configs/model_config_my.yaml` | Create |
| `tests/test_my_model.py` | Create |

---

## 6. Adding a New Dataset

### 6.1 Understand the Data Pipeline

1. **Collect** — `src/data/collectors.py` fetches from Materials Project, OQMD, etc.
2. **Clean** — `src/data/cleaner.py` removes NaNs, filters Ef range, deduplicates.
3. **Split** — `src/data/splitter.py` does composition-based `GroupShuffleSplit`.
4. **Build Graphs** — `src/graphs/builder.py` constructs ALIGNN crystal + line graphs.
5. **Cache** — Graphs are saved as `datasets/{name}/prebuilt_graphs.pt` + `split_indices.pt`.

### 6.2 Adding a New Source

In `src/data/collectors.py`, add a new collector class:

```python
class MyCollector:
    def collect(self, api_key=None, max_structures=5000):
        # Return list of pymatgen Structure + dict of targets
        ...
```

Then wire it into `scripts/preprocess/build_dataset.py` under the `--sources` argument.

### 6.3 Building a Dataset

```bash
python scripts/preprocess/build_dataset.py \
    --sources mp \
    --output datasets/v4_my_dataset \
    --max-structures 10000 \
    --cache-graphs
```

This produces:

```
datasets/v4_my_dataset/
├── prebuilt_graphs.pt   # All graph pairs as a single tensor list
├── split_indices.pt     # train/val/test index dict
├── normalizer.json      # Per-task mean/std for normalization
└── dataset_cache.pt     # Raw structures + targets (optional)
```

### 6.4 Register in the Makefile (Optional)

```makefile
dataset-my:
	python scripts/preprocess/build_dataset.py \
	    --sources mp --output datasets/v4_my_dataset --cache-graphs
```

### 6.5 Training on the New Dataset

```bash
python scripts/train/train.py \
    --config configs/model_config_v3_li.yaml \
    --data_dir datasets/v4_my_dataset
```

Or modify a config's data path or use a dataset-specific config.

---

## 7. Adding a New Training Task

### 7.1 Define the Task

Tasks are defined in YAML configs under the `tasks` key:

```yaml
tasks:
  - name: "my_new_target"
    weight: 1.0
    scale: "linear"
```

### 7.2 Add Target Data to the Dataset

In `scripts/preprocess/build_dataset.py` or the relevant collector, include the new target in the targets dict passed to the dataset builder. The target will be stored as `y_my_new_target` on the graph object.

### 7.3 Add a Loss Function (Optional)

If the task needs a custom loss, add it in `src/training/losses.py`:

```python
class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        return F.huber_loss(pred, target)
```

Wire it into `PINNLoss.forward()` in the same file, checking for the new task name.

### 7.4 Add a Model Head

The model auto-creates heads for every task in `self.tasks`. If the head needs special architecture:

```python
# In ScandiumPINNGNN.__init__
if task == "my_new_target":
    self.task_heads[task] = MyCustomHead(hidden_dim, dropout)
else:
    self.task_heads[task] = default_mlp(...)
```

### 7.5 Add Metrics in Validation

Add the MAE/R² computation in `ScandiumTrainer.validate()` (`src/training/trainer.py:181-194`). The trainer already iterates `model.tasks` and computes denormalized MAE — new tasks will be picked up automatically if targets exist.

### 7.6 Update Config

```yaml
tasks:
  - name: "formation_energy"
    weight: 1.0
  - name: "my_new_target"
    weight: 0.5
```

### 7.7 Add Tests

```python
def test_my_new_target():
    model = ScandiumPINNGNN(tasks=["my_new_target"])
    # Verify output contains the task key
```

---

## 8. Running Experiments

### 8.1 Basic Training

```bash
python scripts/train/train.py \
    --config configs/model_config_v3_li.yaml \
    --data_dir datasets/v3_li_10000
```

Or via Make:

```bash
make train        # Uses model_config_v3_li.yaml + datasets/v3_li_10000
make train-v2     # Uses model_config_v2.yaml + datasets/v2_10000
```

### 8.2 Multi-GPU Training

```bash
python scripts/train/train.py \
    --config configs/model_config_v3_li.yaml \
    --data_dir datasets/v3_li_10000 \
    --gpus 4
```

Uses `torch.distributed` under the hood.

### 8.3 Resuming from a Checkpoint

```bash
python scripts/train/train.py \
    --config configs/model_config_v3_li.yaml \
    --data_dir datasets/v3_li_10000 \
    --checkpoint checkpoints/epoch_42.pt
```

### 8.4 Cross-Validation

```bash
python scripts/evaluate/cross_validate.py \
    --config configs/model_config_v2.yaml \
    --data_dir datasets/v2_10000
```

### 8.5 Benchmarking

```bash
make benchmark
# or: python scripts/benchmark/run_benchmark.py \
#        --checkpoint checkpoints/best_model.pt \
#        --data_dir datasets/v3_li_10000

make benchmark-compare  # Compare across checkpoints
```

### 8.6 Screening / Inference

```bash
make screen
# or: python scripts/inference/screen_candidates.py --formula Li6PS5Cl
```

### 8.7 Autopilot (Full Pipeline)

```bash
bash scripts/autopilot.sh
```

Monitors training, then runs evaluation, benchmark, CV, and generates comparison tables automatically.

### 8.8 Experiment Tracking with W&B

Set `WANDB_API_KEY` and W&B logging activates automatically in `ScandiumTrainer.train()`.

```bash
export WANDB_API_KEY=your_key
python scripts/train/train.py ...
```

---

## 9. Running the API / Frontend

### 9.1 API Server (FastAPI)

```bash
# Development server with auto-reload
uvicorn api.main:app --reload --port 8000

# Or via script
bash start_api.sh
```

### 9.2 Streamlit Frontend

```bash
bash start_streamlit.sh
# or: streamlit run streamlit_app/streamlit_app.py
```

### 9.3 Docker (Full Stack)

```bash
docker compose up --build
```

This starts API (2 replicas), Celery workers (4 replicas, GPU-enabled), TorchServe, PostgreSQL, Redis, and Flower (Celery monitoring at `localhost:5555`).

---

## 10. Debugging Common Issues

### 10.1 "No split indices found"

```text
FileNotFoundError: No split indices found. Build the dataset first:
  python scripts/build_dataset.py ...
```

Run the dataset build step first:

```bash
python scripts/preprocess/build_dataset.py \
    --sources mp \
    --output datasets/v3_li_10000 \
    --cache-graphs
```

### 10.2 CUDA Out of Memory

- Lower `batch_size` in the config (e.g., 8 → 4)
- Enable gradient checkpointing: `use_gradient_checkpointing: true`
- Reduce `hidden_dim` (256 → 128) or `num_alignn_layers` (4 → 2)
- Use `gradient_accumulation_steps: 4` with a smaller batch

### 10.3 NaN Losses

- Check for NaN in targets: run the data audit script
- Ensure targets are properly normalized (check `normalizer.json` exists)
- Lower learning rate (0.001 → 0.0003)
- Increase `gradient_clip` (1.0 → 0.5)

```bash
python scripts/maintenance/data_audit.py --data_dir datasets/v3_li_10000
```

### 10.4 Eah Collapse (All Predictions ≈ 0)

Enable two-stage Eah head in config:

```yaml
model:
    use_two_stage_eah: true
```

And verify the task definition has `two_stage: true`.

### 10.5 "No module named 'src'"

Always run scripts from the project root:

```bash
cd /home/shamique/Scandium\ Labs\ SSB/scandium-labs
python scripts/train/train.py ...
```

The `train.py` entrypoint adds the project root to `sys.path`. If running from elsewhere, set `PYTHONPATH`:

```bash
PYTHONPATH=/home/shamique/Scandium\ Labs\ SSB/scandium-labs python ...
```

### 10.6 Mixed Precision Warnings

If the GPU does not support AMP (e.g., older GPUs), set `mixed_precision: false` in the config.

### 10.7 Checkpoint Shape Mismatch on Resume

If the model architecture changed between runs, checkpoints from old architectures won't load. Either:

- Use `model.load_state_dict(ckpt["model"], strict=False)` (but inspect mismatches)
- Train from scratch after architecture changes
- Archive old checkpoints before refactoring

---

## 11. Profiling & Optimization Tips

### 11.1 Profile Training Speed

```bash
python -m cProfile -o profile.out scripts/train/train.py --config configs/model_config_v3_li.yaml
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumtime').print_stats(20)"
```

### 11.2 Memory Profiling

Monitor VRAM during training:

```bash
watch -n 1 nvidia-smi
```

Or programmatically:

```python
import torch, gc
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")
gc.collect()
torch.cuda.empty_cache()
```

### 11.3 Performance Bottlenecks

| Bottleneck | Mitigation |
|---|---|
| Graph building (CPU-bound) | Prebuild graphs (`--cache-graphs`); use `prebuilt_graphs.pt` |
| Data loading | Increase `num_workers` in DataLoader (default: 4) |
| GPU underutilization | Increase `batch_size`; use gradient accumulation |
| Transformer overhead | Reduce `num_transformer_layers` (4 → 1) |
| ALIGNN message passing | Reduce `num_alignn_layers` (4 → 2) |
| I/O on slow disks | Shard graphs into individual `.pt` files |

### 11.4 Graph Caching Strategy

- **Monolithic** (`prebuilt_graphs.pt`): Fastest for small datasets (<5k), loads entire dataset into RAM.
- **Sharded** (individual `{idx}.pt` files): Lower memory footprint, good for >10k structures.
- **On-the-fly**: No caching, builds graphs per epoch. Slowest, only for debugging.

To shard an existing prebuilt file:

```bash
python scripts/maintenance/shard_graphs.py --input datasets/v3_li_10000/prebuilt_graphs.pt
```

### 11.5 Mixed Precision

Enable in config:

```yaml
training:
    mixed_precision: true
```

Uses `torch.cuda.amp.autocast` + `GradScaler`. Reduces VRAM ~40% with minimal accuracy loss.

### 11.6 Gradient Checkpointing

Enable in config:

```yaml
model:
    use_gradient_checkpointing: true
```

Trades compute for memory — recomputes activations during backward pass. Essential for 4 GB GPUs at 256-dim.

---

## 12. CI/CD Process

### 12.1 Local Pre-Push Checklist

```bash
make lint       # Ruff check + format check
make test       # pytest suite
make test-coverage  # Ensure coverage hasn't dropped
```

### 12.2 CI Pipeline (Planned / In-Progress)

The CI (GitHub Actions) should run on every PR:

1. **Lint** — `ruff check src/ scripts/ tests/`
2. **Type Check** — (optional) `mypy src/`
3. **Test** — `pytest tests/ -v --cov=src --cov-report=xml`
4. **Build** — `pip install -e .` (verify package installs)
5. **Archive** — If tests pass on `main`, trigger model build + Docker images

### 12.3 Docker Builds

Three Dockerfiles in `docker/`:

| Dockerfile | Purpose |
|---|---|
| `Dockerfile.api` | FastAPI inference server |
| `Dockerfile.training` | Training job (HPC/cloud) |
| `Dockerfile.worker` | Celery task worker |

Build & tag example:

```bash
docker build -f docker/Dockerfile.api -t scandium-api:latest .
docker build -f docker/Dockerfile.training -t scandium-train:latest .
```

### 12.4 Release Process

1. Bump version in `pyproject.toml` and `setup.py`
2. Update `CHANGELOG.md`
3. Tag the release: `git tag v0.4.0 && git push origin v0.4.0`
4. CI builds Docker images and pushes to registry

### 12.5 Model Versioning

Checkpoints follow `checkpoints/epoch_{N}.pt` with a symlink/copy at `checkpoints/best_model.pt`. For production:

- Archive checkpoints to `model_store/` with version metadata
- Use TorchServe (`.mar` file) via `docker-compose inference` service
- Tag checkpoints with dataset version + config hash

```bash
python scripts/maintenance/archive_model.py \
    --checkpoint checkpoints/best_model.pt \
    --name scandium_v3_li_10k \
    --version 1.0
```

---

## 13. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://user:pass@localhost:5432/scandium` | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis / Celery broker |
| `WANDB_API_KEY` | (none) | Weights & Biases |
| `MATERIALS_PROJECT_API_KEY` | (none) | MP API |
| `OQMD_API_KEY` | (none) | OQMD API |
| `MODEL_PATH` | `checkpoints/best_model.pt` | Inference model path |

Copy `.env.example` to `.env` and fill in secrets. The `.env` file is loaded by `python-dotenv`.

---

## 14. Quick Reference

```bash
make install       # Install package + deps
make train         # Train v3_li model
make test          # Run tests
make lint          # Check code quality
make format        # Auto-format code
make evaluate      # Cross-validate
make benchmark     # Benchmark trained model
make screen        # Screen a formula
make docs          # Build docs
make clean         # Clean caches
```
