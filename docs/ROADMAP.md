# Product & Research Roadmap

> Strategic roadmap for Scandium Labs' AI-driven solid-state electrolyte
> discovery platform. Organized by completion status and time horizon.
>
> **Last updated:** July 2026
> **Current version:** v0.5.0

---

## Table of Contents

1. [Completed (v0.1.0 – v0.5.0)](#1-completed)
2. [In Progress (v0.6.0 – Current Sprint)](#2-in-progress)
3. [Next Release (v0.7.0 – Q3 2026)](#3-next-release)
4. [Short-Term (v0.8.0 – v0.9.0, Q4 2026)](#4-short-term)
5. [Medium-Term (v1.0.0 – v1.5.0, H1 2027)](#5-medium-term)
6. [Future Research (2027–2028)](#6-future-research)
7. [Future Infrastructure (2027–2028)](#7-future-infrastructure)
8. [Decision Log](#8-decision-log)

---

## 1. Completed (v0.1.0 – v0.5.0)

### ✅ Core Architecture

#### Model

- [x] **ScandiumPINNGNN architecture** — Combined ALIGNN backbone + Graph Transformer + PINN constraints + multi-task heads. Reference: `src/models/scandium_model.py`
- [x] **ALIGNN layers** — Alternating line-graph → crystal-graph message passing via `CrystalMPNN` layers. Reference: `src/models/gnn/alignn.py`
- [x] **Graph Transformer layers** — Stacked multi-head self-attention for long-range interactions. Reference: `src/models/gnn/layers.py`
- [x] **PINN Constraint Module** — Learned physics gates: Arrhenius gating and thermodynamic gating. Reference: `src/models/gnn/layers.py`
- [x] **Two-Stage EaH Head** — Binary stability classifier + conditional magnitude regressor. Reference: `src/models/heads/two_stage_eah.py`
- [x] **Multi-task regression heads** — Separate MLP heads for formation energy, energy above hull, band gap. Reference: `src/models/scandium_model.py`
- [x] **Multi-task PINN loss** — `PINNLoss` with data MSE + Arrhenius constraint + thermodynamic constraint + physics residual. Reference: `src/training/losses.py`

#### Training

- [x] **ScandiumTrainer class** — Full training orchestrator: build_model, build_optimizer, build_loss, train_epoch, validate, train. Reference: `src/training/trainer.py`
- [x] **GradNorm adaptive loss balancing** — The only known implementation with analytical gradient updates for GradNorm weights (no `create_graph=True` overhead). Reduces autograd calls from 7 to 3 per step. Reference: `src/training/losses.py`
- [x] **Cosine annealing scheduler** — `CosineAnnealingWarmRestarts` in `ScandiumTrainer`; `cosine_with_restarts` in `train_v3_li.py`. Reference: `src/training/scheduler.py`
- [x] **Size-bucketed batch sampler** — Reduces padding waste by ~30%. Reference: `src/data/samplers.py`
- [x] **Gradient checkpointing** — 2.4× VRAM savings at 33% speed cost. Auto-detect based on VRAM (< 6 GB = enable). Reference: `configs/model_config_v3_li.yaml`
- [x] **Mixed precision (AMP) training** — `GradScaler` + `autocast` for fp16 training. Reference: `src/training/trainer.py`
- [x] **DataLoader optimizations** — `pin_memory=True`, `multiprocessing_context='fork'` for Python 3.14+ CUDA compatibility, `num_workers=4`. Reference: `scripts/train/train_v3_li.py`

#### Data

- [x] **LazyGraphDataset** — Memory-efficient, on-disk graph loading with optional in-memory cache. Prevents 29-minute first-epoch graph building overhead. Reference: `src/data/dataset.py`
- [x] **SolidElectrolyteDataset** — Legacy in-memory dataset for on-the-fly graph building. Reference: `src/data/dataset.py`
- [x] **DataCleaner** — Structure deduplication via `StructureMatcher`, property range filtering, size filtering. Reference: `src/data/cleaner.py`
- [x] **PropertyNormalizer** — Z-score normalization with fit/transform/inverse_transform/save/load. Reference: `src/data/cleaner.py`
- [x] **MaterialsProjectCollector** — API-based data collection from Materials Project. Reference: `src/data/collectors.py`
- [x] **Multiple data source collectors** — JARVIS, OQMD, AFLOW, NOMAD collectors. Reference: `src/data/collectors.py`
- [x] **Composition-based split** — GroupShuffleSplit by element combination to prevent element leakage. Reference: `src/data/splitter.py`
- [x] **Graph caching pipeline** — Single-process CPU-based graph pre-caching. Reference: `scripts/preprocess/cache_graphs.py`
- [x] **Dataset building pipeline** — End-to-end: collect → clean → split → cache. Reference: `scripts/preprocess/build_dataset.py`
- [x] **v3_li_10000 dataset** — 10k Li-containing structures with 83/6/11 train/val/test split. Li ≥ 5 at.%, family-balanced across halides/oxides/sulfides/phosphates.

#### Inference

- [x] **InferenceEngine** — Full prediction pipeline: graph building → model → MC dropout → denormalization → stability check → recommendation. Reference: `src/inference/engine.py`
- [x] **MC Dropout uncertainty quantification** — Configurable number of Monte Carlo forward passes. Reference: `src/models/scandium_model.py`
- [x] **ParetoRanker** — Multi-objective ranking by ionic conductivity, EaH, and confidence. Non-dominated sorting + weighted composite score. Reference: `src/inference/ranking.py`
- [x] **Stability utilities** — `compute_hull_energy()` for MP convex hull queries, `resolve_stability()` for inconsistency detection. Reference: `src/inference/stability.py`
- [x] **Structure validation** — Pre-inference structure sanity checks (charge, density, distances, cell volume). Reference: `src/inference/validation.py`
- [x] **OOD detection** — Embedding-based out-of-distribution detection. Reference: `src/inference/engine.py`
- [x] **Coverage gating** — Task-level training data coverage report; gates predictions for tasks with insufficient training data. Reference: `src/training/data_audit.py`

#### Experiment Management

- [x] **ExperimentTracker** — Research-grade experiment manager: auto-generated run IDs, per-epoch metrics, checkpoint management, plot generation, report writing. Reference: `src/training/experiment_tracker.py`
- [x] **RunRegistry** — CSV-based run index at `runs/index.csv` with best metrics per run. Reference: `src/training/experiment_tracker.py`
- [x] **MetricsStore** — JSON + CSV persistence of per-epoch metrics. Reference: `src/training/experiment_tracker.py`
- [x] **CheckpointManager** — Saves multiple checkpoints per metric (best_val_loss, best_*_mae, best_*_r2). Reference: `src/training/experiment_tracker.py`
- [x] **PlotGenerator** — Automatic plots: loss curves, per-task MAE/R², GradNorm weights, confusion matrix, ROC/PR, calibration, system metrics. Reference: `src/training/experiment_tracker.py`
- [x] **Auto-generated reports** — TRAINING_SUMMARY.md, BEST_MODEL_REPORT.md, MODEL_CARD.md, EXPERIMENT_LEADERBOARD.md, STOP_REPORT.md. Reference: `src/training/experiment_tracker.py`

#### Analysis

- [x] **analyze_training.py** — Post-hoc analysis: learning curves, per-task metrics, GradNorm trajectories, system metrics, resume audit, timeline, prediction diagnostics. Reference: `scripts/analyze/analyze_training.py`
- [x] **Scorecard generation** — Pass/fail checklist for experiment quality. Reference: `scripts/analyze/analyze_training.py`
- [x] **Resume detection** — Automatic detection of resume points from timestamp gaps. Reference: `scripts/analyze/analyze_training.py`

#### API & Deployment

- [x] **FastAPI application** — 4 endpoints: `/health`, `/screen`, `/screen/upload`, `/job/{job_id}`. Reference: `api/main.py`
- [x] **Celery async workers** — Background task processing for batch screening. Reference: `api/tasks.py`
- [x] **JWT authentication** — Bearer token auth with HS256. Reference: `api/auth.py`
- [x] **PostgreSQL integration** — Job, Material, ScreeningResult models via SQLAlchemy. Reference: `api/database.py`
- [x] **Docker Compose** — 6 services: API (2 replicas), Worker (4 replicas, GPU), TorchServe, Postgres, Redis, Flower. Reference: `docker-compose.yml`
- [x] **TorchServe integration** — Model serving via `.mar` archive. Reference: `docker-compose.yml`

#### Testing & Quality

- [x] **65 passing tests** — Unit and integration tests. Reference: `tests/`
- [x] **Ruff linting** — Automated code formatting. Reference: `pyproject.toml`
- [x] **Pre-commit hooks** — Pre-commit validation. Reference: `.pre-commit-config.yaml`
- [x] **Makefile** — Common development commands. Reference: `Makefile`
- [x] **Comprehensive documentation** — 41 docs files covering architecture, data, training, inference, optimization, deployment.

#### Performance Optimizations

- [x] **Full profiling suite** — DataLoader bench, throughput bench, training profiler, torch compile bench. Reference: `scripts/maintenance/`
- [x] **Before/after benchmarks** — 5.7 → 12.8 graphs/s (132% improvement). Reference: `docs/OPTIMIZATION_REPORT.md`
- [x] **Resource profiles** — Small/Medium/Large GPU config templates. Reference: `docs/RESOURCE_PROFILES.md`
- [x] **Configuration optimization** — hidden_dim=128, 4 ALIGNN layers, 2 Transformer layers, batch=16, accum=2.

---

## 2. In Progress (v0.6.0 — Current Sprint)

These items are actively being worked on in the current development cycle.

### 🚧 Experiment B: GradNorm + Cosine Scheduler

**Status:** Training run E (GradNorm enabled, cosine_with_restarts scheduler)

**Objective:** Validate that GradNorm + cosine scheduler together improve convergence vs. fixed weights.

**Hypothesis:**
```
GradNorm + Scheduler > GradNorm only > Scheduler only > Fixed weights
```

**Expected completion:** Within ongoing training run.

### 🚧 Optuna Hyperparameter Search

**Status:** Planned, implementation deferred until GradNorm + scheduler training run completes.

**Search space:**

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [1e-5, 1e-3] | Log |
| `hidden_dim` | [64, 256] | Linear (step 32) |
| `dropout` | [0.05, 0.35] | Linear |
| `weight_decay` | [1e-6, 1e-4] | Log |
| `gradnorm.alpha` | [0.5, 2.0] | Linear |
| `batch_size` | [8, 32] | Linear (step 8) |

**Trials:** 50, using Optuna's `TPESampler`.

**Success metric:** Validation R² averaged across all 3 tasks.

**Expected completion:** Q3 2026.

### 🚧 Graph Cache Build

**Status:** Running (PID in `AGENTS.md`). ~6.1 graphs/s on CPU.

**Progress:** ~2,295 / 10,000 graphs cached.

**Expected completion:** ~21 minutes remaining.

---

## 3. Next Release (v0.7.0 — Q3 2026)

### 🔜 Band Gap Improvements

**Current state:** Band gap MAE = 0.215 eV, R² = 0.75. This is the weakest-performing task.

**Planned approaches:**

| Approach | Expected Improvement | Priority | Dependencies |
|----------|--------------------|----------|--------------|
| Increase band gap weight in GradNorm | Moderate | High | Config change only |
| Band gap-specific data augmentation (strain) | Moderate | Medium | Data pipeline work |
| Separate band gap fine-tuning pass | High | High | Two-phase training |
| Semi-empirical correction (SCAN-DFT) | High | Low | External DFT calculations |
| Band gap as classification (metal/semi/insulator) | Moderate | Low | Task head redesign |

**Target:** Band gap MAE < 0.15 eV, R² > 0.85.

### 🔜 Energy Above Hull Improvements

**Current state:** EaH MAE = 0.089 eV/atom, R² = 0.87, Stability F1 = 0.87.

**Planned approaches:**

| Approach | Expected Improvement | Priority |
|----------|--------------------|----------|
| Temperature scaling for p_unstable calibration | Moderate | Medium |
| Weighted EaH loss (more weight on < 0.1 eV) | Moderate | Medium |
| Ensemble of two-stage heads | High | Low (compute cost) |

**Target:** Stability F1 > 0.92, EaH MAE < 0.07 eV/atom.

### 🔜 Non-Li Data Collection

**Current state:** Li-only (Li ≥ 5 at.%).

**Planned datasets:**

| System | Target Size | Source | Difficulty |
|--------|------------|--------|------------|
| Na-ion electrolytes | 5,000 | Materials Project | Low — same pipeline |
| Mg-ion electrolytes | 2,000 | Materials Project | Low |
| Zn-ion electrolytes | 1,000 | Materials Project + literature | Low |
| Known SSEs (experimental) | 500 | Literature extraction | High — manual curation |

**Target:** Multi-element model covering Li + Na + Mg + Zn by end of Q3 2026.

---

## 4. Short-Term (v0.8.0 – v0.9.0, Q4 2026)

### Non-Li Model Training

- [ ] Train separate models for Na, Mg, Zn systems
- [ ] Fine-tune Li backbone on non-Li data (transfer learning)
- [ ] Evaluate cross-system generalization (can Li model predict Na properties?)
- [ ] Compare: training from scratch vs. fine-tuning

### Ionic Conductivity Prediction

- [ ] Collect experimental ionic conductivity data from literature (~500 points)
- [ ] Design conductivity prediction head (compositional descriptors + graph features)
- [ ] Train conductivity model (separate or joint with existing tasks)
- [ ] Validate: compare predicted σ with known superionic conductors

### Inference Optimization

- [ ] **ONNX export** — Convert model to ONNX for cross-platform deployment
- [ ] **TensorRT integration** — 5× inference speedup on NVIDIA GPUs
- [ ] **Batch inference optimization** — MC Dropout across batch, not per-structure
- [ ] **Quantization** — INT8 quantization for edge deployment

### Infrastructure

- [ ] **CI/CD pipeline** — GitHub Actions: lint, test, build, deploy
- [ ] **Multi-GPU training** — PyTorch DDP (DistributedDataParallel)
- [ ] **Pre-commit hooks fully configured** — Automated formatting
- [ ] **Docker image optimization** — Multi-stage builds, reduced image size

---

## 5. Medium-Term (v1.0.0 – v1.5.0, H1 2027)

### Scientific Goals

- [ ] **State-of-the-art performance** — Achieve best-known results for Li SSE property prediction on MatBench
- [ ] **MatBench submission** — Benchmark against standardized materials ML datasets
- [ ] **Multi-fidelity learning** — Combine DFT, ML, and experimental data in a unified framework
- [ ] **Uncertainty calibration** — Temperature scaling, ensemble methods for calibrated uncertainty
- [ ] **DFT validation pipeline** — Automated DFT verification of top-predicted candidates

### Production

- [ ] **Production API v1.0** — Rate limiting, usage tracking, multi-tenant, SLA
- [ ] **Enterprise on-prem deployment** — License-managed Docker images
- [ ] **Web dashboard** — Interactive Streamlit app for non-technical users
- [ ] **Experiment database** — PostgreSQL tracking for all experiments (replace CSV)
- [ ] **Prometheus + Grafana monitoring** — Real-time dashboards for infrastructure
- [ ] **Flower monitoring** — Celery task monitoring dashboard

### Community

- [ ] **Open-source launch** — Public repository with comprehensive documentation
- [ ] **Tutorial notebooks** — Jupyter/Colab notebooks for onboarding
- [ ] **Discord community** — User support and discussion channel
- [ ] **Contribution guide** — Clear guidelines for external contributors

---

## 6. Future Research (2027–2028)

### Transfer Learning & Foundation Models

| Direction | Description | Impact |
|-----------|-------------|--------|
| **M3GNet backbone** | Replace ALIGNN with M3GNet universal force-field | Access to pretrained universal atom embeddings |
| **Pre-train on all MP data** | 150k+ structures vs. current 10k | Broader chemical knowledge |
| **Fine-tuning framework** | Customer-specific property prediction | Enterprise value |

### Generative Models for Crystal Design

| Direction | Description | Timeline |
|-----------|-------------|----------|
| **Diffusion models** | Generate crystal structures with target properties | 2027 |
| **GFlowNets** | Generative flow networks for diverse candidate generation | 2027 |
| **Composition-to-structure** | Predict stable crystal structure from composition | 2028 |

### Advanced Physics Integration

| Direction | Description |
|-----------|-------------|
| **Equivariant GNNs (MACE, NequIP)** | SE(3)-equivariant message passing for better energy predictions |
| **Neural force fields** | Learn interatomic potentials for MD simulations |
| **Multi-fidelity active learning** | Query DFT for uncertain predictions; retrain iteratively |

### Autonomous Discovery

| Direction | Description |
|-----------|-------------|
| **ML → DFT → Synthesis loop** | Predict → validate with DFT → synthesize top candidates |
| **Laboratory integration** | API for autonomous synthesis robots |
| **Self-driving lab** | Closed-loop materials discovery |

---

## 7. Future Infrastructure (2027–2028)

### Compute Scaling

| Initiative | Description | Priority |
|------------|-------------|----------|
| **Multi-GPU training** | DDP for 4× A100 training | High |
| **CUDA graphs** | Reduced kernel launch overhead | Medium |
| **FlashAttention integration** | Faster attention for transformer layers | Medium |
| **Distributed inference** | Shard models across GPUs | Low |

### Deployment

| Initiative | Description | Priority |
|------------|-------------|----------|
| **Kubernetes Helm charts** | Production-grade Kubernetes deployment | High |
| **TorchServe in production** | Auto-scaling model serving | High |
| **Redis Cluster / Sentinel** | High-availability Celery broker | Medium |
| **Postgres replication** | Read replicas for job status queries | Medium |
| **CDN for static assets** | Frontend dashboard delivery | Low |

### MLOps

| Initiative | Description | Priority |
|------------|-------------|----------|
| **MLflow integration** | Model registry + experiment comparison | Medium |
| **Feature store** | Centralized crystal graph features | Medium |
| **Model versioning** | Versioned model deploys with rollback | High |
| **A/B testing framework** | Compare model versions in production | Medium |
| **Drift monitoring** | Detect data distribution shift | High |

---

## 8. Decision Log

Key architectural decisions and their rationale.

| Date | Decision | Rationale | Alternative Considered |
|------|----------|-----------|----------------------|
| 2026-05 | ALIGNN backbone | SOTA for crystal property prediction; captures bond angles via line graph | GAT, MPNN, SchNet |
| 2026-05 | LazyGraphDataset | Eliminates 29-min first-epoch overhead; memory-efficient | SolidElectrolyteDataset (on-the-fly) |
| 2026-05 | GradNorm (analytical) | Faster than autograd-based GradNorm; 3 vs 7 autograd calls | Fixed weights, uncertainty weighting |
| 2026-05 | Two-stage EaH | Solves EaH-collapse problem | Single-head regression, log-transform |
| 2026-05 | `fork` context for DataLoader | Required for Python 3.14 + CUDA compatibility | `spawn` (default, causes deadlock) |
| 2026-05 | GC auto-detect | Transparent optimization; enabled for < 6 GB VRAM | Always-on, always-off |
| 2026-06 | `ScandiumTrainer` vs. standalone loop | Train_v3_li.py provides full control; ScandiumTrainer provides convenience API | Single entry point |
| 2026-06 | ExperimentTracker > simple logging | Auto-generated reports, leaderboards, reproducibility | WandB-only, CSV-only |
| 2026-06 | Coverage gating | Prevents production deployment of untrained tasks | Ignore missing tasks |
| 2026-07 | Cache build (single-process CPU) | Avoids CUDA multiprocessing deadlocks | Multi-process GPU |

---

## Progress Visualization

```
v0.1  v0.2  v0.3  v0.4  v0.5  v0.6  v0.7  v0.8  v0.9  v1.0
│     │     │     │     │     │     │     │     │     │
ALIGNN│     │     │     │     │     │     │     │     │
──────●─────●─────●─────●─────●─────●─────●─────●─────●─────▶
      Lazy  Two- │     GC    │     │     │     │     │
      Graph Stage│     Opt  │     │     │     │     │
      Data  EaH  │     Auto │     │     │     │     │
            │     │          │     │     │     │     │
           Grad  │     │     │     │     │     │     │
           Norm  │     │     │     │     │     │     │
                 │     │     │     │     │     │     │
                Buck  │     │     │     │     │     │
                eting │     │     │     │     │     │
                      │     │     │     │     │     │
                     Optuna   │     │     │     │
                      BG  │     │     │     │
                      Imp  │     │     │     │
                           │     │     │     │
                          Non-Li│     │     │
                           Data  │     │     │
                                 │     │     │
                                ONNX /     │
                                TensorRT   │
                                           │
                                          Multi-
                                          GPU
```

---

*For the most up-to-date status, see `AGENTS.md` for current sprint details and `runs/index.csv` for experiment results.*
