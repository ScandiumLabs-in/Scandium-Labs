# Scandium Labs — Complete Technical Documentation

## 1. System Overview

Scandium Labs is a physics-informed graph neural network framework for
high-throughput screening of solid-state electrolyte (SSE) materials.
It predicts thermodynamic stability and electronic properties of
Li-containing crystals using a multi-task ALIGNN architecture with
gradient-normalized loss balancing and pre-cached graph loading.

### 1.1 Capabilities

| Capability | Detail |
|------------|--------|
| Property prediction | formation_energy (eV/atom), energy_above_hull (eV/atom), band_gap (eV) |
| Training data | 10,000 Li-containing structures from Materials Project |
| Multi-task learning | 3 shared-backbone task heads with GradNorm loss balancing |
| Physics constraints | Arrhenius gating, thermodynamic gating, diffusion residual |
| Uncertainty | MC Dropout (20 samples) for prediction intervals |
| OOD detection | Isolation Forest on graph embeddings |
| Screening | Pareto-optimal ranking + chemical family debiasing |
| Inference modes | Single structure, batch JSON, CIF files |
| API | FastAPI with JWT auth, Postgres+Redis backend |
| Interactive UI | Streamlit dashboard with 4 pages |

## 2. Architecture

### 2.1 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  CIF / pymatgen.Structure                                    │
│  • Lattice parameters (a, b, c, α, β, γ)                    │
│  • Atomic species + fractional coordinates                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ALIGNNGraphBuilder                                          │
│  • cutoff = 8.0 Å                                            │
│  • max_neighbors = 16                                        │
│  • GaussianRBF (num_rbf = 64) for pairwise distances        │
│  • SphericalBesselRBF (num_sbf = 32) for bond angles        │
│                                                             │
│  Outputs: {crystal_graph, line_graph}                       │
│                                                             │
│  crystal_graph:                                              │
│    x:          (N, 92)     — atom features                  │
│    edge_index: (2, E)      — neighbor indices               │
│    edge_attr:  (E, 64)     — RBF-encoded distances          │
│    pos:        (N, 3)      — Cartesian coordinates          │
│    global_feats: (16,)     — density, volume, etc.          │
│                                                             │
│  line_graph (bond-angle graph):                              │
│    edge_index: (2, L)      — angle adjacency                │
│    edge_attr:  (L, 32)     — SBF-encoded angles             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  FeatureEngineer                                             │
│  • Concatenates: electronegativity, atomic radius,           │
│    ionization energy, group, period, mendeleev_no,           │
│    melting_point, covalent_radius, electron_affinity         │
│  • Pads/reorders to 92-dim standard space                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ScandiumPINNGNN                                             │
│                                                             │
│  1. AtomEncoder:  Linear(92→128) → LayerNorm → SiLU         │
│                   → Linear(128→128)                          │
│  2. EdgeEncoder:  Linear(64→64) → SiLU → Linear(64→64)      │
│                                                             │
│  3. ALIGNN × 4:                                              │
│     ┌──────────────────────────────────────────────────┐     │
│     │  ALIGNNLayer                                       │     │
│     │  ┌───────────────────┐   ┌────────────────────┐   │     │
│     │  │ lg_conv:           │   │ cg_conv:            │   │     │
│     │  │ CrystalMPNN        │   │ CrystalMPNN         │   │     │
│     │  │ edge(64)→edge(64)  │   │ node(128)→node(128) │   │     │
│     │  │ + line_graph edges │   │ + updated edges     │   │     │
│     │  └───────────────────┘   └────────────────────┘   │     │
│     └──────────────────────────────────────────────────┘     │
│                                                             │
│  4. GraphTransformer × 2:                                    │
│     ┌──────────────────────────────────────────────────┐     │
│     │  Pre-LN → MultiheadAttention(4) → Residual       │     │
│     │  Pre-LN → FFN(GELU, 128→512→128) → Residual      │     │
│     └──────────────────────────────────────────────────┘     │
│                                                             │
│  5. PINNConstraintModule:                                    │
│     ┌──────────────────────────────────────────────────┐     │
│     │  h = LayerNorm(h + h × σ(G_A) × σ(G_T))          │     │
│     │  G_A = Linear(128→1)  — Arrhenius gate           │     │
│     │  G_T = MLP(128→64→1)  — Thermodynamic gate       │     │
│     └──────────────────────────────────────────────────┘     │
│                                                             │
│  6. AttentionGlobalPool:                                     │
│     ┌──────────────────────────────────────────────────┐     │
│     │  gate = Linear(128→64) → SiLU → Linear(64→1)     │     │
│     │  α = softmax(gate)  per-graph                     │     │
│     │  h_graph = sum(α * h_node)                        │     │
│     └──────────────────────────────────────────────────┘     │
│                                                             │
│  7. Global Feature Combiner:                                 │
│     ┌──────────────────────────────────────────────────┐     │
│     │  cat(h_graph, global_feats) → Linear(144→128)    │     │
│     └──────────────────────────────────────────────────┘     │
│                                                             │
│  8. Task Heads:                                              │
│     ┌──────────────────────────────────────────────────┐     │
│     │  formation_energy: Linear(128→64) → SiLU          │     │
│     │                    → Linear(64→1)                 │     │
│     │  band_gap:        Linear(128→64) → SiLU           │     │
│     │                    → Linear(64→1)                 │     │
│     │  energy_above_hull: TwoStageEahHead               │     │
│     │    Stage 1: Linear(128→64) → SiLU → Linear(64→1) │     │
│     │             → Sigmoid → p_unstable                │     │
│     │    Stage 2: Linear(128→64) → SiLU → Linear(64→1) │     │
│     │             → Softplus → magnitude                │     │
│     │    Combined: eah_pred = p_unstable * magnitude    │     │
│     └──────────────────────────────────────────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Predictions:                                                 │
│  • formation_energy:   (B, 1)  — eV/atom                    │
│  • energy_above_hull:  (B, 1)  — eV/atom (TwoStage)         │
│  • band_gap:           (B, 1)  — eV                         │
│  • p_unstable:         (B, 1)  — stability probability      │
│  • eah_magnitude:      (B, 1)  — EaH magnitude              │
│  • log_var_{task}:     (B, 1)  — aleatoric uncertainty       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 CrystalMPNN (Core Message-Passing Layer)

The `CrystalMPNN` is a modified `MessagePassing` from PyTorch Geometric
with sum aggregation:

```
Message:  m_ij = MLP([h_i || h_j || e_ij])    # 2×Linear + SiLU
Update:   h_i' = LayerNorm(h_i + MLP([h_i || sum_j m_ij]))
```

Where:
- `h_i` = node features at node i (dim: hidden_dim)
- `e_ij` = edge features between i and j (dim: edge_dim)
- `MLP` = 2-layer SiLU network: Linear → SiLU → Linear

### 2.3 Two-Stage EaH Head

The energy-above-hull prediction uses a two-stage architecture
to avoid the collapse-to-zero problem where the model always predicts
EaH ≈ 0 (since ~70% of training targets are ~0 eV/atom):

```
Stage 1 (Stability Classifier):
  p_unstable = σ(MLP(128→64→1))

Stage 2 (Magnitude Regressor):
  eah_magnitude = Softplus(MLP(128→64→1))

Combined:
  eah_pred = p_unstable × eah_magnitude
```

Loss: `L = λ_bce × BCE(p_unstable, y > 0.1) + λ_reg × MSE(magnitude, max(y, 0.1)) + λ_stable × MSE(prediction, 0 | y ≤ 0.1)`

## 3. Loss Functions

### 3.1 GradNormLoss

Adaptive multi-task loss balancing via gradient normalization
(Chen et al., 2018):

```
L_total = Σ w_i × L_i

For each task i:
  G_w_i = ||∇_{θ_shared} w_i × L_i||    # weighted gradient norm
  target_i = G_avg × (L_i / L_avg)^α     # target gradient norm
  L_grad = Σ |G_w_i - target_i|          # gradient-level loss

Update: w_i ← w_i - lr × ∂L_grad / ∂w_i
Renormalize: Σ w_i = n_tasks
```

- `α = 1.5` (restoring strength — higher = stronger balancing)
- GradNorm is optimized separately from the main model
- Weights are renormalized after each step to prevent drift
- Initial weights: formation=1.0, eah=1.0, band_gap=0.4

### 3.2 TwoStageEahLoss

```
L = λ_bce × BCE(p_unstable, y_gt > 0.1)
  + λ_reg × MSE(eah_magnitude, max(y_gt, 0.1))
  + λ_stable × MSE(eah_pred[y_gt ≤ 0.1], 0)
```

### 3.3 PINNLoss (optional, not used in current training)

```
L = λ_data × Σ w_task × MSE(pred_task, target_task)
  + λ_arrhenius × Var(log₁₀(σ×T) + Ea/(k_B×T×ln10))
  + λ_thermodynamic × mean(ReLU(-EaH))
  + λ_physics × mean((dc/dt - D×∇²c)²)   # diffusion residual
```

## 4. Training Pipeline

### 4.1 Data Flow

```
build_dataset.py → dataset_cache.pt + split_indices.pt
       │
       ▼
cache_graphs.py → graphs/*.pt (10,000 files)
       │
       ▼
train_v3_li.py → LazyGraphDataset → DataLoader(workers=4)
       │
       ▼
    ScandumPINNGNN.forward() → predictions
       │
       ▼
    GradNormLoss(TwoStageEahLoss, MSELoss) → weighted total
       │
       ▼
    scaler.scale(total).backward() → optimizer.step()
       │
       ▼
    Validation → Early Stopping (patience=40)
       │
       ▼
    Test → metrics.json
```

### 4.2 Optimizer & Scheduler

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW (θ = 0.9, β = 0.999, ε = 1e-8) |
| Weight decay | 1e-5 |
| LR | 0.0005 |
| Warmup | 500 steps (linear) |
| Scheduler | Cosine annealing with warm restarts |
| Gradient clip | 1.0 (max norm) |
| Mixed precision | AMP fp16 with GradScaler |

### 4.3 Batch Processing

```
Batch size: 16
Gradient accumulation: 2
Effective batch: 32

For each micro-batch:
  1. Forward (AMP enabled)
  2. Compute task losses
  3. GradNorm: compute_total + update_weights
  4. Backward (scale if AMP)
  5. If accum steps reached:
     a. Unscale (AMP)
     b. Clip gradients (max 1.0)
     c. Optimizer step
     d. Zero gradients
```

### 4.4 Gradient Checkpointing

- Enabled: `True`
- VRAM saved: 2.4× (470 MB vs 1127 MB)
- Speed cost: 33% (1253ms vs 943ms per step)
- Applied to ALIGNN layers during `checkpoint()` wrapper

### 4.5 DataLoader Configuration

| Setting | Value |
|---------|-------|
| `num_workers` | 4 |
| `pin_memory` | True |
| `prefetch_factor` | 2 |
| `persistent_workers` | True |
| `multiprocessing_context` | fork |
| `collate_fn` | `collate_fn` (Batch.from_data_list) |

## 5. Dataset Reference

### 5.1 Available Versions

| Version | Path | N | Elements | Tasks |
|---------|------|---|----------|-------|
| v1_817 | `datasets/v1_817/` | 817 | Li-families | all 5 |
| v2_1000_smoketest | `datasets/v2_1000_smoketest/` | 1,008 | 76 elements | all 5 |
| v2_10000 | `datasets/v2_10000/` | 3,635 | 76 elements | all 5 |
| v3_li_10000 | `datasets/v3_li_10000/` | 10,000 | Li ≥ 5% | 3 (Ef, EaH, BG) |

### 5.2 File Format

Each dataset directory contains:
```
dataset_cache.pt        → {"structures": list[Structure], "targets": dict[str, array]}
split_indices.pt        → {"train": list[int], "val": list[int], "test": list[int]}
normalizer.json         → {"mean": {}, "std": {}, "min": {}, "max": {}}
metadata.json           → {"version", "n_structures", "elements", "stats"}
dataset_report.json     → per-task statistics summary
graphs/                 → {0..9999}.pt (crystal_graph, line_graph tuples)
prebuilt_graphs.pt      → (legacy) monolithic graph list
```

### 5.3 v3_li_10000 Statistics

| Property | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| Formation energy | -1.96 | 0.92 | -4.77 | 0.69 |
| Energy above hull | 0.14 | 0.42 | 0.00 | 3.52 |
| Band gap | 1.26 | 1.45 | 0.00 | 6.63 |

## 6. Scripts Reference

### 6.1 Training Scripts

| Script | Description |
|--------|-------------|
| `scripts/train/train.py` | Config-based trainer (uses `ScandiumTrainer`) |
| `scripts/train/train_v3_li.py` | Standalone v3 Li training (manual loop + GradNorm) |
| `scripts/train/experiment_sweep.py` | Versioned experiment runner |

### 6.2 Preprocessing Scripts

| Script | Description |
|--------|-------------|
| `scripts/preprocess/build_dataset.py` | Downloads + cleans + splits + normalizes data |
| `scripts/preprocess/cache_graphs.py` | Builds individual `.pt` graph files for all structures |
| `scripts/maintenance/rebuild_li_dataset.py` | Downloads Li-containing MP structures (≥5% Li) |

### 6.3 Maintenance Scripts

| Script | Description |
|--------|-------------|
| `scripts/maintenance/profile_training.py` | PyTorch profiler + param/memory stats |
| `scripts/maintenance/profile_dataloader.py` | DataLoader num_workers benchmark |
| `scripts/maintenance/benchmark_throughput.py` | Throughput benchmark (GC vs no-GC) |

## 7. Configuration Reference

### 7.1 model_config_v3_li.yaml (Active)

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
  use_gradient_checkpointing: true

graph:
  cutoff: 8.0          # Å
  max_neighbors: 16
  num_rbf: 64           # Gaussian RBF for distances
  num_sbf: 32           # Spherical Bessel RBF for angles

tasks:
  - {name: formation_energy,    weight: 1.0, scale: linear}
  - {name: energy_above_hull,   weight: 1.0, scale: linear, two_stage: true}
  - {name: band_gap,            weight: 1.0, scale: linear}

gradnorm:
  enabled: true
  alpha: 1.5

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 0.0005
  warmup_steps: 500
  max_epochs: 150
  patience: 40
  scheduler: cosine_with_restarts
  optimizer: AdamW
  weight_decay: 0.00001
  gradient_clip: 1.0
  mixed_precision: true
  normalize_targets: true
```

### 7.2 Other Configs

| Config | Purpose |
|--------|---------|
| `model_config.yaml` | v1 — 5 tasks, hidden_dim=256, 4 layers, 300 epochs |
| `model_config_v2.yaml` | v2 — reduced, hidden_dim=128, 2 layers, 100 epochs |
| `model_config_v3.yaml` | v3 — hidden_dim=200, 3 layers, 100 epochs |
| `data_config.yaml` | Data sources, cleaning params, split ratios |
| `deploy_config.yaml` | API settings, inference engine defaults |
| `finetune_config.yaml` | Fine-tuning hyperparams |
| `ds_config.json` | DeepSpeed ZeRO-2 distributed config |

## 8. Performance Characteristics

### 8.1 Throughput (GTX 1650, 4 GB)

| Config | Step Time | Graphs/s | VRAM |
|--------|-----------|----------|------|
| Baseline (GC) | 1,253 ms | 12.8 | 470 MB |
| No GC | 943 ms | 17.0 | 1,127 MB |
| DataLoader w=0 | N/A | 5.7 | N/A |
| DataLoader w=4 | N/A | 13.2 | N/A |
| torch.compile | TBD | TBD | TBD |

### 8.2 Memory Breakdown

| Component | GC On | GC Off |
|-----------|-------|--------|
| Model parameters (fp32) | 4.9 MB | 4.9 MB |
| Activations (1 batch) | ~300 MB | ~960 MB |
| Optimizer states | ~20 MB | ~20 MB |
| DataLoader buffers | ~150 MB | ~150 MB |
| **Total** | **~470 MB** | **~1,127 MB** |

### 8.3 Bottleneck Priorities

1. **DataLoader workers** (fixed: 0→4, +132% throughput)
2. **On-the-fly graph building** (fixed: pre-cached all 10k graphs)
3. **Per-batch MSELoss creation** (fixed: moved to instance variable)
4. **Model size** (scaled: 728K→1.28M params, 177→470 MB VRAM)
5. **`torch.compile`** (deferred: needs stable training baseline)

## 9. Checkpoint Format

```python
{
    "epoch": int,             # Last training epoch
    "model": dict,            # state_dict
    "val_loss": float,        # Best validation loss
    "config": dict,           # Model config used for training
    "normalizer": dict,       # (optional) normalization stats
    "optimizer": dict,        # (optional) optimizer state
    "metrics": dict,          # (optional) test metrics
}
```

## 10. Test Suite

```bash
pytest tests/ -v
```

65 tests pass, 11 pre-existing failures:

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_models.py` | 2 | Pass |
| `test_pipeline.py` | 6 | Pass |
| `test_data.py` | 2 | Pass |
| `test_inference.py` | 1 | Pass |
| `test_api.py` | 1 | Pass |
| `test_training_normalization.py` | 5 | Pass |
| `test_data_audit.py` | 4 | Pass |
| `test_reference_materials.py` | 7 | Pass |

## 11. Package Structure

### `src/` (8 subpackages, 28 public symbols)

```
src/
├── data/          # 4 modules: dataset, cleaner, collectors, splitter
├── models/
│   ├── gnn/       # 2 modules: alignn, layers
│   └── heads/     # 2 modules: two_stage_eah, pretrained
├── training/      # 10 modules: trainer, loaders, losses, scheduler, ...
├── inference/     # 4 modules: engine, ranking, stability, validation
├── evaluation/    # 2 modules: metrics, ood
├── chemistry/     # 1 module: family_id
├── graphs/        # 2 modules: builder, features
├── explainability/# 2 modules: attention, gradients
└── utils/         # 3 modules: config, io, logging
```

### Scripts (20+)

```
scripts/
├── train/         # train.py, train_v3_li.py, experiment_sweep.py
├── preprocess/    # build_dataset.py, cache_graphs.py
├── inference/     # screen_candidates.py
├── evaluate/      # cross_validate.py
└── maintenance/   # profile_*.py, benchmark_*.py, rebuild_li_dataset.py
```
