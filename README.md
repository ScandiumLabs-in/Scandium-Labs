# Scandium Labs: Physics-Informed Graph Neural Networks for Solid-State Electrolyte Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

Scandium Labs is a deep learning framework for high-throughput screening of solid-state electrolytes (SSEs) using physics-informed graph neural networks (PINNs). The model jointly predicts thermodynamic stability, band gap, and related properties of Li-containing crystalline materials, targeting next-generation all-solid-state batteries.

The system is trained on ~10k Li-containing materials (Li ≥ 5 at.%) curated from the Materials Project, with family-balanced splits across halides, oxides, sulfides, and phosphates. The architecture combines an **ALIGNN** (Atomistic Line Graph Graph Neural Network) backbone with graph transformers, physics-informed constraints, multi-task regression heads, and automatic loss balancing via **GradNorm**.

## Key Features

- **Multi-Task Regression** — Simultaneously predicts formation energy, energy above hull, and band gap with task-specific heads
- **ALIGNN Architecture** — Alternating line-graph + crystal-graph message passing via `CrystalMPNN` layers for bond-aware representations
- **Graph Transformer** — Stacked multi-head self-attention layers for long-range interaction modeling
- **Physics-Informed Constraints** — Arrhenius gating and thermodynamic gating via `PINNConstraintModule`
- **Two-Stage EaH Head** — Stability classifier + magnitude regressor prevents energy-above-hull collapse to zero
- **GradNorm Loss Balancing** — Automatic per-task weight adaptation during training via gradient normalization
- **Uncertainty Quantification** — Monte Carlo Dropout with configurable samples yields prediction intervals
- **LazyGraphDataset** — On-disk pre-cached graph loading eliminates on-the-fly graph building overhead
- **Performance Optimized** — Gradient checkpointing saves 2.4× VRAM at 33% speed cost; DataLoader with 4 workers achieves 12.8 graphs/s on a 4 GB GPU
- **Memory-Efficient** — Runs on GTX 1650 (4 GB VRAM); model uses 470 MB VRAM with 1.28M parameters

## Model Architecture

```
CIF/Structure
    │
    ▼
ALIGNNGraphBuilder (cutoff=8.0Å, max_neighbors=16)
    │
    ├── CrystalGraph: atom_feats(92) + edge_feats(RBF, 64) + global_feats(16)
    └── LineGraph: bond_angles(SBF, 32)
    │
    ▼
ScandiumPINNGNN
    ├── AtomEncoder: Linear(92→128) → LayerNorm → SiLU → Linear(128→128)
    ├── EdgeEncoder: Linear(64→64) → SiLU → Linear(64→64)
    │
    ├── ALIGNN Layers × 4
    │   ├── LineGraph CrystalMPNN (edge → edge update)
    │   └── CrystalGraph CrystalMPNN (node → node update)
    │
    ├── GraphTransformer Layers × 2
    │   └── MultiheadAttention(4 heads) + FFN(GELU) + Residual
    │
    ├── PINNConstraintModule
    │   ├── Arrhenius Gate: Sigmoid(Linear(128))
    │   └── Thermodynamic Gate: Sigmoid(MLP(128))
    │
    ├── AttentionGlobalPool
    │   └── Gated soft-attention readout
    │
    ├── Global Feature Combiner (16-dim → 128)
    │
    └── Task Heads
        ├── formation_energy: MLP(128→64→1)
        ├── energy_above_hull: TwoStageEahHead
        │   ├── Stage 1: p_unstable = Sigmoid(MLP(128))
        │   └── Stage 2: magnitude = Softplus(MLP(128))
        └── band_gap: MLP(128→64→1)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `ScandiumPINNGNN` | `src/models/scandium_model.py` | Main model: encoder, ALIGNN backbone, transformer, PINN, pool, task heads |
| `ALIGNNLayer` | `src/models/gnn/alignn.py` | Alternating line-graph → crystal-graph message passing |
| `CrystalMPNN` | `src/models/gnn/layers.py` | Edge-augmented message passing: `MLP([h_i, h_j, e_ij])` |
| `GraphTransformerLayer` | `src/models/gnn/layers.py` | Multi-head self-attention + FFN with pre-norm residuals |
| `PINNConstraintModule` | `src/models/gnn/layers.py` | Physics gating: `h * σ(Arrhenius) * σ(Thermodynamic)` |
| `AttentionGlobalPool` | `src/models/gnn/layers.py` | Soft-attention graph readout |
| `TwoStageEahHead` | `src/models/heads/two_stage_eah.py` | Stability classifier + magnitude regressor for EaH |
| `PINNLoss` | `src/training/losses.py` | Multi-component loss: data MSE + Arrhenius + thermodynamic |
| `GradNormLoss` | `src/training/losses.py` | Automatic gradient-based per-task weight balancing |

## Installation

```bash
git clone https://github.com/scandium-labs/scandium-labs.git
cd scandium-labs
python -m venv venv
source venv/bin/activate
pip install -e .
```

For GPU training:
```bash
pip install -e ".[gpu]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Dataset

The primary dataset (`v3_li_10000`) contains **10,000 Li-containing crystalline structures** from the Materials Project, filtered at **Li ≥ 5 at.%**. Chemical family stratification ensures balanced representation.

| Property | Key | Unit | Coverage |
|----------|-----|------|----------|
| Formation Energy | `formation_energy` | eV/atom | ~100% |
| Energy Above Hull | `energy_above_hull` | eV/atom | ~100% |
| Band Gap | `band_gap` | eV | ~100% |

Split: **8,310 train / 586 val / 1,104 test** (83/6/11%). See `docs/DATASETS.md` for full details.

### Pre-cached Graphs

All 10,000 graphs are pre-built as individual `.pt` files in `datasets/v3_li_10000/graphs/`. The `LazyGraphDataset` class loads them on-demand from disk, eliminating the 29-minute first-epoch overhead of on-the-fly graph building.

```python
from src.data.dataset import LazyGraphDataset, collate_fn

ds = LazyGraphDataset(
    structure_list=structures,
    targets=targets,
    graph_dir="datasets/v3_li_10000/graphs",
)
```

## Training

```bash
# Train from scratch (current active config)
python scripts/train/train_v3_li.py

# With custom config
python scripts/train/train.py --config configs/model_config_v3_li.yaml

# Programmatic
python -c "
from src.training import ScandiumTrainer
trainer = ScandiumTrainer('configs/model_config_v3_li.yaml')
trainer.train()
"
```

### Current Training Config (`configs/model_config_v3_li.yaml`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `hidden_dim` | 128 | Hidden dimension size |
| `num_alignn_layers` | 4 | ALIGNN message-passing depth |
| `num_transformer_layers` | 2 | Graph transformer depth |
| `num_attention_heads` | 4 | Multi-head attention heads |
| `dropout` | 0.15 | Dropout / MC Dropout rate |
| `use_gradient_checkpointing` | True | Saves 2.4× VRAM |
| `use_two_stage_eah` | True | Decoupled EaH training |
| `batch_size` | 16 | Per-device batch size |
| `gradient_accumulation_steps` | 2 | Effective batch: 32 |
| `learning_rate` | 0.0005 | AdamW initial LR |
| `max_epochs` | 150 | Maximum epochs |
| `patience` | 40 | Early stopping patience |
| `scheduler` | cosine_with_restarts | LR schedule |
| `optimizer` | AdamW | Weight decay: 1e-5 |
| `mixed_precision` | True | AMP fp16 training |
| `normalize_targets` | True | Z-score normalization |

### Performance (GTX 1650, 4 GB)

| Metric | Value |
|--------|-------|
| Parameters | 1,281,321 |
| Model size (fp32) | 4.9 MB |
| Step time (GC on) | ~1,253 ms |
| Step time (GC off) | ~943 ms |
| Throughput (GC on) | 12.8 graphs/s |
| Throughput (GC off) | 17.0 graphs/s |
| Peak VRAM (GC on) | 470 MB (11.5%) |
| Peak VRAM (GC off) | 1,127 MB (27.5%) |
| Epoch time (8,310 samples) | ~353 s |

## Evaluation

```bash
# 5-fold cross-validation
python scripts/evaluate/cross_validate.py

# Throughput benchmark
python scripts/maintenance/benchmark_throughput.py
```

Metrics: MAE, RMSE, R², stability accuracy, expected calibration error (ECE).

## Inference

```bash
# Screen candidates from JSON
python scripts/inference/screen_candidates.py

# Programmatic
python -c "
from src.inference import InferenceEngine
engine = InferenceEngine('checkpoints/v3_li_10k_fresh/best_model.pt')
results = engine.screen(['Li6PS5Cl', 'Li3YCl6', 'Li7La3Zr2O12'])
"
```

## Project Structure

```
scandium-labs/
├── src/
│   ├── data/           # Dataset, data cleaning, collectors, splitting
│   ├── models/         # ScandiumPINNGNN, GNN layers, heads
│   │   ├── gnn/        #   ALIGNN, CrystalMPNN, GraphTransformer
│   │   └── heads/      #   TwoStageEahHead, PretrainedEncoder
│   ├── training/       # Trainer, losses, schedulers, distributed
│   ├── inference/      # Inference engine, ranking, stability
│   ├── evaluation/     # Metrics, OOD detection, cross-validation
│   ├── chemistry/      # Chemical featurization, family IDs
│   ├── graphs/         # Graph construction, feature engineering
│   ├── explainability/ # Attention visualization, integrated gradients
│   └── utils/          # Config, logging, I/O helpers
├── scripts/
│   ├── train/          # Training entrypoints
│   ├── preprocess/     # Dataset building, graph caching
│   ├── inference/      # Candidate screening
│   ├── evaluate/       # Cross-validation, benchmarks
│   └── maintenance/    # Profiling, benchmarking, dataset rebuild
├── configs/            # YAML/JSON configs
├── api/                # FastAPI backend
├── streamlit_app/      # Interactive dashboard
├── tests/              # Unit & integration tests (65 pass)
├── datasets/           # Preprocessed datasets
├── checkpoints/        # Trained model weights
├── docs/               # Architecture, datasets, optimization reports
└── archive/            # Historical/backup files
```

## Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term
```

Test results: **65 tests pass**, 11 pre-existing failures (unrelated to refactoring).

## Advanced Usage

### Graph Pre-caching

Build individual graph files to avoid on-the-fly construction during training:

```bash
python scripts/preprocess/cache_graphs.py
```

### Cross-Validation

```bash
python scripts/evaluate/cross_validate.py
```

### API Deployment

```bash
bash scripts/maintenance/start_api.sh
```

### Streamlit Dashboard

```bash
bash scripts/maintenance/start_streamlit.sh
```

## Documentation

| Document | Content |
|----------|---------|
| `docs/ARCHITECTURE.md` | Complete architecture deep-dive |
| `docs/DATASETS.md` | Dataset versions, statistics, format |
| `docs/OPTIMIZATION_REPORT.md` | Performance bottleneck analysis |
| `docs/RESOURCE_PROFILES.md` | Configs for 4 GB / 12 GB / 24+ GB GPUs |
| `docs/RESEARCH_PLAN.md` | Research roadmap and experiment results |
| `docs/TROUBLESHOOTING.md` | Common issues and solutions |
| `docs/EXPERIMENTS.md` | Experiment tracking |
| `docs/PROJECT_STRUCTURE.md` | Directory overview |
| `CHANGELOG.md` | Version history |
| `ROADMAP.md` | Development milestones |

## Hardware Requirements

| Tier | GPU | VRAM | Config File |
|------|-----|------|-------------|
| Small | GTX 1650 | 4 GB | `model_config_v3_li.yaml` (current) |
| Medium | RTX 3060 | 12 GB | hidden_dim=256, batch=32 |
| Large | RTX 4090 / A100 | 24-80 GB | hidden_dim=512, batch=64 |

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{scandium_labs_2024,
  title  = {Scandium Labs: AI-Driven Solid Electrolyte Discovery},
  author = {Scandium Labs},
  year   = {2024},
  doi    = {10.5281/zenodo.XXXXX},
  url    = {https://github.com/scandium-labs/scandium-labs}
}
```
