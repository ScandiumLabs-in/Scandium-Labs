# Scandium Labs — Technical Documentation

## Table of Contents

- [Architecture Deep Dive](#architecture-deep-dive)
- [API Reference](#api-reference)
  - [InferenceEngine](#inferenceengine)
  - [ScandiumTrainer](#scandiumtrainer)
  - [PropertyNormalizer](#propertynormalizer)
  - [PINNLoss](#pinnloss)
  - [ScandiumPINNGNN](#scandiumpinngnn)
  - [DataCollectors](#datacollectors)
- [Training Guide](#training-guide)
- [Inference Guide](#inference-guide)
- [Benchmarking Guide](#benchmarking-guide)
- [Configuration Reference](#configuration-reference)
- [Data Pipeline](#data-pipeline)
- [Checkpoint Format](#checkpoint-format)
- [Uncertainty Quantification](#uncertainty-quantification)
- [OOD Detection](#ood-detection)
- [Multi-Task Optimization](#multi-task-optimization)
- [Physics-Informed Losses](#physics-informed-losses)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)

---

## Architecture Deep Dive

### ScandiumPINNGNN (`src/models/scandium_model.py`)

The model is a 5-headed multi-task architecture with shared representation layers and task-specific heads, combined with physics-informed constraints and uncertainty quantification.

```
Input: crystal_graph (PyG Data) + line_graph (PyG Data)
         │
         ▼
    ┌─────────────────────┐
    │    Atom Encoder     │  Linear(92 → hidden) → LayerNorm → SiLU → Linear(hidden → hidden)
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │   Edge Encoder      │  Linear(64 → hidden//2) → SiLU
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │  ALIGNN Block × N   │  [LineGraph MPNN → CrystalGraph MPNN] × N
    │                     │  Each layer updates node and edge features
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │ Graph Transformer ×M│  MultiheadAttention(4 heads, hidden) + FFN(hidden×4)
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │  PINN Constraint    │  Arrhenius gate: σ(hidden) → Sigmoid
    │                     │  Thermodynamic gate: SiLU → Linear → Sigmoid
    │                     │  Output = h + h × gate_A × gate_T
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │  Global Pool        │  Learnable attention: softmax(MLP(h)) · h
    └─────────┬───────────┘
              │
    ┌─────────────────────┐
    │  Global Features    │  Concat(pooled, global_emb) → LayerNorm → SiLU
    └─────────┬───────────┘
              │
    ┌────────────────────────────────────────────┐
    │  Task Heads × 5                            │
    │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───┐ │
    │  │  Ef  │ │  Eah │ │  BG  │ │  σ   │ │Ea │ │
    │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └─┬─┘ │
    │     │        │        │        │        │    │
    │  MLP(h→h/2→1) with SiLU + Dropout           │
    │  Uncertainty head: MLP(h→h/4→1) → log_var   │
    └──────────────────────────────────────────────┘
```

#### Forward Pass (`forward` method)

```python
def forward(self, crystal_graph, line_graph):
    # 1. Encode atom and edge features
    x = self.atom_encoder(crystal_graph.x)         # (N, hidden)
    edge_attr = self.edge_encoder(crystal_graph.edge_attr)

    # 2. ALIGNN message passing
    for alignn_layer in self.alignn_layers:
        x, edge_attr = alignn_layer(x, edge_attr, crystal_graph.edge_index, line_graph)

    # 3. Graph transformer
    for transformer in self.transformers:
        x = transformer(x, crystal_graph.batch)

    # 4. Physics-informed constraint
    x = self.pinn_module(x)

    # 5. Global pooling
    pooled = self.global_pool(x, crystal_graph.batch)

    # 6. Global feature conditioning
    global_feats = self.global_encoder(crystal_graph.global_features)
    out = self.combiner(torch.cat([pooled, global_feats], dim=-1))

    # 7. Task-specific heads
    predictions = {}
    for task, head in self.task_heads.items():
        predictions[task] = head(out)

    return predictions
```

#### MC Dropout Inference (`predict_with_mc_dropout` method)

```python
def predict_with_mc_dropout(self, crystal_graph, line_graph, n_samples=20):
    self.train()  # Enable dropout
    samples = {task: [] for task in self.tasks}

    with torch.no_grad():
        for _ in range(n_samples):
            preds = self.forward(crystal_graph, line_graph)
            for task, val in preds.items():
                samples[task].append(val)

    self.eval()
    results = {}
    for task in self.tasks:
        tensor_samples = torch.stack(samples[task], dim=0)
        results[task] = {
            'mean': tensor_samples.mean(dim=0),
            'std': tensor_samples.std(dim=0),
            'samples': tensor_samples,
        }
    return results
```

#### Task Head Architecture

```python
class TaskHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.mlp(x), self.uncertainty_head(x)
```

### ALIGNN Layer (`src/models/alignn.py`)

The ALIGNN (Atomistic Line Graph Neural Network) layer operates on two graphs:

1. **Line Graph** — Each edge in the crystal graph becomes a node in the line graph. Edges connect pairs of crystal-graph edges that share a node. This encodes bond angles.

2. **Crystal Graph** — Standard message-passing on the atomic structure.

#### Line Graph MPNN

```python
def line_graph_mpnn(line_node_feats, line_edge_index, line_edge_feats):
    # Message: MLP(2*line_node_dim + line_edge_dim → hidden)
    # Aggregate: sum over neighbors
    # Update: MLP(line_node_dim + hidden → line_node_dim) + residual
    return updated_line_node_feats
```

#### Crystal Graph MPNN

```python
def crystal_mpnn(node_feats, edge_index, edge_feats):
    # Message: MLP(2*node_dim + edge_dim → hidden)
    # Aggregate: sum over neighbors
    # Update: MLP(node_dim + hidden → node_dim) + LayerNorm + residual
    return updated_node_feats
```

### Graph Transformer Layer (`src/models/transformer.py`)

Standard transformer with pre-norm architecture:

```python
class GraphTransformerLayer(nn.Module):
    def forward(self, x, batch):
        # Pre-norm
        x_norm = self.norm1(x)
        # Multi-head self-attention (with batch mask)
        attn_out = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        # FFN with pre-norm
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x
```

### PINN Constraint Module (`src/models/pinn.py`)

Enforces physical constraints through learned gating:

```python
class PINNConstraintModule(nn.Module):
    def forward(self, x):
        arrhenius_gate = torch.sigmoid(self.arrhenius_net(x))
        thermo_gate = torch.sigmoid(self.thermo_net(x))
        # Modulate features
        x = x + x * arrhenius_gate * thermo_gate
        return x
```

### Global Pool (`src/models/pinn.py`)

Learnable attention-based pooling:

```python
class AttentionGlobalPool(nn.Module):
    def forward(self, x, batch):
        # Attention scores per node
        scores = self.attn_mlp(x)  # (N, 1)
        # Softmax per graph
        scores = scatter_softmax(scores, batch, dim=0)
        # Weighted sum per graph
        pooled = scatter_sum(x * scores, batch, dim=0)
        return pooled
```

---

## API Reference

### InferenceEngine

The main interface for making predictions.

```python
from src.inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="checkpoints/best_model.pt",
    device="cuda",           # or "cpu"
    use_mc_dropout=True,     # Enable MC Dropout for uncertainty
    mc_samples=20,           # Number of dropout forward passes
    log_eah=False,           # Set True for log-Eah checkpoint
)
```

#### Methods

##### `predict_single(structure, temperature=300.0) -> dict`

Predict properties for one crystal structure.

```python
result = engine.predict_single(structure)

# Returns:
{
    'formation_energy': {
        'value': -1.234,          # eV/atom
        'uncertainty': 0.045,     # eV/atom (std from MC dropout)
    },
    'energy_above_hull': {
        'value': 0.112,           # eV/atom
        'uncertainty': 0.023,     # eV/atom
    },
    'band_gap': {
        'value': 2.104,           # eV
        'uncertainty': 0.312,     # eV
    },
    'log_ionic_conductivity': {
        'value': None,            # 0% training data
        'status': 'insufficient training data',
    },
    'ionic_conductivity': {
        'value': None,
        'status': 'insufficient training data',
        'unit': 'S/cm',
    },
    'stability_check': {
        'suspicious': False,
        'reason': None,
    },
    'recommendation': 'REJECT',
    'recommendation_detail': 'Thermodynamically unstable — E above hull 0.112 eV/atom ...',
    'recommendation_confidence': 'high',
    'recommended_actions': ['...'],
    'ood': {
        'is_ood': False,
        'score': 0.23,
    },
}
```

##### `predict_batch(structures, batch_size=32) -> list[dict]`

Batch prediction for multiple structures.

##### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | str | required | Path to checkpoint .pt file |
| `device` | str | "cuda" | "cuda" or "cpu" |
| `use_mc_dropout` | bool | True | Enable uncertainty estimation |
| `mc_samples` | int | 20 | Number of dropout passes |
| `log_eah` | bool | False | Auto-detected from checkpoint config |

#### Normalizer Handling

The engine auto-loads the normalizer from `normalizer.json` adjacent to the checkpoint. When `normalize_targets=True` in the checkpoint config, predictions are automatically denormalized:

```python
# InferenceEngine._load_model()
self.normalize_targets = checkpoint['config']['training'].get('normalize_targets', False)

# In predict_single():
if self.normalize_targets and self.normalizer:
    stat = self.normalizer.stats[task]
    pred['value'] = pred['value'] * stat['std'] + stat['mean']
```

### ScandiumTrainer

```python
from src.training.trainer import ScandiumTrainer

trainer = ScandiumTrainer(
    config_path="config/model_config_v2.yaml",
    data_dir="datasets/v2_10000",
)

model, test_metrics = trainer.train(resume_from="checkpoints/epoch_45.pt")
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_path` | str | required | Path to model config YAML |
| `data_dir` | str | "data/processed" | Dataset directory with prebuilt_graphs.pt |
| `resume_from` | str | None | Checkpoint path to resume training |

#### Training Loop

```python
def train(self, resume_from=None):
    model = self.build_model()
    optimizer = self.build_optimizer(model)
    loss_fn = self.build_loss()
    train_loader, val_loader, test_loader = self.load_data()

    for epoch in range(start_epoch, max_epochs):
        # Training epoch: normalize targets → forward → loss → backward → clip → step
        train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, loss_fn)

        # Validation: denormalize predictions → compute MAE
        val_metrics = self.validate(model, val_loader, loss_fn)

        # Save checkpoint + early stopping
        self.save_checkpoint(model, optimizer, epoch, val_metrics, train_metrics, is_best)

    # Test evaluation using best checkpoint
    model.load_state_dict(torch.load("checkpoints/best_model.pt")['model'])
    test_metrics = self.validate(model, test_loader, loss_fn)
    return model, test_metrics
```

### PropertyNormalizer

```python
from src.data.cleaner import PropertyNormalizer

# Load existing normalizer
normalizer = PropertyNormalizer.load("datasets/v2_10000/normalizer.json")

# Normalize targets (z-score)
raw_targets = {'formation_energy': torch.tensor([-0.5, -1.2, ...])}
normalized = normalizer.normalize(raw_targets)
# Returns: {'formation_energy': tensor([...])}  # z-scores

# Denormalize predictions back to physical units
raw_preds = normalizer.denormalize(normalized, 'formation_energy')
# Returns tensor in eV/atom

# Access statistics
stats = normalizer.stats  # dict of {task: {mean, std, min, max}}
```

#### Normalizer JSON Format

```json
{
  "formation_energy": {
    "mean": -0.9856,
    "std": 0.7838,
    "min": -3.9382,
    "max": 4.7116
  },
  "energy_above_hull": {
    "mean": 0.1840,
    "std": 0.3556,
    "min": 0.0,
    "max": 4.7118
  },
  "band_gap": {
    "mean": 1.3032,
    "std": 1.3211,
    "min": 0.0,
    "max": 5.5854
  }
}
```

### PINNLoss

```python
from src.training.losses import PINNLoss

loss_fn = PINNLoss(
    task_weights={
        'formation_energy': 1.0,
        'energy_above_hull': 0.8,
        'band_gap': 0.4,
        'log_ionic_conductivity': 1.0,
        'activation_energy': 0.6,
    },
    log_eah=False,       # Set True for log-transformed Eah
    lambda_data=1.0,
    lambda_physics=0.1,
    lambda_arrhenius=0.05,
    lambda_thermodynamic=0.05,
)

losses = loss_fn(predictions, targets, crystal_graph, model)
# Returns: {
#   'total': tensor,
#   'data': tensor,
#   'arrhenius': tensor,
#   'thermodynamic': tensor,
#   'physics': tensor,
# }
```

#### Loss Components

**Data Loss** (weighted MSE):
```python
L_data = Σ task_weight * MSE(pred[task], target[task])
# Only tasks with non-NaN targets contribute
```

**Arrhenius Loss** (enforces σ-T-Ea consistency):
```python
L_arrhenius = Var( log10(sigma * T) + Ea / (kB * T * ln10) )
# Minimized when predicted conductivity and activation energy
# satisfy the Arrhenius equation
```

**Thermodynamic Loss** (penalizes negative Eah):
```python
L_thermo = mean(ReLU(-Eah_pred))
# In log-eah mode: Eah = exp(pred) - eps, then ReLU(-Eah)
```

**Physics Loss** (diffusion PDE residual):
```python
L_physics = mean(|dc/dt - D * ∇²c|)
# Requires concentration head (disabled by default)
```

### ScandiumPINNGNN

```python
from src.models.scandium_model import ScandiumPINNGNN

model = ScandiumPINNGNN(
    hidden_dim=128,
    num_alignn_layers=2,
    num_transformer_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    mc_dropout_samples=20,
    tasks=['log_ionic_conductivity', 'formation_energy',
           'energy_above_hull', 'activation_energy', 'band_gap'],
)

# Forward pass
predictions = model(crystal_graph, line_graph)
# Returns dict: {'formation_energy': tensor(B, 1), ...}

# MC Dropout inference
results = model.predict_with_mc_dropout(crystal_graph, line_graph, n_samples=20)
# Returns dict with 'mean', 'std', 'samples' per task
```

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | 128 | Hidden dimension for all layers |
| `num_alignn_layers` | 2 | Number of ALIGNN message-passing blocks |
| `num_transformer_layers` | 1 | Number of transformer attention blocks |
| `num_attention_heads` | 4 | Multi-head attention heads |
| `dropout` | 0.1 | Dropout rate throughout |
| `mc_dropout_samples` | 20 | MC dropout forward passes |
| `tasks` | [...] | List of task names (5) |

### DataCollectors

Five collectors for different data sources:

```python
from src.data.collectors import (
    MaterialsProjectCollector,
    JARVISCollector,
    OQMDCollector,
    AFLOWCollector,
    NOMADCollector,
)

# Materials Project (requires API key)
mp = MaterialsProjectCollector(api_key="your_key")
data = mp.collect(elements=["Li"], max_structures=10000)

# JARVIS-DFT
jarvis = JARVISCollector()
data = jarvis.collect(dataset="dft_3d")

# OQMD
oqmd = OQMDCollector()
data = oqmd.collect(elements=["Li"], max_structures=50000)

# AFLOW
aflow = AFLOWCollector()
data = aflow.collect(elements=["Li"], max_structures=10000)

# NOMAD
nomad = NOMADCollector()
data = nomad.collect(elements=["Li"], max_entries=10000)
```

---

## Training Guide

### 1. Prepare Data

```bash
# Build dataset from Materials Project
python scripts/build_dataset.py \
    --name "v2_10000" \
    --elements Li Na K Rb Cs Mg Ca Sr Ba \
    --max-structures 10000 \
    --clean \
    --split-method stratified_group_kfold \
    --cache-graphs
```

### 2. Train Model

```bash
# Basic training
python scripts/train.py \
    --config config/model_config_v2.yaml \
    --data_dir datasets/v2_10000

# Resume from checkpoint
python scripts/train.py \
    --config config/model_config_v2.yaml \
    --data_dir datasets/v2_10000 \
    --checkpoint checkpoints/epoch_45.pt

# Multi-GPU
python scripts/train.py \
    --config config/model_config_v2.yaml \
    --data_dir datasets/v2_10000 \
    --gpus 4
```

### 3. Monitor Training

Training logs are printed per epoch:

```
Epoch  45 | [log_ionic_conductivity: 0.0000 | formation_energy: 0.4018 | energy_above_hull: 0.6711 | activation_energy: 0.0000 | band_gap: 0.5693] | [g_log_ionic_conductivity: 0.0000 | g_formation_energy: 0.6658 | g_energy_above_hull: 0.4391 | g_activation_energy: 0.0005 | g_band_gap: 0.2827] | val [formation_energy: 0.3678 | energy_above_hull: 0.1804 | band_gap: 0.8375] ★
```

Legend:
- Brackets before `|`: per-task training losses
- `g_*`: per-task gradient norms (diagnose multi-task balance)
- `val [...]`: validation MAE per task
- `★`: new best total validation loss

#### Interpreting Gradient Norms

Healthy gradient ratio: Ef : Eah : BG ≈ 1.5 : 1 : 0.6 (with normalization)

| Pattern | Diagnosis | Fix |
|---|---|---|
| Ef gradients >> all others | Ef dominates multi-task | Adjust task weights or normalization |
| Eah gradients near zero | Eah head saturated | Check learning rate, head initialization |
| Spikes > 2× running average | Gradient explosion | Increase gradient_clip in config |
| All gradients → 0 | Vanishing gradients | Reduce dropout, check residual connections |

### 4. Evaluate

```bash
# Run all benchmarks
python scripts/benchmark_suite.py
python scripts/compare_benchmarks.py \
    --checkpoints experiments/*/checkpoint.pt \
    --labels "v1,corrected-split,corrected-split+norm"

# 5-fold cross-validation
python scripts/cross_validate.py \
    --config config/model_config_v2.yaml \
    --data_dir datasets/v2_10000
```

---

## Inference Guide

### Quick Start

```python
from src.inference.engine import InferenceEngine
from pymatgen.core import Structure, Lattice

engine = InferenceEngine("checkpoints/best_model.pt", device="cpu")

# Structure from lattice parameters
structure = Structure(
    Lattice.cubic(4.03),
    ["Li", "F"],
    [[0, 0, 0], [0.5, 0.5, 0.5]]
)

result = engine.predict_single(structure)
print(f"Ef:  {result['formation_energy']['value']:.3f} ± {result['formation_energy']['uncertainty']:.3f} eV/atom")
print(f"Eah: {result['energy_above_hull']['value']:.3f} ± {result['energy_above_hull']['uncertainty']:.3f} eV/atom")
print(f"BG:  {result['band_gap']['value']:.3f} ± {result['band_gap']['uncertainty']:.3f} eV")
print(f"Recommendation: {result['recommendation']}")
```

### Batch Screening

```python
structures = [
    Structure(Lattice.cubic(a), ["Li", "F"], [[0,0,0], [0.5,0.5,0.5]])
    for a in [4.0, 4.1, 4.2]
]

results = engine.predict_batch(structures, batch_size=32)

for i, result in enumerate(results):
    eah = result['energy_above_hull']['value']
    ef = result['formation_energy']['value']
    print(f"Structure {i}: Ef={ef:.3f}, Eah={eah:.3f}, {result['recommendation']}")
```

### From CIF Files

```python
from pymatgen.core import Structure

structure = Structure.from_file("path/to/structure.cif")
result = engine.predict_single(structure)
```

### Candidate Screening

```bash
python scripts/screen_candidates.py \
    --input candidates.json \
    --output results.json \
    --checkpoint checkpoints/best_model.pt

# candidates.json format:
{
  "candidates": [
    {
      "species": ["Li", "F"],
      "coords": [[0,0,0], [0.5,0.5,0.5]],
      "lattice": [4.03, 4.03, 4.03, 90, 90, 90]
    }
  ]
}
```

### Understanding Recommendations

| Recommendation | Criteria | Meaning |
|---|---|---|
| `HIGH PRIORITY` | σ > 1e-3 S/cm AND Eah < 0.025 eV/atom | Excellent candidate — proceed to validation |
| `MEDIUM PRIORITY` | σ > 1e-4 S/cm AND Eah < 0.05 eV/atom | Promising — DFT verification recommended |
| `LOW PRIORITY` | σ < 1e-4 OR Eah > 0.05 | Marginal — screen alternatives |
| `REJECT` | Eah - σ_unc > 0.10 eV OR σ < 1e-6 S/cm | Unsuitable |
| `UNCERTAIN` | OOD detected OR stability heads disagree | Needs manual review |

---

## Benchmarking Guide

### 54-Material Synthetic Benchmark

```bash
python scripts/benchmark_suite.py
```

Evaluates on 54 crystal structures spanning 5 families:
- **Halides** (23): LiF, NaCl, CsCl, CaF₂, etc.
- **Oxides** (21): MgO, SrTiO₃, CeO₂, LiCoO₂, etc.
- **Sulfides** (3): Li₂S, Na₂S
- **Semiconductors** (7): Si, GaAs, ZnS, CdTe, etc.

Output: `benchmark_v001_<hash>.json`

### 13-Material Expert Benchmark

```bash
python scripts/compare_benchmarks.py \
    --checkpoints experiments/*/checkpoint.pt \
    --labels "v1,corrected-split,corrected-split+norm"
```

Evaluates on 13 literature-known materials with expected experimental values:
Li₆PS₅Cl, Li₂O, LiF, NaCl, MgO, LiCoO₂, LiFePO₄, Li₃PO₄, Li₂TiO₃, Li₂CO₃, SiO₂, Al₂O₃, Li₂S

Output: Per-material predictions + aggregate metrics (MAE, R², stability accuracy)

### 5-Fold Cross-Validation

```bash
python scripts/cross_validate.py \
    --config config/model_config_v2.yaml \
    --data_dir datasets/v2_10000
```

Chemistry-stratified 5-fold CV. Output: per-fold JSON + aggregated summary.

### Cross-Model Comparison

```python
from scripts.compare_benchmarks import evaluate_checkpoint, compute_summary

results = []
for ckpt_path, label in [("path1.pt", "model A"), ("path2.pt", "model B")]:
    r = evaluate_checkpoint(ckpt_path, label)
    results.append(r)

# Compare
from scripts.compare_benchmarks import print_comparison
print_comparison(results)
```

---

## Configuration Reference

### Model Config (`config/model_config_v2.yaml`)

```yaml
model:
  name: "ScandiumPINNGNN-v2"       # Model identifier
  hidden_dim: 128                   # Hidden dimension for all layers
  num_alignn_layers: 2              # ALIGNN message-passing blocks
  num_transformer_layers: 1         # Transformer attention blocks
  num_attention_heads: 4            # Multi-head attention heads
  dropout: 0.1                      # Dropout rate (also used for MC inference)
  mc_dropout_samples: 20            # MC dropout forward passes
  use_pretrained_alignn: false      # Load pretrained ALIGNN weights
  pretrained_checkpoint: null       # Path to pretrained checkpoint

graph:
  cutoff: 8.0                       # Neighbor cutoff in Ångströms
  max_neighbors: 16                 # Max neighbors per atom (v2)
  rbf_type: "bessel"                # Radial basis function type
  num_rbf: 64                       # Number of RBF basis functions
  num_sbf: 32                       # Number of spherical Bessel functions

tasks:
  - name: "log_ionic_conductivity"
    scale: "log10"
    weight: 1.0
    unit: "log(S/cm)"
  - name: "formation_energy"
    scale: "linear"
    weight: 1.0
    unit: "eV/atom"
  - name: "energy_above_hull"
    scale: "linear"
    weight: 0.8
    unit: "eV/atom"
  - name: "activation_energy"
    scale: "linear"
    weight: 0.6
    unit: "eV"
  - name: "band_gap"
    scale: "linear"
    weight: 0.4
    unit: "eV"

pinn:
  lambda_data: 1.0                  # Data loss weight
  lambda_physics: 0.1               # Physics (PDE) loss weight
  lambda_arrhenius: 0.05            # Arrhenius constraint weight
  lambda_thermodynamic: 0.05        # Thermodynamic constraint weight

training:
  batch_size: 8                     # Samples per batch (VRAM-limited)
  learning_rate: 0.001              # Peak learning rate
  warmup_steps: 1000                # LR warmup steps
  max_epochs: 100                   # Maximum training epochs
  patience: 30                      # Early stopping patience
  scheduler: "cosine_with_restarts"  # LR schedule
  optimizer: "AdamW"                # Optimizer
  weight_decay: 0.00001             # L2 regularization
  gradient_clip: 1.0                # Max gradient norm
  mixed_precision: true             # FP16 training
  normalize_targets: true           # Z-score normalize targets (added dynamically)
```

### Deployment Config (`config/deploy_config.yaml`)

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["*"]

database:
  url: "postgresql://user:pass@postgres:5432/scandium"
  pool_size: 10

inference:
  model_path: "/models/best_model.pt"
  device: "cuda"
  mc_samples: 20
  use_mc_dropout: true

storage:
  type: "s3"
  bucket: "scandium-results"
  region: "us-east-1"

redis:
  url: "redis://redis:6379/0"

auth:
  jwt_secret: "${JWT_SECRET}"
  jwt_algorithm: "HS256"
  token_expiry_hours: 24
```

---

## Data Pipeline

### Build Dataset

```bash
python scripts/build_dataset.py \
    --name "v2_10000" \
    --elements Li Na K Rb Cs Mg Ca Sr Ba \
    --max-structures 10000 \
    --clean \
    --split-method stratified_group_kfold \
    --cache-graphs
```

### Pipeline Steps

```
1. Download
   ├── Materials Project (API key required)
   ├── JARVIS-DFT (figshare)
   ├── OQMD (REST API)
   ├── AFLOW (AFLUX API)
   └── NOMAD (NOMAD Lab API)
       │
       ▼
2. Extract Targets
   ├── formation_energy (eV/atom)
   ├── energy_above_hull (eV/atom)
   ├── band_gap (eV)
   └── (log_ionic_conductivity, activation_energy — NaN)
       │
       ▼
3. Clean
   ├── Remove NaN formation_energy
   ├── Filter: 2 ≤ n_atoms ≤ 200
   ├── Filter: -10 ≤ Ef ≤ 5
   ├── Filter: Eah ≥ 0
   └── Deduplicate (StructureMatcher: ltol=0.2, stol=0.3, angle_tol=5°)
       │
       ▼
4. Split
   ├── Method: StratifiedGroupKFold
   ├── Groups: formula-based
   ├── Strata: Eah quantile bins
   └── Ratio: 80/10/10 (train/val/test)
       │
       ▼
5. Normalize
   ├── Fit z-score statistics per task
   └── Save normalizer.json
       │
       ▼
6. Build Graphs
   ├── Neighbors: up to 16 within 8.0 Å
   ├── Edge features: Bessel RBF (64 dims)
   ├── Line graph: bond angles (SBF, 32 dims)
   └── Save prebuilt_graphs.pt
       │
       ▼
7. Output
   ├── dataset_cache.pt
   ├── prebuilt_graphs.pt
   ├── split_indices.pt
   ├── normalizer.json
   └── metadata.json
```

### Dataset Cache Format

```python
# dataset_cache.pt
{
    'structures': [pymatgen.Structure, ...],  # List of structures
    'targets': {
        'formation_energy': torch.Tensor(N),       # eV/atom
        'energy_above_hull': torch.Tensor(N),       # eV/atom
        'band_gap': torch.Tensor(N),               # eV
        'log_ionic_conductivity': torch.Tensor(N), # NaN-filled
        'activation_energy': torch.Tensor(N),       # NaN-filled
    },
    'formula_groups': {formula: [indices]},
    'metadata': {
        'n_structures': int,
        'n_elements': int,
        'element_set': list,
        'creation_date': str,
    },
}

# prebuilt_graphs.pt
[
    {
        'x': torch.Tensor(n_atoms, 92),           # Atom features
        'edge_index': torch.Tensor(2, n_edges),    # Edge adjacency
        'edge_attr': torch.Tensor(n_edges, 64),    # Edge features (Bessel RBF)
        'y_formation_energy': torch.Tensor(1,),   # Target value
        'y_energy_above_hull': torch.Tensor(1,),
        'y_band_gap': torch.Tensor(1,),
        'global_features': torch.Tensor(16,),     # Global structure features
        'num_nodes': int,
        'formula': str,
    },
    ...
]
```

### Log-Transform Dataset

For the log-Eah experiment:

```bash
python scripts/prepare_log_transform.py \
    --input datasets/v2_10000 \
    --output datasets/v2_10000_log_eah \
    --eps 0.001
```

This creates a copy of the dataset where:
```python
y_energy_above_hull = log(original_Eah + 1e-3)
```

The normalizer gains an additional entry:
```json
{
  "energy_above_hull_log": {
    "mean": -2.0795,
    "std": 1.0765,
    "min": -6.9078,
    "max": 1.5504,
    "eps": 0.001
  }
}
```

---

## Checkpoint Format

Each `.pt` checkpoint file is a dictionary:

```python
checkpoint = {
    'epoch': 45,                              # Epoch number
    'model': OrderedDict,                      # Model state_dict
    'optimizer': dict,                        # Optimizer state
    'scheduler': dict,                        # LR scheduler state
    'metrics': {                              # Validation metrics at this epoch
        'formation_energy_mae': 0.3678,
        'energy_above_hull_mae': 0.1804,
        'band_gap_mae': 0.8375,
    },
    'train_metrics': {                        # Training metrics at this epoch
        'data': 1.175,
        'total': 1.175,
        'task_data': {
            'formation_energy': 0.4018,
            'energy_above_hull': 0.6711,
            'band_gap': 0.5693,
        },
        'grad_norms': {
            'formation_energy': 0.6658,
            'energy_above_hull': 0.4391,
            'band_gap': 0.2827,
        },
    },
    'config': {                               # Full training config (YAML contents)
        'model': {...},
        'training': {...},
        'tasks': [...],
        'pinn': {...},
        'graph': {...},
    },
}
```

### Per-Experiment Directory

```
experiments/<name>/
├── checkpoints/              # All epoch checkpoints
│   ├── epoch_0.pt → epoch_99.pt
│   ├── best_model.pt         # Best validation checkpoint
│   └── normalizer.json       # Co-located normalizer
├── checkpoint.pt             # Copy of best_model.pt (for compare_benchmarks.py)
├── config.yaml               # Copy of model config used for this run
├── train.log                 # Training stdout/stderr
├── benchmark/                # Benchmark outputs
│   ├── per_material_predictions.csv
│   ├── benchmark_results.csv
│   ├── metrics_summary.json
│   └── scatter_plots.png
├── dataset_version.txt       # Dataset ID + hash
├── git_commit.txt            # Git commit hash
└── start_time.txt            # Training start timestamp
```

---

## Uncertainty Quantification

### MC Dropout

The model uses Monte Carlo Dropout at inference: dropout layers remain active during evaluation, generating multiple predictions whose variance quantifies epistemic uncertainty.

```python
# During training: dropout is active
# During inference:
model.train()  # Manually enable dropout
samples = []
for _ in range(20):
    with torch.no_grad():
        pred = model(graph)
        samples.append(pred)

mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)  # Epistemic uncertainty
```

### For Log-Eah Models

When the model is trained on log-transformed Eah:
```python
# Each MC sample is in log-space
log_samples = res['samples']  # shape: (n_samples,)

# Transform each sample to physical space
eah_samples = np.maximum(np.exp(log_samples) - EPS, 0.0)

# Aggregate in physical space (not log-space)
mean = np.mean(eah_samples)
std = np.std(eah_samples)
```

### Uncertainty Interpretation

| Uncertainty (eV/atom) | Reliability | Use Case |
|---|---|---|
| < 0.02 | High | Confident prediction |
| 0.02 — 0.05 | Moderate | Reasonable for screening |
| 0.05 — 0.10 | Low | Needs verification |
| > 0.10 | Unreliable | OOD candidate likely |

### Calibration

Currently the uncertainty estimates are **not calibrated** (scale is relative, not absolute). For calibrated uncertainty, ECE (Expected Calibration Error) analysis should be performed against held-out test data.

---

## OOD Detection

The OOD detector uses an **Isolation Forest** trained on the model's learned embeddings:

```python
from src.evaluation.ood import OODDetector

detector = OODDetector(contamination=0.1)
# Fit on training set embeddings
detector.fit(training_embeddings)

# Score new candidates
result = detector.score(test_embedding)
# Returns: {'is_ood': bool, 'score': float, 'threshold': float}
```

When OOD is detected:
- Prediction `ood.is_ood = True`
- Recommendation becomes `UNCERTAIN`
- Output includes warning for human review

---

## Multi-Task Optimization

### Task Weighting

The default weights balance the tasks by importance:
- **Formation Energy**: 1.0 (primary stability indicator)
- **Energy Above Hull**: 0.8 (secondary stability indicator)
- **Ionic Conductivity**: 1.0 (primary performance metric — no training data)
- **Activation Energy**: 0.6 (supports conductivity — no training data)
- **Band Gap**: 0.4 (useful but secondary)

### Target Normalization

Z-score normalization balances gradient magnitudes across tasks with different units and scales:

| Task | Raw Scale | Normalized Scale | Impact |
|---|---|---|---|
| Formation Energy | [-4, +5] eV/atom | N(0,1) | Gradients balanced |
| Energy Above Hull | [0, 5] eV/atom | N(0,1) | Stronger signal |
| Band Gap | [0, 6] eV | N(0,1) | Regularized |

```python
# Before normalization: Ef dominates gradients (larger raw values)
# After normalization: gradient ratio Ef:Eah:BG ≈ 1.5:1:0.6
```

### Per-Task Gradient Monitoring

Training logs show per-task gradient norms. A healthy pattern:
```
g_Ef=0.67, g_Eah=0.44, g_BG=0.28
```
This gives Ef roughly 1.5× the gradient signal of Eah and 2.4× BG, reflecting real differences in learning difficulty.

---

## Physics-Informed Losses

### Thermodynamic Constraint

```python
L_thermo = λ_thermo · mean(ReLU(-Eah_pred))
```

- Penalizes predictions where Eah < 0 (physically impossible)
- In log-Eah mode, Eah is recovered via `exp(pred) - eps` before computing ReLU
- Forces model to respect Eah ≥ 0 without clipping

### Arrhenius Constraint

```python
L_arrhenius = λ_arrhenius · Var(log₁₀(σᵢ·T) + Eaᵢ / (k_B · T · ln(10)))
```

Where `σᵢ` and `Eaᵢ` are predicted ionic conductivity and activation energy for structure `i`.
- Enforces the Arrhenius equation across predicted conductivity and activation energy
- Only active when both σ and Ea heads have non-NaN targets (currently 0% label coverage)

### Diffusion PDE Residual

```python
L_physics = λ_physics · mean(|∂c/∂t - D · ∇²c|)
```

- Optional physics constraint on concentration dynamics
- Requires model to have a concentration prediction head (disabled by default)
- Uses automatic differentiation (requires gradient-enabled tensors)

---

## Reproducibility

### Experiment Tracking

Each experiment directory records:
- `config.yaml`: Exact configuration used
- `git_commit.txt`: Git commit hash
- `dataset_version.txt`: Dataset identifier
- `start_time.txt`: Training timestamp
- `train.log`: Full training output

### Random Seeds

The trainer sets seeds for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Checkpoint Self-Containment

Each checkpoint contains:
- Model weights (state_dict)
- Optimizer state (for resume)
- Config (full YAML copy)
- Normalizer co-located in experiment dir

### Common Pitfalls

**Split mismatch**: The split indices must match the dataset. Check `split_indices.pt` hash against dataset metadata.

```bash
# Verify split hash
python -c "
import torch, hashlib
data = torch.load('datasets/v2_10000/dataset_cache.pt', weights_only=False)
print(hashlib.md5(str(data['targets']['energy_above_hull'].numpy()).encode()).hexdigest()[:12])
"
```

---

## Troubleshooting

### Hanging Training

If training hangs (GPU utilization drops to 0%, no output for >10 min):

1. Check CUDA memory:
```bash
nvidia-smi
# Look for >90% GPU memory usage
```

2. Kill and resume:
```bash
kill <PID>
python scripts/train.py --checkpoint checkpoints/epoch_XX.pt --data_dir datasets/v2_10000
```

3. If hanging repeats, reduce batch size in config.

### Negative Eah Predictions

Two negative predictions observed in the normalized model. Causes:
- Normalization shifts the output distribution, pushing tail into negative
- Thermodynamic loss (ReLU) may not fully constrain in all cases

Fix options:
```python
# During inference:
pred['value'] = max(pred['value'], 0.0)

# Or increase lambda_thermodynamic in config:
pinn.lambda_thermodynamic: 0.1  # was 0.05
```

### Out of Memory (OOM)

The GTX 1650 4GB limits training to batch size 8 with 3,635 structures:

| Batch Size | Estimated VRAM | Works |
|---|---|---|
| 4 | ~2.0 GB | ✅ |
| 8 | ~3.7 GB | ✅ (borderline) |
| 16 | ~5.5 GB | ❌ (OOM) |

Solutions:
- Reduce `max_neighbors` (16 → 8)
- Reduce `hidden_dim` (128 → 64)
- Enable gradient accumulation (effective batch size = batch × accumulation)
- Use CPU for inference on large batches

### Missing Labels

`log_ionic_conductivity` and `activation_energy` have 0% coverage. The model returns `None` with status `'insufficient training data'`.

To add conductivity data:
1. Collect experimental or DFT-NEB conductivity values
2. Add to dataset in `targets` dict
3. Rebuild dataset and retrain

### GPU/CPU Device Issues

```bash
# Force CPU inference
python scripts/benchmark_suite.py
# (engine automatically uses CPU if CUDA unavailable)

# Check available device
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### Benchmark Failures

If `compare_benchmarks.py` shows `FAIL` for a material:
1. Check if the CIF file exists: `python -c "from pymatgen.core import Structure; s = Structure.from_file('test cif/Li6PS5Cl.cif')"`
2. If missing, structure generation may need updating in the script
3. Check that `CIF_DIR` path exists in `compare_benchmarks.py`
