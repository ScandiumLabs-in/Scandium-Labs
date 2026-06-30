# Scandium Labs Architecture

## High-Level Overview

Scandium Labs implements a Physics-Informed Neural Network (PINN) with an ALIGNN (Atomistic Line Graph Neural Network) backbone for multi-task prediction of solid electrolyte properties. The model jointly predicts ionic conductivity, activation energy, formation energy, energy above hull (EaH), and band gap from crystal structures, enforcing physical consistency through Arrhenius and thermodynamic constraints.

The architecture is structured as: crystal structure input → graph featurization → ALIGNN message passing → graph transformer → PINN constraints → task-specific heads → outputs. Uncertainty is quantified via Monte Carlo (MC) Dropout at inference time.

---

## Data Flow

```
pymatgen Structure
       │
       ▼
CrystalGraphBuilder / ALIGNNGraphBuilder (src/graphs/builder.py)
  ├─ Bessel/Gaussian RBF edge features (cutoff 8.0Å, max 16 neighbors)
  ├─ Atom features via get_atom_features() (92-dim)
  ├─ Global features via get_global_features() (16-dim)
  └─ Bond angles → Spherical Bessel RBF → line graph edge features
       │
       ▼
  (crystal_graph Data, line_graph Data)
       │
       ▼
FeatureEngineer.featurize() — pads/truncates atom features to 92-dim
       │
       ▼
SolidElectrolyteDataset / LazyGraphDataset
  └─ _attach_targets() — attaches y_{task} attributes to crystal_graph
       │
       ▼
collate_fn → torch_geometric Batch.from_data_list()
       │
       ▼
ScandiumPINNGNN (src/models/scandium_model.py)
  ├─ Atom encoder: Linear → LayerNorm → SiLU → Linear (92 → 256)
  ├─ Edge encoder: Linear → SiLU → Linear (64 → 128)
  ├─ Line-graph edge encoder: Linear (→ 64) [optional]
  ├─ ALIGNN layers (×N) — message passing on crystal + line graph
  ├─ GraphTransformer layers (×N) — multi-head self-attention
  ├─ PINNConstraintModule — gated physics constraints
  ├─ AttentionGlobalPool — soft-attention readout
  ├─ Global feature encoder + combiner (concatenates 16-dim global feats)
  └─ Task heads — per-property MLP heads (or TwoStageEahHead)
       │
       ▼
  predictions dict {task: tensor} [, uncertainties dict {task: tensor}]
```

---

## Model Architecture

### 1. ALIGNN Backbone

**File:** `src/models/gnn/alignn.py`

`ALIGNNLayer` performs alternating message passing on the line graph and the crystal graph:

1. **Line graph convolution** — `CrystalMPNN` updates edge features by passing messages over the line graph (edges as nodes, bond angles as edge attributes).
2. **Crystal graph convolution** — `CrystalMPNN` updates node (atom) features using the updated edge features.

`ScandiumPINNGNN` stacks `num_alignn_layers` (default 2–4) `ALIGNNLayer` modules.

#### CrystalMPNN

**File:** `src/models/gnn/layers.py:6`

A `torch_geometric.nn.MessagePassing` subclass with:
- **Message function:** `MLP([x_i || x_j || e_ij])` — concatenates source, target, and edge features through a 2-layer SiLU MLP.
- **Update function:** `MLP([x || ∑messages])` with residual connection + LayerNorm.
- Aggregation: sum.

### 2. Graph Transformer

**File:** `src/models/gnn/layers.py:76`

`GraphTransformerLayer` wraps `nn.MultiheadAttention` (batch-first) with pre-norm residual connections and a position-wise FFN (Linear → GELU → Dropout → Linear → Dropout). Applied as a global graph transformer over node embeddings (batch dimension = 1 per graph). Stacks of `num_transformer_layers` (default 1–4).

### 3. PINN Constraint Module

**File:** `src/models/gnn/layers.py:100`

`PINNConstraintModule` applies learnable physics-informed gating after the transformer layers:

- **Arrhenius gate:** `σ(Linear(node_feats))` — controls conductivity-related signal flow.
- **Thermodynamic gate:** `σ(MLP(node_feats))` — controls stability-related signal flow.
- Output: `LayerNorm(node_feats + node_feats × gate_arrhenius × gate_thermo)`

### 4. Attention Pooling

**File:** `src/models/gnn/layers.py:119`

`AttentionGlobalPool` computes per-node attention weights via a single-neuron gate MLP, applies softmax over each graph, then performs weighted sum pooling via `global_add_pool`.

### 5. Global Feature Combiner

16 global features (density, volume, symmetry, etc.) are encoded through `Linear(16 → 64) → SiLU`, concatenated with pooled graph features, and combined through `Linear(320 → 256) → LayerNorm → SiLU`.

### 6. Task Heads

**File:** `src/models/scandium_model.py:80`

For each task in `["log_ionic_conductivity", "formation_energy", "energy_above_hull", "activation_energy", "band_gap"]`:

- **Standard MLP head:** `Linear(hidden → hidden/2) → SiLU → Dropout → Linear(hidden/2 → hidden/4) → SiLU → Linear(hidden/4 → 1)`
- **TwoStageEahHead** (for `energy_above_hull` when `use_two_stage_eah=True`):

**File:** `src/models/heads/two_stage_eah.py:14`

Decomposes EaH prediction into:
- **Stage 1 — Stability classifier:** `MLP → sigmoid` outputs `p_unstable`.
- **Stage 2 — Magnitude regressor:** `MLP → Softplus` outputs EaH magnitude (positive).
- **Combined output:** `eah_pred = p_unstable × magnitude`.
- An auxiliary `uncertainty_head` predicts log-variance for heteroscedastic uncertainty.

### 7. Uncertainty Heads

Per-task MLP (`Linear(hidden → hidden/4) → SiLU → Linear(hidden/4 → 1)`) predicting log-variance for heteroscedastic uncertainty. Used during forward pass when `return_uncertainty=True`.

### 8. MC Dropout Inference

**File:** `src/models/scandium_model.py:201`

`predict_with_mc_dropout()` runs `num_mc_dropout_samples` (default 20) forward passes with `model.train()` (dropout active) under `torch.no_grad()`, returning per-task `{mean, std, samples}`.

---

## Loss Functions

**File:** `src/training/losses.py`

### PINNLoss

Multi-component physics-informed loss:

| Component | Formula | Description |
|-----------|---------|-------------|
| **Data fidelity** | `∑ w_t · MSE(pred_t, target_t)` | Per-task masked MSE (NaNs ignored) |
| **Arrhenius constraint** | `Var[log₁₀(σT) + Ea/(k_B T · ln 10)]` | Enforces Arrhenius relation between conductivity and activation energy |
| **Thermodynamic constraint** | `mean(ReLU(-EaH))` | Penalizes negative EaH predictions (thermodynamically impossible) |
| **Diffusion residual** | `𝔼[(∂c/∂t - D ∇²c)²]` | PINN PDE residual for Li-ion diffusion (requires `concentration_head`) |

The total loss is: `L_total = λ_data · L_data + λ_arrhenius · L_arrhenius + λ_thermo · L_thermo + λ_physics · L_diffusion`

### TwoStageEahLoss

**File:** `src/models/heads/two_stage_eah.py:73`

For the two-stage EaH head: `L = λ_bce · BCE(p_unstable, is_unstable) + λ_reg · MSE(eah_magnitude, eah_true)[unstable] + λ_stable · MSE(eah_pred, 0)[stable]`. Supports per-sample family weights to counteract composition shortcuts.

---

## Training Pipeline

**File:** `src/training/trainer.py`

### ScandiumTrainer

The `ScandiumTrainer` class orchestrates training:

1. **Initialization:** Loads YAML config, sets device (CUDA/CPU), initializes `GradScaler` for AMP, loads or creates `PropertyNormalizer`.

2. **`build_model()`:** Instantiates `ScandiumPINNGNN` from config, optionally loads pretrained ALIGNN encoder weights via `PretrainedEncoder`.

3. **`build_optimizer()`:** Configures `AdamW` with per-parameter-group weight decay via `get_param_groups()`.

4. **`build_loss()`:** Constructs `PINNLoss` with task weights and PINN hyperparameters from config.

5. **`train_epoch()`:**
   - Mixed precision forward pass (`torch.cuda.amp.autocast`).
   - Target normalization via `PropertyNormalizer.normalize()`.
   - Gradient scaling, unscaling, clipping (max norm 1.0).
   - Tracks per-task data losses and gradient norms.

6. **`validate()`:** Denormalizes predictions, computes per-task MAE.

7. **`train()`:**
   - Data loading via `load_data()`.
   - LR scheduling via `build_scheduler()` (cosine with restarts).
   - WandB logging per epoch.
   - Early stopping with configurable patience.
   - Checkpoint saving (best model + per-epoch).
   - Resume from checkpoint support.
   - Final test evaluation on best checkpoint.

### Key Training Features

- **Mixed Precision:** `torch.cuda.amp.GradScaler` + `autocast`.
- **Gradient Clipping:** Global norm clipping at 1.0.
- **Learning Rate Schedule:** Cosine annealing with warm restarts.
- **Early Stopping:** Patience-based (default 30–40 epochs).
- **Checkpointing:** Full model/optimizer/scheduler state saved.

---

## Uncertainty Quantification

- **Epistemic uncertainty:** MC Dropout with 20 forward passes at inference time (`predict_with_mc_dropout()`).
- **Aleatoric uncertainty:** Per-task heteroscedastic uncertainty heads predicting log-variance.
- Returned as `{task: {"mean": tensor, "std": tensor, "samples": tensor}}`.

---

## Chemical Family Handling

**File:** `src/chemistry/family_id.py`

Composition-based classification into 5 families:

| Family | Condition |
|--------|-----------|
| `pure_halide` | F, Cl, Br, I present, no O/S |
| `sulfohalide` | Halide + S present |
| `oxyhalide` | Halide + O present |
| `sulfide` | S present (no halide) |
| `phosphate` | O + P present (no halide) |
| `oxide` | O present (no P, no halide) |
| `other` | None of the above |

Also provides:
- `family_numeric()` — integer encoding (0–6).
- `has_lithium()` — checks Li presence in composition.

Family labels are used for dataset splitting (`composition_based_split` with `GroupShuffleSplit`) and for per-sample weighting in `TwoStageEahLoss`.

---

## Inference Pipeline

**File:** `src/inference/engine.py`

### InferenceEngine

- **`__init__`:** Loads model checkpoint, builds graph builder and feature engineer with config dimensions.
- **`predict_dataset()`:** Runs batched inference over a list of structures, with optional MC Dropout.
- **`recommend_materials()`:** Ranks materials by conductivity/stability criteria.
- **`screen_candidates()`:** Applies property thresholds to filter candidate solid electrolytes.
- **Resolve stability:** `resolve_stability()` post-processes EaH predictions.
- **Gating:** `gate_predictions()` filters predictions based on data audit status.

---

## Configuration

Config files (YAML) control model architecture, graph parameters, task definitions, PINN loss weights, and training hyperparameters:

- `configs/model_config_v2.yaml` — Full 5-task model with PINN constraints.
- `configs/model_config_v3_li.yaml` — 3-task Li-focused model with two-stage EaH head and GradNorm.
