# Experiments

## Completed Experiments

### v1_817 — Initial Proof of Concept
- **Date**: 2026-04
- **Dataset**: 817 Li-containing structures from Materials Project
- **Model**: ALIGNN with 3 layers, hidden_dim=256
- **Training**: ScandiumTrainer with MSE loss, no physics constraints
- **Results**: Baseline established for Li conductivity prediction
- **Status**: Archived

### v2_10000 — Scaled Training
- **Date**: 2026-05
- **Dataset**: 10,000 Li-containing structures (random subsample)
- **Model**: ALIGNN + GraphTransformer + PINN loss
- **Config**: `configs/model_config_v2.yaml`
- **Key Changes**: Added Arrhenius physics loss, multi-task heads
- **Status**: Archived

### v2_10000_log_eah — Log-Scaled EaH
- **Date**: 2026-05
- **Dataset**: 10,000 Li structures with log-scaled energy_above_hull
- **Config**: `configs/phase3_config_log_eah.yaml`
- **Key Changes**: Log-transformed EaH target, two-stage EaH head
- **Status**: Archived

### v3_li_10000 — Family-Balanced Training
- **Date**: 2026-06
- **Dataset**: 10,000 Li≥5% structures (subsampled from 20,789), family-balanced
- **Config**: `configs/model_config_v3_li.yaml`
- **Model**: ScandiumPINNGNN with TwoStageEahHead
- **Key Changes**: Family-balanced split (no single family > 43%), LazyGraphDataset
- **Status**: Current active experiment

## Archived Experiments

The `archive/experiments/` directory contains older experiment outputs:
- Deprecated config files
- Early training logs
- Ablation study results
- Benchmark outputs from v1 and v2

## Planned Experiments

### Optuna Hyperparameter Search
- Search space: lr (1e-5 to 1e-3), hidden_dim (64-512), num_layers (1-6), dropout (0-0.5)
- Objective: validation MAE on log_ionic_conductivity
- Budget: 100 trials

### Architecture Ablation
1. Baseline: ALIGNN only (no Transformer)
2. ALIGNN + Transformer (current)
3. Transformer only (no ALIGNN)
4. Equivariant GNN (MACE-style)

### Loss Function Ablation
1. MSE only
2. PINNLoss (current)
3. Uncertainty-weighted MultiTaskLoss
4. GradNorm adaptive weighting

### Uncertainty Benchmark
1. MC Dropout (current)
2. Deep Ensemble (5 models)
3. Evidential regression
4. Concrete dropout
