# Research Plan

## Core Finding
- **Ef**: Architecture-limited (R² 0.46, 4.4× data → no improvement)
- **BG**: Data-limited (MAE 1.19→0.51, still improving with data)
- **Eah**: Target/architecture problem (R² -1.69)

## Phase Order

### 1. Diagnose Eah Failure (highest priority)
- Distribution, % zeros, outliers, missing values
- Normalization check
- Train/test distribution
- Histogram + log-scale histogram
- Pearson/Spearman correlation: Ef, BG, density, volume, N_atoms vs Eah
- If mostly zeros + long tail → MAE is misleading, R² collapses

### 2. Error Analysis
- Predicted vs True plot
- Residual plot, error histogram
- Worst 100 predictions
- Error vs: BG, atom count, chemistry, element family
- Answer: oxides? sulfides? halides? large cells? rare elements?

### 3. Ablation Study (quantify what matters)
| Experiment | Purpose |
|---|---|
| Remove attention | Is attention helping? |
| Remove uncertainty head | Does uncertainty improve learning? |
| Reduce hidden size (64) | Capacity test |
| Increase hidden size (256) | Underfitting vs overfitting |
| Single-task model | Multi-task interference |
| Remove PINN losses | PINN value |

### 4. Architecture Improvements
- **Option A** (recommended): Replace encoder with ALIGNN pretrained / Matformer / CrystalFormer
- **Option B**: Pretrain on MP-20, then fine-tune
- **Option C**: Deeper (2→6→10 layers) or wider (128→256→512)

### 5. Scaling Study
- Runs at 500, 1000, 2000, 3635
- Plot MAE vs dataset size, R² vs dataset size

### 6. Uncertainty Calibration
- ECE, NLL, coverage@95%, reliability diagram

### 7. Explainability
- Attention visualization, node importance, integrated gradients, GNNExplainer

### 8. Benchmark Against Published Models
- CGCNN, MEGNet, ALIGNN, Matformer, CrystalFormer on same splits

## Experiment Results Summary (v1 vs v2)

| Metric | v1 (817) | v2 (3635) | Δ |
|--------|----------|-----------|----|
| Ef MAE (test) | 0.2474 | 0.2681 | +8% |
| Ef R² (CV) | ? | 0.4621 | — |
| BG MAE (test) | 1.1874 | **0.5096** | **-57%** |
| BG R² (CV) | ? | 0.3823 | — |
| Eah R² (CV) | ? | -1.6948 | broken |

## Key Insight
1. 4.4× data did NOT improve Ef → architecture bottleneck confirmed
2. BG improved substantially → more data would help
3. Eah needs diagnosis before any architecture work
4. Next step: understand *why*, not just scale
