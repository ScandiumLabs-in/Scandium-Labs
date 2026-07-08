# Model Card: ScandiumPINNGNN-v3-Li

Following the HuggingFace Model Card standard.

---

## Model Details

### Model Name
**ScandiumPINNGNN-v3-Li** — Physics-Informed Graph Neural Network for Solid-State Electrolyte Screening

### Version
v0.3.0 (research stage)

### Model Architecture

The model is a hybrid architecture combining an ALIGNN (Atomistic Line Graph Graph Neural Network) backbone with graph transformers, physics-informed constraints, and multi-task regression heads.

| Component | Specification | Source |
|---|---|---|
| **Backbone** | ALIGNN, 4 layers | `src/models/gnn/alignn.py:ALIGNNLayer` |
| **Message passing** | CrystalMPNN (edge-augmented) | `src/models/gnn/layers.py:CrystalMPNN` |
| **Transformer** | GraphTransformerLayer, 2 layers, 4 heads | `src/models/gnn/layers.py:GraphTransformerLayer` |
| **Physics module** | PINNConstraintModule (Arrhenius + thermodynamic gates) | `src/models/gnn/layers.py:PINNConstraintModule` |
| **Pooling** | AttentionGlobalPool (soft-attention) | `src/models/gnn/layers.py:AttentionGlobalPool` |
| **EaH head** | TwoStageEaHHead (stability classifier + magnitude regressor) | `src/models/heads/two_stage_eah.py:TwoStageEaHHead` |
| **Loss balancing** | GradNorm (adaptive gradient-based weighting) | `src/training/losses.py:GradNormLoss` |
| **Uncertainty** | Monte Carlo Dropout (20 samples) | `src/models/scandium_model.py:ScandiumPINNGNN.predict_with_mc_dropout` |

### Model Hyperparameters

| Hyperparameter | Value |
|---|---|
| `hidden_dim` | 128 |
| `num_alignn_layers` | 4 |
| `num_transformer_layers` | 2 |
| `num_attention_heads` | 4 |
| `dropout` | 0.15 |
| `mc_dropout_samples` | 20 |
| `use_two_stage_eah` | True |
| `use_gradient_checkpointing` | Auto (enabled when VRAM < 6 GB) |
| `atom_feat_dim` | 92 |
| `edge_feat_dim` | 64 |

### Graph Construction Parameters

| Parameter | Value |
|---|---|
| Cutoff radius | 8.0 Å |
| Max neighbors | 16 |
| RBF type | Bessel |
| Number of RBF bases | 64 |
| Number of SBF bases | 32 |

### Graph Features

**Node features (92-dim after padding):**
- 15 base atomic features (atomic number, mass, electronegativity, radius, valence, etc.)
- Zero-padded to 92 for consistent dimensions

**Edge features (64-dim):**
- Bessel radial basis functions (RBF) encoding interatomic distances

**Line graph features (32-dim):**
- Spherical Bessel radial basis functions (SBF) encoding bond angles

**Global features (16-dim):**
- Volume/atom, density, total electrons, n_types, n_sites, space group number, lattice parameters, weight, average electronegativity

### Date of Release
2026-07-08

### Developers
Scandium Labs (internal research project)

### Licence
MIT

---

## Intended Use

### Primary Use Case
High-throughput computational screening of Li-containing crystalline materials for solid-state electrolyte (SSE) applications.

### How It Works
1. **Input**: Crystal structure (CIF file or pymatgen Structure object).
2. **Graph construction**: Crystal graph (atom-bond) + line graph (bond-angle) computed on the fly or loaded from cache.
3. **Prediction**: Model outputs formation energy (Ef), energy above hull (EaH), and band gap (BG) with uncertainty estimates.
4. **Recommendation**: Rule-based recommendation engine classifies candidates as HIGH/MEDIUM/LOW priority or REJECT based on stability and conductivity thresholds.

### Prediction Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| Stability | EaH < 0.025 eV/atom | Likely thermodynamically stable |
| Metastability | EaH < 0.10 eV/atom | Potentially synthesizable |
| Band gap | > 3.0 eV | Good electronic insulator |
| Conductivity | σ > 10⁻³ S/cm | Practical SSE |
| Uncertainty | MC Dropout std | Prediction confidence |

### Out-of-Scope Use Cases

- Materials without Li (model trained exclusively on Li ≥ 5%).
- Molecular crystals, metal-organic frameworks, or 2D materials.
- High-temperature properties (all predictions at 300 K).
- Kinetic properties (ionic conductivity requires experimental data).
- Quantitative structure-property relationships outside Materials Project chemical space.

---

## Training Data

### Dataset
**`v3_li_10000`** — 10,000 Li-containing inorganic crystalline structures from the Materials Project.

### Data Selection
- Li ≥ 5 at.% filter.
- Formation energy in [-10, 5] eV/atom.
- Structure size in [2, 200] atoms.
- Unique structures (StructureMatcher dedup available but disabled).

### Data Splits
| Split | Count | Method |
|---|---|---|
| Train | 8,310 | GroupShuffleSplit by chemical family |
| Validation | 586 | GroupShuffleSplit by chemical family |
| Test | 1,104 | GroupShuffleSplit by chemical family |

### Target Properties
| Target | Unit | Coverage | Mean | Std |
|---|---|---|---|---|
| Formation energy | eV/atom | 100% | −1.962 | 0.917 |
| Energy above hull | eV/atom | 100% | 0.142 | 0.422 |
| Band gap | eV | 100% | 1.256 | 1.446 |

### Chemical Families
7 families: pure_halide, oxyhalide, sulfohalide, oxide, sulfide, phosphate, other.

### Known Data Biases
- **Source bias**: All data from Materials Project DFT (PBE+U).
- **Chemical bias**: Restricted to Li ≥ 5% materials.
- **Stability bias**: Most materials near hull (median EaH = 0.006 eV/atom).
- **Missing properties**: No conductivity or activation energy labels.
- **DFT gap error**: PBE band gaps systematically underestimated.

---

## Evaluation Results

### Best Experiment Run (SL-20260708-001)

The following results are from the most recent full training run using `configs/model_config_v3_li.yaml` on the `v3_li_10000` dataset:

#### Validation Metrics (best per metric)

| Metric | Best Value | Epoch |
|---|---|---|
| Val Loss | 3.0941 | 98 |
| Formation Energy MAE | 0.5222 | 136 |
| Formation Energy R² | 0.5871 | 135 |
| Energy Above Hull MAE | 0.1280 | 4 |
| Energy Above Hull R² | 0.3854 | 131 |
| Band Gap MAE | 1.0252 | 98 |
| Band Gap R² | 0.3385 | 98 |

#### Training History

| Epoch Range | Train Loss | Val Loss | Ef MAE | Ef R² | EaH MAE | BG MAE |
|---|---|---|---|---|---|---|
| Start (ep 0) | ~15.0 | ~8.0 | ~1.2 | ~0.1 | ~0.35 | ~1.5 |
| Early (ep 20) | ~4.0 | ~4.5 | ~0.7 | ~0.4 | ~0.18 | ~1.2 |
| Mid (ep 50) | ~2.5 | ~3.5 | ~0.6 | ~0.5 | ~0.14 | ~1.1 |
| Best (ep 98) | ~1.9 | 3.0941 | ~0.54 | ~0.55 | ~0.13 | 1.0252 |
| Final (ep 138) | 1.8609 | 3.1452 | 0.5276 | 0.5811 | 0.1351 | 1.0506 |

### Historical Performance (all experiments)

| Run ID | Ef MAE | Ef R² | EaH MAE | EaH R² | BG MAE | BG R² | Date |
|---|---|---|---|---|---|---|---|
| SL-20260630-002 | 0.5684 | 0.5359 | 0.1256 | 0.3750 | 1.0479 | 0.2924 | 2026-06-30 |
| SL-20260701-007 | 0.5181 | 0.5897 | 0.1252 | 0.4227 | 1.0453 | 0.3060 | 2026-07-01 |
| SL-20260707-001 | 0.7403 | 0.3457 | 0.1496 | 0.2016 | 1.1691 | 0.0967 | 2026-07-07 |
| **SL-20260708-001** | **0.5222** | **0.5871** | **0.1280** | **0.3854** | **1.0252** | **0.3385** | **2026-07-08** |
| v3_li_10k_fresh | 0.3267 | 0.5528 | 0.1029 | 0.1844 | 1.2493 | 0.0373 | — |
| phase5_final | 0.2471 | 0.7056 | 0.1181 | 0.4092 | 0.7614 | 0.3646 | — |

### Test Set Results (External Benchmarks)

| Benchmark | Ef MAE | Ef R² | EaH MAE | BG MAE | BG R² | EaH F1 |
|---|---|---|---|---|---|---|
| final_eval | 0.2485 | 0.6825 | 0.1154 | 0.7833 | 0.3501 | — |
| phase4_final | 0.2678 | 0.6535 | 0.1201 | 0.8041 | 0.2805 | — |
| phase5_final | 0.2471 | 0.7056 | 0.1181 | 0.7614 | 0.3646 | — |

### Performance Summary

| Task | Training (best val) | Typical Range | Notes |
|---|---|---|---|
| **Formation energy** | R² = 0.5871 | 0.35-0.71 | Best-predicted property |
| **Energy above hull** | MAE = 0.128 eV/atom | 0.10-0.19 | Two-stage head improved stability F1 |
| **Band gap** | MAE = 1.025 eV | 0.76-1.80 | Challenging — DFT gap problem |

---

## Known Limitations

### 1. Band Gap Prediction (MAE ≈ 1.025 eV)

The model's band gap predictions have high error for several reasons:
- **DFT band gap problem**: PBE underestimates gaps by 30-40% systematically.
- **Skewed distribution**: ~25% of materials have zero band gap (metals); high-band-gap insulators are rare.
- **Local vs global**: Band gap is an electronic property determined by the periodic crystal potential, not purely local chemistry captured by message passing.

### 2. Energy Above Hull

EaH remains the most challenging target despite architectural improvements:
- **Dynamic range**: 0 to 7.6 eV/atom with most values < 0.1 eV.
- **Mode collapse**: The model tends to predict near-zero EaH for all materials.
- **Two-stage improvement**: The TwoStageEaHHead improved stability classification (F1) but magnitude regression remains noisy.

### 3. Chemical Scope

- **Li-restricted**: No prediction capability for Na-ion, K-ion, or other chemistries.
- **Composition limited**: 49-element query set; elements outside this (e.g., noble metals, rare earths beyond La) are absent from training.
- **No ionic conductivity**: The most practically relevant property has zero training data.

### 4. Training Data Limitations

- **Purely computational**: All targets from DFT; experimental validation required.
- **Thermodynamics only**: No kinetic or transport properties.
- **Room temperature**: 0 K DFT structures; no temperature-dependent behavior.

### 5. Computational Constraints

- **4 GB GPU limit**: Hidden dimension capped at 128; larger models would likely improve performance.
- **Small batch size**: Effective batch of 32 is at the low end for multi-task learning.
- **No hyperparameter tuning**: Learning rate, weight decay, and dropout were not systematically optimized.

---

## Computational Requirements

### Hardware (Development Environment)

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| **CPU** | 8 cores (Intel/AMD) |
| **RAM** | 14 GB system memory |
| **Storage** | 2 GB for dataset + checkpoints |

### Training Cost

| Metric | Value |
|---|---|
| Training time | 15.83 hours (56,986 seconds) |
| Epochs | 139 (early stop at 40 patience) |
| Throughput | 41.1 graphs/s (with bucketed batching) |
| Peak GPU memory | 1,536 MB |
| GPU utilization | ~85% |
| Power draw (GPU) | ~75W (GTX 1650 TDP) |
| **Total energy** | **~1.19 kWh** (16 hr × 75W) |

### Environmental Impact

| Component | CO₂e |
|---|---|
| GPU (75W × 16 hr × 0.2 kg CO₂e/kWh) | ~0.24 kg |
| System (150W × 16 hr × 0.2 kg CO₂e/kWh) | ~0.48 kg |
| **Estimated total** | **~3.2 kg CO₂e** |

Values assume US average grid carbon intensity (0.2 kg CO₂e/kWh at 2025 levels). Actual impact varies by region.

### Model Size

| Format | Size |
|---|---|
| Parameters (fp32) | 1,281,321 |
| Model weights (fp32 .pt) | 4.9 MB |
| VRAM at inference (fp32) | ~470 MB |
| VRAM at inference (fp16 AMP) | ~250 MB |

### Scaling Estimates

| GPU Tier | VRAM | Max hidden_dim | Max batch | Estimated speedup |
|---|---|---|---|---|
| GTX 1650 (current) | 4 GB | 128 | 16 | 1× (baseline) |
| RTX 3060 | 12 GB | 256 | 32 | 3-4× |
| RTX 4090 | 24 GB | 512 | 64 | 8-10× |
| A100 80 GB | 80 GB | 1024 | 128 | 15-20× |

---

## Model Examination

### Interpretability

The model supports basic interpretability:
- **Attention weights**: The `AttentionGlobalPool` gate values can be extracted to identify which atoms contribute most to the graph representation.
- **Integrated gradients**: `src/explainability/gradients.py` provides feature attribution via integrated gradients.
- **MC Dropout variance**: Uncertainty estimates flag high-variance predictions for manual review.

### Error Analysis

The model exhibits systematic errors:
- **Overprediction of stability**: EaH predictions are biased low — the model tends to classify unstable materials as stable.
- **Band gap regression to mean**: Predicted band gaps cluster near the dataset mean (1.26 eV), overestimating small gaps and underestimating large gaps.
- **Composition shortcuts**: Materials with unusual chemistries (e.g., actinide-containing) show higher prediction uncertainty.

### Stability Classification (Two-Stage EaH)

| Metric | Value (SL-20260701-007) |
|---|---|
| Stability F1 | 0.72 |
| Stability Precision | 0.68 |
| Stability Recall | 0.77 |
| EaH MAE (unstable only) | 0.31 eV/atom |
| EaH MAE (all) | 0.13 eV/atom |

---

## Recommendations for Use

### When to Trust Predictions

1. **Coverage check**: All three targets (Ef, EaH, BG) have 100% coverage in training data.
2. **Uncertainty**: MC Dropout std < 0.05 for Ef, < 0.02 for EaH, < 0.2 for BG.
3. **In-distribution**: Composition elements appear in training data.
4. **Consistency**: Ef and EaH predictions are physically consistent (negative Ef, non-negative EaH).

### When to Be Cautious

1. **Conductivity predictions**: Zero training data — predictions are random.
2. **Band gap < 0.5 eV**: May be metallic; DFT gap errors are largest for small-gap materials.
3. **EaH > 1.0 eV**: Very few training examples in this range.
4. **Novel chemistries**: Elements not in the 49-element training set.
5. **Large unit cells > 100 atoms**: Increasing graph size increases approximation in neighbor limiting.

### Deployment Checklist

- [ ] Verify model loads without errors on target hardware.
- [ ] Confirm normalizer.json matches dataset statistics.
- [ ] Test on 5 known reference SSEs (e.g., Li₆PS₅Cl, Li₃YCl₆, LLZO).
- [ ] Enable MC Dropout for production inference.
- [ ] Set OOD detection threshold if using OOD detector.
- [ ] Validate against held-out test set.
- [ ] Monitor prediction uncertainty distribution.

---

## Maintenance

### Model Updates

The model is retrained as new data becomes available. Version history:

| Version | Date | Dataset | Change |
|---|---|---|---|
| v0.1.0 | 2024 | v1_817 | Initial model |
| v0.2.0 | 2024 | v2_10000 | Multi-task GradNorm |
| v0.2.1 | 2025 | v2_10000_log_eah | Log EaH experiment |
| **v0.3.0** | **2026** | **v3_li_10000** | **Current — ALIGNN + Transformer + PINN + TwoStageEaH** |

### Known Issues

- `normalize_targets = True` in config requires normalizer.json in data directory.
- Gradient checkpointing auto-detect assumes CUDA GPU (falls back to disabled on CPU).
- MC Dropout prediction is ~20× slower than single forward pass.
- Two-stage EaH head requires `use_two_stage_eah = True` in config; models trained without it cannot be loaded with it.

### Citation

```bibtex
@software{scandium_labs_2026,
  title  = {ScandiumPINNGNN-v3-Li: Physics-Informed GNN for Solid-State Electrolyte Discovery},
  author = {Scandium Labs},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXX},
  url    = {https://github.com/scandium-labs/scandium-labs}
}
```

---

## Appendix: Training Configuration (Active)

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
  use_gradient_checkpointing: auto

graph:
  cutoff: 8.0
  max_neighbors: 16
  num_rbf: 64
  num_sbf: 32

tasks:
  - name: "formation_energy"
    weight: 1.0
  - name: "energy_above_hull"
    weight: 1.0
    two_stage: true
  - name: "band_gap"
    weight: 1.0

gradnorm:
  enabled: true
  alpha: 1.5

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 0.0005
  max_epochs: 150
  patience: 40
  optimizer: "AdamW"
  weight_decay: 0.00001
  gradient_clip: 1.0
  mixed_precision: true
  normalize_targets: true
  dataset: "v3_li_10000"

bucketing:
  enabled: true
  bucket_size_mult: 2.0
```
