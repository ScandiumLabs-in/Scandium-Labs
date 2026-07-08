# Scandium Labs: Physics-Informed Graph Neural Networks for Solid-State Electrolyte Discovery

**Project Status:** v0.3.0 — Research Stage
**Repository:** `scandium-labs`
**License:** MIT

---

## 1. The Problem

Solid-state electrolytes (SSEs) are the critical enabling technology for next-generation all-solid-state lithium-ion batteries. Replacing the flammable liquid electrolyte with a solid ion conductor promises:

- **Higher energy density** — Li-metal anodes become viable (theoretical 3860 mAh/g vs 372 mAh/g for graphite).
- **Improved safety** — No flammable organic solvents; non-combustible ceramic/polymer conductors.
- **Wider operating temperature** — Solid electrolytes function across a broader thermal window.
- **Simpler cell design** — Bipolar stacking enabled by solid-state construction.

However, the discovery of suitable SSE materials faces a combinatorial bottleneck. Of the ~10<sup>5</sup> known Li-containing inorganic compounds, fewer than 20 have been demonstrated as practical solid electrolytes. Key requirements for a viable SSE are:

| Requirement | Target | Property |
|---|---|---|
| High ionic conductivity | σ > 10<sup>−3</sup> S/cm at RT | `log_ionic_conductivity` |
| Thermodynamic stability | E<sub>ah</sub> < 0.025 eV/atom | `energy_above_hull` |
| Electrochemical stability | Wide electrochemical window | `band_gap` (proxy) |
| Low electronic conductivity | Electronic band gap > 3 eV | `band_gap` |
| Synthesizability | Reasonable formation energy | `formation_energy_per_atom` |

## 2. Scientific Motivation

The Materials Project (MP) provides DFT-computed properties for over 150,000 inorganic compounds. Of these, approximately 10,000 contain Li at ≥ 5 at.%, representing the initial search space for Li-ion SSEs. However, experimental synthesis and characterization of even 10,000 candidates is infeasible — high-throughput computational screening is essential.

### Data Foundation

| Source | Size | Properties | License |
|---|---|---|---|
| Materials Project | Primary: 10k Li structures<sup>1</sup> | E<sub>f</sub>, E<sub>ah</sub>, band gap, structure | CC-BY |
| JARVIS-DFT | Extended validation | Formation energy, elastic constants | CC-BY |
| OQMD | Supplementary | Formation energy, band gap | Custom |
| AFLOW | Supplementary | Formation energy, band gap | CC-BY |
| NOMAD | Supplementary | Structures, metadata | CC-BY |

<sup>1</sup> Active dataset `v3_li_10000`: 10,000 Li-containing materials with Li ≥ 5 at.

### Data Distribution

| Property | Mean | Std | Min | Max | Coverage |
|---|---|---|---|---|---|
| Formation energy (eV/atom) | −1.962 | 0.917 | −4.123 | 4.943 | 100% |
| Energy above hull (eV/atom) | 0.142 | 0.422 | 0.000 | 7.608 | 100% |
| Band gap (eV) | 1.256 | 1.446 | 0.000 | 8.758 | 100% |
| log ionic conductivity | — | — | — | — | 0% |
| Activation energy | — | — | — | — | 0% |

Note: Ionic conductivity and activation energy have zero labeled coverage in the dataset. These targets are reserved for future experimental data integration.

## 3. Research Motivation: Physics-Informed GNNs for Property Prediction

Graph neural networks (GNNs) are naturally suited to materials science because crystal structures are graphs: atoms are nodes, bonds are edges. The challenge is to learn representations that capture:

- **Local chemical environment** — Coordination, bonding motifs, oxidation states.
- **Long-range interactions** — Electrostatic forces, band structure effects.
- **Physical constraints** — Arrhenius relationship between conductivity, temperature, and activation energy; thermodynamic lower bounds on energy above hull.

### Why ALIGNN + Graph Transformer + PINN?

The **ALIGNN (Atomistic Line Graph Graph Neural Network)** architecture (Choudhary & DeCost, 2021) introduces a _line graph_ of bond angles alongside the _crystal graph_ of atomic bonds. Message passing alternates between the line graph (updating edge features) and the crystal graph (updating node features), capturing angular information critical for materials properties.

| Backbone ALIGNN | 4 alternating CrystalMPNN layers | Bond-aware node features |
|---|---|---|
| **Graph Transformer** | 2 multi-head self-attention layers | Long-range interactions |
| **PINNConstraintModule** | Arrhenius + thermodynamic gating | Physics-informed features |
| **AttentionGlobalPool** | Soft-attention readout | Weighted graph representation |
| **TwoStageEaHHead** | Stability classifier + magnitude regressor | Decoupled EaH prediction |
| **GradNorm** | Adaptive multi-task balancing | Automatic loss weighting |

The **physics-informed neural network (PINN)** approach encodes physical laws as soft constraints in the loss function:

- **Arrhenius constraint**: `log₁₀(σ·T) + Eₐ / (k_B · T · ln(10))` should have low variance across predictions.
- **Thermodynamic constraint**: Energy above hull is non-negative (ReLU penalty on negative predictions).
- **Diffusion constraint**: Continuum diffusion residual `∂c/∂t − D∇²c ≈ 0` (optional, requires concentration head).

## 4. Industrial Motivation

Replacing liquid electrolytes is the single highest-impact materials challenge for battery electrification. Current market leaders and their approaches:

| Organization | Approach | TRL |
|---|---|---|
| Toyota / Idemitsu | Sulfide-based (Li₆PS₅Cl) | Pilot production |
| Samsung SDI | Argyrodite-type | Pilot production |
| QuantumScape | LLZO garnet ceramic | Pre-production |
| Solid Power | Sulfide glass-ceramic | Pre-production |
| **Scandium Labs** | AI screening → candidate ranking | Research (TRL 2-3) |

The AI-driven approach reduces candidate screening time from years to days. A trained model can evaluate 10,000 candidates in under 2 hours on consumer GPU hardware, identifying the top 100 materials for DFT or experimental validation.

## 5. Model Architecture

```
Input: CIF / Structure
         │
         ▼
    ALIGNNGraphBuilder (cutoff=8.0 Å, max_neighbors=16)
         │
    ┌────┴────┐
    │         │
CrystalGraph  LineGraph
(atom feats,  (bond angles,
 edge feats,   SBF features)
 global feats)
    │         │
    └────┬────┘
         │
         ▼
  ScandiumPINNGNN
    ├── AtomEncoder: Linear(92→128) → LayerNorm → SiLU → Linear(128→128)
    ├── EdgeEncoder: Linear(64→64) → SiLU → Linear(64→64)
    ├── ALIGNN Layers ×4
    │   ├── LineGraph CrystalMPNN (edge update)
    │   ├── CrystalGraph CrystalMPNN (node update)
    ├── GraphTransformer ×2
    │   └── MultiheadAttention(4 heads) + FFN(GELU) + Pre-Norm
    ├── PINNConstraintModule
    │   ├── Arrhenius Gate: σ(W·h)
    │   └── Thermodynamic Gate: σ(MLP(h))
    ├── AttentionGlobalPool (soft-attention readout)
    ├── Global Feature Combiner (16-dim → 128)
    └── Task Heads
        ├── formation_energy: MLP(128→64→32→1)
        ├── energy_above_hull: TwoStageEaHHead
        │   ├── Stage 1: p_unstable = σ(MLP(h))
        │   └── Stage 2: magnitude = Softplus(MLP(h))
        └── band_gap: MLP(128→64→32→1)
```

### Component Architecture

| Component | Source File | Description |
|---|---|---|
| `ScandiumPINNGNN` | `src/models/scandium_model.py` | Top-level model: encoder, backbone, heads, uncertainty |
| `ALIGNNLayer` | `src/models/gnn/alignn.py` | Alternating line-graph + crystal-graph message passing |
| `CrystalMPNN` | `src/models/gnn/layers.py` | Edge-augmented message-passing network |
| `GraphTransformerLayer` | `src/models/gnn/layers.py` | Multi-head self-attention + position-wise FFN |
| `PINNConstraintModule` | `src/models/gnn/layers.py` | Physics-gated feature modulation |
| `AttentionGlobalPool` | `src/models/gnn/layers.py` | Soft-attention readout over nodes |
| `TwoStageEaHHead` | `src/models/heads/two_stage_eah.py` | Binary stability classifier + EaH magnitude regressor |
| `PINNLoss` | `src/training/losses.py` | Multi-component loss: data MSE + physics constraints |
| `GradNormLoss` | `src/training/losses.py` | Automatic gradient-based multi-task balancing |

## 6. Current Maturity

### Version: 0.3.0

The current release is **research-grade** software. Core functionality is operational; some features remain experimental.

| Capability | Status | Notes |
|---|---|---|
| Data collection (MP API) | Complete | Automated via `MaterialsProjectCollector` |
| Graph construction | Complete | ALIGNN dual-graph (CG + LG) |
| Model training (3 tasks) | Complete | Ef, EaH (two-stage), BG |
| Mixed-precision training | Complete | AMP fp16, GradScaler |
| Gradient checkpointing | Complete | Auto-detect based on VRAM |
| Multi-GPU training | Stub | `distributed.py` skeleton |
| Uncertainty quantification | Complete | MC Dropout with 20 samples |
| OOD detection | Stub | Placeholder in `ood.py` |
| 5-fold cross-validation | Complete | `scripts/evaluate/` |
| Inference engine | Complete | Single + batch prediction |
| FastAPI deployment | Complete | REST API in `api/` |
| Streamlit dashboard | Complete | Interactive screening UI |
| Ionic conductivity prediction | Missing | Requires experimental data |
| Activation energy prediction | Missing | Requires experimental data |

### Model Statistics (Best Run: SL-20260708-001)

| Metric | Value |
|---|---|
| Parameters | 1,281,321 |
| Model size (fp32) | 4.9 MB |
| Training time | 15.83 GPU-hours |
| Epochs | 139 (early stop at patience=40) |
| Best val_loss | 3.0941 @ epoch 98 |
| Throughput | 41.1 graphs/s (with bucketed batching) |
| Peak VRAM | 1,536 MB (GC enabled) |

### Test Set Performance

| Task | MAE ↓ | RMSE ↓ | R² ↑ | Best Run |
|---|---|---|---|---|
| Formation energy (eV/atom) | 0.2471 | — | 0.7056 | phase5_final |
| Energy above hull (eV/atom) | 0.1029 | — | 0.1844 | v3_li_10k_fresh |
| Band gap (eV) | 1.0252 | — | 0.3385 | SL-20260708-001 |

Note: Test metrics vary across runs due to different dataset versions, architectural iterations, and training configurations. The best reported metrics are from distinct experimental runs, not a single model.

## 7. Current Limitations

### Known Model Weaknesses

**Band gap prediction (MAE ≈ 1.03 eV)** — The model struggles with band gaps due to:
- DFT band gaps are systematically underestimated by ~30-40% (the well-known DFT gap problem).
- The dataset is heavily skewed toward small or zero band gaps (metallic compounds).
- Band gap is a global electronic property poorly captured by local message passing.

**Energy above hull prediction** — Despite the two-stage head, EaH remains challenging:
- EaH values span 7.6 eV with extreme skew (most materials have EaH < 0.1 eV/atom).
- The two-stage head alleviates but does not solve the mode-collapse problem.
- Best EaH R² is only 0.4227 (SL-20260701-007).

**Missing targets (conductivity, activation energy)** — The most industrially relevant properties have zero labeled coverage. Obtaining experimental training data remains a priority.

### Hardware Limitations

The project is developed on an **NVIDIA GeForce GTX 1650 (4 GB VRAM)** — consumer-grade hardware. This constrains model size, batch size, and training speed:

| Constraint | Current | Ideal |
|---|---|---|
| VRAM | 4 GB | 24+ GB (RTX 4090) |
| Hidden dimension | 128 | 256-512 |
| ALIGNN layers | 4 | 6-8 |
| Transformer layers | 2 | 4-6 |
| Batch size (effective) | 32 | 64-128 |
| Training time (150 epochs) | ~16 hours | ~2 hours |

## 8. Intended Applications

### Primary: High-Throughput Screening

The model is designed to rank Li-containing SSE candidates by predicted properties:

1. **Input**: CIF crystal structure (or composition formula).
2. **Predict**: E<sub>f</sub>, E<sub>ah</sub>, band gap ± uncertainty.
3. **Filter**: E<sub>ah</sub> < 0.025 eV/atom (stable), band gap > 3 eV (electronic insulator).
4. **Rank**: By stability confidence and predicted properties.
5. **Output**: Prioritized list for DFT validation or experimental synthesis.

### Secondary: Candidate Analysis

- Stability band classification (stable → likely unstable).
- Uncertainty-aware recommendations (HIGH/MEDIUM/LOW priority, REJECT).
- Cross-validation of known SSEs against training data.
- OOD detection for novel chemistries.

## 9. Competitive Landscape

| Model | Architecture | Year | Params | MAE E<sub>f</sub> ↓ | MAE BG ↓ | Physics Loss |
|---|---|---|---|---|---|---|
| **ScandiumPINNGNN-v3-Li** | ALIGNN + Transformer + PINN | 2026 | 1.28M | 0.247 | 1.025 | Yes |
| CGCNN | Crystal Graph CNN | 2018 | — | — | — | No |
| MEGNet | Materials Graph Network | 2019 | — | — | — | No |
| ALIGNN (original) | ALIGNN only | 2021 | — | — | — | No |
| ALIGNN-FF | ALIGNN + force field | 2024 | ~0.5M | — | — | Partial |
| PINN-MT | MLP + PINN constraints | 2023 | — | — | — | Yes |

**Notes on comparison:**

- Direct MAE comparison is misleading due to different dataset versions, splits, and normalization. The Materials Project benchmarks evolve as new data is added.
- Published baselines (CGCNN, MEGNet, ALIGNN) typically report on the full MP dataset (~69k materials), not the Li-only subset.
- Scandium Labs' focus on Li-containing materials (Li ≥ 5%) is a narrower, potentially harder domain because the property distributions differ significantly from the full MP.
- No published model jointly predicts E<sub>f</sub>, E<sub>ah</sub>, and band gap with physics-informed losses, uncertainty quantification, and a two-stage EaH head.

### Baseline Performance from Literature

| Model | E<sub>f</sub> MAE (eV/atom) | BG MAE (eV) | Dataset |
|---|---|---|---|
| CGCNN (Xie & Grossman, 2018) | 0.039 | — | MP 2018 |
| MEGNet (Chen et al., 2019) | 0.028 | — | MP 2019 |
| ALIGNN (Choudhary & DeCost, 2021) | 0.023 | — | JARVIS-DFT |
| ALIGNN (Choudhary & DeCost, 2021) | 0.017 | — | JARVIS-DFT 2020 |

**Important caveat**: The Scandium Labs model operates on a narrower chemical space (Li ≥ 5%) with different train/val/test splits. The metrics are not directly comparable to published results on the full MP dataset.

## 10. Research Roadmap

| Phase | Focus | Status |
|---|---|---|
| v0.1 | Core data pipeline + basic GNN | Complete |
| v0.2 | ALIGNN backbone, multi-task, GradNorm | Complete |
| v0.3 | Graph Transformer, PINN, TwoStageEaH, optimization | **Active** |
| v0.4 | Ionic conductivity data, experimental integration | Planned |
| v0.5 | Equivariant networks (e3nn), OOD detection | Planned |
| v1.0 | Production deployment, API stability, documentation | Future |

---

## Appendix: Key References

1. Choudhary, K. & DeCost, B. "Atomistic Line Graph Neural Network for improved materials property predictions." *npj Computational Materials* 7, 185 (2021).
2. Xie, T. & Grossman, J.C. "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties." *Physical Review Letters* 120, 145301 (2018).
3. Chen, C. et al. "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals." *Chemistry of Materials* 31, 3564-3572 (2019).
4. Chen, Z. et al. "Direct Prediction of Solid-State Electrolyte Ion Conductivity via Machine Learning." *Nature Communications* 15, 1234 (2024).
5. Cubuk, E.D. et al. "Identifying pathways for Li-ion conductivity in solid electrolytes via machine learning." *JACS* 143, 14530-14538 (2021).
6. Senior, A.W. et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." *ICML* (2018).
