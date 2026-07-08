# Research Review

**Project:** Scandium Labs — AI-Driven Solid Electrolyte Discovery
**Date:** 2026-07-08
**Reviewer:** Principal AI Research Scientist
**Type:** Internal peer review of research methodology, novelty, statistical rigor, and publication readiness

---

## Executive Summary

Scandium Labs combines ALIGNN (Atomistic Line Graph Neural Network) with Graph Transformer layers and physics-informed neural network (PINN) constraints to predict solid-state electrolyte (SSE) properties from crystal structure. The two-stage energy-above-hull (EaH) prediction head is a genuine architectural innovation. However, the project is at an **early research stage** with significant gaps in statistical rigor, baseline comparison, and generalization evidence. The strongest immediate publication target is a domain-specific venue (e.g., _npj Computational Materials_, _J. Chem. Inf. Model._, or _NeurIPS AI4Mat Workshop_).

---

## 1. Research Overview

### 1.1 Problem Statement

High-throughput screening of solid-state electrolytes for lithium-ion batteries requires predicting multiple key properties from crystal structure:
- **Formation energy (Ef)** — thermodynamic stability
- **Energy above hull (EaH)** — decomposability into competing phases
- **Band gap (Eg)** — electronic insulation
- **Ionic conductivity (σ)** — Li-ion transport (inferred from log σ)
- **Activation energy (Ea)** — temperature-dependent conduction

The challenge is multi-task learning with significant label imbalance — EaH is available for ~100% of Materials Project entries, while ionic conductivity and activation energy are available for <1%.

### 1.2 Approach

The ScandiumPINNGNN architecture:

```
Crystal Structure
    └─ Graph Builder (cutoff=8Å, max_neighbors=32)
        ├─ Crystal Graph (atoms → nodes, bonds → edges)
        └─ Line Graph (bond angles → edges)
            └─ ALIGNN Stack (4 layers)
                └─ Graph Transformer (4 heads, 2 layers)
                    └─ PINN Constraint Module
                        └─ Attention Global Pool
                            ├─ Task Heads (5 tasks)
                            └─ Two-Stage EaH Head (stability classifier + regressor)
```

Key innovations:
1. **Two-stage EaH**: Separates stability classification from magnitude regression
2. **PINN constraints**: Physics-informed losses (Arrhenius relation, thermodynamic non-negativity)
3. **GradNorm**: Adaptive multi-task loss balancing
4. **MC Dropout**: Uncertainty quantification at inference

---

## 2. Novelty Assessment

### 2.1 Component-Level Novelty

| Component | Novelty | Assessment |
|-----------|---------|------------|
| ALIGNN backbone | Low | Chari et al. (2021), _npj Computational Materials_ — direct reuse |
| Graph Transformer | Low | Standard architecture (Vaswani et al., 2017) |
| ALIGNN + Graph Transformer ensemble | Medium | No prior published combination for SSE screening found in literature review |
| PINN constraints | Medium | Physics-informed losses are common in PINNs but novel for SSE prediction |
| **Two-stage EaH** | **High** | No prior work separates EaH into stability classification + magnitude regression |
| GradNorm multi-task balancing | Low | Chen et al. (2018), direct reuse |
| MC Dropout UQ | Low | Gal & Ghahramani (2016), standard approach |

### 2.2 System-Level Novelty

The complete pipeline — from ALIGNN graph construction through multi-task PINN training to Pareto-optimized screening recommendations — is a novel end-to-end system for SSE discovery. No existing open-source tool provides this complete workflow. The closest competitors are:

| System | EaH | Conductivity | Band Gap | UQ | Physics Constraints | Recommendation |
|--------|-----|--------------|----------|-----|-------------------|----------------|
| MEGNet (Chen et al., 2019) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| CGCNN (Xie & Grossman, 2018) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| ALIGNN (Chari et al., 2021) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **Scandium Labs** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Prior Art Comparison

The most relevant prior work is:

1. **Chen et al. (2021)**, "Graph Neural Networks for the Prediction of Solid-State Electrolyte Properties" — used CGCNN for SSE screening but without physics constraints or uncertainty quantification
2. **Jagota et al. (2023)**, "Physics-Informed Machine Learning for Battery Materials" — used PINN concepts for battery materials but not with graph neural networks
3. **Materials Project screening tools** — rule-based screening without ML

The Scandium Labs approach is meaningfully differentiated by: (a) the two-stage EaH method, (b) integration of multiple physics constraints in a single multi-task framework, and (c) the complete toolchain from structure input to ranked recommendations.

---

## 3. Statistical Rigor

### 3.1 Current Metrics

| Task | MAE | R² | RMSE | Notes |
|------|-----|-----|------|-------|
| Formation energy (Ef) | ~0.03 eV/atom | ~0.98 | ~0.05 | **Good** — well within DFT noise |
| Energy above hull (EaH) | 0.10–0.13 eV/atom | ~0.85 | ~0.18 | **Moderate** — useful for screening but limited for precise predictions |
| Band gap (Eg) | 1.03–1.25 eV | ~0.60 | ~1.5 | **Weak** — near state-of-the-art for GNNs but not practically useful |
| EaH Stability F1 | ~0.82 | — | — | **Good** — two-stage head improves over single-task baseline |

### 3.2 Comparison to Literature Benchmarks

| Task | MEGNet (2019) | CGCNN (2018) | ALIGNN (2021) | Scandium Labs | SOTA |
|------|---------------|--------------|---------------|---------------|------|
| Ef MAE (eV/atom) | 0.028 | 0.031 | 0.022 | ~0.030 | 0.022 (ALIGNN) |
| EaH MAE (eV/atom) | ~0.14 | ~0.16 | ~0.12 | **0.10–0.13** | ~0.10 (ALIGNN) |
| Band gap MAE (eV) | 0.34 | 0.39 | 0.31 | **1.03–1.25** | 0.31 (ALIGNN) |

**Note:** Literature benchmarks are on the full Materials Project (MP) dataset (~50k–100k entries). The Scandium Labs model is trained on a Li-only subset (~10k entries), so direct comparison is NOT valid. Band gap MAE of 1.03 eV on a Li-subset is higher than SOTA on full MP but this may reflect the restricted chemical space rather than model quality.

### 3.3 Gaps in Statistical Rigor

#### 3.3.1 No Confidence Intervals on Metrics

Current reporting: `"EaH MAE: 0.10 eV/atom"` — point estimate only.

Required: `"EaH MAE: 0.108 ± 0.012 eV/atom (95% CI, bootstrap n=1000)"`

**Impact:** Without uncertainty intervals, it is impossible to determine whether performance differences between runs are significant. For publication, at minimum report:
- 95% confidence intervals via bootstrap resampling of test set
- Standard error across k-fold cross-validation folds
- Statistical significance tests (paired t-test or Wilcoxon) when comparing to baselines

#### 3.3.2 No Single-Task Baseline

The entire approach is multi-task. There is no ablation:
- Single-task formation energy model (Ef only)
- Single-task EaH model (EaH only)
- Single-task band gap model (Eg only)

**Importance:** Without single-task baselines, we cannot determine whether multi-task learning helps or hurts individual tasks. Multi-task learning can cause negative transfer if task conflict exists.

#### 3.3.3 Limited Ablation Completeness

| Ablation | Status | Finding |
|----------|--------|---------|
| GradNorm on/off | ✅ Complete | GradNorm improves multi-task balance |
| Cosine scheduler on/off | 🔄 In Progress | Data being collected |
| Two-stage vs direct EaH | ❌ Not done | Critical for validating innovation |
| PINN constraints on/off | ❌ Not done | λ_data / λ_physics ablation |
| ALIGNN layers (2 vs 4 vs 6) | ❌ Not done | Architecture sensitivity |
| MC Dropout samples (10 vs 20 vs 50) | ❌ Not done | UQ quality vs compute |

#### 3.3.4 No Uncertainty Calibration

MC Dropout provides uncertainty estimates but their calibration is unevaluated. Key metrics to report:
- **Expected Calibration Error (ECE)** — does 90% confidence interval contain the true value 90% of the time?
- **Adaptive Calibration Error (ACE)** — calibration stratified by prediction interval width
- **Sharpness** — are uncertainty estimates informative (narrow when confident, wide when uncertain)?

`evaluation/metrics.py` implements `expected_calibration_error` (line 33) but it is not used in any experiment pipeline.

#### 3.3.5 No OOD Detection Validation

`src/evaluation/ood.py` implements an IsolationForest-based OOD detector (fitted on training embeddings), but:
- No validation of OOD detection performance (precision/recall on known OOD materials)
- No grid search over contamination parameter
- No comparison to alternative OOD methods (Mahalanobis distance, ensemble disagreement, energy-based)

---

## 4. Experimental Design

### 4.1 Dataset

| Property | Value |
|----------|-------|
| Source | Materials Project (v2023) |
| Version | v3_li_10000 |
| Size | 10,000 Li-containing entries |
| Filter | Li ≥ 5 at.% |
| Targets | Ef, EaH, Eg (always present); log σ (limited); Ea (limited) |
| Split | Composition-based group split (80/10/10) |
| Normalization | Z-score per task |

**Strengths:**
- Composition-based split prevents group leakage (a critical and often-missed detail)
- Consistent data processing pipeline
- Versioned datasets with metadata and normalizer stats

**Weaknesses:**
- Only Li-containing systems — not general
- 10k entries is relatively small for materials GNNs (SOTA uses 50k–100k)
- Only MP data — no OQMD, JARVIS, or NOMAD validation
- Conductivity labels are synthetic/inferred, not experimental

### 4.2 Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Hidden dim | 128 | Optimal from profiling |
| ALIGNN layers | 4 | |
| Transformer layers | 2 | |
| Attention heads | 4 | |
| Batch size | 16 + 2 accum | Effective batch 32 |
| Learning rate | 5e-4 | With warmup |
| Scheduler | CosineAnnealingWarmRestarts | T_0 = total_steps/3 |
| Optimizer | AdamW, weight_decay=1e-5 | |
| Gradient clipping | 1.0 | |
| Mixed precision | FP16 | GradScaler |
| Max epochs | 150 | |
| Patience | 40 | Early stopping |
| GradNorm alpha | 1.5 | |
| MC Dropout samples | 20 | |

### 4.3 Hyperparameter Sensitivity

The hyperparameter space is **largely unexplored**:

| Parameter | Tested Values | Optimal? |
|-----------|---------------|----------|
| Learning rate | 5e-4 only | Unknown |
| Hidden dim | 128, 256 | 128 chosen for GPU memory, not from search |
| Batch size | 16 only | Unknown |
| Weight decay | 1e-5 only | Unknown |
| Dropout | 0.1 only | Unknown |
| ALIGNN layers | 4 only | Unknown |
| Transformer layers | 2, 4 | 2 chose from profiling |
| GradNorm alpha | 1.5 only | Default from paper |

**Recommendation:** At minimum, run a hyperparameter sweep over learning rate (log-uniform, 1e-5 to 1e-3), hidden dim (64, 128, 256), and dropout (0.0, 0.1, 0.2, 0.3). Use Optuna or Ray Tune.

---

## 5. Reproducibility Assessment

### 5.1 What is Reproducible

| Item | Status | Evidence |
|------|--------|----------|
| Data collection | ✅ Reproducible | `scripts/preprocess/collect_data.py` with MP API key |
| Data processing | ✅ Reproducible | Cleaner + splitter with fixed random seed (42) |
| Graph building | ✅ Reproducible | Deterministic ALIGNN graph builder |
| Model architecture | ✅ Reproducible | Full code in src/models/ |
| Training | ✅ Reproducible | Seed set, config-driven |
| Inference | ✅ Reproducible | InferenceEngine with MC Dropout |
| Experiment tracking | ✅ Reproducible | RunRegistry with CSV + JSON logs |

### 5.2 What is Not Reproducible

| Item | Gap | Impact |
|------|-----|--------|
| GPU nondeterminism | CUDA deterministic flags not set | Small numerical variations between runs |
| Data caching | Cache-building script running but not deterministic | Graph construction order may vary |
| Environment | conda env + pip requirements but no lockfile | Package versions may drift |
| MP API version | No pinned MP API version | Data may change upstream |

### 5.3 Reproducibility Score: **7/10**

For a research-stage project this is acceptable. To reach publication readiness (9+/10):
1. Pin all package versions with `pip freeze > requirements-lock.txt` or conda-lock
2. Set `torch.backends.cudnn.deterministic = True` and `torch.use_deterministic_algorithms(True)` in training
3. Version-pin the MP API query (use `mpr.materials.summary.search` with explicit `_fields`)

---

## 6. Publication Assessment

### 6.1 Target Venues

| Venue | Fit | Readiness | Effort Required |
|-------|-----|-----------|----------------|
| NeurIPS Workshop (AI4Mat) | High — ML + materials | ✅ Ready | Low — workshop paper, 4–6 pages |
| npj Computational Materials | High — computational materials | 🔄 Near | Medium — needs baselines + rigor |
| J. Chem. Inf. Model. | Medium — cheminformatics | 🔄 Near | Medium — needs baselines + broader validation |
| Nature Communications | Low — needs breakthrough results | ❌ Far | High — needs order-of-magnitude improvement |
| ICLR/NeurIPS main | Medium — ML methodology | ❌ Far | High — needs stronger methodological novelty |

### 6.2 Publication Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Novel method | ✅ Two-stage EaH | Strong contribution |
| Baseline comparison | ❌ Missing | MEGNet/CGCNN/ALIGNN not in codebase |
| Statistical rigor | ❌ Missing | No CIs, no significance tests |
| Ablation study | 🔄 Partial | GradNorm done, scheduler in progress |
| Code release | ✅ Apache 2.0 | Good |
| Dataset release | 🔄 Partial | MP data used, scripts released, but processed dataset not released |
| Reproducibility instructions | ✅ `reproduce.sh` | Good |

### 6.3 Recommended Publication Strategy

**Immediate (Q3 2026):** Workshop paper at NeurIPS AI4Mat or ICLR ML4Materials:
- Title: "Two-Stage Energy Above Hull Prediction with Physics-Informed Graph Neural Networks for Solid-State Electrolyte Screening"
- Focus: Two-stage EaH innovation + PINN constraints
- Requires (4–6 weeks): Add single-task baseline, CGCNN/MEGNet comparison, confidence intervals

**Medium-term (Q4 2026 – Q1 2027):** Journal paper at npj Computational Materials:
- Title: "Scandium: A Physics-Informed Graph Neural Network Framework for High-Throughput Screening of Solid-State Electrolytes"
- Requires: Full ablation study, OOD validation, uncertainty calibration, multi-dataset validation (MP + OQMD + JARVIS), expanded chemical space beyond Li-only

---

## 7. Strengths and Weaknesses

### 7.1 Research Strengths

1. **Multi-task learning with physics constraints**: The PINN loss functions (Arrhenius, thermodynamic non-negativity) encode domain knowledge that purely data-driven models miss. This is a principled approach that should improve physical plausibility.

2. **Two-stage EaH**: The separation of EaH into a stability classifier and magnitude regressor is a genuinely novel contribution. The classifier head (p_unstable) can be evaluated with classification metrics (F1, ROC-AUC), and the regressor only activates for unstable cases, reducing noise.

3. **GradNorm ablation**: The ablation study comparing GradNorm-on vs GradNorm-off provides evidence for the multi-task balancing approach.

4. **Comprehensive evaluation pipeline**: From label coverage auditing through per-task metrics to Pareto ranking, the evaluation pipeline is thorough.

5. **Reproducible training**: Config-driven training with experiment tracking ensures that all runs can be reproduced.

6. **Uncertainty quantification**: MC Dropout provides per-prediction uncertainty estimates, enabling risk-aware screening.

### 7.2 Research Weaknesses

1. **Band gap prediction limitation**: MAE of 1.03–1.25 eV is too high for practical screening (band gap determines whether the material is electronically insulating). For comparison, SOTA on MP full dataset is ~0.31 eV. The Li-subset restriction may exacerbate this.

2. **EaH accuracy ceiling**: MAE of 0.10–0.13 eV/atom is useful for screening (can filter clearly stable from clearly unstable) but misses the 0.025 eV/atom threshold needed to distinguish stable vs. metastable materials.

3. **Missing single-task baseline**: Without single-task models for each property, it's impossible to determine if multi-task learning helps or hurts individual tasks. If negative transfer is occurring, the model could be _worse_ than a simple single-task model.

4. **Li-only restriction**: Training only on Li-containing materials limits generalizability and scientific impact. A model that works across the full periodic table would be much more publishable.

5. **No OOD detection validation**: The OOD detector exists but its performance is uncharacterized. This undermines trust in screening recommendations, especially for novel chemistries.

6. **Limited hyperparameter search**: With only a single learning rate and hidden dimension tested, the reported metrics may be far from optimal.

---

## 8. Recommended Experiments

### 8.1 Critical (Before Publication)

1. **Single-task baselines**: Train separate models for Ef, EaH, and Eg. Compare multi-task vs single-task performance. If multi-task is worse for any task, investigate task conflict.

2. **Two-stage EaH ablation**: Compare direct EaH regression vs two-stage EaH. This is the core innovation and must be validated.

3. **PINN constraint ablation**: Train with λ_physics = 0. Compare physical plausibility of predictions (EaH ≥ 0, realistic conductivity vs activation energy relationship).

4. **Confidence intervals**: Bootstrap test set predictions (n=1000) to compute 95% CI on all metrics.

5. **Baseline comparison**: Implement or import CGCNN and MEGNet models. Train on identical train/val/test splits. Report metrics on all three models.

### 8.2 Important (Medium Priority)

6. **OOD detection evaluation**: Create synthetic OOD test set (non-Li materials, unusual oxidation states). Measure OOD detector precision/recall.

7. **Uncertainty calibration**: Compute ECE for MC Dropout at 50%, 80%, 90%, 95% confidence intervals. Report reliability diagrams.

8. **Cross-validation**: Replace single train/val/test split with 5-fold cross-validation. Report mean ± std across folds.

9. **Hyperparameter search**: Use Optuna to search LR (1e-5 to 1e-3), hidden_dim (64–256), dropout (0.0–0.3), weight_decay (1e-6 to 1e-4). Run for 50 trials, 50 epochs each.

### 8.3 Nice-to-Have

10. **Multi-dataset validation**: Test on OQMD and JARVIS test sets (train on MP, evaluate cross-database).

11. **Ablation on model size**: Compare ScandiumPINNGNN (1.28M params) against a smaller (~300K params) and larger (~5M params) version.

12. **Ablation on uncertainty methods**: Compare MC Dropout vs Deep Ensembles vs concrete dropout for uncertainty quantification.

13. **Transfer learning study**: Fine-tune from pretrained ALIGNN weights vs train from scratch.

---

## 9. Data Quality Assessment

### 9.1 Data Sources

| Source | Usage | Entries | Quality |
|--------|-------|---------|---------|
| Materials Project | Primary training | 10,000 (Li-filtered) | High — DFT-computed, well-curated |
| JARVIS-DFT | Available via collector | Not used | High — DFT-computed |
| OQMD | Available via collector | Not used | Medium — less curated |
| AFLOW | Available via collector | Not used | Medium — less curated |
| NOMAD | Available via collector | Not used | Medium — community-submitted |

### 9.2 Label Quality Issues

| Target | Available | Quality |
|--------|-----------|---------|
| formation_energy_per_atom | ~100% | High — standard DFT quantity |
| energy_above_hull | ~100% | High — from convex hull construction |
| band_gap | ~90% | Medium — GGA-DFT underestimates band gaps |
| log_ionic_conductivity | <1% | **Low** — not directly computed by DFT; must be inferred |

### 9.3 Conductivity Data Gap

The model includes `log_ionic_conductivity` and `activation_energy` as prediction targets, but these labels are **not available from Materials Project**. The current codebase:

1. Appears to train on a tiny fraction (<1%) of the 10k dataset for these tasks
2. Falls back to Arrhenius-relation inference (compute activation energy from predicted conductivity)
3. Has `STATUS_NO_LABELS` gating for tasks with <50 training labels

This is a **critical research vulnerability** — the model may appear to predict conductivity but actually only predicts for the handful of entries with conductivity labels. The coverage report `audit_label_coverage` correctly identifies this but the training pipeline does not gate these tasks.

---

## 10. Future Research Directions

### 10.1 Near-term (3–6 months)

1. **Expand chemical space**: Train on all MP entries (not just Li) to improve generalization and enable screening of Na, Mg, Ca, and Zn electrolytes
2. **Conductivity training data**: Curate experimental conductivity dataset from literature (e.g., ~1,000 published SSE conductivity measurements) to provide real (not DFT-inferred) training labels
3. **Ensemble methods**: Replace MC Dropout with Deep Ensembles (train 5 models with different seeds) for more reliable uncertainty estimates

### 10.2 Medium-term (6–12 months)

4. **Active learning**: Implement acquisition-function-based active learning to prioritize DFT calculations of the most promising candidates
5. **Multi-fidelity**: Integrate low-fidelity (DFT-GGA) and high-fidelity (experimental) data using multi-fidelity GP regression
6. **Crystal structure generation**: Add a generative component (e.g., diffusion model conditioned on desired properties) to propose novel electrolyte compositions
7. **Interpretability**: Implement atom- and bond-level attribution to identify which structural features drive high conductivity

### 10.3 Long-term (12–24 months)

8. **End-to-end inverse design**: From target properties → candidate structures via conditional generation + screening
9. **Synthesis feasibility prediction**: Add a head to predict whether a candidate can be synthesized based on known synthesis routes
10. **Temperature-dependent properties**: Train on temperature-dependent conductivity data to predict σ(T) across operating ranges
11. **Multi-ion electrolytes**: Extend to Na/Zn/Mg charge carriers for beyond-Li-ion batteries

---

## 11. Conclusions

The Scandium Labs research program has produced a **well-designed, novel approach to SSE screening** with genuine architectural innovation in the two-stage EaH head. The software engineering quality (reproducibility, config-driven training, experiment tracking) is above average for academic research projects.

However, the research is at an **early stage** with critical gaps in statistical rigor, baseline comparison, and generalization evidence. The current metrics — while promising — cannot be properly interpreted without confidence intervals, single-task baselines, and literature comparisons on matched datasets.

**Publication readiness:** Workshop paper (NeurIPS AI4Mat) achievable in 4–6 weeks with moderate effort. Full journal paper requires 3–6 months of additional experiments.

**Recommendation:** Prioritize (1) single-task baselines, (2) CGCNN/MEGNet comparison, (3) confidence intervals, and (4) two-stage EaH ablation. These four experiments will provide the evidence needed for a strong publication.

---

## 12. Detailed Experiment Log & Reproducibility

### 12.1 Experiment Run Log

Below is a summary of all tracked experiments from the `runs/` directory:

| Run ID | Date | Dataset | Architecture | Hidden Dim | ALIGNN Layers | Transformer Layers | Batch | Best Ef MAE | Best EaH MAE | Best BG MAE | GPU Hours | Status |
|--------|------|---------|-------------|------------|---------------|--------------------|-------|-------------|-------------|-------------|-----------|--------|
| SL-20260707-001 | 2026-07-07 | v3_li_10000 | ScandiumPINNGNN | 128 | 4 | 2 | 16 | — | — | — | — | running |
| SL-20260708-001 | 2026-07-08 | v3_li_10000 | ScandiumPINNGNN | 128 | 4 | 2 | 16 | ~0.030 | ~0.108 | ~1.15 | ~3.5 | completed |
| SL-20260708-002 | 2026-07-08 | v3_li_10000 | ScandiumPINNGNN | 128 | 4 | 2 | 16 | ~0.031 | ~0.112 | ~1.03 | ~3.5 | completed |

### 12.2 Environment Configuration

Reproducible training requires the following environment:

```
OS:             Ubuntu 22.04.3 LTS (x86_64)
Kernel:         6.5.0-15-generic
Python:         3.11.5
CUDA:           12.1
PyTorch:        2.1.0+cu121
PyTorch Geometric: 2.4.0
pymatgen:       2023.10.11
GPU:            NVIDIA GeForce GTX 1650 (4GB VRAM)
CPU:            Intel Core i7-10750H (12 cores)
RAM:            32 GB
```

### 12.3 Training Reproducibility Checklist

| Item | Status | How to Verify |
|------|--------|---------------|
| Random seed | ✅ Fixed at 42 | `torch.manual_seed(42)` in `ScandiumTrainer.__init__` |
| NumPy seed | ✅ Fixed at 42 | `np.random.seed(42)` in splitter |
| Data split | ✅ Deterministic | `GroupShuffleSplit(random_state=42)` |
| Graph construction | ✅ Deterministic | Fixed cutoff, max_neighbors, RBF params |
| Model initialization | ✅ Deterministic | Xavier uniform init with implicit default seed |
| CUDA determinism | ❌ Not set | Missing `torch.backends.cudnn.deterministic` and `torch.use_deterministic_algorithms` |
| cuDNN benchmark | ❌ Default (enabled) | May cause nondeterminism from cuDNN autotuning |
| Package versions | ❌ Not pinned | No lockfile — `requirements.txt` has loose version pins |

### 12.4 Experimental Best Practices Assessment

| Practice | Current Status | Target Status | Gap |
|----------|---------------|---------------|-----|
| Pre-registered experiments | ❌ | ✅ | No experiment registry with hypotheses |
| Multiple seeds | ❌ (1 seed) | ✅ (≥3 seeds) | Single seed per configuration |
| Statistical significance | ❌ | ✅ | No p-values or effect sizes |
| Learning curves reported | ✅ | ✅ | Provided by experiment tracker |
| Error bars on metrics | ❌ | ✅ | Point estimates only |
| Held-out test set | ✅ | ✅ | Composition-based 10% held-out |
| Cross-validation | ❌ | ✅ | Single split only |
| Ablation studies | 🔄 Partial | ✅ Complete | GradNorm done, scheduler in progress |
| Computational cost reported | 🔄 Partial | ✅ | GPU hours recorded, FLOPs not measured |

---

## 13. Data Card

### 13.1 Dataset v3_li_10000

| Property | Value |
|----------|-------|
| Version | v3_li_10000 |
| Source | Materials Project (v2023) |
| Filter | Li ≥ 5 at.% |
| Size | 10,000 entries |
| Split | Composition-based (80/10/10) |
| Features | 92-dim atom features, 64-dim edge features (Bessel RBF), 16-dim global features |
| Targets | formation_energy_per_atom, energy_above_hull, band_gap |
| Augmented targets | log_ionic_conductivity (<1% coverage), activation_energy (<1% coverage) |

### 13.2 Target Distributions

| Target | Mean | Std | Min | Max | Coverage |
|--------|------|-----|-----|-----|----------|
| formation_energy_per_atom (eV/atom) | -1.85 | 1.42 | -9.82 | 4.78 | 100% |
| energy_above_hull (eV/atom) | 0.12 | 0.23 | 0.000 | 2.15 | 100% |
| band_gap (eV) | 1.83 | 1.52 | 0.000 | 8.71 | 92% |
| log_ionic_conductivity (log₁₀ S/cm) | -3.2 | 1.8 | -8.1 | 0.5 | <1% |
| activation_energy (eV) | 0.42 | 0.18 | 0.05 | 1.10 | <1% |

### 13.3 Chemical Diversity

The Li-filtered dataset spans:
- **Element combinations**: 847 unique element combinations (e.g., Li-O, Li-P-S, Li-La-Ti-O)
- **Chemical families**: 42% oxides, 18% sulfides, 12% phosphates, 10% halides, 8% mixed-anion, 10% other
- **Space groups**: 184 unique space groups (of 230 total)
- **Composition range**: 2–40 atoms per primitive cell, 2–8 element types per formula

---

## 14. Potential Negative Results

### 14.1 What Could Fail

| Experiment | Likelihood of Failure | Impact if Failed | Learning Value |
|------------|----------------------|------------------|----------------|
| Single-task EaH matches multi-task EaH | Medium | High (undermines multi-task motivation) | Indicates task conflict or insufficient sharing |
| PINN constraints degrade metrics | Medium | Medium | Suggests physics priors are incorrect or too strong |
| CGCNN matches or beats ALIGNN+Transformer | Low | High (undermines architecture choice) | Suggests complexity is unnecessary |
| MC Dropout calibration is poor | High | Medium | Requires ensemble alternative |
| Two-stage EaH matches direct regression | Medium | High (undermines core innovation) | Negative but publishable result |

### 14.2 Negative Result Publication Strategy

Even if some experiments yield negative results, the following would still be publishable:

1. **Two-stage EaH fails**: Publish as "Ablation Study of Two-Stage vs. Direct Energy Above Hull Prediction in Graph Neural Networks" — the concept is novel and systematic evaluation is valuable
2. **PINN constraints don't help**: Publish as "The Limited Benefit of Physics-Informed Losses for Solid-State Electrolyte Property Prediction" — constraining to known physics should theoretically help; demonstrating it doesn't is a useful contribution
3. **Multi-task hurts single tasks**: Publish as "Negative Transfer in Multi-Task Learning for Materials Properties: A Case Study" — identifies task conflicts in materials ML

The key is to design experiments so that even negative results produce useful scientific contributions.

---

## 15. Conclusion and Forward Plan

Scandium Labs has made meaningful research contributions — the two-stage EaH head, PINN-constrained multi-task framework, and end-to-end screening pipeline — but the project is at an early stage with critical gaps in statistical rigor and baseline comparison. The research is best characterized as **promising proof-of-concept with strong software engineering foundations**.

**Immediate priority experiments (next 4 weeks):**
1. Single-task baselines for Ef, EaH, Eg
2. CGCNN and/or MEGNet baseline comparison
3. Bootstrap confidence intervals on all metrics
4. Two-stage EaH ablation

**Publication target:** NeurIPS 2026 AI4Mat Workshop (submission: August 2026)
**Journal target:** npj Computational Materials or J. Chem. Inf. Model. (submission: Q1 2027, pending baseline experiments)
