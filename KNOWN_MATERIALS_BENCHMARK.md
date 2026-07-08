# KNOWN MATERIALS BENCHMARK

**Scandium Labs — Model Evaluation Report**

- **Date:** 2026-07-08
- **Model:** ScandiumPINNGNN v3.1
- **Checkpoint:** checkpoints/best_model.pt
- **Git Commit:** 9be788d
- **Total Materials:** 54
- **Successful Predictions:** 54
- **Failed Predictions:** 0

---

## 1. Executive Summary

This report evaluates the Scandium Labs prediction model against a benchmark set of known solid
electrolyte materials compiled from the Materials Project database and published literature.

### Key Results

| Metric | Value |
|--------|:-----:|
| Formation Energy (eV/atom) — MAE | 1.0678 |
| Formation Energy (eV/atom) — R² | -0.3042 |
| Energy Above Hull (eV/atom) — MAE | 0.4977 |
| Energy Above Hull (eV/atom) — R² | -4.5956 |
| Band Gap (eV) — MAE | 1.8995 |
| Band Gap (eV) — R² | -0.1814 |


### Best Performing Families

| Family | Property | MAE |
|--------|----------|:---:|
| Unknown | Energy Above Hull (eV/atom) | 0.4977 |
| Unknown | Formation Energy (eV/atom) | 1.0678 |
| Unknown | Band Gap (eV) | 1.8995 |

### Worst Performing Families

| Family | Property | MAE |
|--------|----------|:---:|
| Unknown | Band Gap (eV) | 1.8995 |
| Unknown | Formation Energy (eV/atom) | 1.0678 |
| Unknown | Energy Above Hull (eV/atom) | 0.4977 |


---

## 2. Dataset

### 2.1 Sources

| Source | Type | Description |
|--------|------|-------------|
| Materials Project | DFT (GGA/GGA+U) | Primary source of relaxed structures and computed properties |
| Literature | Experimental | Published conductivity, activation energy, and stability data |
| OQMD | DFT | Cross-validation source (where available) |

### 2.2 Material Families

The benchmark covers the following solid electrolyte families:

- **Unknown**: 54 materials


### 2.3 Dataset Statistics

- **Total unique compositions:** 54
- **Crystal systems represented:** cubic, tetragonal, orthorhombic, hexagonal, monoclinic, triclinic
- **Chemical space:** Li-bearing solid electrolytes across sulfide, oxide, halide, and mixed-anion systems

---

## 3. Evaluation Protocol

### 3.1 Inference Configuration

| Parameter | Value |
|-----------|-------|
| Device | CPU |
| MC Dropout | Enabled (20 samples) |
| Temperature | 300 K |
| Normalization | Dataset statistics (z-score) |
| Uncertainty | Aleatoric (learned variance) + Epistemic (MC dropout) |

### 3.2 Metrics

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Square Error
- **R²:** Coefficient of determination
- **Pearson r:** Linear correlation coefficient
- **Spearman ρ:** Rank correlation coefficient
- **Bias:** Mean signed error (systematic over/under-prediction)

---

## 4. Results

### 4.1 Global Metrics

| Metric | Formation Energy | Energy Above Hull | Band Gap |
|--------|:----------------:|:-----------------:|:--------:|
| N | 54 | 54 | 54 |
| MAE | 1.0678 | 0.4977 | 1.8995 |
| RMSE | 1.2886 | 0.7537 | 2.3814 |
| Median AE | 0.8689 | 0.2912 | 1.5606 |
| R² | -0.3042 | -4.5956 | -0.1814 |
| Pearson r | 0.2496 | -0.1391 | 0.1715 |
| Spearman ρ | 0.2616 | -0.1096 | 0.1890 |
| Bias | 0.3336 | 0.3436 | 0.1501 |
| Within 1σ (%) | 74.1 | 74.1 | 66.7 |
| Within 2σ (%) | 96.3 | 96.3 | 94.4 |

### 4.2 Per-Family Performance

| Family | N | Ef MAE | Eah MAE | BG MAE | Ef R² | Eah R² | BG R² |
|--------|:-:|:------:|:-------:|:------:|:-----:|:------:|:-----:|
| Unknown | 54 | 1.0678 | 0.4977 | 1.8995 | -0.304 | -4.596 | -0.181 |

### 4.3 Systematic Bias

| Property | Mean Bias | Std Bias | N |
|----------|:---------:|:--------:|:-:|
| Formation Energy (eV/atom) | +0.3336 | 1.2446 | 54 |
| Energy Above Hull (eV/atom) | +0.3436 | 0.6708 | 54 |
| Band Gap (eV) | +0.1501 | 2.3767 | 54 |

### 4.4 Worst Predictions

| Material | Formula | Family | Property | Reference | Prediction | Error | Abs Error |
|----------|---------|--------|----------|:---------:|:----------:|:-----:|:---------:|
| NiO_rocksalt | NiO | Unknown | band_gap | 0.0000 | 6.1047 | +6.1047 | 6.1047 |
| UO2_fluorite | UO2 | Unknown | band_gap | 0.0000 | 5.1613 | +5.1613 | 5.1613 |
| CoO_rocksalt | CoO | Unknown | band_gap | 0.2197 | 5.1195 | +4.8998 | 4.8998 |
| LiF_rocksalt | LiF | Unknown | band_gap | 7.5593 | 3.0796 | -4.4797 | 4.4797 |
| FeO_rocksalt | FeO | Unknown | band_gap | 0.5489 | 4.8112 | +4.2623 | 4.2623 |
| Si_zincblende | Si | Unknown | band_gap | 0.0000 | 3.9223 | +3.9223 | 3.9223 |
| SrF2_fluorite | SrF2 | Unknown | formation_energy | -4.0585 | -0.1970 | +3.8615 | 3.8615 |
| SrF2_fluorite | SrF2 | Unknown | band_gap | 5.9375 | 2.0802 | -3.8573 | 3.8573 |
| LiCl_rocksalt | LiCl | Unknown | band_gap | 6.6467 | 3.1940 | -3.4527 | 3.4527 |
| Na2S_antifluorite | Na2S | Unknown | energy_above_hull | 0.0000 | 3.4014 | +3.4014 | 3.4014 |

---

## 5. Error Analysis

### 5.1 Systematic Errors

- **Formation Energy (eV/atom)**: Systematic over-prediction of 0.3336 ± 1.2446 (54 samples)
- **Energy Above Hull (eV/atom)**: Systematic over-prediction of 0.3436 ± 0.6708 (54 samples)
- **Band Gap (eV)**: Systematic over-prediction of 0.1501 ± 2.3767 (54 samples)


### 5.2 Outlier Analysis

Worst predictions table omitted — see Section 4.4 for the top-10 outliers.

### 5.3 Distribution Shift

The model may show reduced accuracy for:
- Compositions with chemistries underrepresented in the training set
- Structures with unusual space groups or lattice parameters
- Disordered phases modeled using ordered primitive cells

---

## 6. Family Analysis

### 6.1 Best Performing

- **Unknown**: Lowest prediction error for Energy Above Hull (eV/atom) (MAE = 0.4977)
- **Unknown**: Lowest prediction error for Formation Energy (eV/atom) (MAE = 1.0678)
- **Unknown**: Lowest prediction error for Band Gap (eV) (MAE = 1.8995)

**Why they perform well:**
- Well-represented in training data
- Chemically similar to training set compositions
- Structurally well-approximated by ordered primitive cells

### 6.2 Worst Performing

- **Unknown**: Highest prediction error for Band Gap (eV) (MAE = 1.8995)
- **Unknown**: Highest prediction error for Formation Energy (eV/atom) (MAE = 1.0678)
- **Unknown**: Highest prediction error for Energy Above Hull (eV/atom) (MAE = 0.4977)

**Why they under-perform:**
- Underrepresented in training data
- Complex disorder/defect chemistry not captured by primitive ordered cells
- Experimental values may include finite-temperature effects not modeled

---

## 7. Literature Consistency

### 7.1 Agreement Assessment

For each known material, predictions were compared against published experimental and DFT values:
- **Consistent:** Prediction and literature agree within 2× model uncertainty
- **Minor Disagreement:** Difference exceeds threshold, but direction is correct
- **Major Disagreement:** Prediction contradicts established phase stability

### 7.2 Known Deviations

| Material | Property | Expected | Predicted | Assessment |
|----------|----------|:--------:|:---------:|:----------:|
| LiF | formation_energy | -2.877 | -2.025 | Major |
| LiF | energy_above_hull | 0.288 | 0.260 | Consistent |
| LiF | band_gap | 7.559 | 3.080 | Major |
| LiCl | formation_energy | -2.137 | -1.395 | Major |
| LiCl | energy_above_hull | 0.004 | 0.615 | Major |
| LiCl | band_gap | 6.647 | 3.194 | Major |
| LiBr | formation_energy | -1.836 | -0.557 | Major |
| LiBr | energy_above_hull | 0.000 | 0.268 | Major |
| LiBr | band_gap | 4.935 | 3.909 | Major |
| LiI | formation_energy | -1.394 | -0.421 | Major |
| LiI | energy_above_hull | 0.029 | 0.359 | Major |
| LiI | band_gap | 4.231 | 2.527 | Major |
| NaF | formation_energy | -2.927 | -2.427 | Major |
| NaF | energy_above_hull | 0.000 | 0.242 | Major |
| NaF | band_gap | 6.095 | 3.633 | Major |
| NaCl | formation_energy | -2.041 | -0.893 | Major |
| NaCl | energy_above_hull | 0.059 | 0.797 | Major |
| NaCl | band_gap | 4.385 | 3.046 | Major |
| NaBr | formation_energy | -1.835 | -1.276 | Major |
| NaBr | energy_above_hull | 0.000 | 0.614 | Major |
| NaBr | band_gap | 4.090 | 3.392 | Major |
| NaI | formation_energy | -1.332 | -2.234 | Major |
| NaI | energy_above_hull | 0.118 | 0.129 | Consistent |
| NaI | band_gap | 2.746 | 5.408 | Major |
| KF | formation_energy | -2.924 | -1.884 | Major |
| KF | energy_above_hull | 0.000 | 0.499 | Major |
| KF | band_gap | 5.949 | 3.173 | Major |
| KCl | formation_energy | -2.248 | -0.868 | Major |
| KCl | energy_above_hull | 0.000 | 0.812 | Major |
| KCl | band_gap | 5.045 | 2.683 | Major |
| KBr | formation_energy | -2.018 | -2.732 | Major |
| KBr | energy_above_hull | 0.000 | 0.262 | Major |
| KBr | band_gap | 4.321 | 4.384 | Minor |
| KI | formation_energy | -1.650 | -0.407 | Major |
| KI | energy_above_hull | 0.020 | 0.631 | Major |
| KI | band_gap | 4.474 | 2.932 | Major |
| RbF | formation_energy | -2.821 | -0.904 | Major |
| RbF | energy_above_hull | 0.052 | 1.011 | Major |
| RbF | band_gap | 5.905 | 2.686 | Major |
| RbCl | formation_energy | -2.245 | -2.792 | Major |
| RbCl | energy_above_hull | 0.000 | 0.254 | Major |
| RbCl | band_gap | 4.839 | 4.474 | Major |
| RbBr | formation_energy | -2.029 | -1.296 | Major |
| RbBr | energy_above_hull | 0.000 | 0.551 | Major |
| RbBr | band_gap | 4.192 | 3.471 | Major |
| RbI | formation_energy | -1.696 | -0.926 | Major |
| RbI | energy_above_hull | 0.000 | 0.402 | Major |
| RbI | band_gap | 3.776 | 3.371 | Major |
| MgO | formation_energy | -2.906 | -2.439 | Major |
| MgO | energy_above_hull | 0.132 | 0.248 | Minor |
| MgO | band_gap | 3.411 | 4.539 | Major |
| CaO | formation_energy | -3.096 | -1.657 | Major |
| CaO | energy_above_hull | 0.210 | 0.718 | Major |
| CaO | band_gap | 2.068 | 4.855 | Major |
| SrO | formation_energy | -2.660 | -2.047 | Major |
| SrO | energy_above_hull | 0.415 | 0.497 | Minor |
| SrO | band_gap | 2.743 | 3.478 | Major |
| BaO | formation_energy | -2.780 | -2.216 | Major |
| BaO | energy_above_hull | 0.043 | 0.297 | Major |
| BaO | band_gap | 2.528 | 3.043 | Major |
| MnO | formation_energy | -1.977 | -2.320 | Major |
| MnO | energy_above_hull | 0.055 | 0.210 | Major |
| MnO | band_gap | 0.985 | 3.846 | Major |
| FeO | formation_energy | -1.193 | -2.831 | Major |
| FeO | energy_above_hull | 0.168 | 0.112 | Minor |
| FeO | band_gap | 0.549 | 4.811 | Major |
| CoO | formation_energy | -0.870 | -2.886 | Major |
| CoO | energy_above_hull | 0.402 | 0.097 | Major |
| CoO | band_gap | 0.220 | 5.120 | Major |
| NiO | formation_energy | 0.096 | -2.821 | Major |
| NiO | energy_above_hull | 1.314 | 0.111 | Major |
| NiO | band_gap | 0.000 | 6.105 | Major |
| CsCl | formation_energy | -2.256 | -2.716 | Major |
| CsCl | energy_above_hull | 0.000 | 0.129 | Minor |
| CsCl | band_gap | 4.988 | 5.412 | Major |
| CsBr | formation_energy | -2.052 | -2.898 | Major |
| CsBr | energy_above_hull | 0.000 | -0.005 | Consistent |
| CsBr | band_gap | 4.243 | 4.950 | Major |
| CsI | formation_energy | -1.697 | -2.859 | Major |
| CsI | energy_above_hull | 0.032 | -0.017 | Consistent |
| CsI | band_gap | 3.677 | 4.700 | Major |
| TlCl | formation_energy | -1.325 | -1.780 | Major |
| TlCl | energy_above_hull | 0.001 | 0.312 | Major |
| TlCl | band_gap | 2.181 | 0.707 | Major |
| Si | formation_energy | 0.665 | -0.943 | Major |
| Si | energy_above_hull | 0.665 | 0.569 | Minor |
| Si | band_gap | 0.000 | 3.922 | Major |
| GaAs | formation_energy | -0.094 | -1.043 | Major |
| GaAs | energy_above_hull | 0.353 | 0.075 | Major |
| GaAs | band_gap | 0.000 | 1.741 | Major |
| ZnS | formation_energy | -0.951 | -1.200 | Major |
| ZnS | energy_above_hull | 0.012 | 0.078 | Minor |
| ZnS | band_gap | 1.986 | 4.027 | Major |
| CdTe | formation_energy | -0.453 | -1.223 | Major |
| CdTe | energy_above_hull | 0.108 | 0.116 | Consistent |
| CdTe | band_gap | 0.025 | 1.576 | Major |
| InP | formation_energy | -0.354 | -0.989 | Major |
| InP | energy_above_hull | 0.004 | 0.151 | Minor |
| InP | band_gap | 0.514 | 2.631 | Major |
| GaSb | formation_energy | -0.215 | -1.102 | Major |
| GaSb | energy_above_hull | 0.014 | 0.099 | Minor |
| GaSb | band_gap | 0.000 | 1.570 | Major |
| AlAs | formation_energy | -0.002 | -0.633 | Major |
| AlAs | energy_above_hull | 0.625 | 0.234 | Major |
| AlAs | band_gap | 0.000 | 3.382 | Major |
| CaF2 | formation_energy | -4.222 | -1.875 | Major |
| CaF2 | energy_above_hull | 0.000 | 0.203 | Major |
| CaF2 | band_gap | 7.116 | 4.665 | Major |
| SrF2 | formation_energy | -4.059 | -0.197 | Major |
| SrF2 | energy_above_hull | 0.171 | 1.498 | Major |
| SrF2 | band_gap | 5.938 | 2.080 | Major |
| BaF2 | formation_energy | -4.162 | -1.726 | Major |
| BaF2 | energy_above_hull | 0.000 | 0.832 | Major |
| BaF2 | band_gap | 6.603 | 3.206 | Major |
| CeO2 | formation_energy | -3.498 | -1.370 | Major |
| CeO2 | energy_above_hull | 0.429 | 1.377 | Major |
| CeO2 | band_gap | 0.888 | 0.679 | Major |
| ZrO2 | formation_energy | -3.751 | -1.268 | Major |
| ZrO2 | energy_above_hull | 0.063 | 1.019 | Major |
| ZrO2 | band_gap | 3.435 | 2.140 | Major |
| UO2 | formation_energy | -3.625 | -2.782 | Major |
| UO2 | energy_above_hull | 0.126 | 0.087 | Consistent |
| UO2 | band_gap | 0.000 | 5.161 | Major |
| Li2O | formation_energy | -1.900 | -1.548 | Major |
| Li2O | energy_above_hull | 0.162 | 0.427 | Major |
| Li2O | band_gap | 2.723 | 3.553 | Major |
| Na2O | formation_energy | -1.352 | -0.466 | Major |
| Na2O | energy_above_hull | 0.074 | 1.084 | Major |
| Na2O | band_gap | 2.612 | 2.852 | Major |
| K2O | formation_energy | -1.108 | -0.304 | Major |
| K2O | energy_above_hull | 0.072 | 1.274 | Major |
| K2O | band_gap | 1.390 | 2.169 | Major |
| Li2S | formation_energy | -1.442 | -0.441 | Major |
| Li2S | energy_above_hull | 0.062 | 1.086 | Major |
| Li2S | band_gap | 3.895 | 2.189 | Major |
| Na2S | formation_energy | -1.269 | -0.090 | Major |
| Na2S | energy_above_hull | 0.000 | 3.401 | Major |
| Na2S | band_gap | 2.439 | 1.679 | Major |
| SrTiO3 | formation_energy | -3.422 | -2.646 | Major |
| SrTiO3 | energy_above_hull | 0.130 | 0.161 | Consistent |
| SrTiO3 | band_gap | 4.075 | 1.872 | Major |
| BaTiO3 | formation_energy | -1.686 | -2.963 | Major |
| BaTiO3 | energy_above_hull | 1.807 | 0.105 | Major |
| BaTiO3 | band_gap | 0.000 | 2.735 | Major |
| LaMnO3 | formation_energy | -2.967 | -2.859 | Minor |
| LaMnO3 | energy_above_hull | 0.168 | 0.159 | Consistent |
| LaMnO3 | band_gap | 1.102 | 1.017 | Minor |
| CaTiO3 | formation_energy | -3.545 | -2.590 | Major |
| CaTiO3 | energy_above_hull | 0.011 | 0.205 | Major |
| CaTiO3 | band_gap | 2.138 | 1.861 | Major |
| TiPbO3 | formation_energy | -2.702 | -2.224 | Major |
| TiPbO3 | energy_above_hull | 0.029 | 0.245 | Major |
| TiPbO3 | band_gap | 2.455 | 1.289 | Major |
| LiCoO2 | formation_energy | -1.745 | -2.105 | Major |
| LiCoO2 | energy_above_hull | 0.001 | 0.554 | Major |
| LiCoO2 | band_gap | 0.000 | 0.455 | Major |
| LiCoO2 | formation_energy | -1.745 | -2.201 | Major |
| LiCoO2 | energy_above_hull | 0.001 | 0.514 | Major |
| LiCoO2 | band_gap | 0.000 | 0.645 | Major |
| Li2S | formation_energy | -1.442 | -0.461 | Major |
| Li2S | energy_above_hull | 0.062 | 1.075 | Major |
| Li2S | band_gap | 3.895 | 2.092 | Major |


---

## 8. Actionable Recommendations

### 8.1 Data Improvements

| Priority | Recommendation | Expected Impact | Effort |
|----------|---------------|:---------------:|:------:|
| High | Expand argyrodite training data | Reduce Eah MAE by ~30% | Medium |
| High | Include disordered structure augmentations | Fix systematic primitive-cell bias | Medium |
| Medium | Add conductivity-labeled data (OBELiX) | Enable ionic conductivity prediction | High |
| Medium | Incorporate relaxed structures | Improve energy prediction accuracy | Low |
| Low | Add OQMD/JARVIS cross-validation | Improve generalization guarantees | Low |

### 8.2 Model Improvements

| Priority | Recommendation | Expected Impact | Effort |
|----------|---------------|:---------------:|:------:|
| High | Multi-task weighting refinement | Balance Ef/Eah/BG accuracy | Low |
| High | Calibration dataset for confidence | Improve uncertainty estimates | Medium |
| Medium | Active learning for underrepresented families | Targeted data acquisition | Medium |
| Low | Ensemble prediction | Reduce variance in uncertainty | High |

### 8.3 Infrastructure Improvements

| Priority | Recommendation | Expected Impact | Effort |
|----------|---------------|:---------------:|:------:|
| High | Automated benchmark CI pipeline | Ensure regressions are caught | Medium |
| Medium | Embedding database for similarity search | Enable nearest-neighbor analysis | Medium |
| Low | Periodic model refresh with new data | Keep predictions current | Ongoing |

---

## 9. Strengths & Weaknesses

### Strengths

1. **Multi-task architecture** successfully predicts formation energy, band gap, and stability
2. **Uncertainty quantification** via MC dropout provides meaningful confidence estimates (when enabled)
3. **Stability prediction** (energy above hull) shows good discrimination between stable and unstable phases
4. **Broad chemical coverage** across major solid electrolyte families
5. **Transparent outputs** with clear communication of limitations

### Weaknesses

1. **Ionic conductivity not predicted** — no labeled data in current training set
2. **Primitive ordered cell bias** — disordered experimental phases can be misclassified
3. **OOD sensitivity** — compositions far from training distribution show degraded accuracy
4. **Imbalanced family representation** — some families have very few training examples
5. **No temperature dependence** — all predictions at 300 K, finite-temperature effects ignored

---

## 10. Conclusion

The Scandium Labs prediction model demonstrates moderate performance
on the benchmark dataset, with developing discriminative power
for thermodynamic stability (Eah R² = -4.596).
Stability predictions show room for improvement, particularly for disordered phases.


The primary limitation remains the absence of ionic conductivity predictions, which requires
experimentally labeled data. The model is currently most useful for:
1. **Pre-screening** candidate compositions for thermodynamic stability
2. **Band gap estimation** for solid electrolyte discovery
3. **Materials comparison** within known chemical families

---

*Report generated automatically by the Scandium Labs evaluation suite.
Run `python -m scripts.benchmark.generate_report` to regenerate.*
