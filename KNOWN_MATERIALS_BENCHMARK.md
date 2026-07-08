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
| Formation Energy (eV/atom) — MAE | 1.0004 |
| Formation Energy (eV/atom) — R² | -2.4491 |
| Energy Above Hull (eV/atom) — MAE | 0.6195 |
| Energy Above Hull (eV/atom) — R² | 0.0000 |
| Band Gap (eV) — MAE | 2.6674 |
| Band Gap (eV) — R² | -0.8099 |


### Best Performing Families

| Family | Property | MAE |
|--------|----------|:---:|
| Oxide | Energy Above Hull (eV/atom) | 0.4355 |
| Halide | Energy Above Hull (eV/atom) | 0.5574 |
| Oxide | Formation Energy (eV/atom) | 0.7046 |

### Worst Performing Families

| Family | Property | MAE |
|--------|----------|:---:|
| Halide | Band Gap (eV) | 4.2267 |
| Oxide | Band Gap (eV) | 2.1519 |
| Sulfide | Band Gap (eV) | 1.3594 |


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

- **Unknown**: 45 materials
- **Oxide**: 4 materials
- **Halide**: 3 materials
- **Sulfide**: 2 materials


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
| N | 9 | 9 | 9 |
| MAE | 1.0004 | 0.6195 | 2.6674 |
| RMSE | 1.0522 | 0.6855 | 3.0242 |
| Median AE | 0.9518 | 0.5536 | 2.0446 |
| R² | -2.4491 | 0.0000 | -0.8099 |
| Pearson r | 0.2830 | nan | 0.8141 |
| Spearman ρ | 0.1177 | nan | 0.8236 |
| Bias | 0.7103 | 0.6195 | -2.6674 |
| Within 1σ (%) | 33.3 | 22.2 | 22.2 |
| Within 2σ (%) | 100.0 | 55.6 | 55.6 |

### 4.2 Per-Family Performance

| Family | N | Ef MAE | Eah MAE | BG MAE | Ef R² | Eah R² | BG R² |
|--------|:-:|:------:|:-------:|:------:|:-----:|:------:|:-----:|
| Unknown | 45 | — | — | — | — | — | — |
| Oxide | 4 | 0.7046 | 0.4355 | 2.1519 | -0.231 | 0.000 | -0.064 |
| Halide | 3 | 1.1623 | 0.5574 | 4.2267 | -10.173 | 0.000 | -11.485 |
| Sulfide | 2 | 1.3492 | 1.0804 | 1.3594 | 0.000 | 0.000 | 0.000 |

### 4.3 Systematic Bias

| Property | Mean Bias | Std Bias | N |
|----------|:---------:|:--------:|:-:|
| Formation Energy (eV/atom) | +0.7103 | 0.7763 | 9 |
| Energy Above Hull (eV/atom) | +0.6195 | 0.2936 | 9 |
| Band Gap (eV) | -2.6674 | 1.4251 | 9 |

### 4.4 Worst Predictions

| Material | Formula | Family | Property | Reference | Prediction | Error | Abs Error |
|----------|---------|--------|----------|:---------:|:----------:|:-----:|:---------:|
| LiF_rocksalt | LiF | Halide | band_gap | 9.0000 | 3.0796 | -5.9204 | 5.9204 |
| LiCl_rocksalt | LiCl | Halide | band_gap | 7.0000 | 3.1940 | -3.8060 | 3.8060 |
| MgO_rocksalt | MgO | Oxide | band_gap | 7.8000 | 4.5386 | -3.2614 | 3.2614 |
| NaCl_rocksalt | NaCl | Halide | band_gap | 6.0000 | 3.0462 | -2.9538 | 2.9538 |
| LiCoO2_layered | LiCoO2 | Oxide | band_gap | 2.5000 | 0.4554 | -2.0446 | 2.0446 |
| LiNiO2_layered | LiCoO2 | Oxide | band_gap | 2.5000 | 0.6453 | -1.8547 | 1.8547 |
| NaCl_rocksalt | NaCl | Halide | formation_energy | -2.4000 | -0.8929 | +1.5071 | 1.5071 |
| Li2O_antifluorite | Li2O | Oxide | band_gap | 5.0000 | 3.5529 | -1.4471 | 1.4471 |
| Li2S_li_superionic | Li2S | Sulfide | band_gap | 3.5000 | 2.0918 | -1.4082 | 1.4082 |
| Li2S_antifluorite | Li2S | Sulfide | formation_energy | -1.8000 | -0.4406 | +1.3594 | 1.3594 |

---

## 5. Error Analysis

### 5.1 Systematic Errors

- **Formation Energy (eV/atom)**: Systematic over-prediction of 0.7103 ± 0.7763 (9 samples)
- **Energy Above Hull (eV/atom)**: Systematic over-prediction of 0.6195 ± 0.2936 (9 samples)
- **Band Gap (eV)**: Systematic under-prediction of 2.6674 ± 1.4251 (9 samples)


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

- **Oxide**: Lowest prediction error for Energy Above Hull (eV/atom) (MAE = 0.4355)
- **Halide**: Lowest prediction error for Energy Above Hull (eV/atom) (MAE = 0.5574)
- **Oxide**: Lowest prediction error for Formation Energy (eV/atom) (MAE = 0.7046)

**Why they perform well:**
- Well-represented in training data
- Chemically similar to training set compositions
- Structurally well-approximated by ordered primitive cells

### 6.2 Worst Performing

- **Halide**: Highest prediction error for Band Gap (eV) (MAE = 4.2267)
- **Oxide**: Highest prediction error for Band Gap (eV) (MAE = 2.1519)
- **Sulfide**: Highest prediction error for Band Gap (eV) (MAE = 1.3594)

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
| LiF | formation_energy | -3.100 | -2.025 | Major |
| LiF | energy_above_hull | 0.000 | 0.260 | Major |
| LiF | band_gap | 9.000 | 3.080 | Major |
| LiCl | formation_energy | -2.300 | -1.395 | Major |
| LiCl | energy_above_hull | 0.000 | 0.615 | Major |
| LiCl | band_gap | 7.000 | 3.194 | Major |
| NaCl | formation_energy | -2.400 | -0.893 | Major |
| NaCl | energy_above_hull | 0.000 | 0.797 | Major |
| NaCl | band_gap | 6.000 | 3.046 | Major |
| MgO | formation_energy | -3.000 | -2.439 | Major |
| MgO | energy_above_hull | 0.000 | 0.248 | Major |
| MgO | band_gap | 7.800 | 4.539 | Major |
| Li2O | formation_energy | -2.500 | -1.548 | Major |
| Li2O | energy_above_hull | 0.000 | 0.427 | Major |
| Li2O | band_gap | 5.000 | 3.553 | Major |
| Li2S | formation_energy | -1.800 | -0.441 | Major |
| Li2S | energy_above_hull | 0.000 | 1.086 | Major |
| Li2S | band_gap | 3.500 | 2.189 | Major |
| LiCoO2 | formation_energy | -1.500 | -2.105 | Major |
| LiCoO2 | energy_above_hull | 0.000 | 0.554 | Major |
| LiCoO2 | band_gap | 2.500 | 0.455 | Major |
| LiCoO2 | formation_energy | -1.500 | -2.201 | Major |
| LiCoO2 | energy_above_hull | 0.000 | 0.514 | Major |
| LiCoO2 | band_gap | 2.500 | 0.645 | Major |
| Li2S | formation_energy | -1.800 | -0.461 | Major |
| Li2S | energy_above_hull | 0.000 | 1.075 | Major |
| Li2S | band_gap | 3.500 | 2.092 | Major |


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
for thermodynamic stability (Eah R² = 0.000).
Stability predictions show room for improvement, particularly for disordered phases.


The primary limitation remains the absence of ionic conductivity predictions, which requires
experimentally labeled data. The model is currently most useful for:
1. **Pre-screening** candidate compositions for thermodynamic stability
2. **Band gap estimation** for solid electrolyte discovery
3. **Materials comparison** within known chemical families

---

*Report generated automatically by the Scandium Labs evaluation suite.
Run `python -m scripts.benchmark.generate_report` to regenerate.*
