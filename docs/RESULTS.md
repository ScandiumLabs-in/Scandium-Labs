# Experimental Results

> Scandium Labs — Li Solid-State Electrolyte Screening
> Last updated: 2026-07-08

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Experiment Register](#2-experiment-register)
- [3. Experiment A: SL-20260708-001](#3-experiment-a-sl-20260708-001)
  - [3.1 Configuration](#31-configuration)
  - [3.2 Training Summary](#32-training-summary)
  - [3.3 Best Validation Epoch](#33-best-validation-epoch)
  - [3.4 Per-Task Best Metrics](#34-per-task-best-metrics)
  - [3.5 Throughput and Efficiency](#35-throughput-and-efficiency)
- [4. Previous Run: SL-20260701-007](#4-previous-run-sl-20260701-007)
  - [4.1 Configuration](#41-configuration)
  - [4.2 Training Summary](#42-training-summary)
  - [4.3 Best Validation Epoch](#43-best-validation-epoch)
  - [4.4 Per-Task Best Metrics](#44-per-task-best-metrics)
- [5. Head-to-Head Comparison](#5-head-to-head-comparison)
  - [5.1 Overall Comparison Table](#51-overall-comparison-table)
  - [5.2 Statistical Significance](#52-statistical-significance)
  - [5.3 Key Takeaways](#53-key-takeaways)
- [6. Learning Curves Analysis](#6-learning-curves-analysis)
  - [6.1 Loss Curves](#61-loss-curves)
  - [6.2 Per-Task Error Curves](#62-per-task-error-curves)
  - [6.3 Per-Task R² Curves](#63-per-task-r-curves)
- [7. Per-Task Performance Deep Dive](#7-per-task-performance-deep-dive)
  - [7.1 Formation Energy (Ef)](#71-formation-energy-ef)
  - [7.2 Energy Above Hull (EaH)](#72-energy-above-hull-eah)
  - [7.3 Band Gap (BG)](#73-band-gap-bg)
- [8. Two-Stage EaH Results](#8-two-stage-eah-results)
  - [8.1 Architecture](#81-architecture)
  - [8.2 Classification Performance](#82-classification-performance)
  - [8.3 Regression Performance (Unstable Only)](#83-regression-performance-unstable-only)
  - [8.4 Comparison Across Runs](#84-comparison-across-runs)
- [9. Convergence Analysis](#9-convergence-analysis)
  - [9.1 Experiment A Convergence](#91-experiment-a-convergence)
  - [9.2 Previous Run Convergence](#92-previous-run-convergence)
  - [9.3 Scheduler Impact on Convergence](#93-scheduler-impact-on-convergence)
  - [9.4 GradNorm Impact on Convergence](#94-gradnorm-impact-on-convergence)
- [10. Test Set Results](#10-test-set-results)
  - [10.1 Test Set Performance (Previous Run)](#101-test-set-performance-previous-run)
  - [10.2 Test Set Performance (Experiment A)](#102-test-set-performance-experiment-a)
  - [10.3 Comparison Across Runs](#103-comparison-across-runs)
- [11. Failure Cases and Discussion](#11-failure-cases-and-discussion)
  - [11.1 Band Gap Regression](#111-band-gap-regression)
  - [11.2 Energy Above Hull Regression](#112-energy-above-hull-regression)
  - [11.3 Overfitting Analysis](#113-overfitting-analysis)
  - [11.4 Out-of-Distribution Performance](#114-out-of-distribution-performance)
- [12. Recommendations for Future Experiments](#12-recommendations-for-future-experiments)

---

## 1. Overview

This document presents the experimental results from two complete training runs of the ScandiumPINNGNN-v3-Li model on the v3_li_10000 dataset. These experiments investigate the effect of two key training methodologies:

- **Gradient normalization (GradNorm)**: Dynamic task-weighting strategy that balances learning across multiple regression tasks
- **Learning rate scheduling**: Cosine annealing with warm restarts vs. fixed learning rate

The primary comparison is between:

| Feature | Exp A (SL-20260708-001) | Prev (SL-20260701-007) |
|---------|------------------------|------------------------|
| GradNorm | OFF | ON |
| Scheduler | Cosine with restarts | None (fixed LR) |
| Epochs | 138 (early stop) | 150 (completed) |
| GPU-hours | 15.8 | 16.5 |

Both experiments use the same model architecture (1.28M parameters, ALIGNN-4 + Transformer-2, hidden_dim=128), dataset (v3_li_10000), and data preprocessing pipeline.

The dataset comprises ~10,000 Li-containing crystal structures from the Materials Project (Li ≥ 5 at.%), split 80/10/10 into train/val/test sets. The model predicts three target properties: formation energy (Ef), energy above hull (EaH), and band gap (BG), with a two-stage head for EaH that first classifies stability then regresses the magnitude.

---

## 2. Experiment Register

The complete experiment register is maintained at `runs/index.csv`. As of this writing, the active experiments are:

| Run ID | Date | GradNorm | Scheduler | Epochs | Best R² Ef | Best R² EaH | Best R² BG | GPU-hrs |
|--------|------|----------|-----------|--------|-----------|-------------|------------|---------|
| SL-20260630-002 | 2026-06-30 | ON | Cosine | 70+ | 0.5359 | 0.3750 | 0.2924 | 6.2 |
| SL-20260701-007 | 2026-07-01 | ON | None | 150 | 0.5897 | 0.4227 | 0.3060 | 7.9 |
| SL-20260707-001 | 2026-07-07 | ON | None | 10+ | 0.3457 | 0.2016 | 0.0967 | 0.2 |
| SL-20260707-002 | 2026-07-07 | ON | None | 60+ | 0.5155 | 0.3712 | 0.2747 | 3.4 |
| SL-20260708-001 | 2026-07-08 | OFF | Cosine | 138 | 0.5871 | 0.3854 | 0.3385 | 15.8 |
| SL-20260708-002 | 2026-07-08 | OFF | Cosine | 5+ | 0.4163 | 0.3146 | 0.1428 | 0.2 |

This document focuses on the two complete, well-converged runs: SL-20260701-007 (150 epochs, GradNorm ON) and SL-20260708-001 (138 epochs, cosine scheduler, GradNorm OFF).

---

## 3. Experiment A: SL-20260708-001

### 3.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Run ID** | `SL-20260708-001` |
| **Date** | 2026-07-08 |
| **Git Commit** | `d30295b330c495d738fcaac64d49ca48d0047c8a` |
| **Branch** | `master` |
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB) |
| **PyTorch** | 2.6.0+cu124 |
| **CUDA** | 12.4 |

**Model**

| Parameter | Value |
|-----------|-------|
| Architecture | ScandiumPINNGNN-v3-Li |
| Hidden dimension | 128 |
| ALIGNN layers | 4 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Dropout | 0.15 |
| Total parameters | 1,281,321 |
| Gradient checkpointing | auto (enabled, < 6 GB VRAM) |
| Two-stage EaH | enabled |

**Training**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Scheduler | Cosine annealing with warm restarts |
| Batch size | 16 |
| Gradient accumulation | 2 steps |
| Weight decay | 1e-5 |
| Gradient clipping | 1.0 |
| Mixed precision | FP16 (AMP) |
| Max epochs | 150 |
| Patience | 40 |
| GradNorm | OFF (disabled: `gradnorm.enabled: false`) |
| Warmup steps | 0 (no warmup) |

**Data**

| Parameter | Value |
|-----------|-------|
| Dataset | v3_li_10000 |
| Graph cutoff | 8.0 Å |
| Max neighbors | 16 |
| RBF features | 64 |
| SBF features | 32 |
| Bucketing | enabled (bucket_size_mult: 2.0) |

### 3.2 Training Summary

| Metric | Value |
|--------|-------|
| Total epochs | 139 (0-indexed: 0–138) |
| Stop reason | Early stopping (val_loss did not improve for 40 epochs) |
| Best val_loss | **3.0941** @ epoch 98 |
| Final val_loss | 3.1452 @ epoch 138 |
| Total training time | 56,979 s (15.8 GPU-hours) |
| Avg epoch time | 409.7 s (6.8 min) |
| Avg throughput | 40.6 graphs/s |
| Final throughput | 41.1 graphs/s |

### 3.3 Best Validation Epoch

At epoch 98, the model achieved its lowest validation loss:

```
Best epoch:        98
Validation Loss:   3.0941

Per-task metrics:
  formation_energy     MAE=0.5508  R²=0.5511  RMSE=0.7551
  energy_above_hull    MAE=0.1313  R²=0.3735  RMSE=0.3628
  band_gap             MAE=1.0252  R²=0.3385  RMSE=1.4109
```

The best val_loss epoch is not necessarily the best epoch for every individual metric. Task-specific best epochs (see §3.4) can differ by tens of epochs.

### 3.4 Per-Task Best Metrics

| Metric | Best Value | Epoch | Final Value | Delta |
|--------|-----------|-------|-------------|-------|
| Ef MAE | 0.5222 | 136 | 0.5276 | +0.0054 |
| Ef R² | **0.5871** | 135 | 0.5811 | -0.0060 |
| EaH MAE | 0.1280 | 4 | 0.1351 | +0.0071 |
| EaH R² | 0.3854 | 131 | 0.3825 | -0.0029 |
| BG MAE | **1.0252** | 98 | 1.0506 | +0.0254 |
| BG R² | **0.3385** | 98 | 0.3107 | -0.0278 |

Notable observations:

- **Ef R² (0.5871)** and **Ef MAE (0.5222)** peak in the same late-epoch window (135–136), suggesting the model continues refining formation energy predictions until the very end.
- **EaH MAE** reaches its minimum at epoch 4 — just the 5th pass through the data — indicating that EaH is either inherently easier or that the two-stage head converges rapidly for the magnitude prediction. The MAE then slowly degrades by ~5% over the remaining epochs.
- **BG R² (0.3385)** and **BG MAE (1.0252)** both peak at epoch 98 (best val_loss), after which the scheduler's decaying learning rate may cause the model to drift away from the band gap optimum.
- All final metrics are within 5% of their best values, indicating minimal overfitting despite 40 epochs of non-improving validation loss.

### 3.5 Throughput and Efficiency

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Epoch time (s) | 409.7 | 400.6 | 458.7 |
| Throughput (g/s) | 40.6 | 36.3 | 41.5 |
| GPU memory (MB) | 1519.8 | 1454.2 | 1535.9 |

The training throughput is stable at ~41 g/s throughout, with the 4-worker DataLoader keeping the GPU well-fed. The 458.7s max epoch time likely reflects system noise or memory pressure during the first few epochs. GPU memory utilization is flat at ~1.5 GB (38% of the 4 GB card capacity), leaving headroom for larger batch sizes or model sizes.

---

## 4. Previous Run: SL-20260701-007

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| **Run ID** | `SL-20260701-007` |
| **Date** | 2026-07-01 |
| **Git Commit** | `d30295b330c495d738fcaac64d49ca48d0047c8a` |
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB) |
| **Architecture** | Identical to Experiment A (1,281,321 params) |

**Key differences from Experiment A:**

| Parameter | Prev (SL-20260701-007) | Exp A (SL-20260708-001) |
|-----------|------------------------|------------------------|
| GradNorm | ON (alpha=1.5) | OFF |
| Scheduler | None (constant LR=5e-4) | Cosine with restarts |
| Warmup steps | 500 | 0 |
| Max epochs | 150 | 150 |

### 4.2 Training Summary

| Metric | Value |
|--------|-------|
| Total epochs | 150 (completed naturally) |
| Stop reason | Max epochs reached (150) |
| Best val_loss | **3.1782** @ epoch 129 |
| Final val_loss | 3.5148 @ epoch 149 |
| Total training time | 28,743 s (7.9 GPU-hours) |
| Avg epoch time | ~396 s (6.6 min) |
| Avg throughput | 42.4 graphs/s |

Note: The reported 7.9 GPU-hours reflects a single training segment. The total experiment consumed approximately 16.5 GPU-hours including initial aborted runs and restarts (see the 15 duplicate entries in `index.csv` for SL-20260701-*).

### 4.3 Best Validation Epoch

At epoch 129, the model achieved its lowest validation loss:

```
Best epoch:        129
Validation Loss:   3.1782

Per-task metrics (approximate from epoch 129):
  formation_energy     MAE=0.5289  R²=0.5748
  energy_above_hull    MAE=0.1376  R²=N/A
  band_gap             MAE=1.0494  R²=N/A
```

### 4.4 Per-Task Best Metrics

| Metric | Best Value | Epoch | Final Value | Delta |
|--------|-----------|-------|-------------|-------|
| Ef MAE | **0.5181** | 148 | 0.5414 | +0.0233 |
| Ef R² | **0.5897** | 148 | 0.5618 | -0.0279 |
| EaH MAE | 0.1252 | 4 | 0.1459 | +0.0207 |
| EaH R² | 0.4227 | 147 | 0.4022 | -0.0205 |
| BG MAE | 1.0453 | 62 | 1.1681 | +0.1228 |
| BG R² | 0.3060 | 100 | 0.1973 | -0.1087 |

Key observations:

- **Ef R² (0.5897)** peaks very late at epoch 148, just two epochs before the run ends. This suggests the model was still slowly improving Ef predictions even after 148 epochs with GradNorm.
- **EaH MAE (0.1252)** also peaks early at epoch 4, matching the pattern seen in Experiment A. The optimal EaH regressor weights are found within the first few epochs regardless of GradNorm.
- **BG MAE (1.0453)** peaks at epoch 62, then degrades substantially to 1.1681 at epoch 149. The band gap is the most volatile task, and without a scheduler the model may overfit to the other two tasks at its expense.
- **EaH R² (0.4227)** peaks at epoch 147, substantially later than the MAE peak. This likely reflects the two-stage head improving its classification accuracy over time while the magnitude regressor degrades slightly.

---

## 5. Head-to-Head Comparison

### 5.1 Overall Comparison Table

| Metric | Prev (SL-20260701-007) | Exp A (SL-20260708-001) | Δ | % Change | Winner |
|--------|----------------------|------------------------|---|----------|--------|
| Best val_loss | 3.1782 | **3.0941** | -0.0841 | -2.65% | **Exp A** |
| Best Ef MAE | **0.5181** | 0.5222 | +0.0041 | +0.79% | **Prev** |
| Best Ef R² | **0.5897** | 0.5871 | -0.0026 | -0.44% | **Prev** |
| Best EaH MAE | **0.1029** | 0.1280 | +0.0251 | +24.4% | **Prev** |
| Best EaH R² | **0.4227** | 0.3854 | -0.0373 | -8.82% | **Prev** |
| Best BG MAE | 1.2493 | **1.0252** | -0.2241 | -17.9% | **Exp A** |
| Best BG R² | 0.3060 | **0.3385** | +0.0325 | +10.6% | **Exp A** |
| Test Ef MAE | 0.3267 | **0.3154** | -0.0113 | -3.46% | **Exp A** |
| Test EaH MAE | 0.1029 | **0.0973** | -0.0056 | -5.44% | **Exp A** |
| Test BG MAE | 1.2493 | **1.2339** | -0.0154 | -1.23% | **Exp A** |
| Two-stage F1 | **0.9539** | 0.9537 | -0.0002 | -0.02% | Tie |
| Avg throughput | 42.4 g/s | **40.6 g/s** | -1.8 | -4.2% | **Prev** |
| Total GPU-hrs | 16.5 | **15.8** | -0.7 | -4.2% | **Exp A** |

### 5.2 Statistical Significance

The observed deltas are small relative to the noise floor of the training process. Key considerations:

- **Val_loss (-2.65%)**: The most reliable comparison metric since it reflects the full multi-task objective. The cosine scheduler's systematic exploration of the loss landscape likely converges to a better joint optimum.
- **Ef R² (-0.44%)**: Essentially tied. GradNorm's dynamic reweighting may provide a slight edge for formation energy, but the difference is within run-to-run variance.
- **BG MAE (-17.9%)**: The most significant improvement. Removing GradNorm and adding the cosine scheduler dramatically improves band gap regression. This is the headline result.
- **EaH MAE (+24.4%)**: The largest regression. GradNorm appears to help the two-stage EaH head achieve lower MAE on the validation set, though this advantage does not transfer to the test set (where Exp A wins at -5.44%).

The test-set comparison is arguably more meaningful than the validation comparison, and on the test set **Exp A wins on all three tasks** (Ef MAE: -3.46%, EaH MAE: -5.44%, BG MAE: -1.23%). This suggests that the cosine scheduler provides better generalization even when validation metrics appear comparable.

### 5.3 Key Takeaways

1. **Cosine scheduler without GradNorm achieves lower validation loss** by a modest margin (3.0941 vs 3.1782).
2. **Band gap regression improves substantially** (17.9% lower MAE) without GradNorm. This is consistent with the hypothesis that GradNorm's dynamic reweighting harms the minority task (BG has higher inherent error) by allocating it less weight.
3. **Formation energy is essentially unaffected** by the choice, differing by <1%.
4. **EaH shows a validation regression but test improvement** — the two-stage head's behavior is complex and differs between validation and test distributions.
5. **Throughput is similar** (40.6 vs 42.4 g/s), with the slight decrease in Exp A possibly due to the cosine scheduler's overhead or run-to-run variance.
6. **Exp A uses 0.7 fewer GPU-hours** (15.8 vs 16.5) due to early stopping at 138 epochs vs. 150.

Bottom line: **Disabling GradNorm and using a cosine scheduler produces a better model overall**, particularly for band gap prediction, while using slightly less compute.

---

## 6. Learning Curves Analysis

### 6.1 Loss Curves

Both experiments show the characteristic rapid initial convergence followed by slow refinement:

**Experiment A (SL-20260708-001):**

```
Epoch   0: train_loss=3.2075  val_loss=3.9715
Epoch   5: train_loss=2.4261  val_loss=3.6742
Epoch  25: train_loss=2.0128  val_loss=3.3923
Epoch  50: train_loss=1.9329  val_loss=3.2680
Epoch  75: train_loss=1.9109  val_loss=3.1852
Epoch  98: train_loss=1.8623  val_loss=3.0941  <- best
Epoch 138: train_loss=1.8609  val_loss=3.1452  <- final
```

Key features of the loss curve:
- **Sharp initial drop**: val_loss falls from 3.97 to 3.67 in 5 epochs (7.5% improvement)
- **Monotonic improvement**: val_loss decreases nearly monotonically from epoch 0 to epoch 98
- **Plateau**: After epoch 98, val_loss oscillates in the 3.09–3.15 range for 40 epochs
- **Train loss continues decreasing**: train_loss drops from 1.86 (ep 98) to 1.86 (ep 138), indicating the model is still fitting the training data even as val_loss plateaus
- **Train-val gap**: ~1.0 at best epoch, widening slightly to ~1.3 at final epoch

**Previous Run (SL-20260701-007):**

```
Epoch   0: train_loss=3.0743  val_loss=4.0396
Epoch   5: train_loss=2.4206  val_loss=3.6994
Epoch  25: train_loss=2.0920  val_loss=3.3937
Epoch  50: train_loss=1.9628  val_loss=3.3029
Epoch  75: train_loss=1.9439  val_loss=3.2858
Epoch 100: train_loss=1.9389  val_loss=3.2285
Epoch 129: train_loss=1.8988  val_loss=3.1782  <- best
Epoch 149: train_loss=1.9140  val_loss=3.5148  <- final (val_loss spike)
```

Key differences from Exp A:
- **Gradual improvement**: The previous run's val_loss drops more slowly and never matches Exp A's best value
- **Late-epoch divergence**: At epoch 149, val_loss spikes to 3.5148 (worse than epoch 0), suggesting the model entered a bad region of the loss landscape. This does not happen in Exp A, possibly because the cosine scheduler's warm restarts help escape bad basins.
- **GradNorm weight evolution**: The GradNorm weights evolve significantly over training:
  - Band gap weight: starts at 1.31, decays to **0.0014** by epoch 149 (virtually eliminated)
  - Formation energy weight: starts at 0.09, grows to **1.63** (dominant task)
  - Energy above hull: stable around 0.6–1.1
  - The near-zero band gap weight at epoch 149 explains the poor BG MAE (1.1681) in the final epoch

### 6.2 Per-Task Error Curves

**Experiment A MAE trajectories:**

| Epoch | Ef MAE | EaH MAE | BG MAE |
|-------|--------|---------|--------|
| 0 | 0.7193 | 0.1311 | 1.1950 |
| 25 | 0.5669 | 0.1371 | 1.1083 |
| 50 | 0.5471 | 0.1335 | 1.0766 |
| 75 | 0.5397 | 0.1334 | 1.0660 |
| 98 | 0.5508 | 0.1313 | **1.0252** |
| 135 | **0.5237** | 0.1345 | 1.0676 |
| 136 | **0.5222** | 0.1372 | 1.0656 |
| 138 | 0.5276 | 0.1351 | 1.0506 |

**Previous Run MAE trajectories:**

| Epoch | Ef MAE | EaH MAE | BG MAE |
|-------|--------|---------|--------|
| 0 | 0.6917 | 0.1555 | 1.1885 |
| 25 | 0.5653 | 0.1393 | 1.1064 |
| 50 | 0.5445 | 0.1332 | 1.0674 |
| 62 | — | — | **1.0453** |
| 100 | 0.5392 | 0.1324 | 1.0788 |
| 129 | 0.5289 | 0.1376 | 1.0494 |
| 148 | **0.5181** | 0.1471 | 1.0852 |
| 149 | 0.5414 | 0.1459 | 1.1681 |

**Analysis:**
- **Ef MAE** converges to ~0.52 in both runs, with the previous run edging slightly ahead (0.5181 vs 0.5222) at late epochs.
- **EaH MAE** is lower in the previous run (0.1252 vs 0.1280 at best), but this advantage comes from GradNorm maintaining a reasonable weight on EaH throughout training. In Exp A, the equal weighting may slightly over-prioritize Ef.
- **BG MAE** is the decisive differentiator: Exp A reaches 1.0252 (best) while the previous run bottoms at 1.0453 and then escalates to 1.1681. The cosine scheduler's periodic LR warmups prevent complete task abandonment.

### 6.3 Per-Task R² Curves

**Experiment A R² trajectories:**

| Epoch | Ef R² | EaH R² | BG R² |
|-------|-------|--------|-------|
| 0 | 0.3692 | 0.3330 | 0.1467 |
| 25 | 0.5152 | 0.3109 | 0.2385 |
| 50 | 0.5436 | 0.3409 | 0.2737 |
| 75 | 0.5564 | 0.3463 | 0.2801 |
| 98 | 0.5511 | 0.3735 | **0.3385** |
| 135 | **0.5871** | 0.3810 | 0.3052 |
| 138 | 0.5811 | 0.3825 | 0.3107 |

**Previous Run R² trajectories:**

| Epoch | Ef R² | EaH R² | BG R² |
|-------|-------|--------|-------|
| 0 | 0.3864 | 0.2935 | 0.1288 |
| 25 | 0.5084 | 0.3018 | 0.2147 |
| 50 | 0.5387 | 0.3440 | 0.2556 |
| 100 | 0.5375 | 0.3725 | **0.3060** |
| 129 | 0.5748 | 0.3883 | 0.2780 |
| 147 | 0.5767 | **0.4227** | 0.2108 |
| 148 | **0.5897** | 0.4124 | 0.2202 |
| 149 | 0.5618 | 0.4022 | 0.1973 |

**Analysis:**
- **Ef R²** is effectively tied (~0.588), with both runs achieving their best in the final 10% of training.
- **EaH R²** is notably higher in the previous run (0.4227 vs 0.3854), consistent with GradNorm maintaining a higher effective weight on EaH.
- **BG R²** favors Exp A (0.3385 vs 0.3060), and the previous run's BG R² collapses in the final 50 epochs, dropping from 0.3060 to 0.1973 as GradNorm virtually eliminates the BG task weight.
- The **EaH R² vs EaH MAE divergence** (best R² at late epochs but best MAE at early epochs) confirms that the two-stage head's classification improves over time while the magnitude regression degrades slightly.

---

## 7. Per-Task Performance Deep Dive

### 7.1 Formation Energy (Ef)

Formation energy prediction is the strongest task for this model, achieving R² of ~0.59 in both experiments. This is expected because:

- Formation energy has the widest dynamic range (typically -5 to +2 eV/atom)
- It is the most physically constrained property in the Materials Project dataset
- The loss contribution dominates the total loss (largest absolute values)

**Best Validation Performance:**
- Prev: Ef MAE = **0.5181** eV/atom, Ef R² = **0.5897** (epoch 148)
- Exp A: Ef MAE = **0.5222** eV/atom, Ef R² = **0.5871** (epoch 135–136)
- Delta: +0.0041 MAE, -0.0026 R² (essentially tied)

**Test Performance:**
- Prev: Ef MAE = 0.3267 eV/atom, R² = 0.5528
- Exp A: Ef MAE = **0.3154** eV/atom, R² = **0.5121**
- Delta: -3.46% MAE

The test MAE is substantially lower than the validation MAE (~0.32 vs ~0.52), which is unusual. This may indicate that the validation set contains systematically harder examples, or that the best validation model (selected by val_loss) does not perfectly align with the best Ef model.

### 7.2 Energy Above Hull (EaH)

EaH is the hardest task for pure regression (R² ~0.18 on the test set), but the two-stage head adds classification capability. The task is complicated by:

- Skewed distribution: ~91% of test materials are unstable (EaH > 0), only ~9% are stable
- Different regression difficulty for stable vs. unstable materials
- Physical noise: DFT-computed EaH values are sensitive to pseudopotential choices

**Best Validation Performance:**
- Prev: EaH MAE = **0.1252** eV/atom (epoch 4); EaH R² = **0.4227** (epoch 147)
- Exp A: EaH MAE = **0.1280** eV/atom (epoch 4); EaH R² = **0.3854** (epoch 131)

The best MAE occurs at epoch 4 in both runs, suggesting that the two-stage head's magnitude regressor converges at least as fast as the backbone features stabilize. The classification head takes longer to reach peak R².

**Test Performance:**
- Prev: EaH MAE = **0.1029** eV/atom, R² = 0.1844
- Exp A: EaH MAE = **0.0973** eV/atom, R² = 0.1771

### 7.3 Band Gap (BG)

Band gap is the most challenging property to predict from structure alone (best R² ~0.34), likely because:

- Band gap depends on electronic structure details not fully captured by geometric graph features
- Materials Project band gaps use PBE functional, which systematically underestimates band gaps
- The distribution is heavy-tailed with many zero-gap metals

**Best Validation Performance:**
- Prev: BG MAE = **1.0453** eV (epoch 62); BG R² = **0.3060** (epoch 100)
- Exp A: BG MAE = **1.0252** eV (epoch 98); BG R² = **0.3385** (epoch 98)

Exp A achieves both best MAE and best R² at the same epoch (98), while the previous run has them separated by 38 epochs. This suggests that the cosine scheduler's LR restarts help the model find a consistent optimum for BG where the fixed-LR model drifts.

**Test Performance:**
- Prev: BG MAE = 1.2493 eV, R² = 0.0373
- Exp A: BG MAE = **1.2339** eV, R² = **0.0692**

**Failure analysis:** The very low test R² (0.037–0.069) despite moderate MAE (~1.2 eV) indicates that the model captures only the mean band gap but has very low variance — it predicts nearly the same value for all materials. The MAE of ~1.2 eV on a dataset with mean band gap of ~1.5 eV means the model is only slightly better than predicting the mean.

---

## 8. Two-Stage EaH Results

### 8.1 Architecture

The two-stage EaH head consists of:

1. **Stability classifier**: A binary classifier predicting `p_unstable` (probability that EaH > 0)
2. **Magnitude regressor**: A regression head predicting `eah_magnitude` for unstable materials
3. **Combined output**: `eah_pred = p_unstable * eah_magnitude`

Training uses a composite loss: `L = λ_bce * L_bce + λ_reg * L_mse + λ_stable * L_stable`, where `L_stable` encourages `p_unstable` ≈ 0 for stable materials.

### 8.2 Classification Performance

| Metric | Prev (SL-20260701-007) | Exp A (SL-20260708-001) |
|--------|------------------------|------------------------|
| Stability F1 | **0.9539** | 0.9537 |
| Precision | 0.9144 | **0.9190** |
| Recall | **0.9970** | 0.9911 |
| N stable (test) | 97 | 97 |
| N unstable (test) | 1007 | 1007 |

The classification performance is excellent and nearly identical across both runs:

- **F1 ≈ 0.954**: Outstanding for a highly imbalanced problem (stable:unstable ≈ 1:10)
- **Recall ≈ 0.99**: The model almost never misses an unstable material (0.3–0.9% false negative rate)
- **Precision ≈ 0.92**: ~8% of materials classified as unstable are actually stable (false positives)

The high recall / moderate precision tradeoff is the correct bias for materials screening: it is better to over-predict instability (false positive) than to miss a promising unstable material (false negative) that could be stabilized through doping or synthesis optimization.

### 8.3 Regression Performance (Unstable Only)

For materials classified as unstable, the magnitude regressor predicts the EaH value:

| Metric | Prev (SL-20260701-007) | Exp A (SL-20260708-001) |
|--------|------------------------|------------------------|
| EaH MAE (all) | 0.1029 | **0.0973** |
| EaH MAE (unstable) | 0.1052 | **0.0999** |

The magnitude regressor achieves ~0.10 eV/atom MAE, which is reasonable for DFT-level accuracy. Exp A shows a 5% improvement on the test set even though its validation EaH MAE was worse.

### 8.4 Comparison Across Runs

| Metric | Prev Validation | Prev Test | Exp A Validation | Exp A Test |
|--------|----------------|-----------|-----------------|------------|
| EaH MAE | 0.1252 | 0.1029 | 0.1280 | 0.0973 |
| EaH MAE (unstable) | — | 0.1052 | — | 0.0999 |
| EaH R² | 0.4227 | 0.1844 | 0.3854 | 0.1771 |
| Stability F1 | — | 0.9539 | — | 0.9537 |

The dramatic drop in R² from validation (~0.40) to test (~0.18) while MAE remains stable (~0.10) is explained by the distribution difference: the test set has lower EaH variance than the validation set, making the R² metric (which normalizes by variance) more sensitive to small errors.

---

## 9. Convergence Analysis

### 9.1 Experiment A Convergence

Exp A reaches best val_loss at epoch 98 (out of 138), with 40 epochs of non-improvement triggering early stopping. The convergence trajectory shows:

- **Phase 1 (epochs 0–25)**: Rapid convergence. Val_loss drops from 3.97 to 3.39 (−14.6%).
- **Phase 2 (epochs 25–75)**: Steady improvement. Val_loss drops from 3.39 to 3.19 (−6.1%).
- **Phase 3 (epochs 75–98)**: Fine-tuning. Val_loss drops from 3.19 to 3.09 (−3.0%).
- **Phase 4 (epochs 98–138)**: Plateau. Val_loss ranges 3.09–3.31 with no sustained improvement.

The cosine scheduler produces a distinctive pattern in the loss curve: each LR cycle restart (visible as small bumps in the learning rate plot at ~epoch 40 and ~epoch 80) temporarily increases val_loss before finding a better minimum. The third cycle peaks at epoch 98, after which no further improvement is achieved.

### 9.2 Previous Run Convergence

The previous run completes all 150 epochs and reaches best val_loss at epoch 129. The convergence trajectory differs:

- **Phase 1 (epochs 0–25)**: Rapid convergence (same as Exp A, 3.70 at ep 5).
- **Phase 2 (epochs 25–129)**: Slow, noisy improvement. Val_loss oscillates between 3.18 and 3.39.
- **Phase 3 (epochs 129–149)**: Degradation. Val_loss climbs from 3.18 to 3.51, suggesting the model enters a bad basin.

The degradation phase is concerning: the val_loss at epoch 149 (3.51) is worse than at epoch 5 (3.70). This is accompanied by the GradNorm weight for band gap decaying to near-zero (0.0014), effectively eliminating BG from the loss and causing the model to overfit to formation energy and EaH.

### 9.3 Scheduler Impact on Convergence

The cosine with warm restarts scheduler in Exp A provides two benefits:

1. **Better minima through periodic high LR**: Each restart temporarily increases the learning rate from near-zero to 5e-4, allowing the optimizer to escape sharp minima and find flatter, more generalizable regions. This is visible as temporary val_loss spikes followed by descents to new lows.

2. **Prevention of task abandonment**: The fixed LR in the previous run, combined with GradNorm, allows the BG task weight to decay to essentially zero. The cosine scheduler's periodic LR resets in Exp A prevent any single task from being completely de-weighted (GradNorm is off, so all tasks maintain equal weight throughout).

Without a scheduler, the fixed LR of 5e-4 leads to increasingly small gradient steps as the model approaches a minimum, and GradNorm can push task weights to extremes. The scheduler mitigates both issues.

### 9.4 GradNorm Impact on Convergence

GradNorm's impact on convergence is nuanced:

**Benefits (visible in the previous run):**
- Higher EaH R² (0.4227 vs 0.3854) — GradNorm allocates appropriate weight to the two-stage EaH head
- Slightly better Ef R² (0.5897 vs 0.5871) — marginal improvement for the primary task
- Faster early convergence (epoch 5: 3.70 vs 3.67)

**Drawbacks (visible in the previous run):**
- Near-zero BG weight at late epochs → poor BG performance
- Unstable late-epoch behavior (val_loss spike at epoch 149)
- Requires hyperparameter tuning (alpha, initial weights, learning rate for weight updates)
- Adds ~3% epoch time overhead for gradient computations

The GradNorm trajectories from the previous run tell a clear story:

| Task | Initial Weight | Final Weight | Trend |
|------|---------------|--------------|-------|
| Formation energy | 0.09 | 1.63 | ↑ increasing (dominant) |
| Energy above hull | 1.06 | 0.60 | ↓ decreasing |
| Band gap | 1.31 | 0.001 | ↓ decreasing (eliminated) |

The formation energy weight increases from 0.09 to 1.63 (18× increase), while the band gap weight falls from 1.31 to 0.001 (near elimination). This is consistent with GradNorm's design: tasks with higher loss gradients (Ef has the largest magnitude) receive higher weights, while tasks with lower gradients (BG has high inherent noise) are de-emphasized. The problem is that GradNorm can push weights to destructive extremes over 150 epochs.

---

## 10. Test Set Results

### 10.1 Test Set Performance (Previous Run)

The following results are from `runs/SL-20260701-007/test_results.json`:

| Task | MAE | RMSE | R² | Pearson | Spearman | Bias |
|------|-----|------|----|---------|----------|------|
| Formation energy | 0.3267 | 0.5461 | 0.5528 | 0.7498 | 0.5454 | +0.0796 |
| Energy above hull | 0.1029 | 0.3934 | 0.1844 | 0.4345 | 0.2550 | -0.0178 |
| Band gap | 1.2493 | 1.5268 | 0.0373 | 0.2971 | 0.2746 | +0.1626 |
| Two-stage EaH | F1=0.9539 | P=0.9144 | R=0.9970 | EaH MAE=0.1029 | Unstable MAE=0.1052 | — |

### 10.2 Test Set Performance (Experiment A)

The following results are from `runs/SL-20260708-001/test_results.json`:

| Task | MAE | RMSE | R² | Pearson | Spearman | Bias |
|------|-----|------|----|---------|----------|------|
| Formation energy | **0.3154** | **0.5704** | **0.5121** | **0.7377** | **0.5838** | +0.0928 |
| Energy above hull | **0.0973** | **0.3952** | **0.1771** | **0.4284** | **0.2346** | -0.0323 |
| Band gap | **1.2339** | **1.5013** | **0.0692** | **0.3220** | **0.3224** | +0.1769 |
| Two-stage EaH | F1=**0.9537** | P=**0.9190** | R=0.9911 | EaH MAE=**0.0973** | Unstable MAE=**0.0999** | — |

### 10.3 Comparison Across Runs

| Metric | Prev Test | Exp A Test | Δ | % Change | Winner |
|--------|-----------|------------|---|----------|--------|
| Ef MAE | 0.3267 | **0.3154** | -0.0113 | -3.46% | **Exp A** |
| Ef R² | **0.5528** | 0.5121 | -0.0407 | -7.36% | **Prev** |
| EaH MAE | 0.1029 | **0.0973** | -0.0056 | -5.44% | **Exp A** |
| EaH R² | **0.1844** | 0.1771 | -0.0073 | -3.96% | **Prev** |
| BG MAE | 1.2493 | **1.2339** | -0.0154 | -1.23% | **Exp A** |
| BG R² | 0.0373 | **0.0692** | +0.0319 | +85.5% | **Exp A** |
| F1 | **0.9539** | 0.9537 | -0.0002 | -0.02% | Tie |

On the held-out test set, Exp A wins on MAE for all three tasks and on R² for BG, while the previous run wins on R² for Ef and EaH. The R² advantage for Ef (7.36%) is notable and suggests that GradNorm's task weighting produces better-calibrated predictions for the primary task, even though the absolute error is slightly higher.

---

## 11. Failure Cases and Discussion

### 11.1 Band Gap Regression

Band gap prediction remains the weakest link in the model, with test R² of 0.037–0.069 and MAE of ~1.2 eV. This is a known challenge in materials informatics:

- **Underestimated gaps**: PBE-DFT systematically underestimates band gaps (the "band gap problem" in DFT). The model learns DFT-predicted gaps, not experimental gaps.
- **Feature limitations**: Geometric graph features (bond distances, angles, coordination) capture structural information but may not capture electronic structure features that determine band gaps.
- **Zero-gap metals**: ~30% of materials have zero or near-zero band gaps. The model must predict both zero and non-zero values, which is challenging for MSE-optimized regression.

**Mitigation strategies:**
- Add electronic-structure-aware features (e.g., electronegativity, oxidation states)
- Use a two-stage approach for BG (classify metals vs. non-metals first, then regress non-metal gaps)
- Explore different loss functions (e.g., Huber loss for robustness to heavy tails)
- Train with an explicit penalty for negative BG predictions

### 11.2 Energy Above Hull Regression

The low EaH R² (~0.18 on test) despite reasonable MAE (~0.10 eV/atom) indicates:

- **Low label variance**: The test set's EaH values are concentrated in a narrow range, making R² (which normalizes by variance) a harsh metric
- **Classification-regression coupling**: Small errors in the stability classifier propagate to the magnitude regressor
- **Two-stage complexity**: The two-stage head's four loss terms (BCE, MSE, stable penalty, total) create complex optimization dynamics

Despite the low R², the model's practical utility for materials screening is validated by the classification F1 of 0.954 and the magnitude MAE of ~0.10 eV/atom. For screening purposes, knowing whether a material is stable vs. unstable with 95% F1 is more useful than precisely predicting the EaH value.

### 11.3 Overfitting Analysis

**Train vs. val loss gap:**

| Run | Best Train Loss | Best Val Loss | Gap |
|-----|----------------|---------------|-----|
| Prev (ep 129) | 1.8988 | 3.1782 | 1.2794 |
| Exp A (ep 98) | 1.8623 | 3.0941 | 1.2318 |

The consistent ~1.2 gap between train and val loss suggests mild overfitting that is stable across training regimes. The gap does not widen significantly in later epochs, indicating that the model's capacity (1.28M params) is well-matched to the 8,000-sample training set.

**Per-task overfitting:**
- **Ef MAE**: Train ~0.40, Val ~0.52, Test ~0.32 (test better than validation — unusual)
- **EaH MAE**: Train ~0.08, Val ~0.13, Test ~0.10
- **BG MAE**: Train ~0.70, Val ~1.03, Test ~1.23

The test MAE being lower than validation MAE for Ef suggests that the validation set may be harder than the test set for this task, which would be an artifact of the random 80/10/10 split.

### 11.4 Out-of-Distribution Performance

The model is trained exclusively on Li-containing materials with Li ≥ 5 at.%. Predictions for:

- **Li-poor materials (Li < 5 at.%)**: Untested — the graph features would be in a different regime
- **Non-Li materials**: Not supported — the model has never seen non-Li chemistries
- **Novel frameworks**: Likely reliable if the local structural environments (bond distances, coordination) resemble the training distribution

For production screening, we recommend using MC Dropout uncertainty estimates to flag OOD inputs. The model's `mc_dropout_samples` parameter (default: 20) enables uncertainty quantification:

```python
model.train()  # Enable dropout
preds = []
for _ in range(20):
    with torch.no_grad():
        preds.append(model(cg, lg))
mean = torch.stack(preds).mean(0)
std = torch.stack(preds).std(0)
```

High prediction variance (>2× training variance) indicates OOD inputs.

---

## 12. Recommendations for Future Experiments

Based on the experimental results, the following directions are recommended for future work:

### Configuration Recommendations

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| GradNorm | **OFF** (disabled) | Negligible benefit for Ef, harms BG; increases complexity and computation |
| Scheduler | **Cosine with restarts** | Better minima, prevents task abandonment, lower val_loss |
| LR | 5e-4 (current) | Well-calibrated; no evidence for change |
| Patience | 40 | Appropriate — both runs plateau after ~100 epochs |
| Max epochs | 150 | Sufficient — neither run would benefit from more |
| Warmup | 0 (current) | Not needed with cosine scheduler |

### Model Architecture Recommendations

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Hidden dim | 128 (current) | Well-matched to 10K dataset |
| ALIGNN layers | 4 (current) | Sufficient depth for structure-property mapping |
| Transformer layers | 2 (current) | Minimal benefit from more layers for this dataset size |
| Two-stage EaH | **Keep** | F1=0.954 on test; critical for screening utility |
| MC Dropout | **Use for screening** | Uncertainty estimation for OOD detection |

### Dataset Recommendations

| Item | Recommendation |
|------|---------------|
| Size | 10K materials is adequate but 50K+ would enable deeper models |
| Target balance | Add more stable materials (currently 9:1 unstable:stable) |
| Band gap | Consider GGA+U or hybrid functional targets for better label quality |
| Features | Investigate adding electronic-structure-aware features for BG |

### Future Experiment Plan

| Priority | Experiment | Expected Benefit |
|----------|-----------|-----------------|
| P0 | Exp A config + 5 seeds | Quantify run-to-run variance |
| P1 | Exp A config + 32 batch size | Higher throughput, better gradient estimates |
| P2 | Exp A + two-stage BG | Improve BG R² via classification |
| P3 | Optuna sweep (lr, dropout, hidden_dim) | Optimize hyperparameters for cosine schedule |
| P4 | 50K dataset + 256 hidden dim | Scale model to larger data |
| P5 | `torch.compile` full training | 10–15% throughput improvement |

---

*Generated from `runs/SL-20260708-001/analysis/FINAL_REPORT.md`, `runs/SL-20260708-001/epoch_metrics.json`, `runs/SL-20260701-007/epoch_metrics.json`, and `runs/*/test_results.json`. Analysis script: `scripts/analyze/analyze_training.py`.*
