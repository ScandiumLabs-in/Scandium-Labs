# Loss Functions — Scandium Labs

**Overview:** Three loss systems operate in the Scandium Labs training pipeline:
1. **PINNLoss** — physics-informed multi-task loss (legacy, used by `ScandiumTrainer`)
2. **GradNorm** — adaptive task-weighting via gradient normalization
3. **TwoStageEahLoss** — specialized loss for two-stage energy-above-hull prediction

---

## 1. PINNLoss

**File:** `src/training/losses.py:9`

Multi-component physics-informed loss connecting data fidelity with Arrhenius, thermodynamic, and diffusion constraints.

### 1.1 Mathematical Formulation

$$\mathcal{L}_{\text{PINN}} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{arrhenius}} \mathcal{L}_{\text{arrhenius}} + \lambda_{\text{thermo}} \mathcal{L}_{\text{thermo}} + \lambda_{\text{physics}} \mathcal{L}_{\text{diffusion}}$$

#### Data Fidelity

$$\mathcal{L}_{\text{data}} = \sum_{t \in \mathcal{T}} w_t \cdot \frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} (\hat{y}_{t,i} - y_{t,i})^2$$

Where:
- $\mathcal{T} = \{\text{energy\_above\_hull},\ \text{formation\_energy},\ \text{band\_gap}\}$
- $w_t$ = configurable per-task weight (default: 1.0 for all)
- $\mathcal{M}_t$ = mask of non-NaN targets for task $t$
- NaNs in target values are **excluded** from the loss (not zero-filled)

#### Arrhenius Constraint

$$\mathcal{L}_{\text{arrhenius}} = \text{Var}\Big[\log_{10}(\sigma \cdot T) + \frac{E_a}{k_B \cdot T \cdot \ln 10}\Big]$$

Where:
- $\sigma = 10^{\hat{y}_{\text{cond}}}$ = predicted ionic conductivity (S/cm)
- $E_a = \hat{y}_{\text{act}}$ = predicted activation energy (eV)
- $k_B = 8.617 \times 10^{-5}$ eV/K (Boltzmann constant)
- $T$ = temperature (default 300 K)
- $\ln 10 \approx 2.3026$

This enforces the **Arrhenius relation**: $\sigma T = A \exp(-E_a / k_B T)$. The variance penalty ensures consistency between conductivity and activation energy predictions — if the model predicts high conductivity, it must also predict low activation energy, and vice versa.

#### Thermodynamic Constraint

$$\mathcal{L}_{\text{thermo}} = \frac{1}{N} \sum_{i=1}^N \max(0, -\hat{y}_{\text{EaH}, i})$$

Penalizes **negative EaH predictions** (a material cannot have negative energy above the convex hull). The ReLU gating ensures only negative predictions contribute.

When `log_eah=True`, the prediction is exponentiated first: $\hat{y}_{\text{EaH}} = \exp(\hat{y}_{\text{raw}}) - \epsilon$.

#### Diffusion Residual (Disabled Placeholder)

$$\mathcal{L}_{\text{diffusion}} = \frac{1}{N} \sum_{i=1}^N \Big(\frac{\partial c}{\partial t} - D \nabla^2 c\Big)^2$$

Residual of the Fickian diffusion PDE. Requires a `concentration_head` submodule that does not exist in the current model. Always returns 0 in practice.

### 1.2 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_weights` | `{}` (all = 1.0) | Per-task weight dict |
| `lambda_data` | 1.0 | Data fidelity weight |
| `lambda_physics` | 0.1 | Physics constraint weight (total) |
| `lambda_arrhenius` | 0.05 | Arrhenius constraint weight |
| `lambda_thermodynamic` | 0.05 | Thermodynamic constraint weight |
| `log_eah` | False | Whether EaH is predicted in log-space |

### 1.3 Scaling and Normalization

Targets are normalized via `PropertyNormalizer` before reaching the loss:

$$\tilde{y}_t = \frac{y_t - \mu_t}{\sigma_t + \epsilon}$$

Where $\mu_t$, $\sigma_t$ are dataset statistics stored in `normalizer.json`. The loss operates on normalized targets.

### 1.4 Usage Note

PINNLoss is used by `ScandiumTrainer` (`src/training/trainer.py`) but is **not active** in the current `train_v3_li.py` pipeline, which uses only MSE + TwoStageEahLoss + GradNorm. The Arrhenius and thermodynamic components require conductivity and activation energy predictions, which are not trained in the v3-Li 3-task configuration.

---

## 2. GradNorm (Gradient Normalization)

**File:** `src/training/losses.py:66`

Adaptive multi-task loss balancing from Chen et al. (2018). Dynamically adjusts task weights $w_t$ so that all tasks learn at similar speeds, measured by gradient magnitude.

### 2.1 Mathematical Formulation

**Weighted total loss:**

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T w_t \mathcal{L}_t \quad \text{where} \quad w_t = \exp(\log w_t) > 0$$

**Per-task gradient norm:**

$$G_w^{(t)} = \|\nabla_{\theta_{\text{shared}}} (w_t \mathcal{L}_t)\|_2$$

Using the identity $\|\nabla(w_t \mathcal{L}_t)\| = w_t \|\nabla \mathcal{L}_t\|$ (since $w_t > 0$, the weight factors out of the gradient norm).

**Inverse training rate (loss ratio):**

$$\tilde{\mathcal{L}}_t = \frac{\mathcal{L}_t}{\mathcal{L}_t^{(0)}}$$

Where $\mathcal{L}_t^{(0)}$ is the loss at the first GradNorm step.

**Target gradient norm:**

$$\bar{G} = \mathbb{E}_t[G_w^{(t)}]$$

$$\text{target}_t = \bar{G} \cdot \left(\frac{\tilde{\mathcal{L}}_t}{\mathbb{E}_t[\tilde{\mathcal{L}}_t]}\right)^\alpha$$

**Gradient update to log-weights:**

$$\frac{\partial \mathcal{L}_{\text{grad}}}{\partial \log w_t} = \text{sign}(G_w^{(t)} - \text{target}_t) \cdot \|\nabla \mathcal{L}_t\| \cdot w_t$$

$$\log w_t \leftarrow \log w_t - \eta \cdot \frac{\partial \mathcal{L}_{\text{grad}}}{\partial \log w_t}$$

**Renormalization:**

$$w_t \leftarrow \frac{w_t \cdot T}{\sum_t w_t} \quad \text{(so that} \sum w_t = T\text{)}$$

### 2.2 Implementation Details

```python
# Optimized implementation — 3 autograd.grad calls, no create_graph=True
# Analytical gradient for log-weights (not through autograd)

class GradNormLoss(nn.Module):
    def __init__(self, tasks, alpha=1.5, initial_weights=None):
        self.log_weights = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(w).log())
            for t, w in initial_weights.items()
        })

    def update_weights(self, task_losses, shared_params, lr=0.025):
        # 1. Compute raw gradient norms for each task
        raw_norms[t] = grad_norm(task_losses[t], shared_params)

        # 2. Compute loss ratios and targets
        loss_ratios[t] = task_losses[t] / initial_losses[t]
        target[t] = avg_raw_norm * (loss_ratios[t] / avg_loss_ratio)^alpha

        # 3. Analytical gradient update
        grad_log_w = sign(G_w - target) * raw_norms[t] * w_t
        log_w_t -= lr * grad_log_w

        # 4. Renormalize
        log_w_t -= log(sum(w) / n_tasks)
```

### 2.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1.5 | Restoring strength — higher = stronger balancing |
| `lr` | 0.025 | Learning rate for weight updates |
| `initial_weights` | `{ef: 1.0, eah: 1.0, bg: 0.4}` | Starting task weights |

The `alpha` parameter controls how aggressively GradNorm equalizes task learning:
- $\alpha = 0$: no balancing (weights stay at initial values)
- $\alpha = 0.5$: mild balancing
- $\alpha = 1.5$: strong balancing (default)
- $\alpha > 2$: aggressive balancing (may destabilize)

### 2.4 Integration in Training Loop

**File:** `scripts/train/train_v3_li.py:345`

```python
if USE_GRADNORM:
    total_loss = grad_norm.compute_total(task_losses) / GRAD_ACCUM
    if n_batches % 50 == 0:
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith("task_heads")
                           and not n.startswith("uncertainty_heads")]
        grad_norm.update_weights(task_losses, backbone_params, lr=0.025)
```

Key details:
- GradNorm updates every **50 batches** (not every step)
- Only updates **shared backbone** parameters (excludes task heads and uncertainty heads)
- The `compute_total()` call uses **detached weights** (`w[t].detach()`)
- When `--no-gradnorm` is passed, fixed weights are used: `{ef: 1.0, eah: 1.0, bg: 0.4}`

### 2.5 Impact Analysis

| Task | Initial Weight | Effect of GradNorm |
|------|:-------------:|:------------------:|
| formation_energy | 1.0 | Usually stays near 1.0 (dominant task) |
| energy_above_hull | 1.0 | Weight may increase initially (harder task) |
| band_gap | 0.4 | Weight may decrease further (model converges faster) |

**Observed behavior:**
- GradNorm typically **increases** the EaH weight during early training (this task is harder), which helps the TwoStageEahLoss converge faster
- Band gap weight may **decrease** as the model learns it easily — preventing it from dominating the gradient
- Without GradNorm, formation energy tends to dominate because it has the largest magnitude

### 2.6 Ablation: Exp A (GradNorm OFF) vs Current Run (GradNorm ON)

| Setting | Ef MAE | EaH MAE | BG MAE | Notes |
|---------|:------:|:-------:|:------:|-------|
| **Exp A** (no GradNorm, no scheduler) | — | — | — | Baseline; may show Ef dominance |
| **Exp B** (GradNorm ON + CosineWarmRestarts) | — | — | — | **Currently running** |

*Ablation results will be populated from experiment B when it completes.*

---

## 3. TwoStageEahLoss

**File:** `src/models/heads/two_stage_eah.py:73`

Specialized loss for the two-stage energy-above-hull head that separates stability classification from magnitude regression.

### 3.1 Motivation

Approximately 70% of EaH targets are near zero (stable materials on the convex hull). A standard MSE loss causes the model to **collapse to zero** — always predicting EaH ≈ 0 — because that minimizes error for the majority class. The two-stage approach solves this:

1. **Stage 1:** Binary classifier determines if the material is unstable (EaH > threshold)
2. **Stage 2:** Regressor predicts EaH magnitude, trained only on unstable samples

### 3.2 Mathematical Formulation

$$\mathcal{L}_{\text{TwoStage}} = \lambda_{\text{bce}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{reg}} \mathcal{L}_{\text{MSE}}^{\text{(unstable)}} + \lambda_{\text{stable}} \mathcal{L}_{\text{MSE}}^{\text{(stable)}}$$

#### BCE — Stability Classification

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \big[y_i^{\text{us}} \log(p_i) + (1 - y_i^{\text{us}}) \log(1 - p_i)\big]$$

Where $y_i^{\text{us}} = \mathbb{1}[y_i^{\text{EaH}} > \tau]$ and $\tau = 10^{-3}$ eV/atom.

#### MSE — Magnitude Regression (Unstable Only)

$$\mathcal{L}_{\text{MSE}}^{\text{(unstable)}} = \frac{1}{\sum_i y_i^{\text{us}} + \epsilon} \sum_i y_i^{\text{us}} \cdot (m_i - y_i^{\text{EaH}})^2$$

Where $m_i$ is the predicted magnitude (Softplus output, always positive). Only samples classified as unstable contribute.

#### MSE — Stable Regularization

$$\mathcal{L}_{\text{MSE}}^{\text{(stable)}} = \frac{1}{\sum_i (1 - y_i^{\text{us}}) + \epsilon} \sum_i (1 - y_i^{\text{us}}) \cdot (\hat{y}_i^{\text{EaH}} - 0)^2$$

Penalizes the combined prediction $(\hat{y} = p \cdot m)$ for stable samples — forces both $p \to 0$ and/or $m \to 0$ for stable materials.

#### Combined Prediction

$$\hat{y}_i^{\text{EaH}} = \sigma(\text{MLP}_{\text{stab}}(h_i)) \cdot \text{Softplus}(\text{MLP}_{\text{mag}}(h_i))$$

### 3.3 TwoStageEahLoss Configuration

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `lambda_bce` | 1.0 | BCE classification weight |
| `lambda_reg` | 3.0 | Magnitude regression weight (unstable) |
| `lambda_stable` | 0.5 | Stable regularization weight |
| `stability_threshold` | 0.001 eV/atom | Threshold for unstable classification |

### 3.4 Per-Sample Family Weights

Supports optional per-sample weights to counteract composition shortcuts:

```python
def forward(self, output, eah_true, family_weights=None):
    ...
    if family_weights is not None:
        reg = reg * family_weights
```

When provided, family weights amplify the loss for underrepresented chemical families, preventing the model from learning spurious composition→stability correlations.

### 3.5 Loss Flow in Training

```
graph_feats (B, 128)
    → stability_head: Linear(128→64) → LN → SiLU → Dropout → Linear(64→32) → SiLU → Linear(32→1)
    → magnitude_head: Linear(128→64) → LN → SiLU → Dropout → Linear(64→32) → SiLU → Linear(32→1) → Softplus
    → uncertainty_head: Linear(128→32) → SiLU → Linear(32→1)

Outputs:
    p_unstable:     (B,)  ∈ [0, 1]      # Sigmoid output
    eah_magnitude:  (B,)  ∈ [0, ∞)      # Softplus output
    eah_pred:       (B,)  ∈ [0, ∞)      # p_unstable × eah_magnitude
    log_var:        (B,)  ∈ ℝ           # For heteroscedastic uncertainty
```

### 3.6 GradNorm Applied Per-Stage

In `train_v3_li.py`, the TwoStageEahLoss contribution for EaH is included in the GradNorm computation as a **single task loss** (`ts_loss["total"]`). GradNorm treats the two-stage loss as one task, not two separate tasks:

```python
ts_loss = eah_two_stage_loss(eah_out, v)
task_losses["energy_above_hull"] = ts_loss["total"]
```

---

## 4. Weighting Scheme Summary

### 4.1 GradNorm Task Weights

| Task | Initial Weight | Learned Range | Notes |
|------|:-------------:|:-------------:|-------|
| formation_energy | 1.0 | 0.5–2.0 | Dominant task; weight may decrease |
| energy_above_hull | 1.0 | 0.5–3.0 | Hard task; weight typically increases |
| band_gap | 0.4 | 0.1–1.0 | Easier task; weight may decrease |

### 4.2 TwoStageEahLoss Internal Weights

| Component | Weight | Purpose |
|-----------|:------:|---------|
| BCE | 1.0 | Primary: stability classification |
| MSE (unstable) | 3.0 | Primary: magnitude regression |
| MSE (stable) | 0.5 | Regularization: prevent false positives |

### 4.3 PINNLoss Weights (legacy, not active)

| Component | Weight | In Use? |
|-----------|:------:|:-------:|
| data (per task) | w_t (config) | Yes |
| arrhenius | 0.05 | No (tasks not active) |
| thermodynamic | 0.05 | No (log_eah=False) |
| diffusion | 0.1 | No (placeholder) |

---

## 5. Gradient Balancing

### 5.1 GradNorm Update Flow

```
Every 50 batches:
  1. Compute per-task losses L_t (MSE or TwoStageEahLoss)
  2. Store initial losses L_t^(0) on first call
  3. Compute gradient norms ||∇L_t|| for shared backbone
  4. Compute loss ratios L̃_t = L_t / L_t^(0)
  5. Compute targets: target_t = Ḡ · (L̃_t / L̃̄)^α
  6. Update log-weights: Δ log w_t ∝ sign(G_w - target) · ||∇L_t|| · w_t
  7. Renormalize: sum(w_t) = n_tasks
```

### 5.2 Gradient Flow

```
Total Loss = Σ w_t · L_t

    ↓

Backward passes through task_heads (separate parameters)
    → Gradients flow to task-specific parameters

    ↓

Gradients flow through shared backbone
    → GradNorm measures norms at shared backbone
    → GradNorm updates weights to equalize backbone gradient magnitudes
```

### 5.3 Gradient Clipping

Applied after unscaling (AMP), before optimizer step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Global norm clipping at 1.0 — all gradient norms are capped at this value.

### 5.4 Gradient Accumulation

```
Effective batch = BATCH_SIZE × GRAD_ACCUM = 16 × 2 = 32

Loss per micro-batch = total_loss / GRAD_ACCUM  (mean, not sum)
Gradients accumulated over 2 micro-batches
Optimizer step and gradient clipping every 2 batches
```

---

## 6. Ablation Results

### 6.1 Exp A (GradNorm OFF, No Scheduler) — Reference

*Baseline experiment with fixed task weights and constant LR.*

| Metric | formation_energy | energy_above_hull | band_gap |
|--------|:----------------:|:-----------------:|:--------:|
| MAE | — | — | — |
| R² | — | — | — |
| Stability F1 | — | — | — |

### 6.2 Exp B (GradNorm ON, CosineWarmRestarts) — Currently Running

*Experiment with GradNorm adaptive weighting and cosine annealing with warm restarts.*

| Metric | formation_energy | energy_above_hull | band_gap |
|--------|:----------------:|:-----------------:|:--------:|
| MAE | — | — | — |
| R² | — | — | — |
| Stability F1 | — | — | — |

### 6.3 Expected Differences

| Aspect | Exp A (No GradNorm) | Exp B (GradNorm) |
|--------|:-------------------:|:-----------------:|
| EaH loss convergence | Slower (dominated by Ef) | Faster (GradNorm upweights) |
| Ef final MAE | Potentially better (focus) | Potentially worse (balanced) |
| BG final MAE | Potentially worse (overfit) | Potentially better (balanced) |
| Training stability | Lower (Ef dominates) | Higher (balanced grads) |

---

## 7. Complete Loss Computation (train_v3_li.py)

```python
# Per batch:
with torch.amp.autocast("cuda", enabled=use_amp):
    preds = model(cg, lg)  # forward pass

    task_losses = {}
    for task in ["formation_energy", "energy_above_hull", "band_gap"]:
        v = getattr(cg, f"y_{task}")  # target

        if task == "energy_above_hull":
            # Two-stage loss
            eah_out = {
                "eah_pred": preds["energy_above_hull"],
                "p_unstable": preds["p_unstable"],
                "eah_magnitude": preds["eah_magnitude"],
            }
            ts_loss = eah_two_stage_loss(eah_out, v)
            task_losses[task] = ts_loss["total"]
            train_total_loss += ts_loss["total"]
        else:
            # Standard MSE
            loss = mse_loss(preds[task], v)
            task_losses[task] = loss
            train_total_loss += loss

    # GradNorm weighting
    if USE_GRADNORM:
        total_loss = grad_norm.compute_total(task_losses) / GRAD_ACCUM
        if n_batches % 50 == 0:
            grad_norm.update_weights(task_losses, backbone_params, lr=0.025)
    else:
        total_loss = sum(task_losses.values()) / GRAD_ACCUM
```

---

## 8. References

1. Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." *ICML 2018*.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686–707.
3. Choudhary, K., & DeCost, B. (2021). "Atomistic Line Graph Neural Network for improved materials property predictions." *npj Computational Materials*, 7, 185.
