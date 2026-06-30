# Verification Checklist — Normalized Model Pipeline

## 1. Preprocessing
- [✅] Same data source (datasets/v2_10000) for both normalized and non-normalized
- [✅] Same prebuilt graphs (prebuilt_graphs.pt) — identical input features
- [✅] Same normalizer.json (fitted on full dataset) loaded by both trainers
- [✅] Normalizer stats confirmed: Ef mean=-0.986, Eah mean=0.184, BG mean=1.303
- [✅] Log-transform not used (log_eah=False) — confirmed in both checkpoints

## 2. Training Pipeline (Normalized)
- [✅] Trainer normalizes targets: `self.normalizer.normalize(raw_targets)` at line 113
- [✅] Trainer denormalizes predictions for validation metrics: `preds * std + mean` at line 181
- [✅] Model trains in normalized space, outputs z-scores
- [✅] Config identical to non-normalized except training added `normalize_targets=True`

## 3. Training Pipeline (Non-Normalized)
- [✅] Trainer loaded normalizer but did NOT call normalize() — raw targets used
- [✅] Model outputs in raw energy units (eV/atom)
- [✅] No denormalization applied during validation

## 4. Inference Pipeline (Normalized)
- [✅] `InferenceEngine._load_model()` reads `normalize_targets` flag from checkpoint config
- [✅] Flag set to `True` for normalized checkpoint (added post-training)
- [✅] `predict_single()` applies denormalization: `value * (std + 1e-8) + mean` when flag is True
- [✅] Normalizer loaded from checkpoint-adjacent normalizer.json
- [✅] Denormalization applied before gating/stability checks

## 5. Inference Pipeline (Non-Normalized)
- [✅] `normalize_targets=False` in checkpoint config
- [✅] `predict_single()` skips denormalization — raw outputs used directly

## 6. Benchmark Consistency
- [✅] Same 54 MATERIALS list used across all three evaluations
- [✅] Same `pymatgen` structure generators
- [✅] Same `InferenceEngine` class (only flag-driven behavior differs)
- [✅] Same CPU device for all evaluations
- [✅] Eah nonnorm-vs-norm correlation = 0.0286 (confirmed independent predictions)

## 7. Internal Test Metrics (from trainer evaluate)
- [✅] v1:  Ef MAE=0.505,  Eah MAE=0.193,  BG MAE=1.017  (n=817)
- [✅] corrected-split:  Ef MAE=0.360,  Eah MAE=0.176,  BG MAE=0.787  (n=3635)
- [✅] corrected-split+norm:  Ef MAE=0.240,  Eah MAE=0.113,  BG MAE=0.801  (n=3635)

## 8. Benchmark Degradation Analysis
- NonNorm: test Eah=0.176 → benchmark Eah MAE=0.359 (**2.0x degradation**)
- Norm:    test Eah=0.113 → benchmark Eah MAE=0.586 (**5.2x degradation**)
- Conclusion: Synthetic benchmark materials are OOD for both models.
  Norm model is more sensitive to distribution shift.

## 9. Key Config Differences (Norm vs NonNorm)
| Setting             | NonNorm | Norm  |
|--------------------:|:-------:|:-----:|
| hidden_dim          | 128     | 128   |
| num_alignn_layers   | 2       | 2     |
| num_transformer_layers | 1    | 1     |
| batch_size          | 8       | 8     |
| learning_rate       | 0.001   | 0.001 |
| max_epochs          | 100     | 100   |
| target normalization | No     | Yes   |
| tasks               | same    | same  |
| task weights        | same    | same  |

Only change: target normalization during training.
