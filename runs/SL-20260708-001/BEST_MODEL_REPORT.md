# Best Model Report — SL-20260708-001

**Generated:** 2026-07-08T18:12:30.110471

## Test Set Results

| Task | MAE ↓ | RMSE ↓ | R² ↑ |
|------|-------|--------|------|
| formation_energy | 0.3154 | 0.5704 | 0.5121 |
| energy_above_hull | 0.0973 | 0.3952 | 0.1771 |
| band_gap | 1.2339 | 1.5013 | 0.0692 |

## Two-Stage EaH Metrics

| Metric | Value |
|--------|-------|
| Stability F1 | 0.9537 |
| Precision | 0.9190 |
| Recall | 0.9911 |
| EaH MAE (all) | 0.0973 |
| EaH MAE (unstable) | 0.0999 |

## Best Epochs per Metric

| Metric | Best Value | Epoch |
|--------|-----------|-------|
| Val Loss (min) | 3.0941 | 98 |
| Ef MAE (min) | 0.5222 | 136 |
| Ef R² (max) | 0.5871 | 135 |
| EaH MAE (min) | 0.1280 | 4 |
| EaH R² (max) | 0.3854 | 131 |
| BG MAE (min) | 1.0252 | 98 |
| BG R² (max) | 0.3385 | 98 |

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs | 139 |
| GPU hours | 15.83 |
| Parameters | 1,281,321 |
| Hidden dim | 128 |
| ALIGNN layers | 4 |
| Transformer layers | 2 |
| Batch size | 16 |