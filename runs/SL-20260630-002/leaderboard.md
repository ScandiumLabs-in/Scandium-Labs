# Leaderboard — SL-20260630-002
*Note: SL-20260630-002 shows **validation** metrics (best epoch per metric). Previous runs show **test** metrics. Direct comparison requires test evaluation after training completes.*

| Run | Ef MAE ↓ | Ef R² ↑ | EaH MAE ↓ | EaH R² ↑ | BG MAE ↓ | BG R² ↑ | Split |
|-----|----------|---------|-----------|----------|----------|---------|-------|
| final_eval | 0.2485 | 0.6825 | 0.1154 | 0.4130 | 0.7833 | 0.3501 | test |
| phase5_final | 0.2471 | 0.7056 | **0.1181** | **0.4092** | **0.7614** | **0.3646** | test |
| phase4_final | 0.2678 | 0.6535 | 0.1201 | 0.3687 | 0.8041 | 0.2805 | test |
| **SL-20260630-002** (ep 50/50/13/62/59/59) | 0.5684 | 0.5359 | 0.1256 | 0.3750 | 1.0479 | 0.2924 | **val** |

## Gap to phase5_final (val→test comparison is indicative only)

| Metric | phase5 (test) | SL-002 best val | Gap |
|--------|--------------|-----------------|-----|
| Ef MAE | 0.2471 | 0.5684 | +0.3213 |
| EaH MAE | 0.1181 | 0.1256 | +0.0075 |
| BG MAE | 0.7614 | 1.0479 | +0.2865 |

> ⚠️  These MAEs are **not directly comparable** — val metrics are typically better than test. After training finishes, run test evaluation to get a true comparison.

## Run Status
- Epochs completed: 67 / 150
- Patience: 40 (best val loss epoch 59)
- ETA to patience exhaustion: ~194 min (if no improvement)
