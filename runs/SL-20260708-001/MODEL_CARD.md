# Model Card — SL-20260708-001

## Model Details

- **Architecture:** ScandiumPINNGNN-v3-Li
- **Backbone:** ALIGNN (4 layers) + GraphTransformer (2 layers)
- **Hidden dimension:** 128
- **Attention heads:** 4
- **Dropout:** 0.15
- **Gradient checkpointing:** auto
- **Two-stage EaH:** True
- **Parameters:** 1,281,321

## Dataset

- **Dataset:** v3_li_10000
- **Cutoff:** 8.0 Å
- **Max neighbors:** 16
- **RBF features:** 64
- **SBF features:** 32

## Training Procedure

- **Optimizer:** AdamW
- **Learning rate:** 0.0005
- **Warmup steps:** 500
- **Scheduler:** cosine_with_restarts
- **Batch size:** 16
- **Gradient accumulation:** 2
- **Weight decay:** 1e-05
- **Mixed precision:** True
- **Gradient clipping:** 1.0
- **GradNorm alpha:** 1.5
- **Max epochs:** 150
- **Patience:** 40

## Hardware

- **GPU:** NVIDIA GeForce GTX 1650
- **CUDA:** 12.4
- **PyTorch:** 2.6.0+cu124
- **Training time:** 15.83 GPU-hours

## Performance

| Task | MAE | RMSE | R² |
|------|-----|------|----|
| formation_energy | 0.3154 | 0.5704 | 0.5121 |
| energy_above_hull | 0.0973 | 0.3952 | 0.1771 |
| band_gap | 1.2339 | 1.5013 | 0.0692 |

## Intended Use

This model is designed for high-throughput screening of Li-containing
solid-state electrolyte candidates. It predicts formation energy,
energy above hull, and band gap from crystal structure.

## Limitations

- Only trained on Li-containing materials (Li ≥ 5 at.%)
- Does not predict ionic conductivity or activation energy directly
- Limited to Materials Project data (DFT-computed properties)
- Uncertainty estimates via MC Dropout may not be well-calibrated
- Model size is small (1.28M params) — scaling may improve performance
