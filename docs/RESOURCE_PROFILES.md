# Resource Profiles

Configurations optimized for different GPU tiers.

## Small — GTX 1650 (4 GB) [CURRENT]

| Setting | Value | VRAM |
|---------|-------|------|
| hidden_dim | 128 | 470 MB |
| ALIGNN layers | 4 | |
| Transformer layers | 2 | |
| Attention heads | 4 | |
| Batch size | 16 | |
| Gradient accumulation | 2 | |
| Dropout | 0.15 | |
| Gradient checkpointing | Enabled | |
| Mixed precision | fp16 | |
| Optimizer | AdamW | |

Total VRAM estimate: 470 MB model + 300 MB overhead + 500 MB data ≈ 1.3 GB (32% of 4 GB)

## Medium — RTX 3060 (12 GB)

| Setting | Value | VRAM |
|---------|-------|------|
| hidden_dim | 256 | ~1.6 GB |
| ALIGNN layers | 6 | |
| Transformer layers | 4 | |
| Attention heads | 8 | |
| Batch size | 32 | |
| Gradient accumulation | 4 | |
| Dropout | 0.15 | |
| Gradient checkpointing | Enabled | |
| Mixed precision | fp16 | |
| Optimizer | AdamW | |

## Large — RTX 4090 / A100 (24-80 GB)

| Setting | Value | VRAM |
|---------|-------|------|
| hidden_dim | 512 | ~4.5 GB |
| ALIGNN layers | 8 | |
| Transformer layers | 6 | |
| Attention heads | 8 | |
| Batch size | 64 | |
| Gradient accumulation | 8 | |
| Dropout | 0.1 | |
| Gradient checkpointing | Disabled (speed > VRAM) | |
| Mixed precision | bf16 (A100) or fp16 | |
| Optimizer | AdamW + schedule-free | |

## Scaling Rules

- `hidden_dim` × 2 → VRAM × 4 (both params and activations)
- `num_alignn_layers` × 2 → VRAM × 2 (activations dominate)
- `batch_size` × 2 → VRAM × 2 (activations)
- GC trades ~33% speed for 2.4× VRAM savings
- Each layer adds ~0.3M params at hidden_dim=128
