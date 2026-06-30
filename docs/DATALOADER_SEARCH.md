# DataLoader Benchmark Results

Benchmark date: 2026-06-30 18:02:26

## Configuration

- **Device:** `cuda`
- **PyTorch:** `2.12.0+cu130`
- **CPUs:** `12`
- **GPU:** `NVIDIA GeForce GTX 1650`
- **Batch size:** `16`
- **Batches per run:** `100`
- **Repeats per config:** `3`
- **Dataset:** `v3_li_10000` (8310 train samples)

## Fastest Configuration (Overall)

| Parameter | Value |
|-----------|-------|
| `num_workers` | `3` |
| `prefetch_factor` | `2` |
| `pin_memory` | `True` |
| `persistent_workers` | `True` |
| `multiprocessing_context` | `fork` |
| **Throughput** | **543.2 graphs/s** |
| **Time per 100 batches** | **2.9s ± 0.1s** |
| **Samples per second** | **543.2** |

## Fastest per `num_workers`

| Workers | Throughput (graphs/s) | Config |
|---------|----------------------|--------|
| 0 | 265.9 |  PF=4 PM=False |
| 1 | 333.1 |  PF=4 PM=False PW=True CTX=fork |
| 2 | 511.8 |  PF=None PM=True PW=True CTX=fork |
| 3 | 543.2 |  PF=2 PM=True PW=True CTX=fork |
| 4 | 442.9 |  PF=4 PM=True PW=True CTX=def |

## All Results

| # | Workers | PF | PinMem | PersistW | Ctx | Time (s) | ±Std | Graphs/s | ±Std |
|---|---------|----|--------|----------|-----|----------|------|----------|------|
| 1 | 0 | 2 | True | False | - | 11.3 | 4.8 | 165.1 | 54.3 |
| 2 | 0 | 2 | False | False | - | 6.8 | 0.4 | 234.6 | 11.7 |
| 3 | 0 | 4 | True | False | - | 6.3 | 0.2 | 255.5 | 8.9 |
| 4 | 0 | 4 | False | False | - | 6.0 | 0.2 | 265.9 | 7.9 |
| 5 | 0 | None | True | False | - | 6.3 | 0.1 | 254.4 | 3.6 |
| 6 | 0 | None | False | False | - | 6.4 | 0.0 | 249.4 | 1.7 |
| 7 | 1 | 2 | True | True | fork | 6.2 | 0.1 | 256.2 | 4.0 |
| 8 | 1 | 2 | True | True | - | 5.1 | 0.2 | 313.0 | 10.0 |
| 9 | 1 | 2 | True | False | fork | 5.4 | 0.2 | 298.1 | 12.3 |
| 10 | 1 | 2 | True | False | - | 6.0 | 0.1 | 267.3 | 2.6 |
| 11 | 1 | 2 | False | True | fork | 6.1 | 0.1 | 262.9 | 3.5 |
| 12 | 1 | 2 | False | True | - | 6.2 | 0.1 | 258.1 | 5.1 |
| 13 | 1 | 2 | False | False | fork | 5.9 | 0.5 | 272.3 | 26.5 |
| 14 | 1 | 2 | False | False | - | 5.1 | 0.1 | 316.3 | 6.1 |
| 15 | 1 | 4 | True | True | fork | 5.8 | 0.3 | 276.2 | 17.0 |
| 16 | 1 | 4 | True | True | - | 6.1 | 0.1 | 261.0 | 2.7 |
| 17 | 1 | 4 | True | False | fork | 6.2 | 0.1 | 257.2 | 2.3 |
| 18 | 1 | 4 | True | False | - | 5.4 | 0.6 | 301.3 | 31.5 |
| 19 | 1 | 4 | False | True | fork | 4.8 | 0.0 | 333.1 | 3.4 |
| 20 | 1 | 4 | False | True | - | 6.0 | 0.2 | 266.6 | 8.0 |
| 21 | 1 | 4 | False | False | fork | 6.3 | 0.1 | 254.0 | 2.4 |
| 22 | 1 | 4 | False | False | - | 6.7 | 0.5 | 240.2 | 18.1 |
| 23 | 1 | None | True | True | fork | 5.2 | 0.5 | 308.8 | 29.1 |
| 24 | 1 | None | True | True | - | 5.0 | 0.2 | 321.3 | 11.5 |
| 25 | 1 | None | True | False | fork | 6.2 | 0.3 | 256.7 | 13.9 |
| 26 | 1 | None | True | False | - | 6.5 | 0.1 | 246.9 | 2.6 |
| 27 | 1 | None | False | True | fork | 6.0 | 0.2 | 265.2 | 9.0 |
| 28 | 1 | None | False | True | - | 5.7 | 0.3 | 281.0 | 15.5 |
| 29 | 1 | None | False | False | fork | 5.7 | 0.7 | 282.9 | 34.2 |
| 30 | 1 | None | False | False | - | 6.6 | 0.5 | 245.0 | 19.7 |
| 31 | 2 | 2 | True | True | fork | 4.5 | 0.3 | 356.6 | 24.5 |
| 32 | 2 | 2 | True | True | - | 4.2 | 0.1 | 378.5 | 11.2 |
| 33 | 2 | 2 | True | False | fork | 5.4 | 1.0 | 303.6 | 51.3 |
| 34 | 2 | 2 | True | False | - | 3.7 | 0.2 | 433.2 | 20.0 |
| 35 | 2 | 2 | False | True | fork | 4.4 | 0.1 | 366.9 | 11.8 |
| 36 | 2 | 2 | False | True | - | 5.1 | 0.1 | 314.1 | 7.4 |
| 37 | 2 | 2 | False | False | fork | 5.1 | 0.1 | 313.3 | 3.3 |
| 38 | 2 | 2 | False | False | - | 5.1 | 0.1 | 311.8 | 4.7 |
| 39 | 2 | 4 | True | True | fork | 5.0 | 0.1 | 321.7 | 4.2 |
| 40 | 2 | 4 | True | True | - | 3.8 | 0.7 | 426.7 | 64.4 |
| 41 | 2 | 4 | True | False | fork | 3.6 | 0.1 | 447.4 | 10.0 |
| 42 | 2 | 4 | True | False | - | 4.5 | 0.7 | 366.7 | 56.9 |
| 43 | 2 | 4 | False | True | fork | 5.7 | 0.0 | 279.7 | 1.6 |
| 44 | 2 | 4 | False | True | - | 5.8 | 0.1 | 274.4 | 2.5 |
| 45 | 2 | 4 | False | False | fork | 5.3 | 0.5 | 303.1 | 28.3 |
| 46 | 2 | 4 | False | False | - | 4.8 | 0.0 | 333.7 | 3.1 |
| 47 | 2 | None | True | True | fork | 3.1 | 0.0 | 511.8 | 6.7 |
| 48 | 2 | None | True | True | - | 3.3 | 0.0 | 486.0 | 1.8 |
| 49 | 2 | None | True | False | fork | 4.0 | 0.8 | 417.8 | 75.0 |
| 50 | 2 | None | True | False | - | 5.0 | 0.7 | 325.6 | 43.6 |
| 51 | 2 | None | False | True | fork | 4.9 | 0.0 | 325.8 | 1.1 |
| 52 | 2 | None | False | True | - | 5.0 | 0.1 | 322.2 | 6.2 |
| 53 | 2 | None | False | False | fork | 5.2 | 0.1 | 310.4 | 4.1 |
| 54 | 2 | None | False | False | - | 4.9 | 0.4 | 326.3 | 29.7 |
| 55 | 3 | 2 | True | True | fork | 2.9 | 0.1 | 543.2 | 14.7 |
| 56 | 3 | 2 | True | True | - | 7.3 | 3.4 | 286.9 | 155.8 |
| 57 | 3 | 2 | True | False | fork | 4.2 | 0.2 | 380.7 | 15.2 |
| 58 | 3 | 2 | True | False | - | 4.1 | 0.0 | 393.6 | 1.2 |
| 59 | 3 | 2 | False | True | fork | 4.9 | 0.0 | 324.7 | 2.6 |
| 60 | 3 | 2 | False | True | - | 4.6 | 0.1 | 347.2 | 6.4 |
| 61 | 3 | 2 | False | False | fork | 4.1 | 0.0 | 393.2 | 3.5 |
| 62 | 3 | 2 | False | False | - | 4.2 | 0.1 | 383.5 | 7.5 |
| 63 | 3 | 4 | True | True | fork | 3.8 | 0.4 | 428.2 | 43.5 |
| 64 | 3 | 4 | True | True | - | 4.1 | 0.0 | 395.0 | 4.0 |
| 65 | 3 | 4 | True | False | fork | 4.3 | 0.0 | 376.0 | 2.6 |
| 66 | 3 | 4 | True | False | - | 4.4 | 0.0 | 367.6 | 4.1 |
| 67 | 3 | 4 | False | True | fork | 5.1 | 0.0 | 316.3 | 0.0 |
| 68 | 3 | 4 | False | True | - | 5.2 | 0.1 | 305.5 | 6.9 |
| 69 | 3 | 4 | False | False | fork | 4.5 | 0.6 | 360.3 | 47.2 |
| 70 | 3 | 4 | False | False | - | 4.9 | 0.4 | 329.7 | 27.1 |
| 71 | 3 | None | True | True | fork | 3.1 | 0.0 | 509.9 | 7.2 |
| 72 | 3 | None | True | True | - | 3.6 | 0.4 | 447.5 | 42.4 |
| 73 | 3 | None | True | False | fork | 4.7 | 0.0 | 340.5 | 2.7 |
| 74 | 3 | None | True | False | - | 4.7 | 0.1 | 342.0 | 7.0 |
| 75 | 3 | None | False | True | fork | 5.5 | 0.1 | 289.1 | 3.5 |
| 76 | 3 | None | False | True | - | 5.6 | 0.1 | 283.4 | 4.1 |
| 77 | 3 | None | False | False | fork | 5.7 | 0.1 | 282.5 | 5.3 |
| 78 | 3 | None | False | False | - | 5.9 | 0.0 | 272.6 | 0.8 |
| 79 | 4 | 2 | True | True | fork | 4.5 | 0.0 | 354.4 | 1.8 |
| 80 | 4 | 2 | True | True | - | 4.4 | 0.1 | 359.8 | 4.4 |
| 81 | 4 | 2 | True | False | fork | 4.6 | 0.1 | 347.1 | 6.7 |
| 82 | 4 | 2 | True | False | - | 4.2 | 0.6 | 390.8 | 64.8 |
| 83 | 4 | 2 | False | True | fork | 4.7 | 0.2 | 340.3 | 13.1 |
| 84 | 4 | 2 | False | True | - | 5.8 | 0.0 | 277.2 | 1.5 |
| 85 | 4 | 2 | False | False | fork | 5.3 | 0.4 | 302.1 | 25.6 |
| 86 | 4 | 2 | False | False | - | 5.0 | 0.4 | 320.5 | 22.6 |
| 87 | 4 | 4 | True | True | fork | 3.8 | 0.2 | 423.1 | 26.1 |
| 88 | 4 | 4 | True | True | - | 3.6 | 0.1 | 442.9 | 14.0 |
| 89 | 4 | 4 | True | False | fork | 3.8 | 0.1 | 420.3 | 5.6 |
| 90 | 4 | 4 | True | False | - | 3.9 | 0.1 | 411.5 | 6.1 |
| 91 | 4 | 4 | False | True | fork | 5.1 | 0.1 | 316.0 | 7.7 |
| 92 | 4 | 4 | False | True | - | 4.4 | 0.3 | 364.3 | 21.5 |
| 93 | 4 | 4 | False | False | fork | 4.4 | 0.1 | 364.8 | 7.3 |
| 94 | 4 | 4 | False | False | - | 4.5 | 0.0 | 357.3 | 0.8 |
| 95 | 4 | None | True | True | fork | 3.7 | 0.3 | 440.5 | 35.2 |
| 96 | 4 | None | True | True | - | 3.8 | 0.1 | 426.1 | 7.4 |
| 97 | 4 | None | True | False | fork | 4.4 | 0.8 | 374.9 | 58.0 |
| 98 | 4 | None | True | False | - | 4.0 | 0.1 | 401.7 | 5.9 |
| 99 | 4 | None | False | True | fork | 5.2 | 0.2 | 310.1 | 12.0 |
| 100 | 4 | None | False | True | - | 5.0 | 0.1 | 319.4 | 3.7 |
| 101 | 4 | None | False | False | fork | 4.9 | 0.0 | 325.8 | 1.4 |
| 102 | 4 | None | False | False | - | 4.9 | 0.1 | 326.2 | 9.0 |

## Notes

- `PF` = `prefetch_factor`, `PM` = `pin_memory`, `PW` = `persistent_workers`, `CTX` = `multiprocessing_context`
- `-` in PersistW/Ctx means the parameter was not passed (num_workers=0 or default)
- All measurements include `to(device)` + `torch.cuda.synchronize()` (full pipeline)
- 3 repeats per config, mean ± standard deviation reported

