# Data Directory

| File | Description |
|------|-------------|
| `baseline_v1.0.json` | Original v1 baseline (817 structures, deprecated). Not used for current comparisons. |
| `normalizer.json` | Z-score normalization stats (mean, std) per target. Used by training and inference. |
| `benchmark_cifs/` | Reference material CIF files for sanity-check testing (e.g., Li6PS5Cl). |

Current evaluations use test results from tracked experiment runs in `runs/<run_id>/test_results.json`.
