# Training Summary — SL-20260630-002

**Current Epoch:** 37
**Status:** Running
**Last Updated:** 2026-06-30T21:16:57.718583

## Latest Epoch

| Metric | Value |
|--------|-------|
| Train Loss | 2.2567 |
| Val Loss | 3.4651 |
| Epoch Time | 354.3s |
| formation_energy MAE | 0.5766 |
| formation_energy R² | 0.5238 |
| energy_above_hull MAE | 0.1391 |
| energy_above_hull R² | 0.3692 |
| band_gap MAE | 1.1497 |
| band_gap R² | 0.2333 |
| GradNorm Weights | {'band_gap': 4.2928, 'energy_above_hull': 0.8377, 'formation_energy': 0.1891} |
| GPU Memory | 604 MB |
| Throughput | 47.0 g/s |

## Best So Far

| Metric | Best Value | Epoch |
|--------|-----------|-------|
| Train Loss (min) | 2.2288 | 36 |
| Val Loss (min) | 3.4568 | 36 |
| formation_energy MAE min | 0.5743 | 31 |
| formation_energy R2 max | 0.5276 | 36 |
| energy_above_hull MAE min | 0.1256 | 13 |
| energy_above_hull R2 max | 0.3740 | 35 |
| band_gap MAE min | 1.0949 | 13 |
| band_gap R2 max | 0.2395 | 36 |

## Comparison vs Previous Experiments

| Run | Ef MAE | Ef R² | EaH MAE | EaH R² | BG MAE | BG R² |
|-----|--------|-------|---------|--------|--------|-------|
| final_eval | 0.24851323664188385 | 0.6825149059295654 | 0.11539901793003082 | 0.41297316551208496 | 0.7833346724510193 | 0.35012519359588623 |
| phase4_final | 0.26781073212623596 | 0.6534528732299805 | 0.1201116219162941 | 0.3686751127243042 | 0.8040952086448669 | 0.28048568964004517 |
| phase5_final | 0.247115358710289 | 0.7055881023406982 | 0.11814559996128082 | 0.4091660976409912 | 0.7613657116889954 | 0.36457592248916626 |
| v3_upgraded | 0.7673068046569824 | -0.5295946598052979 | 0.18706099689006805 | -0.0471644401550293 | 1.7981042861938477 | -1.3906745910644531 |
| **SL-20260630-002 (best so far)** | 0.5742859244346619 | 0.5275540351867676 | 0.12561039626598358 | 0.37399256229400635 | 1.094860553741455 | 0.23946768045425415 |