# Training Summary — SL-20260630-002

**Current Epoch:** 70
**Status:** Running
**Last Updated:** 2026-07-01T00:31:38.953129

## Latest Epoch

| Metric | Value |
|--------|-------|
| Train Loss | 2.0921 |
| Val Loss | 3.4181 |
| Epoch Time | 352.2s |
| formation_energy MAE | 0.5694 |
| formation_energy R² | 0.5299 |
| energy_above_hull MAE | 0.1415 |
| energy_above_hull R² | 0.3736 |
| band_gap MAE | 1.0859 |
| band_gap R² | 0.2485 |
| GradNorm Weights | {'band_gap': 3.3269, 'energy_above_hull': 0.3364, 'formation_energy': 0.0002} |
| GPU Memory | 666 MB |
| Throughput | 47.2 g/s |

## Best So Far

| Metric | Best Value | Epoch |
|--------|-----------|-------|
| Train Loss (min) | 2.0741 | 66 |
| Val Loss (min) | 3.3163 | 59 |
| formation_energy MAE min | 0.5684 | 50 |
| formation_energy R2 max | 0.5359 | 50 |
| energy_above_hull MAE min | 0.1256 | 13 |
| energy_above_hull R2 max | 0.3750 | 62 |
| band_gap MAE min | 1.0479 | 59 |
| band_gap R2 max | 0.2924 | 59 |

## Comparison vs Previous Experiments

| Run | Ef MAE | Ef R² | EaH MAE | EaH R² | BG MAE | BG R² |
|-----|--------|-------|---------|--------|--------|-------|
| final_eval | 0.24851323664188385 | 0.6825149059295654 | 0.11539901793003082 | 0.41297316551208496 | 0.7833346724510193 | 0.35012519359588623 |
| phase4_final | 0.26781073212623596 | 0.6534528732299805 | 0.1201116219162941 | 0.3686751127243042 | 0.8040952086448669 | 0.28048568964004517 |
| phase5_final | 0.247115358710289 | 0.7055881023406982 | 0.11814559996128082 | 0.4091660976409912 | 0.7613657116889954 | 0.36457592248916626 |
| v3_upgraded | 0.7673068046569824 | -0.5295946598052979 | 0.18706099689006805 | -0.0471644401550293 | 1.7981042861938477 | -1.3906745910644531 |
| **SL-20260630-002 (best so far)** | 0.568403422832489 | 0.5358825922012329 | 0.12561039626598358 | 0.3749995827674866 | 1.0478742122650146 | 0.2923750877380371 |