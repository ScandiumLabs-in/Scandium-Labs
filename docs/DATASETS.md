# Scandium Labs Datasets

## Overview

Scandium Labs curates a collection of Li-containing solid electrolyte crystal structures and computed properties sourced primarily from the Materials Project (MP). Each dataset version is a self-contained directory containing serialized `pymatgen` structures, target property dictionaries, train/val/test split indices, and a fitted `PropertyNormalizer` for standardization.

The core collection target is Li-containing inorganic solids (Li ≥ 5 at.%) for solid-state battery electrolyte screening.

---

## Dataset Versions

| Version | Path | N Structures | Val/Test Split | Source | Elements | Key Difference |
|---------|------|-------------|----------------|--------|----------|----------------|
| **v1_817** | `datasets/v1_817/` | 817 (train: 653, val: 82, test: 82) | 10%/10% | MP + OQMD | Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba | Initial prototype; multi-source; low-N |
| **v2_10000** | `datasets/v2_10000/` | 3,635 (train: 2,814, val: 255, test: 566) | 7%/15% | MP only | 49 elements (Li, alkali, alkaline earth, transition metals, main group) | Full Materials Project query; broader chemistry |
| **v2_1000_smoketest** | `datasets/v2_1000_smoketest/` | 1,008 (train: 798, val: 105, test: 105) | 10%/10% | MP only | Li, Na, K, Mg, Ca, Sr, Ba | Subset for rapid CI testing |
| **v2_10000_log_eah** | `datasets/v2_10000_log_eah/` | 3,635 | same as v2_10000 | MP only | 49 elements | Same data as v2_10000 with log-transformed EaH targets |
| **v3_li** | — | 10,000 (subsampled from 20k) | 10%/10% | MP only | All (Li ≥ 5% filter) | Li-focused expansion; larger pool |

> **Note:** The `v2_10000` dataset name reflects the target collection size (10,000), but actual retrieved count after filtering and deduplication is 3,635. The full MP query returned fewer Li-containing structures that passed quality filters.

---

## Data Collection

### Collector Classes

**File:** `src/data/collectors.py`

| Collector | Source | API | Key Method |
|-----------|--------|-----|-----------|
| `MaterialsProjectCollector` | Materials Project | `mp_api.client.MPRester` | `collect(elements, fields, max_results)` |
| `JARVISCollector` | JARVIS-DFT | `jarvis.db.figshare` | `collect(dataset_name)` |
| `OQMDCollector` | OQMD | REST API | `collect(limit, offset)` |
| `AFLOWCollector` | AFLOW | REST API (`aflux/`) | `collect(elements, max_results)` |
| `NOMADCollector` | NOMAD | REST API | `collect(elements, page_size, max_entries)` |

### Materials Project Collection (Primary)

`MaterialsProjectCollector` queries `mpr.materials.summary.search()` with:
- **Elements filter:** Li (or broader 49-element set with Li ≥ 5% threshold applied post-query).
- **Requested fields:** `material_id`, `formula_pretty`, `structure`, `formation_energy_per_atom`, `energy_above_hull`, `band_gap`, `volume`, `density`, `symmetry`, `is_stable`.
- **Chunking:** Auto-chunked in units of 1,000, up to 5 chunks for 5,000 total max per call.
- **API key:** Read from `MP_API_KEY` or `MATERIALS_PROJECT_API_KEY` environment variables.

Additional collectors exist for OQMD, JARVIS-DFT, AFLOW, and NOMAD but the primary production datasets use MP only.

### Sulfide-Specific Data

Pre-curated lists of known sulfide solid electrolytes (e.g., `Li6PS5Cl`, `Li10GeP2S12`, `Li3PS4`, `Li7P3S11`) are defined as `KNOWN_SULFIDES` and used for targeted filtering.

---

## Target Properties

| Property | Key | Unit | Source Field | Notes |
|----------|-----|------|-------------|-------|
| Ionic conductivity | `log_ionic_conductivity` | log₁₀(S/cm) | Computed (MD/NEB) | Primary target; log-scaled |
| Activation energy | `activation_energy` | eV | Computed | Arrhenius-related; linked to conductivity via PINN loss |
| Formation energy | `formation_energy` | eV/atom | `formation_energy_per_atom` | Thermodynamic stability indicator |
| Energy above hull | `energy_above_hull` | eV/atom | `energy_above_hull` | 0 = on convex hull (stable); may be log-transformed |
| Band gap | `band_gap` | eV | `band_gap` | Electronic insulator property |

### Log-Transformed EaH

The `v2_10000_log_eah` variant applies `eah_log = log(eah + ε)` transformation to handle the heavy-tailed distribution of EaH values (most materials near hull, few very unstable). The `log_eah` flag in config controls this transformation in the loss function.

---

## Preprocessing

### Data Cleaning

**File:** `src/data/cleaner.py:6`

`DataCleaner.clean()` applies the following pipeline:

1. **Drop NaNs** on required columns (`formation_energy_per_atom`, `structure`).
2. **Energy filtering:** `-10 ≤ formation_energy_per_atom ≤ 5 eV/atom`, `energy_above_hull ≥ 0`.
3. **Size filtering:** Structures with 2–200 sites.
4. **Deduplication:** `StructureMatcher`-based (ltol=0.2, stol=0.3, angle_tol=5°) pairwise comparison.

### Normalization

**File:** `src/data/cleaner.py:48`

`PropertyNormalizer` implements z-score standardization:

- **`fit(df, columns)`:** Computes mean, std, min, max per target column.
- **`transform(df)` / `normalize(raw_targets)`:** Applies `(x - mean) / (std + 1e-8)`.
- **`inverse_transform(values, col)` / `denormalize(predictions)`:** Reverses normalization.
- **Persistence:** Save/load to/from `normalizer.json` (stored in dataset directory).

Normalization statistics are fit on the training split only to prevent data leakage.

### Outlier Handling

Energy bounds (`ef_range: [-10, 5]`) act as hard filters. No additional outlier clipping is applied, allowing the model to learn the full distribution.

### NaN Imputation

NaN target values are preserved as `torch.tensor(float("nan"))` in dataset tensors and masked during loss computation. No imputation is performed.

---

## Dataset Splits

### Composition-Based Split

Splits are computed using `GroupShuffleSplit` grouped by chemical family (from `family_id.py`). This ensures:

- All materials from a given family appear in only one split.
- Realistic generalization assessment (no compositionally similar structures across train/test).

### Default Ratios

| Split | v1_817 | v2_10000 | v2_1000_smoketest |
|-------|--------|----------|-------------------|
| Train | 80% (653) | 77% (2,814) | 79% (798) |
| Val | 10% (82) | 7% (255) | 10% (105) |
| Test | 10% (82) | 16% (566) | 10% (105) |

### Split Indices

Per-dataset JSON metadata files record `n_train`, `n_val`, `n_test` counts. Actual index files (e.g., `train_indices.npy`) are stored in each dataset directory alongside the structures.

---

## Chemical Families

**File:** `src/chemistry/family_id.py`

Seven chemical families are identified from composition, with mixed-anion separation:

| Family | Numeric ID | Example | Condition |
|--------|-----------|---------|-----------|
| `pure_halide` | 0 | LiCl, LiBr | F/Cl/Br/I, no O or S |
| `oxyhalide` | 1 | Li3OCl | Halide + O |
| `sulfohalide` | 2 | Li6PS5Cl | Halide + S |
| `oxide` | 3 | Li7La3Zr2O12 | O, no P, no halide |
| `sulfide` | 4 | Li3PS4 | S, no halide |
| `phosphate` | 5 | LiFePO4 | O + P, no halide |
| `other` | 6 | — | None of the above |

Family labels are used for:
- **Dataset splitting** — `GroupShuffleSplit` to prevent family leakage.
- **Loss weighting** — `TwoStageEahLoss` accepts optional per-sample family weights.

---

## Dataset Format

Each dataset directory contains:

```
datasets/{version}/
├── metadata.json              # Version, hash, timestamps, config, stats
├── structures.pt              # List of pymatgen Structure objects
├── targets.pt                 # Dict[str, List[float]] — per-task target arrays
├── train_indices.npy          # Training split indices
├── val_indices.npy            # Validation split indices
├── test_indices.npy           # Test split indices
├── normalizer.json            # Per-task mean/std for z-score normalization
├── normalizer_v2.json         # (v1_817 only) v2-normalizer for backward compat
└── data/                      # (optional) pre-cached per-structure graphs
```

### metadata.json Structure

```json
{
  "version": "v2_10000",
  "dataset_hash": "sha256-of-contents",
  "timestamp": "2026-06-29T00:41:24",
  "sources": ["mp"],
  "elements_requested": ["Li", "Na", ...],
  "n_structures": 3635,
  "n_train": 2814,
  "n_val": 255,
  "n_test": 566,
  "min_atoms": 2,
  "max_atoms": 200,
  "ef_range": [-10.0, 5.0],
  "deduplicated": true,
  "normalized": true,
  "graphs_cached": false,
  "config": { ... },
  "baseline_metrics": { ... }     // (v1_817 only)
}
```

### Dataset Loading

Two dataset classes in `src/data/dataset.py`:

- **`SolidElectrolyteDataset`:** In-memory; builds graphs on-the-fly via `graph_builder.build(structure)`, featurizes, attaches targets. Primary training dataset.
- **`LazyGraphDataset`:** Supports disk-cached graphs; falls back to on-the-fly building; loads from individual `.pt` files or monolithic `prebuilt_graphs.pt`.

Both classes produce `(crystal_graph, line_graph)` tuples collated via `collate_fn()` using `torch_geometric.data.Batch.from_data_list()`.
