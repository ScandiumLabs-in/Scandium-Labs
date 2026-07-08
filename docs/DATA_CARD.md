# Data Card: Scandium Labs — v3_li_10000

Following the HuggingFace Dataset Card standard.

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Data Sources](#2-data-sources)
3. [Collection Methodology](#3-collection-methodology)
4. [Preprocessing](#4-preprocessing)
5. [Splits](#5-splits)
6. [Target Properties](#6-target-properties)
7. [Distribution Statistics](#7-distribution-statistics)
8. [Normalization](#8-normalization)
9. [Known Biases](#9-known-biases)
10. [Licensing](#10-licensing)
11. [Dataset Versions](#11-dataset-versions)
12. [Maintenance and Updates](#12-maintenance-and-updates)

---

## 1. Dataset Description

**Dataset Name:** `v3_li_10000`
**Current Version:** v3 (active)
**Size:** 10,000 crystalline structures
**Domain:** Li-containing inorganic solids (Li ≥ 5 at.%)
**Primary Use:** Multi-task regression for solid-state electrolyte property prediction
**Format:** PyTorch serialized cache + individual graph `.pt` files
**Hash:** `e7ec8725a74047d2`

### Quick Statistics

| Property | Value |
|---|---|
| Total structures | 10,000 |
| Unique elements in dataset | 82 |
| Elements in search query | 49 |
| Unique formulas | 811 |
| Formula uniqueness ratio | 8.1% |
| Crystal systems represented | Triclinic, Orthorhombic, Hexagonal, Cubic, Tetragonal, Monoclinic |
| Mean atoms per structure | 33.15 (±24.43) |
| Min atoms | 3 |
| Max atoms | 177 |
| Chemical families | 7 |

### Target Coverage

| Target | Coverage | N Labeled | N Missing |
|---|---|---|---|
| `formation_energy` | 100% | 10,000 | 0 |
| `energy_above_hull` | 100% | 10,000 | 0 |
| `band_gap` | 100% | 10,000 | 0 |
| `log_ionic_conductivity` | **0%** | 0 | 10,000 |
| `activation_energy` | **0%** | 0 | 10,000 |

---

## 2. Data Sources

### Primary Source: Materials Project

The Materials Project (https://materialsproject.org) provides DFT-computed properties for all known inorganic crystalline materials. Data is accessed via the official MP REST API (`mp-api` Python package).

**API endpoint:** `mp_api.client.MPRester.materials.summary.search()`

**Fields collected:**

| Field | Type | Description |
|---|---|---|
| `material_id` | string | MP unique identifier (e.g., "mp-1234") |
| `formula_pretty` | string | Chemical formula |
| `structure` | object | `pymatgen.Structure` object |
| `formation_energy_per_atom` | float | Formation energy (eV/atom) |
| `energy_above_hull` | float | Energy above convex hull (eV/atom) |
| `band_gap` | float | Electronic band gap (eV) |
| `volume` | float | Unit cell volume (Å³) |
| `density` | float | Density (g/cm³) |
| `symmetry` | object | Space group information |
| `is_stable` | bool | Whether on convex hull |

**Query parameters:**
- `elements`: List of 49 elements (alkali, alkaline earth, transition metals, main group, halogens, chalcogens)
- `num_chunks`: ≤ 5 (for pagination, max 5000 documents per query)

### Secondary Sources (Code Support, Not in Current Dataset)

| Source | Collector Class | API Method | Fields |
|---|---|---|---|
| JARVIS-DFT | `JARVISCollector` | `jarvis.db.figshare.data("dft_3d")` | Formation energy, band gap, elastic constants |
| OQMD | `OQMDCollector` | `https://oqmd.org/oqmdapi/formationenergy` | Delta E, stability, band gap |
| AFLOW | `AFLOWCollector` | `https://aflow.org/API/aflux/` | E_gap, formation enthalpy |
| NOMAD | `NOMADCollector` | POST `https://nomad-lab.eu/prod/v1/api/v1/entries/query` | Entry metadata |

The current active dataset (`v3_li_10000`) uses **only the Materials Project** as its data source. Support code for the other sources exists in `src/data/collectors.py` for future data augmentation.

---

## 3. Collection Methodology

### Step-by-Step Collection Pipeline

```
1. Initialize MPRester with API key
2. Query: mpr.materials.summary.search(elements=element_list, fields=field_list)
3. Paginate: num_chunks = min(max(1, max_results // 1000), 5)
4. Collect documents → list[MaterialsDoc]
5. Convert to DataFrame: pd.DataFrame([d.dict() for d in docs])
```

### Element Filter

The dataset targets Li-containing materials. The query uses a 49-element superset:

```
Li, Na, K, Rb, Cs           # Alkali metals
Mg, Ca, Sr, Ba              # Alkaline earth
Sc, Y, La, Ti, Zr, Hf       # Early transition
V, Nb, Ta, Cr, Mo, W        # Middle transition
Mn, Fe, Co, Ni, Cu, Zn      # Late transition
B, Al, Ga, In               # Group 13
C, Si, Ge, Sn, Pb           # Group 14
N, P, As, Sb, Bi            # Pnictogens
O, S, Se, Te                # Chalcogens
F, Cl, Br, I                # Halogens
```

Li is not explicitly required in the query — the 5% post-filter ensures all returned materials contain Li.

### Sample Python Usage

```python
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(
        elements=["Li", "S", "P", "Cl"],
        fields=["material_id", "formula_pretty", "structure",
                "formation_energy_per_atom", "energy_above_hull", "band_gap"],
        num_chunks=5,
    )
```

---

## 4. Preprocessing

### Cleaning Pipeline

The `DataCleaner` class in `src/data/cleaner.py` applies these steps in order:

#### Step 1: Drop NaN in Required Columns
```python
required = ["formation_energy_per_atom", "structure"]
df = df.dropna(subset=[c for c in required if c in df.columns])
```

#### Step 2: Energy Range Filter
```python
df = df[df["formation_energy_per_atom"].between(-10, 5)]       # eV/atom
df = df[df["energy_above_hull"] >= 0]                           # Non-negative
```

**Rationale:**
- Formation energy beyond [-10, 5] eV/atom is physically unreasonable for condensed phases.
- Energy above hull is inherently non-negative (materials on or above the convex hull).

#### Step 3: Size Filter
```python
df = df[df["structure"].apply(lambda s: 2 <= len(s) <= 200)]
```

**Rationale:** Remove:
- Atomistic calculations (single atoms, dimers — < 2 atoms).
- Very large unit cells (> 200 atoms) that would exceed memory constraints.

#### Step 4: Deduplication (Disabled for v3_li_10000)
```python
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
```

The `StructureMatcher` from pymatgen identifies structurally identical materials. Deduplication is **disabled** in the current dataset version (`config.deduplicate = false`).

### Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `target` | 10,000 | Target number of structures |
| `min_atoms` | 2 | Minimum atoms per structure |
| `max_atoms` | 200 | Maximum atoms per structure |
| `ef_range` | [-10.0, 5.0] | Formation energy acceptance range |
| `deduplicate` | false | StructureMatcher dedup disabled |
| `normalize` | true | Z-score normalization applied |
| `cache_graphs` | false | Graph caching done separately |
| `seed` | 42 | Random seed for reproducibility |
| `max_neighbors` | 16 | Max neighbors in graph construction |
| `cutoff` | 8.0 | Neighbor search cutoff (Å) |
| `hidden_dim` | 128 | Model hidden dimension |

---

## 5. Splits

### Split Method: `composition_based_split()`

The splitter in `src/data/splitter.py` uses chemical-family-based grouping to prevent data leakage:

```python
def composition_based_split(dataset, val_ratio=0.1, test_ratio=0.1):
```

**Algorithm:**

1. Extract chemical formula from each structure.
2. Parse element groups: sort unique elements → join with "-" (e.g., "Li-O-P").
3. `GroupShuffleSplit(random_state=42)` splits by element group → train/test (90/10).
4. Second `GroupShuffleSplit(random_state=42)` splits train → train/val (92.2/7.8 adjusted).

### Split Sizes

| Split | Count | Ratio |
|---|---|---|
| Train | 8,310 | 83.1% |
| Validation | 586 | 5.9% |
| Test | 1,104 | 11.0% |
| **Total** | **10,000** | **100%** |

### Chemical Family Distribution

Based on `family_id()` from `src/chemistry/family_id.py`:

| Family | Composition Rule | Expected Count |
|---|---|---|
| `pure_halide` | Contains F/Cl/Br/I, no O or S | ~15% |
| `oxyhalide` | Contains halogen + O | ~5% |
| `sulfohalide` | Contains halogen + S | ~3% |
| `oxide` | Contains O (no P, no halogen) | ~40% |
| `sulfide` | Contains S (no halogen) | ~10% |
| `phosphate` | Contains O + P | ~20% |
| `other` | None of the above | ~7% |

### Split Rationale

`GroupShuffleSplit` by chemical family prevents the common problem of chemically similar materials appearing in both train and test splits. For example, LiFePO₄ variants are all phosphates and would be assigned to the same split, ensuring the model generalizes to unseen chemical families rather than memorizing formula patterns.

---

## 6. Target Properties

### Formation Energy (`formation_energy`)

- **Symbol**: E<sub>f</sub>
- **Unit**: eV/atom
- **Computation**: DFT total energy difference from elemental reference states
- **Range**: [-4.12, 4.94] eV/atom
- **Lower is better**: More negative = more stable
- **Physical meaning**: Energy released when forming a compound from its constituent elements. Negative values indicate thermodynamic stability.

### Energy Above Hull (`energy_above_hull`)

- **Symbol**: E<sub>ah</sub>
- **Unit**: eV/atom
- **Computation**: Difference between formation energy and the convex hull of competing phases
- **Range**: [0.00, 7.61] eV/atom
- **Lower is better**: E<sub>ah</sub> = 0 means on the hull (stable)
- **Physical meaning**: Decomposition energy — how much energy the material would gain by decomposing into the phases on the convex hull.
- **Thresholds**: < 0.025 eV/atom for synthesis typically achievable.

### Band Gap (`band_gap`)

- **Symbol**: E<sub>g</sub>
- **Unit**: eV
- **Computation**: DFT-PBE (semilocal) band gap
- **Range**: [0.00, 8.76] eV
- **Higher is better** (for SSEs): Wider band gap = better electronic insulation.
- **DFT limitation**: PBE systematically underestimates band gaps by ~30-40%.

### Log Ionic Conductivity (`log_ionic_conductivity`)

- **Coverage**: **0%** — no labeled data in current dataset
- **Unit**: log₁₀(S/cm)
- **Status**: Reserved for future experimental data integration

### Activation Energy (`activation_energy`)

- **Coverage**: **0%** — no labeled data in current dataset
- **Unit**: eV
- **Status**: Reserved for future experimental data integration

---

## 7. Distribution Statistics

### Formation Energy Distribution

| Statistic | Value |
|---|---|
| Mean | −1.962 eV/atom |
| Standard deviation | 0.917 eV/atom |
| Minimum | −4.123 eV/atom |
| 25th percentile | −2.531 eV/atom |
| Median (50th) | −2.151 eV/atom |
| 75th percentile | −1.498 eV/atom |
| Maximum | 4.943 eV/atom |
| Skew | 0.74 (right-tailed) |

### Energy Above Hull Distribution

| Statistic | Value |
|---|---|
| Mean | 0.142 eV/atom |
| Standard deviation | 0.422 eV/atom |
| Minimum | 0.000 eV/atom |
| 25th percentile | 0.000 eV/atom |
| Median (50th) | 0.006 eV/atom |
| 75th percentile | 0.086 eV/atom |
| Maximum | 7.608 eV/atom |
| Skew | 7.29 (heavily right-tailed) |

### Band Gap Distribution

| Statistic | Value |
|---|---|
| Mean | 1.256 eV |
| Standard deviation | 1.446 eV |
| Minimum | 0.000 eV |
| 25th percentile | 0.000 eV |
| Median (50th) | 0.638 eV |
| 75th percentile | 2.120 eV |
| Maximum | 8.758 eV |
| Percent zero | ~25% (metallic materials) |

### Structure Size Distribution

| Statistic | Value |
|---|---|
| Mean atoms | 33.15 |
| Standard deviation | 24.43 |
| Minimum | 3 |
| Maximum | 177 |
| Median | ~26 |

### Crystal System Distribution

| System | Count | Percentage |
|---|---|---|
| Triclinic | 816 | 81.6% |
| Orthorhombic | 85 | 8.5% |
| Hexagonal | 45 | 4.5% |
| Tetragonal | 25 | 2.5% |
| Monoclinic | 17 | 1.7% |
| Cubic | 12 | 1.2% |

---

## 8. Normalization

The `PropertyNormalizer` class applies per-task Z-score normalization:

```python
normalized_value = (raw_value - mean) / (std + 1e-8)
```

### Normalization Statistics (from `datasets/v3_li_10000/normalizer.json`)

| Target | Mean | Std | Min | Max |
|---|---|---|---|---|
| `formation_energy` | −1.962 | 0.917 | −4.123 | 4.943 |
| `energy_above_hull` | 0.142 | 0.422 | 0.000 | 7.608 |
| `band_gap` | 1.256 | 1.446 | 0.000 | 8.758 |

Normalization is essential for multi-task learning because:
1. Target scales differ by orders of magnitude (E<sub>ah</sub> ~0.1 vs BG ~1.0).
2. Z-score normalization ensures each task contributes equally to the initial loss.
3. GradNorm further adjusts task weights during training.

---

## 9. Known Biases

### Source Bias

- **Materials Project only**: All data comes from a single computational database. MP data is computed with consistent DFT settings (PBE+U), but this means systematic DFT errors propagate to all predictions.
- **No experimental data**: Properties are computed, not measured. The model learns DFT-predicted property relationships, not real-world physics.

### Chemical Bias

| Bias | Description | Impact |
|---|---|---|
| Li restriction | Only Li ≥ 5% materials | Model has zero knowledge of Li-poor or non-Li chemistries |
| Element imbalance | O, F, P, S overrepresented; noble metals underrepresented | Predictions for rare elements unreliable |
| Crystal system skew | Triclinic (81.6%) dominates | Performance on high-symmetry structures may differ |
| Stability skew | Most materials near hull (median E<sub>ah</sub> = 0.006) | Model trained on mostly stable materials |

### Data Quality Bias

- **DFT band gap error**: PBE band gaps are systematically underestimated by 30-40%.
- **Missing metastable phases**: The convex hull omits kinetically stabilized phases relevant for SSEs.
- **Structure relaxation**: MP structures are DFT-relaxed at 0 K — no temperature effects.

### Measurement Bias

- **No experimental validation**: All targets are computational. A model that achieves high R² on DFT-computed properties may not translate to experimental accuracy.
- **Conductivity gap**: The two most practically relevant properties (ionic conductivity, activation energy) have zero training data.

---

## 10. Licensing

### Materials Project Data

The Materials Project data is made available under the **Creative Commons Attribution 4.0 International License (CC-BY 4.0)**.

**Requirements:**
- Attribution must be given to the Materials Project.
- Users must cite the Materials Project when publishing results derived from this data.

**Citation:**
```
A. Jain, S.P. Ong, G. Hautier, et al.
"Commentary: The Materials Project: A materials genome approach to accelerating materials innovation."
APL Materials 1, 011002 (2013).
```

### Scandium Labs Code

The Scandium Labs codebase is licensed under the **MIT License**.

**Third-party dependencies:**
- PyTorch / torch-geometric: BSD-style licenses
- pymatgen: MIT License
- Materials Project API: MIT License
- scikit-learn: BSD License

---

## 11. Dataset Versions

| Version | Date | N | Description | Status |
|---|---|---|---|---|
| `v1_817` | 2024 | 817 | Initial Li-only | **Deprecated** |
| `v2_1000_smoketest` | 2024 | 1,000 | Pipeline validation | **Deprecated** |
| `v2_3635` | 2024 | 3,635 | Mid-scale v2 | **Deprecated** |
| `v2_10000` | 2024 | 10,000 | Large v2 | **Deprecated** |
| `v2_10000_log_eah` | 2025 | 10,000 | v2 + log EaH targets | **Deprecated** |
| **`v3_li_10000`** | 2026 | 10,000 | Current active | **Active** |

### Changes in v3_li_10000

- Updated element query (49 elements vs 20 in v2).
- Improved cleaning pipeline (energy filters, size filters).
- `composition_based_split()` with `GroupShuffleSplit`.
- Deduplication disabled (was causing sample count instability).
- Normalization statistics pre-computed and saved.

### History: Dataset Evolution

```
v1_817 (2024)
  └─ First working dataset: 817 Li-containing materials
  └─ Simple random split, manual cleaning
  └─ Basic ALIGNN model (phase1)
      │
      ▼
v2_1000_smoketest → v2_3635 → v2_10000 (2024-2025)
  └─ Scaled up to 10k
  └─ Improved splitter (group-based)
  └─ Introduced graph caching
  └─ Two-stage EaH head developed
      │
      ▼
v2_10000_log_eah (2025)
  └─ Experiment: log-transformed EaH targets
  └─ Abandoned — log transform hurt performance
      │
      ▼
v3_li_10000 (2026) ← ACTIVE
  └─ 49-element query, Li ≥ 5%
  └─ Family-balanced splits
  └─ PropertyNormalizer integrated
  └─ 7 chemical families
  └─ Pre-cached graphs (individual .pt files)
```

---

## 12. Maintenance and Updates

### Regenerating the Dataset

```bash
# Full pipeline: collect → clean → split → normalize → cache
python scripts/preprocess/build_dataset.py \
    --sources mp \
    --target 10000 \
    --output datasets/v3_li_10000 \
    --min-atoms 2 \
    --max-atoms 200 \
    --ef-range -10 5 \
    --seed 42

# Pre-cache graphs (optional, for faster training)
python scripts/preprocess/cache_graphs.py \
    --data-dir datasets/v3_li_10000 \
    --num-workers 1
```

### Dataset Integrity

Each dataset includes:
- `metadata.json` — version hash, config, target statistics.
- `dataset_report.json` — detailed per-target statistics.
- `normalizer.json` — Z-score normalization parameters.
- `split_indices.pt` — deterministic train/val/test split.

### Adding New Data

The collector code supports multiple data sources. To add experimental conductivity data:

1. Subclass `DataCollector` in `src/data/collectors.py`.
2. Add fields to the collector output.
3. Extend the cleaning pipeline to handle new targets.
4. Regenerate the dataset with `--extra-fields conductivity`.

---

## Appendix: Target Definition Details

### Formation Energy Definition

```
E_f = E_total - Σ(n_i * μ_i)
```

Where:
- E_total = DFT total energy of the compound
- n_i = number of atoms of element i
- μ_i = chemical potential of element i in its standard state

### Energy Above Hull Definition

```
E_ah = E_f(compound) - E_f(hull)
```

Where E_f(hull) is the minimum formation energy achievable by mixing phases at the same composition. Computed by:
1. Construct convex hull of all competing phases.
2. For a given composition, find the hull energy via Lever rule.
3. E_ah = difference between compound energy and hull energy.

### Band Gap Definition

The Kohn-Sham band gap from DFT-PBE:
```
E_g = E_{CBM} - E_{VBM}
```

PBE systematically underestimates band gaps. Hybrid functionals (HSE06) or GW would give more accurate values but are computationally prohibitive for 10,000 structures.
