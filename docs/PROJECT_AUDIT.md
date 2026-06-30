# Scandium Labs — Project Audit

> Generated: 2026-06-30 | Repository: scandium-labs v0.3.0

---

## 1. Repository Overview

| Metric | Value |
|---|---|
| Python files | 80 |
| Python lines of code | 9,470 |
| Non-Python files | 74 (configs, docs, Docker, shell, etc.) |
| Markdown files | 27 |
| Package structure | `src/` with 10 subpackages |
| Scripts | 12 across 6 subdirectories |
| Config files | 9 (YAML + JSON) |
| Test files | 9 |
| Test cases | 83 collected (63 pass, 13 fail, 7 skip) |
| API | FastAPI (5 files) |
| Frontend | Streamlit app (6 files) |
| Docker | 3 Dockerfiles + `docker-compose.yml` |

### Package structure

```
scandium-labs/
├── src/
│   ├── data/           # 5 files — dataset, collectors, cleaner, splitter
│   ├── models/
│   │   ├── gnn/        # 2 files — ALIGNN layers + GNN building blocks
│   │   ├── heads/      # 2 files — TwoStageEah head + PretrainedEncoder stub
│   │   └── scandium_model.py  # Main model (223 lines)
│   ├── training/       # 12 files — trainer, losses, engine, recom'd, coverage, etc.
│   ├── graphs/         # 3 files — builder + feature engineering
│   ├── chemistry/      # 2 files — family_id.py
│   ├── inference/      # 5 files — engine, ranking, stability, validation
│   ├── evaluation/     # 3 files — metrics, OOD detector
│   ├── explainability/ # 3 files — attention viz, integrated gradients
│   └── utils/          # 4 files — config, IO, logging
├── scripts/            # 12 files — train, evaluate, inference, benchmark, etc.
├── configs/            # 9 files — model variants, data, deploy, finetune, deepspeed
├── api/                # 5 files — FastAPI server, auth, DB, tasks, models
├── streamlit_app/      # 6 files — multi-page dashboard
├── tests/              # 9 files — 83 test cases
├── docs/               # 14 files
└── archive/            # 33+ stale files (historical experiments, scripts)
```

---

## 2. Directory-by-Directory Analysis

### `src/data/` — 5 files

| File | Lines | Status |
|---|---|---|
| `dataset.py` | 138 | **Clean.** `SolidElectrolyteDataset`, `LazyGraphDataset`, `_attach_targets`, `collate_fn`. One dead class (`MemoryOptimizedDataset`) was already removed per AGENTS.md. |
| `cleaner.py` | 115 | **Clean.** `DataCleaner` + `PropertyNormalizer`. Round-trip safe (fit → transform → inverse_transform). |
| `collectors.py` | 152 | **Clean.** `MaterialsProjectCollector`, `JARVISCollector`, `OQMDCollector`, `AFLOWCollector`, `NOMADCollector`. The non-MP collectors have bare `except` handlers (noted in AGENTS.md as to-fix). |
| `splitter.py` | 28 | **Clean.** `composition_based_split` using `GroupShuffleSplit`. |
| `__init__.py` | 15 | **Clean.** Exports all 6 public symbols. |

**Verdict:** well-organized. The `LazyGraphDataset` extraction noted in AGENTS.md has already been done — it lives inside `dataset.py`.

### `src/models/` — 5 files across 3 directories

| File | Lines | Status |
|---|---|---|
| `scandium_model.py` | 223 | **Main model.** `ScandiumPINNGNN` — ALIGNN encoder + GraphTransformer + PINN module + task heads + uncertainty heads + MC Dropout. Clean, well-structured. |
| `gnn/alignn.py` | 85 | **Clean.** `ALIGNNLayer` + `ALIGNN` (standalone model). |
| `gnn/layers.py` | 135 | **Clean.** `CrystalMPNN`, `GraphTransformerLayer`, `AttentionGlobalPool`, `PINNConstraintModule`, `EquivariantConv`. |
| `heads/two_stage_eah.py` | 168 | **Clean.** `TwoStageEahHead`, `TwoStageEahLoss`, `two_stage_metrics`. Two-stage: binary stability classifier + magnitude regressor. |
| `heads/pretrained.py` | 20 | **Stub.** Documented as "not actively used — kept for import compatibility." No-op `load_encoder`. |
| `__init__.py` | 25 | **Clean.** Exports all 10 public symbols from both subpackages + main model. |
| `gnn/__init__.py` | 18 | **Clean.** |
| `heads/__init__.py` | 13 | **Clean.** |

**Verdict:** recently reorganized into subpackages. Clean across the board. The `PretrainedEncoder` stub could be removed or revived.

### `src/training/` — 12 files

| File | Lines | Status |
|---|---|---|
| `trainer.py` | 264 | **Clean. Core training loop.** `ScandiumTrainer` — build model, optimizer, loss, train_epoch, validate, save_checkpoint, resume. |
| `loaders.py` | 89 | **Clean.** `load_data()` — handles prebuilt graphs or on-the-fly graph building. |
| `losses.py` | 90 | **Clean.** `PINNLoss` — data loss + Arrhenius physics loss + thermodynamic constraint. 5 dead classes already removed (GradNormLoss, FamilyContrastiveLoss, GradientReversal, FamilyAdversary, MultiTaskLoss). |
| `engine.py` | 233 | **Clean.** `evaluate_model`, `compute_test_metrics`, `predict_dataset`. Substantial overlap with `src/inference/engine.py`'s `predict_single` (see Duplicate Detection §4). |
| `scheduler.py` | 25 | **Clean.** `build_scheduler` (CosineAnnealingWarmRestarts) + `get_cosine_schedule_with_warmup`. |
| `distributed.py` | 90 | **Clean.** `train_distributed` (DDP) + `train_with_deepspeed`. |
| `pretrained.py` | 27 | **Clean.** `get_param_groups` — differential LR for pretrained vs new params. |
| `activation.py` | 16 | **Clean.** `compute_activation_energies` — Arrhenius-derived Ea from sigma(T). |
| `recommend.py` | 145 | **Clean.** `recommend_materials`, `recommend_by_formula`, `stability_bands`. Recommender logic (REJECT / UNCERTAIN / HIGH PRIORITY / etc.). Duplicated from `inference/engine.py`'s `_make_recommendation` and `_stability_bands`. |
| `coverage.py` | 30 | **Clean.** `generate_coverage_report`, `format_coverage_metrics`. |
| `data_audit.py` | 64 | **Active.** `audit_label_coverage`, `gate_predictions`, `fit_activation_energy`. Imported by `engine.py`, `coverage.py`, and `inference/engine.py`. |
| `curriculum.py` | 17 | **⚠️ ORPHANED.** `CurriculumDataLoader` — not imported anywhere in the codebase. Zero references. |
| `__init__.py` | 37 | **Clean.** Exports all 17 public symbols. Does NOT export `curriculum` or `data_audit` (correctly). |

**Verdict:** recently split from 2→12 files. Well-organized. One orphaned file (`curriculum.py`) should be removed.

### `src/graphs/` — 3 files

| File | Lines | Status |
|---|---|---|
| `builder.py` | 133 | **Clean.** `CrystalGraphBuilder` + `ALIGNNGraphBuilder` + `FeatureEngineer`. |
| `features.py` | 177 | **Clean.** Atom features (14-dim), Bessel/Gaussian/SphericalBessel RBFs, bond angles, global features (16-dim). Minor warning for mendeleev_no on Ts/Og (benign). |
| `__init__.py` | 0 | **Empty.** Should export `ALIGNNGraphBuilder`, `FeatureEngineer`, etc. |

**Verdict:** focused and clean. Empty `__init__.py` should be populated.

### `src/chemistry/` — 2 files

| File | Lines | Status |
|---|---|---|
| `family_id.py` | — | **Clean.** Chemical family classification. |
| `__init__.py` | 0 | **Empty.** Should export from `family_id.py`. |

**Verdict:** clean, minimal.

### `src/inference/` — 5 files

| File | Lines | Status |
|---|---|---|
| `engine.py` | 348 | **Clean.** `InferenceEngine` — model loading, single/batch prediction, stability resolution, recommendations, activation energy inference. |
| `stability.py` | 84 | **Clean.** `compute_hull_energy`, `hull_consistency_flag`, `resolve_stability`. |
| `ranking.py` | 79 | **Clean.** `ParetoRanker` — multi-objective Pareto ranking with composite scores. |
| `validation.py` | 54 | **Clean.** `validate_structure` — sanity checks on input structures. |
| `__init__.py` | 0 | **Empty.** Should export `InferenceEngine`, `ParetoRanker`, `resolve_stability`, etc. |

**Verdict:** clean and focused. Substantial code overlap with `src/training/engine.py` (both have near-identical `_load_model`, `predict_dataset` with gating, denormalization, and recommendation logic).

### `src/evaluation/` — 3 files

| File | Lines | Status |
|---|---|---|
| `metrics.py` | 46 | **Clean.** `compute_metrics`, `expected_calibration_error`. |
| `ood.py` | 23 | **Clean.** `OODDetector` — IsolationForest-based out-of-distribution detection. |
| `__init__.py` | 0 | **Empty.** Should export. |

**Verdict:** clean.

### `src/explainability/` — 3 files

| File | Lines | Status |
|---|---|---|
| `attention.py` | 61 | **Clean.** `AttentionVisualizer` — hooks into attention layers, plots crystal graph with attention weights. |
| `gradients.py` | 23 | **Clean.** `integrated_gradients` — IG attribution for atom features. |
| `__init__.py` | 0 | **Empty.** Should export. |

**Verdict:** functional but sparse. Covers basic interpretability needs. Could be expanded with Grad-CAM or other methods.

### `src/utils/` — 4 files

| File | Lines | Status |
|---|---|---|
| `config.py` | 22 | **Clean.** `load_config`, `merge_configs`. |
| `io.py` | 46 | **Clean.** `ensure_dir`, `safe_save` (atomic replace via tempfile), `load_json`, `save_json`. |
| `logging.py` | 22 | **Clean.** `setup_logging`, `get_logger`. |
| `__init__.py` | 16 | **Clean.** Exports all 8 public symbols. |

**Verdict:** recently created. Good coverage of shared utilities. No duplicate utility functions found across `src/`, `scripts/`, or `api/`.

### `scripts/` — 12 files across 6 subdirectories

| Category | Files |
|---|---|
| `train/` | `train.py`, `train_v3_li.py`, `experiment_sweep.py` |
| `evaluate/` | `cross_validate.py` |
| `inference/` | `screen_candidates.py` |
| `preprocess/` | `build_dataset.py` |
| `benchmark/` | `benchmark_suite.py`, `compare_benchmarks.py`, `run_benchmark.py`, `_utils.py` |
| `maintenance/` | `rebuild_li_dataset.py` |

**Verdict:** recently cleaned up. The scripts noted in AGENTS.md (`train_gpu.py`, `run_training.py`) have already been removed.

### `configs/` — 9 files

| Config | Type | Active? | Notes |
|---|---|---|---|
| `model_config.yaml` | v1 | **Stale** | No code references. Hidden_dim=256, batch_size=64. |
| `model_config_v2.yaml` | v2 | **Stale** | No code references. Checkpoints exist in `experiments/` dirs. |
| `model_config_v3.yaml` | v3 | **Stale** | Only referenced in archive experiment reports. |
| `model_config_v3_li.yaml` | v3-Li | **Active** | Referenced by `reproduce.sh` (default config). Li-focused with two_stage_eah. |
| `phase3_config_log_eah.yaml` | Phase 3 | **Stale** | Referenced only in docs/experiments.md. Log-transformed EaH experiment. |
| `finetune_config.yaml` | Fine-tune | **Stale** | Referenced only in docs. |
| `data_config.yaml` | Data | **Likely active** | Data collection/build parameters. Not directly referenced from Python code. |
| `deploy_config.yaml` | Deploy | **Stale** | Referenced only in docs. Deployment uses env vars instead. |
| `ds_config.json` | DeepSpeed | **Active** | Referenced by `src/training/distributed.py:71`. |

**Verdict:** 3 of 9 configs are actively referenced. The remaining 6 should be archived or documented as historical.

### `api/` — 5 files

| File | Status |
|---|---|
| `main.py` (194 lines) | **Functional.** FastAPI with `/screen`, `/screen/upload`, `/job/{job_id}`, `/health`. |
| `auth.py` | JWT token verification. |
| `database.py` | SQLAlchemy models + session management. |
| `tasks.py` | Celery task definitions. |
| `models.py` | Pydantic schemas. |

**Verdict:** functional but several bare `except` handlers. Requires MP_API_KEY in `.env` and running Celery/Redis stack.

### `streamlit_app/` — 6 files

| File | Status |
|---|---|
| `streamlit_app.py` (448 lines) | Main app — navigation, CIF upload, model selection. |
| `pages/dashboard.py` (144 lines) | **Clean.** Dashboard page. |
| `pages/batch.py` (223 lines) | **Clean.** Batch screening. |
| `pages/results.py` (203 lines) | **Clean.** Results page. |
| `pages/screen.py` (759 lines) | **Clean.** Single-crystal screen with structure viewer. |

**Verdict:** well-organized multi-page Streamlit app.

### `tests/` — 9 files

| Test file | Status |
|---|---|
| `conftest.py` | Fixtures for model + graph data. |
| `test_api.py` | 3 tests (2 async — require pytest-asyncio). |
| `test_data.py` | 3 tests — DataCleaner + PropertyNormalizer. |
| `test_data_audit.py` | 8 tests — coverage, gating, activation energy fitting. |
| `test_inference.py` | 3 tests — ParetoRanker. |
| `test_models.py` | 3 tests — model creation + forward pass. |
| `test_pipeline.py` | 18 tests — normalizer round-trip, log transform, checkpoint self-containment, loss functions. |
| `test_reference_materials.py` | 24 tests — reference materials (Li6PS5Cl), hull consistency, recommendations. |
| `test_training_normalization.py` | 19 tests — normalizer, delta method, subset comparison, inference. |

**Results:** 83 tests collected, **63 pass, 13 fail, 7 skipped**.

**Pre-existing failures:**
- `test_reference_materials.py::TestFineTuner::test_recommendation_v3_high_priority` — recommendation logic mismatch
- `test_reference_materials.py::TestFineTuner::test_recommendation_v3_none_uncertainty` — recommendation logic mismatch  
- `test_training_normalization.py::TestCommonSubsetComparison::test_common_subset_intersection` — missing data files
- `test_training_normalization.py::TestCommonSubsetComparison::test_common_subset_excluded_reported` — missing data files
- Remaining 9 failures also from data-dependent tests in `test_pipeline.py` (missing experiment directories)

### `docs/` — 14 files

Comprehensive set: README, ARCHITECTURE, DATASETS, DEVELOPMENT, TROUBLESHOOTING, INSTALLATION, API, INFERENCE, TRAINING, BENCHMARKS, FAQ, EXPERIMENTS, RESEARCH_PLAN, PROJECT_AUDIT.

---

## 3. Dead Code Assessment

### Orphaned files

| File | Reason | Action |
|---|---|---|
| `src/training/curriculum.py` | **Zero imports** across the entire codebase. Not in `__init__.py`. `CurriculumDataLoader` is unused. | **Remove.** |
| `src/models/heads/pretrained.py` | Documented as "not actively used — kept for import compatibility." No-op stub. | **Consider removing** or implementing real pretrained loading. |
| `archive/` (33+ files) | Historical experiments, scripts, and old modules. Most predate the codebase reorganization. | **Clean up** — archive to a `.tar.gz` or delete after verification. |

### Dead classes already removed

Per AGENTS.md, the following were removed from `losses.py`:
- `GradNormLoss`
- `FamilyContrastiveLoss`
- `GradientReversal`
- `FamilyAdversary`
- `MultiTaskLoss`

Confirmed: zero references to these classes remain in the codebase.

### Files referenced but non-critical

| File | Status |
|---|---|
| `src/training/data_audit.py` | **Active** — imported by `src/inference/engine.py:10`, `src/training/engine.py:14`, `src/training/coverage.py:5`, `tests/test_data_audit.py`. |

---

## 4. Duplicate Detection

### Utility functions — no duplicates found

| Function | Location | Duplicates? |
|---|---|---|
| `load_config` | `src/utils/config.py:9` | None |
| `ensure_dir` | `src/utils/io.py:10` | None |
| `setup_logging` | `src/utils/logging.py:7` | None |
| `save_json` | `src/utils/io.py:42` | None |
| `load_json` | `src/utils/io.py:37` | None |

### Cross-module duplication — significant overlap

The largest redundancy exists between `src/inference/engine.py` and `src/training/engine.py`:

| Logic | `inference/engine.py` | `training/engine.py` | Lines duplicated |
|---|---|---|---|
| `_load_model` | Lines 67-93 | Lines 22-58 | ~35 lines, near-identical |
| `predict_single` / `predict_dataset` | Lines 96-206 | Lines 112-233 | ~110 lines, structurally identical |
| `_make_recommendation` / `recommend_materials` | Lines 216-324 | `src/training/recommend.py:24-135` | ~110 lines, nearly identical |
| `_stability_bands` / `stability_bands` | Lines 327-340 | `src/training/recommend.py:8-21` | ~13 lines, identical |

**Total estimated duplication:** ~270 lines across ~3 files.

### Recommendation code duplication

`inference/engine.py:_make_recommendation` and `training/recommend.py:recommend_materials` are near-identical forks. Both implement the same REJECT/UNCERTAIN/HIGH/MEDIUM/LOW PRIORITY logic with the same thresholds and action suggestions.

### Stability band duplication

`inference/engine.py:_stability_bands` and `training/recommend.py:stability_bands` are identical in logic (only icon strings differ slightly: `green` vs `🟢`).

---

## 5. Configuration Files

### Active configs

| Config | Referenced by | Purpose |
|---|---|---|
| `configs/model_config_v3_li.yaml` | `reproduce.sh:11` (default) | Li-2000 fine-tuning config with two-stage EAH |
| `configs/ds_config.json` | `src/training/distributed.py:71` | DeepSpeed ZeRO-2 config |

### Stale configs (no code references)

| Config | Purpose | Recommendation |
|---|---|---|
| `configs/model_config.yaml` | v1 model (256-dim, 4+4 layers) | **Archive** — superseded by v3 |
| `configs/model_config_v2.yaml` | v2 model (128-dim, 2+1 layers) | **Archive** — superseded by v3 |
| `configs/model_config_v3.yaml` | v3 model (200-dim, 3+2 layers) | **Archive** — Li variant is the active one |
| `configs/phase3_config_log_eah.yaml` | Phase 3 log-EAH experiment | **Archive** — historical |
| `configs/finetune_config.yaml` | Generic fine-tune config | **Archive** or keep as template |
| `configs/deploy_config.yaml` | Deployment parameters | **Keep** but note it's only documented |

### Ambiguity

`model_config.yaml` v2 → v3 progression is not documented anywhere. Newcomers cannot tell which config is current without reading `reproduce.sh` and `archive/experiments/reports/`.

---

## 6. Recommendations

### Priority: High

1. **Remove orphaned `src/training/curriculum.py`** — Zero imports, no references.
2. **Deduplicate inference engine** — Consolidate `src/inference/engine.py` and `src/training/engine.py`. Both have near-identical `_load_model` and prediction logic. Either have `training/engine.py` delegate to `inference/engine.py`, or extract a shared base.
3. **Deduplicate recommendation logic** — `inference/engine.py:_make_recommendation` and `training/recommend.py:recommend_materials` are identical forks. Pick one and have the other delegate.
4. **Fix 13 pre-existing test failures** — Especially `test_reference_materials.py` (recommendation logic drift) and `test_training_normalization.py` (missing data subset comparisons).

### Priority: Medium

5. **Populate empty `__init__.py` files** — `src/graphs/`, `src/chemistry/`, `src/inference/`, `src/evaluation/`, `src/explainability/` all have empty `__init__.py`. Should export public symbols.
6. **Clean up stale configs** — Archive `model_config.yaml`, `model_config_v2.yaml`, `model_config_v3.yaml`, and `phase3_config_log_eah.yaml` with a note about which checkpoints they were used for. Leave only actively referenced configs.
7. **Replace bare `except` handlers** — Found in `api/main.py` (lines 32-33, 68, 93, 180) and `src/data/collectors.py` (lines 103, 142). Should log the exception at minimum.

### Priority: Low

8. **Implement or remove `PretrainedEncoder`** — Currently a no-op stub documented as "not actively used."
9. **Add entry points to `pyproject.toml`** — No `[project.scripts]` section. Consider adding `scandium-train`, `scandium-screen`, `scandium-serve`.
10. **Remove `archive/` from repository** — 33+ stale files. If needed for provenance, move them outside the repo or into a compressed archive.
11. **Add scripts section to `pyproject.toml`** — So users can run `scandium-train`, `scandium-screen`, etc. without knowing the module paths.
12. **Fix `test_api.py` pytest-asyncio marker** — Tests have `@pytest.mark.asyncio` but no `pytest-asyncio` in `pyproject.toml` optional dev deps. Causes `PytestUnknownMarkWarning`.

---

## Summary

Scandium Labs is a well-structured GNN-based solid-state electrolyte screening platform. The codebase has undergone significant refactoring (training 2→12 files, extraction of `src/utils/`, removal of dead losses). Key remaining issues are ~270 lines of duplicated inference logic across two engine modules, one orphaned file (`curriculum.py`), 13 pre-existing test failures, and 6 stale config files. The recommendations above should take approximately 2–3 days to fully address.
