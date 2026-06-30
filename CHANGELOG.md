# Changelog

All notable changes to Scandium Labs will be documented in this file.

## [0.3.0] — 2026-06-30

### Changed
- Major architectural refactoring: `src/training/` split into 8 focused modules (trainer, loaders, distributed, scheduler, pretrained, engine, recommend, coverage, activation)
- `src/models/` organized into subpackages: `gnn/` and `heads/`
- All imports standardized to absolute package imports
- Unused imports removed across 10 files

### Removed
- 160 lines of dead code from `losses.py` (5 unused loss classes)
- Stale scripts: `run_training.py`, `train_gpu.py`, `prepare_data.py`
- Parent workspace cleanup: stale venvs (`Labs/`, `SSB/`) and duplicated CIFs removed
- Obsolete `setup.py` (migrated fully to `pyproject.toml`)

### Fixed
- `PretrainedEncoder.load_encoder()` — was called by `ScandiumTrainer` but didn't exist
- Syntax errors in 3 Streamlit pages (`dashboard.py`, `batch.py`, `results.py`)
- `predict_with_mc_dropout` `KeyError` for `p_unstable`
- Build backend in `pyproject.toml` (`setuptools.backends._legacy` → `build_meta:__legacy__`)

### Added
- `src/utils/` populated with `logging.py`, `config.py`, `io.py`
- `__init__.py` exports for all 5 `src/` packages (28 public symbols)
- `.editorconfig`, `.gitattributes`, `Makefile`, `reproduce.sh`
- `AGENTS.md` — refactoring plan for AI agents
- Documentation: `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_STRUCTURE.md`

## [0.2.0] — 2026-06-15

### Added
- Li-constrained dataset rebuild: 20,789 Li≥5% structures, subsampled to 10k
- `LazyGraphDataset` — on-the-fly graph building with disk caching
- `scripts/train/train_v3_li.py` — training script for Li dataset
- Config files for v3 training (`configs/model_config_v3_li.yaml`)
- Docker deployment support (`docker-compose.yml`)

### Changed
- Ruff/Black/isort applied to all `src/` and `scripts/` files
- `print()` → `logging` in `trainer.py` and `pretrained.py`
- `.gitignore` updated for venvs, node_modules, logs
- Security: JWT secret, DB URL, Redis URL, model path → environment variables

## [0.1.0] — 2026-05-01

### Added
- Initial release
- ALIGNN-based GNN for solid electrolyte property prediction
- Multi-task PINN framework with physics-informed losses
- MC Dropout for uncertainty quantification
- API server (FastAPI) and Streamlit dashboard
- Materials Project data collection pipeline
- `ScandiumTrainer` with AMP, gradient clipping, early stopping, wandb logging
- 5 chemical families classification (halide/oxide/sulfide/phosphate/other)
- Composition-based dataset splitting
- GPU training with DeepSpeed support
