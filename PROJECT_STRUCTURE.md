# Project Structure

```
scandium-labs/
├── README.md                      # Project overview, installation, quick start
├── PROJECT_STRUCTURE.md           # This file — comprehensive directory reference
├── ARCHITECTURE.md                # Model architecture, training flow, deployment
├── DATASETS.md                    # Dataset versions, coverage, curation methodology
├── CHANGELOG.md                   # Version history (v0.1.0 → v0.3.0)
├── ROADMAP.md                     # Short/medium/long-term development goals
├── AGENTS.md                      # Instructions for AI coding agents (refactoring plan)
├── Makefile                       # install, train, evaluate, test, lint, format, typecheck, clean
├── CITATION.cff                   # Machine-readable citation metadata
├── pyproject.toml                 # Python package config: setuptools, ruff, pytest
├── .gitignore
├── .env                           # API keys (MP_API_KEY, JWT_SECRET_KEY) — not committed
│
├── src/                           # Core Python package `scandium-labs` (under src/)
│   ├── __init__.py                # Re-exports data, models, training, utils
│   │
│   ├── data/                      # Data acquisition, cleaning, loading, splitting
│   │   ├── __init__.py            # Exports: PropertyNormalizer, MaterialsProjectCollector,
│   │   │                          #   LazyGraphDataset, SolidElectrolyteDataset, collate_fn,
│   │   │                          #   composition_based_split
│   │   ├── collectors.py          # MPRester-based MaterialsProjectCollector (primary),
│   │   │                          #   JARVISCollector, OQMDCollector, AFLOWCollector,
│   │   │                          #   NOMADCollector for multi-source data acquisition
│   │   ├── cleaner.py             # DataCleaner (structure dedup via StructureMatcher,
│   │   │                          #   range filtering, unit normalization),
│   │   │                          #   PropertyNormalizer (z-score fit/transform/inverse,
│   │   │                          #   save/load JSON)
│   │   ├── dataset.py             # SolidElectrolyteDataset (on-the-fly graph building),
│   │   │                          #   LazyGraphDataset (memory-efficient disk-backed loading),
│   │   │                          #   collate_fn (PyG batching helper)
│   │   └── splitter.py            # composition_based_split — GroupShuffleSplit by element
│   │                              #   composition to prevent element-wise leakage
│   │
│   ├── graphs/                    # Crystal graph construction & feature engineering
│   │   ├── __init__.py            # (empty — imports via full paths)
│   │   ├── builder.py             # CrystalGraphBuilder (periodic neighbor search via
│   │   │                          #   pymatgen's find_points_in_spheres, edge limiting),
│   │   │                          #   ALIGNNGraphBuilder (adds line graph with bond-angle
│   │   │                          #   edge features via SphericalBesselRBF),
│   │   │                          #   FeatureEngineer (pad/truncate atom features to 92-dim)
│   │   └── features.py            # get_atom_features (16-dim via mendeleev table),
│   │                              #   get_global_features (16-dim: density, volume, lattice,
│   │                              #   sg number, electronegativity),
│   │                              #   BesselRBF, GaussianRBF, SphericalBesselRBF,
│   │                              #   compute_bond_angles, compute_soap (optional dscribe)
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── __init__.py            # Exports: ALIGNN, ALIGNNLayer, AttentionGlobalPool,
│   │   │                          #   CrystalMPNN, GraphTransformerLayer, PINNConstraintModule,
│   │   │                          #   PretrainedEncoder, ScandiumPINNGNN, TwoStageEahHead,
│   │   │                          #   TwoStageEahLoss
│   │   ├── scandium_model.py      # ScandiumPINNGNN — main multi-task GNN:
│   │   │                          #   ALIGNN backbone → GraphTransformer × M →
│   │   │                          #   PINNConstraintModule → AttentionGlobalPool →
│   │   │                          #   task-specific heads (5 tasks). Includes MC Dropout
│   │   │                          #   uncertainty estimation, gradient checkpointing,
│   │   │                          #   TwoStageEahHead support
│   │   ├── gnn/                   # GNN layer subpackage
│   │   │   ├── __init__.py        # Exports: ALIGNN, ALIGNNLayer, AttentionGlobalPool,
│   │   │   │                      #   CrystalMPNN, EquivariantConv, GraphTransformerLayer,
│   │   │   │                      #   PINNConstraintModule
│   │   │   ├── alignn.py          # ALIGNN (standalone — embedding + ALIGNNLayers + pool +
│   │   │   │                      #   heads), ALIGNNLayer (one dual message-passing step:
│   │   │   │                      #   line-graph conv → crystal-graph conv)
│   │   │   └── layers.py          # CrystalMPNN (PyG MessagePassing, SiLU-gated),
│   │   │                          #   EquivariantConv (e3nn SO(3) convolution, optional),
│   │   │                          #   GraphTransformerLayer (MultiheadAttention + FFN + LN),
│   │   │                          #   PINNConstraintModule (Arrhenius + thermodynamic gating),
│   │   │                          #   AttentionGlobalPool (learned softmax-weighted pooling)
│   │   └── heads/                 # Task head subpackage
│   │       ├── __init__.py        # Exports: PretrainedEncoder, TwoStageEahHead,
│   │       │                      #   TwoStageEahLoss, two_stage_metrics
│   │       ├── two_stage_eah.py   # TwoStageEahHead — decomposes EaH into binary stability
│   │       │                      #   classifier + magnitude regressor (Softplus-constrained);
│   │       │                      #   TwoStageEahLoss — weighted BCE + MSE(unstable) +
│   │       │                      #   MSE(stable→0) with per-sample family weights;
│   │       │                      #   two_stage_metrics — precision, recall, F1, MAE
│   │       └── pretrained.py      # PretrainedEncoder — stub for future ALIGNN pretrained-
│   │                              #   checkpoint loading (currently pass-through)
│   │
│   ├── training/                  # Training loop, losses, evaluation, scheduling
│   │   ├── __init__.py            # Exports: ScandiumTrainer, PINNLoss, compute_activation_
│   │   │                          #   energies, load_data, build_scheduler, evaluate_model,
│   │   │                          #   predict_dataset, recommend_materials, stability_bands,
│   │   │                          #   train_distributed, train_with_deepspeed, etc.
│   │   ├── trainer.py             # ScandiumTrainer — full training orchestrator:
│   │   │                          #   builds model/optimizer/loss from YAML config, trains
│   │   │                          #   with AMP + GradScaler, validates, early-stops, saves
│   │   │                          #   checkpoints, W&B logging, resume-from-checkpoint
│   │   ├── losses.py              # PINNLoss — data loss + Arrhenius physics (logσ–Ea
│   │   │                          #   consistency) + thermodynamic (EaH ≥ 0) regularization;
│   │   │                          #   compute_diffusion_residual — PDE residual via
│   │   │                          #   autograd (Fick's 2nd law — optional)
│   │   ├── engine.py              # _load_model (checkpoint → model + normalizer),
│   │   │                          #   evaluate_model, compute_test_metrics (per-task MAE),
│   │   │                          #   predict_dataset (full batch inference with MC dropout,
│   │   │                          #   coverage gating, stability checks, conductivity calc)
│   │   ├── loaders.py             # load_data — loads prebuilt .pt graphs or builds on-the-fly
│   │   │                          #   from dataset_cache.pt + split_indices.pt
│   │   ├── scheduler.py           # build_scheduler (CosineAnnealingWarmRestarts),
│   │   │                          #   get_cosine_schedule_with_warmup
│   │   ├── distributed.py         # train_distributed (DDP via torch.distributed),
│   │   │                          #   train_with_deepspeed (ZeRO-2 via DeepSpeed)
│   │   ├── pretrained.py          # get_param_groups — differential LR for pretrained
│   │   │                          #   (ALIGNN layers: 0.1×) vs new params
│   │   ├── curriculum.py          # CurriculumDataLoader — progressive complexity increase
│   │   │                          #   (by # elements × # sites)
│   │   ├── data_audit.py          # audit_label_coverage, gate_predictions, fit_activation_
│   │   │                          #   energy (Arrhenius Ea from sigma(T)),
│   │   │                          #   STATUS_NO_LABELS / STATUS_MC_DISABLED constants
│   │   ├── coverage.py            # generate_coverage_report, format_coverage_metrics
│   │   │                          #   (label availability per task)
│   │   ├── recommend.py           # recommend_materials — rule-based: HIGH/MEDIUM/LOW/REJECT/
│   │   │                          #   UNCERTAIN based on σ, EaH, OOD, uncertainty;
│   │   │                          #   stability_bands — EaH threshold labels/colors;
│   │   │                          #   recommend_by_formula
│   │   └── activation.py          # compute_activation_energies — Arrhenius Ea from
│   │                              #   single-temperature sigma (kB = 8.617e-5, A = 1e6)
│   │
│   ├── inference/                 # Prediction pipeline & candidate ranking
│   │   ├── __init__.py            # (empty)
│   │   ├── engine.py              # InferenceEngine — loads checkpoint + normalizer +
│   │   │                          #   graph builder; predict_single (full pipeline: build
│   │   │                          #   graph → MC dropout → coverage gating → denormalize →
│   │   │                          #   stability check → recommendation), predict_batch;
│   │   │                          #   also: _make_recommendation, _stability_bands,
│   │   │                          #   _infer_activation_energy
│   │   ├── ranking.py             # ParetoRanker — multi-objective Pareto front ranking
│   │   │                          #   (conductivity ↑, −EaH ↑, confidence ↑) with
│   │   │                          #   weighted composite score
│   │   ├── stability.py           # compute_hull_energy (MP convex hull lookup via API),
│   │   │                          #   hull_consistency_flag (detects Ef≈0 + EaH>0 conflict),
│   │   │                          #   resolve_stability (combined check + hull query)
│   │   └── validation.py          # validate_structure — CIF quality checks: n_atoms,
│   │                              #   volume, min distance, charge, density, formula
│   │
│   ├── evaluation/                # Metrics & out-of-distribution detection
│   │   ├── __init__.py            # (empty)
│   │   ├── metrics.py             # compute_metrics (MAE, RMSE, R², MAPE; task-specific:
│   │   │                          #   Within_1_OOM for conductivity, Stability_Accuracy
│   │   │                          #   for EaH); expected_calibration_error
│   │   └── ood.py                 # OODDetector — IsolationForest on graph embeddings
│   │                              #   with StandardScaler preprocessing
│   │
│   ├── explainability/            # Model interpretability tools
│   │   ├── __init__.py            # (empty)
│   │   ├── attention.py           # AttentionVisualizer — hooks into transformer layers,
│   │   │                          #   produces NetworkX spring-layout attention graph
│   │   └── gradients.py           # integrated_gradients — baseline attribution by
│   │                              #   Riemann approximation over interpolation path
│   │
│   ├── chemistry/                 # Chemistry utilities
│   │   ├── __init__.py            # (empty)
│   │   └── family_id.py           # family_id — classifies formula into 7 families:
│   │                              #   pure_halide, oxyhalide, sulfohalide, oxide, sulfide,
│   │                              #   phosphate, other; family_numeric, has_lithium
│   │
│   └── utils/                     # Shared helpers
│       ├── __init__.py            # Exports: setup_logging, get_logger, load_config,
│       │                          #   merge_configs, ensure_dir, safe_save, load_json,
│       │                          #   save_json
│       ├── config.py              # load_config (YAML), merge_configs (deep recursive merge)
│       ├── io.py                  # ensure_dir, safe_save (atomic write via tempfile+replace),
│       │                          #   load_json, save_json
│       └── logging.py             # setup_logging (stdout handler, custom format),
│                                  #   get_logger
│
├── scripts/                       # CLI tools, organized by purpose
│   ├── train/
│   │   ├── train.py               # Config-based harness for ScandiumTrainer; supports
│   │   │                          #   single-GPU and multi-GPU (torch.distributed.spawn)
│   │   ├── train_v3_li.py         # Standalone end-to-end training for v3_li_10k:
│   │   │                          #   LazyGraphDataset + manual loop with GradNorm +
│   │   │                          #   TwoStageEahLoss + AMP + early stopping + test eval
│   │   └── experiment_sweep.py    # Structured experiment runner: auto-creates versioned
│   │                              #   experiment directories with config, metrics, logs,
│   │                              #   parity plots, git commit hash
│   ├── evaluate/
│   │   └── cross_validate.py      # 5-fold cross-validation with chemistry-stratified
│   │                              #   folds, per-task MAE/RMSE/R², denormalized metrics,
│   │                              #   early stopping, saves per-fold + summary JSON
│   ├── inference/
│   │   └── screen_candidates.py   # CLI screening: reads candidate list JSON, runs
│   │                              #   InferenceEngine, outputs ranked results JSON
│   ├── preprocess/
│   │   └── build_dataset.py       # Unified dataset builder: 10-step pipeline —
│   │                              #   download (MP/OQMD/JARVIS/AFLOW/NOMAD) →
│   │                              #   extract → clean → deduplicate → report →
│   │                              #   split → normalize → cache dataset → cache graphs
│   │                              #   → metadata. 718 lines, 80+ configurable CLI flags
│   ├── benchmark/
│   │   ├── _utils.py              # 13-material benchmark definitions with expected
│   │   │                          #   properties (Ef, EaH, band gap, stability flag),
│   │   │                          #   structure generators (rocksalt, anti-fluorite,
│   │   │                          #   hexagonal layered, olivine, beta-Li-like)
│   │   ├── run_benchmark.py       # Evaluate benchmark set against a single checkpoint,
│   │   │                          #   outputs CSV report + critical analysis summary
│   │   ├── benchmark_suite.py     # ~160 generated crystal structures across 10 families
│   │   │                          #   (rocksalt, CsCl, zincblende, fluorite, anti-fluorite,
│   │   │                          #   perovskite, layered), versioned JSON output
│   │   └── compare_benchmarks.py  # Side-by-side comparison of 2+ checkpoints on the
│   │                              #   13-material benchmark, common-subset filtering,
│   │                              #   per-material CSV export
│   └── maintenance/
│       ├── rebuild_li_dataset.py  # Download Li≥5% MP structures (20k+ confirmed),
│       │                          #   filter by coverage + size, subsample to 10k,
│       │                          #   save as raw pipeline input, report family distribution
│       ├── start_api.sh           # Launch FastAPI via uvicorn on port 8000
│       └── start_streamlit.sh     # Launch Streamlit app on port 8501 (headless)
│
├── configs/                       # YAML/JSON configuration files
│   ├── model_config.yaml          # v1 architecture: hidden_dim=256, 4 ALIGNN layers
│   ├── model_config_v2.yaml       # v2 architecture: hidden_dim=128, 2 ALIGNN layers
│   ├── model_config_v3.yaml       # v3 with log-Eah transform, 5 tasks, PINN coefficients
│   ├── model_config_v3_li.yaml    # Active training config: Li-only, 3 tasks, TwoStageEah,
│   │                              #   GradNorm enabled, no log-Eah
│   ├── phase3_config_log_eah.yaml # Phase 3 experiment: log-transformed EaH targets
│   ├── finetune_config.yaml       # Fine-tuning parameters
│   ├── data_config.yaml           # Dataset build parameters
│   ├── deploy_config.yaml         # Deployment configuration
│   └── ds_config.json             # DeepSpeed ZeRO-2 optimization config
│
├── streamlit_app/                 # Browser-based interactive screening dashboard
│   ├── streamlit_app.py           # Main app: monochrome design system, 4-page navigation
│   │                              #   (Dashboard, Screen, Batch, Results)
│   ├── requirements.txt           # Python dependencies for streamlit deployment
│   └── pages/
│       ├── screen.py              # Single-material CIF upload → validation → property
│       │                          #   prediction → literature comparison → stability
│       │                          #   bar → conductivity bar → reliability score →
│       │                          #   recommendations → scientific summary (~760 lines)
│       ├── batch.py               # Batch screening via MP IDs or formulas, API submission,
│       │                          #   family-diversity reranking, Altair bar chart
│       ├── dashboard.py           # System status (inference engine, API, coverage),
│       │                          #   material family table, capabilities cards
│       └── results.py             # Job status polling, progress bar, ranked candidate
│                                  #   table, Altair highlight chart
│
├── api/                           # FastAPI REST backend
│   ├── __init__.py
│   ├── main.py                    # FastAPI app: /screen (POST async), /screen/upload
│   │                              #   (CIF file), /job/{job_id} (polling), /health;
│   │                              #   JWT-protected, Celery task dispatch, SQLAlchemy
│   ├── models.py                  # Pydantic models: MaterialScreeningResult, JobStatus
│   ├── database.py                # SQLAlchemy ORM: Material, ScreeningResult, Job tables;
│   │                              #   PostgreSQL default; get_engine, get_session
│   ├── auth.py                    # JWT auth: create_access_token, verify_token (HS256)
│   └── tasks.py                   # Celery app: screen_materials_task (async screening
│                                  #   with progress tracking, Pareto ranking, retry)
│
├── frontend/                      # React web dashboard (Vite-based)
│   ├── public/
│   │   └── scandium.svg           # App logo
│   ├── src/
│   │   ├── main.jsx               # React entry point
│   │   ├── App.jsx                # React Router (4 routes), nav bar with gradient
│   │   ├── index.css              # Tailwind-style global CSS
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # System status cards, material families
│   │   │   ├── Screening.jsx      # Single-material CIF upload + prediction
│   │   │   ├── Results.jsx        # Batch job status + ranked candidates
│   │   │   └── ApiDocs.jsx        # API reference documentation
│   │   ├── components/            # Reusable React components (empty, WIP)
│   │   └── utils/
│   │       └── api.js             # Axios-based API client
│   └── dist/                      # Built static assets (deployment)
│
├── tests/                         # Test suite (pytest)
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures (empty)
│   ├── test_data.py               # DataCleaner empty-input, PropertyNormalizer
│   │                              #   fit/transform, save/load round-trip
│   ├── test_models.py             # CrystalMPNN forward shape, ScandiumPINNGNN
│   │                              #   creation + forward pass
│   ├── test_pipeline.py           # Normalizer round-trip (v1, v2), log-Eah
│   │                              #   round-trip, checkpoint self-containment
│   │                              #   (has config/model/metrics), checkpoint
│   │                              #   normalizer consistency, PINNLoss in
│   │                              #   physical units (thermodynamic reg),
│   │                              #   inference must NOT denormalize outputs
│   ├── test_inference.py          # ParetoRanker (if it exists)
│   ├── test_api.py                # API endpoint tests
│   ├── test_training_normalization.py  # Training normalization tests
│   ├── test_data_audit.py         # Label coverage analysis
│   └── test_reference_materials.py     # Li6PS5Cl end-to-end smoke test
│
├── datasets/                      # Preprocessed graph datasets (versioned)
│   ├── v1_817/                    # v1: 817 structures (general inorganic, superseded)
│   │   └── raw/                   # Raw downloaded MP/OQMD data
│   ├── v2_1000_smoketest/         # v2 smoke-test subset (1000 structures)
│   │   └── raw/
│   ├── v2_10000/                  # v2: 10k structures (superseded, general inorganic)
│   │   └── raw/                   # Contains dataset_cache.pt, split_indices.pt,
│   │                              #   normalizer.json, metadata.json
│   ├── v2_10000_log_eah/          # v2 variant: log-transformed EaH targets
│   └── v3_li_10000/               # Active dataset: 10k Li≥5% structures from MP,
│                                  #   family-balanced splits, TwoStageEah labels
│
├── data/                          # Preprocessed / cached data & artifacts
│   ├── raw/                       # Empty — raw downloads go to datasets/*/raw/
│   ├── processed/                 # data/processed/prebuilt_graphs.pt,
│   │                              #   dataset_cache.pt, split_indices.pt (legacy)
│   ├── splits/                    # Split index files (empty, unused)
│   ├── benchmark_cifs/            # Benchmark CIF files (e.g., Li6PS5Cl.cif)
│   ├── baseline_v1.0.json         # Baseline results: 817-structure model metrics,
│   │                              #   normalizer stats, sample predictions, pipeline
│   │                              #   bug-fix history
│   └── normalizer.json            # Global PropertyNormalizer statistics
│
├── checkpoints/                   # Trained model weights
│   ├── best_model.pt              # Best validation model (v2_10000, hidden_dim=128)
│   └── norm_best_model.pt         # Normalized variant
│
├── experiments/                   # Versioned experiment artifacts (created by
│   │                              #   experiment_sweep.py or cross_validate.py)
│   ├── cv/                        # Cross-validation results
│   └── reports/                   # Phase report JSON files
│
├── docs/                          # Documentation (14 files)
│   ├── ARCHITECTURE.md            # Model & system architecture deep dive
│   ├── DATASETS.md                # Dataset documentation
│   ├── installation.md            # Installation guide
│   ├── training.md                # Training guide
│   ├── inference.md               # Inference guide
│   ├── api.md                     # API endpoint reference
│   ├── benchmarks.md              # Benchmark methodology & results
│   ├── experiments.md             # Experiment tracking
│   ├── DEVELOPMENT.md             # Developer setup & workflow
│   ├── troubleshooting.md         # Common issues & fixes
│   ├── faq.md                     # Frequently asked questions
│   ├── DOCS.md                    # Documentation index
│   ├── PROJECT_AUDIT.md           # Code quality audit
│   └── RESEARCH_PLAN.md           # Research roadmap & objectives
│
├── logs/                          # Training log files
├── outputs/                       # Generated output files
│
├── docker/                        # Containerization
│   ├── Dockerfile.api             # FastAPI inference server
│   ├── Dockerfile.training        # Training environment (GPU)
│   ├── Dockerfile.worker          # Celery async worker
│   └── docker-compose.yml         # 6-service orchestration (api, worker, redis,
│                                  #   postgres, training, frontend)
│
├── archive/                       # Historical/experimental code (not in active use)
│   ├── scripts/                   # Archived research scripts
│   ├── src/                       # Archived source modules
│   ├── datasets/                  # Archived datasets
│   ├── checkpoints/               # Archived checkpoints
│   └── experiments/               # Archived experiment runs
│
├── .local/                        # Local environment artifacts
└── .ruff_cache/                   # Ruff linter cache (0.15.20)
```
