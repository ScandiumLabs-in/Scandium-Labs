# Roadmap

## Short-term (Q3 2026)

### Model Improvements
- [x] Multi-task PINN framework with Arrhenius physics constraint
- [x] MC Dropout uncertainty quantification
- [x] Family-balanced dataset splitting
- [x] Two-stage EaH training head
- [ ] Implement proper hyperparameter search (Optuna)
- [ ] Experiment with equivariant graph networks (e.g., MACE, NequIP)
- [ ] Add ensemble-based uncertainty estimation
- [ ] Implement temperature scaling for calibration

### Data & Preprocessing
- [x] Li-structured dataset (20k+ structures from Materials Project)
- [x] LazyGraphDataset for memory-efficient training
- [ ] Add non-Li solid electrolyte datasets (NASICON, garnet, perovskite)
- [ ] Implement data augmentation: supercell perturbation, vacancy doping
- [ ] Create standardized benchmarking suite

### Infrastructure
- [x] Package installation via `pip install -e .`
- [x] Automated code formatting (Ruff, Black, isort)
- [x] Makefile with common commands
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Pre-commit hooks fully configured
- [ ] Docker image for reproducible training

## Medium-term (Q4 2026)

### Scientific Goals
- [ ] Achieve state-of-the-art ionic conductivity prediction for Li solid electrolytes
- [ ] Publish benchmark results on MatBench and other standardized datasets
- [ ] Extify to Na, Mg, and Zn solid electrolyte systems
- [ ] Implement crystal structure generation (diffusion-based or GFlowNet)
- [ ] Add DFT validation pipeline for top predictions

### Engineering
- [ ] TorchScript / ONNX export for deployment
- [ ] Model quantization for edge deployment
- [ ] REST API with authentication and rate limiting
- [ ] Web dashboard with interactive visualizations
- [ ] Experiment tracking database (PostgreSQL)

### Community
- [ ] Open-source release with comprehensive documentation
- [ ] Contribution guide and issue templates
- [ ] Tutorial notebooks (Jupyter/Colab)
- [ ] Discord/Slack community channel

## Long-term (2027+)

### Research Directions
- [ ] Multi-fidelity learning (combining DFT, MD, and experimental data)
- [ ] Active learning for efficient materials exploration
- [ ] Inverse design: generate structures with target property profiles
- [ ] Composition-structure-property foundation model for ionic conductors
- [ ] Integration with autonomous synthesis platforms

### Production
- [ ] Managed cloud API service
- [ ] Materials discovery campaign with experimental validation
- [ ] Patent filings for novel compositions discovered
- [ ] Journal publication in Nature Computational Science / NPJ Computational Materials
