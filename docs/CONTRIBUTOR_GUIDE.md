# Contributor Guide

Thank you for your interest in contributing to Scandium Labs! This guide describes how to contribute code, documentation, tests, or bug reports to the project.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Coding Standards](#2-coding-standards)
3. [Naming Conventions](#3-naming-conventions)
4. [Git Workflow](#4-git-workflow)
5. [Pull Request Process](#5-pull-request-process)
6. [Testing](#6-testing)
7. [Documentation Standards](#7-documentation-standards)
8. [Issue Reporting](#8-issue-reporting)
9. [Feature Request Process](#9-feature-request-process)
10. [Code Review Process](#10-code-review-process)
11. [Community Guidelines](#11-community-guidelines)

---

## 1. Repository Structure

```
scandium-labs/
├── src/                    # Core Python package
│   ├── chemistry/          # Chemical featurization
│   ├── data/               # Data collection, cleaning, splitting
│   ├── evaluation/         # Metrics, OOD detection
│   ├── explainability/     # Model interpretability
│   ├── graphs/             # Graph construction, features
│   ├── inference/          # Prediction engine, ranking
│   ├── models/             # Architectures
│   │   ├── gnn/            #   ALIGNN, CrystalMPNN, layers
│   │   └── heads/          #   Two-stage EaH, pretrained encoder
│   ├── training/           # Trainer, losses, schedulers
│   └── utils/              # Config, logging, I/O
├── scripts/                # CLI entrypoints
│   ├── train/              # Training scripts
│   ├── preprocess/         # Dataset building, caching
│   ├── inference/          # Candidate screening
│   ├── evaluate/           # Cross-validation, benchmarks
│   ├── maintenance/        # Profiling, debugging
│   └── analyze/            # Analysis helpers
├── configs/                # YAML/JSON configurations
├── api/                    # FastAPI backend
├── streamlit_app/          # Streamlit dashboard
├── frontend/               # Vue.js web app
├── tests/                  # Pytest test suite
├── docs/                   # Documentation
├── datasets/               # Preprocessed dataset versions
├── checkpoints/            # Trained model weights
├── runs/                   # Experiment outputs
└── archive/                # Historical/deprecated code
```

### Module Responsibilities

| Module | Responsibility | Maintainer |
|---|---|---|
| `src/data/` | All data operations: collect, clean, split, normalize, dataset classes | Data team |
| `src/models/` | Model architectures: layers, heads, full model definition | Model team |
| `src/graphs/` | Crystal/line graph construction and featurization | Graph team |
| `src/training/` | Training loops, losses, optimization, experiment tracking | Training team |
| `src/inference/` | Inference engine, candidate screening, recommendation | Inference team |
| `src/evaluation/` | Metrics computation, cross-validation, OOD | Evaluation team |
| `src/explainability/` | Attention visualization, integrated gradients | Interpretability team |
| `src/chemistry/` | Chemical family classification, property computation | Chemistry team |
| `src/utils/` | Shared utilities: config loading, logging, file I/O | Infrastructure team |

---

## 2. Coding Standards

This project follows the coding conventions defined in `STYLE_GUIDE.md`. Below is a summary of the most important rules enforced by CI.

### Python Version

**Python 3.10+** is required. The development environment uses Python 3.12.13. Use type hints compatible with `from __future__ import annotations`.

### Formatting and Linting (Enforced by CI)

| Tool | Purpose | Configuration |
|---|---|---|
| **Ruff** | Linting + import sorting | `pyproject.toml` — `[tool.ruff]` |
| **Black** | Code formatting | Line length: 100 |
| **mypy** | Type checking | `--ignore-missing-imports` |

Run checks before committing:

```bash
make lint        # ruff check src/ scripts/ tests/ api/
make format      # ruff format + isort
make typecheck   # mypy (best-effort, non-blocking)
```

### Ruff Configuration (from `pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

Selected rules:
- **E** — pycodestyle errors
- **F** — pyflakes (logic errors, undefined names)
- **I** — import order
- **N** — naming conventions
- **W** — pycodestyle warnings
- **UP** — pyupgrade (modern Python idioms)

### Python Conventions

- **Line length**: 100 characters maximum.
- **Indentation**: 4 spaces. No tabs.
- **Blank lines**: Two between top-level definitions, one between methods.
- **Trailing whitespace**: Not permitted.
- **File ending**: Single trailing newline.

### Imports (Grouped and Sorted)

```python
# 1. Standard library
import json
import os
from pathlib import Path
from typing import Optional

# 2. Third-party libraries
import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit

# 3. Local application imports
from src.data.cleaner import DataCleaner, PropertyNormalizer
from src.data.dataset import LazyGraphDataset
```

### Type Hints

All public functions and methods must have type hints:

```python
from typing import Dict, List, Optional

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    ...
```

### Logging

Use `logging.getLogger(__name__)` — never `print()` in production code:

```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Epoch {epoch}: loss={loss:.4f}")
logger.debug("Loading graph %d from cache", idx)
```

### Error Handling

- Catch specific exceptions only — no bare `except:`.
- Raise specific exception types (`ValueError`, `TypeError`, `FileNotFoundError`).
- Use `raise` without args when re-raising.

```python
# Correct
try:
    result = risky_operation()
except (ValueError, KeyError) as exc:
    logger.error("Operation failed: %s", exc)
    raise

# Wrong
try:
    result = risky_operation()
except:
    pass
```

### Docstrings (Google Style)

```python
def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Model output probabilities, shape (N,).
        targets: Ground-truth binary labels, shape (N,).
        threshold: Decision boundary for converting probabilities to labels.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score.

    Raises:
        ValueError: If predictions and targets have different lengths.
    """
    ...
```

---

## 3. Naming Conventions

| Construct | Convention | Example |
|---|---|---|
| Classes | `PascalCase` | `ScandiumTrainer`, `CrystalGraphBuilder` |
| Functions/methods | `snake_case` | `compute_metrics()`, `load_data()` |
| Variables | `snake_case` | `batch_size`, `hidden_dim` |
| Constants | `UPPER_CASE` | `MAX_EPOCHS`, `DEFAULT_SEED` |
| Private members | `_leading_underscore` | `_internal_state` |
| Name-mangled | `__double_leading` | `__cache` (use sparingly) |
| Modules/files | `snake_case` | `data_loader.py`, `train.py` |
| Packages | `short_lower_case` | `scandium_labs` |

### Consistency Rules

- Single-character names are forbidden except for loop indices (`i`, `j`, `k`).
- Avoid abbreviations unless universally understood (`lr`, `config`, `img`).
- Boolean variables should read as questions: `is_training`, `use_amp`, `has_structure`.
- Private methods indicate internal APIs not intended for external use.

---

## 4. Git Workflow

### Branch Model

```
main              Production-ready code (protected)
  ├── feat/*      New features (branched from main)
  ├── fix/*       Bug fixes (branched from main)
  ├── refactor/*  Code restructuring (branched from main)
  ├── perf/*      Performance improvements (branched from main)
  └── docs/*      Documentation changes (branched from main)
```

### Branch Naming

```
feat/add-ionic-conductivity-head
fix/normalizer-nan-handling
refactor/extract-graph-builder
perf/dataloader-multiprocessing
docs/update-model-card
```

### Commit Style: Conventional Commits

```
<type>(<scope>): <short summary>

<body (optional)>

<footer (optional)>
```

**Types:**

| Type | Usage | Example |
|---|---|---|
| `feat` | New feature | `feat(models): add equivariant convolution layer` |
| `fix` | Bug fix | `fix(data): handle NaN in energy_above_hull targets` |
| `docs` | Documentation | `docs(data): update dataset card with v3 stats` |
| `style` | Formatting only | `style: apply ruff formatting to all src/ files` |
| `refactor` | Code restructure | `refactor(trainer): split trainer.py into submodules` |
| `perf` | Performance | `perf(dataloader): increase num_workers to 4` |
| `test` | Tests | `test(models): add test for TwoStageEahHead` |
| `chore` | Tooling/CI | `chore: update ruff config to v0.5` |

**Rules:**
- Summary uses imperative mood, capitalized, no trailing period.
- Keep summary under 72 characters.
- Use body to explain *why* the change was made.
- Reference issues in footer: `Closes #42`.

```
feat(models): add Arrhenius gating to PINNConstraintModule

Implements physics-informed feature gating using σ(W·h) for
Arrhenius-like conductivity-temperature relationships.

Closes #128
```

### Git Best Practices

- **Atomic commits**: Each commit should be a single logical change.
- **Commit early, commit often**: Small commits are easier to review and revert.
- **Write descriptive messages**: The summary should be enough to understand the change.
- **Don't commit broken code**: Always run tests before committing.

---

## 5. Pull Request Process

### Step 1: Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feat/my-feature
```

### Step 2: Make Changes and Commit

```bash
# Stage changes
git add src/models/new_module.py
git add tests/test_new_module.py

# Commit with Conventional Commits format
git commit -m "feat(models): add new graph transformer variant"
```

### Step 3: Run All Checks

```bash
make lint        # Must pass with zero errors
make typecheck   # Address any type errors (best-effort)
make test        # All tests must pass
```

### Step 4: Push and Open PR

```bash
git push origin feat/my-feature
```

Then open a pull request on GitHub using the template below.

### Pull Request Template

```markdown
## Summary
<!-- One-line description of the change -->

## Related Issue
Closes #ISSUE_NUMBER

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows project style (ruff + black pass with zero warnings)
- [ ] Type hints are correct (mypy passes)
- [ ] No new warnings or errors
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow Conventional Commits
```

### Step 5: Address Review Feedback

- Make additional commits to address review comments.
- **Do not force-push** unless specifically requested by the reviewer.
- Respond to each comment, even if only with a "Fixed" acknowledgment.

### Step 6: Merge

- PRs require at least **one approval** from a maintainer.
- All CI checks must pass.
- Use **squash merge** for feature branches (clean single commit).
- Use **merge commit** for collaborative branches with multiple contributors.

---

## 6. Testing

### Test Framework

This project uses **pytest** as the test framework.

```bash
# Run all tests
make test
# or
python -m pytest tests/ -q --tb=short

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term

# Run specific test file
python -m pytest tests/test_models.py -v

# Run specific test
python -m pytest tests/test_models.py::test_model_forward -v
```

### Test Organization

Test files mirror the `src/` structure:

```
tests/
├── conftest.py               # Shared fixtures (currently empty)
├── test_data.py               # Data collection, cleaning, normalization
├── test_models.py             # Model construction, forward pass
├── test_training_normalization.py  # Training + normalization integration
├── test_pipeline.py           # End-to-end pipeline
├── test_inference.py          # Inference engine
├── test_api.py                # API endpoints
├── test_data_audit.py         # Coverage gating, filtering
└── test_reference_materials.py # Known SSE validation
```

### Test Naming

- File: `test_<module_name>.py`
- Function: `test_<feature>[_<scenario>]`

```python
def test_collector_returns_dataframe():
    ...

def test_cleaner_removes_nan_formation_energy():
    ...

def test_model_forward_produces_correct_output_shapes():
    ...
```

### Writing Tests

```python
import pytest
import torch
import numpy as np

def test_model_forward():
    """Test that ScandiumPINNGNN produces correct output shapes."""
    model = ScandiumPINNGNN(hidden_dim=64, num_alignn_layers=2)
    crystal_graph, line_graph = create_dummy_graphs()
    output = model(crystal_graph, line_graph)

    assert "formation_energy" in output
    assert output["formation_energy"].ndim == 1
    assert output["formation_energy"].shape[0] == 1

@pytest.mark.parametrize("hidden_dim", [64, 128])
def test_model_different_sizes(hidden_dim):
    """Test model works with different hidden dimensions."""
    model = ScandiumPINNGNN(hidden_dim=hidden_dim, num_alignn_layers=2)
    assert sum(p.numel() for p in model.parameters()) > 0
```

### Test Requirements

Before submitting a PR:

- [ ] All existing tests pass (`make test`).
- [ ] New code includes tests (unit tests for logic, integration for workflows).
- [ ] Test coverage does not decrease meaningfully.
- [ ] Edge cases (empty inputs, invalid data, NaN handling) are covered.
- [ ] Tests are deterministic (no random failures).

### Test Configuration (from `pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

---

## 7. Documentation Standards

### Documentation System

Documentation is maintained as Markdown files in the `docs/` directory. There are currently 25 documentation files covering:

| Category | Documents |
|---|---|
| **Architecture** | `ARCHITECTURE.md`, `PROJECT_STRUCTURE.md` |
| **Data** | `DATASETS.md`, `DATA_CARD.md` |
| **Model** | `MODEL_CARD.md` |
| **Training** | `training.md`, `TRAINING_SPEEDUP_PLAN.md` |
| **Optimization** | `OPTIMIZATION_REPORT.md`, `RESOURCE_PROFILES.md`, `BOTTLENECK_REPORT.md` |
| **Graph pipeline** | `GRAPH_PIPELINE.md`, `MEMORY_PROFILE.md` |
| **Research** | `RESEARCH_PLAN.md`, `experiments.md`, `benchmarks.md` |
| **API** | `api.md`, `inference.md` |
| **DevOps** | `installation.md`, `DEVELOPMENT.md`, `troubleshooting.md`, `faq.md` |

### Documentation Guidelines

1. **Use GitHub-flavored Markdown** with fenced code blocks, tables, and lists.
2. **Include code examples** for all user-facing functionality.
3. **Keep documents current** — update docs when code changes.
4. **Link to source code** with file paths and line numbers where possible.
5. **Use Mermaid diagrams** for architecture and data flow.

### Docstrings

All public Python functions and classes require Google-style docstrings:

```python
def load_data(config: dict, data_dir: Path) -> tuple:
    """Load train/val/test DataLoaders from dataset.

    Args:
        config: Training configuration dictionary.
        data_dir: Path to the dataset directory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If dataset cache or split indices are missing.
    """
```

### README

The `README.md` is the project's front page. Keep it focused on:
- What the project does (one paragraph)
- Quick start (install → run → verify in < 5 steps)
- Key features and architecture overview
- Links to detailed documentation

---

## 8. Issue Reporting

### Bug Report Template

```markdown
## Bug Description
<!-- Clear, concise description of the bug -->

## Steps to Reproduce
1. Run command: `python scripts/train/train_v3_li.py --config ...`
2. Observe output at epoch 5
3. See error: ...

## Expected Behavior
<!-- What should happen instead -->

## Environment
- OS: [e.g., Linux 7.0]
- Python: [e.g., 3.12.13]
- PyTorch: [e.g., 2.6.0+cu124]
- GPU: [e.g., GTX 1650 4GB]
- Commit: [e.g., d30295b]

## Logs / Screenshots
<!-- Error messages, stack traces, screenshots -->

## Possible Solution
<!-- If you have an idea for the fix -->
```

### When to File a Bug

- Code crashes or produces incorrect results.
- Documentation is incorrect or misleading.
- Unexpected behavior contradicts the specification.
- Performance regression compared to previous versions.

### Before Filing

1. Search existing issues (both open and closed) for duplicates.
2. Check the troubleshooting guide (`docs/troubleshooting.md`).
3. Try the minimal reproduction case (smallest possible input that triggers the bug).
4. Include exact error messages and stack traces.

---

## 9. Feature Request Process

### Feature Request Template

```markdown
## Feature Description
<!-- What would you like to add or change -->

## Motivation
<!-- Why is this feature needed? What problem does it solve? -->

## Proposed Implementation
<!-- How might this be implemented? High-level approach -->

## Alternative Approaches
<!-- What other solutions have you considered? -->

## Impact
<!-- Who benefits from this feature? What is the expected effort? -->

- [ ] I am willing to implement this feature
```

### Feature Lifecycle

```
1. Feature Request (Issue)
        │
        ▼
2. Discussion (Maintainer + Community)
        │
        ▼
3. Acceptance Decision
   ├── Accepted → Added to roadmap
   └── Rejected → Closed with explanation
        │
        ▼
4. Implementation (PR)
        │
        ▼
5. Review + Merge
        │
        ▼
6. Release
```

### Feature Evaluation Criteria

Each feature request is evaluated on:
- **Alignment with project goals**: Does it fit the SSE discovery mission?
- **User impact**: How many users benefit?
- **Maintenance cost**: Is this a one-time addition or ongoing commitment?
- **Complexity**: Can it be implemented cleanly?
- **Testability**: Can we verify it works correctly?

### Current Priority Areas

| Priority | Area | Examples |
|---|---|---|
| High | Model architecture | Equivariant networks, improved EaH head |
| High | Data augmentation | Ionic conductivity dataset, external benchmarks |
| Medium | Infrastructure | Hyperparameter optimization, multi-GPU |
| Medium | Interpretability | Attention visualization, SHAP |
| Low | Deployment | Kubernetes, model serving |

---

## 10. Code Review Process

### Review Flow

1. Developer opens PR.
2. CI runs `make lint`, `make test`, `make typecheck`.
3. At least one maintainer reviews the code.
4. Developer addresses feedback with additional commits.
5. Reviewer approves.
6. PR is merged.

### What Reviewers Look For

| Category | Check |
|---|---|
| **Correctness** | Does the code do what it claims? Are edge cases handled? |
| **Security** | Are there injection vulnerabilities? Are secrets handled correctly? |
| **Performance** | Are there obvious performance issues? GPU memory considered? |
| **Style** | Does it pass ruff? Follows naming conventions? |
| **Types** | Are type hints correct? Any unchecked type violations? |
| **Tests** | Are tests added? Do they cover edge cases? |
| **Documentation** | Are docstrings updated? Is the README affected? |
| **Reproducibility** | Are random seeds set? Are configurations documented? |

### Reviewing Your Own Code

Before requesting review, self-check:

- [ ] `make lint` passes with zero warnings.
- [ ] `make test` passes (all tests).
- [ ] All public functions have type hints.
- [ ] All public functions have docstrings.
- [ ] No `print()` statements — use `logging`.
- [ ] No bare `except:` clauses.
- [ ] GPU memory usage is reasonable for target hardware.
- [ ] Configuration changes are backward-compatible (or documented).

---

## 11. Community Guidelines

### Code of Conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to:

- Be respectful and inclusive.
- Provide constructive feedback.
- Accept constructive criticism gracefully.
- Focus on what is best for the community.
- Show empathy toward other community members.

### Communication Channels

| Channel | Purpose |
|---|---|
| **GitHub Issues** | Bug reports, feature requests, task tracking |
| **Pull Requests** | Code review, technical discussion |
| **Project Board** | Roadmap and sprint tracking |

### Getting Help

- Read the documentation in `docs/`.
- Check the FAQ (`docs/faq.md`).
- Search existing issues for similar problems.
- Open an issue with the "question" label.

### Recognition

Contributors are recognized in:
- `CITATION.cff` — Authors list for software citation.
- Release notes — Key contributors per release.
- Pull requests — Merged PRs are attributed to their authors.

---

## Appendix A: Quick Reference

### Common Make Commands

```bash
make install        # pip install -e .
make install-dev    # pip install -e ".[dev]"
make train          # python scripts/train/train_v3_li.py
make test           # python -m pytest tests/ -q --tb=short
make lint           # ruff check src/ scripts/ tests/ api/
make format         # ruff format + isort
make typecheck      # mypy (best-effort)
make clean          # Remove __pycache__, .pyc, .egg-info
make reproduce      # bash reproduce.sh
make docs           # Print documentation message
```

### Pre-commit Hook Setup

```bash
pip install pre-commit
pre-commit install
```

Pre-commit runs ruff checks automatically before each commit.

### Updating Dependencies

```bash
# Add a new dependency to pyproject.toml [project.dependencies]
# Then reinstall
pip install -e .
```

---

*This contributor guide evolves with the project. Propose changes by opening a PR against `docs/CONTRIBUTOR_GUIDE.md`.*
