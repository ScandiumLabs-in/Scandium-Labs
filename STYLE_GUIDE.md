# Scandium Labs Style Guide

This document defines the coding conventions and best practices for all Scandium Labs projects. Adherence ensures consistency, readability, and maintainability across the codebase.

---

## 1. Python Coding Conventions

All Python code follows **PEP 8** with the exceptions noted below.

- **Line length**: 88 characters for code (matching Black's default), 72 characters for docstrings and comments.
- **Indentation**: 4 spaces. No tabs.
- **Blank lines**:
  - Two blank lines between top-level definitions (classes, functions).
  - One blank line between method definitions inside a class.
- **Trailing whitespace**: Not permitted. End every file with a single newline.
- Use **Black** for auto-formatting and **Ruff** for linting. Configuration lives in `pyproject.toml`.

```bash
black src/ tests/
ruff check src/ tests/
```

---

## 2. Import Ordering

Imports must be grouped in the following order, with a blank line between each group:

1. **Standard library** (`os`, `sys`, `json`, `pathlib`, etc.)
2. **Third-party libraries** (`torch`, `numpy`, `pandas`, `fastapi`, etc.)
3. **Local application imports** (modules within `scandium_labs`)

Within each group, sort alphabetically. Prefer **absolute imports** over relative imports.

```python
# Correct
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter

from scandium_labs.api.schemas import HealthResponse
from scandium_labs.core.config import Settings
```

---

## 3. Naming Conventions

| Construct        | Convention                     | Example                      |
|------------------|--------------------------------|------------------------------|
| Classes          | `PascalCase`                   | `ModelTrainer`, `DatasetLoader` |
| Functions/methods| `snake_case`                   | `train_model()`, `load_config` |
| Variables        | `snake_case`                   | `batch_size`, `learning_rate`  |
| Constants        | `UPPER_CASE`                   | `MAX_EPOCHS`, `DEFAULT_SEED`   |
| Private members  | `_leading_underscore`          | `_internal_state`              |
| Name-mangled     | `__double_leading_underscore`  | `__cache` (use sparingly)      |
| Modules / files  | `snake_case`                   | `data_loader.py`, `train.py`   |
| Packages         | `short_lower_case`             | `scandium_labs`                |

- Single-character names are forbidden except for trivial loop indices (`i`, `j`, `k`).
- Avoid abbreviations unless they are universally understood (`lr`, `img`, `config`).

---

## 4. Type Hints

**Always** annotate public functions and methods. Use type hints from the `typing` module where applicable.

```python
from typing import Dict, List, Optional, Tuple


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    ...
```

- Use `Optional[X]` instead of `Union[X, None]`.
- Use `list[X]`, `dict[K, V]` (lowercase) for Python 3.9+.
- Prefer `Self` return type on class methods returning `self`.
- Private helper functions should also be typed; untyped code will be rejected in review.

---

## 5. Docstrings (Google Style)

Every public module, class, method, and function must have a docstring. Use **Google-style** docstrings.

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

- **One-line docstrings**: Use only when the function is trivial and needs no explanation.
- Sections: `Args:`, `Returns:`, `Raises:`, `Yields:`, `Note:`.
- Document types in docstrings only when the type hint is insufficient (e.g., shape of arrays).

---

## 6. Logging

Use Python's `logging` module. **Never** use `print()` in production code.

```python
import logging

logger = logging.getLogger(__name__)
```

- Module-level logger instantiated as `logger = logging.getLogger(__name__)`.
- Log levels: `DEBUG` → `INFO` → `WARNING` → `ERROR` → `CRITICAL`.
- F-strings are acceptable in log messages (Python 3.8+ lazy evaluation is not required).
- Avoid logging sensitive data (passwords, tokens, PII).

---

## 7. Error Handling

- **Never** use bare `except:` — always catch specific exception types.
- **Never** use `except Exception:` without a very good reason (document it).
- Raise specific exceptions: `ValueError`, `TypeError`, `FileNotFoundError`, etc.
- Define custom exception classes in `scandium_labs/exceptions.py` for domain-specific errors.

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

- Use `raise` without argument when re-raising the same exception.
- Context managers (`with` blocks) are preferred for resource management.

---

## 8. File Organization

---

One logical concept per file where reasonable.

```
src/
└── scandium_labs/
    ├── __init__.py
    ├── exceptions.py          # All custom exceptions
    ├── config.py              # Configuration / settings
    ├── models/                # One class per file
    │   ├── __init__.py
    │   ├── base.py
    │   ├── transformer.py
    │   └── classifier.py
    ├── data/                  # Data loading and processing
    │   ├── __init__.py
    │   ├── loader.py
    │   └── augmentations.py
    ├── training/              # Training loops and utilities
    │   ├── __init__.py
    │   ├── trainer.py
    │   └── metrics.py
    └── api/                   # API layer
        ├── __init__.py
        ├── routes.py
        └── schemas.py
```

- `__init__.py` files should be minimal (re-export public symbols only).
- Keep functions short (under 50 lines); if they grow, refactor.

---

## 9. Testing Conventions

- Framework: **pytest**.
- Test files live in `tests/` and mirror the `src/` structure.
- File naming: `test_<module_name>.py`.
- Test function naming: `test_<feature>[_<scenario>]`.
- Use `conftest.py` for shared fixtures.

```
tests/
├── conftest.py
├── test_config.py
├── test_metrics.py
└── models/
    ├── conftest.py
    ├── test_transformer.py
    └── test_classifier.py
```

```python
# tests/test_metrics.py
def test_accuracy_perfect_match() -> None:
    preds = np.array([1, 0, 1])
    targets = np.array([1, 0, 1])
    assert calculate_accuracy(preds, targets) == 1.0


def test_accuracy_no_match() -> None:
    preds = np.array([1, 1, 0])
    targets = np.array([0, 0, 1])
    assert calculate_accuracy(preds, targets) == 0.0
```

- Aim for at least 80% test coverage on new code.
- Use `pytest.mark.parametrize` for testing multiple input combinations.
- Avoid mocking external services in unit tests; use fixtures or dependency injection.

---

## 10. Git / Commit Conventions

Follow **Conventional Commits**:

```
<type>(<scope>): <short summary>

<body (optional)>

<footer (optional)>
```

**Types**:

| Type       | Usage                                  |
|------------|----------------------------------------|
| `feat`     | New feature                            |
| `fix`      | Bug fix                                |
| `docs`     | Documentation changes                   |
| `style`    | Formatting (no logic change)            |
| `refactor` | Code restructuring (no behavior change) |
| `perf`     | Performance improvement                 |
| `test`     | Adding or modifying tests               |
| `chore`    | Build / CI / tooling changes            |

**Scopes** (examples): `api`, `models`, `data`, `trainer`, `config`.

**Rules**:
- Summary uses imperative mood, capitalized, no trailing period.
- Keep the summary under 72 characters.
- Use the body to explain *why* the change was made.
- Reference issues/PRs in the footer: `Closes #42`.

```
feat(trainer): add gradient accumulation support

Implements gradient accumulation across N batches to simulate larger
batch sizes on memory-constrained hardware.

Closes #127
```

- Commits must compile and pass tests before being pushed.
- Write atomic commits: each commit is a single logical change.

---

## 11. Configurations & Dependencies

- All project-wide configuration lives in `pyproject.toml`.
- Dependencies are pinned to exact versions in `requirements/` (with `requirements.txt` as the lockfile for production).
- Use `pip-tools` for dependency management.
- Environment-specific settings go into `.env` files (never committed).

---

## 12. Code Review Checklist

Before requesting review, verify:

- [ ] Formatted with Black (passes `ruff check` with zero warnings).
- [ ] All public functions have type hints.
- [ ] All public functions have Google-style docstrings.
- [ ] No `print()` statements; logging is used correctly.
- [ ] No bare `except` clauses.
- [ ] Tests pass (`pytest tests/`).
- [ ] Coverage is >= 80% on new code.
- [ ] Commit messages follow Conventional Commits.
- [ ] No secrets, keys, or local paths committed.

---

*This style guide evolves. Propose changes by opening a PR against `STYLE_GUIDE.md`.*
