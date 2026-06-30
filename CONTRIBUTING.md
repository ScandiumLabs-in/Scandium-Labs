# Contributing to Scandium Labs

Thank you for your interest in contributing! We welcome bug reports, feature requests, documentation improvements, and code contributions.

## How to Contribute

### Reporting Bugs
- Search [existing issues](https://github.com/scandium-labs/scandium-labs/issues) to avoid duplicates
- Use the **Bug Report** template
- Include: environment details, steps to reproduce, expected vs actual behavior, logs/screenshots if relevant

### Feature Requests
- Open a **Feature Request** issue
- Describe the problem you want to solve and the proposed solution
- Explain why it fits within the project's scope

### Pull Requests
1. Fork the repository and create a feature branch from `main`
2. Follow the [development workflow](#development-workflow)
3. Open a PR and fill out the [pull request template](#pull-request-template)
4. Ensure all status checks pass

## Development Workflow

1. **Set up your environment**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Create a branch**
   ```bash
   git checkout -b feat/my-feature  # or fix/my-bugfix
   ```

3. **Make changes** — keep commits small and descriptive

4. **Run checks before committing**
   ```bash
   make lint        # ruff + black
   make typecheck   # mypy or pyright
   make test        # pytest
   ```

5. **Push and open a PR**

## Code Review Process

- Every PR requires at least **one approval** from a maintainer
- Reviews focus on correctness, security, performance, and style
- Address review feedback with additional commits — avoid force-pushing until requested
- Large or risky changes may require multiple reviewers

## Testing Requirements

Before submitting a PR:
- [ ] All existing tests pass (`make test`)
- [ ] New code includes tests (unit tests for logic, integration tests for APIs)
- [ ] Test coverage does not decrease meaningfully
- [ ] Edge cases (empty inputs, network errors, invalid data) are handled

Run tests with:
```bash
pytest tests/ -v
```

## Style Guide

This project enforces code style via:
- **Ruff** for linting and import sorting
- **Black** for formatting (line length: 88)
- **mypy** / **pyright** for type checking

Run `make lint` and `make typecheck` before pushing. CI will reject PRs that fail these checks.

## Pull Request Template

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
- [ ] Code follows project style (ruff + black)
- [ ] Type hints are correct (mypy passes)
- [ ] No new warnings or errors
- [ ] Documentation updated (if applicable)
```

## Community Guidelines

- Be respectful and inclusive — see our [Code of Conduct](CODE_OF_CONDUCT.md)
- Provide constructive feedback on others' work
- Ask questions early if something is unclear
- Keep discussions focused and on-topic
