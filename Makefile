.PHONY: install train evaluate test lint format clean docs

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

train:
	python scripts/train/train_v3_li.py

evaluate:
	python scripts/evaluate/cross_validate.py

test:
	python -m pytest tests/ -q --tb=short

lint:
	ruff check src/ scripts/ tests/ api/

format:
	ruff format src/ scripts/ tests/ api/
	isort src/ scripts/ tests/ api/

typecheck:
	python -m mypy src/ --ignore-missing-imports || true

docs:
	python -c "print('Documentation available in docs/')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info

reproduce:
	bash reproduce.sh

docker-build:
	docker-compose build

docker-up:
	docker-compose up
