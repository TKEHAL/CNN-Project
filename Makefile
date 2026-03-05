.PHONY: install train eval test lint

install:
	pip install -e .[dev]

train:
	python -m src.models.train --config configs/train.yaml

eval:
	python -m src.models.evaluate --config configs/evaluate.yaml

test:
	pytest

lint:
	ruff check src tests
