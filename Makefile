.PHONY: install train eval gui test lint

install:
	pip install -e .[dev]

train:
	python -m src.models.train --config configs/train.yaml

eval:
	python -m src.models.evaluate --config configs/evaluate.yaml

gui:
	python -m src.models.gui_predict --config configs/predict.yaml

test:
	pytest

lint:
	ruff check src tests
