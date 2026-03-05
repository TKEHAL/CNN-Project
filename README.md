# Fruit Image Classification (CNN)

Professional ML project structure for training and evaluating a CNN on the fruit dataset.

## Project Structure

- `configs/`: YAML configuration files
- `data/raw/`: raw datasets (current source: `data/raw/fruits`)
- `data/processed/`: generated metadata/splits/features
- `notebooks/`: experimentation notebooks
- `reports/`: metrics and visual outputs
- `src/`: source code package
  - `src/data/`: dataset inspection and loading tools
  - `src/features/`: preprocessing/transforms
  - `src/models/`: model definition, train/eval/predict scripts
  - `src/utils/`: config/logging helpers
- `tests/`: unit tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Train:

```bash
python -m src.models.train --config configs/train.yaml
```

Evaluate:

```bash
python -m src.models.evaluate --config configs/evaluate.yaml
```

Predict one image:

```bash
python -m src.models.predict --config configs/predict.yaml --image /path/to/image.png
```

GUI interface (upload an image and predict):

```bash
python -m src.models.gui_predict --config configs/predict.yaml
```

or:

```bash
make gui
```

## Notes

- Existing notebook: `notebooks/CNNmodel.ipynb`
- Existing model weights: `src/models/modelCNN.h5`
- Default classes are inferred from subfolders in train directory.
