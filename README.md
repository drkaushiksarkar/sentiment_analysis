# Sentiment Analysis – Modular Package

This repository now exposes the original notebook experiments (IMDB sentiment and sarcasm detection) as a reusable Python package.  The code is organized into datasets, model builders, and training helpers, so the models can be trained from a CLI or imported into other services.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

> **TensorFlow note:**  
> - Linux/Windows CPU: `pip install tensorflow` (pulled automatically via `pip install -e .[dev]`).  
> - Apple Silicon: `pip install tensorflow-macos tensorflow-metal`.  
> - NVIDIA GPU: install CUDA/cuDNN as per the [official guide](https://www.tensorflow.org/install/pip) before running the pip command.

After installing, activate the virtual environment whenever you work on the project: `source .venv/bin/activate`.

## Package Layout

```
src/sentiment_package/
  imdb/      # IMDB loaders, model builders, and training API
  sarcasm/   # Sarcasm headline dataset utilities + Glove support
scripts/
  train_imdb.py
  train_sarcasm.py
```

## Example Usage

Train the dense IMDB baseline:

```bash
python scripts/train_imdb.py --model dense --epochs 4 --checkpoint-dir artifacts/imdb_dense
```

Train the CNN sarcasm model with a GloVe matrix:

```bash
python scripts/train_sarcasm.py --model conv --glove-path /path/to/glove.6B.100d.txt --epochs 20
```

Both scripts expose flags for vocabulary size, max sequence length, epoch count, and checkpoint directory.

You can also import the modules directly:

```python
from sentiment_package.imdb import data, models, train

splits = data.load_dataset(data.ImdbDatasetConfig(max_length=128))
model = models.build_dense_model(models.DenseModelConfig(vocab_size=10000, max_length=128))
train.train_model(model, splits, train.TrainingConfig(epochs=4))
```

## Verifying Checkpoints

1. Activate the virtual environment and ensure TensorFlow is installed (see Getting Started).  
2. Run a short training job (single epoch) to keep runtime low:

```bash
python scripts/train_imdb.py --model dense --epochs 1 --checkpoint-dir artifacts/imdb_dense
python scripts/train_sarcasm.py --model dense --epochs 1 --checkpoint-dir artifacts/sarcasm_dense
```

3. Each command creates directories under `artifacts/` containing `.keras` checkpoint files. Adjust `--model`, `--glove-path`, and `--epochs` as needed for longer experiments.

## Tests

Basic import tests are available and can be executed with:

```bash
pytest
```

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) provisions Python 3.11, installs the package inside a virtual environment, runs `ruff check .`, and executes `pytest` on every push or pull request targeting `main`. Ensure commits keep lint/test results green before requesting reviews.
