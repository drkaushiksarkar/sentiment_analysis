"""Training utilities for IMDB sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from . import data as imdb_data
from . import models as imdb_models


@dataclass
class TrainingConfig:
    """Generic Keras training settings."""

    batch_size: int = 128
    epochs: int = 4
    learning_rate: float = 1e-3
    checkpoint_dir: Path | str = Path("artifacts/imdb")
    checkpoint_pattern: str = "weights.{epoch:02d}.keras"
    use_early_stopping: bool = True
    patience: int = 2


def _compile(model: Sequential, learning_rate: float) -> Sequential:
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model


def train_model(
    model: Sequential,
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: TrainingConfig,
) -> Sequential:
    """Compile and fit a Keras model, persisting checkpoints to disk."""

    x_train, y_train, x_valid, y_valid = data
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_dir / config.checkpoint_pattern)),
    ]
    if config.use_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True))
    model = _compile(model, learning_rate=config.learning_rate)
    model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks,
        verbose=1,
    )
    return model


def train_dense_classifier(
    dataset_cfg: Optional[imdb_data.ImdbDatasetConfig] = None,
    model_cfg: Optional[imdb_models.DenseModelConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
) -> Sequential:
    """High-level helper to reproduce the dense baseline experiment."""

    dataset_cfg = dataset_cfg or imdb_data.ImdbDatasetConfig()
    model_cfg = model_cfg or imdb_models.DenseModelConfig(
        vocab_size=dataset_cfg.vocab_size,
        max_length=dataset_cfg.max_length,
    )
    train_cfg = train_cfg or TrainingConfig(checkpoint_dir="artifacts/imdb_dense")
    data_splits = imdb_data.load_dataset(dataset_cfg)
    model = imdb_models.build_dense_model(model_cfg)
    return train_model(model, data_splits, train_cfg)


def train_conv_classifier(
    dataset_cfg: Optional[imdb_data.ImdbDatasetConfig] = None,
    model_cfg: Optional[imdb_models.ConvModelConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
) -> Sequential:
    """High-level helper for the CNN baseline."""

    dataset_cfg = dataset_cfg or imdb_data.ImdbDatasetConfig()
    model_cfg = model_cfg or imdb_models.ConvModelConfig(
        vocab_size=dataset_cfg.vocab_size,
        max_length=dataset_cfg.max_length,
    )
    train_cfg = train_cfg or TrainingConfig(epochs=10, checkpoint_dir="artifacts/imdb_conv")
    data_splits = imdb_data.load_dataset(dataset_cfg)
    model = imdb_models.build_conv_model(model_cfg)
    return train_model(model, data_splits, train_cfg)
