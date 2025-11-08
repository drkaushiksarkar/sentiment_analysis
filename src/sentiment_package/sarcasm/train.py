"""Training helpers for sarcasm detection models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from . import data as sarcasm_data
from . import glove as glove_utils
from . import models as sarcasm_models


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    checkpoint_dir: Path | str = Path("artifacts/sarcasm")
    checkpoint_pattern: str = "weights.{epoch:02d}.keras"
    patience: int = 5


def _compile(model: Sequential, learning_rate: float) -> Sequential:
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model


def _embedding_matrix_from_path(
    glove_path: Optional[Path],
    tokenizer_word_index: dict,
    config: sarcasm_data.SarcasmDatasetConfig,
) -> Optional[np.ndarray]:
    if glove_path is None:
        return None
    glove_vectors = glove_utils.load_glove_vectors(Path(glove_path))
    return glove_utils.build_embedding_matrix(
        tokenizer_word_index,
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        glove_vectors=glove_vectors,
    )


def prepare_dataset(
    config: sarcasm_data.SarcasmDatasetConfig,
    glove_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, Optional[np.ndarray]]:
    """Load data, tokenize, and optionally derive an embedding matrix."""

    df = sarcasm_data.load_dataframe(config)
    x_train, x_test, y_train, y_test = sarcasm_data.train_test_split_texts(df, config)
    train_padded, test_padded, tokenizer = sarcasm_data.tokenize_texts(x_train, x_test, config)
    embedding_matrix = _embedding_matrix_from_path(glove_path, tokenizer.word_index, config)
    return train_padded, test_padded, y_train, y_test, tokenizer.word_index, embedding_matrix


def train_model(
    model: Sequential,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    val_inputs: np.ndarray,
    val_labels: np.ndarray,
    train_cfg: TrainingConfig,
) -> Sequential:
    """Generic fit function with checkpointing + early stopping."""

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_dir / train_cfg.checkpoint_pattern)),
        EarlyStopping(monitor="val_loss", patience=train_cfg.patience, restore_best_weights=True),
    ]
    model = _compile(model, learning_rate=train_cfg.learning_rate)
    model.fit(
        train_inputs,
        train_labels,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        validation_data=(val_inputs, val_labels),
        callbacks=callbacks,
        verbose=1,
    )
    return model


def train_dense_classifier(
    dataset_cfg: Optional[sarcasm_data.SarcasmDatasetConfig] = None,
    model_cfg: Optional[sarcasm_models.DenseSarcasmConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    glove_path: Optional[Path] = None,
) -> Sequential:
    dataset_cfg = dataset_cfg or sarcasm_data.SarcasmDatasetConfig()
    train_cfg = train_cfg or TrainingConfig(epochs=100, checkpoint_dir="artifacts/sarcasm_dense")
    train_inputs, val_inputs, train_labels, val_labels, _, embedding_matrix = prepare_dataset(dataset_cfg, glove_path)
    model_cfg = model_cfg or sarcasm_models.DenseSarcasmConfig(
        vocab_size=dataset_cfg.vocab_size,
        embedding_dim=dataset_cfg.embedding_dim,
        max_length=dataset_cfg.max_length,
        trainable_embeddings=glove_path is None,
    )
    model = sarcasm_models.build_dense_model(model_cfg, embedding_matrix)
    return train_model(model, train_inputs, train_labels, val_inputs, val_labels, train_cfg)


def train_conv_classifier(
    dataset_cfg: Optional[sarcasm_data.SarcasmDatasetConfig] = None,
    model_cfg: Optional[sarcasm_models.ConvSarcasmConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    glove_path: Optional[Path] = None,
) -> Sequential:
    dataset_cfg = dataset_cfg or sarcasm_data.SarcasmDatasetConfig()
    train_cfg = train_cfg or TrainingConfig(checkpoint_dir="artifacts/sarcasm_conv")
    train_inputs, val_inputs, train_labels, val_labels, _, embedding_matrix = prepare_dataset(dataset_cfg, glove_path)
    model_cfg = model_cfg or sarcasm_models.ConvSarcasmConfig(
        vocab_size=dataset_cfg.vocab_size,
        embedding_dim=dataset_cfg.embedding_dim,
        max_length=dataset_cfg.max_length,
        trainable_embeddings=glove_path is None,
    )
    model = sarcasm_models.build_conv_model(model_cfg, embedding_matrix)
    return train_model(model, train_inputs, train_labels, val_inputs, val_labels, train_cfg)


def train_bilstm_classifier(
    dataset_cfg: Optional[sarcasm_data.SarcasmDatasetConfig] = None,
    model_cfg: Optional[sarcasm_models.BiLSTMSarcasmConfig] = None,
    train_cfg: Optional[TrainingConfig] = None,
    glove_path: Optional[Path] = None,
) -> Sequential:
    dataset_cfg = dataset_cfg or sarcasm_data.SarcasmDatasetConfig()
    train_cfg = train_cfg or TrainingConfig(checkpoint_dir="artifacts/sarcasm_bilstm")
    train_inputs, val_inputs, train_labels, val_labels, _, embedding_matrix = prepare_dataset(dataset_cfg, glove_path)
    model_cfg = model_cfg or sarcasm_models.BiLSTMSarcasmConfig(
        vocab_size=dataset_cfg.vocab_size,
        embedding_dim=dataset_cfg.embedding_dim,
        max_length=dataset_cfg.max_length,
        trainable_embeddings=glove_path is None,
    )
    model = sarcasm_models.build_bilstm_model(model_cfg, embedding_matrix)
    return train_model(model, train_inputs, train_labels, val_inputs, val_labels, train_cfg)
