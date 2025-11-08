"""Keras models for sarcasm classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import layers


@dataclass
class BaseSarcasmModelConfig:
    vocab_size: int
    embedding_dim: int = 100
    max_length: int = 32
    trainable_embeddings: bool = True


@dataclass
class DenseSarcasmConfig(BaseSarcasmModelConfig):
    dense_units: int = 32
    aux_units: int = 10
    dropout: float = 0.5


@dataclass
class ConvSarcasmConfig(BaseSarcasmModelConfig):
    conv_filters: int = 128
    conv_kernel_size: int = 5
    dense_units: int = 64
    dropout: float = 0.3
    spatial_dropout: float = 0.2


@dataclass
class BiLSTMSarcasmConfig(BaseSarcasmModelConfig):
    lstm_units: int = 32
    dense_units: int = 6


def _embedding_layer(config: BaseSarcasmModelConfig, embedding_matrix: Optional[np.ndarray] = None) -> layers.Embedding:
    layer = layers.Embedding(config.vocab_size, config.embedding_dim)
    if embedding_matrix is not None:
        layer.build((None,))
        layer.set_weights([embedding_matrix])
        layer.trainable = config.trainable_embeddings
    return layer


def build_dense_model(
    config: DenseSarcasmConfig,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Sequential:
    """Dense network that mirrors the notebook baseline."""

    model = Sequential(
        [
            _embedding_layer(config, embedding_matrix),
            layers.Flatten(),
            layers.Dense(config.dense_units, activation="relu"),
            layers.Dropout(config.dropout),
            layers.Dense(config.aux_units, activation="relu"),
            layers.Dropout(config.dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def build_conv_model(
    config: ConvSarcasmConfig,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Sequential:
    """CNN inspired by the notebook's Glove + Conv1D experiment."""

    model = Sequential(
        [
            _embedding_layer(config, embedding_matrix),
            layers.SpatialDropout1D(config.spatial_dropout),
            layers.Conv1D(config.conv_filters, config.conv_kernel_size, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(config.dense_units, activation="relu"),
            layers.Dropout(config.dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def build_bilstm_model(
    config: BiLSTMSarcasmConfig,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Sequential:
    """Bidirectional LSTM equivalent."""

    model = Sequential(
        [
            _embedding_layer(config, embedding_matrix),
            layers.Bidirectional(layers.LSTM(config.lstm_units)),
            layers.Dense(config.dense_units, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model
