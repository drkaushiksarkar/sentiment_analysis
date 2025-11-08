"""Model builders for IMDB sentiment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    SpatialDropout1D,
)


@dataclass
class DenseModelConfig:
    """Hyperparameters for the dense baseline."""

    vocab_size: int
    embedding_dim: int = 64
    dense_units: int = 64
    dropout: float = 0.3
    max_length: int = 256


@dataclass
class ConvModelConfig:
    """Hyperparameters for the CNN baseline."""

    vocab_size: int
    embedding_dim: int = 64
    conv_filters: int = 256
    conv_kernel_size: int = 3
    dense_units: int = 256
    dropout: float = 0.2
    embed_dropout: float = 0.2
    max_length: int = 256


def build_dense_model(config: DenseModelConfig) -> Sequential:
    """Return the dense network defined in the notebook."""

    model = Sequential()
    model.add(Embedding(config.vocab_size, config.embedding_dim))
    model.add(Flatten())
    model.add(Dense(config.dense_units, activation="relu"))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.dense_units, activation="relu"))
    model.add(Dropout(config.dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model


def build_conv_model(config: ConvModelConfig) -> Sequential:
    """Return the CNN network with SpatialDropout + Conv1D + Global max pooling."""

    model = Sequential()
    model.add(Embedding(config.vocab_size, config.embedding_dim))
    model.add(SpatialDropout1D(config.embed_dropout))
    model.add(Conv1D(config.conv_filters, config.conv_kernel_size, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(config.dense_units, activation="relu"))
    model.add(Dropout(config.dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model
