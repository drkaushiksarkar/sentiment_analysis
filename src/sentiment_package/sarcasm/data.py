"""Sarcasm headline dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/"
    "master/Sarcasm_Headlines_Dataset.json"
)


@dataclass
class SarcasmDatasetConfig:
    """Configuration for sarcasm data preparation."""

    dataset_url: str = DEFAULT_DATASET_URL
    test_size: float = 0.3
    random_state: int = 42
    vocab_size: int = 10000
    embedding_dim: int = 100
    max_length: int = 32
    padding_type: str = "post"
    trunc_type: str = "post"
    oov_token: str = "<oov>"


def load_dataframe(config: SarcasmDatasetConfig) -> pd.DataFrame:
    """Load the sarcasm dataset and add helper columns."""

    df = pd.read_json(config.dataset_url, lines=True)
    df["sentence_length"] = df["headline"].str.split().apply(len)
    return df


def train_test_split_texts(df: pd.DataFrame, config: SarcasmDatasetConfig) -> Tuple[np.ndarray, ...]:
    """Split headline text and labels into train/test splits."""

    x = df["headline"].values
    y = df["is_sarcastic"].values
    return train_test_split(x, y, test_size=config.test_size, random_state=config.random_state)


def tokenize_texts(
    x_train: np.ndarray,
    x_test: np.ndarray,
    config: SarcasmDatasetConfig,
) -> Tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Tokenize and pad sarcasm headlines."""

    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token=config.oov_token)
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    train_padded = pad_sequences(
        train_sequences,
        maxlen=config.max_length,
        padding=config.padding_type,
        truncating=config.trunc_type,
    )
    test_padded = pad_sequences(
        test_sequences,
        maxlen=config.max_length,
        padding=config.padding_type,
        truncating=config.trunc_type,
    )
    return train_padded, test_padded, tokenizer
