"""Data loading helpers for the IMDB sentiment dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


@dataclass
class ImdbDatasetConfig:
    """Configuration for fetching and preparing IMDB reviews."""

    vocab_size: int = 10000
    skip_top: int = 0
    max_length: int = 256
    pad_type: str = "post"
    trunc_type: str = "post"


def load_dataset(config: ImdbDatasetConfig) -> Tuple[np.ndarray, ...]:
    """Load IMDB data and return padded train/validation splits."""

    (x_train, y_train), (x_valid, y_valid) = keras.datasets.imdb.load_data(
        num_words=config.vocab_size,
        skip_top=config.skip_top,
    )
    x_train = pad_sequences(
        x_train,
        maxlen=config.max_length,
        padding=config.pad_type,
        truncating=config.trunc_type,
        value=0,
    )
    x_valid = pad_sequences(
        x_valid,
        maxlen=config.max_length,
        padding=config.pad_type,
        truncating=config.trunc_type,
        value=0,
    )
    return x_train, y_train, x_valid, y_valid


def build_word_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return token-to-word and word-to-token mappings identical to the notebook."""

    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    word_index["UNK"] = 2
    index_word = {v: k for k, v in word_index.items()}
    return index_word, word_index


def decode_review(tokens: np.ndarray, index_word: Dict[int, str]) -> str:
    """Turn a padded token sequence back into space-separated text."""

    return " ".join(index_word.get(int(token), "?") for token in tokens if token != 0)
