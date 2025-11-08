"""Utilities for working with pre-trained GloVe embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_glove_vectors(glove_path: Path) -> Dict[str, np.ndarray]:
    """Read GloVe embeddings from disk."""

    vectors: Dict[str, np.ndarray] = {}
    with Path(glove_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype="float32")
            vectors[word] = coeffs
    return vectors


def build_embedding_matrix(
    word_index: Dict[str, int],
    vocab_size: int,
    embedding_dim: int,
    glove_vectors: Dict[str, np.ndarray],
) -> np.ndarray:
    """Create the embedding matrix aligned with the tokenizer indices."""

    matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        vector = glove_vectors.get(word)
        if vector is not None:
            matrix[idx] = vector
    return matrix
