"""TensorFlow-backed sentiment service using the sentiment_package models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend_app.schemas import SentimentResponse
from sentiment_package.imdb import data as imdb_data
from sentiment_package.imdb import models as imdb_models

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


class SentimentService:
    """Loads the IMDB dense classifier and exposes an inference-friendly interface."""

    def __init__(self, weights_path: Path, max_length: int = 256) -> None:
        self.dataset_cfg = imdb_data.ImdbDatasetConfig(max_length=max_length)
        self.model_cfg = imdb_models.DenseModelConfig(
            vocab_size=self.dataset_cfg.vocab_size,
            max_length=self.dataset_cfg.max_length,
        )
        self.model = imdb_models.build_dense_model(self.model_cfg)
        self._load_weights(weights_path)
        _, self.word_index = imdb_data.build_word_mappings()
        self.unknown_token = self.word_index.get("UNK", 2)

    def _load_weights(self, weights_path: Path) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        # Build the model to materialize weights shapes before loading.
        self.model.build((None, self.dataset_cfg.max_length))
        self.model.load_weights(str(weights_path))

    def _tokenize(self, text: str) -> List[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    def _encode(self, text: str) -> Tuple[np.ndarray, int]:
        tokens = self._tokenize(text)
        indices = [self.word_index.get(token, self.unknown_token) for token in tokens]
        if not indices:
            indices = [self.unknown_token]
        padded = pad_sequences(
            [indices],
            maxlen=self.dataset_cfg.max_length,
            padding=self.dataset_cfg.pad_type,
            truncating=self.dataset_cfg.trunc_type,
            value=0,
        )
        return padded, len(tokens)

    def predict(self, text: str) -> SentimentResponse:
        encoded, token_count = self._encode(text)
        probability = float(self.model.predict(encoded, verbose=0)[0][0])
        signed_score = (probability - 0.5) * 2  # scale to [-1, 1]
        label = "neutral"
        if signed_score >= 0.1:
            label = "positive"
        elif signed_score <= -0.1:
            label = "negative"
        confidence = min(1.0, abs(signed_score))
        return SentimentResponse(
            label=label,
            score=float(round(signed_score, 3)),
            confidence=float(round(confidence, 3)),
            tokens_analyzed=token_count,
        )
