"""CLI for training sarcasm detection models."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_package.sarcasm import data as sarcasm_data
from sentiment_package.sarcasm import models as sarcasm_models
from sentiment_package.sarcasm import train as sarcasm_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sarcasm detection models")
    parser.add_argument("--model", choices=["dense", "conv", "bilstm"], default="dense")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--glove-path", type=Path, default=None, help="Optional path to a GloVe .txt file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_cfg = sarcasm_data.SarcasmDatasetConfig(
        max_length=args.max_length,
        vocab_size=args.vocab_size,
    )

    train_cfg = sarcasm_train.TrainingConfig()
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir

    model_cfg = None
    if args.model == "dense":
        model_cfg = sarcasm_models.DenseSarcasmConfig(
            vocab_size=dataset_cfg.vocab_size,
            embedding_dim=dataset_cfg.embedding_dim,
            max_length=dataset_cfg.max_length,
            trainable_embeddings=args.glove_path is None,
        )
        sarcasm_train.train_dense_classifier(dataset_cfg, model_cfg, train_cfg, glove_path=args.glove_path)
    elif args.model == "conv":
        model_cfg = sarcasm_models.ConvSarcasmConfig(
            vocab_size=dataset_cfg.vocab_size,
            embedding_dim=dataset_cfg.embedding_dim,
            max_length=dataset_cfg.max_length,
            trainable_embeddings=args.glove_path is None,
        )
        sarcasm_train.train_conv_classifier(dataset_cfg, model_cfg, train_cfg, glove_path=args.glove_path)
    else:
        model_cfg = sarcasm_models.BiLSTMSarcasmConfig(
            vocab_size=dataset_cfg.vocab_size,
            embedding_dim=dataset_cfg.embedding_dim,
            max_length=dataset_cfg.max_length,
            trainable_embeddings=args.glove_path is None,
        )
        sarcasm_train.train_bilstm_classifier(dataset_cfg, model_cfg, train_cfg, glove_path=args.glove_path)


if __name__ == "__main__":
    main()
