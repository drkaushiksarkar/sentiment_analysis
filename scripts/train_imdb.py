"""CLI for training IMDB sentiment models."""

from __future__ import annotations

import argparse

from sentiment_package.imdb import data as imdb_data
from sentiment_package.imdb import models as imdb_models
from sentiment_package.imdb import train as imdb_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMDB sentiment models")
    parser.add_argument("--model", choices=["dense", "conv"], default="dense")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_cfg = imdb_data.ImdbDatasetConfig(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
    )

    train_cfg = imdb_train.TrainingConfig()
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir

    if args.model == "dense":
        model_cfg = imdb_models.DenseModelConfig(vocab_size=dataset_cfg.vocab_size, max_length=dataset_cfg.max_length)
        imdb_train.train_dense_classifier(dataset_cfg, model_cfg, train_cfg)
    else:
        model_cfg = imdb_models.ConvModelConfig(vocab_size=dataset_cfg.vocab_size, max_length=dataset_cfg.max_length)
        imdb_train.train_conv_classifier(dataset_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
