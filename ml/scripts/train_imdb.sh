#!/usr/bin/env bash
set -euo pipefail

python scripts/train_imdb.py --model dense --epochs 1 --checkpoint-dir artifacts/imdb_docker
