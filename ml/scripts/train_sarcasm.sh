#!/usr/bin/env bash
set -euo pipefail

python scripts/train_sarcasm.py --model dense --epochs 1 --checkpoint-dir artifacts/sarcasm_docker
