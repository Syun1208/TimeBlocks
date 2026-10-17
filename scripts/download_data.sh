#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py data download \
    --config configs/data/download.yaml \
    "$@"
