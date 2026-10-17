#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py report \
    --config configs/report/forecasting.continual.yaml \
    --inputs "$@"
