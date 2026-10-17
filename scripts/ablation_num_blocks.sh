#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=5,7
CONFIG=configs/timeblocks/forecasting/zero_shot.yaml
TARGETS="[ETTh2, ETTm1]"
HORIZONS="[96]"
BLOCK_COUNTS=(1 2 3 4 5 6)

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
for INDEX in "${!BLOCK_COUNTS[@]}"; do
    BLOCKS=${BLOCK_COUNTS[INDEX]}
    CUDA_VISIBLE_DEVICES=${GPUS[INDEX % ${#GPUS[@]}]} python main.py timeblocks evaluate \
        --config "${CONFIG}" \
        --set "target_datasets=${TARGETS}" \
        --set "horizons=${HORIZONS}" \
        --set "selection.num_blocks=${BLOCKS}" \
        --output-dir "ablation_num_blocks/j${BLOCKS}" \
        "$@" &
done
wait
