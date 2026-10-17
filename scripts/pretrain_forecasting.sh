#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=5,7
WORKERS_PER_GPU=4
CONFIG=configs/timeblocks/forecasting/pretrain.yaml

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
for SHARD in $(seq 0 $(( NUM_SHARDS - 1 ))); do
    CUDA_VISIBLE_DEVICES=${GPUS[SHARD % ${#GPUS[@]}]} python main.py timeblocks pretrain \
        --config "${CONFIG}" \
        --shard-index "${SHARD}" \
        --num-shards "${NUM_SHARDS}" \
        --set optimization.num_workers=2 \
        "$@" &
done
wait
