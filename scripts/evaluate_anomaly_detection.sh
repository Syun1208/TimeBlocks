#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=5,7
CONFIG=configs/timeblocks/anomaly_detection/zero_shot.yaml

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
for SHARD in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[SHARD]} python main.py timeblocks evaluate \
        --config "${CONFIG}" \
        --shard-index "${SHARD}" \
        --num-shards "${#GPUS[@]}" \
        "$@" &
done
wait
