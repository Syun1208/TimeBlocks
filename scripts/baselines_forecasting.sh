#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=3,4
EXPERIMENT=forecasting
MODELS=(autoformer patchtst timesnet timemixer ttm moment lag_llama symtime)

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
for INDEX in "${!MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[INDEX % ${#GPUS[@]}]} python main.py baseline run \
        --config "configs/baselines/${MODELS[INDEX]}/${EXPERIMENT}.yaml" \
        "$@" &
done
wait
