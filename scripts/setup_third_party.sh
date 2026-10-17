#!/usr/bin/env bash
set -euo pipefail

REPOSITORIES=(
    "thuml/Time-Series-Library 4e938a1"
    "time-series-foundation-models/lag-llama df7531a"
    "thuml/Anomaly-Transformer b0ee470"
    "DAMO-DI-ML/KDD2023-DCdetector 9d107ca"
    "wwhenxuan/SymTime 24a9386"
)
SYMTIME_CHECKPOINT=https://drive.google.com/uc?id=1SXixHUkBWX5tGTVANFaZDovnA5P77QIo

cd "$(dirname "$0")/.."
THIRD_PARTY=$(python -c "from pathlib import Path; from src.utils.env import resolve_storage; print(resolve_storage(start=Path.cwd()).third_party)")
mkdir -p "${THIRD_PARTY}/symtime_drive"
for ENTRY in "${REPOSITORIES[@]}"; do
    read -r REPOSITORY COMMIT <<< "${ENTRY}"
    TARGET="${THIRD_PARTY}/$(basename "${REPOSITORY}")"
    [ -d "${TARGET}" ] || git clone -q "https://github.com/${REPOSITORY}.git" "${TARGET}"
    git -C "${TARGET}" checkout -q "${COMMIT}" || echo "Keeping current commit of ${REPOSITORY}"
done
[ -f "${THIRD_PARTY}/symtime_drive/finetuning.pth" ] || gdown "${SYMTIME_CHECKPOINT}" -O "${THIRD_PARTY}/symtime_drive/finetuning.pth"
