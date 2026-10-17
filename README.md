# TimeBlocks

## Overview

A reimplementation of TimeBlocks, which builds small time-series models at inference time by routing through a pool of pre-trained, interchangeable blocks (the Blockbase), and keeps them calibrated on a stream with the StreamCore coreset. Every design choice the paper leaves open is listed in [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

## Highlights

- Blocks (MLP, LSTM, Attention) with multi-scale sampling `2^(P-p)` and output standardization to a common vector size, pre-trained by Algorithm A1.
- First-layer fingerprints, K-Means with one cluster per context length, and a Conv1d router trained by Algorithm A2.
- Inference-time model building by Algorithm A3 for forecasting, imputation, outlier detection and UCR classification.
- StreamCore streaming coreset with O(|S|) updates and continual calibration of the last block, plus the random 5% buffer used for baselines.
- Twelve baselines behind one adapter interface: Autoformer, PatchTST, TimesNet, TimeMixer, AnomalyTransformer, DCdetector, InceptionTime, ResNet, TTM, Moment, Lag-Llama, SymTime.

## Installation

Python 3.10, PyTorch 2.3.1 with CUDA 12.1 (the server driver supports CUDA 12.2).

```bash
conda create -n timeblocks python=3.10 -y
conda activate timeblocks
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-deps granite-tsfm==0.3.3 momentfm==0.1.4 reformer_pytorch==1.4.4 axial-positional-embedding==0.2.1 product-key-memory==0.1.10 local-attention==1.9.0
bash scripts/setup_third_party.sh
```

`.env` in the parent folder (`time-series-forecasting/.env`, keys in `.env.example`):

| Key | Value |
|---|---|
| `TIMEBLOCKS_STORAGE_ROOT` | `/mnt/HDD6/longpm/timeblocks`, holds `data/`, `logs/`, `weights/`, `outputs/`, `hf_cache/`, `third_party/` |
| `HF_TOKEN` | optional |

## Dataset Preparation

```bash
bash scripts/download_data.sh
```

Sources: the THUML mirror on Hugging Face (`thuml/Time-Series-Library`) for ETT, Weather, SMD, MSL, SMAP and SWaT, and timeseriesclassification.com for the 128 UCR datasets. Splits follow Table A1.

```
${TIMEBLOCKS_STORAGE_ROOT}/data/
├── forecasting/            ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, Weather.csv
├── anomaly_detection/
│   ├── SMD/                SMD_train.npy, SMD_test.npy, SMD_test_label.npy
│   ├── MSL/                MSL_train.npy, MSL_test.npy, MSL_test_label.npy
│   ├── SMAP/               SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
│   └── SWaT/               swat_train2.csv, swat2.csv (last column is the label)
└── classification/UCR/<Name>/  <Name>_TRAIN.ts, <Name>_TEST.ts
```

## Training

Run the steps of one task in order: pretrain, then blockbase, then evaluate. Every GPU script starts with `GPU_IDS=3` or `GPU_IDS=3,4`, using the ids printed by `nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). With several ids the work is split across them: pretraining jobs, target datasets, baseline models or values of J.

| Step | Command | Config | Output |
|---|---|---|---|
| Blockbase pre-training (Alg A1) | `bash scripts/pretrain_forecasting.sh` | `configs/timeblocks/forecasting/pretrain.yaml` | `weights/blockbase/paper/<model_id>/` |
| Router and clusters (Alg A2, Sec 3.2.3) | `bash scripts/blockbase_forecasting.sh` | `configs/timeblocks/forecasting/blockbase.yaml` | `weights/blockbase_index/paper/forecasting/without_<target>/` |
| Same for the other tasks | `bash scripts/pretrain_<task>.sh`, `bash scripts/blockbase_<task>.sh` | `configs/timeblocks/<task>/` | as above |
| Baselines | `bash scripts/baselines_<task>.sh` | `configs/baselines/<model>/<task>.yaml` | `outputs/<date>_baselines.<model>.<task>/` |
| Continual baselines (Table 2) | `bash scripts/baselines_forecasting_continual.sh` | `configs/baselines/<model>/forecasting.continual.yaml` | `outputs/<date>_baselines.<model>.forecasting.continual/` |

`<task>` is one of `forecasting`, `imputation`, `anomaly_detection`, `classification`. Extra arguments go to `main.py`, for example `bash scripts/pretrain_forecasting.sh --set optimization.samples_per_epoch=null`.

Trained weights are saved once and loaded again on the next run, so a rerun of a test skips training:

| Weights | Saved at | Reused when |
|---|---|---|
| Pre-trained TimeBlocks models | `weights/blockbase/<name>/<model_id>/` | the same `model_id` exists (`skip_existing: true`) |
| Baselines trained on the source datasets | `weights/baselines/<model>/<digest>/<dataset>_h<horizon>/` | model, sources, hyperparameters, optimization and seed are unchanged |
| Calibrated models after the stream (Table 2) | `weights/calibrated/<model>/<digest>/<dataset>_h<horizon>/` | the same blocks or baseline weights and the same `stream` settings |

`<digest>` is a hash of the settings, written next to the weights in `identity.json`, so a changed config gets a new folder instead of loading stale weights. Only trainable parameters and buffers are saved, frozen parts (Blockbase blocks, the Moment encoder) are rebuilt from their source. The calibrated rows carry `calibration_loaded`. Delete a folder to train it again.

## Evaluation / Inference

| Experiment | Command | Output |
|---|---|---|
| Table 1, forecasting inference-time | `bash scripts/evaluate_forecasting.sh` | `outputs/<date>_timeblocks.forecasting.zero_shot/` |
| Table 2, continual with StreamCore | `bash scripts/continual_streamcore.sh` | `outputs/<date>_timeblocks.forecasting.continual.streamcore/` |
| Table 2, TimeBlocks + Random | `bash scripts/continual_random.sh` | `outputs/<date>_timeblocks.forecasting.continual.random/` |
| Table 3, imputation | `bash scripts/evaluate_imputation.sh` | `outputs/<date>_timeblocks.imputation.zero_shot/` |
| Table 4, outlier detection | `bash scripts/evaluate_anomaly_detection.sh` | `outputs/<date>_timeblocks.anomaly_detection.zero_shot/` |
| Figure 7, classification | `bash scripts/evaluate_classification.sh` | `outputs/<date>_timeblocks.classification.zero_shot/` |
| Figure A1, number of blocks J | `bash scripts/ablation_num_blocks.sh` | `outputs/ablation_num_blocks/j<J>/` |

Each run directory has `config.yaml` (the resolved config), `metrics.jsonl` (one line per dataset and horizon or mask ratio, with `mse`, `mae`, `f1`, `accuracy`, `size_mb`, `seconds_per_sample`, `selection_seconds`) and `summary.csv`.

Tables and figures take run directories relative to `outputs/`:

```bash
bash scripts/report_forecasting.sh 2026-10-15_timeblocks.forecasting.zero_shot 2026-10-15_baselines.patchtst.forecasting
bash scripts/report_classification.sh <run dirs>
```

Reports land in `outputs/report_<name>/`: `table.csv`, `table.md`, `ranking.json`, and `figures/*.pdf|png`.

## Citation

```bibtex
@misc{campos2026timeblocksfoundationalcontinualtimeseries,
      title={TimeBlocks: Foundational and Continual Time-Series Blockbase -- Extended Version},
      author={David Campos and Bin Yang and Tung Kieu and Lei Chen and Chenjuan Guo and Christian S. Jensen},
      year={2026},
      eprint={2606.02142},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.02142},
}
```

## Acknowledgements / License

- [TimeBlocks: Foundational and Continual Time-Series Blockbase, Extended Version](https://arxiv.org/abs/2606.02142), Campos et al., arXiv 2026. No official code was released, this is an independent reimplementation.
- Baselines come from [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer), [DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector), [lag-llama](https://github.com/time-series-foundation-models/lag-llama), [SymTime](https://github.com/wwhenxuan/SymTime), [granite-tsfm](https://github.com/ibm-granite/granite-tsfm) (TTM) and [momentfm](https://github.com/moment-timeseries-foundation-model/moment).
- Datasets: ETT, Weather, SMD, MSL, SMAP and SWaT through the THUML Hugging Face mirror, and the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
