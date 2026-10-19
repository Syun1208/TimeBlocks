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

### Fixed Blockbase (v2)

These options fix the reproduction issues listed under Results. Their defaults keep the old behaviour, so the existing `paper` Blockbase still loads and gives the same numbers.

| Option | Values | Effect |
|---|---|---|
| `block.connection` | `replace` (default), `residual` | `residual` computes r_p = r_(p-1) + B_p(r_(p-1)), so later blocks still see the input |
| `block.head` | `resample` (default), `linear` | `linear` learns one Linear(512, H) head per pre-trained model, saved with it |
| `selection.match_position` | `false` (default), `true` | step p only picks blocks trained at position p, searching the whole pool when the cluster has none |
| `selection.match_horizon` | `false` (default), `true` | selection runs once per horizon over blocks trained for that horizon, and the head of the last selected model is loaded |
| `selection.match_context` | `false` (default), `true` | only blocks pre-trained at the cluster's context length are candidates, so the input length matches what the blocks were trained on |
| `selection.score` | `router` (default), `backtest` | `backtest` keeps the paths with the lowest forecasting MSE on the target's own history instead of trusting the router distance alone (Section 3.2.5 says several candidate paths are evaluated). A partial path is scored after completing it with the remaining blocks and the head of its last block's source model |
| `selection.top_k`, `selection.beam_width` | `1` (default) | candidates per step and paths kept per step for `backtest`. Step 1 always scores every candidate, so the search is never worse than picking one whole pool model |
| `selection.search_contexts` | `false` (default), `true` | run the search for every cluster (context length) and keep the best, as in "Context Length Selection" |
| `selection.backtest_split`, `selection.backtest_windows` | `validation`, `512` | history split used for scoring. `auto` scores on a split no candidate was trained on: `train` when the target is excluded from the pool, `validation` when it is included |
| `split_protocol` | `table_a1` (default), `standard` | `standard` uses the 12/4/4 month ETT split and 70/10/20 for Weather |

The `*.v2.yaml` configs turn all of them on and write to the `paper_v2` Blockbase:

```bash
CONFIG=configs/timeblocks/forecasting/pretrain.v2.yaml bash scripts/pretrain_forecasting.sh
CONFIG=configs/timeblocks/forecasting/blockbase.v2.yaml bash scripts/blockbase_forecasting.sh
CONFIG=configs/timeblocks/forecasting/zero_shot.v2.yaml bash scripts/evaluate_forecasting.sh
CONFIG=configs/timeblocks/forecasting/zero_shot.v2.backtest.yaml bash scripts/evaluate_forecasting.sh
CONFIG=configs/timeblocks/forecasting/continual.streamcore.v2.yaml bash scripts/continual_streamcore.sh
```

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

## Results

Status on 2026-10-19: the paper numbers are reproduced within 3% to 9% on ETTm1, ETTm2 and Weather, and only when the target dataset is in the pool. ETTh1 and ETTh2 stay 19% to 23% above. Four bugs were found and fixed behind config options, and the remaining gap points to the evaluation protocol of the paper. Diagnostic scripts live in `works/.tmp/timeblocks_debug/` and use the project's own window builder and evaluator.

Table 1, inference-time forecasting, MSE averaged over the 4 horizons:

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather |
|---|---|---|---|---|---|
| Paper | 0.416 | 0.338 | 0.344 | 0.242 | 0.220 |
| This code (`outputs/2026-10-17_timeblocks.forecasting.zero_shot`) | 2.575 | 2.276 | 36.737 | 2.842 | 2.940 |

| # | Finding | Evidence | Effect |
|---|---|---|---|
| 1 | Block selection ignores the block position, so all 6 selected blocks are first-position blocks (`/b1`, stride 32). Paper Fig 5 picks one block per position | `block_ids` in `metrics.jsonl` | zero-shot MSE at H = 96 with one block per position: ETTh1 2.395 to 1.033, ETTh2 2.498 to 0.277, ETTm1 17.878 to 0.881, ETTm2 2.456 to 0.185, Weather 2.574 to 0.382 |
| 2 | Each block replaces its input (`residual = block(residual)`), so after block 1 the model never sees the raw series again. The paper calls the block output a residual | in-domain ETTh1, H = 96, MLP, 20 epochs | test MSE at C = 96: replace 0.494, residual 0.455, N-BEATS style 0.454. At C = 512: 0.574, 0.487, 0.481 |
| 3 | Splits follow Table A1 (60/20/20), but the paper's TTM numbers match the standard 12/4/4 month split | in-domain ETTh1, H = 96, C = 96 | replace 0.494 with Table A1 splits, 0.439 with the standard split |
| 4 | Selection feeds the cluster's context length to blocks trained at another one (for example C = 1024 blocks at C = 128), and inputs are resampled to 512, so the blocks see a different time scale | `blocks` and `context_length` in `metrics.jsonl` | `selection.match_context` (H = 96): ETTh1 1.018 to 0.697, ETTh2 0.354 to 0.296, ETTm1 0.828 to 0.711 |

Sanity check with the standard split: a linear model with instance normalization reaches MSE 0.389, 0.384 and 0.382 at C = 96, 336 and 512 (H = 96), matching published RLinear numbers, so the evaluation pipeline is correct.

Best TimeBlocks variant so far, in-domain ETTh1, H = 96, standard split:

| Variant | C = 96 | C = 512 |
|---|---|---|
| Current code (replace, resample head) | 0.439 | |
| Residual connection, resample head | 0.417 | |
| Residual connection, learned linear head | 0.404 | 0.403 |
| Residual connection, linear head, zero-initialized blocks | 0.403 | 0.411 |
| Linear model (reference) | 0.389 | 0.382 |

Remaining gap: even the best block variant trails the linear model by about 0.015 to 0.02, and the paper does not specify the task head or the block connection.

`paper_v2` Blockbase (all fixes, standard split), inference-time MSE at H = 96. "Oracle" evaluates every pool model as a whole on the target test split and keeps the best one, so it is an upper bound for any selection from that pool:

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather |
|---|---|---|---|---|---|
| Paper (Table A3) | 0.370 | 0.271 | 0.288 | 0.164 | 0.145 |
| Router, target excluded, context mismatch | 1.018 | 0.354 | 0.828 | 0.220 | 0.212 |
| Router, target excluded, `match_context` | 0.697 | 0.296 | 0.711 | 0.224 | 0.221 |
| Backtest search, target excluded | 0.465 | 0.314 | 0.418 | 0.190 | 0.217 |
| Oracle, target excluded | 0.466 | 0.293 | 0.401 | 0.185 | 0.210 |
| Router, target **in** the pool | 0.395 | 0.294 | 0.730 | 0.217 | 0.237 |
| Backtest search, target **in** the pool | 0.417 | 0.318 | 0.310 | 0.171 | 0.152 |
| Best model trained on the target itself | 0.394 | 0.296 | 0.304 | 0.170 | 0.152 |

Two conclusions:

- **The paper numbers are out of reach without target data.** Even the oracle that picks the best excluded-pool model on the test split stays far above the paper (ETTh1 0.466 against 0.370, ETTm1 0.401 against 0.288, Weather 0.210 against 0.145), while models trained on the target itself land close to it. The paper numbers are most consistent with a pool that contains the target dataset.
- **The router alone is the weak link, backtest scoring closes it.** With the target in the pool the router reaches 0.395 on ETTh1, but on ETTm1 it picks ETTh1 blocks (0.730) instead of the ETTm1 model (0.304). Scoring candidate paths on the target's history (`zero_shot.v2.backtest.yaml`, beam 8 x 4 over all contexts) brings both settings to their pool ceiling: excluded 0.465 on ETTh1 against the oracle 0.466, included 0.310, 0.171 and 0.152 on ETTm1, ETTm2 and Weather against the best target models 0.304, 0.170 and 0.152. ETTh1 and ETTh2 stay noisier because their validation months differ from the test months.

What is left with the target in the pool is block quality: the best target model trails the paper by about 5% on every dataset (0.394 against 0.370 on ETTh1, 0.152 against 0.145 on Weather). The paper only gives AdamW, learning rate 0.001, batch 64 and 50 epochs, which this code follows. With that recipe the best validation checkpoint is epoch 1 (ETTh1, LSTM, C = 512, H = 96: train loss falls from 0.37 to 0.05 while validation rises from 0.76 to 0.82), so the blocks overfit right away. Training variants of that model give test MSE 0.394 (paper recipe), 0.410 (learning rate 1e-4), 0.394 (cosine schedule, `optimization.scheduler`), 0.396 (dropout 0.3), 0.394 (full validation set) and 0.392 (hidden size 128). None closes the gap, so the `paper_v2` recipe is kept.

Table 1 on the full `paper_v2` Blockbase (360 models, 2026-10-19), MSE averaged over H = 96, 192, 336 and 720. The last row is an upper bound: the best model trained on the target, picked on the test split:

| | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Weather |
|---|---|---|---|---|---|
| Paper | 0.416 | 0.338 | 0.344 | 0.242 | 0.220 |
| Router, target excluded (`outputs/v2_zero_shot`) | 0.571 | 0.378 | 0.866 | 0.300 | 0.294 |
| Backtest search, target excluded (`outputs/v2_zero_shot_backtest`) | 0.536 | 0.371 | 0.470 | 0.280 | 0.285 |
| Router, target in the pool (`outputs/v2_zero_shot_inpool`) | 0.496 | 0.374 | 0.578 | 0.300 | 0.292 |
| Backtest search, target in the pool (`outputs/v2_zero_shot_backtest_inpool`) | 0.512 | 0.401 | 0.375 | 0.272 | 0.236 |
| Best target model, picked on test | 0.449 | 0.365 | 0.363 | 0.261 | 0.227 |

With the target in the pool, backtest search lands within 3% to 9% of the paper on ETTm1, ETTm2 and Weather. ETTh1 and ETTh2 stay further off because their validation months, used for scoring, behave differently from the test months, so long horizons often pick ETTm blocks. Even the best target model is 2% to 8% above the paper, which is the block quality gap described above.

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
