# Reproduction Notes

Paper: [TimeBlocks: Foundational and Continual Time-Series Blockbase, Extended Version](https://arxiv.org/abs/2606.02142), Campos et al., 2026.
Official code: none found (A.5.4 says "available as supplementary material", no public repository on 2026-10-15).

Every choice below is tagged:

| Tag | Meaning |
|---|---|
| SPECIFIED | Stated in the paper, with the section |
| PARTIAL | Mentioned but ambiguous, our reading is given |
| UNSPECIFIED | Not in the paper, our default plus alternatives |
| DEVIATION | We knowingly differ from the paper, with the reason |

---

## Scope

| Component | Status | Code |
|---|---|---|
| Blocks with multi-scale sampling and output standardization (Sec 3.2.2, Fig 4) | implemented | `src/models/timeblocks/block.py` |
| Block pre-training, Algorithm A1 | implemented | `src/services/pretraining.py` |
| Fingerprints and K-Means clustering (Sec 3.2.3) | implemented | `src/services/clustering.py` |
| Router, Algorithm A2 | implemented | `src/models/timeblocks/router.py`, `src/services/router_training.py` |
| Block selection, Algorithm A3 | implemented | `src/services/block_selection.py` |
| StreamCore and continual calibration (Sec 3.3, A.4) | implemented | `src/services/coresets.py`, `src/services/continual.py` |
| Random 5% buffer baseline (Table 2) | implemented | `ReservoirBuffer` in `src/services/coresets.py` |
| Tasks: forecasting, imputation, outlier detection, classification | implemented | `src/models/tasks.py` |
| Metrics MSE, MAE, F1, accuracy, Friedman plus Wilcoxon-Holm ranking (A.5.2) | implemented | `src/utils/metrics.py`, `src/utils/ranking.py` |
| Baselines (A.5.3) | adapters, see the baseline table | `src/models/baselines/` |
| Multiple independent selection paths evaluated in parallel (Sec 3.2.5, last paragraph) | not implemented | single greedy path |
| DER++ and Camel coresets (Table A5) | not implemented | out of scope |
| Model size, inference time, building time (Sec 4.2.7, 4.2.8) | logged per run | `size_mb`, `seconds_per_sample`, `selection_seconds` |

---

## Architecture

| Item | Status | Paper | Our choice | Alternatives |
|---|---|---|---|---|
| Block input and output | PARTIAL | Sec 3.2.2: output standardization is "a fully-connected layer that standardizes the output to a common vector size" | every block maps a residual of length `common_length` (512) to length 512 | per-model vector size |
| Raw input before the first block | PARTIAL | Alg A3 line 7: `r0 = Standardization(x_st)` before any block is chosen | parameter-free: instance normalization (mean, std per window) then linear resampling from C to 512 | learned projection per context length (cannot run before a block is chosen) |
| Scaling inside a block | PARTIAL | Fig 4 and Alg A1 line 5: block p samples "every 2^(P-p)" | block keeps every 2^(P-p)-th value of the incoming residual, aligned to the most recent value | average pooling |
| Patching (Sec 3.2.2, [47]) | PARTIAL | "enhanced with a patching structure" | non-overlapping patches of length 16 | PatchTST overlap with stride 8 |
| Fingerprint | SPECIFIED | Sec 3.2.3: first layer b1, In x V x Out, homogeneous over all blocks | first layer is the patch embedding `Linear(16, 64)`, fingerprint size 1024 | |
| Variables V in the fingerprint | DEVIATION | Sec 3.2.3 lists V in In x V x Out | V = 1 because every series is processed channel-independently, so fingerprints from ETT (7 variables) and Weather (21) stay comparable | per-variable weights |
| Layers per block s | UNSPECIFIED | | 3 (first layer plus 2 body layers) | 2, 4 |
| Hidden size | UNSPECIFIED | | 64 | 32, 128 |
| MLP body | UNSPECIFIED | | token-wise pre-norm residual `LayerNorm, Linear, GELU, Dropout` | Mixer token mixing |
| LSTM body | UNSPECIFIED | | `nn.LSTM(64, 64, layers=s-1)` | bidirectional |
| Attention body | UNSPECIFIED | | learned positional embedding plus pre-norm `TransformerEncoderLayer`, 4 heads, FFN 2x | sinusoidal positions |
| Activations, dropout, init | UNSPECIFIED | | GELU, dropout 0.1, PyTorch default init | ReLU, Xavier |
| Task head | PARTIAL | the paper only says the last block output is the model output | forecasting and reconstruction: parameter-free resampling of the last residual from 512 to F (or C) then de-normalization, so any block can be last. Classification: `Linear(512, K)` | learned linear head per horizon |
| Router | SPECIFIED | Sec 3.2.4: "two 1D convolutional layers followed by a fully-connected layer" | `Conv1d(1,32,k5,s2)`, GELU, `Conv1d(32,32,k5,s2)`, GELU, adaptive average pool to 32, `Linear(1024, 1024)` | channel counts, kernels and pooling are UNSPECIFIED |
| Router loss | UNSPECIFIED | Alg A2 line 7: `Loss(Router(r_{p-1}), b1_p)` | MSE (config `router.loss: cosine` also available) | cosine distance |
| Number of clusters K | SPECIFIED | Sec 3.2.3: K = number of context lengths | K = number of distinct context lengths in the pool | |
| Cluster representative | SPECIFIED | "employs the block that is closest to the cluster center" | yes, and the cluster context length is that block's context length | majority context length |
| K-Means features | UNSPECIFIED | | L2-normalized fingerprints, `n_init=10` | raw fingerprints |
| Selection distance | SPECIFIED | A.5.4: cosine distance | cosine distance | |
| Default number of blocks J | SPECIFIED | A.5.4: J = 6 | 6 | |
| Repeated blocks in a path | PARTIAL | Hamiltonian path argument in Sec 3.2.4 | a block is used at most once | allow repeats |
| Router estimate for a dataset | UNSPECIFIED | Alg A3 runs on "a time series x" | router outputs averaged over 256 evenly spaced windows of the target train split (inputs only, no labels) | test inputs, single window |
| Cluster has fewer than J blocks | UNSPECIFIED | | stop early and log a warning (`restrict_to_cluster: false` searches the whole pool) | fall back to the whole pool |

## Training

| Item | Status | Paper | Our choice | Alternatives |
|---|---|---|---|---|
| Optimizer | SPECIFIED | A.5.4: AdamW | AdamW, `weight_decay=0.01` (PyTorch default, UNSPECIFIED) | |
| Learning rate, epochs, batch | SPECIFIED | A.5.4: 50 epochs, lr 0.001, batch 64 | same | |
| Samples per epoch | DEVIATION | full epochs | `samples_per_epoch: 20000`, `validation_samples: 5000`, because the grid has 360 forecasting models and a full ETTm epoch at C = 64 is 292k univariate windows. Set them to `null` for full epochs | |
| Model selection | UNSPECIFIED | | best validation epoch | last epoch |
| Pre-training grid | PARTIAL | Sec 3.2.2: "multiple context lengths and forecasting horizons", A.5.1: contexts 64 to 2048 | datasets, {mlp, lstm, attention}, contexts {64, 128, 256, 512, 1024, 2048}, horizons {96, 192, 336, 720}, P = 6 | P from 2 to 6 |
| Pre-training loss | UNSPECIFIED | Alg A1 line 7 | MSE (forecasting, reconstruction), masked MSE (imputation), cross entropy (classification) | MAE |
| Router epochs E | UNSPECIFIED | Alg A2 line 6 | 10 epochs per model, models visited in random order, 1 round | |
| Router training data | PARTIAL | Alg A2 line 3: residuals of the trained model | 256 evenly spaced training windows per pretrained model | |
| Imputation masks | UNSPECIFIED | three levels 10%, 20%, 30% | random point masks, ratio drawn per window from {0.1, 0.2, 0.3} during training, fixed seed at evaluation | block masks |
| Seed | UNSPECIFIED | | 2026 | |

## StreamCore and continual calibration

| Item | Status | Paper | Our choice |
|---|---|---|---|
| Replacement rule | PARTIAL | Sec 3.3: "Randomly selecting an existing point in S and deleting the point that is closer" | pick a random s in S, compare the nearest-neighbour distance of s and of the new point q to S without s, keep the one that is farther (more diverse) |
| Feature space | PARTIAL | A.4 bound uses the input distance between x_i and x_j | instance-normalized input window |
| Coreset weights w | UNSPECIFIED | Eq 1 | unweighted calibration |
| Subset size | SPECIFIED | Sec 4.2.2: 5% of the data | `coreset_fraction: 0.05` of the stream windows |
| Calibration period | SPECIFIED | Sec 4.2.2: R = 10 | calibrate every 10 periods |
| Number of periods | UNSPECIFIED | | 100 periods over the target train split, in time order |
| What is calibrated | SPECIFIED | Sec 3.3: all blocks frozen except the last | `BlockModel.freeze_all_but_last_block`. Baselines fine-tune all parameters |
| Calibration optimizer | UNSPECIFIED | | AdamW lr 1e-4, 1 epoch over the coreset, batch 64 |
| Random baseline buffer | PARTIAL | "a random 5% of the incoming data" | reservoir sampling |
| Calibrated weights | UNSPECIFIED | | saved to `weights/calibrated/`, never written back to the Blockbase, which stays shared by every leave-one-dataset-out target. A rerun loads them and reports the stored calibration time |

## Data

| Item | Status | Paper | Our choice |
|---|---|---|---|
| Datasets | SPECIFIED | Sec 4.1.1 | ETTh1, ETTh2, ETTm1, ETTm2, Weather, SMD, MSL, SMAP, SWaT, UCR 128 |
| Source | UNSPECIFIED | | THUML mirror `thuml/Time-Series-Library` on Hugging Face, UCR zips from timeseriesclassification.com |
| Forecasting splits | SPECIFIED | Table A1 | ETTh 10452/3484/3484, ETTm 41808/13936/13936, Weather 24388/14154/14154 |
| Anomaly splits | PARTIAL | Table A1 lists valid = test = half of the original test set | train = original train, validation = first half of the original test, test = second half |
| Scaling | UNSPECIFIED | | z-score per variable fitted on the train split, metrics on the scaled values |
| Channel handling | UNSPECIFIED | | every model, including the baselines, sees univariate windows, so models move between datasets with different numbers of variables |
| Baseline context length | SPECIFIED | A.5.1: 96 | 96 (TTM uses 512, see below) |
| Inference-time protocol | PARTIAL | Sec 4.2.1: "trained on a large group of data sets and then tested on one unseen data set" | leave-one-dataset-out: the target dataset's models are removed from the Blockbase, and baselines train on the other datasets of the task |
| UCR protocol | UNSPECIFIED | | blocks pre-trained on 12 large UCR datasets, evaluated on the other 116. For each target the selected blocks are frozen and only `Linear(512, K)` is trained on the target train split |
| UCR variable length and missing values | UNSPECIFIED | | linear interpolation of NaNs, resampled to the longest series of the dataset |
| Outlier threshold | PARTIAL | A.5.2: "self-determined threshold [22]" | threshold with the best F1 on the validation half (option `percentile` reproduces TSLib's anomaly ratio rule) |
| Point adjustment | UNSPECIFIED | | on, as in AnomalyTransformer, DCdetector and TSLib |

## Baselines

| Model | Tasks here | Source | Notes |
|---|---|---|---|
| Autoformer, PatchTST, TimesNet, TimeMixer | all four (TimeMixer, TimesNet also classification) | TSLib commit `4e938a1` | hyperparameters from the TSLib scripts for ETTh1 and MSL |
| AnomalyTransformer | outlier detection | official repo `b0ee470` | minimax loss and score from `solver.py` |
| DCdetector | outlier detection | official repo `9d107ca` | patch sizes 3, 4, 6 so they divide the window of 96 (DEVIATION from 3, 5, 7) |
| InceptionTime, ResNet | classification | own implementation of Fawaz 2020 and Wang 2017 | single network instead of the 5-network ensemble, odd kernels (39, 19, 9 and 7, 5, 3) for same padding, 500 epochs instead of 1500 |
| TTM | all four | `ibm-granite/granite-timeseries-ttm-r2` | DEVIATION: context 512 because no TTM-r2 checkpoint forecasts 96 or more steps from 96 points. Other tasks use the TTM backbone plus a linear head |
| Moment | all four | `AutonLab/MOMENT-1-large` | input left-padded to 512 with the padding mask, encoder frozen (linear probing) |
| Lag-Llama | forecasting | official repo `df7531a` plus HF checkpoint | mean of 100 samples (A.5.4) with the original probabilistic sampling and the KV cache. The repo's `use_single_pass_sampling` mode gave MSE 5.3 against 0.33 on 64 Weather windows, so it stays off. DEVIATION: test windows taken every 96 steps instead of every step to keep the autoregressive cost tractable |
| SymTime | forecasting | official repo `24a9386` plus `finetuning.pth` | Table 1 only, as in the paper |
| Pretrained baselines | PARTIAL | | fine-tuned on the source datasets (`finetune_on_source: true`), set it to `false` for pure zero-shot |

Lag-Llama imputation and outlier detection appear in Tables 3 and 4 but the paper does not explain how an autoregressive forecaster was adapted, so these two rows are not reproduced.

## Contradictions in the paper

- Sec 4.2.1 says the models are "pre-trained on ETTh1, ETTh2, ETTm1, ETTm2 and then only test on the unseen Weather data set", while Table 1 reports all five datasets. We use leave-one-dataset-out, which covers both readings for Weather.
- Table A1 lists SMD with 25 variables and SMAP with 38. The released files have 38 (SMD) and 25 (SMAP), so the two rows are swapped.
- Table A1 lists SMD with 7.18 million steps and SWaT with 489600, the released files have 1416825 (SMD) and 944919 (SWaT). We use the files as released. MSL (132046) and SMAP (562800) match Table A1 exactly.
- Fig 4 places Scaling before b1, yet b1 must have a homogeneous shape across positions. Our parameter-free sampling of a fixed-length residual keeps both properties.

## Expected numbers (paper)

| Table | Metric | TimeBlocks |
|---|---|---|
| Table 1, forecasting inference-time | average MSE | ETTh1 0.416, ETTh2 0.338, ETTm1 0.344, ETTm2 0.242, Weather 0.220 |
| Table 2, continual | average MSE | ETTh1 0.393, ETTh2 0.334, ETTm1 0.342, ETTm2 0.244, Weather 0.220 |
| Table 3, imputation | average MSE | ETTh1 0.518, ETTh2 0.298, ETTm1 0.385, ETTm2 0.248, Weather 0.234 |
| Table 4, outlier detection | F1 | SMD 0.854, MSL 0.848, SMAP 0.866, SWaT 0.915 |
| Fig 8a | size | 2.985 MB with J = 6 |

With the defaults here a 6-block model is about 5.6 MB, mostly in the output standardization layers of the finer blocks (`32 x 64 x 512` weights). A smaller `common_length` or `hidden_size` brings it to the paper's size.

## Open issues from the smoke tests

- Lag-Llama on Weather h96 (252 test windows, stride 500): zero-shot MSE 0.692, after one tiny fine-tuning epoch on ETTh1 (8 steps, lr 1e-4) MSE 1.750. Fine-tuning through `_compute_loss` runs without errors but has not been shown to help. Before the full run, compare `finetune_on_source: true` and `false` on one dataset and keep the better one.
- The smoke tests use 1 epoch and a few hundred samples, so their numbers only show that each path runs. None of the paper numbers have been reproduced yet.
