from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BlockSettings:
    common_length: int = 512
    patch_length: int = 16
    hidden_size: int = 64
    layers_per_block: int = 3
    attention_heads: int = 4
    dropout: float = 0.1
    normalization_epsilon: float = 1.0e-5


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    epochs: int = 50
    learning_rate: float = 1.0e-3
    batch_size: int = 64
    weight_decay: float = 1.0e-2
    patience: int | None = None
    samples_per_epoch: int | None = None
    validation_samples: int | None = None
    gradient_clip_norm: float | None = None
    num_workers: int = 4
    minimum_free_gigabytes: float = 2.0


@dataclass(frozen=True, slots=True)
class WindowSettings:
    train_stride: int = 1
    evaluation_stride: int = 1
    evaluation_batch_size: int = 512


@dataclass(frozen=True, slots=True)
class ImputationSettings:
    mask_ratios: tuple[float, ...] = (0.1, 0.2, 0.3)
    evaluation_seed: int = 2026


@dataclass(frozen=True, slots=True)
class AnomalySettings:
    threshold: str = "validation_f1"
    anomaly_ratio_percent: float = 1.0
    point_adjustment: bool = True


@dataclass(frozen=True, slots=True)
class PretrainSettings:
    task: str
    blockbase_name: str
    datasets: tuple[str, ...]
    architectures: tuple[str, ...] = ("mlp", "lstm", "attention")
    context_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    horizons: tuple[int, ...] = (96, 192, 336, 720)
    num_blocks: tuple[int, ...] = (6,)
    block: BlockSettings = field(default_factory=BlockSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    windows: WindowSettings = field(default_factory=WindowSettings)
    imputation: ImputationSettings = field(default_factory=ImputationSettings)
    skip_existing: bool = True
    seed: int = 2026
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class RouterSettings:
    channels: int = 32
    kernel_size: int = 5
    pooled_length: int = 32
    epochs: int = 10
    rounds: int = 1
    learning_rate: float = 1.0e-3
    batch_size: int = 64
    loss: str = "mse"
    windows_per_model: int = 256


@dataclass(frozen=True, slots=True)
class ClusteringSettings:
    num_clusters: int | None = None
    normalize: bool = True
    initializations: int = 10


@dataclass(frozen=True, slots=True)
class BlockbaseSettings:
    blockbase_name: str
    pool_tasks: tuple[str, ...]
    target_datasets: tuple[str, ...] = ()
    router: RouterSettings = field(default_factory=RouterSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    imputation: ImputationSettings = field(default_factory=ImputationSettings)
    seed: int = 2026
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    num_blocks: int = 6
    windows: int = 256
    split: str = "train"
    restrict_to_cluster: bool = True


@dataclass(frozen=True, slots=True)
class StreamSettings:
    coreset: str = "streamcore"
    coreset_fraction: float = 0.05
    num_periods: int = 100
    calibration_period: int = 10
    calibration_epochs: int = 1
    learning_rate: float = 1.0e-4
    batch_size: int = 64
    stream_stride: int = 1


@dataclass(frozen=True, slots=True)
class ZeroShotSettings:
    task: str
    blockbase_name: str
    pool_tasks: tuple[str, ...]
    target_datasets: tuple[str, ...]
    horizons: tuple[int, ...] = (96, 192, 336, 720)
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    windows: WindowSettings = field(default_factory=WindowSettings)
    imputation: ImputationSettings = field(default_factory=ImputationSettings)
    anomaly: AnomalySettings = field(default_factory=AnomalySettings)
    head: OptimizationSettings = field(default_factory=OptimizationSettings)
    stream: StreamSettings | None = None
    exclude_target_from_pool: bool = True
    seed: int = 2026
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    model: str
    task: str
    target_datasets: tuple[str, ...]
    source_datasets: tuple[str, ...] = ()
    context_length: int = 96
    horizons: tuple[int, ...] = (96, 192, 336, 720)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    finetune_on_source: bool = True
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    windows: WindowSettings = field(default_factory=WindowSettings)
    imputation: ImputationSettings = field(default_factory=ImputationSettings)
    anomaly: AnomalySettings = field(default_factory=AnomalySettings)
    stream: StreamSettings | None = None
    seed: int = 2026
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class DownloadSettings:
    forecasting: tuple[str, ...] = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
    anomaly_detection: tuple[str, ...] = ("SMD", "MSL", "SMAP", "SWaT")
    classification: tuple[str, ...] = ("all",)
    retries: int = 3


@dataclass(frozen=True, slots=True)
class ReportSettings:
    task: str
    metric: str
    kind: str = "table"
    setting: str = "inference_time"
    lower_is_better: bool = True
    figure_name: str = ""
