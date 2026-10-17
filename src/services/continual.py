from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from src.data.classes.series import Batch
from src.data.classes.settings import OptimizationSettings, StreamSettings
from src.data.loaders.windows import build_loader
from src.models.timeblocks.standardization import instance_normalize
from src.services.checkpoints import CALIBRATED_DIRECTORY, CheckpointKey, CheckpointStore
from src.services.coresets import build_coreset
from src.services.training import Trainer
from src.utils.progress import STREAM, finish, progress

logger = logging.getLogger(__name__)

FEATURE_EPSILON = 1.0e-5
FEATURE_BATCH_SIZE = 1024
NO_WORKERS = 0
TIMEBLOCKS = "TimeBlocks"
DEFAULT_CORESET = {True: "streamcore", False: "random"}


@dataclass(frozen=True, slots=True)
class ContinualReport:
    stream_samples: int
    coreset_size: int
    calibrations: int
    calibration_seconds: float
    loaded: bool = False


def continual_model_name(
    *,
    model: str,
    coreset: str,
) -> str:
    return model if coreset == DEFAULT_CORESET[model == TIMEBLOCKS] else f"{model} + {coreset.capitalize()}"


def stream_features(*, batch: Batch) -> torch.Tensor:
    normalized, _ = instance_normalize(
        values=batch.inputs[..., 0],
        mask=None,
        epsilon=FEATURE_EPSILON,
    )
    return normalized


class ContinualCalibrationService:
    def __init__(
        self,
        *,
        settings: StreamSettings,
        store: CheckpointStore,
        device: torch.device,
        seed: int,
    ) -> None:
        self._settings = settings
        self._store = store
        self._device = device
        self._seed = seed

    def _calibration_trainer(self, *, label: str) -> Trainer:
        return Trainer(
            settings=OptimizationSettings(
                epochs=self._settings.calibration_epochs,
                learning_rate=self._settings.learning_rate,
                batch_size=self._settings.batch_size,
                num_workers=NO_WORKERS,
            ),
            device=self._device,
            seed=self._seed,
            label=label,
        )

    def checkpoint_key(
        self,
        *,
        model_name: str,
        name: str,
        identity: dict[str, Any],
    ) -> CheckpointKey:
        return CheckpointKey(
            group=f"{CALIBRATED_DIRECTORY}/{model_name}",
            name=name,
            identity={**identity, "stream": asdict(self._settings), "seed": self._seed},
        )

    def run(
        self,
        *,
        model: nn.Module,
        stream: Dataset,
        label: str,
        key: CheckpointKey,
    ) -> ContinualReport:
        if self._store.exists(key=key):
            metadata = self._store.load(
                key=key,
                model=model,
            )
            logger.info("%s reused the calibrated weights instead of calibrating again", label)
            return ContinualReport(**{**metadata, "loaded": True})
        report = self._calibrate(
            model=model,
            stream=stream,
            label=label,
        )
        self._store.save(
            key=key,
            model=model,
            metadata=asdict(report),
        )
        return report

    def _calibrate(
        self,
        *,
        model: nn.Module,
        stream: Dataset,
        label: str,
    ) -> ContinualReport:
        total = len(stream)
        capacity = max(1, math.ceil(self._settings.coreset_fraction * total))
        coreset = build_coreset(
            kind=self._settings.coreset,
            capacity=capacity,
            seed=self._seed,
            device=self._device,
        )
        periods = np.array_split(np.arange(total), self._settings.num_periods)
        trainer = self._calibration_trainer(label=f"{label} calibrate")
        calibrations, seconds = 0, 0.0
        bar = progress(
            stage=STREAM,
            total=len(periods),
            unit="period",
            detail=label,
        )
        for period, indices in enumerate(periods, start=1):
            loader = build_loader(
                dataset=Subset(stream, indices.tolist()),
                batch_size=FEATURE_BATCH_SIZE,
                shuffle=False,
                num_workers=NO_WORKERS,
            )
            offset = 0
            for items in loader:
                batch = Batch.from_mapping(
                    items=items,
                    device=self._device,
                )
                coreset.insert(
                    features=stream_features(batch=batch),
                    indices=torch.from_numpy(indices[offset : offset + batch.size]),
                )
                offset += batch.size
            if period % self._settings.calibration_period == 0:
                result = trainer.fit(
                    model=model,
                    train_dataset=Subset(stream, coreset.selected()),
                    validation_dataset=None,
                )
                calibrations += 1
                seconds += result.seconds
            bar.update(1)
            bar.set_postfix_str(f"coreset {len(coreset.selected())}/{capacity} calibrations {calibrations}")
        finish(bar=bar)
        return ContinualReport(
            stream_samples=total,
            coreset_size=len(coreset.selected()),
            calibrations=calibrations,
            calibration_seconds=seconds,
        )
