from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.data.classes.blocks import ModelRecord
from src.data.classes.enums import Architecture, Split, Task
from src.data.classes.settings import PretrainSettings
from src.services.blockbase_store import BlockbaseStore
from src.services.model_building import build_fresh_blocks, build_task_model
from src.services.training import Trainer
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import PRETRAIN, finish, progress
from src.utils.runtime import count_parameters, set_seed

logger = logging.getLogger(__name__)

NO_HORIZON = 0
SINGLE_CLASS = 1


@dataclass(frozen=True, slots=True)
class PretrainJob:
    task: Task
    dataset: str
    architecture: Architecture
    context_length: int
    horizon: int
    num_blocks: int

    @property
    def model_id(self) -> str:
        return (
            f"{self.task.value}-{self.dataset}-{self.architecture.value}"
            f"-c{self.context_length}-h{self.horizon}-p{self.num_blocks}"
        )


class PretrainGrid:
    def __init__(
        self,
        *,
        settings: PretrainSettings,
        series_length: Callable[[str], int],
    ) -> None:
        self._settings = settings
        self._series_length = series_length

    def jobs(self) -> list[PretrainJob]:
        task = Task(self._settings.task)
        jobs = []
        for dataset, architecture, num_blocks in itertools.product(
            self._settings.datasets,
            self._settings.architectures,
            self._settings.num_blocks,
        ):
            context_lengths = (
                (self._series_length(dataset),) if task is Task.CLASSIFICATION else self._settings.context_lengths
            )
            horizons = self._settings.horizons if task is Task.FORECASTING else (NO_HORIZON,)
            for context_length, horizon in itertools.product(context_lengths, horizons):
                jobs.append(
                    PretrainJob(
                        task=task,
                        dataset=dataset,
                        architecture=Architecture(architecture),
                        context_length=context_length,
                        horizon=horizon,
                        num_blocks=num_blocks,
                    )
                )
        return jobs


class BlockPretrainingService:
    def __init__(
        self,
        *,
        settings: PretrainSettings,
        windows: WindowFactory,
        store: BlockbaseStore,
        device: torch.device,
    ) -> None:
        self._settings = settings
        self._windows = windows
        self._store = store
        self._device = device

    def _request(
        self,
        *,
        job: PretrainJob,
        split: Split,
    ) -> WindowRequest:
        return WindowRequest(
            task=job.task,
            dataset=job.dataset,
            split=split.value,
            context_length=job.context_length,
            horizon=job.horizon,
            stride=self._settings.windows.train_stride,
            mask_ratios=self._settings.imputation.mask_ratios,
            mask_seed=None if split is Split.TRAIN else self._settings.imputation.evaluation_seed,
        )

    def _datasets(self, *, job: PretrainJob) -> tuple[Dataset, Dataset | None]:
        train = self._windows.build(
            request=self._request(
                job=job,
                split=Split.TRAIN,
            )
        )
        if job.task is Task.CLASSIFICATION:
            return train, None
        return train, self._windows.build(
            request=self._request(
                job=job,
                split=Split.VALIDATION,
            )
        )

    def _num_classes(self, *, job: PretrainJob) -> int:
        if job.task is not Task.CLASSIFICATION:
            return SINGLE_CLASS
        return self._windows.repository.load_classification(name=job.dataset).num_classes

    def train_job(self, *, job: PretrainJob) -> ModelRecord:
        set_seed(seed=self._settings.seed)
        model = build_task_model(
            task=job.task,
            settings=self._settings.block,
            blocks=build_fresh_blocks(
                settings=self._settings.block,
                architecture=job.architecture,
                num_blocks=job.num_blocks,
            ),
            context_length=job.context_length,
            horizon=job.horizon,
            num_classes=self._num_classes(job=job),
        )
        train, validation = self._datasets(job=job)
        logger.info("Training %s on %d windows", job.model_id, len(train))
        result = Trainer(
            settings=self._settings.optimization,
            device=self._device,
            seed=self._settings.seed,
            label=job.model_id,
        ).fit(
            model=model,
            train_dataset=train,
            validation_dataset=validation,
        )
        record = ModelRecord(
            model_id=job.model_id,
            task=job.task.value,
            dataset=job.dataset,
            architecture=job.architecture.value,
            context_length=job.context_length,
            horizon=job.horizon,
            num_blocks=job.num_blocks,
            output_size=self._num_classes(job=job),
            validation_loss=result.best_validation_loss,
            num_parameters=count_parameters(module=model.network.blocks),
        )
        self._store.save(
            record=record,
            model=model.network,
            settings=self._settings.block,
        )
        logger.info(
            "Saved %s, validation loss %.5f, best epoch %d, %.1fs",
            job.model_id,
            result.best_validation_loss,
            result.best_epoch,
            result.seconds,
        )
        return record

    def run(
        self,
        *,
        jobs: list[PretrainJob],
        shard_index: int,
        num_shards: int,
    ) -> list[ModelRecord]:
        remaining = [
            job for job in jobs if not (self._settings.skip_existing and self._store.exists(model_id=job.model_id))
        ]
        pending = remaining[shard_index::num_shards]
        logger.info(
            "Shard %d/%d has %d of %d pending jobs, %d already trained",
            shard_index,
            num_shards,
            len(pending),
            len(remaining),
            len(jobs) - len(remaining),
        )
        bar = progress(
            stage=PRETRAIN,
            total=len(pending),
            unit="model",
        )
        records = []
        for job in pending:
            bar.set_postfix_str(job.model_id)
            records.append(self.train_job(job=job))
            bar.update(1)
        finish(bar=bar)
        return records
