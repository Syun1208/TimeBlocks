from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.classes.enums import Split, Task
from src.data.classes.settings import BaselineSettings
from src.data.loaders.catalog import DATASETS
from src.data.loaders.windows import classification_samples
from src.models.baselines.foundation import WindowRequirements
from src.models.baselines.registry import BaselineFactory, BaselineRequest, BuiltBaseline
from src.services.checkpoints import BASELINE_DIRECTORY, CheckpointKey, CheckpointStore
from src.services.continual import ContinualCalibrationService, continual_model_name
from src.services.evaluation import AnomalyEvaluation, Evaluator
from src.services.results import ResultStore
from src.services.thresholds import build_threshold_strategy
from src.services.training import Trainer
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import EXPERIMENT, finish, progress
from src.utils.runtime import module_size_megabytes, set_seed

logger = logging.getLogger(__name__)

SINGLE_CLASS = 1
NO_HORIZON = 0


def default_sources(
    *,
    task: Task,
    target: str,
) -> tuple[str, ...]:
    return tuple(name for name, spec in DATASETS.items() if spec.task is task and name != target)


class BaselineExperimentService:
    def __init__(
        self,
        *,
        settings: BaselineSettings,
        windows: WindowFactory,
        factory: BaselineFactory,
        evaluator: Evaluator,
        results: ResultStore,
        checkpoints: CheckpointStore,
        continual: ContinualCalibrationService | None,
        device: torch.device,
    ) -> None:
        self._settings = settings
        self._windows = windows
        self._factory = factory
        self._evaluator = evaluator
        self._results = results
        self._checkpoints = checkpoints
        self._continual = continual
        self._device = device
        self._task = Task(settings.task)

    def _sources(self, *, target: str) -> tuple[str, ...]:
        return self._settings.source_datasets or default_sources(
            task=self._task,
            target=target,
        )

    def _identity(
        self,
        *,
        target: str,
        context_length: int,
    ) -> dict[str, Any]:
        return {
            "model": self._settings.model,
            "task": self._task.value,
            "context_length": context_length,
            "hyperparameters": self._settings.hyperparameters,
            "finetune_on_source": self._settings.finetune_on_source,
            "optimization": asdict(self._settings.optimization),
            "imputation": asdict(self._settings.imputation),
            "train_stride": self._settings.windows.train_stride,
            "sources": list(self._sources(target=target)) if self._task is not Task.CLASSIFICATION else [],
            "seed": self._settings.seed,
        }

    def _checkpoint_key(
        self,
        *,
        target: str,
        horizon: int,
        context_length: int,
    ) -> CheckpointKey:
        return CheckpointKey(
            group=f"{BASELINE_DIRECTORY}/{self._settings.model}",
            name=f"{target}_h{horizon}" if horizon else target,
            identity=self._identity(
                target=target,
                context_length=context_length,
            ),
        )

    def _fit_or_load(
        self,
        *,
        model: nn.Module,
        key: CheckpointKey,
        train: Callable[[], None],
        label: str,
    ) -> None:
        if self._checkpoints.exists(key=key):
            self._checkpoints.load(
                key=key,
                model=model,
            )
            logger.info("%s reused its trained weights instead of training again", label)
            return
        train()
        self._checkpoints.save(
            key=key,
            model=model,
            metadata={"label": label},
        )

    def _request(
        self,
        *,
        dataset: str,
        split: Split,
        horizon: int,
        requirements: WindowRequirements,
        stride: int,
        mask_ratios: tuple[float, ...] | None = None,
    ) -> WindowRequest:
        return WindowRequest(
            task=self._task,
            dataset=dataset,
            split=split.value,
            context_length=self._settings.context_length,
            horizon=horizon,
            stride=stride,
            mask_ratios=mask_ratios or self._settings.imputation.mask_ratios,
            mask_seed=None if split is Split.TRAIN else self._settings.imputation.evaluation_seed,
            history_length=requirements.history_length,
            time_features=requirements.time_features,
        )

    def _source_datasets(
        self,
        *,
        target: str,
        horizon: int,
        requirements: WindowRequirements,
    ) -> tuple[Dataset, Dataset]:
        built = {}
        for split in (Split.TRAIN, Split.VALIDATION):
            built[split] = self._windows.build_many(
                requests=[
                    self._request(
                        dataset=source,
                        split=split,
                        horizon=horizon,
                        requirements=requirements,
                        stride=self._settings.windows.train_stride,
                    )
                    for source in self._sources(target=target)
                ]
            )
        return built[Split.TRAIN], built[Split.VALIDATION]

    def _build(
        self,
        *,
        horizon: int,
        num_classes: int,
    ) -> BuiltBaseline:
        set_seed(seed=self._settings.seed)
        built = self._factory.build(
            name=self._settings.model,
            request=BaselineRequest(
                task=self._task,
                context_length=self._settings.context_length,
                horizon=horizon,
                num_classes=num_classes,
                hyperparameters=self._settings.hyperparameters,
            ),
        )
        built.model.to(self._device)
        return built

    def _fit(
        self,
        *,
        model: nn.Module,
        train: Dataset,
        validation: Dataset | None,
        label: str,
    ) -> None:
        Trainer(
            settings=self._settings.optimization,
            device=self._device,
            seed=self._settings.seed,
            label=label,
        ).fit(
            model=model,
            train_dataset=train,
            validation_dataset=validation,
        )

    def _train_on_sources(
        self,
        *,
        built: BuiltBaseline,
        target: str,
        horizon: int,
        label: str,
    ) -> None:
        if built.pretrained and not self._settings.finetune_on_source:
            logger.info("%s used zero-shot from its public pretrained weights", self._settings.model)
            return

        def train_on_sources() -> None:
            train, validation = self._source_datasets(
                target=target,
                horizon=horizon,
                requirements=built.requirements,
            )
            logger.info("%s training on %d source windows from %s", label, len(train), self._sources(target=target))
            self._fit(
                model=built.model,
                train=train,
                validation=validation,
                label=label,
            )

        self._fit_or_load(
            model=built.model,
            key=self._checkpoint_key(
                target=target,
                horizon=horizon,
                context_length=self._settings.context_length,
            ),
            train=train_on_sources,
            label=label,
        )

    def _record(
        self,
        *,
        target: str,
        model: nn.Module,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "model": self._settings.model,
            "task": self._task.value,
            "dataset": target,
            "context_length": self._settings.context_length,
            "size_mb": module_size_megabytes(module=model),
            "sources": list(self._sources(target=target)) if self._task is not Task.CLASSIFICATION else [],
        }
        record.update(extra)
        return record

    def _forecasting(self, *, target: str) -> None:
        for horizon in self._settings.horizons:
            built = self._build(
                horizon=horizon,
                num_classes=SINGLE_CLASS,
            )
            label = f"{self._settings.model} {target} h{horizon}"
            self._train_on_sources(
                built=built,
                target=target,
                horizon=horizon,
                label=label,
            )
            test = self._windows.build(
                request=self._request(
                    dataset=target,
                    split=Split.TEST,
                    horizon=horizon,
                    requirements=built.requirements,
                    stride=self._settings.windows.evaluation_stride,
                )
            )
            record = self._record(
                target=target,
                model=built.model,
                extra={"horizon": horizon, "setting": "inference_time"},
            )
            record.update(
                self._evaluator.regression(
                    model=built.model,
                    dataset=test,
                    label=label,
                    masked_only=False,
                )
            )
            self._results.append(record=record)
            if self._continual is not None:
                self._calibrate(
                    built=built,
                    target=target,
                    horizon=horizon,
                    test=test,
                    label=label,
                )

    def _calibrate(
        self,
        *,
        built: BuiltBaseline,
        target: str,
        horizon: int,
        test: Dataset,
        label: str,
    ) -> None:
        stream = self._windows.build(
            request=self._request(
                dataset=target,
                split=Split.TRAIN,
                horizon=horizon,
                requirements=built.requirements,
                stride=self._settings.stream.stream_stride,
            )
        )
        report = self._continual.run(
            model=built.model,
            stream=stream,
            label=label,
            key=self._continual.checkpoint_key(
                model_name=self._settings.model,
                name=f"{target}_h{horizon}",
                identity=self._identity(
                    target=target,
                    context_length=self._settings.context_length,
                ),
            ),
        )
        record = self._record(
            target=target,
            model=built.model,
            extra={
                "horizon": horizon,
                "setting": "continual",
                "model": continual_model_name(
                    model=self._settings.model,
                    coreset=self._settings.stream.coreset,
                ),
                "coreset": self._settings.stream.coreset,
                "coreset_size": report.coreset_size,
                "stream_samples": report.stream_samples,
                "calibrations": report.calibrations,
                "calibration_seconds": report.calibration_seconds,
                "calibration_loaded": report.loaded,
            },
        )
        record.update(
            self._evaluator.regression(
                model=built.model,
                dataset=test,
                label=f"{label} calibrated",
                masked_only=False,
            )
        )
        self._results.append(record=record)

    def _imputation(self, *, target: str) -> None:
        built = self._build(
            horizon=NO_HORIZON,
            num_classes=SINGLE_CLASS,
        )
        label = f"{self._settings.model} {target}"
        self._train_on_sources(
            built=built,
            target=target,
            horizon=NO_HORIZON,
            label=label,
        )
        for ratio in self._settings.imputation.mask_ratios:
            test = self._windows.build(
                request=self._request(
                    dataset=target,
                    split=Split.TEST,
                    horizon=NO_HORIZON,
                    requirements=built.requirements,
                    stride=self._settings.windows.evaluation_stride,
                    mask_ratios=(ratio,),
                )
            )
            record = self._record(
                target=target,
                model=built.model,
                extra={"mask_ratio": ratio, "setting": "inference_time"},
            )
            record.update(
                self._evaluator.regression(
                    model=built.model,
                    dataset=test,
                    label=f"{label} mask {ratio}",
                    masked_only=True,
                )
            )
            self._results.append(record=record)

    def _anomaly(self, *, target: str) -> None:
        built = self._build(
            horizon=NO_HORIZON,
            num_classes=SINGLE_CLASS,
        )
        self._train_on_sources(
            built=built,
            target=target,
            horizon=NO_HORIZON,
            label=f"{self._settings.model} {target}",
        )
        evaluation = AnomalyEvaluation(
            evaluator=self._evaluator,
            threshold=build_threshold_strategy(settings=self._settings.anomaly),
            point_adjustment=self._settings.anomaly.point_adjustment,
        )
        record = self._record(
            target=target,
            model=built.model,
            extra={"setting": "inference_time", "threshold_strategy": self._settings.anomaly.threshold},
        )
        record.update(
            evaluation.run(
                scorer=built.model,
                windows_by_split={
                    split.value: self._windows.scoring(
                        dataset=target,
                        split=split.value,
                        context_length=self._settings.context_length,
                    )
                    for split in Split
                },
                labels=self._windows.repository.load_series(name=target).labels,
            )
        )
        self._results.append(record=record)

    def _classification(self, *, target: str) -> None:
        splits = self._windows.repository.load_classification(name=target)
        built = self._factory.build(
            name=self._settings.model,
            request=BaselineRequest(
                task=self._task,
                context_length=splits.series_length,
                horizon=NO_HORIZON,
                num_classes=splits.num_classes,
                hyperparameters=self._settings.hyperparameters,
            ),
        )
        label = f"{self._settings.model} {target}"
        self._fit_or_load(
            model=built.model,
            key=self._checkpoint_key(
                target=target,
                horizon=NO_HORIZON,
                context_length=splits.series_length,
            ),
            train=lambda: self._fit(
                model=built.model,
                train=classification_samples(
                    splits=splits,
                    split=Split.TRAIN.value,
                ),
                validation=None,
                label=label,
            ),
            label=label,
        )
        record = self._record(
            target=target,
            model=built.model,
            extra={"setting": "inference_time", "context_length": splits.series_length},
        )
        record.update(
            self._evaluator.classification(
                model=built.model,
                dataset=classification_samples(
                    splits=splits,
                    split=Split.TEST.value,
                ),
                label=target,
            )
        )
        self._results.append(record=record)

    def _handlers(self) -> dict[Task, Callable[..., None]]:
        return {
            Task.FORECASTING: self._forecasting,
            Task.IMPUTATION: self._imputation,
            Task.ANOMALY_DETECTION: self._anomaly,
            Task.CLASSIFICATION: self._classification,
        }

    def run(self) -> None:
        handler = self._handlers()[self._task]
        bar = progress(
            stage=EXPERIMENT,
            total=len(self._settings.target_datasets),
            unit="dataset",
        )
        for target in self._settings.target_datasets:
            bar.set_postfix_str(target)
            set_seed(seed=self._settings.seed)
            handler(target=target)
            bar.update(1)
        finish(bar=bar)
        self._results.write_summary()
