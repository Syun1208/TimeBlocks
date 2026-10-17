from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from src.data.classes.blocks import SelectionResult
from src.data.classes.enums import Split, Task
from src.data.classes.settings import BlockSettings, ZeroShotSettings
from src.data.loaders.windows import classification_samples
from src.models.timeblocks.block import TimeBlock
from src.services.block_selection import BlockSelector, SelectionWindowSampler
from src.services.blockbase_store import BlockbaseIndexStore, BlockbaseStore, freeze
from src.services.continual import ContinualCalibrationService, continual_model_name
from src.services.evaluation import AnomalyEvaluation, Evaluator
from src.services.model_building import build_task_model
from src.services.results import ResultStore
from src.services.thresholds import build_threshold_strategy
from src.services.training import Trainer
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import EXPERIMENT, finish, progress
from src.utils.runtime import module_size_megabytes, set_seed

logger = logging.getLogger(__name__)

MODEL_NAME = "TimeBlocks"
SINGLE_CLASS = 1
NO_HORIZON = 0


class TimeBlocksExperimentService:
    def __init__(
        self,
        *,
        settings: ZeroShotSettings,
        windows: WindowFactory,
        store: BlockbaseStore,
        index_store: BlockbaseIndexStore,
        evaluator: Evaluator,
        results: ResultStore,
        continual: ContinualCalibrationService | None,
        device: torch.device,
    ) -> None:
        self._settings = settings
        self._windows = windows
        self._store = store
        self._index_store = index_store
        self._evaluator = evaluator
        self._results = results
        self._continual = continual
        self._device = device
        self._task = Task(settings.task)

    def _selector(self, *, target: str) -> tuple[BlockSelector, BlockSettings]:
        router, fingerprints, blocks, clusters = self._index_store.load(
            directory=self._index_store.directory(
                blockbase_name=self._settings.blockbase_name,
                pool_tasks=self._settings.pool_tasks,
                excluded_dataset=target if self._settings.exclude_target_from_pool else None,
            )
        )
        block_settings = self._store.block_settings(model_id=blocks[0].model_id)
        selector = BlockSelector(
            router=router,
            store=self._store,
            fingerprints=fingerprints,
            blocks=blocks,
            clusters=clusters,
            block_settings=block_settings,
            settings=self._settings.selection,
            device=self._device,
        )
        return selector, block_settings

    def _assemble(
        self,
        *,
        block_settings: BlockSettings,
        blocks: list[TimeBlock],
        context_length: int,
        horizon: int,
        num_classes: int,
    ) -> nn.Module:
        return build_task_model(
            task=self._task,
            settings=block_settings,
            blocks=copy.deepcopy(blocks),
            context_length=context_length,
            horizon=horizon,
            num_classes=num_classes,
        ).to(self._device)

    def _base_record(
        self,
        *,
        target: str,
        selection: SelectionResult,
        model: nn.Module,
    ) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "task": self._task.value,
            "dataset": target,
            "num_blocks": len(selection.block_ids),
            "context_length": selection.context_length,
            "cluster": selection.cluster,
            "selection_seconds": selection.seconds,
            "size_mb": module_size_megabytes(module=model),
            "blocks": list(selection.block_ids),
        }

    def _forecasting(
        self,
        *,
        target: str,
        selection: SelectionResult,
        blocks: list[TimeBlock],
        block_settings: BlockSettings,
    ) -> None:
        for horizon in self._settings.horizons:
            model = self._assemble(
                block_settings=block_settings,
                blocks=blocks,
                context_length=selection.context_length,
                horizon=horizon,
                num_classes=SINGLE_CLASS,
            )
            test = self._windows.build(
                request=WindowRequest(
                    task=self._task,
                    dataset=target,
                    split=Split.TEST.value,
                    context_length=selection.context_length,
                    horizon=horizon,
                    stride=self._settings.windows.evaluation_stride,
                )
            )
            record = self._base_record(
                target=target,
                selection=selection,
                model=model,
            )
            record.update({"horizon": horizon, "setting": "inference_time"})
            record.update(
                self._evaluator.regression(
                    model=model,
                    dataset=test,
                    label=f"{target} h{horizon}",
                    masked_only=False,
                )
            )
            self._results.append(record=record)
            if self._continual is None:
                continue
            self._calibrate(
                target=target,
                model=model,
                selection=selection,
                horizon=horizon,
                test=test,
            )

    def _calibrate(
        self,
        *,
        target: str,
        model: nn.Module,
        selection: SelectionResult,
        horizon: int,
        test,
    ) -> None:
        model.network.freeze_all_but_last_block()
        stream = self._windows.build(
            request=WindowRequest(
                task=self._task,
                dataset=target,
                split=Split.TRAIN.value,
                context_length=selection.context_length,
                horizon=horizon,
                stride=self._settings.stream.stream_stride,
            )
        )
        report = self._continual.run(
            model=model,
            stream=stream,
            label=f"{target} h{horizon}",
            key=self._continual.checkpoint_key(
                model_name=MODEL_NAME,
                name=f"{target}_h{horizon}",
                identity={
                    "task": self._task.value,
                    "blockbase_name": self._settings.blockbase_name,
                    "pool_tasks": list(self._settings.pool_tasks),
                    "exclude_target_from_pool": self._settings.exclude_target_from_pool,
                    "context_length": selection.context_length,
                    "blocks": list(selection.block_ids),
                },
            ),
        )
        record = self._base_record(
            target=target,
            selection=selection,
            model=model,
        )
        record.update(
            {
                "horizon": horizon,
                "setting": "continual",
                "model": continual_model_name(
                    model=MODEL_NAME,
                    coreset=self._settings.stream.coreset,
                ),
                "coreset": self._settings.stream.coreset,
                "coreset_size": report.coreset_size,
                "stream_samples": report.stream_samples,
                "calibrations": report.calibrations,
                "calibration_seconds": report.calibration_seconds,
                "calibration_loaded": report.loaded,
            }
        )
        record.update(
            self._evaluator.regression(
                model=model,
                dataset=test,
                label=f"{target} h{horizon} calibrated",
                masked_only=False,
            )
        )
        self._results.append(record=record)

    def _imputation(
        self,
        *,
        target: str,
        selection: SelectionResult,
        blocks: list[TimeBlock],
        block_settings: BlockSettings,
    ) -> None:
        model = self._assemble(
            block_settings=block_settings,
            blocks=blocks,
            context_length=selection.context_length,
            horizon=NO_HORIZON,
            num_classes=SINGLE_CLASS,
        )
        for ratio in self._settings.imputation.mask_ratios:
            test = self._windows.build(
                request=WindowRequest(
                    task=self._task,
                    dataset=target,
                    split=Split.TEST.value,
                    context_length=selection.context_length,
                    stride=self._settings.windows.evaluation_stride,
                    mask_ratios=(ratio,),
                    mask_seed=self._settings.imputation.evaluation_seed,
                )
            )
            record = self._base_record(
                target=target,
                selection=selection,
                model=model,
            )
            record.update({"mask_ratio": ratio, "setting": "inference_time"})
            record.update(
                self._evaluator.regression(
                    model=model,
                    dataset=test,
                    label=f"{target} mask {ratio}",
                    masked_only=True,
                )
            )
            self._results.append(record=record)

    def _anomaly(
        self,
        *,
        target: str,
        selection: SelectionResult,
        blocks: list[TimeBlock],
        block_settings: BlockSettings,
    ) -> None:
        model = self._assemble(
            block_settings=block_settings,
            blocks=blocks,
            context_length=selection.context_length,
            horizon=NO_HORIZON,
            num_classes=SINGLE_CLASS,
        )
        evaluation = AnomalyEvaluation(
            evaluator=self._evaluator,
            threshold=build_threshold_strategy(settings=self._settings.anomaly),
            point_adjustment=self._settings.anomaly.point_adjustment,
        )
        windows_by_split = {
            split.value: self._windows.scoring(
                dataset=target,
                split=split.value,
                context_length=selection.context_length,
            )
            for split in Split
        }
        record = self._base_record(
            target=target,
            selection=selection,
            model=model,
        )
        record.update({"setting": "inference_time", "threshold_strategy": self._settings.anomaly.threshold})
        record.update(
            evaluation.run(
                scorer=model,
                windows_by_split=windows_by_split,
                labels=self._windows.repository.load_series(name=target).labels,
            )
        )
        self._results.append(record=record)

    def _classification(
        self,
        *,
        target: str,
        selection: SelectionResult,
        blocks: list[TimeBlock],
        block_settings: BlockSettings,
    ) -> None:
        splits = self._windows.repository.load_classification(name=target)
        model = self._assemble(
            block_settings=block_settings,
            blocks=blocks,
            context_length=splits.series_length,
            horizon=NO_HORIZON,
            num_classes=splits.num_classes,
        )
        freeze(module=model.network.blocks)
        Trainer(
            settings=self._settings.head,
            device=self._device,
            seed=self._settings.seed,
            label=f"{target} head",
        ).fit(
            model=model,
            train_dataset=classification_samples(
                splits=splits,
                split=Split.TRAIN.value,
            ),
            validation_dataset=None,
        )
        record = self._base_record(
            target=target,
            selection=selection,
            model=model,
        )
        record.update({"setting": "inference_time"})
        record.update(
            self._evaluator.classification(
                model=model,
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
            selector, block_settings = self._selector(target=target)
            selection, blocks = selector.select(
                sampler=SelectionWindowSampler(
                    windows=self._windows,
                    task=self._task,
                    dataset=target,
                    settings=self._settings.selection,
                )
            )
            handler(
                target=target,
                selection=selection,
                blocks=blocks,
                block_settings=block_settings,
            )
            self._store.release()
            bar.update(1)
        finish(bar=bar)
        self._results.write_summary()
