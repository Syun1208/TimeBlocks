from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.data.classes.enums import Task
from src.data.classes.settings import (
    BaselineSettings,
    BlockbaseSettings,
    DownloadSettings,
    OptimizationSettings,
    PretrainSettings,
    StreamSettings,
    WindowSettings,
    ZeroShotSettings,
)
from src.data.loaders.downloaders import HuggingFaceDatasetDownloader, UcrArchiveDownloader
from src.data.loaders.readers import DatasetRepository
from src.data.loaders.time_features import TimeFeatureEncoder
from src.models.baselines.registry import BaselineFactory
from src.services.baseline_experiment import BaselineExperimentService
from src.services.blockbase_index import BlockbaseIndexService
from src.services.blockbase_store import BLOCKBASE_DIRECTORY, INDEX_DIRECTORY, BlockbaseIndexStore, BlockbaseStore
from src.services.checkpoints import CheckpointStore
from src.services.clustering import BlockClusterer
from src.services.continual import ContinualCalibrationService
from src.services.download import DatasetDownloadService
from src.services.evaluation import Evaluator
from src.services.pretraining import BlockPretrainingService, PretrainGrid
from src.services.results import ResultStore
from src.services.router_training import ResidualCollector, RouterTrainingService
from src.data.loaders.catalog import TABLE_A1_SPLIT
from src.services.windowing import WindowFactory
from src.utils.env import StoragePaths
from src.utils.runtime import ensure_free_memory, resolve_device
from src.utils.third_party import IsolatedImporter

logger = logging.getLogger(__name__)


def build_window_factory(
    *,
    storage: StoragePaths,
    split_protocol: str = TABLE_A1_SPLIT,
) -> WindowFactory:
    return WindowFactory(
        repository=DatasetRepository(
            data_root=storage.data,
            split_protocol=split_protocol,
        ),
        time_encoder=TimeFeatureEncoder(),
    )


def build_blockbase_store(
    *,
    storage: StoragePaths,
    blockbase_name: str,
) -> BlockbaseStore:
    return BlockbaseStore(root=storage.weights / BLOCKBASE_DIRECTORY / blockbase_name)


def build_evaluator(
    *,
    device: Any,
    windows: WindowSettings,
    optimization: OptimizationSettings,
) -> Evaluator:
    return Evaluator(
        device=device,
        batch_size=windows.evaluation_batch_size,
        num_workers=optimization.num_workers,
    )


def build_continual(
    *,
    stream: StreamSettings | None,
    storage: StoragePaths,
    device: Any,
    seed: int,
) -> ContinualCalibrationService | None:
    if stream is None:
        return None
    return ContinualCalibrationService(
        settings=stream,
        store=CheckpointStore(root=storage.weights),
        device=device,
        seed=seed,
    )


def build_download_service(
    *,
    settings: DownloadSettings,
    storage: StoragePaths,
) -> DatasetDownloadService:
    huggingface = HuggingFaceDatasetDownloader()
    return DatasetDownloadService(
        settings=settings,
        downloaders={
            Task.FORECASTING: huggingface,
            Task.IMPUTATION: huggingface,
            Task.ANOMALY_DETECTION: huggingface,
            Task.CLASSIFICATION: UcrArchiveDownloader(retries=settings.retries),
        },
        data_root=storage.data,
    )


def build_pretraining(
    *,
    settings: PretrainSettings,
    storage: StoragePaths,
) -> tuple[BlockPretrainingService, PretrainGrid]:
    device = resolve_device(preferred=settings.device)
    ensure_free_memory(
        device=device,
        minimum_gigabytes=settings.optimization.minimum_free_gigabytes,
    )
    windows = build_window_factory(
        storage=storage,
        split_protocol=settings.split_protocol,
    )
    service = BlockPretrainingService(
        settings=settings,
        windows=windows,
        store=build_blockbase_store(
            storage=storage,
            blockbase_name=settings.blockbase_name,
        ),
        device=device,
    )
    grid = PretrainGrid(
        settings=settings,
        series_length=lambda name: windows.repository.load_classification(name=name).series_length,
    )
    return service, grid


def build_blockbase_index(
    *,
    settings: BlockbaseSettings,
    storage: StoragePaths,
) -> BlockbaseIndexService:
    device = resolve_device(preferred=settings.device)
    store = build_blockbase_store(
        storage=storage,
        blockbase_name=settings.blockbase_name,
    )
    return BlockbaseIndexService(
        settings=settings,
        store=store,
        index_store=BlockbaseIndexStore(root=storage.weights / INDEX_DIRECTORY),
        clusterer=BlockClusterer(
            settings=settings.clustering,
            seed=settings.seed,
        ),
        router_trainer=RouterTrainingService(
            settings=settings.router,
            collector=ResidualCollector(
                store=store,
                windows=build_window_factory(
                    storage=storage,
                    split_protocol=settings.split_protocol,
                ),
                imputation=settings.imputation,
                windows_per_model=settings.router.windows_per_model,
                device=device,
            ),
            device=device,
            seed=settings.seed,
        ),
    )


def build_timeblocks_experiment(
    *,
    settings: ZeroShotSettings,
    storage: StoragePaths,
    results: ResultStore,
):
    from src.services.zero_shot import TimeBlocksExperimentService

    device = resolve_device(preferred=settings.device)
    ensure_free_memory(
        device=device,
        minimum_gigabytes=settings.head.minimum_free_gigabytes,
    )
    return TimeBlocksExperimentService(
        settings=settings,
        windows=build_window_factory(
            storage=storage,
            split_protocol=settings.split_protocol,
        ),
        store=build_blockbase_store(
            storage=storage,
            blockbase_name=settings.blockbase_name,
        ),
        index_store=BlockbaseIndexStore(root=storage.weights / INDEX_DIRECTORY),
        evaluator=build_evaluator(
            device=device,
            windows=settings.windows,
            optimization=settings.head,
        ),
        results=results,
        continual=build_continual(
            stream=settings.stream,
            storage=storage,
            device=device,
            seed=settings.seed,
        ),
        device=device,
    )


def build_baseline_experiment(
    *,
    settings: BaselineSettings,
    storage: StoragePaths,
    results: ResultStore,
) -> BaselineExperimentService:
    device = resolve_device(preferred=settings.device)
    ensure_free_memory(
        device=device,
        minimum_gigabytes=settings.optimization.minimum_free_gigabytes,
    )
    return BaselineExperimentService(
        settings=settings,
        windows=build_window_factory(
            storage=storage,
            split_protocol=settings.split_protocol,
        ),
        factory=BaselineFactory(importer=IsolatedImporter(third_party_root=storage.third_party)),
        checkpoints=CheckpointStore(root=storage.weights),
        evaluator=build_evaluator(
            device=device,
            windows=settings.windows,
            optimization=settings.optimization,
        ),
        results=results,
        continual=build_continual(
            stream=settings.stream,
            storage=storage,
            device=device,
            seed=settings.seed,
        ),
        device=device,
    )


def build_result_store(
    *,
    storage: StoragePaths,
    config_path: Path,
    configs_root: Path,
    resolved: Mapping[str, Any],
    output_dir: Path | None,
) -> ResultStore:
    from src.services.results import run_name_from_config

    if output_dir is not None:
        store = ResultStore(directory=output_dir)
    else:
        store = ResultStore.for_run(
            outputs_root=storage.outputs,
            run_name=run_name_from_config(
                config_path=config_path,
                configs_root=configs_root,
            ),
        )
    store.save_config(values=resolved)
    logger.info("Outputs go to %s", store.directory)
    return store
