from __future__ import annotations

import logging
from pathlib import Path

from src.data.classes.settings import BlockbaseSettings
from src.services.blockbase_store import BlockbaseIndexStore, BlockbaseStore
from src.services.clustering import BlockClusterer
from src.services.router_training import RouterTrainingService, fingerprint_matrix
from src.utils.runtime import set_seed

logger = logging.getLogger(__name__)


class BlockbaseIndexService:
    def __init__(
        self,
        *,
        settings: BlockbaseSettings,
        store: BlockbaseStore,
        index_store: BlockbaseIndexStore,
        clusterer: BlockClusterer,
        router_trainer: RouterTrainingService,
    ) -> None:
        self._settings = settings
        self._store = store
        self._index_store = index_store
        self._clusterer = clusterer
        self._router_trainer = router_trainer

    def build(self, *, excluded_dataset: str | None) -> Path:
        set_seed(seed=self._settings.seed)
        records = self._store.records(
            tasks=self._settings.pool_tasks,
            excluded_datasets=() if excluded_dataset is None else (excluded_dataset,),
        )
        if not records:
            raise ValueError(f"No pretrained models for tasks {self._settings.pool_tasks} without {excluded_dataset}")
        fingerprints, blocks = fingerprint_matrix(
            store=self._store,
            records=records,
        )
        logger.info(
            "Blockbase without %s: %d models, %d blocks, fingerprint size %d",
            excluded_dataset,
            len(records),
            len(blocks),
            fingerprints.shape[1],
        )
        clusters = self._clusterer.cluster(
            fingerprints=fingerprints,
            context_lengths=[block.context_length for block in blocks],
        )
        router = self._router_trainer.train(
            records=records,
            fingerprint_size=fingerprints.shape[1],
        )
        directory = self._index_store.directory(
            blockbase_name=self._settings.blockbase_name,
            pool_tasks=self._settings.pool_tasks,
            excluded_dataset=excluded_dataset,
        )
        self._index_store.save(
            directory=directory,
            router=router.cpu(),
            router_settings=self._settings.router,
            fingerprints=fingerprints,
            blocks=blocks,
            clusters=clusters,
        )
        self._store.release()
        logger.info("Saved blockbase index to %s", directory)
        return directory

    def run(self) -> list[Path]:
        if not self._settings.target_datasets:
            return [self.build(excluded_dataset=None)]
        return [self.build(excluded_dataset=target) for target in self._settings.target_datasets]
