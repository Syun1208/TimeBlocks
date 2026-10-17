from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data.classes.blocks import BlockRecord, ClusterIndex, SelectionResult
from src.data.classes.enums import Split, Task
from src.data.classes.settings import BlockSettings, SelectionSettings
from src.data.loaders.windows import evenly_spaced_subset
from src.models.timeblocks.block import TimeBlock
from src.models.timeblocks.router import Router
from src.models.timeblocks.standardization import InputStandardization
from src.services.blockbase_store import BlockbaseStore
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import SELECT, finish, progress

logger = logging.getLogger(__name__)

WindowSampler = Callable[..., torch.Tensor]


class SelectionWindowSampler:
    def __init__(
        self,
        *,
        windows: WindowFactory,
        task: Task,
        dataset: str,
        settings: SelectionSettings,
    ) -> None:
        self._windows = windows
        self._task = task
        self._dataset = dataset
        self._settings = settings

    def __call__(self, *, context_length: int) -> torch.Tensor:
        request = WindowRequest(
            task=Task.CLASSIFICATION if self._task is Task.CLASSIFICATION else Task.ANOMALY_DETECTION,
            dataset=self._dataset,
            split=self._settings.split if self._task is not Task.CLASSIFICATION else Split.TRAIN.value,
            context_length=context_length,
        )
        subset = evenly_spaced_subset(
            dataset=self._windows.build(request=request),
            limit=self._settings.windows,
        )
        return torch.stack([subset[index]["inputs"] for index in range(len(subset))])


def cosine_distances(
    *,
    estimate: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(estimate.unsqueeze(0), candidates, dim=-1)


class BlockSelector:
    def __init__(
        self,
        *,
        router: Router,
        store: BlockbaseStore,
        fingerprints: np.ndarray,
        blocks: Sequence[BlockRecord],
        clusters: ClusterIndex,
        block_settings: BlockSettings,
        settings: SelectionSettings,
        device: torch.device,
    ) -> None:
        self._router = router.to(device).eval()
        self._store = store
        self._fingerprints = torch.from_numpy(fingerprints).to(device)
        self._blocks = list(blocks)
        self._clusters = clusters
        self._settings = settings
        self._device = device
        self._standardization = InputStandardization(
            common_length=block_settings.common_length,
            epsilon=block_settings.normalization_epsilon,
        )

    def _standardize(self, *, windows: torch.Tensor) -> torch.Tensor:
        residual, _ = self._standardization(values=windows[..., 0].to(self._device))
        return residual

    def _estimate(self, *, residual: torch.Tensor) -> torch.Tensor:
        return self._router(residual=residual).mean(dim=0)

    def _choose_cluster(self, *, sampler: WindowSampler) -> tuple[int, list[float]]:
        distances = []
        for cluster in range(self._clusters.num_clusters):
            residual = self._standardize(windows=sampler(context_length=self._clusters.context_lengths[cluster]))
            representative = self._fingerprints[self._clusters.representatives[cluster]].unsqueeze(0)
            distances.append(
                float(
                    cosine_distances(
                        estimate=self._estimate(residual=residual),
                        candidates=representative,
                    )[0]
                )
            )
        return int(np.argmin(distances)), distances

    def _candidates(self, *, cluster: int) -> list[int]:
        if self._settings.restrict_to_cluster:
            return self._clusters.members(cluster=cluster)
        return list(range(len(self._blocks)))

    @torch.no_grad()
    def select(self, *, sampler: WindowSampler) -> tuple[SelectionResult, list[TimeBlock]]:
        started = time.perf_counter()
        cluster, cluster_distances = self._choose_cluster(sampler=sampler)
        context_length = self._clusters.context_lengths[cluster]
        residual = self._standardize(windows=sampler(context_length=context_length))
        candidates = self._candidates(cluster=cluster)
        chosen: list[int] = []
        modules: list[TimeBlock] = []
        block_distances: list[float] = []
        bar = progress(
            stage=SELECT,
            total=self._settings.num_blocks,
            unit="block",
            leave=False,
        )
        for _ in range(self._settings.num_blocks):
            if modules:
                residual = modules[-1](residual=residual)
            remaining = [index for index in candidates if index not in chosen]
            if not remaining:
                logger.warning("Cluster %d has only %d blocks, stopping early", cluster, len(chosen))
                break
            distances = cosine_distances(
                estimate=self._estimate(residual=residual),
                candidates=self._fingerprints[remaining],
            )
            best = remaining[int(torch.argmin(distances))]
            chosen.append(best)
            block_distances.append(float(distances.min()))
            modules.append(self._store.load_block(block=self._blocks[best]).to(self._device).eval())
            bar.update(1)
            bar.set_postfix_str(self._blocks[best].block_id)
        finish(bar=bar)
        result = SelectionResult(
            cluster=cluster,
            context_length=context_length,
            block_ids=tuple(self._blocks[index].block_id for index in chosen),
            cluster_distances=tuple(cluster_distances),
            block_distances=tuple(block_distances),
            seconds=time.perf_counter() - started,
        )
        logger.info(
            "Selected cluster %d (context %d) and blocks %s in %.2fs",
            cluster,
            context_length,
            list(result.block_ids),
            result.seconds,
        )
        return result, [module.cpu() for module in modules]
