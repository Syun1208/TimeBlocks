from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.data.classes.blocks import BlockRecord, ClusterIndex, SelectionResult
from src.data.classes.enums import Split, Task
from src.data.classes.settings import BlockSettings, SelectionSettings
from src.data.loaders.windows import evenly_spaced_subset
from src.models.timeblocks.block import TimeBlock
from src.models.timeblocks.block_model import RESIDUAL_CONNECTION
from src.models.timeblocks.router import Router
from src.models.timeblocks.standardization import InputStandardization, InstanceStatistics, denormalize
from src.services.blockbase_store import BlockbaseStore
from src.services.model_building import LINEAR_HEAD, build_head
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import SELECT, finish, progress

logger = logging.getLogger(__name__)

WindowSampler = Callable[..., torch.Tensor]
BACKTEST_SCORE = "backtest"
SINGLE_CLASS = 1


class MissingHorizonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PartialPath:
    chosen: tuple[int, ...]
    residual: torch.Tensor
    distances: tuple[float, ...]
    score: float


@dataclass(frozen=True, slots=True)
class BacktestWindows:
    residual: torch.Tensor
    statistics: InstanceStatistics
    targets: torch.Tensor
    context_length: int
    horizon: int


class SelectionWindowSampler:
    def __init__(
        self,
        *,
        windows: WindowFactory,
        task: Task,
        dataset: str,
        settings: SelectionSettings,
        backtest_split: str | None = None,
    ) -> None:
        self._windows = windows
        self._task = task
        self._dataset = dataset
        self._settings = settings
        self._backtest_split = settings.backtest_split if backtest_split is None else backtest_split

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

    def backtest(
        self,
        *,
        context_length: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subset = evenly_spaced_subset(
            dataset=self._windows.build(
                request=WindowRequest(
                    task=self._task,
                    dataset=self._dataset,
                    split=self._backtest_split,
                    context_length=context_length,
                    horizon=horizon,
                )
            ),
            limit=self._settings.backtest_windows,
        )
        samples = [subset[index] for index in range(len(subset))]
        return (
            torch.stack([sample["inputs"] for sample in samples]),
            torch.stack([sample["targets"] for sample in samples]),
        )


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
        self._block_settings = block_settings
        self._device = device
        self._horizons: dict[str, int] = {}
        self._modules: dict[int, TimeBlock] = {}
        self._heads: dict[tuple[str, int], torch.nn.Module] = {}
        self._positions = {(block.model_id, block.position): index for index, block in enumerate(self._blocks)}
        self._residual_connection = block_settings.connection == RESIDUAL_CONNECTION
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

    def _horizon_of(self, *, index: int) -> int:
        model_id = self._blocks[index].model_id
        if model_id not in self._horizons:
            self._horizons[model_id] = self._store.record(model_id=model_id).horizon
        return self._horizons[model_id]

    def _with_horizon(
        self,
        *,
        pool: list[int],
        horizon: int | None,
        context_length: int | None,
    ) -> list[int]:
        if horizon is not None:
            pool = [index for index in pool if self._horizon_of(index=index) == horizon]
        if context_length is not None:
            pool = [index for index in pool if self._blocks[index].context_length == context_length]
        return pool

    def _candidates(
        self,
        *,
        cluster: int,
        horizon: int | None,
    ) -> tuple[list[int], list[int]]:
        context_length = self._clusters.context_lengths[cluster] if self._settings.match_context else None
        everything = self._with_horizon(
            pool=list(range(len(self._blocks))),
            horizon=horizon,
            context_length=context_length,
        )
        if not self._settings.restrict_to_cluster:
            return everything, everything
        members = self._with_horizon(
            pool=self._clusters.members(cluster=cluster),
            horizon=horizon,
            context_length=context_length,
        )
        if not members:
            logger.warning("Cluster %d has no block trained for horizon %s, searching the whole pool", cluster, horizon)
            return everything, everything
        return members, everything

    def _step_pool(
        self,
        *,
        candidates: list[int],
        fallback: list[int],
        chosen: list[int],
        step: int,
    ) -> list[int]:
        remaining = [index for index in candidates if index not in chosen]
        if not self._settings.match_position:
            return remaining
        for pool in (candidates, fallback):
            same_position = [index for index in pool if index not in chosen and self._blocks[index].position == step]
            if same_position:
                return same_position
        logger.warning("No block at position %d in the Blockbase, using every position", step)
        return remaining

    def _module(self, *, index: int) -> TimeBlock:
        if index not in self._modules:
            self._modules[index] = self._store.load_block(block=self._blocks[index]).to(self._device).eval()
        return self._modules[index]

    def _advance(
        self,
        *,
        residual: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        update = self._module(index=index)(residual=residual)
        return residual + update if self._residual_connection else update

    def _head(
        self,
        *,
        model_id: str,
        windows: BacktestWindows,
    ) -> torch.nn.Module:
        key = (model_id, windows.horizon)
        if key not in self._heads:
            head = build_head(
                task=Task.FORECASTING,
                settings=self._block_settings,
                context_length=windows.context_length,
                horizon=windows.horizon,
                num_classes=SINGLE_CLASS,
            )
            if self._block_settings.head == LINEAR_HEAD:
                head.load_state_dict(self._store.head_state(model_id=model_id))
            self._heads[key] = head.to(self._device).eval()
        return self._heads[key]

    def _completion(self, *, chosen: tuple[int, ...]) -> list[int] | None:
        donor = self._blocks[chosen[-1]].model_id
        suffix = [
            self._positions.get((donor, position)) for position in range(len(chosen) + 1, self._settings.num_blocks + 1)
        ]
        return None if None in suffix else suffix

    def _score(
        self,
        *,
        chosen: tuple[int, ...],
        residual: torch.Tensor,
        windows: BacktestWindows,
    ) -> float:
        suffix = self._completion(chosen=chosen)
        if suffix is None:
            return math.inf
        for index in suffix:
            residual = self._advance(
                residual=residual,
                index=index,
            )
        head = self._head(
            model_id=self._blocks[chosen[-1]].model_id,
            windows=windows,
        )
        output = head(residual=residual)
        if head.denormalizes:
            output = denormalize(
                values=output,
                statistics=windows.statistics,
            ).unsqueeze(-1)
        return float((output - windows.targets).pow(2).mean())

    def _backtest_windows(
        self,
        *,
        sampler: SelectionWindowSampler,
        context_length: int,
        horizon: int,
    ) -> BacktestWindows:
        inputs, targets = sampler.backtest(
            context_length=context_length,
            horizon=horizon,
        )
        residual, statistics = self._standardization(values=inputs[..., 0].to(self._device))
        return BacktestWindows(
            residual=residual,
            statistics=statistics,
            targets=targets.to(self._device),
            context_length=context_length,
            horizon=horizon,
        )

    def _expand(
        self,
        *,
        beam: PartialPath,
        candidates: list[int],
        fallback: list[int],
        windows: BacktestWindows,
    ) -> list[PartialPath]:
        remaining = self._step_pool(
            candidates=candidates,
            fallback=fallback,
            chosen=list(beam.chosen),
            step=len(beam.chosen) + 1,
        )
        if not remaining:
            return []
        distances = cosine_distances(
            estimate=self._estimate(residual=beam.residual),
            candidates=self._fingerprints[remaining],
        )
        top_k = len(remaining) if not beam.chosen else self._settings.top_k
        expansions = []
        for position in torch.argsort(distances)[:top_k].tolist():
            chosen = beam.chosen + (remaining[position],)
            residual = self._advance(
                residual=beam.residual,
                index=remaining[position],
            )
            expansions.append(
                PartialPath(
                    chosen=chosen,
                    residual=residual,
                    distances=beam.distances + (float(distances[position]),),
                    score=self._score(
                        chosen=chosen,
                        residual=residual,
                        windows=windows,
                    ),
                )
            )
        return expansions

    def _beam_search(
        self,
        *,
        cluster: int,
        sampler: SelectionWindowSampler,
        horizon: int,
    ) -> PartialPath:
        windows = self._backtest_windows(
            sampler=sampler,
            context_length=self._clusters.context_lengths[cluster],
            horizon=horizon,
        )
        candidates, fallback = self._candidates(
            cluster=cluster,
            horizon=horizon,
        )
        beams = [
            PartialPath(
                chosen=(),
                residual=windows.residual,
                distances=(),
                score=math.inf,
            )
        ]
        for _ in range(self._settings.num_blocks):
            expansions = [
                expansion
                for beam in beams
                for expansion in self._expand(
                    beam=beam,
                    candidates=candidates,
                    fallback=fallback,
                    windows=windows,
                )
            ]
            if not expansions:
                break
            beams = sorted(expansions, key=lambda path: path.score)[: self._settings.beam_width]
        return min(beams, key=lambda path: path.score)

    def _select_by_backtest(
        self,
        *,
        sampler: SelectionWindowSampler,
        horizon: int | None,
    ) -> tuple[SelectionResult, list[TimeBlock]]:
        if horizon is None:
            raise MissingHorizonError("Backtest scoring needs a forecasting horizon, set selection.match_horizon")
        started = time.perf_counter()
        routed, cluster_distances = self._choose_cluster(sampler=sampler)
        clusters = range(self._clusters.num_clusters) if self._settings.search_contexts else [routed]
        paths = {}
        bar = progress(
            stage=SELECT,
            total=len(clusters),
            unit="cluster",
            leave=False,
        )
        for cluster in clusters:
            paths[cluster] = self._beam_search(
                cluster=cluster,
                sampler=sampler,
                horizon=horizon,
            )
            bar.update(1)
            bar.set_postfix_str(f"context {self._clusters.context_lengths[cluster]} mse {paths[cluster].score:.4f}")
        finish(bar=bar)
        cluster = min(paths, key=lambda key: paths[key].score)
        best = paths[cluster]
        modules = [self._module(index=index).cpu() for index in best.chosen]
        self._modules.clear()
        self._heads.clear()
        result = SelectionResult(
            cluster=cluster,
            context_length=self._clusters.context_lengths[cluster],
            block_ids=tuple(self._blocks[index].block_id for index in best.chosen),
            cluster_distances=tuple(cluster_distances),
            block_distances=best.distances,
            seconds=time.perf_counter() - started,
            backtest_mse=best.score,
        )
        logger.info(
            "Selected cluster %d (context %d) and blocks %s with backtest MSE %.4f in %.2fs",
            cluster,
            result.context_length,
            list(result.block_ids),
            best.score,
            result.seconds,
        )
        return result, modules

    @torch.no_grad()
    def select(
        self,
        *,
        sampler: WindowSampler,
        horizon: int | None = None,
    ) -> tuple[SelectionResult, list[TimeBlock]]:
        if self._settings.score == BACKTEST_SCORE:
            return self._select_by_backtest(
                sampler=sampler,
                horizon=horizon,
            )
        started = time.perf_counter()
        cluster, cluster_distances = self._choose_cluster(sampler=sampler)
        context_length = self._clusters.context_lengths[cluster]
        residual = self._standardize(windows=sampler(context_length=context_length))
        candidates, fallback = self._candidates(
            cluster=cluster,
            horizon=horizon,
        )
        chosen: list[int] = []
        modules: list[TimeBlock] = []
        block_distances: list[float] = []
        bar = progress(
            stage=SELECT,
            total=self._settings.num_blocks,
            unit="block",
            leave=False,
        )
        for step in range(1, self._settings.num_blocks + 1):
            if modules:
                update = modules[-1](residual=residual)
                residual = residual + update if self._residual_connection else update
            remaining = self._step_pool(
                candidates=candidates,
                fallback=fallback,
                chosen=chosen,
                step=step,
            )
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
