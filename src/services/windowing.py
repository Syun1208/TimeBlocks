from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch.utils.data import Dataset

from src.data.classes.enums import Task
from src.data.loaders.readers import DatasetRepository
from src.data.loaders.time_features import TimeFeatureEncoder
from src.data.loaders.windows import (
    ForecastingWindows,
    ImputationWindows,
    ReconstructionWindows,
    ScoringWindows,
    classification_samples,
    concatenate,
)


@dataclass(frozen=True, slots=True)
class WindowRequest:
    task: Task
    dataset: str
    split: str
    context_length: int
    horizon: int = 0
    stride: int = 1
    mask_ratios: tuple[float, ...] = ()
    mask_seed: int | None = None
    history_length: int = 0
    time_features: bool = False


class WindowFactory:
    def __init__(
        self,
        *,
        repository: DatasetRepository,
        time_encoder: TimeFeatureEncoder,
    ) -> None:
        self._repository = repository
        self._time_encoder = time_encoder

    @property
    def repository(self) -> DatasetRepository:
        return self._repository

    def _forecasting(self, *, request: WindowRequest) -> Dataset:
        series = self._repository.load_series(name=request.dataset)
        return ForecastingWindows(
            series=series,
            split=request.split,
            context_length=request.context_length,
            horizon=request.horizon,
            stride=request.stride,
            history_length=request.history_length,
            time_features=self._time_encoder.encode(series=series) if request.time_features else None,
        )

    def _imputation(self, *, request: WindowRequest) -> Dataset:
        return ImputationWindows(
            series=self._repository.load_series(name=request.dataset),
            split=request.split,
            context_length=request.context_length,
            stride=request.stride,
            mask_ratios=request.mask_ratios,
            seed=request.mask_seed,
        )

    def _reconstruction(self, *, request: WindowRequest) -> Dataset:
        return ReconstructionWindows(
            series=self._repository.load_series(name=request.dataset),
            split=request.split,
            context_length=request.context_length,
            stride=request.stride,
        )

    def _classification(self, *, request: WindowRequest) -> Dataset:
        return classification_samples(
            splits=self._repository.load_classification(name=request.dataset),
            split=request.split,
        )

    def build(self, *, request: WindowRequest) -> Dataset:
        builders = {
            Task.FORECASTING: self._forecasting,
            Task.IMPUTATION: self._imputation,
            Task.ANOMALY_DETECTION: self._reconstruction,
            Task.CLASSIFICATION: self._classification,
        }
        return builders[request.task](request=request)

    def build_many(self, *, requests: Sequence[WindowRequest]) -> Dataset:
        return concatenate(datasets=[self.build(request=request) for request in requests])

    def scoring(
        self,
        *,
        dataset: str,
        split: str,
        context_length: int,
    ) -> ScoringWindows:
        return ScoringWindows(
            series=self._repository.load_series(name=dataset),
            split=split,
            context_length=context_length,
        )
