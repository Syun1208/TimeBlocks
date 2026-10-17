from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Subset

from src.data.classes.series import ClassificationSplits, SeriesSplits

OBSERVED = 1.0
MASKED = 0.0


class EmptyWindowsError(ValueError):
    pass


def window_starts(
    *,
    first: int,
    last: int,
    stride: int,
) -> np.ndarray:
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


class UnivariateWindows(Dataset):
    def __init__(
        self,
        *,
        series: SeriesSplits,
        starts: np.ndarray,
    ) -> None:
        self._values = torch.from_numpy(np.ascontiguousarray(series.values))
        self._starts = starts
        self._num_variables = series.num_variables
        self.series_name = series.name

    def __len__(self) -> int:
        return len(self._starts) * self._num_variables

    @property
    def starts(self) -> np.ndarray:
        return self._starts

    @property
    def num_variables(self) -> int:
        return self._num_variables

    def locate(self, *, index: int) -> tuple[int, int]:
        window, variable = divmod(index, self._num_variables)
        return int(self._starts[window]), variable

    def segment(
        self,
        *,
        start: int,
        length: int,
        variable: int,
    ) -> torch.Tensor:
        return self._values[start : start + length, variable : variable + 1]


class ForecastingWindows(UnivariateWindows):
    def __init__(
        self,
        *,
        series: SeriesSplits,
        split: str,
        context_length: int,
        horizon: int,
        stride: int,
        history_length: int = 0,
        time_features: np.ndarray | None = None,
    ) -> None:
        begin, end = series.borders(split=split)
        super().__init__(
            series=series,
            starts=window_starts(
                first=max(begin, context_length),
                last=end - horizon,
                stride=stride,
            ),
        )
        self._context_length = context_length
        self._horizon = horizon
        self._history_length = history_length
        self._time_features = None if time_features is None else torch.from_numpy(time_features)

    def _padded(
        self,
        *,
        start: int,
        length: int,
        variable: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding = max(0, -start)
        values = self.segment(
            start=max(start, 0),
            length=length - padding,
            variable=variable,
        )[:, 0]
        observed = torch.ones(length)
        observed[:padding] = MASKED
        return torch.cat([torch.zeros(padding), values]), observed

    def _time(
        self,
        *,
        start: int,
        length: int,
    ) -> torch.Tensor:
        padding = max(0, -start)
        features = self._time_features[max(start, 0) : start + length]
        return torch.cat([torch.zeros(padding, features.shape[1]), features]).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_start, variable = self.locate(index=index)
        item = {
            "inputs": self.segment(
                start=target_start - self._context_length,
                length=self._context_length,
                variable=variable,
            ),
            "targets": self.segment(
                start=target_start,
                length=self._horizon,
                variable=variable,
            ),
        }
        if self._history_length:
            history_start = target_start - self._context_length - self._history_length
            history_size = self._context_length + self._history_length
            item["history"], item["history_observed"] = self._padded(
                start=history_start,
                length=history_size,
                variable=variable,
            )
            if self._time_features is not None:
                item["history_time"] = self._time(
                    start=history_start,
                    length=history_size,
                )
                item["future_time"] = self._time(
                    start=target_start,
                    length=self._horizon,
                )
        return item


class ReconstructionWindows(UnivariateWindows):
    def __init__(
        self,
        *,
        series: SeriesSplits,
        split: str,
        context_length: int,
        stride: int,
    ) -> None:
        begin, end = series.borders(split=split)
        super().__init__(
            series=series,
            starts=window_starts(
                first=begin,
                last=end - context_length,
                stride=stride,
            ),
        )
        self._context_length = context_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, variable = self.locate(index=index)
        window = self.segment(
            start=start,
            length=self._context_length,
            variable=variable,
        )
        return {"inputs": window, "targets": window}


class ImputationWindows(ReconstructionWindows):
    def __init__(
        self,
        *,
        series: SeriesSplits,
        split: str,
        context_length: int,
        stride: int,
        mask_ratios: Sequence[float],
        seed: int | None,
    ) -> None:
        super().__init__(
            series=series,
            split=split,
            context_length=context_length,
            stride=stride,
        )
        self._mask_ratios = tuple(mask_ratios)
        self._seed = seed

    def _generator(self, *, index: int) -> np.random.Generator:
        if self._seed is None:
            return np.random.default_rng()
        return np.random.default_rng((self._seed, index))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        generator = self._generator(index=index)
        ratio = self._mask_ratios[generator.integers(len(self._mask_ratios))]
        observed = generator.random(item["inputs"].shape) > ratio
        item["mask"] = torch.from_numpy(observed.astype(np.float32))
        return item


class ScoringWindows(UnivariateWindows):
    def __init__(
        self,
        *,
        series: SeriesSplits,
        split: str,
        context_length: int,
    ) -> None:
        begin, end = series.borders(split=split)
        if end - begin < context_length:
            raise EmptyWindowsError(f"{series.name} {split} is shorter than the context length {context_length}")
        starts = list(range(begin, end - context_length + 1, context_length))
        if starts[-1] + context_length < end:
            starts.append(end - context_length)
        super().__init__(
            series=series,
            starts=np.array(starts, dtype=np.int64),
        )
        self._context_length = context_length
        self.begin = begin
        self.end = end

    @property
    def context_length(self) -> int:
        return self._context_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, variable = self.locate(index=index)
        window = self.segment(
            start=start,
            length=self._context_length,
            variable=variable,
        )
        return {"inputs": window, "targets": window}


class ClassificationSamples(Dataset):
    def __init__(
        self,
        *,
        inputs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._inputs = torch.from_numpy(np.ascontiguousarray(inputs[:, :, None]))
        self._labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self._inputs[index], "labels": self._labels[index]}


def classification_samples(
    *,
    splits: ClassificationSplits,
    split: str,
) -> ClassificationSamples:
    inputs = splits.train_inputs if split == "train" else splits.test_inputs
    labels = splits.train_labels if split == "train" else splits.test_labels
    return ClassificationSamples(
        inputs=inputs,
        labels=labels,
    )


def concatenate(*, datasets: Sequence[Dataset]) -> Dataset:
    non_empty = [dataset for dataset in datasets if len(dataset) > 0]
    if not non_empty:
        raise EmptyWindowsError("No windows available for the requested datasets and split")
    return non_empty[0] if len(non_empty) == 1 else ConcatDataset(non_empty)


def evenly_spaced_subset(
    *,
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    if limit is None or len(dataset) <= limit:
        return dataset
    indices = np.linspace(0, len(dataset) - 1, limit).round().astype(np.int64)
    return Subset(dataset, indices.tolist())


def build_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_samples: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    sampler = None
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=min(num_samples, len(dataset)) if num_samples else None,
            generator=generator,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )
