from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class SeriesSplits:
    name: str
    values: np.ndarray
    train_end: int
    validation_end: int
    labels: np.ndarray | None = None
    timestamps: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_variables(self) -> int:
        return int(self.values.shape[1])

    def borders(self, *, split: str) -> tuple[int, int]:
        mapping = {
            "train": (0, self.train_end),
            "validation": (self.train_end, self.validation_end),
            "test": (self.validation_end, self.length),
        }
        return mapping[split]


@dataclass(frozen=True, slots=True)
class ClassificationSplits:
    name: str
    train_inputs: np.ndarray
    train_labels: np.ndarray
    test_inputs: np.ndarray
    test_labels: np.ndarray
    class_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def series_length(self) -> int:
        return int(self.train_inputs.shape[1])


@dataclass(slots=True)
class Batch:
    inputs: "torch.Tensor"
    targets: "torch.Tensor | None" = None
    mask: "torch.Tensor | None" = None
    labels: "torch.Tensor | None" = None
    history: "torch.Tensor | None" = None
    history_observed: "torch.Tensor | None" = None
    history_time: "torch.Tensor | None" = None
    future_time: "torch.Tensor | None" = None

    @classmethod
    def from_mapping(
        cls,
        *,
        items: dict,
        device: "torch.device",
    ) -> "Batch":
        names = {item.name for item in fields(cls)}
        return cls(
            **{name: value.to(device=device, non_blocking=True) for name, value in items.items() if name in names}
        )

    @property
    def size(self) -> int:
        return int(self.inputs.shape[0])
