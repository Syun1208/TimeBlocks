from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from src.data.classes.series import Batch


@runtime_checkable
class TaskModel(Protocol):
    def training_loss(self, *, batch: Batch) -> torch.Tensor: ...

    def predict(self, *, batch: Batch) -> torch.Tensor: ...


@runtime_checkable
class AnomalyScorer(Protocol):
    def score(self, *, batch: Batch) -> torch.Tensor: ...
