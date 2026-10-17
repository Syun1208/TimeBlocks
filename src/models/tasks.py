from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.data.classes.enums import Task
from src.data.classes.series import Batch

MINIMUM_MASKED = 1.0


class ForecastingTask(nn.Module):
    def __init__(self, *, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def predict(self, *, batch: Batch) -> torch.Tensor:
        return self.network(x=batch.inputs)

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        return F.mse_loss(
            input=self.predict(batch=batch),
            target=batch.targets,
        )


class ImputationTask(nn.Module):
    def __init__(self, *, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def predict(self, *, batch: Batch) -> torch.Tensor:
        return self.network(
            x=batch.inputs * batch.mask,
            mask=batch.mask,
        )

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        missing = 1.0 - batch.mask
        squared = (self.predict(batch=batch) - batch.targets).pow(2) * missing
        return squared.sum() / missing.sum().clamp_min(MINIMUM_MASKED)


class ReconstructionTask(nn.Module):
    def __init__(self, *, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def predict(self, *, batch: Batch) -> torch.Tensor:
        return self.network(x=batch.inputs)

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        return F.mse_loss(
            input=self.predict(batch=batch),
            target=batch.inputs,
        )

    def score(self, *, batch: Batch) -> torch.Tensor:
        return (self.predict(batch=batch) - batch.inputs).pow(2).mean(dim=-1)


class ClassificationTask(nn.Module):
    def __init__(self, *, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def predict(self, *, batch: Batch) -> torch.Tensor:
        return self.network(x=batch.inputs)

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        return F.cross_entropy(
            input=self.predict(batch=batch),
            target=batch.labels,
        )


TASK_WRAPPERS: dict[Task, type[nn.Module]] = {
    Task.FORECASTING: ForecastingTask,
    Task.IMPUTATION: ImputationTask,
    Task.ANOMALY_DETECTION: ReconstructionTask,
    Task.CLASSIFICATION: ClassificationTask,
}


def wrap_network(
    *,
    task: Task,
    network: nn.Module,
) -> nn.Module:
    return TASK_WRAPPERS[task](network=network)
