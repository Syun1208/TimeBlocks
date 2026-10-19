from __future__ import annotations

import torch
from torch import nn

from src.models.timeblocks.standardization import resample


class ResampleHead(nn.Module):
    denormalizes = True

    def __init__(self, *, output_length: int) -> None:
        super().__init__()
        self.output_length = output_length

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return resample(
            values=residual,
            length=self.output_length,
        )


class LinearForecastHead(nn.Module):
    denormalizes = True

    def __init__(
        self,
        *,
        common_length: int,
        output_length: int,
    ) -> None:
        super().__init__()
        self.output_length = output_length
        self.projection = nn.Linear(
            in_features=common_length,
            out_features=output_length,
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.projection(residual)


class ClassificationHead(nn.Module):
    denormalizes = False

    def __init__(
        self,
        *,
        common_length: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(
            in_features=common_length,
            out_features=num_classes,
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.projection(residual)
