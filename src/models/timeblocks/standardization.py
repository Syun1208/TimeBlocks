from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class InstanceStatistics:
    mean: torch.Tensor
    deviation: torch.Tensor


def instance_normalize(
    *,
    values: torch.Tensor,
    mask: torch.Tensor | None,
    epsilon: float,
) -> tuple[torch.Tensor, InstanceStatistics]:
    if mask is None:
        mean = values.mean(dim=1, keepdim=True)
        deviation = torch.sqrt(values.var(dim=1, keepdim=True, unbiased=False) + epsilon)
        return (values - mean) / deviation, InstanceStatistics(
            mean=mean,
            deviation=deviation,
        )
    observed = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=1, keepdim=True) / observed
    centred = (values - mean) * mask
    deviation = torch.sqrt(centred.pow(2).sum(dim=1, keepdim=True) / observed + epsilon)
    return centred / deviation, InstanceStatistics(
        mean=mean,
        deviation=deviation,
    )


def denormalize(
    *,
    values: torch.Tensor,
    statistics: InstanceStatistics,
) -> torch.Tensor:
    return values * statistics.deviation + statistics.mean


def resample(
    *,
    values: torch.Tensor,
    length: int,
) -> torch.Tensor:
    if values.shape[-1] == length:
        return values
    return F.interpolate(
        values.unsqueeze(1),
        size=length,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


class InputStandardization(nn.Module):
    def __init__(
        self,
        *,
        common_length: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.common_length = common_length
        self.epsilon = epsilon

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, InstanceStatistics]:
        normalized, statistics = instance_normalize(
            values=values,
            mask=mask,
            epsilon=self.epsilon,
        )
        return (
            resample(
                values=normalized,
                length=self.common_length,
            ),
            statistics,
        )
