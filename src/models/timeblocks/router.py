from __future__ import annotations

import torch
from torch import nn

from src.data.classes.settings import RouterSettings

CONVOLUTION_STRIDE = 2


class Router(nn.Module):
    def __init__(
        self,
        *,
        settings: RouterSettings,
        fingerprint_size: int,
    ) -> None:
        super().__init__()
        padding = settings.kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=settings.channels,
                kernel_size=settings.kernel_size,
                stride=CONVOLUTION_STRIDE,
                padding=padding,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=settings.channels,
                out_channels=settings.channels,
                kernel_size=settings.kernel_size,
                stride=CONVOLUTION_STRIDE,
                padding=padding,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size=settings.pooled_length),
        )
        self.projection = nn.Linear(
            in_features=settings.channels * settings.pooled_length,
            out_features=fingerprint_size,
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        features = self.features(residual.unsqueeze(1))
        return self.projection(features.flatten(start_dim=1))
