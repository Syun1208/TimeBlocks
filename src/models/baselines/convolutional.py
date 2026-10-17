from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

INCEPTION_DEFAULTS: dict[str, Any] = {
    "filters": 32,
    "depth": 6,
    "kernel_sizes": [39, 19, 9],
    "bottleneck": 32,
    "residual_every": 3,
}
RESNET_DEFAULTS: dict[str, Any] = {
    "filters": [64, 128, 128],
    "kernel_sizes": [7, 5, 3],
}
POOL_SIZE = 3
INCEPTION_BRANCHES = 4


class InceptionModule(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        filters: int,
        kernel_sizes: Sequence[int],
        bottleneck: int,
    ) -> None:
        super().__init__()
        use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck,
                kernel_size=1,
                bias=False,
            )
            if use_bottleneck
            else nn.Identity()
        )
        branch_input = bottleneck if use_bottleneck else in_channels
        self.branches = nn.ModuleList(
            nn.Conv1d(
                in_channels=branch_input,
                out_channels=filters,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in kernel_sizes
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=POOL_SIZE,
                stride=1,
                padding=1,
            ),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=1,
                bias=False,
            ),
        )
        self.normalize = nn.Sequential(
            nn.BatchNorm1d(num_features=filters * (len(kernel_sizes) + 1)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(x)
        outputs = [branch(reduced) for branch in self.branches] + [self.pool(x)]
        return self.normalize(torch.cat(outputs, dim=1))


class InceptionTime(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        super().__init__()
        values = {**INCEPTION_DEFAULTS, **hyperparameters}
        width = values["filters"] * (len(values["kernel_sizes"]) + 1)
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        channels, residual_channels = 1, 1
        for depth in range(values["depth"]):
            self.modules_list.append(
                InceptionModule(
                    in_channels=channels,
                    filters=values["filters"],
                    kernel_sizes=values["kernel_sizes"],
                    bottleneck=values["bottleneck"],
                )
            )
            channels = width
            if (depth + 1) % values["residual_every"] == 0:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=residual_channels,
                            out_channels=width,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.BatchNorm1d(num_features=width),
                    )
                )
                residual_channels = width
        self.residual_every = values["residual_every"]
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(
            in_features=width,
            out_features=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = x.transpose(1, 2)
        residual = features
        for depth, module in enumerate(self.modules_list):
            features = module(features)
            if (depth + 1) % self.residual_every == 0:
                shortcut = self.shortcuts[(depth + 1) // self.residual_every - 1]
                features = self.activation(features + shortcut(residual))
                residual = features
        return self.classifier(features.mean(dim=-1))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for index, kernel in enumerate(kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.BatchNorm1d(num_features=out_channels),
                ]
            )
            if index < len(kernel_sizes) - 1:
                layers.append(nn.ReLU())
            channels = out_channels
        self.body = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        super().__init__()
        values = {**RESNET_DEFAULTS, **hyperparameters}
        blocks = []
        channels = 1
        for filters in values["filters"]:
            blocks.append(
                ResidualBlock(
                    in_channels=channels,
                    out_channels=filters,
                    kernel_sizes=values["kernel_sizes"],
                )
            )
            channels = filters
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(
            in_features=channels,
            out_features=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.blocks(x.transpose(1, 2)).mean(dim=-1))
