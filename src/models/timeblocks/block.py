from __future__ import annotations

import torch
from torch import nn

from src.data.classes.enums import Architecture
from src.data.classes.settings import BlockSettings
from src.models.timeblocks.bodies import build_body

MINIMUM_BLOCK_LAYERS = 2


class BlockShapeError(ValueError):
    pass


class TimeBlock(nn.Module):
    def __init__(
        self,
        *,
        settings: BlockSettings,
        architecture: Architecture,
        sampling_stride: int,
    ) -> None:
        super().__init__()
        if settings.common_length % (sampling_stride * settings.patch_length):
            raise BlockShapeError(
                f"common_length={settings.common_length} must be divisible by "
                f"sampling_stride*patch_length={sampling_stride * settings.patch_length}"
            )
        if settings.layers_per_block < MINIMUM_BLOCK_LAYERS:
            raise BlockShapeError(f"layers_per_block must be at least {MINIMUM_BLOCK_LAYERS}")
        self.architecture = architecture
        self.sampling_stride = sampling_stride
        self.patch_length = settings.patch_length
        self.num_tokens = settings.common_length // sampling_stride // settings.patch_length
        self.first_layer = nn.Linear(
            in_features=settings.patch_length,
            out_features=settings.hidden_size,
        )
        self.activation = nn.GELU()
        self.body = build_body(
            architecture=architecture,
            settings=settings,
            num_layers=settings.layers_per_block - 1,
            num_tokens=self.num_tokens,
        )
        self.output_standardization = nn.Linear(
            in_features=self.num_tokens * settings.hidden_size,
            out_features=settings.common_length,
        )

    def scale(self, *, residual: torch.Tensor) -> torch.Tensor:
        return residual[:, self.sampling_stride - 1 :: self.sampling_stride]

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        sampled = self.scale(residual=residual)
        patches = sampled.reshape(sampled.shape[0], self.num_tokens, self.patch_length)
        tokens = self.activation(self.first_layer(patches))
        tokens = self.body(tokens=tokens)
        return self.output_standardization(tokens.flatten(start_dim=1))

    def fingerprint(self) -> torch.Tensor:
        return self.first_layer.weight.detach().flatten()
