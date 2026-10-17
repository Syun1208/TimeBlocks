from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.data.classes.settings import BlockSettings
from src.models.timeblocks.block import TimeBlock
from src.models.timeblocks.standardization import InputStandardization, InstanceStatistics, denormalize


class BlockModel(nn.Module):
    def __init__(
        self,
        *,
        settings: BlockSettings,
        blocks: Sequence[TimeBlock],
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.standardization = InputStandardization(
            common_length=settings.common_length,
            epsilon=settings.normalization_epsilon,
        )
        self.blocks = nn.ModuleList(blocks)
        self.head = head

    def _prepare(
        self,
        *,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, InstanceStatistics]:
        return self.standardization(
            values=values[..., 0],
            mask=None if mask is None else mask[..., 0],
        )

    def residuals(
        self,
        *,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        depth: int | None = None,
    ) -> list[torch.Tensor]:
        residual, _ = self._prepare(
            values=values,
            mask=mask,
        )
        outputs = [residual]
        for block in self.blocks[: len(self.blocks) if depth is None else depth]:
            residual = block(residual=residual)
            outputs.append(residual)
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, statistics = self._prepare(
            values=x,
            mask=mask,
        )
        for block in self.blocks:
            residual = block(residual=residual)
        output = self.head(residual=residual)
        if not self.head.denormalizes:
            return output
        return denormalize(
            values=output,
            statistics=statistics,
        ).unsqueeze(-1)

    def freeze_all_but_last_block(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.blocks[-1].parameters():
            parameter.requires_grad = True
