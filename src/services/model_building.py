from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from src.data.classes.enums import Architecture, Task
from src.data.classes.settings import BlockSettings
from src.models.tasks import wrap_network
from src.models.timeblocks.block import TimeBlock
from src.models.timeblocks.block_model import BlockModel
from src.models.timeblocks.heads import ClassificationHead, ResampleHead


def build_head(
    *,
    task: Task,
    settings: BlockSettings,
    context_length: int,
    horizon: int,
    num_classes: int,
) -> nn.Module:
    if task is Task.CLASSIFICATION:
        return ClassificationHead(
            common_length=settings.common_length,
            num_classes=num_classes,
        )
    output_length = horizon if task is Task.FORECASTING else context_length
    return ResampleHead(output_length=output_length)


def build_fresh_blocks(
    *,
    settings: BlockSettings,
    architecture: Architecture,
    num_blocks: int,
) -> list[TimeBlock]:
    return [
        TimeBlock(
            settings=settings,
            architecture=architecture,
            sampling_stride=2 ** (num_blocks - position),
        )
        for position in range(1, num_blocks + 1)
    ]


def build_task_model(
    *,
    task: Task,
    settings: BlockSettings,
    blocks: Sequence[TimeBlock],
    context_length: int,
    horizon: int,
    num_classes: int,
) -> nn.Module:
    network = BlockModel(
        settings=settings,
        blocks=blocks,
        head=build_head(
            task=task,
            settings=settings,
            context_length=context_length,
            horizon=horizon,
            num_classes=num_classes,
        ),
    )
    return wrap_network(
        task=task,
        network=network,
    )
