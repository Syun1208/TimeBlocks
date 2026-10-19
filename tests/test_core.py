from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from src.data.classes.enums import Architecture, Task
from src.data.classes.settings import BlockSettings, RouterSettings
from src.models.timeblocks.block import BlockShapeError, TimeBlock
from src.models.timeblocks.router import Router
from src.services.coresets import ReservoirBuffer, StreamCore
from src.services.model_building import build_fresh_blocks, build_task_model
from src.utils.config_loader import ConfigError, build_dataclass
from src.utils.metrics import adjust_predictions, detection_scores, search_best_threshold

SMALL_BLOCK = BlockSettings(
    common_length=64,
    patch_length=4,
    hidden_size=8,
    layers_per_block=2,
    attention_heads=2,
)


@dataclass(frozen=True, slots=True)
class InnerSettings:
    value: int = 1


@dataclass(frozen=True, slots=True)
class OuterSettings:
    name: str
    inner: InnerSettings = field(default_factory=InnerSettings)
    sizes: tuple[int, ...] = ()


@pytest.mark.parametrize("architecture", list(Architecture))
def test_block_keeps_common_length_and_fingerprint_size(architecture: Architecture) -> None:
    block = TimeBlock(
        settings=SMALL_BLOCK,
        architecture=architecture,
        sampling_stride=4,
    )
    output = block(residual=torch.randn(3, SMALL_BLOCK.common_length))
    assert output.shape == (3, SMALL_BLOCK.common_length)
    assert block.fingerprint().numel() == SMALL_BLOCK.patch_length * SMALL_BLOCK.hidden_size


def test_block_rejects_incompatible_stride() -> None:
    with pytest.raises(BlockShapeError):
        TimeBlock(
            settings=SMALL_BLOCK,
            architecture=Architecture.MLP,
            sampling_stride=32,
        )


def test_fresh_blocks_follow_decreasing_power_of_two() -> None:
    blocks = build_fresh_blocks(
        settings=SMALL_BLOCK,
        architecture=Architecture.MLP,
        num_blocks=3,
    )
    assert [block.sampling_stride for block in blocks] == [4, 2, 1]


@pytest.mark.parametrize(
    ("task", "expected"),
    [(Task.FORECASTING, (2, 24, 1)), (Task.IMPUTATION, (2, 40, 1)), (Task.CLASSIFICATION, (2, 5))],
)
def test_task_models_output_shapes(task: Task, expected: tuple[int, ...]) -> None:
    model = build_task_model(
        task=task,
        settings=SMALL_BLOCK,
        blocks=build_fresh_blocks(
            settings=SMALL_BLOCK,
            architecture=Architecture.LSTM,
            num_blocks=2,
        ),
        context_length=40,
        horizon=24,
        num_classes=5,
    )
    output = model.network(
        x=torch.randn(2, 40, 1),
        mask=torch.ones(2, 40, 1) if task is Task.IMPUTATION else None,
    )
    assert tuple(output.shape) == expected


def test_router_outputs_fingerprint_size() -> None:
    router = Router(
        settings=RouterSettings(
            channels=4,
            kernel_size=3,
            pooled_length=8,
        ),
        fingerprint_size=32,
    )
    assert router(residual=torch.randn(5, SMALL_BLOCK.common_length)).shape == (5, 32)


def test_streamcore_keeps_capacity_and_prefers_spread_points() -> None:
    coreset = StreamCore(
        capacity=3,
        generator=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    clustered = torch.tensor([[0.0], [0.01], [0.02]])
    coreset.insert(
        features=clustered,
        indices=torch.arange(3),
    )
    coreset.insert(
        features=torch.tensor([[10.0]]),
        indices=torch.tensor([3]),
    )
    assert len(coreset.selected()) == 3
    assert 3 in coreset.selected()


def test_reservoir_buffer_respects_capacity() -> None:
    buffer = ReservoirBuffer(
        capacity=5,
        generator=np.random.default_rng(0),
    )
    buffer.insert(
        features=torch.zeros(100, 1),
        indices=torch.arange(100),
    )
    assert len(buffer.selected()) == 5


def test_point_adjustment_marks_whole_segment() -> None:
    labels = np.array([0, 1, 1, 1, 0, 1, 1])
    predictions = np.array([0, 0, 1, 0, 0, 0, 0])
    adjusted = adjust_predictions(
        predictions=predictions,
        labels=labels,
    )
    assert adjusted.tolist() == [False, True, True, True, False, False, False]
    f1, precision, recall = detection_scores(
        predictions=predictions,
        labels=labels,
        point_adjustment=True,
    )
    assert precision == 1.0 and recall == pytest.approx(0.6)


def test_threshold_search_finds_perfect_split() -> None:
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.zeros(90), np.ones(10)])
    result = search_best_threshold(
        scores=scores,
        labels=labels,
        point_adjustment=False,
    )
    assert result.f1 == 1.0


def test_config_loader_builds_nested_and_rejects_unknown_keys() -> None:
    settings = build_dataclass(
        settings_type=OuterSettings,
        values={"name": "x", "inner": {"value": 3}, "sizes": [1, 2]},
    )
    assert settings.inner.value == 3 and settings.sizes == (1, 2)
    with pytest.raises(ConfigError):
        build_dataclass(
            settings_type=OuterSettings,
            values={"name": "x", "typo": 1},
        )
    with pytest.raises(ConfigError):
        build_dataclass(
            settings_type=OuterSettings,
            values={},
        )


def test_residual_connection_with_linear_head_shapes() -> None:
    settings = BlockSettings(
        common_length=64,
        patch_length=4,
        hidden_size=8,
        layers_per_block=2,
        attention_heads=2,
        connection="residual",
        head="linear",
    )
    model = build_task_model(
        task=Task.FORECASTING,
        settings=settings,
        blocks=build_fresh_blocks(
            settings=settings,
            architecture=Architecture.MLP,
            num_blocks=2,
        ),
        context_length=40,
        horizon=24,
        num_classes=1,
    )
    model.eval()
    inputs = torch.randn(2, 40, 1)
    assert model.network(x=inputs).shape == (2, 24, 1)
    residuals = model.network.residuals(values=inputs)
    update = model.network.blocks[0](residual=residuals[0])
    assert torch.allclose(residuals[1], residuals[0] + update)


def test_standard_split_protocol_overrides_counts() -> None:
    from src.data.loaders.catalog import STANDARD_SPLIT, UnknownSplitProtocolError, resolve_spec

    paper = resolve_spec(name="ETTh1")
    standard = resolve_spec(
        name="ETTh1",
        split_protocol=STANDARD_SPLIT,
    )
    assert paper.split_counts.train == 10452
    assert standard.split_counts.train == 8640
    with pytest.raises(UnknownSplitProtocolError):
        resolve_spec(
            name="ETTh1",
            split_protocol="unknown",
        )


def test_selector_restricts_each_step_to_its_position() -> None:
    from src.data.classes.blocks import BlockRecord
    from src.data.classes.settings import SelectionSettings
    from src.services.block_selection import BlockSelector

    selector = object.__new__(BlockSelector)
    selector._settings = SelectionSettings(match_position=True)
    selector._blocks = [
        BlockRecord(
            block_id=f"m{index // 2}/b{index % 2 + 1}",
            model_id=f"m{index // 2}",
            position=index % 2 + 1,
            task="forecasting",
            dataset="ETTh1",
            architecture="mlp",
            context_length=96,
            sampling_stride=2 ** (1 - index % 2),
        )
        for index in range(4)
    ]
    pool = selector._step_pool(
        candidates=[0, 1, 2, 3],
        fallback=[0, 1, 2, 3],
        chosen=[0],
        step=2,
    )
    assert pool == [1, 3]
    fallback_pool = selector._step_pool(
        candidates=[0, 2],
        fallback=[0, 1, 2, 3],
        chosen=[0],
        step=2,
    )
    assert fallback_pool == [1, 3]


def test_backtest_completion_uses_the_donor_suffix() -> None:
    from src.data.classes.blocks import BlockRecord
    from src.data.classes.settings import SelectionSettings
    from src.services.block_selection import BlockSelector

    selector = object.__new__(BlockSelector)
    selector._settings = SelectionSettings(num_blocks=2)
    selector._blocks = [
        BlockRecord(
            block_id=f"m{index // 2}/b{index % 2 + 1}",
            model_id=f"m{index // 2}",
            position=index % 2 + 1,
            task="forecasting",
            dataset="ETTh1",
            architecture="mlp",
            context_length=96,
            sampling_stride=2 ** (1 - index % 2),
        )
        for index in range(3)
    ]
    selector._positions = {(block.model_id, block.position): index for index, block in enumerate(selector._blocks)}
    assert selector._completion(chosen=(0,)) == [1]
    assert selector._completion(chosen=(0, 1)) == []
    assert selector._completion(chosen=(2,)) is None
