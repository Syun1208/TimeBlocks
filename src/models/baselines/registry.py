from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from torch import nn

from src.data.classes.enums import Task
from src.models.baselines.association_discrepancy import build_anomaly_transformer, build_dcdetector
from src.models.baselines.convolutional import InceptionTime, ResNetClassifier
from src.models.baselines.foundation import (
    WindowRequirements,
    build_lag_llama,
    build_moment,
    build_symtime,
    build_ttm,
)
from src.models.baselines.tslib import build_tslib_network
from src.models.tasks import wrap_network
from src.utils.third_party import IsolatedImporter

ALL_TASKS = frozenset(Task)
FORECASTING_ONLY = frozenset({Task.FORECASTING})
ANOMALY_ONLY = frozenset({Task.ANOMALY_DETECTION})
CLASSIFICATION_ONLY = frozenset({Task.CLASSIFICATION})


class UnsupportedBaselineError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BaselineRequest:
    task: Task
    context_length: int
    horizon: int
    num_classes: int
    hyperparameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BuiltBaseline:
    model: nn.Module
    requirements: WindowRequirements
    pretrained: bool


Builder = Callable[..., BuiltBaseline]


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    tasks: frozenset[Task]
    builder: Builder


def _wrapped(
    *,
    request: BaselineRequest,
    network: nn.Module,
    pretrained: bool,
) -> BuiltBaseline:
    return BuiltBaseline(
        model=wrap_network(
            task=request.task,
            network=network,
        ),
        requirements=WindowRequirements(),
        pretrained=pretrained,
    )


def _tslib(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
    name: str,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=build_tslib_network(
            importer=importer,
            name=name,
            task=request.task,
            context_length=request.context_length,
            horizon=request.horizon,
            num_classes=request.num_classes,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=False,
    )


def _anomaly_transformer(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return BuiltBaseline(
        model=build_anomaly_transformer(
            importer=importer,
            context_length=request.context_length,
            hyperparameters=request.hyperparameters,
        ),
        requirements=WindowRequirements(),
        pretrained=False,
    )


def _dcdetector(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return BuiltBaseline(
        model=build_dcdetector(
            importer=importer,
            context_length=request.context_length,
            hyperparameters=request.hyperparameters,
        ),
        requirements=WindowRequirements(),
        pretrained=False,
    )


def _inception_time(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=InceptionTime(
            num_classes=request.num_classes,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=False,
    )


def _resnet(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=ResNetClassifier(
            num_classes=request.num_classes,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=False,
    )


def _ttm(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=build_ttm(
            task=request.task,
            context_length=request.context_length,
            horizon=request.horizon,
            num_classes=request.num_classes,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=True,
    )


def _moment(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=build_moment(
            task=request.task,
            horizon=request.horizon,
            num_classes=request.num_classes,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=True,
    )


def _lag_llama(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    model, requirements = build_lag_llama(
        importer=importer,
        context_length=request.context_length,
        horizon=request.horizon,
        hyperparameters=request.hyperparameters,
    )
    return BuiltBaseline(
        model=model,
        requirements=requirements,
        pretrained=True,
    )


def _symtime(
    *,
    importer: IsolatedImporter,
    request: BaselineRequest,
) -> BuiltBaseline:
    return _wrapped(
        request=request,
        network=build_symtime(
            importer=importer,
            context_length=request.context_length,
            horizon=request.horizon,
            hyperparameters=request.hyperparameters,
        ),
        pretrained=True,
    )


BASELINES: dict[str, BaselineSpec] = {
    "Autoformer": BaselineSpec(tasks=ALL_TASKS, builder=partial(_tslib, name="Autoformer")),
    "PatchTST": BaselineSpec(tasks=ALL_TASKS, builder=partial(_tslib, name="PatchTST")),
    "TimesNet": BaselineSpec(tasks=ALL_TASKS, builder=partial(_tslib, name="TimesNet")),
    "TimeMixer": BaselineSpec(tasks=ALL_TASKS, builder=partial(_tslib, name="TimeMixer")),
    "AnomalyTransformer": BaselineSpec(tasks=ANOMALY_ONLY, builder=_anomaly_transformer),
    "DCdetector": BaselineSpec(tasks=ANOMALY_ONLY, builder=_dcdetector),
    "InceptionTime": BaselineSpec(tasks=CLASSIFICATION_ONLY, builder=_inception_time),
    "ResNet": BaselineSpec(tasks=CLASSIFICATION_ONLY, builder=_resnet),
    "TTM": BaselineSpec(tasks=ALL_TASKS, builder=_ttm),
    "Moment": BaselineSpec(tasks=ALL_TASKS, builder=_moment),
    "LagLlama": BaselineSpec(tasks=FORECASTING_ONLY, builder=_lag_llama),
    "SymTime": BaselineSpec(tasks=FORECASTING_ONLY, builder=_symtime),
}


class BaselineFactory:
    def __init__(self, *, importer: IsolatedImporter) -> None:
        self._importer = importer

    def build(
        self,
        *,
        name: str,
        request: BaselineRequest,
    ) -> BuiltBaseline:
        if name not in BASELINES:
            raise UnsupportedBaselineError(f"Unknown baseline {name!r}, choose from {sorted(BASELINES)}")
        spec = BASELINES[name]
        if request.task not in spec.tasks:
            supported = sorted(task.value for task in spec.tasks)
            raise UnsupportedBaselineError(f"{name} does not support {request.task.value}, supported: {supported}")
        return spec.builder(
            importer=self._importer,
            request=request,
        )
