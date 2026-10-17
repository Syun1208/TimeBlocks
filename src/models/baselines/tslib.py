from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.data.classes.enums import Task
from src.utils.third_party import TIME_SERIES_LIBRARY, IsolatedImporter

TSLIB_TASK_NAMES = {
    Task.FORECASTING: "long_term_forecast",
    Task.IMPUTATION: "imputation",
    Task.ANOMALY_DETECTION: "anomaly_detection",
    Task.CLASSIFICATION: "classification",
}
TSLIB_DEFAULTS: dict[str, Any] = {
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "dropout": 0.1,
    "factor": 1,
    "moving_avg": 25,
    "embed": "timeF",
    "freq": "h",
    "activation": "gelu",
    "top_k": 5,
    "num_kernels": 6,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 0,
    "down_sampling_window": 1,
    "down_sampling_method": None,
    "label_len": 48,
}
UNIVARIATE = 1


class TslibNetwork(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        task: Task,
        label_length: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task
        self.label_length = label_length
        self.horizon = horizon

    def _forecast(self, *, x: torch.Tensor) -> torch.Tensor:
        decoder = torch.zeros(x.shape[0], self.label_length + self.horizon, x.shape[2], device=x.device)
        return self.model(x, None, decoder, None)[:, -self.horizon :, :]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task is Task.FORECASTING:
            return self._forecast(x=x)
        if self.task is Task.IMPUTATION:
            return self.model(x, None, None, None, mask)
        if self.task is Task.CLASSIFICATION:
            return self.model(x, torch.ones(x.shape[0], x.shape[1], device=x.device), None, None)
        return self.model(x, None, None, None)


def build_tslib_network(
    *,
    importer: IsolatedImporter,
    name: str,
    task: Task,
    context_length: int,
    horizon: int,
    num_classes: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    module = importer.load(
        repository=TIME_SERIES_LIBRARY,
        module=f"models.{name}",
    )
    values = {**TSLIB_DEFAULTS, **hyperparameters}
    is_forecasting = task is Task.FORECASTING
    values.update(
        {
            "task_name": TSLIB_TASK_NAMES[task],
            "seq_len": context_length,
            "pred_len": horizon if is_forecasting else 0,
            "label_len": min(values["label_len"], context_length),
            "enc_in": UNIVARIATE,
            "dec_in": UNIVARIATE,
            "c_out": UNIVARIATE,
            "num_class": num_classes,
        }
    )
    configs = SimpleNamespace(**values)
    return TslibNetwork(
        model=module.Model(configs),
        task=task,
        label_length=configs.label_len,
        horizon=horizon,
    )
