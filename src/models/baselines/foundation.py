from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.data.classes.enums import Task
from src.data.classes.series import Batch
from src.models.baselines.fixed_length import fit_length, fit_mask
from src.utils.third_party import LAG_LLAMA, SYMTIME, SYMTIME_CHECKPOINT, IsolatedImporter

TTM_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
TTM_DEFAULTS: dict[str, Any] = {"model_context_length": 512, "head_dropout": 0.1}
MOMENT_MODEL = "AutonLab/MOMENT-1-large"
MOMENT_LENGTH = 512
MOMENT_DEFAULTS: dict[str, Any] = {
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
    "head_dropout": 0.1,
}
LAG_LLAMA_REPOSITORY = "time-series-foundation-models/Lag-Llama"
LAG_LLAMA_CHECKPOINT = "lag-llama.ckpt"
LAG_LLAMA_DEFAULTS: dict[str, Any] = {"num_samples": 100, "single_pass_sampling": False, "rope_scaling": True}
SYMTIME_DEFAULTS: dict[str, Any] = {
    "patch_len": 16,
    "stride": 8,
    "padding_patch": True,
    "forward_layers": 3,
    "individual": False,
    "out_dropout": 0.1,
    "use_avg": True,
    "moving_avg": 25,
    "model_config": "configs/SymTime_base.yaml",
}
UNIVARIATE = 1


@dataclass(frozen=True, slots=True)
class WindowRequirements:
    history_length: int = 0
    time_features: bool = False


class TtmForecaster(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        context_length: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.context_length = context_length
        self.horizon = horizon

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, observed = fit_length(
            x=x,
            length=self.context_length,
        )
        output = self.model(
            past_values=values,
            past_observed_mask=observed.unsqueeze(-1).bool(),
        )
        return output.prediction_outputs[:, : self.horizon, :]


class TtmBackboneHead(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        context_length: int,
        feature_size: int,
        output_size: int,
        task: Task,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.context_length = context_length
        self.task = task
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=feature_size,
                out_features=output_size,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, observed = fit_length(
            x=x,
            length=self.context_length,
        )
        if mask is not None:
            observed = observed * fit_mask(
                mask=mask,
                length=self.context_length,
            )
        output = self.backbone(
            past_values=values,
            past_observed_mask=observed.unsqueeze(-1).bool(),
        )
        features = self.head(output.last_hidden_state.flatten(start_dim=1))
        if self.task is Task.CLASSIFICATION:
            return features
        return (features.unsqueeze(-1) * output.scale + output.loc)[:, -x.shape[1] :, :]


def build_ttm(
    *,
    task: Task,
    context_length: int,
    horizon: int,
    num_classes: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    from tsfm_public.toolkit.get_model import get_model

    values = {**TTM_DEFAULTS, **hyperparameters}
    model_context = values["model_context_length"]
    model = get_model(
        TTM_MODEL,
        context_length=model_context,
        prediction_length=horizon if task is Task.FORECASTING else MOMENT_LENGTH // 4,
    )
    if task is Task.FORECASTING:
        return TtmForecaster(
            model=model,
            context_length=model_context,
            horizon=horizon,
        )
    with torch.no_grad():
        probe = model.backbone(past_values=torch.zeros(1, model_context, UNIVARIATE)).last_hidden_state
    return TtmBackboneHead(
        backbone=model.backbone,
        context_length=model_context,
        feature_size=probe[0].numel(),
        output_size=num_classes if task is Task.CLASSIFICATION else context_length,
        task=task,
        dropout=values["head_dropout"],
    )


class MomentNetwork(nn.Module):
    def __init__(
        self,
        *,
        pipeline: nn.Module,
        task: Task,
        horizon: int,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.task = task
        self.horizon = horizon

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values, observed = fit_length(
            x=x,
            length=MOMENT_LENGTH,
        )
        arguments = {"x_enc": values.transpose(1, 2), "input_mask": observed}
        if mask is not None:
            arguments["mask"] = fit_mask(
                mask=mask,
                length=MOMENT_LENGTH,
            )
        output = self.pipeline(**arguments)
        if self.task is Task.FORECASTING:
            return output.forecast.transpose(1, 2)
        if self.task is Task.CLASSIFICATION:
            return output.logits
        return output.reconstruction.transpose(1, 2)[:, -x.shape[1] :, :]


MOMENT_TASK_NAMES = {
    Task.FORECASTING: "forecasting",
    Task.IMPUTATION: "reconstruction",
    Task.ANOMALY_DETECTION: "reconstruction",
    Task.CLASSIFICATION: "classification",
}


def build_moment(
    *,
    task: Task,
    horizon: int,
    num_classes: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    from momentfm import MOMENTPipeline

    values = {**MOMENT_DEFAULTS, **hyperparameters}
    model_kwargs = {
        "task_name": MOMENT_TASK_NAMES[task],
        "freeze_encoder": values["freeze_encoder"],
        "freeze_embedder": values["freeze_embedder"],
        "freeze_head": values["freeze_head"],
        "head_dropout": values["head_dropout"],
    }
    if task is Task.FORECASTING:
        model_kwargs["forecast_horizon"] = horizon
    if task is Task.CLASSIFICATION:
        model_kwargs.update({"n_channels": UNIVARIATE, "num_class": num_classes})
    pipeline = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL,
        model_kwargs=model_kwargs,
    )
    pipeline.init()
    return MomentNetwork(
        pipeline=pipeline,
        task=task,
        horizon=horizon,
    )


class LagLlamaForecastingTask(nn.Module):
    def __init__(
        self,
        *,
        module: nn.Module,
        horizon: int,
    ) -> None:
        super().__init__()
        self.module = module
        self.horizon = horizon

    def _arguments(self, *, batch: Batch) -> dict[str, torch.Tensor]:
        return {
            "past_target": batch.history,
            "past_observed_values": batch.history_observed,
            "past_time_feat": batch.history_time,
            "future_time_feat": batch.future_time,
        }

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        arguments = self._arguments(batch=batch)
        arguments.update(
            {
                "future_target": batch.targets[..., 0],
                "future_observed_values": torch.ones_like(batch.targets[..., 0]),
            }
        )
        return self.module._compute_loss(arguments)

    def predict(self, *, batch: Batch) -> torch.Tensor:
        samples = self.module(**self._arguments(batch=batch))
        return samples.reshape(batch.size, -1, self.horizon).mean(dim=1).unsqueeze(-1)


def build_lag_llama(
    *,
    importer: IsolatedImporter,
    context_length: int,
    horizon: int,
    hyperparameters: Mapping[str, Any],
) -> tuple[nn.Module, WindowRequirements]:
    from huggingface_hub import hf_hub_download

    values = {**LAG_LLAMA_DEFAULTS, **hyperparameters}
    checkpoint = hf_hub_download(
        repo_id=LAG_LLAMA_REPOSITORY,
        filename=LAG_LLAMA_CHECKPOINT,
    )
    arguments = torch.load(checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]["model_kwargs"]
    estimator_module = importer.load(
        repository=LAG_LLAMA,
        module="lag_llama.gluon.estimator",
    )
    rope_scaling = None
    if values["rope_scaling"]:
        factor = max(1.0, (context_length + horizon) / arguments["context_length"])
        rope_scaling = {"type": "linear", "factor": factor}
    estimator = estimator_module.LagLlamaEstimator(
        ckpt_path=checkpoint,
        prediction_length=horizon,
        context_length=context_length,
        input_size=arguments["input_size"],
        n_layer=arguments["n_layer"],
        n_embd_per_head=arguments["n_embd_per_head"],
        n_head=arguments["n_head"],
        scaling=arguments["scaling"],
        time_feat=arguments["time_feat"],
        rope_scaling=rope_scaling,
        num_parallel_samples=values["num_samples"],
        device=torch.device("cpu"),
    )
    module = estimator.create_lightning_module(use_kv_cache=True)
    module.use_kv_cache = True
    module.use_single_pass_sampling = values["single_pass_sampling"]
    requirements = WindowRequirements(
        history_length=max(module.model.lags_seq),
        time_features=bool(arguments["time_feat"]),
    )
    return (
        LagLlamaForecastingTask(
            module=module,
            horizon=horizon,
        ),
        requirements,
    )


class SymTimeNetwork(nn.Module):
    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x)


def build_symtime(
    *,
    importer: IsolatedImporter,
    context_length: int,
    horizon: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    import yaml

    values = {**SYMTIME_DEFAULTS, **hyperparameters}
    repository = importer.repository(name=SYMTIME)
    checkpoint = repository.parent / SYMTIME_CHECKPOINT
    configs = yaml.safe_load((repository / values["model_config"]).read_text(encoding="utf-8"))
    arguments = SimpleNamespace(
        task_name="long_term_forecast",
        patch_len=values["patch_len"],
        stride=values["stride"],
        padding_patch=values["padding_patch"],
        seq_len=context_length,
        pred_len=horizon,
        forward_layers=values["forward_layers"],
        individual=values["individual"],
        pretrain_path=str(checkpoint),
        out_dropout=values["out_dropout"],
        use_avg=values["use_avg"],
        moving_avg=values["moving_avg"],
        enc_in=UNIVARIATE,
    )
    module = importer.load(
        repository=SYMTIME,
        module="models.finetune_model",
    )
    return SymTimeNetwork(model=module.SymTime(arguments, configs))
