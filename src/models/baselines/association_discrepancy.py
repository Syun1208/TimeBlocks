from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.classes.series import Batch
from src.utils.third_party import ANOMALY_TRANSFORMER, DCDETECTOR, IsolatedImporter

KL_EPSILON = 1.0e-4
UNIVARIATE = 1
ANOMALY_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 3,
    "d_ff": 512,
    "dropout": 0.0,
    "k": 3.0,
    "temperature": 50.0,
}
DCDETECTOR_DEFAULTS: dict[str, Any] = {
    "d_model": 256,
    "n_heads": 1,
    "e_layers": 3,
    "d_ff": 512,
    "dropout": 0.0,
    "patch_size": [3, 4, 6],
    "temperature": 50.0,
}


def kl_divergence(
    *,
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    result = p * (torch.log(p + KL_EPSILON) - torch.log(q + KL_EPSILON))
    return torch.mean(torch.sum(result, dim=-1), dim=1)


def normalize_prior(*, prior: torch.Tensor) -> torch.Tensor:
    return prior / prior.sum(dim=-1, keepdim=True)


def discrepancy_losses(
    *,
    series: Sequence[torch.Tensor],
    priors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    series_loss, prior_loss = 0.0, 0.0
    for association, prior in zip(series, priors):
        normalized = normalize_prior(prior=prior)
        series_loss = (
            series_loss
            + torch.mean(
                kl_divergence(
                    p=association,
                    q=normalized.detach(),
                )
            )
            + torch.mean(
                kl_divergence(
                    p=normalized.detach(),
                    q=association,
                )
            )
        )
        prior_loss = (
            prior_loss
            + torch.mean(
                kl_divergence(
                    p=normalized,
                    q=association.detach(),
                )
            )
            + torch.mean(
                kl_divergence(
                    p=association.detach(),
                    q=normalized,
                )
            )
        )
    return series_loss / len(priors), prior_loss / len(priors)


def pointwise_discrepancy(
    *,
    series: Sequence[torch.Tensor],
    priors: Sequence[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    total = 0.0
    for association, prior in zip(series, priors):
        normalized = normalize_prior(prior=prior)
        total = (
            total
            + kl_divergence(
                p=association,
                q=normalized.detach(),
            )
            * temperature
        )
        total = (
            total
            + kl_divergence(
                p=normalized,
                q=association.detach(),
            )
            * temperature
        )
    return total


class AnomalyTransformerTask(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        k: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.k = k
        self.temperature = temperature

    def predict(self, *, batch: Batch) -> torch.Tensor:
        output, _, _, _ = self.model(batch.inputs)
        return output

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        output, series, priors, _ = self.model(batch.inputs)
        series_loss, prior_loss = discrepancy_losses(
            series=series,
            priors=priors,
        )
        reconstruction = F.mse_loss(
            input=output,
            target=batch.inputs,
        )
        return (reconstruction - self.k * series_loss) + (reconstruction + self.k * prior_loss)

    def score(self, *, batch: Batch) -> torch.Tensor:
        output, series, priors, _ = self.model(batch.inputs)
        reconstruction = (batch.inputs - output).pow(2).mean(dim=-1)
        metric = torch.softmax(
            -pointwise_discrepancy(
                series=series,
                priors=priors,
                temperature=self.temperature,
            ),
            dim=-1,
        )
        return metric * reconstruction


class DcdetectorTask(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        temperature: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def predict(self, *, batch: Batch) -> torch.Tensor:
        return batch.inputs

    def training_loss(self, *, batch: Batch) -> torch.Tensor:
        series, priors = self.model(batch.inputs)
        series_loss, prior_loss = discrepancy_losses(
            series=series,
            priors=priors,
        )
        return prior_loss - series_loss

    def score(self, *, batch: Batch) -> torch.Tensor:
        series, priors = self.model(batch.inputs)
        return torch.softmax(
            -pointwise_discrepancy(
                series=series,
                priors=priors,
                temperature=self.temperature,
            ),
            dim=-1,
        )


def build_anomaly_transformer(
    *,
    importer: IsolatedImporter,
    context_length: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    values = {**ANOMALY_TRANSFORMER_DEFAULTS, **hyperparameters}
    module = importer.load(
        repository=ANOMALY_TRANSFORMER,
        module="model.AnomalyTransformer",
    )
    model = module.AnomalyTransformer(
        win_size=context_length,
        enc_in=UNIVARIATE,
        c_out=UNIVARIATE,
        d_model=values["d_model"],
        n_heads=values["n_heads"],
        e_layers=values["e_layers"],
        d_ff=values["d_ff"],
        dropout=values["dropout"],
    )
    return AnomalyTransformerTask(
        model=model,
        k=values["k"],
        temperature=values["temperature"],
    )


def build_dcdetector(
    *,
    importer: IsolatedImporter,
    context_length: int,
    hyperparameters: Mapping[str, Any],
) -> nn.Module:
    values = {**DCDETECTOR_DEFAULTS, **hyperparameters}
    module = importer.load(
        repository=DCDETECTOR,
        module="model.DCdetector",
    )
    model = module.DCdetector(
        win_size=context_length,
        enc_in=UNIVARIATE,
        c_out=UNIVARIATE,
        n_heads=values["n_heads"],
        d_model=values["d_model"],
        e_layers=values["e_layers"],
        patch_size=list(values["patch_size"]),
        channel=UNIVARIATE,
        d_ff=values["d_ff"],
        dropout=values["dropout"],
    )
    return DcdetectorTask(
        model=model,
        temperature=values["temperature"],
    )
