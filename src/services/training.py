from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.classes.series import Batch
from src.data.classes.settings import OptimizationSettings
from src.data.loaders.windows import build_loader, evenly_spaced_subset
from src.utils.progress import TRAIN, VALIDATE, finish, progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FitResult:
    best_validation_loss: float
    best_epoch: int
    epochs_run: int
    seconds: float


class Trainer:
    def __init__(
        self,
        *,
        settings: OptimizationSettings,
        device: torch.device,
        seed: int,
        label: str,
    ) -> None:
        self._settings = settings
        self._device = device
        self._seed = seed
        self._label = label

    def _optimizer(self, *, model: nn.Module) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            params=parameters,
            lr=self._settings.learning_rate,
            weight_decay=self._settings.weight_decay,
        )

    def _train_epoch(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        epoch: int,
    ) -> float:
        loader = build_loader(
            dataset=dataset,
            batch_size=self._settings.batch_size,
            shuffle=True,
            num_workers=self._settings.num_workers,
            num_samples=self._settings.samples_per_epoch,
            seed=self._seed + epoch,
        )
        model.train()
        bar = progress(
            stage=TRAIN,
            total=len(loader),
            unit="batch",
            leave=False,
            detail=f"{self._label} epoch {epoch}",
        )
        total_loss = 0.0
        for step, items in enumerate(loader, start=1):
            batch = Batch.from_mapping(
                items=items,
                device=self._device,
            )
            loss = model.training_loss(batch=batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self._settings.gradient_clip_norm:
                nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=self._settings.gradient_clip_norm,
                )
            optimizer.step()
            total_loss += float(loss.detach())
            bar.update(1)
            bar.set_postfix_str(f"loss {total_loss / step:.4f}")
        finish(bar=bar)
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def validation_loss(
        self,
        *,
        model: nn.Module,
        dataset: Dataset,
    ) -> float:
        subset = evenly_spaced_subset(
            dataset=dataset,
            limit=self._settings.validation_samples,
        )
        loader = build_loader(
            dataset=subset,
            batch_size=self._settings.batch_size,
            shuffle=False,
            num_workers=self._settings.num_workers,
        )
        model.eval()
        bar = progress(
            stage=VALIDATE,
            total=len(loader),
            unit="batch",
            leave=False,
            detail=self._label,
        )
        total, count = 0.0, 0
        for items in loader:
            batch = Batch.from_mapping(
                items=items,
                device=self._device,
            )
            total += float(model.training_loss(batch=batch)) * batch.size
            count += batch.size
            bar.update(1)
        finish(bar=bar)
        return total / max(count, 1)

    def fit(
        self,
        *,
        model: nn.Module,
        train_dataset: Dataset,
        validation_dataset: Dataset | None,
    ) -> FitResult:
        started = time.perf_counter()
        model.to(self._device)
        optimizer = self._optimizer(model=model)
        best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
        epoch = 0
        for epoch in range(1, self._settings.epochs + 1):
            train_loss = self._train_epoch(
                model=model,
                optimizer=optimizer,
                dataset=train_dataset,
                epoch=epoch,
            )
            if validation_dataset is None:
                logger.debug("%s epoch %d train %.5f", self._label, epoch, train_loss)
                continue
            validation_loss = self.validation_loss(
                model=model,
                dataset=validation_dataset,
            )
            logger.debug("%s epoch %d train %.5f validation %.5f", self._label, epoch, train_loss, validation_loss)
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            else:
                stale += 1
                if self._settings.patience is not None and stale >= self._settings.patience:
                    logger.info("%s early stopping at epoch %d", self._label, epoch)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return FitResult(
            best_validation_loss=best_loss if validation_dataset is not None else math.nan,
            best_epoch=best_epoch if validation_dataset is not None else epoch,
            epochs_run=epoch,
            seconds=time.perf_counter() - started,
        )
