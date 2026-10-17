from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.data.classes.blocks import ModelRecord
from src.data.classes.enums import RouterLoss, Split, Task
from src.data.classes.series import Batch
from src.data.classes.settings import ImputationSettings, RouterSettings
from src.data.loaders.windows import build_loader, evenly_spaced_subset
from src.models.timeblocks.router import Router
from src.services.blockbase_store import BlockbaseStore, block_records_of
from src.services.windowing import WindowFactory, WindowRequest
from src.utils.progress import RESIDUALS, ROUTER, finish, progress

logger = logging.getLogger(__name__)

RESIDUAL_BATCH_SIZE = 512
NO_WORKERS = 0


@dataclass(frozen=True, slots=True)
class ResidualPairs:
    model_id: str
    residuals: torch.Tensor
    fingerprints: torch.Tensor


def router_loss(
    *,
    kind: RouterLoss,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if kind is RouterLoss.COSINE:
        return (1.0 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
    return F.mse_loss(
        input=predictions,
        target=targets,
    )


class ResidualCollector:
    def __init__(
        self,
        *,
        store: BlockbaseStore,
        windows: WindowFactory,
        imputation: ImputationSettings,
        windows_per_model: int,
        device: torch.device,
    ) -> None:
        self._store = store
        self._windows = windows
        self._imputation = imputation
        self._windows_per_model = windows_per_model
        self._device = device

    def _dataset(self, *, record: ModelRecord):
        request = WindowRequest(
            task=Task(record.task),
            dataset=record.dataset,
            split=Split.TRAIN.value,
            context_length=record.context_length,
            horizon=record.horizon,
            mask_ratios=self._imputation.mask_ratios,
            mask_seed=self._imputation.evaluation_seed,
        )
        return evenly_spaced_subset(
            dataset=self._windows.build(request=request),
            limit=self._windows_per_model,
        )

    @torch.no_grad()
    def collect(self, *, record: ModelRecord) -> ResidualPairs:
        backbone = self._store.load_backbone(record=record).to(self._device).eval()
        loader = build_loader(
            dataset=self._dataset(record=record),
            batch_size=RESIDUAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NO_WORKERS,
        )
        inputs, targets = [], []
        fingerprints = torch.stack([block.fingerprint() for block in backbone.blocks])
        for items in loader:
            batch = Batch.from_mapping(
                items=items,
                device=self._device,
            )
            values = batch.inputs if batch.mask is None else batch.inputs * batch.mask
            residuals = backbone.residuals(
                values=values,
                mask=batch.mask,
            )
            for position in range(len(backbone.blocks)):
                inputs.append(residuals[position].cpu())
                targets.append(fingerprints[position].expand(residuals[position].shape[0], -1).cpu())
        return ResidualPairs(
            model_id=record.model_id,
            residuals=torch.cat(inputs),
            fingerprints=torch.cat(targets),
        )


class RouterTrainingService:
    def __init__(
        self,
        *,
        settings: RouterSettings,
        collector: ResidualCollector,
        device: torch.device,
        seed: int,
    ) -> None:
        self._settings = settings
        self._collector = collector
        self._device = device
        self._seed = seed

    def collect(self, *, records: Sequence[ModelRecord]) -> list[ResidualPairs]:
        bar = progress(
            stage=RESIDUALS,
            total=len(records),
            unit="model",
        )
        pairs = []
        for record in records:
            pairs.append(self._collector.collect(record=record))
            bar.update(1)
        finish(bar=bar)
        return pairs

    def _fit_model_pairs(
        self,
        *,
        router: Router,
        optimizer: torch.optim.Optimizer,
        pairs: ResidualPairs,
        generator: torch.Generator,
    ) -> float:
        loss_kind = RouterLoss(self._settings.loss)
        total = 0.0
        for _ in range(self._settings.epochs):
            order = torch.randperm(len(pairs.residuals), generator=generator)
            for start in range(0, len(order), self._settings.batch_size):
                selection = order[start : start + self._settings.batch_size]
                loss = router_loss(
                    kind=loss_kind,
                    predictions=router(residual=pairs.residuals[selection].to(self._device)),
                    targets=pairs.fingerprints[selection].to(self._device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total = float(loss.detach())
        return total

    def train(
        self,
        *,
        records: Sequence[ModelRecord],
        fingerprint_size: int,
    ) -> Router:
        pairs = self.collect(records=records)
        router = Router(
            settings=self._settings,
            fingerprint_size=fingerprint_size,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            params=router.parameters(),
            lr=self._settings.learning_rate,
        )
        generator = torch.Generator().manual_seed(self._seed)
        rng = np.random.default_rng(self._seed)
        bar = progress(
            stage=ROUTER,
            total=self._settings.rounds * len(pairs),
            unit="model",
        )
        router.train()
        for round_index in range(self._settings.rounds):
            for position in rng.permutation(len(pairs)):
                loss = self._fit_model_pairs(
                    router=router,
                    optimizer=optimizer,
                    pairs=pairs[position],
                    generator=generator,
                )
                bar.update(1)
                bar.set_postfix_str(f"round {round_index + 1} loss {loss:.5f}")
        finish(bar=bar)
        return router.eval()


def fingerprint_matrix(
    *,
    store: BlockbaseStore,
    records: Sequence[ModelRecord],
) -> tuple[np.ndarray, list]:
    blocks, fingerprints = [], []
    for record in records:
        backbone = store.load_backbone(record=record)
        for block_record, block in zip(block_records_of(record=record), backbone.blocks):
            blocks.append(block_record)
            fingerprints.append(block.fingerprint().numpy())
    sizes = {len(fingerprint) for fingerprint in fingerprints}
    if len(sizes) != 1:
        raise ValueError(f"Fingerprints have different sizes {sizes}, all blocks must share BlockSettings")
    return np.stack(fingerprints).astype(np.float32), blocks
