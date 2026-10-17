from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.classes.series import Batch
from src.data.loaders.windows import ScoringWindows, build_loader
from src.interface.threshold import ThresholdStrategy
from src.utils.metrics import RegressionAccumulator, accuracy, evaluate_threshold
from src.utils.progress import EVALUATE, finish, progress


class Evaluator:
    def __init__(
        self,
        *,
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers

    def _batches(
        self,
        *,
        dataset: Dataset,
        label: str,
    ):
        loader = build_loader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
        bar = progress(
            stage=EVALUATE,
            total=len(loader),
            unit="batch",
            leave=False,
            detail=label,
        )
        for items in loader:
            yield Batch.from_mapping(
                items=items,
                device=self._device,
            )
            bar.update(1)
        finish(bar=bar)

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def regression(
        self,
        *,
        model: nn.Module,
        dataset: Dataset,
        label: str,
        masked_only: bool,
    ) -> dict[str, float]:
        model.eval()
        accumulator = RegressionAccumulator()
        seconds, batches, samples = 0.0, 0, 0
        for batch in self._batches(
            dataset=dataset,
            label=label,
        ):
            self._synchronize()
            started = time.perf_counter()
            predictions = model.predict(batch=batch)
            self._synchronize()
            seconds += time.perf_counter() - started
            batches += 1
            samples += batch.size
            accumulator.update(
                predictions=predictions,
                targets=batch.targets,
                weights=(1.0 - batch.mask) if masked_only else None,
            )
        metrics = accumulator.compute()
        return {
            "mse": metrics.mse,
            "mae": metrics.mae,
            "points": float(metrics.count),
            "samples": float(samples),
            "seconds_per_batch": seconds / max(batches, 1),
            "seconds_per_sample": seconds / max(samples, 1),
        }

    @torch.no_grad()
    def classification(
        self,
        *,
        model: nn.Module,
        dataset: Dataset,
        label: str,
    ) -> dict[str, float]:
        model.eval()
        predicted, expected = [], []
        for batch in self._batches(
            dataset=dataset,
            label=label,
        ):
            predicted.append(model.predict(batch=batch).argmax(dim=-1).cpu().numpy())
            expected.append(batch.labels.cpu().numpy())
        return {
            "accuracy": accuracy(
                predictions=np.concatenate(predicted),
                labels=np.concatenate(expected),
            ),
            "samples": float(sum(len(item) for item in expected)),
        }

    @torch.no_grad()
    def point_scores(
        self,
        *,
        scorer: nn.Module,
        windows: ScoringWindows,
        label: str,
    ) -> np.ndarray:
        scorer.eval()
        totals = np.zeros(windows.end - windows.begin, dtype=np.float64)
        counts = np.zeros_like(totals)
        index = 0
        for batch in self._batches(
            dataset=windows,
            label=label,
        ):
            scores = scorer.score(batch=batch).cpu().numpy()
            for row in scores:
                start, _ = windows.locate(index=index)
                offset = start - windows.begin
                totals[offset : offset + windows.context_length] += row
                counts[offset : offset + windows.context_length] += 1.0
                index += 1
        return totals / np.maximum(counts, 1.0)


class AnomalyEvaluation:
    def __init__(
        self,
        *,
        evaluator: Evaluator,
        threshold: ThresholdStrategy,
        point_adjustment: bool,
    ) -> None:
        self._evaluator = evaluator
        self._threshold = threshold
        self._point_adjustment = point_adjustment

    def run(
        self,
        *,
        scorer: nn.Module,
        windows_by_split: dict[str, ScoringWindows],
        labels: np.ndarray,
    ) -> dict[str, float]:
        scores = {
            split: self._evaluator.point_scores(
                scorer=scorer,
                windows=windows,
                label=split,
            )
            for split, windows in windows_by_split.items()
        }
        validation = windows_by_split["validation"]
        test = windows_by_split["test"]
        threshold = self._threshold.choose(
            train_scores=scores["train"],
            validation_scores=scores["validation"],
            validation_labels=labels[validation.begin : validation.end],
            test_scores=scores["test"],
        )
        result = evaluate_threshold(
            scores=scores["test"],
            labels=labels[test.begin : test.end],
            threshold=threshold,
            point_adjustment=self._point_adjustment,
        )
        return {
            "f1": result.f1,
            "precision": result.precision,
            "recall": result.recall,
            "threshold": result.threshold,
        }
