from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

THRESHOLD_CANDIDATES = 512


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mse: float
    mae: float
    count: int


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    f1: float
    precision: float
    recall: float
    threshold: float


class RegressionAccumulator:
    def __init__(self) -> None:
        self._squared_error = 0.0
        self._absolute_error = 0.0
        self._count = 0

    def update(
        self,
        *,
        predictions: "torch.Tensor",
        targets: "torch.Tensor",
        weights: "torch.Tensor | None" = None,
    ) -> None:
        difference = (predictions - targets).double()
        if weights is None:
            self._squared_error += float(difference.pow(2).sum())
            self._absolute_error += float(difference.abs().sum())
            self._count += difference.numel()
            return
        selected = weights.double()
        self._squared_error += float((difference.pow(2) * selected).sum())
        self._absolute_error += float((difference.abs() * selected).sum())
        self._count += int(selected.sum())

    def compute(self) -> RegressionMetrics:
        denominator = max(self._count, 1)
        return RegressionMetrics(
            mse=self._squared_error / denominator,
            mae=self._absolute_error / denominator,
            count=self._count,
        )


def adjust_predictions(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    adjusted = predictions.astype(bool).copy()
    truth = labels.astype(bool)
    boundaries = np.flatnonzero(np.diff(np.concatenate(([0], truth.astype(np.int8), [0]))))
    for start, end in zip(boundaries[0::2], boundaries[1::2]):
        if adjusted[start:end].any():
            adjusted[start:end] = True
    return adjusted


def detection_scores(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    point_adjustment: bool,
) -> tuple[float, float, float]:
    truth = labels.astype(bool)
    predicted = (
        adjust_predictions(
            predictions=predictions,
            labels=labels,
        )
        if point_adjustment
        else predictions.astype(bool)
    )
    true_positive = float(np.sum(predicted & truth))
    precision = true_positive / max(float(predicted.sum()), 1.0)
    recall = true_positive / max(float(truth.sum()), 1.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def evaluate_threshold(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    point_adjustment: bool,
) -> DetectionMetrics:
    f1, precision, recall = detection_scores(
        predictions=scores > threshold,
        labels=labels,
        point_adjustment=point_adjustment,
    )
    return DetectionMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        threshold=threshold,
    )


def search_best_threshold(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    point_adjustment: bool,
    candidates: int = THRESHOLD_CANDIDATES,
) -> DetectionMetrics:
    quantiles = np.linspace(0.5, 1.0, candidates, endpoint=False)
    thresholds = np.unique(np.quantile(scores, quantiles))
    results = [
        evaluate_threshold(
            scores=scores,
            labels=labels,
            threshold=float(threshold),
            point_adjustment=point_adjustment,
        )
        for threshold in thresholds
    ]
    return max(results, key=lambda result: result.f1)


def accuracy(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    return float(np.mean(predictions == labels))
